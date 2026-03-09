"""
学習塾 時間割自動生成システム
Google OR-Tools CP-SAT Solver を使用
"""

from ortools.sat.python import cp_model
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# 定数
# ============================================================
SLOT_MINUTES = 5          # 1スロット = 5分
DAY_START_HOUR = 9
DAY_START_MIN = 30
MAX_SLOT = 144            # 9:30(t=0) ~ 21:30(t=144)
BREAK_SLOTS = 3           # 科目変更時の休憩 = 15分 = 3スロット


def slot_to_time(t: int) -> str:
    """スロット番号 -> "HH:MM" 文字列"""
    total_min = DAY_START_HOUR * 60 + DAY_START_MIN + t * SLOT_MINUTES
    return f"{total_min // 60:02d}:{total_min % 60:02d}"


# ============================================================
# データクラス
# ============================================================
@dataclass
class Room:
    campus: str          # 校舎名
    room_id: str         # 教室ID (例: "長野校_R1")
    capacity: int        # 定員

@dataclass
class Teacher:
    teacher_id: str      # 講師ID
    name: str
    available: list      # [(start_slot, end_slot), ...] 稼働可能時間帯

@dataclass
class Lesson:
    lesson_id: str
    name: str            # 講座名
    campus: str          # 対象校舎
    subject: str         # 科目
    teacher_id: str      # 担当講師ID
    duration_slots: int  # 授業時間（スロット数）
    num_students: int    # 想定生徒数
    is_test_zemi: bool = False
    test_slots: int = 0          # テスト時間（スロット数）
    explanation_slots: int = 0   # 解説時間（スロット数）
    fixed_start: Optional[int] = None  # 開始時間固定（スロット番号）
    tags: list = field(default_factory=list)  # ["最難関大", "選抜"] 等
    broadcast_targets: list = field(default_factory=list)
    # [(campus_name, num_students), ...] 配信先校舎と生徒数
    # 例: [("福岡校", 8), ("長野校", 5), ("上田校", 3)]


# ============================================================
# ソルバー本体
# ============================================================
def build_and_solve(
    rooms: list[Room],
    teachers: list[Teacher],
    lessons: list[Lesson],
    time_limit_sec: float = 60.0,
):
    """
    CP-SAT モデルを構築し、時間割を最適化する。

    Returns:
        result_df: 結果の DataFrame (成功時) or None
        status: ソルバーステータス文字列
    """
    model = cp_model.CpModel()

    # --- 前処理: 辞書化 ---
    teacher_map = {t.teacher_id: t for t in teachers}
    rooms_by_campus: dict[str, list[Room]] = {}
    for r in rooms:
        rooms_by_campus.setdefault(r.campus, []).append(r)

    # ================================================================
    # 変数の定義
    # ================================================================
    # lesson_vars[i] = {
    #   "start": IntVar (開始スロット),
    #   "end":   IntVar (終了スロット),
    #   "interval": IntervalVar,
    #   "room_bools": {room_id: BoolVar},  # どの教室に割り当てるか
    #   --- テストゼミ専用 ---
    #   "test_start", "test_end", "test_interval",
    #   "expl_start", "expl_end", "expl_interval",
    # }
    lesson_vars = []

    for i, les in enumerate(lessons):
        lv = {}

        if les.is_test_zemi:
            total = les.test_slots + les.explanation_slots
            # テスト開始
            ts = model.new_int_var(0, MAX_SLOT - total, f"test_start_{i}")
            te = model.new_int_var(0, MAX_SLOT, f"test_end_{i}")
            model.Add(te == ts + les.test_slots)
            t_intv = model.new_interval_var(ts, les.test_slots, te, f"test_intv_{i}")

            # 解説開始 = テスト終了
            es = model.new_int_var(0, MAX_SLOT, f"expl_start_{i}")
            ee = model.new_int_var(0, MAX_SLOT, f"expl_end_{i}")
            model.Add(es == te)  # 解説はテスト直後
            model.Add(ee == es + les.explanation_slots)
            e_intv = model.new_interval_var(es, les.explanation_slots, ee, f"expl_intv_{i}")

            # 全体の start / end (教室占有用)
            lv["start"] = ts
            lv["end"] = ee
            overall_dur = total
            overall_intv = model.new_interval_var(ts, total, ee, f"overall_intv_{i}")
            lv["interval"] = overall_intv

            lv["test_start"] = ts
            lv["test_end"] = te
            lv["test_interval"] = t_intv
            lv["expl_start"] = es
            lv["expl_end"] = ee
            lv["expl_interval"] = e_intv

            # 固定開始
            if les.fixed_start is not None:
                model.Add(ts == les.fixed_start)

            # 時間枠内
            model.Add(ee <= MAX_SLOT)

        else:
            s = model.new_int_var(0, MAX_SLOT - les.duration_slots, f"start_{i}")
            e = model.new_int_var(0, MAX_SLOT, f"end_{i}")
            model.Add(e == s + les.duration_slots)
            intv = model.new_interval_var(s, les.duration_slots, e, f"intv_{i}")
            lv["start"] = s
            lv["end"] = e
            lv["interval"] = intv

            if les.fixed_start is not None:
                model.Add(s == les.fixed_start)

            model.Add(e <= MAX_SLOT)

        # --- 教室割当 BoolVar ---
        campus_rooms = rooms_by_campus.get(les.campus, [])
        room_bools = {}
        for r in campus_rooms:
            if r.capacity >= les.num_students:
                b = model.new_bool_var(f"room_{i}_{r.room_id}")
                room_bools[r.room_id] = b
        lv["room_bools"] = room_bools

        # ちょうど1教室を選択
        if not room_bools:
            raise ValueError(
                f"授業 '{les.name}' (生徒数{les.num_students}) を収容できる教室が "
                f"校舎 '{les.campus}' に存在しません。"
            )
        model.add_exactly_one(room_bools.values())

        # --- 配信先教室割当 BoolVar ---
        shadow_room_bools_list = []  # [(campus, {room_id: BoolVar}), ...]
        for t_idx, (target_campus, target_students) in enumerate(les.broadcast_targets):
            if target_campus == les.campus:
                raise ValueError(
                    f"授業 '{les.name}': 配信先校舎 '{target_campus}' が "
                    f"配信元校舎と同じです。"
                )
            target_campus_rooms = rooms_by_campus.get(target_campus, [])
            shadow_bools = {}
            for r in target_campus_rooms:
                if r.capacity >= target_students:
                    b = model.new_bool_var(f"shadow_room_{i}_t{t_idx}_{r.room_id}")
                    shadow_bools[r.room_id] = b
            if not shadow_bools:
                raise ValueError(
                    f"配信授業 '{les.name}' (配信先: {target_campus}, "
                    f"生徒数{target_students}) を収容できる教室が "
                    f"校舎 '{target_campus}' に存在しません。"
                )
            model.add_exactly_one(shadow_bools.values())
            shadow_room_bools_list.append((target_campus, shadow_bools))
        lv["shadow_room_bools"] = shadow_room_bools_list

        lesson_vars.append(lv)

    # ================================================================
    # 制約1 & 3: 教室の重複禁止 (同校舎・同教室で NoOverlap)
    #   配信先教室も含めて統合的に NoOverlap を適用
    # ================================================================
    # room_id → [(lesson_idx, BoolVar, start, end, duration, tag)] のマップ構築
    room_assignments: dict[str, list] = {}
    for i, les in enumerate(lessons):
        lv = lesson_vars[i]
        dur = les.test_slots + les.explanation_slots if les.is_test_zemi else les.duration_slots

        # 配信元の教室割当
        for rid, bv in lv["room_bools"].items():
            room_assignments.setdefault(rid, []).append(
                (i, bv, lv["start"], lv["end"], dur, f"opt_room_{i}_{rid}")
            )

        # 配信先の教室割当（同じ start/end を共有）
        for t_idx, (_, shadow_bools) in enumerate(lv["shadow_room_bools"]):
            for rid, bv in shadow_bools.items():
                room_assignments.setdefault(rid, []).append(
                    (i, bv, lv["start"], lv["end"], dur, f"opt_shadow_{i}_t{t_idx}_{rid}")
                )

    # room_id ごとに NoOverlap
    for rid, assignments in room_assignments.items():
        optional_intervals = []
        for (lesson_idx, bool_var, start_var, end_var, dur, name) in assignments:
            opt = model.new_optional_interval_var(
                start_var, dur, end_var, bool_var, name,
            )
            optional_intervals.append(opt)
        if len(optional_intervals) > 1:
            model.add_no_overlap(optional_intervals)

    # ================================================================
    # 制約4a: 講師の重複禁止
    # ================================================================
    teachers_lessons: dict[str, list[int]] = {}
    for i, les in enumerate(lessons):
        teachers_lessons.setdefault(les.teacher_id, []).append(i)

    teacher_intervals_for_no_overlap: dict[str, list] = {}

    for tid, idxs in teachers_lessons.items():
        intervals = []
        for i in idxs:
            les = lessons[i]
            if les.is_test_zemi:
                # 講師は解説時間のみ拘束
                intervals.append(lesson_vars[i]["expl_interval"])
            else:
                intervals.append(lesson_vars[i]["interval"])
        if len(intervals) > 1:
            model.add_no_overlap(intervals)
        teacher_intervals_for_no_overlap[tid] = intervals

    # ================================================================
    # 制約4b: 講師のシフト時間内に収める
    # ================================================================
    for i, les in enumerate(lessons):
        teacher = teacher_map[les.teacher_id]
        if not teacher.available:
            continue
        # 各 available 区間のどれかに入る
        if les.is_test_zemi:
            t_start = lesson_vars[i]["expl_start"]
            t_end = lesson_vars[i]["expl_end"]
        else:
            t_start = lesson_vars[i]["start"]
            t_end = lesson_vars[i]["end"]

        shift_bools = []
        for k, (a_s, a_e) in enumerate(teacher.available):
            b = model.new_bool_var(f"shift_{i}_{k}")
            model.Add(t_start >= a_s).only_enforce_if(b)
            model.Add(t_end <= a_e).only_enforce_if(b)
            shift_bools.append(b)
        model.add_exactly_one(shift_bools)

    # ================================================================
    # 制約5: 生徒被り回避（裏番組NGルール）
    # ================================================================
    elite_lessons: list[int] = []
    for i, les in enumerate(lessons):
        if any(tag in les.name for tag in ["最難関大", "選抜"]):
            elite_lessons.append(i)

    group_a_subjects = {"国語", "数学", "英語"}
    group_b_subjects = {"物理", "化学", "世界史", "日本史"}

    # ルールA: group_a と group_b は被ってはいけない
    overlap_ok_pairs = {
        frozenset({"物理", "生物"}),
        frozenset({"物理", "世界史"}),
        frozenset({"物理", "日本史"}),
        frozenset({"物理", "地理"}),
    }
    history_geo = {"日本史", "世界史", "地理"}

    for idx_i in range(len(elite_lessons)):
        for idx_j in range(idx_i + 1, len(elite_lessons)):
            i = elite_lessons[idx_i]
            j = elite_lessons[idx_j]
            li = lessons[i]
            lj = lessons[j]
            si, sj = li.subject, lj.subject

            must_not_overlap = False

            # ルールA: group_a vs group_b
            if (si in group_a_subjects and sj in group_b_subjects) or \
               (sj in group_a_subjects and si in group_b_subjects):
                pair = frozenset({si, sj})
                # ルールB: 例外ペアは被ってOK
                if pair not in overlap_ok_pairs:
                    must_not_overlap = True

            # ルールC: 日本史・世界史・地理は互いに被ってはいけない
            if si in history_geo and sj in history_geo and si != sj:
                must_not_overlap = True

            if must_not_overlap:
                # NoOverlap2 で被り禁止
                model.add_no_overlap(
                    [lesson_vars[i]["interval"], lesson_vars[j]["interval"]]
                )

    # ================================================================
    # 制約6: 講師の休憩ルール (科目が異なれば15分以上空ける)
    # ================================================================
    for tid, idxs in teachers_lessons.items():
        if len(idxs) < 2:
            continue
        for a_pos in range(len(idxs)):
            for b_pos in range(a_pos + 1, len(idxs)):
                ia = idxs[a_pos]
                ib = idxs[b_pos]
                la = lessons[ia]
                lb = lessons[ib]

                if la.subject == lb.subject:
                    continue  # 同科目なら休憩不要

                # 講師拘束の開始・終了を取得
                if la.is_test_zemi:
                    sa = lesson_vars[ia]["expl_start"]
                    ea = lesson_vars[ia]["expl_end"]
                else:
                    sa = lesson_vars[ia]["start"]
                    ea = lesson_vars[ia]["end"]

                if lb.is_test_zemi:
                    sb = lesson_vars[ib]["expl_start"]
                    eb = lesson_vars[ib]["expl_end"]
                else:
                    sb = lesson_vars[ib]["start"]
                    eb = lesson_vars[ib]["end"]

                # a が先 OR b が先 (NoOverlap で既に保証されているが、
                # 休憩分の gap を追加で確保する)
                b_a_first = model.new_bool_var(f"order_{ia}_{ib}")
                # a が先 → ea + BREAK_SLOTS <= sb
                model.Add(ea + BREAK_SLOTS <= sb).only_enforce_if(b_a_first)
                # b が先 → eb + BREAK_SLOTS <= sa
                model.Add(eb + BREAK_SLOTS <= sa).only_enforce_if(b_a_first.negated())

    # ================================================================
    # ソフト制約（目的関数）: 講師の待機時間を最小化
    # ================================================================
    penalty_vars = []
    for tid, idxs in teachers_lessons.items():
        if len(idxs) < 2:
            continue

        # 講師拘束区間の start / end を集める
        starts = []
        ends = []
        for i in idxs:
            les = lessons[i]
            if les.is_test_zemi:
                starts.append(lesson_vars[i]["expl_start"])
                ends.append(lesson_vars[i]["expl_end"])
            else:
                starts.append(lesson_vars[i]["start"])
                ends.append(lesson_vars[i]["end"])

        first_start = model.new_int_var(0, MAX_SLOT, f"first_{tid}")
        last_end = model.new_int_var(0, MAX_SLOT, f"last_{tid}")
        model.add_min_equality(first_start, starts)
        model.add_max_equality(last_end, ends)

        span = model.new_int_var(0, MAX_SLOT, f"span_{tid}")
        model.Add(span == last_end - first_start)

        total_teaching = sum(
            lessons[i].explanation_slots if lessons[i].is_test_zemi else lessons[i].duration_slots
            for i in idxs
        )
        idle = model.new_int_var(0, MAX_SLOT, f"idle_{tid}")
        model.Add(idle == span - total_teaching)
        penalty_vars.append(idle)

    if penalty_vars:
        model.minimize(sum(penalty_vars))

    # ================================================================
    # 求解
    # ================================================================
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_workers = 8

    status = solver.solve(model)
    status_name = solver.status_name(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"[WARN] ソルバーステータス: {status_name}")
        return None, status_name

    # ================================================================
    # 結果の抽出
    # ================================================================
    results = []
    for i, les in enumerate(lessons):
        lv = lesson_vars[i]
        has_broadcast = len(les.broadcast_targets) > 0

        # 割り当て教室
        assigned_room = None
        for rid, bv in lv["room_bools"].items():
            if solver.value(bv):
                assigned_room = rid
                break

        broadcast_label = "📡 配信元" if has_broadcast else ""

        if les.is_test_zemi:
            ts_val = solver.value(lv["test_start"])
            te_val = solver.value(lv["test_end"])
            es_val = solver.value(lv["expl_start"])
            ee_val = solver.value(lv["expl_end"])
            results.append({
                "授業ID": les.lesson_id,
                "講座名": les.name,
                "種別": "テストゼミ(テスト)",
                "科目": les.subject,
                "校舎": les.campus,
                "教室": assigned_room,
                "講師": teacher_map[les.teacher_id].name,
                "開始": slot_to_time(ts_val),
                "終了": slot_to_time(te_val),
                "開始slot": ts_val,
                "終了slot": te_val,
                "生徒数": les.num_students,
                "配信": broadcast_label,
            })
            results.append({
                "授業ID": les.lesson_id,
                "講座名": les.name,
                "種別": "テストゼミ(解説)",
                "科目": les.subject,
                "校舎": les.campus,
                "教室": assigned_room,
                "講師": teacher_map[les.teacher_id].name,
                "開始": slot_to_time(es_val),
                "終了": slot_to_time(ee_val),
                "開始slot": es_val,
                "終了slot": ee_val,
                "生徒数": les.num_students,
                "配信": broadcast_label,
            })
        else:
            s_val = solver.value(lv["start"])
            e_val = solver.value(lv["end"])
            results.append({
                "授業ID": les.lesson_id,
                "講座名": les.name,
                "種別": "通常授業",
                "科目": les.subject,
                "校舎": les.campus,
                "教室": assigned_room,
                "講師": teacher_map[les.teacher_id].name,
                "開始": slot_to_time(s_val),
                "終了": slot_to_time(e_val),
                "開始slot": s_val,
                "終了slot": e_val,
                "生徒数": les.num_students,
                "配信": broadcast_label,
            })

        # --- 配信先の結果行を追加 ---
        for t_idx, (target_campus, target_students) in enumerate(les.broadcast_targets):
            _, shadow_bools = lv["shadow_room_bools"][t_idx]
            assigned_shadow_room = None
            for rid, bv in shadow_bools.items():
                if solver.value(bv):
                    assigned_shadow_room = rid
                    break

            shadow_label = f"📡 {les.campus}より配信"

            if les.is_test_zemi:
                results.append({
                    "授業ID": les.lesson_id,
                    "講座名": les.name,
                    "種別": "テストゼミ(テスト)",
                    "科目": les.subject,
                    "校舎": target_campus,
                    "教室": assigned_shadow_room,
                    "講師": teacher_map[les.teacher_id].name,
                    "開始": slot_to_time(ts_val),
                    "終了": slot_to_time(te_val),
                    "開始slot": ts_val,
                    "終了slot": te_val,
                    "生徒数": target_students,
                    "配信": shadow_label,
                })
                results.append({
                    "授業ID": les.lesson_id,
                    "講座名": les.name,
                    "種別": "テストゼミ(解説)",
                    "科目": les.subject,
                    "校舎": target_campus,
                    "教室": assigned_shadow_room,
                    "講師": teacher_map[les.teacher_id].name,
                    "開始": slot_to_time(es_val),
                    "終了": slot_to_time(ee_val),
                    "開始slot": es_val,
                    "終了slot": ee_val,
                    "生徒数": target_students,
                    "配信": shadow_label,
                })
            else:
                results.append({
                    "授業ID": les.lesson_id,
                    "講座名": les.name,
                    "種別": "通常授業",
                    "科目": les.subject,
                    "校舎": target_campus,
                    "教室": assigned_shadow_room,
                    "講師": teacher_map[les.teacher_id].name,
                    "開始": slot_to_time(s_val),
                    "終了": slot_to_time(e_val),
                    "開始slot": s_val,
                    "終了slot": e_val,
                    "生徒数": target_students,
                    "配信": shadow_label,
                })

    result_df = pd.DataFrame(results)
    result_df.sort_values(["校舎", "開始slot", "教室"], inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    obj_val = solver.objective_value if penalty_vars else 0
    print(f"\n[INFO] ステータス: {status_name}")
    print(f"[INFO] 目的関数値 (講師空き時間合計スロット): {obj_val}")
    print(f"[INFO] 講師空き時間合計: {obj_val * SLOT_MINUTES:.0f} 分\n")

    return result_df, status_name


# ============================================================
# モックデータ生成
# ============================================================
def generate_mock_data():
    """テスト用のモックデータを生成"""

    # --- 教室 ---
    rooms = [
        Room("長野校", "長野校_R1", 20),
        Room("長野校", "長野校_R2", 15),
        Room("長野校", "長野校_R3", 10),
        Room("上田校", "上田校_R1", 30),
        Room("上田校", "上田校_R2", 6),
        Room("上田校", "上田校_R3", 3),
        Room("福岡校", "福岡校_R1", 10),
        Room("福岡校", "福岡校_R2", 10),
        Room("福岡校", "福岡校_R3", 15),
        Room("代々木校", "代々木校_R1", 8),
        Room("代々木校", "代々木校_R2", 30),
        Room("代々木校", "代々木校_R3", 15),
        Room("代々木校", "代々木校_R4", 45),
        Room("代々木校", "代々木校_R5", 15),
        Room("代々木別館", "代々木別館_R1", 12),
        Room("代々木別館", "代々木別館_R2", 12),
        Room("代々木別館", "代々木別館_R3", 50),
    ]

    # --- 講師 (稼働: 全日 9:30-21:30 = slot 0-144) ---
    full_day = [(0, 144)]
    morning  = [(0, 72)]    # 9:30-15:30
    afternoon = [(36, 144)]  # 12:30-21:30

    teachers = [
        Teacher("T01", "佐藤先生",  full_day),
        Teacher("T02", "鈴木先生",  full_day),
        Teacher("T03", "高橋先生",  afternoon),
        Teacher("T04", "田中先生",  full_day),
        Teacher("T05", "渡辺先生",  morning),
        Teacher("T06", "伊藤先生",  full_day),
        Teacher("T07", "山本先生",  full_day),
        Teacher("T08", "中村先生",  afternoon),
        Teacher("T09", "小林先生",  full_day),
        Teacher("T10", "加藤先生",  full_day),
    ]

    # --- 授業 ---
    # 時間換算: 45分=9, 60分=12, 90分=18, 120分=24
    lessons = [
        # === 代々木校: 最難関大/選抜 講座 (裏番組NGルールの検証用) ===
        # L01, L02: 全校舎へ配信
        Lesson("L01", "最難関大 英語α", "代々木校", "英語", "T01", 18, 25,
               tags=["最難関大"],
               broadcast_targets=[("福岡校", 8), ("長野校", 5), ("上田校", 3)]),
        Lesson("L02", "最難関大 数学α", "代々木校", "数学", "T02", 18, 20,
               tags=["最難関大"],
               broadcast_targets=[("福岡校", 7), ("長野校", 4), ("上田校", 3)]),
        Lesson("L03", "最難関大 国語α", "代々木校", "国語", "T04", 18, 20, tags=["最難関大"]),
        # L04: 福岡・長野のみ配信
        Lesson("L04", "選抜 物理",      "代々木校", "物理", "T06", 18, 12,
               tags=["選抜"],
               broadcast_targets=[("福岡校", 5), ("長野校", 3)]),
        Lesson("L05", "選抜 化学",      "代々木校", "化学", "T07", 18, 12, tags=["選抜"]),
        Lesson("L06", "選抜 世界史",    "代々木校", "世界史", "T09", 18, 14, tags=["選抜"]),
        Lesson("L07", "選抜 日本史",    "代々木校", "日本史", "T10", 18, 14, tags=["選抜"]),

        # === 代々木校: 通常授業 ===
        Lesson("L08", "標準 英語B",    "代々木校", "英語", "T01", 18, 12),
        Lesson("L09", "標準 数学B",    "代々木校", "数学", "T02", 18, 10),

        # === 長野校 ===
        Lesson("L10", "基礎 英語",     "長野校", "英語", "T03", 18, 8),
        Lesson("L11", "基礎 数学",     "長野校", "数学", "T03", 18, 8),
        Lesson("L12", "基礎 国語",     "長野校", "国語", "T05", 12, 10),

        # === 上田校 ===
        Lesson("L13", "基礎 英語",     "上田校", "英語", "T04", 18, 5),
        Lesson("L14", "基礎 数学",     "上田校", "数学", "T08", 12, 3),

        # === 福岡校 ===
        Lesson("L15", "標準 英語",     "福岡校", "英語", "T06", 18, 8),
        Lesson("L16", "標準 数学",     "福岡校", "数学", "T07", 18, 9),

        # === 代々木別館 ===
        Lesson("L17", "公開講座 英語", "代々木別館", "英語", "T09", 24, 40),

        # === テストゼミ (代々木校) ===
        Lesson("L18", "最難関大 英語テストゼミ", "代々木校", "英語", "T01",
               0, 25, is_test_zemi=True, test_slots=12, explanation_slots=12,
               tags=["最難関大"]),

        # === 時間固定の授業 (代々木校, 13:00開始 = slot 42) ===
        Lesson("L19", "人気講座 数学特別", "代々木校", "数学", "T02", 18, 28,
               fixed_start=42),
    ]

    return rooms, teachers, lessons


# ============================================================
# 結果の見やすい表示
# ============================================================
def print_timetable(df: pd.DataFrame):
    """時間割を校舎ごとに見やすく表示"""
    if df is None or df.empty:
        print("結果がありません。")
        return

    campuses = df["校舎"].unique()
    for campus in campuses:
        sub = df[df["校舎"] == campus]
        print("=" * 80)
        print(f"  【{campus}】")
        print("=" * 80)
        print(f"{'講座名':<28} {'種別':<16} {'科目':<6} {'教室':<14} "
              f"{'開始':>5} ~ {'終了':<5} {'講師':<10} {'生徒数':>4}")
        print("-" * 80)
        for _, row in sub.iterrows():
            print(f"{row['講座名']:<26} {row['種別']:<14} {row['科目']:<6} "
                  f"{row['教室']:<14} {row['開始']:>5} ~ {row['終了']:<5} "
                  f"{row['講師']:<10} {row['生徒数']:>4}")
        print()


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  学習塾 時間割自動生成システム")
    print("  (Google OR-Tools CP-SAT Solver)")
    print("=" * 60)

    rooms, teachers, lessons = generate_mock_data()

    print(f"\n[DATA] 教室数: {len(rooms)}")
    print(f"[DATA] 講師数: {len(teachers)}")
    print(f"[DATA] 授業数: {len(lessons)}")
    print(f"[DATA] うちテストゼミ: {sum(1 for l in lessons if l.is_test_zemi)}")
    print(f"[DATA] うち時間固定: {sum(1 for l in lessons if l.fixed_start is not None)}")
    elite = [l for l in lessons if any(t in l.name for t in ["最難関大", "選抜"])]
    print(f"[DATA] うち最難関大/選抜: {len(elite)}")

    print("\n[SOLVING] 最適化を実行中...")
    result_df, status = build_and_solve(rooms, teachers, lessons, time_limit_sec=30.0)

    if result_df is not None:
        print_timetable(result_df)
    else:
        print(f"\n解が見つかりませんでした。ステータス: {status}")
