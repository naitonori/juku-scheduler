"""
学習塾 時間割自動生成システム — Streamlit Web UI
"""

import streamlit as st
import pandas as pd
import io
import math
from datetime import date, timedelta
from scheduler import (
    Room, Teacher, Lesson,
    build_and_solve, slot_to_time,
    SLOT_MINUTES, DAY_START_HOUR, DAY_START_MIN, MAX_SLOT,
    IntensiveTeacher, IntensiveLesson,
    build_and_solve_intensive, build_and_solve_intensive_phased,
    diagnose_intensive,
    intensive_slot_to_time, intensive_time_to_slot,
)

# ============================================================
# ページ設定
# ============================================================
st.set_page_config(
    page_title="学習塾 時間割ジェネレーター",
    page_icon="📚",
    layout="wide",
)

# ============================================================
# 共通カラーパレット
# ============================================================
SUBJECT_COLORS = {
    "英語": "#4A90D9", "数学": "#D94A4A", "国語": "#6BBF59",
    "物理": "#D9A74A", "化学": "#9B59B6", "世界史": "#1ABC9C",
    "日本史": "#E67E22", "地理": "#3498DB", "生物": "#2ECC71",
}
DEFAULT_COLOR = "#95A5A6"

# ============================================================
# ユーティリティ
# ============================================================

def time_to_slot(time_str: str) -> int:
    """'HH:MM' -> スロット番号"""
    h, m = map(int, time_str.strip().split(":"))
    total_min = h * 60 + m
    base_min = DAY_START_HOUR * 60 + DAY_START_MIN
    slot = (total_min - base_min) / SLOT_MINUTES
    return int(slot)


def read_csv_with_bom(path: str) -> bytes:
    """CSVをUTF-8 BOM付きで読み込み（Excel文字化け防止）"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content.encode("utf-8-sig")


def parse_date_str(date_str: str, year: int) -> date:
    """'M/D' 形式の日付文字列をdateに変換"""
    parts = str(date_str).strip().split("/")
    return date(year, int(parts[0]), int(parts[1]))


def parse_rooms_csv(df: pd.DataFrame) -> list[Room]:
    rooms = []
    for _, row in df.iterrows():
        rooms.append(Room(
            campus=str(row["校舎"]).strip(),
            room_id=str(row["教室ID"]).strip(),
            capacity=int(row["定員"]),
        ))
    return rooms


def parse_teachers_csv(df: pd.DataFrame) -> list[Teacher]:
    teachers = []
    for _, row in df.iterrows():
        avail = []
        for k in range(1, 4):  # 最大3区間
            sc = f"稼働開始{k}"
            ec = f"稼働終了{k}"
            if sc in df.columns and ec in df.columns:
                sv = row.get(sc)
                ev = row.get(ec)
                if pd.notna(sv) and pd.notna(ev) and str(sv).strip() and str(ev).strip():
                    avail.append((time_to_slot(str(sv)), time_to_slot(str(ev))))
        if not avail:
            avail = [(0, MAX_SLOT)]
        teachers.append(Teacher(
            teacher_id=str(row["講師ID"]).strip(),
            name=str(row["名前"]).strip(),
            available=avail,
        ))
    return teachers


def parse_lessons_csv(df: pd.DataFrame) -> list[Lesson]:
    lessons = []
    for _, row in df.iterrows():
        name = str(row["講座名"]).strip()
        dur_min = int(row["授業時間(分)"]) if pd.notna(row["授業時間(分)"]) else 0
        dur_slots = dur_min // SLOT_MINUTES

        is_tz = str(row.get("テストゼミ", "")).strip() in ("はい", "TRUE", "true", "1", "Yes", "yes")
        test_min = int(row["テスト時間(分)"]) if pd.notna(row.get("テスト時間(分)")) and str(row.get("テスト時間(分)")).strip() else 0
        expl_min = int(row["解説時間(分)"]) if pd.notna(row.get("解説時間(分)")) and str(row.get("解説時間(分)")).strip() else 0

        fixed = None
        fv = row.get("固定開始時刻")
        if pd.notna(fv) and str(fv).strip():
            fixed = time_to_slot(str(fv).strip())

        tags = []
        if "最難関大" in name:
            tags.append("最難関大")
        if "選抜" in name:
            tags.append("選抜")

        # --- 配信先校舎の解析 ---
        broadcast_targets = []
        bt_raw = row.get("配信先校舎")
        bs_raw = row.get("配信先生徒数")
        if pd.notna(bt_raw) and str(bt_raw).strip():
            target_campuses = [c.strip() for c in str(bt_raw).split(";") if c.strip()]
            if pd.notna(bs_raw) and str(bs_raw).strip():
                target_students = [int(s.strip()) for s in str(bs_raw).split(";") if s.strip()]
            else:
                target_students = [int(row["生徒数"])] * len(target_campuses)
            if len(target_students) != len(target_campuses):
                raise ValueError(
                    f"授業 '{name}': 配信先校舎の数({len(target_campuses)})と"
                    f"配信先生徒数の数({len(target_students)})が一致しません。"
                )
            broadcast_targets = list(zip(target_campuses, target_students))

        lessons.append(Lesson(
            lesson_id=str(row["授業ID"]).strip(),
            name=name,
            campus=str(row["校舎"]).strip(),
            subject=str(row["科目"]).strip(),
            teacher_id=str(row["講師ID"]).strip(),
            duration_slots=dur_slots,
            num_students=int(row["生徒数"]),
            is_test_zemi=is_tz,
            test_slots=test_min // SLOT_MINUTES,
            explanation_slots=expl_min // SLOT_MINUTES,
            fixed_start=fixed,
            tags=tags,
            broadcast_targets=broadcast_targets,
        ))
    return lessons


def parse_intensive_teachers_csv(df: pd.DataFrame, date_list: list[date]) -> list[IntensiveTeacher]:
    """講習モード講師CSVをパース。同一講師で複数行（日による時間帯違い）に対応"""
    date_to_idx = {d: i for i, d in enumerate(date_list)}
    year = date_list[0].year

    # 講師IDごとにグルーピング
    grouped = {}
    for _, row in df.iterrows():
        tid = str(row["講師ID"]).strip()
        name = str(row["名前"]).strip()
        start_date = parse_date_str(row["開始日"], year)
        end_date = parse_date_str(row["終了日"], year)
        start_time = intensive_time_to_slot(str(row["稼働開始"]).strip())
        end_time = intensive_time_to_slot(str(row["稼働終了"]).strip())

        if tid not in grouped:
            grouped[tid] = {"name": name, "slots": {}}

        # 該当日のインデックスを算出して登録
        d = start_date
        while d <= end_date:
            if d in date_to_idx:
                grouped[tid]["slots"][date_to_idx[d]] = (start_time, end_time)
            d += timedelta(days=1)

    teachers = []
    for tid, info in grouped.items():
        teachers.append(IntensiveTeacher(
            teacher_id=tid,
            name=info["name"],
            available_slots=info["slots"],
        ))
    return teachers


def parse_intensive_lessons_csv(df: pd.DataFrame, date_list: list[date]) -> list[IntensiveLesson]:
    """講習モード授業CSVをパース"""
    date_to_idx = {d: i for i, d in enumerate(date_list)}
    year = date_list[0].year

    lessons = []
    for _, row in df.iterrows():
        name = str(row["講座名"]).strip()
        dur_min = int(row["1回の授業時間(分)"])
        dur_slots = dur_min // SLOT_MINUTES

        start_date = parse_date_str(row["開始日"], year)
        end_date = parse_date_str(row["終了日"], year)

        avail_days = []
        d = start_date
        while d <= end_date:
            if d in date_to_idx:
                avail_days.append(date_to_idx[d])
            d += timedelta(days=1)

        ng_group = str(row.get("NGグループ", "")).strip() if pd.notna(row.get("NGグループ")) else ""

        # 配信先
        broadcast_targets = []
        bt_raw = row.get("配信先校舎")
        bs_raw = row.get("配信先生徒数")
        if pd.notna(bt_raw) and str(bt_raw).strip():
            target_campuses = [c.strip() for c in str(bt_raw).split(";") if c.strip()]
            if pd.notna(bs_raw) and str(bs_raw).strip():
                target_students = [int(s.strip()) for s in str(bs_raw).split(";") if s.strip()]
            else:
                target_students = [int(row["生徒数"])] * len(target_campuses)
            broadcast_targets = list(zip(target_campuses, target_students))

        # 時間帯制限 (新機能)
        time_band = str(row.get("時間帯制限", "")).strip() if pd.notna(row.get("時間帯制限")) else ""
        # NG優先度 (新機能): 0=ハード, 1=ソフト(重), 2=ソフト(軽)
        ng_priority = int(row.get("NG優先度", 0)) if pd.notna(row.get("NG優先度")) and str(row.get("NG優先度")).strip() else 0

        lessons.append(IntensiveLesson(
            lesson_id=str(row["授業ID"]).strip(),
            name=name,
            campus=str(row["校舎"]).strip(),
            subject=str(row["科目"]).strip(),
            teacher_id=str(row["講師ID"]).strip(),
            duration_slots=dur_slots,
            num_students=int(row["生徒数"]),
            num_sessions=int(row["授業回数"]),
            available_days=avail_days,
            ng_group=ng_group,
            broadcast_targets=broadcast_targets,
            time_band=time_band,
            ng_group_priority=ng_priority,
        ))
    return lessons


# ============================================================
# ガントチャート（通常モード）
# ============================================================
def build_gantt_html(df: pd.DataFrame) -> str:
    """結果DFからガントチャート風HTMLを生成"""
    if df is None or df.empty:
        return ""

    campuses = df["校舎"].unique()
    html_parts = []

    html_parts.append(_gantt_css())

    # 凡例
    html_parts.append(_build_legend(df))

    # 時間軸: 9:30-21:30 を 30分刻みで表示
    time_labels = []
    for slot in range(0, MAX_SLOT + 1, 6):  # 6 slots = 30 min
        time_labels.append((slot, slot_to_time(slot)))
    num_cols = len(time_labels) - 1

    for campus in campuses:
        sub = df[df["校舎"] == campus]
        rooms_in_campus = sorted(sub["教室"].unique())

        html_parts.append(f'<div class="gantt-container">')
        html_parts.append(f'<div class="gantt-title">📍 {campus}</div>')
        html_parts.append('<table class="gantt-table">')

        html_parts.append('<tr><th class="gantt-label">教室</th>')
        for k in range(num_cols):
            html_parts.append(f'<th class="time-header">{time_labels[k][1]}</th>')
        html_parts.append('</tr>')

        for room in rooms_in_campus:
            room_lessons = sub[sub["教室"] == room]
            html_parts.append(f'<tr><td class="gantt-label">{room.split("_")[-1] if "_" in room else room}</td>')
            html_parts.append(f'<td colspan="{num_cols}" style="position:relative;">')
            for _, row in room_lessons.iterrows():
                s = row["開始slot"]
                e = row["終了slot"]
                left_pct = (s / MAX_SLOT) * 100
                width_pct = ((e - s) / MAX_SLOT) * 100
                color = SUBJECT_COLORS.get(row["科目"], DEFAULT_COLOR)
                label = row["講座名"]
                if len(label) > 12:
                    label = label[:11] + "…"
                kind = ""
                if "テスト" in row["種別"]:
                    kind = "🔸" if "テスト)" in row["種別"] else "🔹"
                    if "テスト)" in row["種別"]:
                        color = "#7f8c8d"

                broadcast_info = row.get("配信", "") if "配信" in row.index else ""
                extra_style = ""
                broadcast_icon = ""
                if broadcast_info:
                    if "配信元" in broadcast_info and "より" not in broadcast_info:
                        broadcast_icon = "📡"
                    elif "より配信" in broadcast_info:
                        broadcast_icon = "📡"
                        extra_style = "border:2px dashed rgba(255,255,255,0.7);"

                tooltip = f'{row["講座名"]} ({row["開始"]}~{row["終了"]}) {row["講師"]} {row["種別"]}'
                if broadcast_info:
                    tooltip += f' [{broadcast_info}]'
                html_parts.append(
                    f'<div class="gantt-bar" style="left:{left_pct}%;width:{width_pct}%;background:{color};{extra_style}" '
                    f'title="{tooltip}">{broadcast_icon}{kind}{label}</div>'
                )
            html_parts.append('</td></tr>')

        html_parts.append('</table></div>')

    return "\n".join(html_parts)


# ============================================================
# ガントチャート（講習モード — 日付別）
# ============================================================
def build_intensive_gantt_html(df: pd.DataFrame, selected_date: str) -> str:
    """講習モード結果DFから、指定日付のガントチャートHTMLを生成"""
    if df is None or df.empty:
        return ""

    day_df = df[df["日付"] == selected_date]
    if day_df.empty:
        return f'<p>📅 {selected_date} には授業がありません。</p>'

    # 時間軸の範囲を算出
    min_slot = int(day_df["開始slot"].min())
    max_slot = int(day_df["終了slot"].max())
    # 30分刻みで丸める
    axis_start = (min_slot // 6) * 6
    axis_end = ((max_slot + 5) // 6) * 6
    total_range = max(axis_end - axis_start, 6)

    time_labels = []
    for slot in range(axis_start, axis_end + 1, 6):
        time_labels.append((slot, intensive_slot_to_time(slot)))
    num_cols = len(time_labels) - 1

    html_parts = [_gantt_css(), _build_legend(day_df)]

    campuses = day_df["校舎"].unique()
    for campus in campuses:
        sub = day_df[day_df["校舎"] == campus]
        rooms_in_campus = sorted(sub["教室"].unique())

        html_parts.append(f'<div class="gantt-container">')
        html_parts.append(f'<div class="gantt-title">📍 {campus}</div>')
        html_parts.append('<table class="gantt-table">')

        html_parts.append('<tr><th class="gantt-label">教室</th>')
        for k in range(num_cols):
            html_parts.append(f'<th class="time-header">{time_labels[k][1]}</th>')
        html_parts.append('</tr>')

        for room in rooms_in_campus:
            room_lessons = sub[sub["教室"] == room]
            html_parts.append(f'<tr><td class="gantt-label">{room.split("_")[-1] if "_" in room else room}</td>')
            html_parts.append(f'<td colspan="{num_cols}" style="position:relative;">')
            for _, row in room_lessons.iterrows():
                s = row["開始slot"]
                e = row["終了slot"]
                left_pct = ((s - axis_start) / total_range) * 100
                width_pct = ((e - s) / total_range) * 100
                color = SUBJECT_COLORS.get(row["科目"], DEFAULT_COLOR)
                label = row["講座名"]
                if len(label) > 12:
                    label = label[:11] + "…"
                session_info = f' [{row["回"]}]' if "回" in row.index else ""
                tooltip = f'{row["講座名"]}{session_info} ({row["開始"]}~{row["終了"]}) {row["講師"]}'
                html_parts.append(
                    f'<div class="gantt-bar" style="left:{left_pct}%;width:{width_pct}%;background:{color};" '
                    f'title="{tooltip}">{label}</div>'
                )
            html_parts.append('</td></tr>')

        html_parts.append('</table></div>')

    return "\n".join(html_parts)


def _gantt_css() -> str:
    return """
    <style>
    .gantt-container { margin-bottom: 30px; }
    .gantt-title { font-size: 18px; font-weight: bold; margin: 15px 0 8px 0; color: #2c3e50; }
    .gantt-table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    .gantt-table th { background: #34495e; color: #fff; padding: 6px 4px; font-size: 11px;
                      text-align: center; border: 1px solid #2c3e50; }
    .gantt-table td { padding: 0; height: 38px; border: 1px solid #ddd; position: relative;
                      vertical-align: middle; }
    .gantt-label { width: 130px; min-width: 130px; padding: 4px 6px !important;
                   font-size: 12px; font-weight: 500; background: #f8f9fa; }
    .gantt-bar { position: absolute; top: 4px; bottom: 4px; border-radius: 4px; color: #fff;
                 font-size: 10px; display: flex; align-items: center; justify-content: center;
                 overflow: hidden; white-space: nowrap; text-overflow: ellipsis; padding: 0 4px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.2); cursor: default; z-index: 1; }
    .gantt-bar:hover { opacity: 0.85; z-index: 10; }
    .time-header { font-size: 10px; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 20px 0; }
    .legend-item { display: flex; align-items: center; gap: 4px; font-size: 12px; }
    .legend-swatch { width: 14px; height: 14px; border-radius: 3px; }
    </style>
    """


def _build_legend(df: pd.DataFrame) -> str:
    used_subjects = df["科目"].unique()
    legend_html = '<div class="legend">'
    for subj in sorted(used_subjects):
        c = SUBJECT_COLORS.get(subj, DEFAULT_COLOR)
        legend_html += f'<div class="legend-item"><div class="legend-swatch" style="background:{c}"></div>{subj}</div>'
    has_broadcast = "配信" in df.columns and df["配信"].str.contains("配信", na=False).any()
    if has_broadcast:
        legend_html += (
            '<div class="legend-item">'
            '<div class="legend-swatch" style="background:#4A90D9;border:2px dashed #fff;"></div>'
            '📡 配信（受信側）'
            '</div>'
        )
    legend_html += '</div>'
    return legend_html


# ============================================================
# 通常モード UI
# ============================================================
def run_regular_mode(data_mode: str, time_limit: int):
    rooms, teachers, lessons = None, None, None

    if data_mode == "📄 CSVアップロード":
        st.header("📁 CSVファイルをアップロード")

        st.subheader("テンプレートのダウンロード")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("🏫 教室テンプレート", read_csv_with_bom("templates/rooms.csv"),
                             "rooms_template.csv", "application/octet-stream")
        with col2:
            st.download_button("👨‍🏫 講師テンプレート", read_csv_with_bom("templates/teachers.csv"),
                             "teachers_template.csv", "application/octet-stream")
        with col3:
            st.download_button("📖 授業テンプレート", read_csv_with_bom("templates/lessons.csv"),
                             "lessons_template.csv", "application/octet-stream")

        st.divider()

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            rooms_file = st.file_uploader("🏫 教室CSV", type=["csv"], key="rooms_csv")
        with col_b:
            teachers_file = st.file_uploader("👨‍🏫 講師CSV", type=["csv"], key="teachers_csv")
        with col_c:
            lessons_file = st.file_uploader("📖 授業CSV", type=["csv"], key="lessons_csv")

        if rooms_file and teachers_file and lessons_file:
            try:
                rooms_df = pd.read_csv(rooms_file)
                teachers_df = pd.read_csv(teachers_file)
                lessons_df = pd.read_csv(lessons_file)

                rooms = parse_rooms_csv(rooms_df)
                teachers = parse_teachers_csv(teachers_df)
                lessons = parse_lessons_csv(lessons_df)

                st.success(f"✅ 読み込み完了: 教室{len(rooms)}件 / 講師{len(teachers)}件 / 授業{len(lessons)}件")

                with st.expander("📊 読み込みデータのプレビュー"):
                    tab1, tab2, tab3 = st.tabs(["教室", "講師", "授業"])
                    with tab1:
                        st.dataframe(rooms_df, use_container_width=True)
                    with tab2:
                        st.dataframe(teachers_df, use_container_width=True)
                    with tab3:
                        st.dataframe(lessons_df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ CSV解析エラー: {e}")
        else:
            st.info("3つのCSVファイルをすべてアップロードしてください。")

    else:  # サンプルデータ
        st.info("🧪 サンプルデータで最適化を実行します。")
        from scheduler import generate_mock_data
        rooms, teachers, lessons = generate_mock_data()

        with st.expander("📊 サンプルデータの概要", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("教室数", len(rooms))
            with col2:
                st.metric("講師数", len(teachers))
            with col3:
                st.metric("授業数", len(lessons))
            room_data = [{"校舎": r.campus, "教室": r.room_id, "定員": r.capacity} for r in rooms]
            st.dataframe(pd.DataFrame(room_data), use_container_width=True, hide_index=True)

    # 最適化
    st.divider()

    if rooms and teachers and lessons:
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            run_btn = st.button("🚀 時間割を生成する", type="primary", use_container_width=True)
        with col_info:
            elite = [l for l in lessons if any(t in l.name for t in ["最難関大", "選抜"])]
            tz = [l for l in lessons if l.is_test_zemi]
            fixed = [l for l in lessons if l.fixed_start is not None]
            st.caption(
                f"授業 {len(lessons)}件 "
                f"（最難関大/選抜: {len(elite)} / テストゼミ: {len(tz)} / 時間固定: {len(fixed)}）"
            )

        if run_btn:
            with st.spinner("⏳ 最適化を実行中..."):
                try:
                    result_df, status = build_and_solve(rooms, teachers, lessons, time_limit)
                except ValueError as e:
                    st.error(f"❌ データエラー: {e}")
                    return

            if result_df is not None and not result_df.empty:
                st.success(f"✅ 最適化完了！ ステータス: **{status}**")

                st.subheader("📊 ガントチャート")
                gantt_html = build_gantt_html(result_df)
                st.html(gantt_html)

                st.subheader("📋 時間割一覧")
                campuses = result_df["校舎"].unique()
                tabs = st.tabs([f"📍 {c}" for c in campuses])
                for tab, campus in zip(tabs, campuses):
                    with tab:
                        display_cols = ["講座名", "種別", "科目", "教室", "開始", "終了", "講師", "生徒数"]
                        if "配信" in result_df.columns:
                            display_cols.append("配信")
                        campus_df = result_df[result_df["校舎"] == campus][display_cols]
                        st.dataframe(
                            campus_df, use_container_width=True, hide_index=True,
                            column_config={"生徒数": st.column_config.NumberColumn("生徒数", format="%d人")},
                        )

                st.subheader("💾 結果のダウンロード")
                csv_buf = io.BytesIO()
                result_df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
                st.download_button("📥 結果CSVをダウンロード", csv_buf.getvalue(),
                                 "timetable_result.csv", "application/octet-stream")
            else:
                st.error(
                    f"❌ 解が見つかりませんでした（ステータス: {status}）。\n\n"
                    "制約条件が厳しすぎる可能性があります。教室数や講師のシフトを見直してください。"
                )
    else:
        st.warning("データを読み込んでから「時間割を生成する」ボタンを押してください。")


# ============================================================
# 夏期講習モード UI
# ============================================================
def run_intensive_mode(data_mode: str, time_limit: int):
    rooms, teachers, lessons, date_list = None, None, None, None

    # --- 期間設定 ---
    st.header("📅 講習期間の設定")
    col_s, col_e = st.columns(2)
    with col_s:
        period_start = st.date_input("開始日", value=date(2026, 7, 20), key="intensive_start")
    with col_e:
        period_end = st.date_input("終了日", value=date(2026, 8, 30), key="intensive_end")

    if period_start >= period_end:
        st.error("終了日は開始日より後にしてください。")
        return

    num_days = (period_end - period_start).days + 1
    date_list = [period_start + timedelta(days=d) for d in range(num_days)]
    st.caption(f"講習期間: {period_start.strftime('%Y/%m/%d')} 〜 {period_end.strftime('%Y/%m/%d')}（{num_days}日間）")

    st.divider()

    if data_mode == "📄 CSVアップロード":
        st.header("📁 CSVファイルをアップロード")

        # テンプレートダウンロード
        st.subheader("テンプレートのダウンロード")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("🏫 教室テンプレート", read_csv_with_bom("templates/rooms.csv"),
                             "rooms_template.csv", "application/octet-stream", key="int_rooms_dl")
        with col2:
            st.download_button("👨‍🏫 講師テンプレート（講習用）",
                             read_csv_with_bom("templates/teachers_intensive.csv"),
                             "teachers_intensive_template.csv", "application/octet-stream")
        with col3:
            st.download_button("📖 授業テンプレート（講習用）",
                             read_csv_with_bom("templates/lessons_intensive.csv"),
                             "lessons_intensive_template.csv", "application/octet-stream")

        # 入力フォーマット説明
        with st.expander("📝 CSVフォーマットの説明"):
            st.markdown("""
**講師CSV（講習用）の列:**
| 列名 | 説明 | 例 |
|---|---|---|
| 講師ID | 一意のID | T01 |
| 名前 | 講師名 | 佐藤先生 |
| 開始日 | 出講開始日（M/D形式） | 7/20 |
| 終了日 | 出講終了日（M/D形式） | 8/10 |
| 稼働開始 | 1日の稼働開始時刻 | 15:00 |
| 稼働終了 | 1日の稼働終了時刻 | 21:00 |

> 同一講師で複数行を書くと、期間ごとに異なる稼働時間を設定できます。

**授業CSV（講習用）の列:**
| 列名 | 説明 | 例 |
|---|---|---|
| 授業ID | 一意のID | L01 |
| 講座名 | 講座の名称 | 数学講習 |
| 校舎 | 実施校舎 | 代々木校 |
| 科目 | 科目名 | 数学 |
| 講師ID | 担当講師のID | T01 |
| 1回の授業時間(分) | 1回あたりの授業時間 | 120 |
| 生徒数 | 受講生徒数 | 20 |
| 授業回数 | 期間内の総授業回数 | 8 |
| NGグループ | 同時開講NGのグループ | A |
| 開始日 | 講座開始日（M/D形式） | 7/25 |
| 終了日 | 講座終了日（M/D形式） | 8/10 |
| 配信先校舎 | 配信先（;区切り） | |
| 配信先生徒数 | 配信先生徒数（;区切り） | |
| 時間帯制限 | 配置可能な時間帯（HH:MM-HH:MM） | 15:00-18:00 |
| NG優先度 | 0=絶対NG, 1=できれば避ける, 2=許容 | 0 |
            """)

        st.divider()

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            rooms_file = st.file_uploader("🏫 教室CSV", type=["csv"], key="int_rooms_csv")
        with col_b:
            teachers_file = st.file_uploader("👨‍🏫 講師CSV（講習用）", type=["csv"], key="int_teachers_csv")
        with col_c:
            lessons_file = st.file_uploader("📖 授業CSV（講習用）", type=["csv"], key="int_lessons_csv")

        if rooms_file and teachers_file and lessons_file:
            try:
                rooms_df = pd.read_csv(rooms_file)
                teachers_df = pd.read_csv(teachers_file)
                lessons_df = pd.read_csv(lessons_file)

                rooms = parse_rooms_csv(rooms_df)
                teachers = parse_intensive_teachers_csv(teachers_df, date_list)
                lessons = parse_intensive_lessons_csv(lessons_df, date_list)

                st.success(f"✅ 読み込み完了: 教室{len(rooms)}件 / 講師{len(teachers)}件 / 講座{len(lessons)}件")

                with st.expander("📊 読み込みデータのプレビュー"):
                    tab1, tab2, tab3 = st.tabs(["教室", "講師", "授業"])
                    with tab1:
                        st.dataframe(rooms_df, use_container_width=True)
                    with tab2:
                        st.dataframe(teachers_df, use_container_width=True)
                    with tab3:
                        st.dataframe(lessons_df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ CSV解析エラー: {e}")
        else:
            st.info("3つのCSVファイルをすべてアップロードしてください。")

    else:  # サンプルデータ
        st.info("🧪 サンプルデータで最適化を実行します。")
        from scheduler import generate_intensive_mock_data
        rooms, teachers, lessons, date_list = generate_intensive_mock_data()

        with st.expander("📊 サンプルデータの概要", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("教室数", len(rooms))
            with col2:
                st.metric("講師数", len(teachers))
            with col3:
                st.metric("講座数", len(lessons))
            with col4:
                total_sessions = sum(l.num_sessions for l in lessons)
                st.metric("総授業回数", total_sessions)

            # 講座一覧
            lesson_data = [{
                "講座名": l.name, "科目": l.subject, "講師": l.teacher_id,
                "授業時間": f"{l.duration_slots * SLOT_MINUTES}分",
                "授業回数": l.num_sessions, "生徒数": l.num_students,
                "NGグループ": l.ng_group or "-",
            } for l in lessons]
            st.dataframe(pd.DataFrame(lesson_data), use_container_width=True, hide_index=True)

    # --- 最適化の実行 ---
    st.divider()

    if rooms and teachers and lessons and date_list:
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            run_btn = st.button("🚀 講習時間割を生成する", type="primary", use_container_width=True)
        with col_info:
            total_sessions = sum(l.num_sessions for l in lessons)
            ng_groups = set(l.ng_group for l in lessons if l.ng_group)
            st.caption(
                f"講座 {len(lessons)}件 / 総授業回数 {total_sessions}回 / "
                f"NGグループ {len(ng_groups)}組 / 期間 {len(date_list)}日間"
            )

        if run_btn:
            # --- 事前診断 ---
            diagnostics = diagnose_intensive(rooms, teachers, lessons, date_list)
            if diagnostics:
                critical = [d for d in diagnostics if d["severity"] == "critical"]
                warns = [d for d in diagnostics if d["severity"] == "warning"]
                if critical:
                    st.error("**事前診断: 致命的な問題が見つかりました**")
                    for diag in critical:
                        st.markdown(f"- **{diag['category']}**: {diag['message']}")
                if warns:
                    with st.expander(f"⚠️ 事前診断: 警告 {len(warns)}件", expanded=False):
                        for diag in warns:
                            st.warning(f"**{diag['category']}**: {diag['message']}")

            # --- 段階的求解 ---
            with st.spinner("⏳ 講習時間割を最適化中... （段階的に制約を緩和しながら最適解を探索します）"):
                try:
                    result_df, status, relaxations = build_and_solve_intensive_phased(
                        rooms, teachers, lessons, date_list, time_limit
                    )
                except ValueError as e:
                    st.error(f"❌ データエラー: {e}")
                    return

            if result_df is not None and not result_df.empty:
                st.success(f"✅ 最適化完了！ ステータス: **{status}**")
                if relaxations:
                    st.info("**制約緩和あり:** " + " / ".join(relaxations))

                # --- 日付選択 ---
                st.subheader("📊 日付別ガントチャート")
                dates_with_lessons = sorted(result_df["日付"].unique(),
                                           key=lambda x: result_df[result_df["日付"]==x]["日付index"].iloc[0])
                selected_date = st.selectbox(
                    "表示する日付を選択",
                    dates_with_lessons,
                    format_func=lambda d: f"{d}（{_weekday_jp(d, date_list)}）"
                )
                gantt_html = build_intensive_gantt_html(result_df, selected_date)
                st.html(gantt_html)

                # --- 表形式 ---
                st.subheader("📋 講習時間割一覧")
                view_mode = st.radio("表示切替", ["日付別", "講座別"], horizontal=True, key="int_view")

                if view_mode == "日付別":
                    tabs = st.tabs([f"📅 {d}" for d in dates_with_lessons])
                    for tab, d in zip(tabs, dates_with_lessons):
                        with tab:
                            display_cols = ["講座名", "科目", "教室", "開始", "終了", "講師", "生徒数", "回"]
                            if "NGグループ" in result_df.columns:
                                display_cols.append("NGグループ")
                            day_df = result_df[result_df["日付"] == d][display_cols]
                            st.dataframe(day_df, use_container_width=True, hide_index=True,
                                        column_config={"生徒数": st.column_config.NumberColumn("生徒数", format="%d人")})
                else:
                    lesson_names = sorted(result_df["講座名"].unique())
                    tabs = st.tabs([f"📖 {n}" for n in lesson_names])
                    for tab, name in zip(tabs, lesson_names):
                        with tab:
                            display_cols = ["日付", "科目", "教室", "開始", "終了", "講師", "生徒数", "回"]
                            lesson_df = result_df[result_df["講座名"] == name][display_cols]
                            st.dataframe(lesson_df, use_container_width=True, hide_index=True)

                # --- CSVダウンロード ---
                st.subheader("💾 結果のダウンロード")
                csv_buf = io.BytesIO()
                result_df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
                st.download_button("📥 結果CSVをダウンロード", csv_buf.getvalue(),
                                 "intensive_timetable_result.csv", "application/octet-stream")
            else:
                st.error(f"❌ 解が見つかりませんでした（ステータス: {status}）")
                if diagnostics:
                    st.subheader("問題の原因分析")
                    for diag in diagnostics:
                        icon = "🔴" if diag["severity"] == "critical" else "🟡"
                        st.markdown(f"{icon} **{diag['category']}**: {diag['message']}")
                st.info(
                    "**対処法:**\n"
                    "- 授業CSVの「NG優先度」列を `1`（ソフト）に変更する\n"
                    "- 授業回数を減らす / 講師の出講可能期間を広げる\n"
                    "- ソルバー制限時間を長くする（120秒以上推奨）"
                )
    else:
        st.warning("データを読み込んでから「講習時間割を生成する」ボタンを押してください。")


def _weekday_jp(date_str: str, date_list: list[date]) -> str:
    """日付文字列(M/D)から曜日を取得"""
    weekdays = ["月", "火", "水", "木", "金", "土", "日"]
    for d in date_list:
        if d.strftime("%m/%d") == date_str:
            return weekdays[d.weekday()]
    return ""


# ============================================================
# メインUI
# ============================================================
def main():
    st.title("📚 学習塾 時間割自動生成システム")
    st.caption("Google OR-Tools CP-SAT Solver でスケジュールを最適化")

    with st.sidebar:
        st.header("⚙️ 設定")

        schedule_mode = st.radio(
            "モード選択",
            ["📅 通常モード（1日）", "🌴 夏期講習モード（期間）"],
            index=0,
        )

        st.divider()

        data_mode = st.radio(
            "データ入力方法",
            ["📄 CSVアップロード", "🧪 サンプルデータで試す"],
            index=1,
        )

        st.divider()
        default_limit = 30 if "通常" in schedule_mode else 60
        time_limit = st.slider("ソルバー制限時間 (秒)", 5, 300, default_limit, step=5)

    if "通常" in schedule_mode:
        run_regular_mode(data_mode, time_limit)
    else:
        run_intensive_mode(data_mode, time_limit)


if __name__ == "__main__":
    main()
