"""
学習塾 時間割自動生成システム — Streamlit Web UI
"""

import streamlit as st
import pandas as pd
import io
import math
from scheduler import (
    Room, Teacher, Lesson,
    build_and_solve, slot_to_time,
    SLOT_MINUTES, DAY_START_HOUR, DAY_START_MIN, MAX_SLOT,
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
# ユーティリティ
# ============================================================

def time_to_slot(time_str: str) -> int:
    """'HH:MM' -> スロット番号"""
    h, m = map(int, time_str.strip().split(":"))
    total_min = h * 60 + m
    base_min = DAY_START_HOUR * 60 + DAY_START_MIN
    slot = (total_min - base_min) / SLOT_MINUTES
    return int(slot)


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
                # 配信先生徒数が未指定の場合、配信元の生徒数をデフォルトに
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


def build_gantt_html(df: pd.DataFrame) -> str:
    """結果DFからガントチャート風HTMLを生成"""
    if df is None or df.empty:
        return ""

    # カラーパレット (科目ごと)
    subject_colors = {
        "英語": "#4A90D9", "数学": "#D94A4A", "国語": "#6BBF59",
        "物理": "#D9A74A", "化学": "#9B59B6", "世界史": "#1ABC9C",
        "日本史": "#E67E22", "地理": "#3498DB", "生物": "#2ECC71",
    }
    default_color = "#95A5A6"

    campuses = df["校舎"].unique()
    html_parts = []

    html_parts.append("""
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
    """)

    # 凡例
    used_subjects = df["科目"].unique()
    legend_html = '<div class="legend">'
    for subj in sorted(used_subjects):
        c = subject_colors.get(subj, default_color)
        legend_html += f'<div class="legend-item"><div class="legend-swatch" style="background:{c}"></div>{subj}</div>'
    # 配信授業がある場合は凡例に追加
    has_broadcast = "配信" in df.columns and df["配信"].str.contains("配信", na=False).any()
    if has_broadcast:
        legend_html += (
            '<div class="legend-item">'
            '<div class="legend-swatch" style="background:#4A90D9;border:2px dashed #fff;"></div>'
            '📡 配信（受信側）'
            '</div>'
        )
    legend_html += '</div>'
    html_parts.append(legend_html)

    # 時間軸: 9:30-21:30 を 30分刻みで表示
    time_labels = []
    for slot in range(0, MAX_SLOT + 1, 6):  # 6 slots = 30 min
        time_labels.append((slot, slot_to_time(slot)))
    num_cols = len(time_labels) - 1  # 表示列数

    for campus in campuses:
        sub = df[df["校舎"] == campus]
        rooms_in_campus = sorted(sub["教室"].unique())

        html_parts.append(f'<div class="gantt-container">')
        html_parts.append(f'<div class="gantt-title">📍 {campus}</div>')
        html_parts.append('<table class="gantt-table">')

        # ヘッダー行
        html_parts.append('<tr><th class="gantt-label">教室</th>')
        for k in range(num_cols):
            html_parts.append(f'<th class="time-header">{time_labels[k][1]}</th>')
        html_parts.append('</tr>')

        for room in rooms_in_campus:
            room_lessons = sub[sub["教室"] == room]
            html_parts.append(f'<tr><td class="gantt-label">{room.split("_")[-1] if "_" in room else room}</td>')
            # 1つの td に全バーを描画
            html_parts.append(f'<td colspan="{num_cols}" style="position:relative;">')
            for _, row in room_lessons.iterrows():
                s = row["開始slot"]
                e = row["終了slot"]
                left_pct = (s / MAX_SLOT) * 100
                width_pct = ((e - s) / MAX_SLOT) * 100
                color = subject_colors.get(row["科目"], default_color)
                label = row["講座名"]
                if len(label) > 12:
                    label = label[:11] + "…"
                kind = ""
                if "テスト" in row["種別"]:
                    kind = "🔸" if "テスト)" in row["種別"] else "🔹"
                    if "テスト)" in row["種別"]:
                        color = "#7f8c8d"

                # 配信の視覚表示
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
# メインUI
# ============================================================
def main():
    st.title("📚 学習塾 時間割自動生成システム")
    st.caption("Google OR-Tools CP-SAT Solver でスケジュールを最適化")

    # ----------------------------------------------------------
    # サイドバー: データ入力
    # ----------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ 設定")

        data_mode = st.radio(
            "データ入力方法",
            ["📄 CSVアップロード", "🧪 サンプルデータで試す"],
            index=1,
        )

        st.divider()
        time_limit = st.slider("ソルバー制限時間 (秒)", 5, 120, 30, step=5)

    # ----------------------------------------------------------
    # データの取得
    # ----------------------------------------------------------
    rooms, teachers, lessons = None, None, None

    if data_mode == "📄 CSVアップロード":
        st.header("📁 CSVファイルをアップロード")

        # テンプレートダウンロード
        st.subheader("テンプレートのダウンロード")
        col1, col2, col3 = st.columns(3)

        def read_csv_with_bom(path: str) -> bytes:
            """CSVをUTF-8 BOM付きで読み込み（Excel文字化け防止）"""
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return content.encode("utf-8-sig")

        with col1:
            st.download_button("🏫 教室テンプレート", read_csv_with_bom("templates/rooms.csv"),
                             "rooms_template.csv", "text/csv")
        with col2:
            st.download_button("👨‍🏫 講師テンプレート", read_csv_with_bom("templates/teachers.csv"),
                             "teachers_template.csv", "text/csv")
        with col3:
            st.download_button("📖 授業テンプレート", read_csv_with_bom("templates/lessons.csv"),
                             "lessons_template.csv", "text/csv")

        st.divider()

        # アップロード
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

                # プレビュー
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

        # サンプルデータの概要
        with st.expander("📊 サンプルデータの概要", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("教室数", len(rooms))
            with col2:
                st.metric("講師数", len(teachers))
            with col3:
                st.metric("授業数", len(lessons))

            # 校舎ごとの教室
            room_data = [{"校舎": r.campus, "教室": r.room_id, "定員": r.capacity} for r in rooms]
            st.dataframe(pd.DataFrame(room_data), use_container_width=True, hide_index=True)

    # ----------------------------------------------------------
    # 最適化の実行
    # ----------------------------------------------------------
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

                # --- ガントチャート ---
                st.subheader("📊 ガントチャート")
                gantt_html = build_gantt_html(result_df)
                st.html(gantt_html)

                # --- 表形式 ---
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
                            campus_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "生徒数": st.column_config.NumberColumn("生徒数", format="%d人"),
                            }
                        )

                # --- CSV ダウンロード ---
                st.subheader("💾 結果のダウンロード")
                csv_buf = io.StringIO()
                result_df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
                st.download_button(
                    "📥 結果CSVをダウンロード",
                    csv_buf.getvalue(),
                    "timetable_result.csv",
                    "text/csv",
                )
            else:
                st.error(
                    f"❌ 解が見つかりませんでした（ステータス: {status}）。\n\n"
                    "制約条件が厳しすぎる可能性があります。教室数や講師のシフトを見直してください。"
                )
    else:
        st.warning("データを読み込んでから「時間割を生成する」ボタンを押してください。")


if __name__ == "__main__":
    main()
