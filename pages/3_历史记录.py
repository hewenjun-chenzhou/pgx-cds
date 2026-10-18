"""
pages/3_历史记录.py — Step 7
历史分析记录页面
"""

import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="历史记录 · PGx CDS", page_icon="📁", layout="wide")

st.title("📁 历史分析记录")

# ── 初始化 ─────────────────────────────────────────────────
history = st.session_state.get("history", [])

if not history:
    st.info("📭 暂无历史记录，完成分析后自动保存。")
    st.page_link("app.py", label="← 返回首页", use_container_width=True)
    st.stop()

# ── 统计摘要 ─────────────────────────────────────────────────
col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("总记录数", len(history))
col_s2.metric("高风险DDI（合计）", sum(h.get("high_risk_ddi", 0) for h in history))
col_s3.metric("紧急干预（合计）", sum(h.get("urgent_count", 0) for h in history))

st.divider()

# ── 记录列表 ─────────────────────────────────────────────────
st.subheader("📋 分析记录列表")

for i, entry in enumerate(history):
    meds_str = ", ".join(entry.get("meds", [])[:5])
    if len(entry.get("meds", [])) > 5:
        meds_str += f" ... (+{len(entry['meds']) - 5}种)"

    phenotypes_str = ", ".join(
        f"{g}: {ph}" for g, ph in entry.get("phenotypes", {}).items()
    )
    gen_time = entry.get("generated_at", "")[:19] if entry.get("generated_at") else "—"

    with st.expander(
        f"**{entry.get('case_id', 'N/A')}** · {entry.get('diagnosis', '未知诊断')} · "
        f"{entry.get('population', '')} · {gen_time} · "
        f"⚠️DDI:{entry.get('high_risk_ddi', 0)} 🚨:{entry.get('urgent_count', 0)}"
    ):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**病例ID:** {entry.get('case_id', 'N/A')}")
            st.markdown(f"**诊断:** {entry.get('diagnosis', '未知')}")
            st.markdown(f"**人群:** {entry.get('population', '未知')}")
            st.markdown(f"**用药:** {meds_str}")
        with c2:
            st.markdown(f"**表型推断:** {phenotypes_str}")
            st.markdown(f"**生成时间:** {gen_time}")
            st.markdown(f"**高风险DDI数:** {entry.get('high_risk_ddi', 0)}")
            st.markdown(f"**紧急干预数:** {entry.get('urgent_count', 0)}")

        # 加载历史报告
        if i < len(st.session_state.get("history", [])):
            hist_entry = st.session_state["history"][i]
            if "report" in hist_entry:
                st.warning("报告详情请重新进入「分析报告」页面查看（报告存储于会话状态）。")

st.divider()

# ── 工具栏 ─────────────────────────────────────────────────
col_b1, col_b2 = st.columns([1, 4])
with col_b1:
    if st.button("🗑️ 清空历史记录", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()
with col_b2:
    st.caption("注意：历史记录存储在当前浏览器会话中，关闭页面后将清除。")

st.page_link("app.py", label="← 返回首页", use_container_width=False)
