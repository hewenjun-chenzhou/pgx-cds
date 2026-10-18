"""
app.py — Step 7
Streamlit 主入口

多页面架构：
  /           → 欢迎页（快速入口）
  /pages/1_病例录入.py  → 病例信息输入
  /pages/2_分析报告.py → CDS 报告展示
  /pages/3_历史记录.py → 历史分析记录
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# ── 路径设置 ─────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Session State 初始化 ────────────────────────────────────
def _init_state():
    defaults = {
        "case_id": "",
        "diagnosis": "",
        "current_meds": [],
        "population": "东亚",
        "chief_complaint": "",
        "age": None,
        "gender": "",
        "signals_cyp2d6": "",
        "signals_cyp2c19": "",
        "report": None,
        "report_dict": None,
        "html_report": None,
        "history": [],          # list of dicts
        "submitted": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── 侧边栏导航 ──────────────────────────────────────────────
st.set_page_config(
    page_title="PGx CDS · 药物基因组学临床决策支持",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "🏠 首页": "home",
    "📝 病例录入": "pages/1_病例录入",
    "📊 分析报告": "pages/2_分析报告",
    "📁 历史记录": "pages/3_历史记录",
}

with st.sidebar:
    st.title("🧬 PGx CDS")
    st.caption("药物基因组学临床决策支持系统")
    st.divider()

    selection = st.radio(
        "导航",
        list(PAGES.keys()),
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"版本 1.0.0  ·  {datetime.now().strftime('%Y-%m-%d')}")

# ── 路由 ────────────────────────────────────────────────────
page = PAGES[selection]

if page == "home":
    pass  # 主页内容在下方直接渲染

elif page.startswith("pages/"):
    from pathlib import Path as P
    # Streamlit 多页面自动从 web/pages 目录读取
    # 这里通过重定向确保 navigation 正常
    pass

# ── 主页内容 ────────────────────────────────────────────────
if page == "home" or selection == "🏠 首页":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("🧬 PGx CDS 系统")
        st.header("药物基因组学临床决策支持平台")

        st.markdown("""
        ### 功能概览

        本系统基于 **CYP2D6** 和 **CYP2C19** 两大药物代谢基因，
        为临床医生提供：

        - 🔬 **表型推断** — 从基因检测结果或临床信号推断患者代谢表型（PM/IM/NM/UM）
        - 💊 **药物推荐** — CPIC/FDA 级别药物剂量调整建议
        - ⚠️ **DDI 检测** — 药物相互作用风险评估
        - 🌏 **人群分析** — 人群表型频率分布参考
        - 📊 **可视化报告** — 交互式 HTML 报告导出
        """)

        st.divider()

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.page_link("app.py", label="📝 新建病例", icon="📝", use_container_width=True)
        with col_b:
            st.page_link("app.py", label="📁 历史记录", icon="📁", use_container_width=True)
        with col_c:
            st.page_link("app.py", label="ℹ️ 使用说明", icon="ℹ️", use_container_width=True)

    with col2:
        st.subheader("系统状态")
        try:
            from src.data_loader import get_cache
            cache = get_cache()
            st.success("✅ 数据加载正常")
            st.markdown(f"""
            - **数据覆盖范围**
              - CYP2D6 底物药物: `{len(cache.drugs)}` 种
              - CYP2C19 底物药物: `{len(cache.drugs_cyp2c19)}` 种
              - DDI数据: `{len(cache.drug_interactions)}` 条
              - 表型频率: 已加载
            - **覆盖的治疗领域**: 心血管、精神、肿瘤、镇痛、抗菌等
            """)
        except Exception as e:
            st.error(f"❌ 数据加载异常: {e}")

        st.divider()
        st.caption("数据来源: CPIC / PharmGKB / FDA 药物基因组学标签")
