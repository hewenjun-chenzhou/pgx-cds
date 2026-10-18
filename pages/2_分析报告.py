"""
pages/2_分析报告.py
====================
PGx CDS 分析报告页面（按用户 UI 设想重写）

4 板块布局：
1. 大标题 + 病人基本信息
2. 基因检测建议两列（CYP2D6 + CYP2C19）
3. 用药建议（多药品药方表格）
4. 可视化 tabs（基因型 + 推理 + 对比 + 知识库引用）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from pages._agent_section import render_agent_section  # noqa: E402

# ── Matplotlib 中文字体配置 ─────────────────────────────────────
# 此页面所有 matplotlib 图都必须能渲染中文（"后验概率""证据 1"等），
# 否则会出现方框。优先挑已安装的中文字体回写 matplotlib rcParams。
import matplotlib

try:
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Microsoft JhengHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正常显示
except Exception:
    pass

st.set_page_config(
    page_title="分析报告 · PGx CDS",
    page_icon="📊",
    layout="wide",
)

# ── Session state 安全初始化 ──────────────────────────────────
for _k, _v in {
    "history": [],
    "report": None,
    "report_dict": None,
    "html_report": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── 检查 session ─────────────────────────────────────────────
if not st.session_state.get("report"):
    st.warning("⚠️ 尚未生成报告，请先在「📝 病例录入」页面提交分析。")
    st.page_link("app.py", label="← 返回录入", use_container_width=True)
    st.stop()


report = st.session_state["report"]
patient = report.patient
phenotypes = report.phenotypes
drug_recs = report.drug_recs


# ══════════════════════════════════════════════════════════════════
# 板块 1：大标题 + 病人基本信息
# ══════════════════════════════════════════════════════════════════
# 推断最严重表型
critical_phen = None
for gene, phen in phenotypes.items():
    if phen.phenotype in ("PM", "UM"):
        critical_phen = (gene, phen.phenotype)
        break

if critical_phen:
    critical_text = (
        f"严重基因异常：{critical_phen[0]} {critical_phen[1]} "
        f"（置信 {critical_phen[1] and phen.confidence:.0%}）"
    )
elif any(p.phenotype == "IM" for p in phenotypes.values()):
    critical_text = "中度基因异常（IM），建议谨慎用药"
else:
    critical_text = "基因型正常，常规用药"

st.title("📊 " + critical_text)
st.caption(f"病例ID: {patient.case_id}  ·  生成时间: {report.generated_at}")
st.divider()

# 病人基本信息卡片
st.subheader("📋 病人基本信息")
col_basic1, col_basic2, col_basic3, col_basic4 = st.columns(4)
with col_basic1:
    st.metric("病例 ID", patient.case_id)
with col_basic2:
    st.metric("诊断", patient.diagnosis or "未填")
with col_basic3:
    if patient.age:
        st.metric("年龄", f"{patient.age} 岁")
    else:
        st.metric("年龄", "未填")
with col_basic4:
    st.metric("人群", patient.population or "未填")

st.markdown(f"**当前用药**: {patient.current_meds or '未填'}")
if patient.chief_complaint:
    st.markdown(f"**主诉**: {patient.chief_complaint}")

st.divider()


# ══════════════════════════════════════════════════════════════════
# 板块 2：基因检测建议 - 两列
# ══════════════════════════════════════════════════════════════════
st.subheader("🧬 基因检测建议")


def _strip_marker_parens(step_text: str) -> str:
    """把 '测 CYP2C19*2 (splicing defect no function)' 简化为 'CYP2C19*2'"""
    import re
    # 移除括号注释（如 "(splicing defect no function)"）
    return re.sub(r"\s*\(.*?\)\s*", "", step_text).strip()


def get_test_advice(
    phen: str, confidence: float, has_critical_drug: bool,
    evidence_source: str = "none",
) -> tuple[str, str]:
    """
    根据表型 + 置信度 + 关键药物 + 证据来源，返回(建议, 颜色)。

    evidence_source:
      - "clinical"  临床症状（如可待因明确失败）→ 已有强证据，可直接按推断表型用药
      - "blood"     血液标记 CCA 推断（器官功能不全）→ 必须测以区分"基因PM" vs "NM合并器官功能障碍"
      - "mixed"     两者都有
      - "none"      无证据 → 仅人群先验
    """
    if evidence_source == "clinical":
        # 临床症状直接推断（如可待因明确失败）→ 表型已封顶在 0.85
        # APP 目标：帮病人少花钱，置信度高时不再要求测基因
        if phen in ("PM", "UM"):
            return "✅ 可按推断表型用药", "success"
        elif phen == "IM":
            return "⚠️ 建议检测（中等风险）", "warning"
        else:
            return "✅ 无需检测", "success"
    elif evidence_source == "blood":
        # 纯血液标记 → 推断的是"有效代谢能力"（器官功能受损），
        # 不代表真实基因型。NM 患者也可能因肝功能不全而代谢差
        # → 仍建议检测以区分"基因 PM" vs "NM 合并器官功能障碍"
        if phen in ("PM", "UM"):
            return "🩸 建议检测（区分基因型 vs 器官功能障碍）", "warning"
        elif phen == "IM":
            return "⚠️ 建议检测", "warning"
        else:
            return "ℹ️ 可选检测", "info"
    elif evidence_source == "mixed":
        # 临床+血液 都有 → 临床症状优先，但器官功能问题叠加
        if phen in ("PM", "UM"):
            return "✅ 可按推断表型用药（注意器官功能）", "success"
        else:
            return "⚠️ 建议检测", "warning"
    else:  # none
        if confidence >= 0.80:
            if phen in ("PM", "UM"):
                return "⚠️ 建议检测（仅基于先验）", "warning"
            else:
                return "✅ 无需检测", "success"
        elif confidence >= 0.50:
            if phen in ("PM", "UM") or has_critical_drug:
                return "⚠️ 建议检测", "warning"
            else:
                return "ℹ️ 可选检测", "info"
        else:
            if has_critical_drug or phen in ("PM", "UM"):
                return "⚠️ 建议检测（置信低）", "warning"
            else:
                return "ℹ️ 可选检测（无强信号）", "info"


CRITICAL_DRUGS = {
    "CYP2D6": ["阿米替林", "可待因", "曲马多", "文拉法辛", "他莫昔芬"],
    "CYP2C19": ["氯吡格雷"],
}
current_drug = patient.current_meds or ""

PHENO_COLOR = {
    "PM": "🔴",
    "IM": "🟠",
    "NM": "🟢",
    "UM": "🔴",
}

# 两列布局：CYP2D6 + CYP2C19
col_d, col_c = st.columns(2)

for col, gene in zip([col_d, col_c], ["CYP2D6", "CYP2C19"]):
    with col:
        phen = phenotypes.get(gene)
        if not phen:
            st.error(f"❌ {gene} 表型推断失败")
            continue

        # 若需要基因检测，用 ⚪ 替代表型本身的颜色（避免误导）
        icon_for_header = "⚪" if getattr(phen, "needs_detection", False) else PHENO_COLOR[phen.phenotype]
        with st.container():
            st.markdown(f"### {icon_for_header} {gene}")

            phen_label = {
                "PM": "Poor Metabolizer（弱代谢）",
                "IM": "Intermediate Metabolizer（中间代谢）",
                "NM": "Normal Metabolizer（正常代谢）",
                "UM": "Ultrarapid Metabolizer（超速代谢）",
            }.get(phen.phenotype, phen.phenotype)
            st.markdown(f"**表型（基因型推导）**: {phen_label}")

            # ── 两栏：基因型 vs 有效代谢（phenoconversion）──
            if phen.best_genotype:
                col_g, col_e = st.columns(2)
                with col_g:
                    st.markdown("**🧬 基因型层面**")
                    st.markdown(f"`{phen.best_genotype}`")
                    st.caption(f"activity score: {phen.activity_score}")
                    st.caption(f"→ {phen.phenotype}")
                with col_e:
                    st.markdown("**💊 有效代谢**（基因型+器官）")
                    eff_label = {
                        "PM": "🔴 PM（弱代谢）", "IM": "🟠 IM（中间）",
                        "NM": "🟢 NM（正常）", "UM": "🟣 UM（超速）",
                    }.get(phen.effective_phenotype, phen.effective_phenotype)
                    st.markdown(f"{eff_label}")
                    st.caption(f"organ factor: ×{phen.organ_factor}")
                    st.caption(f"effective score: {phen.activity_score * phen.organ_factor:.2f}")
                if phen.organ_reasons:
                    with st.expander("📋 器官功能修正依据", expanded=False):
                        for r in phen.organ_reasons:
                            st.caption(f"• {r}")
                if phen.needs_detection:
                    st.warning(
                        f"⚠️ **基因型表型 ({phen.phenotype}) 与有效代谢 ({phen.effective_phenotype}) 不一致**"
                        f" — 强烈建议基因检测以确定基线（区分基因型 vs 器官功能）"
                    )

                if phen.top_genotypes:
                    st.markdown("**🔬 其他可能基因型（Top 5）**")
                    st.caption("饼内仅显示百分比；基因型-表型-后验信息见右侧 legend。")
                    import matplotlib.pyplot as plt
                    from matplotlib.patches import Patch

                    # 按后验倒序，方便人眼对照
                    sorted_tops = sorted(phen.top_genotypes, key=lambda x: -x[2])
                    best_genotype = phen.best_genotype or ""

                    sizes = [post for _, _, post in sorted_tops]
                    pheno_color = {
                        "PM": "#e74c3c",   # 红
                        "IM": "#f39c12",   # 橙
                        "NM": "#27ae60",   # 绿
                        "UM": "#9b59b6",   # 紫
                    }
                    colors = [
                        pheno_color.get(ph, "#95a5a6")
                        for _, ph, _ in sorted_tops
                    ]

                    # 最佳基因型稍微外推
                    explode = [
                        0.05 if g == best_genotype else 0.0
                        for g, _, _ in sorted_tops
                    ]

                    # 关键：<5% 的切片不画百分比（避免小切片文字互相挤）
                    def _autopct(pct: float) -> str:
                        return f"{pct:.1f}%" if pct >= 5.0 else ""

                    fig, ax = plt.subplots(figsize=(7.5, 5.0))
                    fig.subplots_adjust(left=0.02, right=0.55, top=0.92, bottom=0.06)

                    wedges, _texts, autotexts = ax.pie(
                        sizes,
                        labels=None,
                        colors=colors,
                        explode=explode,
                        autopct=_autopct,
                        startangle=90,
                        pctdistance=0.72,
                        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                        textprops={"fontsize": 11},
                    )
                    plt.setp(autotexts, size=11, weight="bold", color="white")

                    # 右侧 legend：完整呈现 基因型-表型-后验，★ 标记最佳
                    legend_labels = [
                        f"{'★ ' if g == best_genotype else '   '}"
                        f"{g}  ({ph})  —  {s * 100:.1f}%"
                        for (g, ph, _), s in zip(sorted_tops, sizes)
                    ]
                    ax.legend(
                        wedges,
                        legend_labels,
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        fontsize=10.5,
                        frameon=False,
                        title="基因型 (表型) — 后验",
                        title_fontsize=10.5,
                    )
                    ax.set_title(
                        f"其他可能基因型 Top {len(sorted_tops)}"
                        + (f"  ·  最佳 `{best_genotype}`" if best_genotype else ""),
                        fontsize=12,
                        pad=14,
                        loc="center",
                    )
                    ax.axis("equal")
                    st.pyplot(fig)
                    plt.close(fig)

                # ── 分层检测推荐（4 档判断 + 三方案对比）──
                try:
                    from src.detection_recommender import recommend_detection
                    plan = recommend_detection(
                        gene, phen.top_genotypes, phen.best_genotype, phen.phenotype,
                        population=phen.population or "东亚",
                    )

                    # ── 顶部：4 档判断 badge（关键修正）──
                    amb = plan.ambiguity_score
                    rec_level = plan.recommendation_level

                    # 4 档等级标识
                    LEVEL_BADGE = {
                        "CLEAR": ("🟢", "清晰"),
                        "AMBIGUOUS": ("🟡", "模糊"),
                        "EXTREMELY_AMBIGUOUS": ("🔴", "极度模糊"),
                        "PHENOTYPE_DIVERGENT": ("🔴", "表型分歧"),
                        "EXTREME_AMBIGUOUS": ("🔴", "极端"),
                        "NO_DATA": ("⚪", "无数据"),
                    }
                    badge_icon, badge_text = LEVEL_BADGE.get(rec_level, ("⚪", rec_level))

                    # 计算过程摘要（即使折叠也显示关键数字）
                    if plan.top2_genotype:
                        amb_caption = (
                            f"top1 `{plan.best_genotype}` ({plan.best_phenotype}, {plan.top_genotypes[0][2]:.1%}) "
                            f"vs top2 `{plan.top2_genotype}` ({plan.top2_phenotype}, {plan.top2_prob:.1%}) "
                            f"· 歧义度 {amb:.1%}"
                        )
                        if plan.phenotype_divergent:
                            amb_caption += " · ⚠️ 表型不同"
                    else:
                        amb_caption = (
                            f"top1 `{plan.best_genotype}` ({plan.best_phenotype}, {plan.top_genotypes[0][2]:.1%}) "
                            f"· 无 top2 候选"
                        )

                    # 根据 4 档等级显示不同颜色 badge
                    if plan.test_advice_kind == "success":
                        st.success(f"{badge_icon} **判断等级：{badge_text}** · {plan.test_advice}")
                    elif plan.test_advice_kind == "warning":
                        st.warning(f"{badge_icon} **判断等级：{badge_text}** · {plan.test_advice}")
                    elif plan.test_advice_kind == "error":
                        st.error(f"{badge_icon} **判断等级：{badge_text}** · {plan.test_advice}")
                    else:
                        st.info(f"{badge_icon} **判断等级：{badge_text}** · {plan.test_advice}")

                    st.caption(amb_caption)

                    # ── 详细方案（默认展开）──
                    with st.expander("🧪 详细检测方案对比", expanded=True):
                        # 方案 1：全测序
                        full = plan.plan_full
                        st.markdown("**方案 1 · 全基因测序**")
                        st.caption(f"覆盖: {full['covers']} | {full['resolves_uncertainty']}")
                        st.markdown("---")
                        # 方案 2：经济分层检测（顺序工作流）
                        eco = plan.plan_economical
                        st.markdown("**方案 2 · 经济分层检测**（推荐 ⭐）")
                        st.markdown(f"- {eco.get('name', '')}")
                        st.markdown("**检测顺序：**")

                        # ── 收集所有步骤 + 判断末步（用于"可确定表型"标注）──
                        import re

                        steps: list[dict] = []  # [{"label": "第 1 步", "test": "测 XXX"}, ...]

                        if "step1" in eco:
                            steps.append({
                                "label": "第 1 步",
                                "test": eco["step1"],
                                "is_final": False,  # 占位，稍后判断
                            })
                        if "step2" in eco:
                            steps.append({
                                "label": "（若为阴）第 2 步",
                                "test": eco["step2"],
                                "is_final": False,
                            })

                        # 从 step1_outcome_if_negative 解析 next_snp_hint
                        has_full_seq = False
                        if "step1_outcome_if_negative" in eco:
                            neg_text = eco["step1_outcome_if_negative"]
                            more_snp = re.search(
                                r"阴性后\*\*还需测\*\*\s*([^\s]+(?:[^\n]*?[^\s])?)\s",
                                neg_text,
                            )
                            if more_snp:
                                steps.append({
                                    "label": f"（若为阴）第 {len(steps)+1} 步",
                                    "test": f"测 {more_snp.group(1).strip()}",
                                    "is_final": False,
                                })
                            if "建议全测序" in neg_text:
                                steps.append({
                                    "label": f"（若为阴）第 {len(steps)+1} 步",
                                    "test": "建议**全基因测序**",
                                    "is_final": True,  # 全测序一定确定
                                })
                                has_full_seq = True

                        # 标记最后一步为 "is_final"
                        if steps:
                            for s in steps[:-1]:
                                s["is_final"] = False
                            # 如果没有全测序兜底，最后一步视为"可确定"
                            if not has_full_seq and steps:
                                steps[-1]["is_final"] = True

                        # ── 渲染步骤 ──
                        # 提取 step1 阳性的表型（用于第 1 步）
                        pos_pheno_str = " → 进一步缩小候选"
                        if "step1_outcome_if_positive" in eco:
                            ph_match = re.search(
                                r"建议按最可能 `([^`]+)`（([^）]+)）",
                                eco["step1_outcome_if_positive"],
                            )
                            if ph_match:
                                pos_pheno_str = (
                                    f" → 确定为 {ph_match.group(1)} ({ph_match.group(2).strip()})"
                                )

                        for i, s in enumerate(steps):
                            test_clean = _strip_marker_parens(s["test"])
                            if i == 0:
                                # 第 1 步：若阳 → 确定为具体表型
                                note = pos_pheno_str
                            elif s["is_final"] and has_full_seq and "全基因测序" in s["test"]:
                                # 全测序末步
                                note = " → **一定可确定**表型"
                            elif s["is_final"]:
                                # 链自然结束的末步
                                note = " → 可确定表型"
                            else:
                                # 中间步骤
                                note = " → 缩小候选"

                            st.markdown(
                                f"&nbsp;&nbsp;**{s['label']}**："
                                f"{test_clean}"
                                f"（若阳{note}）"
                            )
                        st.markdown("---")
                        # 方案 3：免测（当 must_test=True 时变红警告）
                        mini = plan.plan_minimal
                        st.markdown("**方案 3 · 不测，直接按 top1 用药**")
                        if plan.must_test:
                            st.error(
                                f"⛔ **不建议采用** · {plan.test_advice}\n\n"
                                f"原因：歧义度高或表型分歧，按 top1 用药风险过大。"
                                f"实际基因型若为 top2（`{plan.top2_genotype}`），"
                                f"可能用药不当。"
                            )
                        else:
                            st.caption(f"风险: {mini.get('risk', '')}")
                            st.caption(f"建议: {mini.get('recommended_when', '')}")
                except Exception as e:
                    st.caption(f"（检测推荐暂不可用: {e}）")

            # ── best_genotype_confidence 提示（与 4 档判断建议互补）──
            _bg_conf = getattr(phen, "best_genotype_confidence", 0.0) or 0.0
            if phen.best_genotype and _bg_conf < 0.60:
                st.warning(
                    f"⚠️ **最可能基因型 `{phen.best_genotype}` 后验仅 {_bg_conf:.1%}（< 60%）** "
                    f"—— 推断不确定性强，**强烈建议基因检测以确定基线**。",
                    icon="🧬",
                )
            elif phen.best_genotype and _bg_conf < 0.80:
                st.info(
                    f"ℹ️ **最可能基因型 `{phen.best_genotype}` 后验 {_bg_conf:.1%}（60-80%）** "
                    f"—— 中等置信度，可考虑基因检测复核。",
                    icon="🧬",
                )
            elif phen.best_genotype:
                st.success(
                    f"✅ **最可能基因型 `{phen.best_genotype}` 后验 {_bg_conf:.1%}（≥ 80%）** "
                    f"—— 高置信度，无需立即检测。",
                    icon="🧬",
                )

            has_crit = any(d in current_drug for d in CRITICAL_DRUGS.get(gene, []))
            # 判断证据来源
            has_clin = any(not s.signal_name.startswith("[CCA") for s in phen.matched_signals)
            has_cca = any(s.signal_name.startswith("[CCA") for s in phen.matched_signals)
            if has_clin and has_cca:
                evidence_source = "mixed"
            elif has_clin:
                evidence_source = "clinical"
            elif has_cca:
                evidence_source = "blood"
            else:
                evidence_source = "none"

            st.divider()


# ══════════════════════════════════════════════════════════════════
# Agent 推理区块（含 3 种回答类型 + 用药工具）
# ══════════════════════════════════════════════════════════════════
render_agent_section(report)

st.divider()
st.page_link("app.py", label="← 返回首页", use_container_width=False)