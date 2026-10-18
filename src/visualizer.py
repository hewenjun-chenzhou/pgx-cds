"""
visualizer.py — Step 6
Plotly 交互可视化生成器

生成：
1. 表型推断雷达图（CYP2D6 vs CYP2C19 置信度）
2. 高优先级药物推荐卡片
3. DDI 风险水平柱状图
4. 人群表型频率堆叠柱状图
5. 药物-基因相互作用矩阵热力图
6. 完整 HTML 报告页
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional

from .phenotype_inferrer import get_phenotype_description

from .report_generator import CDSReport, report_to_dict
from .config import GRADE_COLORS, WARNING_COLORS, SIGNAL_STRENGTH_COLORS
from .population_analyzer import get_population_analysis


# ── 颜色常量 ───────────────────────────────────────────────────
PGX_COLORS = {
    "CYP2D6": "#4FC3F7",
    "CYP2C19": "#81C784",
    "PM": "#FF4B4B",
    "IM": "#FFA500",
    "NM": "#4CAF50",
    "UM": "#9C27B0",
    "高": "#FF4B4B",
    "中": "#FFA500",
    "低": "#28A745",
}

# ── 药物类别颜色 ───────────────────────────────────────────────
DRUG_CATEGORY_COLORS = {
    "心血管": "#E3F2FD",
    "精神": "#F3E5F5",
    "肿瘤": "#FFEBEE",
    "镇痛": "#FFF3E0",
    "抗菌": "#E8F5E9",
    "抗病毒": "#E0F7FA",
    "其他": "#F5F5F5",
}
DRUG_CATEGORY_LABEL_COLORS = {
    "心血管": "#1565C0",
    "精神": "#6A1B9A",
    "肿瘤": "#C62828",
    "镇痛": "#E65100",
    "抗菌": "#2E7D32",
    "抗病毒": "#00838F",
    "其他": "#616161",
}


# ── 临床分析文本生成 ─────────────────────────────────────────────

def _build_clinical_analysis(rec, gene: str, population: str) -> str:
    """
    为单个药物推荐生成临床分析文字。
    格式：【基因】药物 · 等级：X（强推荐/推荐/...）
    依据：患者为X代谢者（人群频率%），[具体数据]，建议...
    """
    grade = rec.recommendation_grade
    grade_map = {
        "A": "强推荐",
        "B": "推荐",
        "C": "弱推荐",
        "D": "不推荐",
    }
    grade_text = grade_map.get(grade, f"等级{grade}")

    phenotype = rec.phenotype
    phenotype_name = {
        "PM": "慢代谢者（PM）",
        "IM": "中间代谢者（IM）",
        "NM": "正常代谢者（NM）",
        "UM": "超快代谢者（UM）",
    }.get(phenotype, phenotype)

    # 人群频率
    pop_data = get_population_analysis(gene, population)
    pop_freq_text = ""
    if pop_data:
        target_freq = {
            "PM": pop_data.pm_freq,
            "IM": pop_data.im_freq,
            "NM": pop_data.nm_freq,
            "UM": pop_data.um_freq,
        }.get(phenotype, None)
        if target_freq is not None and target_freq > 0:
            pop_freq_text = f"该人群{phenotype}频率约 {target_freq:.1%}"

    # 证据强度
    evidence_map = {
        "Strong": "强",
        "Moderate": "中等",
        "Weak": "弱",
        "Optional": "可选",
    }
    evidence_text = evidence_map.get(rec.evidence.strip(), rec.evidence.strip()) if rec.evidence.strip() else "证据有限"

    # 核心建议
    action = rec.current_action.strip() if rec.current_action.strip() else "维持常规剂量"

    # 组合依据
    lines = [f"【{gene}】{rec.drug_name} · 等级：{grade}（{grade_text}）"]
    lines.append(f"依据：患者为{phenotype_name}，")
    if pop_freq_text:
        lines.append(f"{pop_freq_text}，")
    if rec.evidence.strip():
        lines.append(f"证据强度：{evidence_text}，")
    lines.append(f"建议：{action}。")
    if rec.alternatives.strip():
        lines.append(f"备选药物：{rec.alternatives.strip()}。")

    return "".join(lines)


def _build_population_analysis_text(gene: str, phenotype: str, population: str, report: "CDSReport") -> str:
    """
    生成人群差异分析文字（创新点说明）。
    基于患者基因型表型、当前用药、人群频率，给出具体临床指导。
    核心创新：人群精准匹配的贝叶斯先验——直接决定后验概率的基准。
    """
    from .data_loader import get_cache
    pop_data = get_population_analysis(gene, population)
    if pop_data is None:
        return ""

    cache = get_cache()
    # 参考人群对比（固定用欧洲人群做对比基准）
    ref_pop = "欧洲" if population != "欧洲" else "非洲裔"
    ref_data = get_population_analysis(gene, ref_pop)

    phenotype_name = {
        "PM": "慢代谢者（PM）",
        "IM": "中间代谢者（IM）",
        "NM": "正常代谢者（NM）",
        "UM": "超快代谢者（UM）",
    }.get(phenotype, phenotype)

    # 当前人群频率
    freq_map = {
        "PM": pop_data.pm_freq,
        "IM": pop_data.im_freq,
        "NM": pop_data.nm_freq,
        "UM": pop_data.um_freq,
    }
    cur_freq = freq_map.get(phenotype, 0)

    lines = [f"【{gene} · {population}人群精准分析】"]

    # ── 1. 人群频率差异（核心创新点说明）─────────────────────────
    if ref_data:
        ref_freq = {
            "PM": ref_data.pm_freq,
            "IM": ref_data.im_freq,
            "NM": ref_data.nm_freq,
            "UM": ref_data.um_freq,
        }.get(phenotype, 0)
        diff = cur_freq - ref_freq
        if abs(diff) > 0.005:
            direction = "高于" if diff > 0 else "低于"
            lines.append(
                f"患者为{phenotype_name}，该表型在{population}人群中出现频率为 {cur_freq:.1%}，"
                f"较{ref_pop}人群（{ref_freq:.1%}）{direction}约 {abs(diff):.1%}。"
                f"这一差异直接影响贝叶斯推断的先验概率——"
                f"若错误使用{ref_pop}人群数据，该患者表型的后验概率基准偏差将达 {abs(diff):.1%}。"
            )
    else:
        lines.append(f"{population}人群{phenotype_name}频率约 {cur_freq:.1%}，")
        lines.append(f"数据来源：{pop_data.source}。")

    # ── 2. 人群特异性临床意义 ────────────────────────────────────
    lines.append(_build_population_clinical_guidance(gene, phenotype, population, ref_pop, ref_data, report))

    return "".join(lines)


def _build_population_clinical_guidance(
    gene: str,
    phenotype: str,
    population: str,
    ref_pop: str,
    ref_data,
    report: "CDSReport",
) -> str:
    """
    基于人群表型分布特征，给出具体临床指导。
    不同人群同一表型对相同药物的剂量建议可能不同。
    """
    from .data_loader import get_cache
    cache = get_cache()

    # 选择对应基因的药物表
    drug_df = cache.drugs if gene == "CYP2D6" else cache.drugs_cyp2c19

    # 筛选出当前用药且该基因有重要意义的药物
    current_meds = [m.strip() for m in report.patient.current_meds if m.strip()]
    relevant_drugs = drug_df[drug_df["drug_name"].isin(current_meds)]

    # 按表型取对应的处理建议
    action_col = "pm_action" if phenotype == "PM" else (
        "um_action" if phenotype == "UM" else None
    )

    lines = []
    drug_specifics = []

    for _, row in relevant_drugs.iterrows():
        effect = str(row.get("cyp2d6_effect" if gene == "CYP2D6" else "cyp2c19_effect", "")).strip()
        if effect in ("non_substrate", "") or not effect:
            continue

        action = None
        if phenotype == "PM":
            action = str(row.get("pm_action", "")).strip()
        elif phenotype == "UM":
            action = str(row.get("um_action", "")).strip()
        elif phenotype == "IM":
            # IM 通常参考 PM 但更保守
            pm_a = str(row.get("pm_action", "")).strip()
            if pm_a and pm_a not in ("不适用", "无需调整", "影响小"):
                action = f"参考PM方案（{pm_a}），可适度调整"
        # NM 通常无需特别说明

        if action and action not in ("不适用", "无需调整", "影响小", ""):
            drug_specifics.append(f"**{row['drug_name']}**：{action}")

    if drug_specifics:
        lines.append(f"**🔴 该人群中需重点关注的药物（{phenotype}）**：")
        for d in drug_specifics:
            lines.append(f"  · {d}")

    # ── 人群表型分布特征带来的特殊风险 ──────────────────────────
    pop_data = get_population_analysis(gene, population)
    if pop_data is None:
        return "".join(lines)

    # 识别该人群的表型优势/劣势
    dominant_phenotype = max(
        ["PM", "IM", "NM", "UM"],
        key=lambda p: [pop_data.pm_freq, pop_data.im_freq, pop_data.nm_freq, pop_data.um_freq][
            ["PM", "IM", "NM", "UM"].index(p)]
    )

    # 高风险表型频率与参考人群对比
    high_risk_phenotypes = {"PM", "UM"}
    risk_notes = []

    for rp in ["PM", "UM"]:
        freq = {"PM": pop_data.pm_freq, "UM": pop_data.um_freq}[rp]
        ref_freq_map = {
            "PM": (ref_data.pm_freq if ref_data else None),
            "UM": (ref_data.um_freq if ref_data else None),
        }
        ref_f = ref_freq_map[rp]

        if ref_f is not None and freq > ref_f + 0.03:
            pop_label = {
                "PM": "慢代谢者",
                "UM": "超快代谢者",
            }[rp]
            diff_pct = freq - ref_f
            risk_notes.append(
                f"⚠️ {population}{pop_label}频率（{freq:.1%}）显著高于{ref_pop}（{ref_f:.1%}），"
                f"相差 {diff_pct:.1%}。若错误使用{ref_pop}人群数据做先验，"
                f"该表型的贝叶斯后验概率将系统性低估 {diff_pct:.1%}，可能导致剂量调整建议被忽略。"
            )
        elif ref_f is not None and freq < ref_f - 0.03:
            pop_label = {
                "PM": "慢代谢者",
                "UM": "超快代谢者",
            }[rp]
            diff_pct = ref_f - freq
            risk_notes.append(
                f"ℹ️ {population}{pop_label}频率（{freq:.1%}）低于{ref_pop}（{ref_f:.1%}），"
                f"相差 {diff_pct:.1%}。该人群使用{ref_pop}数据时后验概率将系统性高估，"
                f"可能导致不必要的剂量干预。"
            )

    if risk_notes:
        lines.extend(risk_notes)

    if not lines:
        return f"该人群{gene}表型分布与参考人群差异无显著临床意义。"
    return "".join(lines)


def _phenotype_radar(report: CDSReport) -> go.Figure:
    """雷达图：两个基因的表型置信度"""
    genes = ["CYP2D6", "CYP2C19"]
    phenotypes = ["PM", "IM", "NM", "UM"]
    phenotype_values = {p: [0.0] * len(genes) for p in phenotypes}

    for i, gene in enumerate(genes):
        pheno = report.phenotypes[gene].phenotype
        conf = report.phenotypes[gene].confidence
        for p in phenotypes:
            phenotype_values[p][i] = 1.0 if p == pheno else 0.0

    fig = go.Figure()
    for p in phenotypes:
        fig.add_trace(go.Scatterpolar(
            r=phenotype_values[p] + [phenotype_values[p][0]],
            theta=genes + [genes[0]],
            mode="lines+markers+text",
            name=p,
            line=dict(color=PGX_COLORS[p], width=2.5),
            marker=dict(size=8),
            text=[p if v > 0 else "" for v in phenotype_values[p]],
            textposition="top center",
        ))

    # 标注入推断表型文字（含星型等位基因）
    for i, gene in enumerate(genes):
        pheno = report.phenotypes[gene].phenotype
        conf = report.phenotypes[gene].confidence
        genotype = report.phenotypes[gene].genotype
        geno_text = f" ({genotype})" if genotype else ""
        # 置信度等级标签
        if conf >= 0.80:
            conf_label = "强置信"
        elif conf >= 0.60:
            conf_label = "中置信"
        elif conf >= 0.40:
            conf_label = "偏低"
        else:
            conf_label = "低置信"
        fig.add_annotation(
            x=0.5 if i == 0 else 0.5,
            y=0.85 - i * 0.1,
            xref="paper", yref="paper",
            text=f"{gene}{geno_text}: <b>{pheno}</b> (后验概率 {conf:.0%} · {conf_label})",
            showarrow=False,
            font=dict(size=12, color=PGX_COLORS[gene]),
        )

    fig.update_layout(
        title=dict(text="表型推断雷达图", font=dict(size=16)),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.2])),
        showlegend=True,
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1),
    )
    return fig


def _ddi_bar_chart(report: CDSReport) -> go.Figure:
    """DDI 风险水平柱状图"""
    levels = ["高", "中", "低"]
    counts = [
        len(report.ddi_report.high_risk),
        len(report.ddi_report.medium_risk),
        len(report.ddi_report.low_risk),
    ]
    colors = [PGX_COLORS[l] for l in levels]

    fig = go.Figure(go.Bar(
        x=levels,
        y=counts,
        marker_color=colors,
        text=counts,
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="药物相互作用（DDI）风险分布", font=dict(size=14)),
        xaxis_title="风险级别",
        yaxis_title="数量",
        height=300,
        showlegend=False,
    )
    return fig


def _population_stacked_bar(report: CDSReport, gene: str) -> go.Figure:
    """人群表型频率堆叠柱状图"""
    pop_data = report.population_data.get(gene)
    if pop_data is None:
        return go.Figure()

    phenotypes = ["PM", "IM", "NM", "UM"]
    freqs = {
        "PM": pop_data.pm_freq,
        "IM": pop_data.im_freq,
        "NM": pop_data.nm_freq,
        "UM": pop_data.um_freq,
    }

    fig = go.Figure()
    bottom = 0.0
    for p in phenotypes:
        fig.add_trace(go.Bar(
            x=[pop_data.population],
            y=[freqs[p]],
            name=p,
            marker_color=PGX_COLORS[p],
            offset=0,
            text=f"{freqs[p]:.1%}",
            textposition="inside",
        ))
        bottom += freqs[p]

    fig.update_layout(
        title=dict(text=f"{gene} 人群表型频率（来源: {pop_data.source[:30]}...）",
                   font=dict(size=13)),
        barmode="stack",
        xaxis_title="人群",
        yaxis_title="频率",
        yaxis_tickformat=".0%",
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    return fig


def _population_comparison_chart(gene: str, patient_population: str) -> go.Figure:
    """
    跨人群表型频率对比图（创新点图表）。
    突出患者所属人群与其他人群的差异。
    """
    from .population_analyzer import get_population_comparison

    all_pops = get_population_comparison(gene)
    if not all_pops:
        return go.Figure()

    phenotypes = ["PM", "IM", "NM", "UM"]
    pop_labels = list(all_pops.keys())

    # 每种表型一条grouped bar
    fig = go.Figure()
    for p in phenotypes:
        freqs = [all_pops[pop].__dict__[f"{p.lower()}_freq"] for pop in pop_labels]
        fig.add_trace(go.Bar(
            x=pop_labels,
            y=freqs,
            name=p,
            marker_color=PGX_COLORS[p],
            text=[f"{f:.1%}" for f in freqs],
            textposition="outside",
        ))

    # 高亮患者人群
    fig.update_layout(
        title=dict(
            text=f"🌏 {gene} 人群表型频率对比 | 患者人群: <b>{patient_population}</b>",
            font=dict(size=14),
        ),
        barmode="group",
        xaxis_title="人群",
        yaxis_title="频率",
        yaxis_tickformat=".0%",
        height=340,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        # 用形状高亮患者人群的柱子（通过添加注释）
        annotations=[
            dict(
                x=pop_labels[i] if pop_labels[i] == patient_population else "",
                y=1.05,
                xref="x", yref="paper",
                text="← 患者人群",
                showarrow=False,
                font=dict(size=11, color="#1a237e"),
            )
            for i in range(len(pop_labels))
            if pop_labels[i] == patient_population
        ],
    )
    return fig


def _drug_recommendation_table(report: CDSReport) -> go.Figure:
    """高优先级药物推荐表格"""
    urgent = report.urgent_interventions
    if not urgent:
        # 无紧急干预时显示提示
        fig = go.Figure()
        fig.add_annotation(
            text="✓ 暂无需要紧急干预的药物",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#28A745"),
        )
        fig.update_layout(height=200, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    drugs = [r.drug_name for r in urgent]
    genes = [r.gene for r in urgent]
    actions = [r.current_action for r in urgent]
    grades = [r.recommendation_grade for r in urgent]
    colors = [GRADE_COLORS.get(g, "#888888") for g in grades]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>基因</b>", "<b>药物</b>", "<b>推荐等级</b>", "<b>建议</b>"],
            fill_color="#37474F",
            font=dict(color="white", size=12),
            align="left",
            height=28,
        ),
        cells=dict(
            values=[genes, drugs, grades, actions],
            fill_color=[["#FFF3E0"] * len(drugs)],
            font=dict(size=11),
            align="left",
            height=26,
        ),
    )])
    fig.update_layout(
        title=dict(text="⚠ 高优先级药物调整建议", font=dict(size=14, color="#FF4B4B")),
        height=max(60 + len(drugs) * 30, 200),
        margin=dict(l=5, r=5, t=40, b=5),
    )
    return fig


def _ddi_detail_table(report: CDSReport) -> go.Figure:
    """DDI 详情表"""
    interactions = report.ddi_report.interactions
    if not interactions:
        fig = go.Figure()
        fig.add_annotation(
            text="✓ 未检测到药物相互作用",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="#28A745"),
        )
        fig.update_layout(height=200, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    drugs_a = [d.drug_a for d in interactions]
    drugs_b = [d.drug_b for d in interactions]
    levels = [d.warning_level for d in interactions]
    consequences = [d.clinical_consequence for d in interactions]
    level_colors = [WARNING_COLORS.get(l, "#888888") for l in levels]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>药物A</b>", "<b>药物B</b>", "<b>风险</b>", "<b>临床后果</b>"],
            fill_color="#37474F",
            font=dict(color="white", size=11),
            align="left",
            height=26,
        ),
        cells=dict(
            values=[drugs_a, drugs_b, levels, consequences],
            fill_color=[level_colors],
            font=dict(size=10),
            align="left",
            height=24,
        ),
    )])
    fig.update_layout(
        title=dict(text=f"💊 药物相互作用（{len(interactions)} 项）", font=dict(size=13)),
        height=max(60 + len(interactions) * 28, 200),
        margin=dict(l=5, r=5, t=40, b=5),
    )
    return fig


def generate_report_html(report: CDSReport, output_path: Optional[str] = None) -> str:
    """
    生成完整的单文件 HTML 报告。

    Parameters
    ----------
    report : CDSReport
    output_path : 若指定，则同时写入文件

    Returns
    -------
    HTML 字符串
    """
    fig_radar = _phenotype_radar(report)
    fig_ddi_bar = _ddi_bar_chart(report)
    fig_ddi_table = _ddi_detail_table(report)
    fig_rec_table = _drug_recommendation_table(report)
    fig_pop_cyp2d6 = _population_stacked_bar(report, "CYP2D6")
    fig_pop_cyp2c19 = _population_stacked_bar(report, "CYP2C19")
    fig_pop_cmp_cyp2d6 = _population_comparison_chart("CYP2D6", report.patient.population)
    fig_pop_cmp_cyp2c19 = _population_comparison_chart("CYP2C19", report.patient.population)

    # 拼成完整 HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='zh-CN'>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<title>PGx CDS 报告</title>",
        "<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>",
        "<style>",
        "body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 0; background: #f5f7fa; }",
        ".container { max-width: 1100px; margin: 0 auto; padding: 20px; }",
        ".header { background: linear-gradient(135deg, #1a237e 0%, #283593 100%); "
            "color: white; padding: 28px 32px; border-radius: 12px; margin-bottom: 24px; }",
        ".header h1 { margin: 0 0 8px 0; font-size: 22px; }",
        ".header p { margin: 0; opacity: 0.85; font-size: 13px; }",
        ".patient-card { background: white; border-radius: 10px; padding: 20px 24px; "
            "margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }",
        ".patient-card h2 { margin: 0 0 14px 0; font-size: 15px; color: #1a237e; border-bottom: 2px solid #c5cae9; "
            "padding-bottom: 8px; }",
        ".patient-meta { display: flex; gap: 24px; flex-wrap: wrap; }",
        ".meta-item { font-size: 13px; }",
        ".meta-item .label { color: #757575; }",
        ".meta-item .value { font-weight: 600; color: #212121; }",
        ".grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }",
        ".grid-1 { margin-bottom: 20px; }",
        ".card { background: white; border-radius: 10px; padding: 20px; "
            "box-shadow: 0 2px 8px rgba(0,0,0,0.08); }",
        ".card h3 { margin: 0 0 14px 0; font-size: 14px; color: #1a237e; }",
        ".alert-box { background: #FFEBEE; border-left: 4px solid #FF4B4B; "
            "border-radius: 4px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; }",
        ".ok-box { background: #E8F5E9; border-left: 4px solid #28A745; "
            "border-radius: 4px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; }",
        ".footer { text-align: center; color: #9e9e9e; font-size: 11px; padding: 20px 0; }",
        ".plotly-graph-div { margin-bottom: 8px; }",
        ".section-title { font-size: 15px; font-weight: 600; color: #1a237e; "
            "margin: 0 0 12px 0; border-left: 4px solid #3949ab; padding-left: 10px; }",
        ".exec-summary { background: white; border-radius: 10px; padding: 20px 24px; "
            "margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }",
        ".exec-summary h2 { margin: 0 0 14px 0; font-size: 15px; color: #1a237e; "
            "border-bottom: 2px solid #c5cae9; padding-bottom: 8px; }",
        ".exec-item { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; "
            "font-size: 13px; line-height: 1.5; }",
        ".exec-num { background: #3949ab; color: white; border-radius: 50%; "
            "width: 20px; height: 20px; display: inline-flex; align-items: center; "
            "justify-content: center; font-size: 11px; flex-shrink: 0; margin-top: 2px; }",
        ".legend-box { background: white; border-radius: 10px; padding: 16px 20px; "
            "margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }",
        ".legend-title { font-size: 13px; font-weight: 600; color: #1a237e; margin-bottom: 10px; }",
        ".legend-items { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; }",
        ".legend-item { display: flex; align-items: center; gap: 5px; }",
        ".legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }",
        ".category-group { margin-bottom: 16px; }",
        ".category-header { display: flex; align-items: center; gap: 8px; "
            "margin-bottom: 8px; font-size: 13px; font-weight: 600; }",
        ".category-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; "
            "font-size: 11px; font-weight: 600; }",
        ".clinical-analysis { background: #f3f4f6; border-radius: 6px; padding: 10px 14px; "
            "margin-bottom: 10px; font-size: 13px; }",
        ".source-link { color: #1565C0; text-decoration: underline; cursor: pointer; }",
        ".source-link:hover { color: #0D47A1; }",
        "@media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }",
        "</style>",
        "</head>",
        "<body>",
        "<div class='container'>",
    ]

    # ── 报告头部 ───────────────────────────────────────────────
    p = report.patient
    html_parts.extend([
        "<div class='header'>",
        f"<h1>🧬 药物基因组学（PGx）临床决策支持报告</h1>",
        f"<p>病例ID: {p.case_id} &nbsp;|&nbsp; 生成时间: {report.generated_at}</p>",
        "</div>",
    ])

    # ── 患者信息卡片 ───────────────────────────────────────────
    # 构建表型+基因型展示文字
    phen_cyp2d6 = report.phenotypes.get("CYP2D6")
    phen_cyp2c19 = report.phenotypes.get("CYP2C19")
    pheno_d_text = f"{phen_cyp2d6.phenotype}" if phen_cyp2d6 else "NM"
    pheno_d_geno = f"（{phen_cyp2d6.genotype}）" if phen_cyp2d6 and phen_cyp2d6.genotype else ""
    pheno_c_text = f"{phen_cyp2c19.phenotype}" if phen_cyp2c19 else "NM"
    pheno_c_geno = f"（{phen_cyp2c19.genotype}）" if phen_cyp2c19 and phen_cyp2c19.genotype else ""

    html_parts.extend([
        "<div class='patient-card'>",
        "<h2>📋 患者信息</h2>",
        "<div class='patient-meta'>",
        f"<div class='meta-item'><span class='label'>诊断:</span> <span class='value'>{p.diagnosis}</span></div>",
        f"<div class='meta-item'><span class='label'>人群:</span> <span class='value'>{p.population}</span></div>",
        f"<div class='meta-item'><span class='label'>当前用药:</span> <span class='value'>{', '.join(p.current_meds) if p.current_meds else '无'}</span></div>",
        f"<div class='meta-item'><span class='label'>CYP2D6:</span> <span class='value'>{pheno_d_text}{pheno_d_geno}</span></div>",
        f"<div class='meta-item'><span class='label'>CYP2C19:</span> <span class='value'>{pheno_c_text}{pheno_c_geno}</span></div>",
        "</div>",
        "</div>",
    ])

    # ── 执行摘要（Executive Summary）───────────────────────────
    # 4条核心结论
    summary_lines = []
    # 结论1：表型+功能描述
    desc_d = get_phenotype_description(pheno_d_text, "CYP2D6") if phen_cyp2d6 else ""
    desc_c = get_phenotype_description(pheno_c_text, "CYP2C19") if phen_cyp2c19 else ""
    def _conf_label(c):
        if c >= 0.80: return "强置信"
        elif c >= 0.60: return "中置信"
        elif c >= 0.40: return "偏低"
        return "低置信"

    summary_lines.append(
        f"患者为 <b>CYP2D6 {pheno_d_text}{pheno_d_geno}</b> / "
        f"<b>CYP2C19 {pheno_c_text}{pheno_c_geno}</b> 代谢者，"
        f"后验概率分别为 {phen_cyp2d6.confidence:.0%}（{_conf_label(phen_cyp2d6.confidence)}）"
        f" / {phen_cyp2c19.confidence:.0%}（{_conf_label(phen_cyp2c19.confidence)}）。"
        f"（{desc_d}；{desc_c}）"
    )
    # 结论2：DDI
    if report.high_risk_ddi_count > 0:
        summary_lines.append(
            f"⚠️ 检测到 <b>{report.high_risk_ddi_count} 项高风险药物相互作用</b>，"
            "可能需要调整用药方案。"
        )
    else:
        summary_lines.append("✓ 未检测到高风险药物相互作用。")
    # 结论3：紧急干预
    urgent = report.urgent_interventions
    if urgent:
        drug_names = "、".join([f"<b>{r.drug_name}</b>（{r.gene}）" for r in urgent[:3]])
        more = f"等{len(urgent)}种" if len(urgent) > 3 else ""
        summary_lines.append(
            f"🚨 <b>{len(urgent)} 项药物</b>（{drug_names}{more}）需要调整，"
            "建议优先处理。"
        )
    else:
        summary_lines.append("✓ 暂无需要紧急干预的药物。")

    exec_html = "".join(
        f"<div class='exec-item'><span class='exec-num'>{i+1}</span>"
        f"<span>{line}</span></div>"
        for i, line in enumerate(summary_lines)
    )
    html_parts.extend([
        "<div class='exec-summary'>",
        "<h2>📌 执行摘要</h2>",
        exec_html,
        "</div>",
    ])

    # ── 颜色图例 ───────────────────────────────────────────────
    legend_items = [
        ("🟢 无需调整", "#28A745"),
        ("🟡 注意/监测", "#FFA500"),
        ("🟠 考虑替代", "#FFD700"),
        ("🔴 需干预", "#FF4B4B"),
    ]
    legend_html = "".join(
        f"<div class='legend-item'><span class='legend-dot' style='background:{c}'></span>{lbl.split(' ', 1)[1]}</div>"
        for lbl, c in legend_items
    )
    html_parts.extend([
        "<div class='legend-box'>",
        "<div class='legend-title'>等级颜色说明</div>",
        f"<div class='legend-items'>{legend_html}</div>",
        "</div>",
    ])

    # ── 警示框 ─────────────────────────────────────────────────
    if report.high_risk_ddi_count > 0 or report.urgent_interventions:
        html_parts.append(
            f"<div class='alert-box'>⚠ 检测到 {report.high_risk_ddi_count} 项高风险药物相互作用，"
            f"另有 {len(report.urgent_interventions)} 项药物需要调整，请优先处理。</div>"
        )
    else:
        html_parts.append(
            "<div class='ok-box'>✓ 未检测到高风险药物相互作用，药物基因组学推荐暂无紧急干预指征。</div>"
        )

    # ── 图表区域 ───────────────────────────────────────────────
    # 药物推荐临床分析（按药物类别分组，显示CPIC链接）
    drug_analysis_parts = []
    # 按类别归组
    from collections import defaultdict
    category_groups: dict[str, list] = defaultdict(list)
    for gene, rec in report.drug_recs.items():
        for r in rec.recommendations:
            cat = r.drug_category if r.drug_category else "其他"
            category_groups[cat].append((gene, r))

    for cat, items in category_groups.items():
        bg = DRUG_CATEGORY_COLORS.get(cat, "#F5F5F5")
        lbl_color = DRUG_CATEGORY_LABEL_COLORS.get(cat, "#616161")
        cat_html = f"<div class='category-group'>"
        cat_html += f"<div class='category-header'>"
        cat_html += f"<span class='category-badge' style='background:{bg};color:{lbl_color}'>{cat}</span>"
        cat_html += f"</div>"
        for gene, r in items:
            analysis_text = _build_clinical_analysis(r, gene, report.patient.population)
            # CPIC链接（如果source中有URL）
            source_html = ""
            if r.source and ("cpic" in r.source.lower() or "http" in r.source.lower()):
                source_html = f" <a class='source-link' href='{r.source}' target='_blank'>CPIC指南↗</a>"
            grade_icon = {"A": "🔴", "B": "🟠", "C": "🟡", "D": "🟢"}.get(r.recommendation_grade, "⚪")
            cat_html += (
                f"<div class='clinical-analysis'>"
                f"📌 {grade_icon} <b>{r.drug_name}</b>（{gene}）{analysis_text.split('建议：', 1)[-1] if '建议：' in analysis_text else ''}"
                f"{source_html}</div>"
            )
        cat_html += "</div>"
        drug_analysis_parts.append(cat_html)
    drug_analysis_html = "".join(drug_analysis_parts)

    # 表型雷达图
    html_parts.extend([
        "<div class='grid-2'>",
        "<div class='card'>",
        "<h3 class='section-title'>📡 表型推断结果</h3>",
        fig_radar.to_html(full_html=False),
        "</div>",
        "<div class='card'>",
        "<h3 class='section-title'>💊 高优先级药物调整</h3>",
        fig_rec_table.to_html(full_html=False),
        drug_analysis_html,
        "</div>",
        "</div>",
    ])

    # DDI 图表
    html_parts.extend([
        "<div class='grid-2'>",
        "<div class='card'>",
        "<h3 class='section-title'>⚠ 药物相互作用风险分布</h3>",
        fig_ddi_bar.to_html(full_html=False),
        "</div>",
        "<div class='card'>",
        "<h3 class='section-title'>💊 DDI 详情</h3>",
        fig_ddi_table.to_html(full_html=False),
        "</div>",
        "</div>",
    ])

    # 人群频率（堆叠 + 跨人群对比）
    html_parts.extend([
        "<div class='grid-2'>",
        "<div class='card'>",
        "<h3 class='section-title'>🌏 CYP2D6 人群频率</h3>",
        fig_pop_cyp2d6.to_html(full_html=False),
        "</div>",
        "<div class='card'>",
        "<h3 class='section-title'>🌏 CYP2C19 人群频率</h3>",
        fig_pop_cyp2c19.to_html(full_html=False),
        "</div>",
        "</div>",
    ])

    # 跨人群对比（创新点）—— 先算出分析文字再插入
    _phen_cyp2d6 = report.phenotypes.get("CYP2D6")
    _phen_cyp2c19 = report.phenotypes.get("CYP2C19")
    pop_text_cyp2d6 = _build_population_analysis_text(
        "CYP2D6",
        _phen_cyp2d6.phenotype if _phen_cyp2d6 else "NM",
        report.patient.population,
        report,
    )
    pop_text_cyp2c19 = _build_population_analysis_text(
        "CYP2C19",
        _phen_cyp2c19.phenotype if _phen_cyp2c19 else "NM",
        report.patient.population,
        report,
    )
    html_parts.extend([
        "<div class='grid-2'>",
        "<div class='card'>",
        "<h3 class='section-title'>🔬 CYP2D6 人群差异分析（创新点）</h3>",
        f"<p style='font-size:13px;color:#1a237e;background:#e8eaf6;border-radius:6px;padding:10px 14px;'>"
        f"{pop_text_cyp2d6}</p>",
        fig_pop_cmp_cyp2d6.to_html(full_html=False),
        "</div>",
        "<div class='card'>",
        "<h3 class='section-title'>🔬 CYP2C19 人群差异分析（创新点）</h3>",
        f"<p style='font-size:13px;color:#1a237e;background:#e8eaf6;border-radius:6px;padding:10px 14px;'>"
        f"{pop_text_cyp2c19}</p>",
        fig_pop_cmp_cyp2c19.to_html(full_html=False),
        "</div>",
        "</div>",
    ])

    # ── 页脚 ───────────────────────────────────────────────────
    html_parts.extend([
        "<div class='footer'>",
        "本报告由 AI PGx CDS 系统自动生成 · 数据来源: CPIC / PharmGKB / FDA 药物基因组学标签<br>",
        "仅供临床参考，不替代专业医学判断",
        "</div>",
        "</div>",
        "</body>",
        "</html>",
    ])

    html = "\n".join(html_parts)

    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")

    return html
