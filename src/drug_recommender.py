"""
drug_recommender.py — Step 3
表型 → 药物推荐（CYP2D6 + CYP2C19）

根据推断表型和患者当前用药，从 drugs / drugs_cyp2c19 表中
查找需要关注/调整的药物，给出 CPIC Grade、具体建议和替代选项。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from .data_loader import get_cache
from .config import GRADE_LABELS, PHENOTYPES


@dataclass
class DrugRecommendation:
    """单个药物的推荐结果"""
    gene: str
    drug_name: str
    drug_category: str                    # 仅 CYP2D6 有意义
    phenotype: str                        # 患者表型 PM/IM/NM/UM
    cyp_effect: str                       # substrate / prodrug / inhibitor / ...
    pm_action: str                        # PM 建议
    im_action: str                        # IM 建议
    um_action: str                        # UM 建议
    current_action: str                   # 根据患者实际表型取对应建议
    recommendation_grade: str            # A/B/C/D
    grade_label: str                      # 等级中文说明
    evidence: str                         # 证据强度
    alternatives: str                     # 替代药物建议
    source: str                           # 数据来源
    recommendation_index: float = 0.0    # 推荐指数 0-100，综合等级+表型+证据计算

    @property
    def needs_adjustment(self) -> bool:
        """
        是否需要剂量调整。

        CPIC 2016 TCA 指南明确：PM 减量 50%；IM 起始剂量低（部分指南）或密切监测；
        NM 维持常规剂量。后续 CPIC 指南（SSRI/SNRI、阿片类、他莫昔芬等）也普遍
        把 IM 视为需要谨慎用药的群体。

        修复前：IM 永远 False → IM 用 Grade A/B 药（如 CYP2D6 IM 阿米替林）
        会被误标"安全"——属 P0 临床错误。

        判定规则：
        - current_action 显式"无需调整/不适用" → False
        - PM/UM → True（明确调整）
        - IM + Grade A/B → True（CPIC 明确建议起始剂量低/减量）
        - IM + Grade C/D → True（CPIC 指南通常给"起始剂量低/密切监测"，
          比 NM 更严格；除 non_substrate 的 grade=D 之外）
        - NM → False（默认常规剂量）
        """
        no_adj_actions = (
            "无需调整", "无数据（无推荐）", "无CPIC推荐", "不适用", "",
            "建议常规剂量（CPIC对NM表型无需调整）",
            "建议常规剂量",
        )
        if self.current_action in no_adj_actions:
            return False

        if self.phenotype in ("PM", "UM"):
            return True

        if self.phenotype == "IM":
            # IM + Grade A/B：CPIC 明确给"起始剂量低"建议
            # IM + Grade D（non_substrate）：无需调整
            return self.recommendation_grade in ("A", "B", "C")

        # NM
        return False


# ── 处方评估新增 dataclass ───────────────────────────────────────

@dataclass
class AlternativeDrugDetail:
    """
    替代药物的详细信息（含推荐指数）。
    """
    drug_name: str           # 替代药物名称
    rec_index: float = 0.0  # 推荐指数 0-100
    grade: str = ""          # CPIC等级 A/B/C/D
    grade_label: str = ""    # 等级中文说明
    reason: str = ""         # 为什么推荐此替代药
    source: str = ""         # 证据来源


@dataclass
class PrescriptionDrugEval:
    """
    单个药物的处方合理性评估结果。
    """
    drug_name: str
    gene: str                              # 涉及基因 CYP2D6 / CYP2C19 / -
    effect: str                            # 作用类型 substrate / prodrug / inhibitor / non_substrate / -
    # 合理性判断
    is_appropriate: bool                   # 原方案是否合理
    reason: str                            # 判断理由
    # 调整建议
    adjusted_action: str                   # 调整后建议（合理则同原方案）
    adjusted_grade: str                    # 调整后推荐等级
    alternatives: str                      # 替代药物名称列表（逗号分隔，用于兼容旧逻辑）
    # 证据
    evidence: str
    source: str
    # DDI 相关
    has_ddi_risk: bool = False             # 是否有相互作用风险
    ddi_partner: str = ""                  # 相互作用的药物
    ddi_warning: str = ""                  # 相互作用说明
    # 替代药详情（含推荐指数）
    alternative_details: list[AlternativeDrugDetail] = field(default_factory=list)


@dataclass
class OtherGeneDrugInfo:
    """
    非CYP2D6/2C19药物的说明信息。
    """
    drug_name: str           # 药物名称
    main_pathway: str        # 主要代谢途径
    therapeutic_class: str   # 治疗类别
    note: str                # 备注
    source: str              # 数据来源


@dataclass
class PrescriptionEvaluation:
    """
    完整处方评估报告。
    """
    overall_appropriate: bool              # 整体处方是否合理
    overall_grade: str                     # 整体推荐等级：A/B/C/D
    eval_details: list[PrescriptionDrugEval] = field(default_factory=list)
    # 综合建议
    keep_meds: list[str] = field(default_factory=list)    # 可继续使用的药物
    replace_meds: list[tuple[str, str]] = field(default_factory=list)  # (原药, 替代药)
    add_meds: list[str] = field(default_factory=list)      # 建议加用的药物
    caution_meds: list[str] = field(default_factory=list)  # 需密切监测的药物
    # 整体说明
    summary: str = ""
    # 未被数据库识别的药物（处方中有但无法评估）
    unmatched_meds: list[str] = field(default_factory=list)
    # 非CYP2D6/2C19药物（在drugs_other_genes.csv中找到）
    other_gene_drugs: list[OtherGeneDrugInfo] = field(default_factory=list)


@dataclass
class GeneRecommendations:
    """单个基因的全部药物推荐"""
    gene: str
    phenotype: str
    confidence: float
    is_fallback: bool
    recommendations: list[DrugRecommendation] = field(default_factory=list)
    high_priority: list[DrugRecommendation] = field(default_factory=list)
    prescription_eval: Optional[PrescriptionEvaluation] = None
    # 高优先级：Grade A/B 且需要调整的药物

    @property
    def needs_intervention(self) -> bool:
        return any(r.needs_adjustment for r in self.recommendations)


def _compute_recommendation_index(phenotype: str, grade: str) -> float:
    """
    计算推荐指数（0-100），综合 CPIC Grade 和表型。
    - Grade A=100分，B=75分，C=50分，D=25分
    - 表型偏离越大、风险越高，分数越低
    """
    grade_score = {"A": 100, "B": 75, "C": 50, "D": 25}.get(grade.upper(), 50)

    # 表型风险系数（基于基因型对药物代谢的影响程度）
    phenotype_coef = {
        "PM": 0.80,   # PM 对多数底物影响大，剂量需调整
        "IM": 0.90,   # IM 略有影响，建议监测
        "NM": 1.00,   # NM 为正常，标准剂量
        "UM": 0.85,   # UM 对部分药物（特别是前药）影响大
    }.get(phenotype, 0.90)

    # 综合推荐指数（0-100）
    index = grade_score * phenotype_coef
    return round(index, 1)


def _build_recommendation(
    gene: str,
    phenotype: str,
    row: pd.Series,
    phenotype_col: str,          # "cyp2d6_effect" 或 "cyp2c19_effect"
) -> DrugRecommendation:
    """从 drugs 表的一行构建 DrugRecommendation 对象"""
    effect_col = phenotype_col          # e.g. "cyp2d6_effect"

    # 决定当前执行的建议
    if phenotype == "PM":
        current_action = str(row.get("pm_action", "")).strip()
    elif phenotype == "UM":
        current_action = str(row.get("um_action", "")).strip()
    elif phenotype == "IM":
        # IM 患者：CPIC 通常建议常规剂量起始，密切监测
        im_action = str(row.get("im_action", "")).strip()
        current_action = im_action if im_action else "建议常规剂量起始，密切监测疗效与不良反应（CPIC对IM通常无需强制减量）"
    else:
        current_action = "建议常规剂量（CPIC对NM表型无需调整）"

    grade = str(row.get("recommendation_grade", "")).strip().upper()
    if grade not in ("A", "B", "C", "D"):
        grade = "C"

    drug_category = ""
    if gene == "CYP2D6" and "drug_category" in row.index:
        drug_category = str(row.get("drug_category", "")).strip()

    return DrugRecommendation(
        gene=gene,
        drug_name=str(row.get("drug_name", "")).strip(),
        drug_category=drug_category,
        phenotype=phenotype,
        cyp_effect=str(row.get(effect_col, "")).strip(),
        pm_action=str(row.get("pm_action", "")).strip(),
        im_action=str(row.get("im_action", "")).strip(),
        um_action=str(row.get("um_action", "")).strip(),
        current_action=current_action,
        recommendation_grade=grade,
        grade_label=GRADE_LABELS.get(grade, "未知"),
        evidence=str(row.get("evidence", "")).strip(),
        alternatives=str(row.get("alternatives", "")).strip(),
        source=str(row.get("source", "")).strip(),
        recommendation_index=_compute_recommendation_index(phenotype, grade),
    )


def recommend_for_gene(
    gene: str,
    phenotype: str,
    current_meds: Optional[list[str]] = None,
) -> GeneRecommendations:
    """
    根据基因表型推荐需要关注的药物。

    Parameters
    ----------
    gene : "CYP2D6" 或 "CYP2C19"
    phenotype : PM / IM / NM / UM
    current_meds : 患者当前用药列表（子串匹配），可传 None（仅返回所有药物）

    Returns
    -------
    GeneRecommendations
    """
    if phenotype not in PHENOTYPES:
        phenotype = "NM"

    cache = get_cache()

    if gene == "CYP2D6":
        df = cache.drugs
        phenotype_col = "cyp2d6_effect"
    elif gene == "CYP2C19":
        df = cache.drugs_cyp2c19
        phenotype_col = "cyp2c19_effect"
    else:
        return GeneRecommendations(gene=gene, phenotype=phenotype,
                                    confidence=0.0, is_fallback=True)

    all_recs: list[DrugRecommendation] = []

    for _, row in df.iterrows():
        rec = _build_recommendation(gene, phenotype, row, phenotype_col)

        # 若提供了 current_meds，仅返回患者在用的药
        if current_meds is not None:
            drug_name = rec.drug_name.lower()
            if not any(med.lower() in drug_name or drug_name in med.lower()
                       for med in current_meds):
                continue

        all_recs.append(rec)

    # 高优先级：Grade A/B 且需要调整
    high_priority = [
        r for r in all_recs
        if r.recommendation_grade in ("A", "B") and r.needs_adjustment
    ]

    return GeneRecommendations(
        gene=gene,
        phenotype=phenotype,
        confidence=1.0,
        is_fallback=False,
        recommendations=all_recs,
        high_priority=high_priority,
    )


def recommend_both_genes(
    phenotypes: dict[str, str],           # {"CYP2D6": "PM", "CYP2C19": "NM"}
    current_meds: Optional[list[str]] = None,
) -> dict[str, GeneRecommendations]:
    """
    同时生成两个基因的药物推荐。
    phenotypes: 包含 CYP2D6 和 CYP2C19 表型的字典
    """
    return {
        gene: recommend_for_gene(gene, phenotypes.get(gene, "NM"), current_meds)
        for gene in ("CYP2D6", "CYP2C19")
    }


# ── 处方合理性评估（综合DDI）─────────────────────────────────────

def _find_ddi(drug_a: str, drug_b: str, ddi_report) -> tuple[bool, str, str]:
    """
    在 ddi_report 中查找两药是否有相互作用。
    Returns (has_ddi, warning_level_str, consequence_str)
    """
    if ddi_report is None or not ddi_report.interactions:
        return False, "", ""
    da = drug_a.strip().lower()
    db = drug_b.strip().lower()
    for ddi in ddi_report.interactions:
        if (ddi.drug_a.lower() in da or da in ddi.drug_a.lower()) and \
           (ddi.drug_b.lower() in db or db in ddi.drug_b.lower()):
            return True, ddi.warning_level, ddi.clinical_consequence
        if (ddi.drug_b.lower() in da or da in ddi.drug_b.lower()) and \
           (ddi.drug_a.lower() in db or db in ddi.drug_a.lower()):
            return True, ddi.warning_level, ddi.clinical_consequence
    return False, "", ""


def _get_alternative_details(
    alt_names: str,
    gene: str,
    phenotype: str,
) -> list[AlternativeDrugDetail]:
    """
    根据替代药名称字符串（用 / 分隔），查询每种替代药的推荐指数和等级。
    如果数据库中找不到该替代药，仍返回基本信息（名称+空评分）。
    """
    if not alt_names or not alt_names.strip():
        return []

    cache = get_cache()
    details: list[AlternativeDrugDetail] = []

    for alt in alt_names.split("/"):
        alt = alt.strip()
        if not alt:
            continue

        # 在对应基因的药物库中查找该替代药
        df = cache.drugs if gene == "CYP2D6" else cache.drugs_cyp2c19
        match = df[df["drug_name"].str.strip() == alt]
        if not match.empty:
            row = match.iloc[0]
            grade = str(row.get("recommendation_grade", "")).strip().upper()
            if grade not in ("A", "B", "C", "D"):
                grade = "C"
            rec_index = _compute_recommendation_index(phenotype, grade)
            details.append(AlternativeDrugDetail(
                drug_name=alt,
                rec_index=rec_index,
                grade=grade,
                grade_label=GRADE_LABELS.get(grade, "未知"),
                reason=f"该药不经过{gene}代谢，指南推荐等级{grade}（{GRADE_LABELS.get(grade, '')}）",
                source=str(row.get("source", "")).strip(),
            ))
        else:
            # 数据库中没有该替代药的基本信息，仍展示名称
            details.append(AlternativeDrugDetail(
                drug_name=alt,
                rec_index=0.0,
                grade="",
                grade_label="未知（建议参考药品说明书）",
                reason="该药不在当前数据库中，建议参考药品说明书或咨询专科",
                source="未收录",
            ))

    return details


def _grade_to_priority(grade: str) -> int:
    """等级越高优先级越大（用于综合评级）"""
    return {"A": 4, "B": 3, "C": 2, "D": 1}.get(grade.upper(), 0)


def evaluate_prescription(
    gene_recs: dict[str, GeneRecommendations],
    current_meds: list[str],
    ddi_report,                          # DDICheckReport
) -> dict[str, PrescriptionEvaluation]:
    """
    对患者当前处方进行综合合理性评估，结合药物基因组学和DDI。

    评估逻辑：
    1. 逐药判断：基于基因型判断是否适合该药
    2. DDI检查：该药是否与处方中其他药存在相互作用
    3. 综合调整：若有DDI，即便基因型适合也标记为需关注/不适当
    4. 整体处方等级：取各药最高风险等级
    5. 标识无处方评估的药物（数据库中不存在）
    """
    result: dict[str, PrescriptionEvaluation] = {}

    # 收集所有已匹配到推荐结果的药品（用于识别未匹配药物）
    matched_drug_names: set[str] = set()
    for rec in gene_recs.values():
        for r in rec.recommendations:
            matched_drug_names.add(r.drug_name.strip().lower())

    # 识别无处方评估的药物，并查询是否属于非CYP2D6/2C19药物
    unmatched_meds: list[str] = []
    other_gene_drugs: list[OtherGeneDrugInfo] = []
    cache = get_cache()
    for med in current_meds:
        med_lower = med.strip().lower()
        if not any(med_lower in d or d in med_lower for d in matched_drug_names):
            unmatched_meds.append(med)
            # 查询是否在 drugs_other_genes.csv 中（不经过CYP2D6/2C19代谢）
            other_row = cache.get_other_gene_drug(med)
            if other_row is not None:
                other_gene_drugs.append(OtherGeneDrugInfo(
                    drug_name=str(other_row.get("drug_name", "")).strip(),
                    main_pathway=str(other_row.get("main_pathway", "")).strip(),
                    therapeutic_class=str(other_row.get("therapeutic_class", "")).strip(),
                    note=str(other_row.get("note", "")).strip(),
                    source=str(other_row.get("source", "")).strip(),
                ))

    for gene, rec in gene_recs.items():
        if not rec.recommendations:
            continue

        evals: list[PrescriptionDrugEval] = []
        keep: list[str] = []
        replace: list[tuple[str, str]] = []
        caution: list[str] = []

        for r in rec.recommendations:
            # 查找该药与处方中其他药的DDI
            ddi_has_risk = False
            ddi_partner = ""
            ddi_warning = ""
            for other_med in current_meds:
                if other_med.strip() == r.drug_name.strip():
                    continue
                has_ddi, wlevel, consequence = _find_ddi(r.drug_name, other_med, ddi_report)
                if has_ddi:
                    ddi_has_risk = True
                    ddi_partner = other_med
                    ddi_warning = f"与{other_med}合用：{consequence}（{wlevel}风险）"
                    break

            # ── 合理性判断 ───────────────────────────────────────
            # 非 NM 表型但建议"维持常规剂量" → 不合理（应该调整）
            phenotype_actionable = r.phenotype != "NM" and r.needs_adjustment

            # DDI 高风险 → 无论如何都不合理
            ddi_high_risk = ddi_has_risk and ddi_warning and "高" in ddi_warning

            if ddi_high_risk:
                is_appropriate = False
                reason = f"DDI风险过高：{ddi_warning}"
                adjusted_action = f"建议换用替代药物，备选：{r.alternatives}" if r.alternatives else "建议与专科协商调整方案"
                adjusted_grade = "A"  # 高优先级干预
            elif ddi_has_risk:
                is_appropriate = False
                reason = f"存在药物相互作用：{ddi_warning}"
                adjusted_action = f"需密切监测，{r.current_action}"
                adjusted_grade = r.recommendation_grade
            elif phenotype_actionable:
                is_appropriate = False
                reason = f"患者{gene}为{r.phenotype}，{r.current_action}"
                adjusted_action = r.current_action
                adjusted_grade = r.recommendation_grade
            else:
                is_appropriate = True
                reason = f"患者{gene}为{r.phenotype}，{r.current_action}"
                adjusted_action = r.current_action
                adjusted_grade = r.recommendation_grade

            evals.append(PrescriptionDrugEval(
                drug_name=r.drug_name,
                gene=gene,
                effect=r.cyp_effect,
                is_appropriate=is_appropriate,
                reason=reason,
                adjusted_action=adjusted_action,
                adjusted_grade=adjusted_grade,
                alternatives=r.alternatives,
                alternative_details=_get_alternative_details(r.alternatives, gene, r.phenotype),
                evidence=r.evidence,
                source=r.source,
                has_ddi_risk=ddi_has_risk,
                ddi_partner=ddi_partner,
                ddi_warning=ddi_warning,
            ))

            # 分类
            if is_appropriate and not ddi_has_risk:
                keep.append(r.drug_name)
            elif ddi_high_risk and r.alternatives:
                # 找一个替代药
                alt_list = [a.strip() for a in r.alternatives.split("/") if a.strip()]
                if alt_list:
                    replace.append((r.drug_name, alt_list[0]))
            elif ddi_has_risk:
                caution.append(r.drug_name)

        # 整体评级 = 最严重那药的等级
        worst_grade = max(evals, key=lambda e: _grade_to_priority(e.adjusted_grade)).adjusted_grade if evals else "A"
        overall_appropriate = all(e.is_appropriate for e in evals) if evals else True

        # 生成总结
        summary_parts = []
        if keep:
            summary_parts.append(f"可继续：{', '.join(keep)}")
        if replace:
            for old, new in replace:
                summary_parts.append(f"建议替换：{old} → {new}")
        if caution:
            summary_parts.append(f"需监测：{', '.join(caution)}")

        result[gene] = PrescriptionEvaluation(
            overall_appropriate=overall_appropriate,
            overall_grade=worst_grade,
            eval_details=evals,
            keep_meds=keep,
            replace_meds=replace,
            add_meds=[],
            caution_meds=caution,
            summary="；".join(summary_parts) if summary_parts else "处方合理，无特殊干预",
            unmatched_meds=unmatched_meds,
            other_gene_drugs=other_gene_drugs,
        )

    return result
