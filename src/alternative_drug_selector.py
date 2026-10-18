"""
alternative_drug_selector.py — Step 5
基于治疗类别（drug_category）推荐 PGx 安全的替代药物

当某药物存在高风险 DDI 或表型禁忌时，
系统根据 drug_category 字段匹配同类药物，筛选出对当前患者表型安全的替代选项。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from .data_loader import get_cache
from .config import PHENOTYPES


@dataclass
class AlternativeDrug:
    """单个替代药物建议"""
    original_drug: str           # 原药物名称
    alternative_drug: str        # 替代药物名称
    gene: str                    # 相关基因
    phenotype: str               # 患者表型
    category: str                # 药物类别
    recommendation_grade: str    # 替代药物对该表型的推荐等级
    recommendation_text: str     # 替代建议文字
    reason: str                  # 为什么推荐此替代药物


def _is_safe_for_phenotype(
    drug_row,
    gene: str,
    phenotype: str,
) -> bool:
    """
    根据 drug 表中的 guidance 列判断某药物对特定表型是否安全。
    guidance 示例: "CYP2D6 NM: 正常剂量; CYP2D6 PM: 考虑减量..."
    """
    guidance = str(drug_row.get("guidance", "")).lower()
    gene_key = gene.lower()

    if gene_key not in guidance:
        return True  # 无该基因指导信息，默认安全

    # 简单策略：若 guidance 中出现该表型关键词且未标注"禁忌"或"避免"，视为可用
    phenotype_key = phenotype.lower()
    prohibited = any(k in guidance for k in ["避免", "禁用", "禁忌", "contraindicated"])
    if prohibited and phenotype_key in guidance:
        return False
    return True


def select_alternatives(
    problematic_drug: str,
    gene: str,
    phenotype: str,
    current_meds: list[str],
    max_alternatives: int = 3,
) -> list[AlternativeDrug]:
    """
    为指定药物推荐安全替代选项。

    Parameters
    ----------
    problematic_drug : str
        存在问题的原药物名称
    gene : str
        相关基因（CYP2D6 / CYP2C19）
    phenotype : str
        患者表型（PM / IM / NM / UM）
    current_meds : list[str]
        当前用药列表（排除已使用的药物）
    max_alternatives : int
        最多返回几条推荐

    Returns
    -------
    list[AlternativeDrug]
    """
    cache = get_cache()
    drug_row = cache.get_drug_row(problematic_drug, gene)
    if drug_row is None:
        return []

    category = str(drug_row.get("drug_category", "")).strip()
    if not category:
        return []

    # 在 drug 表中查找同类药物
    gene_col = {
        "CYP2D6": "gene_cyp2d6",
        "CYP2C19": "gene_cyp2c19",
    }.get(gene, "")

    candidates = []
    for _, row in cache.drugs.iterrows():
        if str(row.get("drug_category", "")).strip() != category:
            continue
        if str(row.get("drug_name", "")).strip() == problematic_drug:
            continue
        if str(row.get("drug_name", "")).strip() in current_meds:
            continue
        if gene_col and str(row.get(gene_col, "")).strip().upper() not in (
            gene, "BOTH", ""
        ):
            continue

        # 检查该替代药物对当前表型是否安全
        if not _is_safe_for_phenotype(row, gene, phenotype):
            continue

        rec_grade = str(row.get(f"grade_{phenotype}", "U")).strip()
        rec_text = str(row.get(f"recommendation_{phenotype}", "请参考CPIC指南")).strip()

        candidates.append(AlternativeDrug(
            original_drug=problematic_drug,
            alternative_drug=str(row.get("drug_name", "")).strip(),
            gene=gene,
            phenotype=phenotype,
            category=category,
            recommendation_grade=rec_grade or "U",
            recommendation_text=rec_text or "请参考CPIC指南",
            reason=f"同属「{category}」类别，对{gene} {phenotype}型患者无禁忌",
        ))

    # 按推荐等级排序（A > B > C > D > U），取前 max_alternatives
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "U": 4}
    candidates.sort(key=lambda x: grade_order.get(x.recommendation_grade, 5))
    return candidates[:max_alternatives]


def get_all_alternatives_for_case(
    drug_recommendations: dict,
    ddi_report,
    current_meds: list[str],
) -> dict[str, list[AlternativeDrug]]:
    """
    扫描所有药物推荐和 DDI，批量生成替代药物建议。

    Returns
    -------
    dict[str, list[AlternativeDrug]]
        key = 原药物名, value = 替代药物列表
    """
    results: dict[str, list[AlternativeDrug]] = {}

    # 1. 从高优先级药物推荐中获取问题药物
    for gene, rec in drug_recommendations.items():
        for r in rec.high_priority:
            phenotype = rec.phenotype
            alts = select_alternatives(r.drug_name, gene, phenotype, current_meds)
            if alts:
                results[r.drug_name] = alts

    # 2. 从高风险 DDI 中获取问题药物
    for ddi in ddi_report.high_risk:
        for drug in [ddi.drug_a, ddi.drug_b]:
            # 尝试推断该药关联的基因（简化：查 drug_row）
            cache = get_cache()
            row = cache.get_drug_row(drug, "CYP2D6")
            gene = "CYP2D6"
            if row is None:
                row = cache.get_drug_row(drug, "CYP2C19")
                gene = "CYP2C19"
            if row is None:
                continue
            # 找到患者该基因的表型
            phenotype = drug_recommendations.get(gene, None)
            if phenotype:
                phenotype = phenotype.phenotype
            else:
                phenotype = "NM"
            alts = select_alternatives(drug, gene, phenotype, current_meds)
            if alts and drug not in results:
                results[drug] = alts

    return results
