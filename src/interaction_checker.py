"""
interaction_checker.py — Step 4
DDI（药物相互作用）检测

检测患者当前用药两两之间的药物相互作用，
返回警示级别、临床后果和建议。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from itertools import combinations

from .data_loader import get_cache
from .config import WARNING_LEVELS


@dataclass
class DDIResult:
    """单个 DDI 检测结果"""
    drug_a: str
    drug_b: str
    mechanism: str
    warning_level: str                   # 高/中/低
    clinical_consequence: str
    suggestion: str                     # 临床建议


@dataclass
class DDICheckReport:
    """完整 DDI 报告"""
    interactions: list[DDIResult] = field(default_factory=list)
    high_risk: list[DDIResult] = field(default_factory=list)
    medium_risk: list[DDIResult] = field(default_factory=list)
    low_risk: list[DDIResult] = field(default_factory=list)

    @property
    def has_interactions(self) -> bool:
        return len(self.interactions) > 0

    @property
    def has_high_risk(self) -> bool:
        return len(self.high_risk) > 0


def _check_pair(
    drug_a: str,
    drug_b: str,
    ddi_df: pd.DataFrame,
) -> list[DDIResult]:
    """检测一对药物的相互作用"""
    mask = (
        (
            ddi_df["drug_a"].str.contains(drug_a, case=False, na=False)
            & ddi_df["drug_b"].str.contains(drug_b, case=False, na=False)
        )
        | (
            ddi_df["drug_a"].str.contains(drug_b, case=False, na=False)
            & ddi_df["drug_b"].str.contains(drug_a, case=False, na=False)
        )
    )
    results: list[DDIResult] = []
    for _, row in ddi_df[mask].iterrows():
        wl = str(row.get("warning_level", "低")).strip()
        if wl not in WARNING_LEVELS:
            wl = "低"
        results.append(DDIResult(
            drug_a=str(row["drug_a"]).strip(),
            drug_b=str(row["drug_b"]).strip(),
            mechanism=str(row.get("mechanism", "")).strip(),
            warning_level=wl,
            clinical_consequence=str(row.get("clinical_consequence", "")).strip(),
            suggestion=_generate_suggestion(wl, row),
        ))
    return results


def _generate_suggestion(warning_level: str, row: pd.Series) -> str:
    """根据警示级别生成临床建议"""
    if warning_level == "高":
        return "高风险组合，建议避免联用或更换方案；如必须联用需密切监测疗效和不良反应。"
    elif warning_level == "中":
        return "中等风险，建议监测不良反应，必要时调整剂量。"
    else:
        return "低风险，常规监测即可。"


def check_ddi(current_meds: list[str]) -> DDICheckReport:
    """
    检测当前用药组合中的所有两两药物相互作用。

    Parameters
    ----------
    current_meds : 患者当前药物名称列表

    Returns
    -------
    DDICheckReport
    """
    if len(current_meds) < 2:
        return DDICheckReport()

    cache = get_cache()
    ddi_df = cache.drug_interactions

    all_interactions: list[DDIResult] = []
    seen: set[tuple[str, str]] = set()

    for drug_a, drug_b in combinations(current_meds, 2):
        key = tuple(sorted([drug_a.lower(), drug_b.lower()]))
        if key in seen:
            continue
        results = _check_pair(drug_a, drug_b, ddi_df)
        for r in results:
            k = tuple(sorted([r.drug_a.lower(), r.drug_b.lower()]))
            if k not in seen:
                seen.add(k)
                all_interactions.append(r)

    report = DDICheckReport(
        interactions=all_interactions,
        high_risk=[d for d in all_interactions if d.warning_level == "高"],
        medium_risk=[d for d in all_interactions if d.warning_level == "中"],
        low_risk=[d for d in all_interactions if d.warning_level == "低"],
    )
    return report
