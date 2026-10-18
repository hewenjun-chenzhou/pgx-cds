"""
population_analyzer.py — Step 5
人群基因表型频率分析

查询不同人群的表型频率分布，
辅助评估患者所属人群的基准风险。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .data_loader import get_cache
from .config import POPULATIONS, PHENOTYPES


@dataclass
class PopulationPhenotypeFreq:
    """某人群的表型频率"""
    population: str
    pm_freq: float
    im_freq: float
    nm_freq: float
    um_freq: float
    source: str

    def highest_risk_phenotype(self) -> str:
        """返回风险最高的表型（PM 或 UM，频率较高者）"""
        if self.pm_freq >= self.um_freq:
            return "PM"
        return "UM"

    def to_dict(self) -> dict:
        return {
            "population": self.population,
            "PM": self.pm_freq,
            "IM": self.im_freq,
            "NM": self.nm_freq,
            "UM": self.um_freq,
            "source": self.source,
        }


def get_population_analysis(
    gene: str,
    population: str,
) -> Optional[PopulationPhenotypeFreq]:
    """
    查询某人群的表型频率。

    Parameters
    ----------
    gene : "CYP2D6" 或 "CYP2C19"
    population : "东亚" / "欧洲" / "非洲裔" / "拉美裔"

    Returns
    -------
    PopulationPhenotypeFreq 或 None（未找到）
    """
    cache = get_cache()
    row = cache.get_population_freq(gene, population)
    if row is None:
        return None

    try:
        return PopulationPhenotypeFreq(
            population=str(row.get("population", "")).strip(),
            pm_freq=float(row.get("pm_freq", 0)),
            im_freq=float(row.get("im_freq", 0)),
            nm_freq=float(row.get("nm_freq", 0)),
            um_freq=float(row.get("um_freq", 0)),
            source=str(row.get("source", "")).strip(),
        )
    except (ValueError, TypeError):
        return None


def get_population_comparison(gene: str) -> dict[str, PopulationPhenotypeFreq]:
    """
    返回某基因所有人群的频率数据字典。
    key = 人群名，用于跨人群对比。
    """
    cache = get_cache()
    if gene == "CYP2D6":
        df = cache.phenotype_frequencies_cyp2d6
    elif gene == "CYP2C19":
        df = cache.phenotype_frequencies_cyp2c19
    else:
        return {}

    results = {}
    for _, row in df.iterrows():
        try:
            pf = PopulationPhenotypeFreq(
                population=str(row["population"]).strip(),
                pm_freq=float(row["pm_freq"]),
                im_freq=float(row["im_freq"]),
                nm_freq=float(row["nm_freq"]),
                um_freq=float(row["um_freq"]),
                source=str(row["source"]).strip(),
            )
            results[pf.population] = pf
        except (ValueError, TypeError):
            continue
    return results
