"""
detection_recommender.py
========================
根据基因型推断结果 + 东亚 AIM panel 数据，
推荐最优检测方案（精准 vs 经济），帮病人节约成本。

核心思路：
  - 全基因测序 = 金标准，但贵（~1500 元）
  - 单点 TaqMan = 便宜（~80 元/位点），但只能回答"是否携带该等位基因"
  - 分层检测：先用便宜的单点排除高先验基因型，剩下歧义再补测
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from .data_loader import get_cache


@dataclass
class DetectionPlan:
    """单个基因的检测方案"""
    gene: str
    population: str

    # 当前推断
    best_genotype: str
    best_phenotype: str
    top_genotypes: list[tuple[str, str, float]]  # [(genotype, phenotype, posterior), ...]

    # 东亚 AIM 位点（按价值排序）
    key_snps: list[dict] = field(default_factory=list)

    # 推荐方案
    plan_full: dict = field(default_factory=dict)       # 全测序
    plan_economical: dict = field(default_factory=dict)  # 经济分层
    plan_minimal: dict = field(default_factory=dict)     # 最少必要

    # ── 4 档判断结果（新增：避免歧义度高时误判用药）──
    ambiguity_score: float = 0.0          # 歧义度：top1 - top2 后验差
    top2_genotype: str = ""               # 次可能基因型
    top2_phenotype: str = ""              # 次可能表型
    top2_prob: float = 0.0                # 次可能后验
    phenotype_divergent: bool = False     # top1 vs top2 表型是否不同
    recommendation_level: str = "CLEAR"   # 4 档之一
    must_test: bool = False               # 是否必须基因检测
    test_advice: str = ""                 # 人类可读建议
    test_advice_kind: str = "info"        # success/warning/error/info


# ── 4 档判断常量 ──
RECOMMENDATION_LEVEL_ADVICE = {
    # 4 档判断 + 建议文本 + Streamlit 颜色
    "CLEAR":               ("✅ 可按推断用药，无需检测",            "success"),
    "AMBIGUOUS":           ("⚠️ 建议测 SNP 区分（中等歧义）",     "warning"),
    "EXTREMELY_AMBIGUOUS": ("🚨 歧义度高，必须基因检测",          "error"),
    "PHENOTYPE_DIVERGENT": ("🚨 表型不同，必须基因检测",          "error"),
    "EXTREME_AMBIGUOUS":   ("🚨 推断概率过低，必须基因检测",      "error"),
    "NO_DATA":             ("⚠️ 数据不足，无法推断",              "warning"),
}


def _classify_recommendation_level(
    top1: tuple[str, str, float] | None,
    top2: tuple[str, str, float] | None,
) -> tuple[str, float, bool, str]:
    """
    4 档判断（修正：考虑歧义度 + 表型差异 + 极端小概率）。

    返回：(recommendation_level, ambiguity_score, phenotype_divergent, reason)

    阈值（用户确认）：
      - 清晰 (>0.20):    可按推断用药
      - 模糊 (0.05-0.20): 测 SNP 区分
      - 极度模糊 (≤0.05): 必须基因检测
      - 表型差异:        top1_prob < 0.70 且 top1 表型 != top2 表型 → 必须基因检测
      - 极端小概率:       top1_prob < 0.20 → 必须基因检测
    """
    if top1 is None:
        return ("NO_DATA", 0.0, False, "无候选基因型")

    top1_g, top1_p, top1_prob = top1

    if top2 is None:
        # 只有 1 个候选 → 当作清晰
        return ("CLEAR", top1_prob, False, f"仅 1 个候选，置信 {top1_prob:.1%}")

    top2_g, top2_p, top2_prob = top2
    ambiguity_score = top1_prob - top2_prob
    phenotype_divergent = (top1_p != top2_p)

    # ── 第 1 档：极端小概率（top1_prob < 0.20）──
    if top1_prob < 0.20:
        return (
            "EXTREME_AMBIGUOUS",
            ambiguity_score,
            phenotype_divergent,
            f"top1 后验仅 {top1_prob:.1%}（< 20%），推断置信度过低",
        )

    # ── 第 2 档：极度模糊（歧义度 ≤ 0.05）──
    if ambiguity_score <= 0.05:
        return (
            "EXTREMELY_AMBIGUOUS",
            ambiguity_score,
            phenotype_divergent,
            f"歧义度仅 {ambiguity_score:.1%}（≤ 5%），top1 与 top2 难分",
        )

    # ── 第 3 档：表型差异 + top1 置信不足 ──
    # 关键修正：即使歧义度 > 5%，只要 top1 < 70% 且表型不同，仍必须测
    if phenotype_divergent and top1_prob < 0.70:
        return (
            "PHENOTYPE_DIVERGENT",
            ambiguity_score,
            phenotype_divergent,
            f"top1 `{top1_g}` ({top1_p}) vs top2 `{top2_g}` ({top2_p}) 表型不同，"
            f"top1 置信 {top1_prob:.1%} < 70%，必须基因检测",
        )

    # ── 第 4 档：模糊（歧义度 0.05 - 0.20）──
    if ambiguity_score <= 0.20:
        return (
            "AMBIGUOUS",
            ambiguity_score,
            phenotype_divergent,
            f"歧义度 {ambiguity_score:.1%}（5-20%），建议测 1 个 SNP 区分",
        )

    # ── 默认：清晰 ──
    return (
        "CLEAR",
        ambiguity_score,
        phenotype_divergent,
        f"歧义度 {ambiguity_score:.1%}（> 20%），top1 显著领先",
    )


def _select_key_snps(gene: str, population: str) -> list[dict]:
    """
    从 panel 选出该基因 + 该人群最值得测的 SNP。
    优先级：1) CPIC 1A 证据  2) 该人群频率高
    """
    cache = get_cache()
    panel = cache.pgx_detection_panel

    # 过滤出该基因的单点检测（排除全测序）
    snps = panel[
        (panel["gene"] == gene)
        & (panel["snp_id"] != "FULL_GENE")
    ].copy()

    if snps.empty:
        return []

    # 按"是否在覆盖人群"过滤
    def is_relevant(row) -> bool:
        coverage = str(row.get("coverage_pop", ""))
        pop_short = {
            "东亚": "东亚",
            "欧洲": "欧洲",
            "非洲裔": "非洲",
        }.get(population, population)
        return pop_short in coverage

    snps["relevant"] = snps.apply(is_relevant, axis=1)

    # 优先级：1) 相关人群 2) CPIC 1A 3) 频率高
    def priority(row) -> int:
        if not row["relevant"]:
            return 99
        if "1A" in str(row.get("evidence_level", "")):
            return 1
        return 2

    snps["priority"] = snps.apply(priority, axis=1)
    snps = snps.sort_values(["priority", "cost_cny"])

    return [row.to_dict() for _, row in snps.iterrows()]


def recommend_detection(
    gene: str,
    top_genotypes: list[tuple[str, str, float]],
    best_genotype: str,
    best_phenotype: str,
    population: str = "东亚",
) -> DetectionPlan:
    """
    给出 3 个检测方案，由医生/患者选择。
    """
    cache = get_cache()
    panel = cache.pgx_detection_panel
    full_row = panel[
        (panel["gene"] == gene) & (panel["snp_id"] == "FULL_GENE")
    ]
    full_cost = float(full_row.iloc[0]["cost_cny"]) if not full_row.empty else 1500.0

    key_snps = _select_key_snps(gene, population)
    top1_prob = top_genotypes[0][2] if top_genotypes else 0.0

    # ── 方案 1：全测序（金标准）──
    plan_full = {
        "name": "全基因测序",
        "cost": full_cost,
        "turnaround_days": 7,
        "covers": "全部已知 + 新变异 + 拷贝数变异",
        "resolves_uncertainty": "100%",
        "recommended_when": "推断置信度低（top1 < 50%）或患者要求最准确",
    }

    # ── 方案 2：经济分层（AIM 单点）──
    # 核心策略：top 1 vs top 2 哪个等位基因不同 → 测那个 SNP
    # 例：top 1=*10/*10, top 2=*10/*5 → diff={*5} → 只需测 *5
    # 阈值用 top1 后验（= 系统对最优基因型的置信度）
    top1_prob = top_genotypes[0][2] if top_genotypes else 0.0
    if len(top_genotypes) >= 2 and top1_prob < 0.95:
        g1 = top_genotypes[0][0]
        g2 = top_genotypes[1][0]
        # 提取 top 1 vs top 2 不同的等位基因
        diff_alleles = set(g1.split("/")) ^ set(g2.split("/"))

        # 找 panel 中能区分 top 1/top 2 的 SNP（必须落在 diff 里）
        candidate_snps = [
            s for s in key_snps
            if any(a == s.get("allele") for a in diff_alleles)
        ]
        if not candidate_snps:
            # fallback: 用 top 1 含有的等位基因
            g1_alleles = set(g1.split("/"))
            candidate_snps = [s for s in key_snps if s.get("allele") in g1_alleles] or key_snps[:1]

        if len(diff_alleles) == 1:
            # ── 1 个 SNP 就够（diff 只 1 个等位基因）──
            snp = candidate_snps[0]
            cost = float(snp["cost_cny"])
            snp_allele = snp.get("allele", "")

            # ── 阳性分支：含 snp_allele 的基因型胜出 ──
            # 重新计算"含该 allele"的所有基因型的后验和
            positive_total = sum(
                post for geno, post in
                zip([g[0] for g in top_genotypes], [g[2] for g in top_genotypes])
                if snp_allele in geno.split("/")
            )
            # 阳性下最可能的基因型
            positive_top = sorted(
                [(g, ph, post) for g, ph, post in top_genotypes
                 if snp_allele in g.split("/")],
                key=lambda x: -x[2]
            )
            pos_list_text = ", ".join(f"`{g}`({p:.0%})" for g, _, p in positive_top[:3]) or "无"

            # ── 阴性分支：不含 snp_allele 的基因型胜出 ──
            negative_top = sorted(
                [(g, ph, post) for g, ph, post in top_genotypes
                 if snp_allele not in g.split("/")],
                key=lambda x: -x[2]
            )
            neg_list_text = ", ".join(f"`{g}`({p:.0%})" for g, _, p in negative_top[:4]) or "无"

            # ── 如果阴性下 top 仍有歧义，提示下一个 SNP ──
            next_snp_hint = ""
            if len(negative_top) >= 2:
                neg_top1_prob = negative_top[0][2]
                # 阈值：top 1 < 70% 时提示进一步测
                if neg_top1_prob < 0.70:
                    ng1 = negative_top[0][0]
                    ng2 = negative_top[1][0]
                    next_diff = set(ng1.split("/")) ^ set(ng2.split("/"))
                    next_snp_candidates = [
                        s for s in key_snps
                        if s.get("allele") in next_diff
                    ]
                    if next_snp_candidates:
                        ns = next_snp_candidates[0]
                        next_snp_hint = (
                            f"\n    阴性后**还需测** {ns['marker_name']}"
                            f" 区分 `{ng1}` vs `{ng2}`"
                        )
                    else:
                        # panel 没有覆盖的等位基因（如 *1 wildtype）
                        # 建议全测序
                        next_snp_hint = (
                            f"\n    阴性后仍有歧义（`{ng1}` {neg_top1_prob:.0%} vs `{ng2}` {negative_top[1][2]:.0%}）"
                            f"，但区分它们的等位基因（如 `*1`）通常不专门测 SNP"
                            f" → **建议全测序**"
                        )

            if_pos_text = (
                f"**结果**：{pos_list_text}\n"
                f"    建议按最可能 `{positive_top[0][0]}`（{positive_top[0][1]}）用药"
            ) if positive_top else "无结果"

            if_neg_text = (
                f"**结果**：{neg_list_text}\n"
                f"    建议按最可能 `{negative_top[0][0]}`（{negative_top[0][1]}）用药"
                f"{next_snp_hint}"
            ) if negative_top else "无结果"

            plan_economical = {
                "name": f"经济单点（测 {snp_allele}，区分 {g1} vs {g2}）",
                "step1": f"测 {snp['marker_name']}",
                "step1_outcome_if_positive": if_pos_text,
                "step1_outcome_if_negative": if_neg_text,
                "total_cost": cost,
                "saved_vs_full": f"{int(full_cost - cost)}",
            }
        else:
            # ── 2 个 SNP（diff 含 2 个等位基因，比如 *10/*10 vs *5/*5）──
            snp1 = candidate_snps[0] if len(candidate_snps) > 0 else key_snps[0]
            # 第二个 SNP：选另一个 diff allele
            used_allele = snp1.get("allele")
            snp2 = next(
                (s for s in candidate_snps if s.get("allele") != used_allele),
                candidate_snps[1] if len(candidate_snps) > 1 else key_snps[1]
            )
            cost1 = float(snp1["cost_cny"])
            cost2 = float(snp2["cost_cny"])
            plan_economical = {
                "name": f"经济双点（测 {snp1.get('allele')} + {snp2.get('allele')}，区分 {g1} vs {g2}）",
                "step1": f"测 {snp1['marker_name']}",
                "step2": f"测 {snp2['marker_name']}",
                "total_cost": cost1 + cost2,
                "saved_vs_full": f"{int(full_cost - cost1 - cost2)}",
            }
    else:
        # top 1 后验高（≥ 95%），只测 1 个最关键的 SNP 即可
        first_snp = key_snps[0] if key_snps else None
        if first_snp:
            cost1 = float(first_snp["cost_cny"])
            plan_economical = {
                "name": f"经济单点（top1 后验高，测 {first_snp['allele']} 确认）",
                "step1": f"测 {first_snp['marker_name']}",
                "step1_outcome_if_positive": f"阳性 → 确认含 {first_snp['allele']}，倾向 {top_genotypes[0][0]}",
                "step1_outcome_if_negative": f"阴性 → 排除 {first_snp['allele']}",
                "total_cost": cost1,
                "saved_vs_full": f"{int(full_cost - cost1)}",
            }
        else:
            plan_economical = {"name": "无可用 SNP，仅全测", "total_cost": full_cost}

    # ── 方案 3：最少必要（直接按 top 1 用药，不测）──
    plan_minimal = {
        "name": "不测，按推断表型用药（仅建议 top1 后验高时）",
        "total_cost": 0,
        "risk": (
            f"若实际基因型是 top 2（{top_genotypes[1][0] if len(top_genotypes) > 1 else '?'}），"
            f"可能用药剂量不当"
        ) if top1_prob < 0.5 else "top1 后验高，可信",
        "recommended_when": f"top1 后验 {top1_prob:.0%} ≥ 50% 时可考虑",
    }

    # ── 4 档判断（修正歧义度高时误判用药的 bug）──
    top1_tuple = top_genotypes[0] if len(top_genotypes) >= 1 else None
    top2_tuple = top_genotypes[1] if len(top_genotypes) >= 2 else None
    rec_level, ambiguity, phen_div, reason = _classify_recommendation_level(
        top1_tuple, top2_tuple
    )

    # 解析 top2 信息
    top2_g = top2_tuple[0] if top2_tuple else ""
    top2_p = top2_tuple[1] if top2_tuple else ""
    top2_prob_val = top2_tuple[2] if top2_tuple else 0.0

    # 决定是否必须基因检测
    # 必须在以下情况触发"必须测"：
    #   - EXTREMELY_AMBIGUOUS / EXTREME_AMBIGUOUS / PHENOTYPE_DIVERGENT
    #   - AMBIGUOUS（也是建议测，但不是"必须"）
    must_test = rec_level in ("EXTREMELY_AMBIGUOUS", "EXTREME_AMBIGUOUS", "PHENOTYPE_DIVERGENT")

    advice_text, advice_kind = RECOMMENDATION_LEVEL_ADVICE.get(
        rec_level, ("⚠️ 未知状态", "warning")
    )

    return DetectionPlan(
        gene=gene,
        population=population,
        best_genotype=best_genotype,
        best_phenotype=best_phenotype,
        top_genotypes=top_genotypes,
        key_snps=key_snps,
        plan_full=plan_full,
        plan_economical=plan_economical,
        plan_minimal=plan_minimal,
        # ── 新增 4 档判断字段 ──
        ambiguity_score=round(ambiguity, 4),
        top2_genotype=top2_g,
        top2_phenotype=top2_p,
        top2_prob=round(top2_prob_val, 4),
        phenotype_divergent=phen_div,
        recommendation_level=rec_level,
        must_test=must_test,
        test_advice=advice_text,
        test_advice_kind=advice_kind,
    )
