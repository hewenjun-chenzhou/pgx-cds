"""
genotype_inferrer.py
====================
从人群等位基因频率 + 临床/血液证据，推断具体基因型（*10/*10, *1/*10 等），
再由基因型映射到表型（PM/IM/NM/UM）。

推理层次：
  1. 加载等位基因频率（哈代-温伯格平衡）
  2. 枚举所有可能基因型（diplotype）及其先验概率
  3. 临床症状 + 血液标记 → 贝叶斯更新每个基因型的后验
  4. 取后验最高的基因型，CPIC 功能组合 → 表型
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from .data_loader import get_cache


# ── CPIC 功能组合 → 表型映射 ─────────────────────────────────
# 用 activity score 而非函数对映射（与 _score_to_phenotype 保持一致）
def _alleles_to_phenotype(f1: str, f2: str) -> str:
    """两个等位基因功能 → 表型（基于 activity score）"""
    s1 = _ACTIVITY_SCORE.get(f1, 1.0)
    s2 = _ACTIVITY_SCORE.get(f2, 1.0)
    return _score_to_phenotype(s1 + s2)


# ── 似然比表（按"临床效应方向"分类，方案 2）────────────────
# 设计原则：
#   - no_effect (完全无效): PM 强支持；IM/NM/UM 都应被强烈排除
#   - low_effect (效果差): PM/IM 都可能，但 IM > PM
#   - normal_effect (标准有效): NM 强支持
#   - too_strong (效果过强/中毒): UM 强支持；PM/IM/NM 都被排除

# 关键修正（修复 *10/*10 IM 在 "完全无效" 信号下后验过高的 bug）：
#   - 之前 IM 在 PM 信号下 LR=5.0（错误——IM 是"部分有效"）
#   - 现在 IM 在 no_effect 信号下 LR=0.3（正确——IM 不会完全无效）
_LR_BY_DIRECTION = {
    "no_effect":     {"PM": 50.0, "IM": 0.3,  "NM": 0.01, "UM": 0.01},
    "low_effect":    {"PM": 5.0,  "IM": 30.0, "NM": 0.2,  "UM": 0.05},
    "normal_effect": {"PM": 0.01, "IM": 0.3,  "NM": 30.0, "UM": 5.0},
    "too_strong":    {"PM": 0.01, "IM": 0.05, "NM": 0.5,  "UM": 50.0},
}

# 兼容旧版（如果临床信号 CSV 没有 effect_direction 列则用此表）
_LR_BY_PHENOTYPE_MATCH = {
    "PM":  {"PM": 50.0, "IM": 5.0,  "NM": 0.01, "UM": 0.01},
    "IM":  {"PM": 0.5,  "IM": 30.0, "NM": 0.05, "UM": 0.01},
    "NM":  {"PM": 0.05, "IM": 0.2,  "NM": 50.0, "UM": 0.5},
    "UM":  {"PM": 0.01, "IM": 0.01, "NM": 0.2,  "UM": 50.0},
}

# effect_direction 默认值（用于无此列的旧数据）
_DIRECTION_DEFAULTS = {
    "PM": "no_effect",
    "IM": "low_effect",
    "NM": "normal_effect",
    "UM": "too_strong",
}

# 血液标记的 LR（**软证据**，仅作"基因型倾向"，绝不能把肝损伤归因于基因型）
# 设计原则：所有方向 LR 都在 0.5-2.0 之间，最多 2 倍倾向，不污染先验
# 理由：血液反映器官功能，与基因型只有弱关联（*10 蛋白折叠异常可能轻度升 ALT）
_LR_BLOOD_GENOTYPE = {
    "PM":  {"PM": 1.8, "IM": 1.1, "NM": 0.7,  "UM": 0.6},
    "IM":  {"PM": 0.9, "IM": 1.5, "NM": 0.9,  "UM": 0.6},
    "NM":  {"PM": 0.6, "IM": 0.8, "NM": 1.5,  "UM": 0.9},
    "UM":  {"PM": 0.6, "IM": 0.7, "NM": 0.8,  "UM": 1.8},
}


# ── Activity Score（CPIC 标准）────────────────────────────────
# 基因型 → 基础活性分数，再由肝/肾/炎症修正 → 有效代谢表型
# https://cpicpgx.org/guidelines/
_ACTIVITY_SCORE = {
    "normal": 1.0,       # *1, *2, *27, *33, *34, *35, *39, *45, *46
    "decreased": 0.5,    # *9, *10, *17, *29, *36, *41
    "none": 0.0,         # *3, *4, *5, *6, *7, *8, *11, *12, *13, *14, *15, *16, *18, *19, *20
    "increased": 2.0,    # *1xN, *2xN, *35xN
}


def _genotype_activity_score(a1: str, a2: str, funcs: dict) -> float:
    """基因型 → activity score (CPIC)"""
    f1 = funcs.get(a1, "normal")
    f2 = funcs.get(a2, "normal")
    s1 = _ACTIVITY_SCORE.get(f1, 1.0)
    s2 = _ACTIVITY_SCORE.get(f2, 1.0)
    return s1 + s2


def _score_to_phenotype(score: float) -> str:
    """Activity score → 表型（CPIC 阈值）"""
    if score == 0:
        return "PM"
    elif score < 1.25:
        return "IM"  # IM: 0.25 - 1.25 (CPIC 2019)
    elif score <= 2.25:
        return "NM"  # NM: 1.25 - 2.25
    else:
        return "UM"  # UM: > 2.25


def _organ_function_factor(lab_results: dict) -> tuple[float, list[str]]:
    """
    肝/肾/炎症 → 代谢能力乘数（独立于基因型，反映 phenoconversion）。
    返回 (multiply_factor, reasons)
    multiply_factor 范围 0.2-1.0
    """
    factor = 1.0
    reasons = []

    # ── 肝功能（影响所有 CYP 酶）──
    alt = _safe_float(lab_results.get("ALT"))
    ast = _safe_float(lab_results.get("AST"))
    t_bili = _safe_float(lab_results.get("T.Bili"))
    alb = _safe_float(lab_results.get("Alb"))
    nh3 = _safe_float(lab_results.get("NH3"))

    if alt is not None:
        if alt > 200:    # 重度
            factor *= 0.30; reasons.append(f"ALT={alt:.0f} 重度肝损伤 → 代谢×0.30")
        elif alt > 120:  # 3xULN
            factor *= 0.50; reasons.append(f"ALT={alt:.0f} 肝功能中度异常 → 代谢×0.50")
        elif alt > 80:
            factor *= 0.70; reasons.append(f"ALT={alt:.0f} 肝功能轻度异常 → 代谢×0.70")
        elif alt > 40:
            factor *= 0.90; reasons.append(f"ALT={alt:.0f} 临界高 → 代谢×0.90（软证据：可能 *10 携带者）")

    if ast is not None and ast > 200:
        factor *= 0.85; reasons.append(f"AST={ast:.0f} 肝损伤 → 代谢×0.85")

    if t_bili is not None and t_bili > 2.0:
        factor *= 0.70; reasons.append(f"T.Bili={t_bili:.1f} 黄疸 → 代谢×0.70")

    if alb is not None and alb < 2.5:
        factor *= 0.85; reasons.append(f"Alb={alb:.1f} 低蛋白 → 游离药物↑ ×0.85")

    if nh3 is not None and nh3 > 67.5:
        factor *= 0.80; reasons.append(f"NH3={nh3:.0f} 肝性脑病前兆 → ×0.80")

    # ── 肾功能（影响经肾清除的药物蓄积）──
    crcl = _safe_float(lab_results.get("CrCl"))
    egfr = _safe_float(lab_results.get("eGFR"))
    if crcl is not None and crcl < 30:
        factor *= 0.65; reasons.append(f"CrCl={crcl:.0f} 重度肾衰 → 蓄积风险 ×0.65")
    elif crcl is not None and crcl < 50:
        factor *= 0.85; reasons.append(f"CrCl={crcl:.0f} 轻中度肾损 → 蓄积 ×0.85")
    if egfr is not None and egfr < 30:
        factor *= 0.70; reasons.append(f"eGFR={egfr:.0f} 重度肾衰 → ×0.70")

    # ── 炎症（IL-6 抑制 CYP 活性）──
    crp = _safe_float(lab_results.get("CRP"))
    if crp is not None and crp > 100:
        factor *= 0.70; reasons.append(f"CRP={crp:.0f} 严重炎症 → CYP 抑制 ×0.70")
    elif crp is not None and crp > 50:
        factor *= 0.85; reasons.append(f"CRP={crp:.0f} 中度炎症 → ×0.85")

    return max(factor, 0.20), reasons


def _safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class GenotypePosterior:
    """单个基因型的后验结果"""
    genotype: str             # 如 "*10/*10"
    allele1: str              # *10
    allele2: str              # *10
    function1: str            # decreased
    function2: str            # decreased
    phenotype: str            # PM
    prior_prob: float         # 哈代-温伯格先验
    posterior_prob: float     # 贝叶斯后验


@dataclass
class GenotypeInferenceResult:
    """基因型推断完整结果（含 phenoconversion 后的有效代谢）"""
    gene: str
    population: str
    best_genotype: str                        # 最可能的基因型
    best_phenotype: str                       # 由基因型映射的表型
    best_posterior: float                     # 最可能基因型的后验
    top_genotypes: list[GenotypePosterior]    # top 5 基因型（按后验排序）
    all_posteriors: dict[str, float]          # 所有基因型后验
    evidence_summary: str

    # ── 有效代谢（phenoconversion 后）──
    activity_score: float = 0.0               # 基础 activity score
    organ_factor: float = 1.0                 # 肝/肾/炎症乘数
    organ_reasons: list[str] = field(default_factory=list)
    effective_score: float = 0.0              # activity_score × organ_factor
    effective_phenotype: str = ""             # 临床实际表型（phenoconversion 后）
    needs_detection: bool = False             # 基因型 vs 有效代谢不一致 → 必须测


# ── 复用 phenotype_inferrer 的信号匹配逻辑 ─────────────────
def _match_clinical_signals(gene: str, signals: list[str]) -> list[dict]:
    """
    复用 phenotype_inferrer 的 clinical_signals 表匹配，
    返回 [{target_phenotype, strength, signal_name}, ...]
    """
    from .phenotype_inferrer import _STRENGTH_TO_CONFIDENCE
    cache = get_cache()
    if gene == "CYP2D6":
        sig_df = cache.clinical_signals
    elif gene == "CYP2C19":
        sig_df = cache.clinical_signals_cyp2c19
    else:
        return []

    matches = []
    for text in signals:
        text = text.strip()
        if not text:
            continue
        mask = (
            sig_df["signal_name"].str.contains(text, case=False, na=False)
            | sig_df["context"].str.contains(text, case=False, na=False)
            | sig_df["interpretation"].str.contains(text, case=False, na=False)
        )
        rows = sig_df[mask]
        for _, row in rows.iterrows():
            strength = str(row.get("strength", "弱")).strip()
            if strength not in ("强", "中", "弱"):
                strength = "弱"
            tp = str(row.get("target_phenotype", "NM")).strip().upper()
            if tp not in ("PM", "IM", "NM", "UM"):
                tp = "NM"
            # 取 effect_direction（缺失则按 target_phenotype 默认）
            effect_dir = str(row.get("effect_direction", "")).strip()
            if effect_dir not in ("no_effect", "low_effect", "normal_effect", "too_strong"):
                effect_dir = _DIRECTION_DEFAULTS.get(tp, "low_effect")
            matches.append({
                "target_phenotype": tp,
                "effect_direction": effect_dir,
                "strength": strength,
                "signal_name": str(row["signal_name"]).strip(),
            })
    return matches


# ── 复用 _BLOOD_MARKER_MODIFIERS ──────────────────────────
def _match_blood_signals(gene: str, lab_results: dict) -> list[dict]:
    """
    复用 phenotype_inferrer._BLOOD_MARKER_MODIFIERS，
    返回 [{target_phenotype, strength, marker, value}, ...]
    """
    from .phenotype_inferrer import _BLOOD_MARKER_MODIFIERS
    sigs = []
    if not lab_results:
        return sigs

    gene_mods = _BLOOD_MARKER_MODIFIERS.get(gene, {})
    for (dim, marker, op), (threshold, desc, strength) in gene_mods.items():
        raw = lab_results.get(marker)
        if raw is None or raw == "":
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        triggered = (val > threshold) if op == ">" else (val < threshold)
        if not triggered:
            continue
        sigs.append({
            "target_phenotype": "PM",  # 血液标记 → 推动 PM
            "strength": "弱",
            "marker": marker,
            "value": val,
            "dim": dim,
        })
    return sigs


def _log_safe(x: float) -> float:
    return math.log(max(x, 1e-300)) if x > 0 else -700.0


def infer_genotype(
    gene: str,
    signals: list[str],
    lab_results: Optional[dict] = None,
    population: str = "东亚",
) -> GenotypeInferenceResult:
    """
    基因型层面推断。
    1. 加载该人群等位基因频率
    2. 枚举所有 diplotype（合子对），计算 HWE 先验
    3. 临床/血液证据 → 贝叶斯更新每个 diplotype 的后验
    4. top 5 diplotype + 最可能 diplotype 对应的表型
    """
    cache = get_cache()
    allele_df = cache.cyp_allele_freq

    # ── 加载等位基因频率 ──
    pop_df = allele_df[
        (allele_df["population"] == population) & (allele_df["gene"] == gene)
    ]
    if pop_df.empty:
        # fallback: 东亚
        pop_df = allele_df[
            (allele_df["population"] == "东亚") & (allele_df["gene"] == gene)
        ]

    alleles = list(pop_df["allele"])
    freqs = {row["allele"]: float(row["freq"]) for _, row in pop_df.iterrows()}
    funcs = {row["allele"]: row["function"] for _, row in pop_df.iterrows()}

    # ── 枚举所有 diplotype ──
    diplotypes = []  # (allele1, allele2)
    for i, a1 in enumerate(alleles):
        for a2 in alleles[i:]:
            diplotypes.append((a1, a2))

    # ── 先验（哈代-温伯格：杂合 2pq，纯合 p²）──
    priors: dict[tuple[str, str], float] = {}
    for a1, a2 in diplotypes:
        p1 = freqs[a1]
        p2 = freqs[a2]
        if a1 == a2:
            priors[(a1, a2)] = p1 * p2
        else:
            priors[(a1, a2)] = 2 * p1 * p2

    # ── 收集临床/血液信号 ──
    clin_sigs = _match_clinical_signals(gene, signals)
    blood_sigs = _match_blood_signals(gene, lab_results or {})

    # ── 贝叶斯更新（log 数值稳定）──
    log_scores: dict[tuple[str, str], float] = {}
    for geno in diplotypes:
        a1, a2 = geno
        f1, f2 = funcs[a1], funcs[a2]
        phen = _alleles_to_phenotype(f1, f2)

        log_post = _log_safe(priors[geno])

        # 临床症状（按 effect_direction 查 LR 表；方案 2）
        for s in clin_sigs:
            effect_dir = s.get("effect_direction") or _DIRECTION_DEFAULTS.get(
                s["target_phenotype"], "low_effect"
            )
            # 优先用 effect_direction 表；缺失则用 target_phenotype 表
            if effect_dir in _LR_BY_DIRECTION:
                lr_table = _LR_BY_DIRECTION[effect_dir]
            else:
                lr_table = _LR_BY_PHENOTYPE_MATCH[s["target_phenotype"]]
            base_lr_full = lr_table[phen]
            if s["strength"] == "强":
                base_lr = base_lr_full
            elif s["strength"] == "中":
                base_lr = 1 + (base_lr_full - 1) * 0.5
            else:  # 弱
                base_lr = 1 + (base_lr_full - 1) * 0.2
            log_post += _log_safe(max(base_lr, 1e-10))

        # ⚠️ 血液标记不参与贝叶斯后验计算（方案 A：保守做法）
        # 理由：血液反映器官功能，与基因型只有间接关联
        # 化验数据只用于 phenoconversion（effective_phenotype 调整），不参与后验
        # 避免"肝功能异常→推断为 PM"的逻辑污染
        # blood_sigs 仅收集供 phenoconversion 使用，**不**进入 LR 累乘

        log_scores[geno] = log_post

    # ── 归一化 ──
    max_log = max(log_scores.values())
    unnorm = {g: math.exp(log_scores[g] - max_log) for g in diplotypes}
    total = sum(unnorm.values())
    posteriors = {g: v / total for g, v in unnorm.items()}

    # ── 排序，取 top 5 ──
    sorted_geno = sorted(posteriors.items(), key=lambda x: -x[1])
    top5 = sorted_geno[:5]

    # ── 临床置信度封顶（沿用 phenotype_inferrer 规则）──
    has_clin = len(clin_sigs) > 0
    has_cca = len(blood_sigs) > 0
    if has_clin and has_cca:
        cap = 0.88
    elif has_clin:
        cap = 0.85
    elif has_cca:
        cap = 0.65
    else:
        cap = 1.0

    # 构造返回结果
    top_list = []
    for (a1, a2), post in top5:
        f1, f2 = funcs[a1], funcs[a2]
        phen = _alleles_to_phenotype(f1, f2)
        top_list.append(GenotypePosterior(
            genotype=f"{a1}/{a2}",
            allele1=a1, allele2=a2,
            function1=f1, function2=f2,
            phenotype=phen,
            prior_prob=priors[(a1, a2)],
            posterior_prob=post,
        ))

    best_geno, best_post = top5[0]
    a1, a2 = best_geno
    best_phen = _alleles_to_phenotype(funcs[a1], funcs[a2])
    best_post_capped = min(best_post, cap)

    # 证据描述
    ev_parts = []
    if clin_sigs:
        ev_parts.append(f"临床×{len(clin_sigs)}")
    if blood_sigs:
        ev_parts.append(f"血液×{len(blood_sigs)}")
    if not ev_parts:
        ev_parts.append("仅先验")
    ev_parts.append(f"封顶 ≤{cap:.0%}")

    # ── 计算有效代谢（phenoconversion）──
    # 基因型 → activity score
    activity = _genotype_activity_score(a1, a2, funcs)
    # 肝/肾/炎症 → 乘数
    organ_factor, organ_reasons = _organ_function_factor(lab_results or {})
    # 有效 score = activity × organ_factor
    effective_score = activity * organ_factor
    # 有效表型
    eff_phen = _score_to_phenotype(effective_score)

    # 是否需要基因检测：基因型表型 vs 有效表型不一致时
    needs_detection = best_phen != eff_phen

    return GenotypeInferenceResult(
        gene=gene,
        population=population,
        best_genotype=f"{a1}/{a2}",
        best_phenotype=best_phen,
        best_posterior=best_post_capped,
        top_genotypes=top_list,
        all_posteriors={f"{a1}/{a2}": p for (a1, a2), p in posteriors.items()},
        evidence_summary=" | ".join(ev_parts),
        activity_score=round(activity, 2),
        organ_factor=round(organ_factor, 2),
        organ_reasons=organ_reasons,
        effective_score=round(effective_score, 2),
        effective_phenotype=eff_phen,
        needs_detection=needs_detection,
    )
