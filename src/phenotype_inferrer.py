"""
phenotype_inferrer.py — Step 2
临床信号 → 表型推断（CYP2D6 + CYP2C19）

核心逻辑：
1. 将患者的文字信号（症状、用药史、反应描述）与
   clinical_signals 表做子串模糊匹配；
2. 按 signal_name / interpretation 双重匹配；
3. 多信号结果投票，综合强度，推断最终表型；
4. 支持"无信号"时回退到默认表型（NM）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from .data_loader import get_cache
from .config import PHENOTYPES, SIGNAL_STRENGTHS


@dataclass
class PhenotypeSignalMatch:
    """单个信号匹配结果"""
    gene: str                         # "CYP2D6" / "CYP2C19"
    signal_name: str                  # 匹配到的信号名
    signal_text: str                  # 用户输入的原始信号文本
    target_phenotype: str             # 推断表型 PM/IM/NM/UM
    strength: str                     # 强/中/弱
    confidence: float                 # 0.0–1.0，信号强度映射
    interpretation: str               # 临床解释


@dataclass
class InferredPhenotype:
    """单个基因的推断表型结果"""
    gene: str                         # "CYP2D6" / "CYP2C19"
    phenotype: str                    # PM / IM / NM / UM
    confidence: float                 # 0.0–1.0，临床信号一致性（基于投票，非先验）
    genotype: str = ""                # 星型等位基因，如 *2/*2，若无检测数据则为空
    matched_signals: list[PhenotypeSignalMatch] = field(default_factory=list)
    is_fallback: bool = False         # True=无信号，用人群频率作为先验
    # 人群先验信息（用于解释为何该表型被选中）
    population: str = "东亚"           # 使用的人群
    prior_prob: float = 0.0           # 该表型在该人群中的先验概率
    posterior_prob: float = 0.0       # 综合信号后的后验概率
    evidence_strength: str = ""       # 综合证据强度描述
    # 基因型层面推断（top 5 diplotype 后验）
    best_genotype: str = ""           # 如 "*1/*10"
    top_genotypes: list = field(default_factory=list)  # [(genotype, phenotype, post), ...]
    # ── v1.3 新增：基因型详细推理数据（让选中节点可展示具体计算过程）──
    # 每个元素含: genotype / allele1 / allele2 / function1 / function2 /
    #            phenotype / prior_prob / posterior_prob
    top_genotypes_detailed: list = field(default_factory=list)
    # 所有候选基因型的后验（含非 Top 5），用于 UI 查任意节点的概率
    gt_all_posteriors: dict = field(default_factory=dict)  # {"*1/*10": 0.336, ...}
    # 有效代谢（phenoconversion：基因型 + 肝/肾/炎症）
    activity_score: float = 0.0
    organ_factor: float = 1.0
    organ_reasons: list = field(default_factory=list)
    effective_phenotype: str = ""
    needs_detection: bool = False
    # ── 表型节点后验分布（v1.1 新增：用于 UI 展示每个表型节点概率）──
    # 例 {"PM": 0.41, "IM": 0.50, "NM": 0.08, "UM": 0.01}，四者之和 = 1.0
    phenotype_posteriors: dict = field(default_factory=dict)
    # ── 最可能基因型的后验置信度（用于 fallback 提示：<60% 建议基因检测）──
    best_genotype_confidence: float = 0.0

    def get_node_lr_breakdown(self, node_idx: int) -> list[dict]:
        """
        v1.3 新增：返回第 node_idx 个基因型节点的"每条信号 LR 贡献"列表。

        用于 UI 展示具体计算过程：
          每个 matched_signal 在该基因型（其表型 = self.top_genotypes_detailed[node_idx].phenotype）
          下的似然比 LR。

        复用 genotype_inferrer 的 LR 表，保持计算逻辑与后端一致。

        Returns:
            [
                {"signal_name": ..., "target_phenotype": ..., "strength": ...,
                 "is_blood": bool, "lr": float},
                ...
            ]
        """
        if node_idx >= len(self.top_genotypes_detailed):
            return []
        if not self.matched_signals:
            return []

        node_phen = self.top_genotypes_detailed[node_idx].phenotype

        from .genotype_inferrer import _LR_BY_PHENOTYPE_MATCH, _LR_BLOOD_GENOTYPE

        breakdown = []
        for sig in self.matched_signals:
            is_blood = sig.signal_name.startswith("[CCA")
            if is_blood:
                # 血液 LR（0.5-2.0，软证据）
                lr = _LR_BLOOD_GENOTYPE.get(sig.target_phenotype, {}).get(node_phen, 1.0)
            else:
                # 临床 LR（强度调节）
                base_lr_full = _LR_BY_PHENOTYPE_MATCH.get(sig.target_phenotype, {}).get(node_phen, 1.0)
                if sig.strength == "强":
                    lr = base_lr_full
                elif sig.strength == "中":
                    lr = 1 + (base_lr_full - 1) * 0.5
                else:
                    lr = 1 + (base_lr_full - 1) * 0.2
            breakdown.append({
                "signal_name": sig.signal_name,
                "target_phenotype": sig.target_phenotype,
                "strength": sig.strength,
                "is_blood": is_blood,
                "lr": float(lr),
            })
        return breakdown


# ── 表型功能描述 ──────────────────────────────────────────────
_PHENOTYPE_DESCRIPTIONS = {
    "PM": "携带两个无功能等位基因，酶活性严重受损，药物代谢能力极弱",
    "IM": "携带一个无功能等位基因和一个功能降低等位基因，酶活性降低，药物代谢能力弱于正常",
    "NM": "携带两个功能正常等位基因（或一个功能增强等位基因），酶活性正常，药物代谢能力符合预期",
    "UM": "携带两个功能增强等位基因，或基因 duplication，酶活性显著增强，药物代谢速度快于正常",
}


def get_phenotype_description(phenotype: str, gene: str = "") -> str:
    """
    返回表型的功能描述文字。

    Parameters
    ----------
    phenotype : PM / IM / NM / UM
    gene : 可选基因名，用于个性化描述

    Returns
    -------
    str : 中文功能描述
    """
    base = _PHENOTYPE_DESCRIPTIONS.get(phenotype.upper(), "未知表型")
    if gene:
        return f"【{gene}】{base}"
    return base


# ── 强度 → 置信度映射 ──────────────────────────────────────────
_STRENGTH_TO_CONFIDENCE = {
    "强": 1.0,
    "中": 0.65,
    "弱": 0.35,
}

# ── 表型优先级（用于综合多信号时决策）────────────────────────────
_PHENOTYPE_PRIORITY = {p: i for i, p in enumerate(["PM", "UM", "IM", "NM"])}


def _normalize_phenotype(raw: str) -> str:
    """将各种文本表型名称标准化为 PM/IM/NM/UM"""
    if not isinstance(raw, str):
        return "NM"
    raw = raw.strip().upper()
    mapping = {
        "POOR": "PM", "慢代谢": "PM", "PM_": "PM",
        "INTERMEDIATE": "IM", "中间": "IM", "IM_": "IM",
        "NORMAL": "NM", "正常": "NM", "NM_": "NM",
        "ULTRA-RAPID": "UM", "超快": "UM", "UM_": "UM",
    }
    for key, val in mapping.items():
        if key in raw:
            return val
    if raw in PHENOTYPES:
        return raw
    return "NM"


import re as _re
import math as _math

# ── 贝叶斯数值稳定性辅助 ────────────────────────────────────────
_LOG_MIN = -700.0   # 避免 log(0) 和下溢


def _log_safe(x: float) -> float:
    """数值稳定的 log，避免 log(0) → -inf"""
    return _math.log(max(x, 1e-300)) if x > 0 else _LOG_MIN


def _exp_log(x: float) -> float:
    """数值稳定的 exp，避免上溢"""
    return _math.exp(min(x, 700.0))


# ── 星型等位基因提取 ────────────────────────────────────────────
_STAR_ALLELE_RE = _re.compile(r"\*\d+(?:[*/]\d+)*")


def _extract_genotype(signals: list[str]) -> str:
    """从信号文本中提取第一个星型等位基因型，如 *1/*1, *10/*10"""
    for s in signals:
        m = _STAR_ALLELE_RE.search(s)
        if m:
            return m.group(0)
    return ""


def _parse_phenotype_from_text(text: str) -> str | None:
    """
    从信号文本中直接解析表型关键词（PM/IM/NM/UM）。
    优先于症状匹配使用。
    """
    text_upper = text.upper()
    # 按优先级匹配：PM最严重先检出
    if any(k in text_upper for k in ["PM", "POOR", "慢代谢", "弱代谢"]):
        return "PM"
    if any(k in text_upper for k in ["UM", "ULTRA-RAPID", "超快", "加速代谢"]):
        return "UM"
    if any(k in text_upper for k in ["IM", "INTERMEDIATE", "中间"]):
        return "IM"
    if any(k in text_upper for k in ["NM", "NORMAL", "正常", "EM"]):
        return "NM"
    return None


# ── 血液标记物 → 表型修饰信号（CCA 反馈层）────────────────
# 基于老师 3 步链第 1 步 CCA：血液标记异常 ↔ 代谢表型关联
# 这些是"弱证据"信号（LR=3），仅修正贝叶斯后验
_BLOOD_MARKER_MODIFIERS = {
    "CYP2D6": {
        # ── 临界高：弱证据（LR 1.1-1.5），提示 *10 携带者倾向 ──
        # 临床：*10 蛋白折叠异常 → 内质网应激 → 临界 ALT
        ("肝临界", "ALT", ">"):  (40,   "ALT>40 临界（*10 携带者倾向）",     "弱"),
        ("肝临界", "AST", ">"):  (35,   "AST>35 临界（*10 携带者倾向）",     "弱"),
        # ── 肝功能异常 → CYP2D6 活性下降 → 推向 PM/IM ──
        ("肝", "ALT", ">"):     (120,  "ALT>3xULN → CYP2D6 代谢能力下降", "弱"),
        ("肝", "AST", ">"):     (105,  "AST>3xULN → CYP2D6 代谢能力下降", "弱"),
        ("肝", "T.Bili", ">"):  (2.0,  "T.Bili>2 → CYP2D6 代谢能力下降",  "弱"),
        ("肝", "Alb", "<"):     (2.5,  "Alb<2.5 → 游离药物↑",              "弱"),
        ("肝", "NH3", ">"):     (67.5, "NH3>1.5xULN → 严重肝功能障碍",     "弱"),
        # 炎症 → IL-6 抑制 CYP
        ("炎症", "CRP", ">"):    (50,   "CRP>50 → 全身炎症抑制 CYP 活性",   "弱"),
        # 肾功能 → 药物蓄积（间接）
        ("肾", "CrCl", "<"):     (30,   "CrCl<30 → 药物蓄积风险",           "弱"),
        ("肾", "eGFR", "<"):     (30,   "eGFR<30 → 药物蓄积风险",           "弱"),
    },
    "CYP2C19": {
        ("肝临界", "ALT", ">"):  (40,   "ALT>40 临界",                        "弱"),
        ("肝临界", "AST", ">"):  (35,   "AST>35 临界",                        "弱"),
        ("肝", "ALT", ">"):     (120,  "ALT>3xULN → CYP2C19 代谢能力下降", "弱"),
        ("肝", "AST", ">"):     (105,  "AST>3xULN → CYP2C19 代谢能力下降", "弱"),
        ("肝", "T.Bili", ">"):  (2.0,  "T.Bili>2 → CYP2C19 代谢能力下降",  "弱"),
        ("肝", "Alb", "<"):     (2.5,  "Alb<2.5 → 游离药物↑",               "弱"),
        ("肝", "NH3", ">"):     (67.5, "NH3>1.5xULN → 严重肝功能障碍",      "弱"),
        ("炎症", "CRP", ">"):    (50,   "CRP>50 → 全身炎症抑制 CYP 活性",    "弱"),
        ("肾", "CrCl", "<"):     (30,   "CrCl<30 → PPI/氯吡格雷蓄积",       "弱"),
        ("肾", "eGFR", "<"):     (30,   "eGFR<30 → 药物蓄积风险",            "弱"),
    },
}


def _generate_blood_marker_signals(
    gene: str, lab_results: dict
) -> list[PhenotypeSignalMatch]:
    """
    从血液标记物生成 CCA 修饰信号。
    这是老师 3 步链第 1 步（CCA 发现血液↔基因型关联）的反馈层。
    所有信号强度统一为"弱"（LR=3），仅参与贝叶斯修正。
    """
    signals: list[PhenotypeSignalMatch] = []
    if not lab_results:
        return signals

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

        signals.append(PhenotypeSignalMatch(
            gene=gene,
            signal_name=f"[CCA·{dim}] {desc}",
            signal_text=f"{marker}={val}",
            target_phenotype="PM",  # 器官功能异常 → 推向弱代谢
            strength=strength,
            confidence=_STRENGTH_TO_CONFIDENCE.get(strength, 0.35),
            interpretation=f"{marker}={val}（阈值 {op} {threshold}）",
        ))

    return signals


def infer_phenotype(
    gene: str,
    signals: list[str],
    default_phenotype: str = "NM",
    population: str = "东亚",
    lab_results: Optional[dict] = None,
) -> InferredPhenotype:
    """
    从信号列表推断指定基因的表型（贝叶斯框架 + 人群先验）。

    Parameters
    ----------
    gene : "CYP2D6" 或 "CYP2C19"
    signals : 患者临床信号文本列表，如 ["可待因镇痛完全无效", "QT延长"]
    default_phenotype : 无匹配信号时返回的默认表型（被人群先验覆盖）
    population : 人群名称（用于获取先验概率），默认"东亚"

    Returns
    -------
    InferredPhenotype
        confidence 字段表示信号一致性（信号投票归一化分），
        posterior_prob 表示贝叶斯后验概率（人群先验 × 证据权重），
        prior_prob 为人群先验概率。
    """
    cache = get_cache()

    # 提取星型基因型（用于显示）
    genotype = _extract_genotype(signals)

    # ── 加载人群先验频率 ─────────────────────────────────────────
    pop_row = cache.get_population_freq(gene, population)
    if pop_row is not None:
        prior_map: dict[str, float] = {
            "PM": float(pop_row.get("pm_freq", 0.0)),
            "IM": float(pop_row.get("im_freq", 0.0)),
            "NM": float(pop_row.get("nm_freq", 0.0)),
            "UM": float(pop_row.get("um_freq", 0.0)),
        }
        freq_source = str(pop_row.get("source", "")).strip()
    else:
        # 无先验数据时使用均匀先验
        prior_map = {p: 0.25 for p in PHENOTYPES}
        freq_source = "无数据（使用均匀先验）"

    # ── 优先级1：从信号文本中识别表型关键词（PM/IM/NM/UM）───
    # v1.1 修复：不再提前 return —— 让下方贝叶斯 + 基因型推断照常接管
    # 否则 best_genotype/top_genotypes/phenotype_posteriors 全部为空
    # 改为：把识别到的关键词作为 LR=100 的强证据加入下方的 matches 列表
    text_phenotype_hints: list[str] = []
    for text in signals:
        text = text.strip()
        if not text:
            continue
        parsed = _parse_phenotype_from_text(text)
        if parsed and parsed not in text_phenotype_hints:
            text_phenotype_hints.append(parsed)

    # ── 优先级2：临床信号症状匹配 ────────────────────────────────
    if gene == "CYP2D6":
        sig_df = cache.clinical_signals
    elif gene == "CYP2C19":
        sig_df = cache.clinical_signals_cyp2c19
    else:
        return InferredPhenotype(gene=gene, phenotype=default_phenotype,
                                 genotype=genotype,
                                 confidence=0.0, is_fallback=True)

    matches: list[PhenotypeSignalMatch] = []

    def _is_subseq(needle: str, haystack: str) -> bool:
        """中文模糊匹配：needle 的中文字符在 haystack 中按序出现（忽略英文缩写如ADR/CYP）"""
        if not needle:
            return False
        import re as _re
        cn = ''.join(_re.findall(r'[一-龥]', needle))
        if len(cn) < 3:
            return False
        i = 0
        for c in cn:
            idx = haystack.find(c, i)
            if idx == -1:
                return False
            i = idx + 1
        return True

    for text in signals:
        text = text.strip()
        if not text:
            continue

        # 1) 精确子串匹配
        mask_exact = (
            sig_df["signal_name"].str.contains(text, case=False, na=False)
            | sig_df["context"].str.contains(text, case=False, na=False)
            | sig_df["interpretation"].str.contains(text, case=False, na=False)
        )
        # 2) 中文模糊子序列匹配（用户短词也能命中长 signal_name）
        if len(text) >= 3:
            mask_fuzzy = (
                sig_df["signal_name"].apply(lambda x: _is_subseq(text, str(x)) if pd.notna(x) else False)
                | sig_df["context"].apply(lambda x: _is_subseq(text, str(x)) if pd.notna(x) else False)
                | sig_df["interpretation"].apply(lambda x: _is_subseq(text, str(x)) if pd.notna(x) else False)
            )
            mask = mask_exact | mask_fuzzy
        else:
            mask = mask_exact
        rows = sig_df[mask]

        for _, row in rows.iterrows():
            strength_raw = str(row.get("strength", "弱")).strip()
            strength = strength_raw if strength_raw in SIGNAL_STRENGTHS else "弱"
            signal_conf = _STRENGTH_TO_CONFIDENCE.get(strength, 0.35)
            matches.append(PhenotypeSignalMatch(
                gene=gene,
                signal_name=str(row["signal_name"]).strip(),
                signal_text=text,
                target_phenotype=_normalize_phenotype(row.get("target_phenotype", "NM")),
                strength=strength,
                confidence=signal_conf,
                interpretation=str(row.get("interpretation", "")).strip(),
            ))

    # ── 合并：血液标记物 → CCA 修饰信号（老师链第1步反馈层）──
    # 即使没有临床症状，血液标记超阈值也会生成弱证据
    blood_signals = _generate_blood_marker_signals(gene, lab_results or {})
    matches.extend(blood_signals)

    # ── 合并：文本识别关键词（v1.1 新增：作为 LR=100 强证据）──
    # 把"病历文本明确写 PM/IM/NM/UM"这种强提示，注入贝叶斯循环
    # 这样下游的基因型推断能正常输出 best_genotype/top_genotypes/phenotype_posteriors
    for hint in text_phenotype_hints:
        matches.append(PhenotypeSignalMatch(
            gene=gene,
            signal_name=f"[文本识别] 病历明确写明 {hint}",
            signal_text=",".join(signals),
            target_phenotype=hint,
            strength="强",       # → LR=100，强烈指向该表型
            confidence=1.0,
            interpretation=f"病历文本含 '{hint}' 关键词（直接识别）",
        ))

    # ── 无临床信号：回退到人群频率先验（而非固定NM）──────────────
    if not matches:
        # 取人群中出现概率最高的表型作为默认
        fallback_phenotype = max(prior_map, key=lambda p: prior_map[p])
        fallback_prior = prior_map[fallback_phenotype]
        result = InferredPhenotype(
            gene=gene,
            phenotype=fallback_phenotype,
            genotype=genotype,
            confidence=round(fallback_prior, 3),       # 用先验概率作为置信度
            is_fallback=True,
            population=population,
            prior_prob=fallback_prior,
            posterior_prob=fallback_prior,
            evidence_strength=f"无临床信号，回退至人群频率先验（{population}，来源：{freq_source}）",
            # fallback 时节点后验 = 人群先验（PM/IM/NM/UM 各自人群频率）
            phenotype_posteriors={
                p: round(prior_map.get(p, 0.0), 4) for p in PHENOTYPES
            },
        )

        # ── v1.1 新增：fallback 路径也调用 infer_genotype() 填充基因型节点 ──
        # 让医生能看到"无信号时人群最常见的 5 个基因型分布"，
        # 而非"无候选基因型节点"——更信息丰富
        try:
            from .genotype_inferrer import infer_genotype
            gt_result = infer_genotype(gene, [], lab_results=None, population=population)
            result.best_genotype = gt_result.best_genotype
            result.top_genotypes = [
                (g.genotype, g.phenotype, g.posterior_prob)
                for g in gt_result.top_genotypes
            ]
            # ── v1.3 新增：填充详细推理数据（让 UI 能展示具体计算过程）──
            result.top_genotypes_detailed = list(gt_result.top_genotypes)
            result.gt_all_posteriors = dict(gt_result.all_posteriors)
            result.activity_score = gt_result.activity_score
            result.organ_factor = gt_result.organ_factor
            result.organ_reasons = gt_result.organ_reasons
            result.effective_phenotype = gt_result.effective_phenotype
            result.needs_detection = gt_result.needs_detection

            # ── best_genotype_confidence（用于 fallback 提示：<60% 建议基因检测）──
            result.best_genotype_confidence = round(gt_result.best_posterior, 4)
        except Exception:
            # 兜底：万一 infer_genotype 失败，仍返回（不影响主流程）
            pass

        return result

    # ── 有临床信号：贝叶斯更新 ──────────────────────────────────
    # 诊断似然比（Diagnostic Likelihood Ratio, DLR）——
    # 临床PGx信号（药物失效/严重不良反应）对应强诊断证据：
    #   强信号：LR = 100（"可待因完全无效"在PM中几乎必然，在NM/IM/UM中极罕见）
    #   中信号：LR = 10
    #   弱信号：LR = 3
    #   否定信号（不支持某表型）：LR = 0.01（强力排除）
    _LR_SUPPORT = {"强": 100.0, "中": 10.0, "弱": 3.0}
    _LR_REJECT = 0.01   # 信号明确不支持该表型 → 强力否定

    phenotype_posteriors: dict[str, float] = {}
    phenotype_log_score: dict[str, float] = {}

    for p in PHENOTYPES:
        log_posterior = _log_safe(prior_map[p])
        for m in matches:
            if m.target_phenotype == p:
                # 信号支持该表型
                lr = _LR_SUPPORT.get(m.strength, 1.5)
                log_posterior += _log_safe(lr)
            else:
                # 信号明确不支持该表型 → 强力否定
                log_posterior += _log_safe(_LR_REJECT)
        phenotype_log_score[p] = log_posterior
        phenotype_posteriors[p] = _exp_log(log_posterior)

    # 归一化后验概率
    total_posterior = sum(phenotype_posteriors.values())
    if total_posterior > 0:
        for p in PHENOTYPES:
            phenotype_posteriors[p] /= total_posterior

    # 取后验概率最高的表型
    best_phenotype = max(phenotype_posteriors, key=lambda p: phenotype_posteriors[p])
    best_posterior = phenotype_posteriors[best_phenotype]

    # ── 用基因型推断覆盖表型+置信度（保证表型与最可能基因型一致）──
    # 比如可待因无效 → 旧逻辑直接说 PM 85%，
    # 但东亚人群 *1/*10 (IM, 33.6% 先验) >> *5/*5 (PM, 0.3% 先验)，
    # 真实最可能基因型是 *1/*10，IM 才是真相
    try:
        from .genotype_inferrer import infer_genotype
        gt_result = infer_genotype(gene, signals, lab_results=lab_results, population=population)
        best_phenotype = gt_result.best_phenotype
        best_posterior = gt_result.best_posterior   # 已在 genotype_inferrer 内封顶
    except Exception:
        # 兜底：原表型推断 + 手动封顶
        has_clin = any(not m.signal_name.startswith("[CCA") for m in matches)
        has_cca = any(m.signal_name.startswith("[CCA") for m in matches)
        if has_clin and has_cca:
            cap = 0.88
        elif has_clin:
            cap = 0.85
        elif has_cca:
            cap = 0.65
        else:
            cap = 1.0
        if best_posterior > cap:
            best_posterior = cap

    # 置信度 = 后验概率（基因型推断已封顶，兜底路径也封顶）
    signal_consistency = round(best_posterior, 4)

    # 证据强度描述（区分临床症状 vs 血液标记 CCA 信号）
    clinical_signals_list = [m for m in matches if not m.signal_name.startswith("[CCA")]
    cca_signals_list = [m for m in matches if m.signal_name.startswith("[CCA")]
    strong_count = sum(1 for m in matches if m.strength == "强")
    medium_count = sum(1 for m in matches if m.strength == "中")
    weak_count = sum(1 for m in matches if m.strength == "弱")
    # 在 evidence 描述里追加"封顶"提示
    has_clin_disp = bool(clinical_signals_list)
    has_cca_disp = bool(cca_signals_list)
    if has_clin_disp and has_cca_disp:
        cap_disp = 0.88
    elif has_clin_disp:
        cap_disp = 0.85
    elif has_cca_disp:
        cap_disp = 0.65
    else:
        cap_disp = 1.0
    cap_label = f"（临床封顶 ≤{cap_disp:.0%}）" if cap_disp < 1.0 else ""
    evidence_parts = [f"贝叶斯后验={best_posterior:.1%}{cap_label}"]
    if clinical_signals_list:
        evidence_parts.append(
            f"临床症状×{len(clinical_signals_list)}（"
            f"强{sum(1 for m in clinical_signals_list if m.strength=='强')}/"
            f"中{sum(1 for m in clinical_signals_list if m.strength=='中')}/"
            f"弱{sum(1 for m in clinical_signals_list if m.strength=='弱')}）"
        )
    if cca_signals_list:
        evidence_parts.append(f"CCA血液标记×{len(cca_signals_list)}")
    evidence_parts.append(f"先验来源：{freq_source}")
    evidence_strength = " | ".join(evidence_parts)

    result = InferredPhenotype(
        gene=gene,
        phenotype=best_phenotype,
        genotype=genotype,
        confidence=round(signal_consistency, 3),
        matched_signals=matches,
        is_fallback=False,
        population=population,
        prior_prob=round(prior_map.get(best_phenotype, 0.0), 4),
        posterior_prob=round(best_posterior, 4),
        evidence_strength=evidence_strength,
        # ── 表型节点后验（4 个表型各自的贝叶斯后验）──
        phenotype_posteriors={
            p: round(phenotype_posteriors.get(p, 0.0), 4) for p in PHENOTYPES
        },
    )

    # ── 附加：基因型层面推断（哈代-温伯格先验 + 贝叶斯 + phenoconversion）──
    try:
        from .genotype_inferrer import infer_genotype
        gt_result = infer_genotype(gene, signals, lab_results=lab_results, population=population)
        result.best_genotype = gt_result.best_genotype
        result.top_genotypes = [
            (g.genotype, g.phenotype, g.posterior_prob)
            for g in gt_result.top_genotypes
        ]
        # ── v1.3 新增：填充详细推理数据（让 UI 能展示具体计算过程）──
        result.top_genotypes_detailed = list(gt_result.top_genotypes)
        result.gt_all_posteriors = dict(gt_result.all_posteriors)
        # phenoconversion 信息
        result.activity_score = gt_result.activity_score
        result.organ_factor = gt_result.organ_factor
        result.organ_reasons = gt_result.organ_reasons
        result.effective_phenotype = gt_result.effective_phenotype
        result.needs_detection = gt_result.needs_detection

        # ── best_genotype_confidence（用于 fallback 提示：<60% 建议基因检测）──
        result.best_genotype_confidence = round(gt_result.best_posterior, 4)
    except Exception:
        pass

    return result


def infer_both_genes(
    signals_cyp2d6: list[str],
    signals_cyp2c19: list[str],
    population: str = "东亚",
) -> dict[str, InferredPhenotype]:
    """
    同时推断两个基因的表型。
    population: 人群名称（用于先验概率），默认"东亚"
    Returns {"CYP2D6": InferredPhenotype, "CYP2C19": InferredPhenotype}
    """
    return {
        "CYP2D6": infer_phenotype("CYP2D6", signals_cyp2d6, population=population),
        "CYP2C19": infer_phenotype("CYP2C19", signals_cyp2c19, population=population),
    }
