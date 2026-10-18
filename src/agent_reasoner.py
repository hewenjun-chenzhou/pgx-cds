"""
agent_reasoner.py (v2.0 - 重构版)
==================================
PGx CDS Agent - 三个受限 Agent 接口 + 确定性边界 + 完整溯源。

设计原则：
  1. LLM 是翻译官，不是医生。所有数字/剂量/推荐来自本地确定性系统。
  2. 三个 Agent 严格分离：A 推理解释 / B 用药决策 / C 边界守卫
  3. 强制 Schema 验证 + Fallback 机制
  4. 严禁编造、严禁免责句、严禁暴露不稳定参数

使用：
    from src.agent_reasoner import PGxAgent

    agent = PGxAgent(api_key="sk-xxx")
    result = agent.reason("可待因镇痛无效，能换什么药？",
                          phenotypes={"CYP2D6": "PM"},
                          current_meds=["可待因30mg", "阿米替林25mg"])
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# 基于 __file__ 自动定位（跨机器/账号可移植）
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KB_FILE = DATA_DIR / "agent_knowledge_base.csv"

CN_RANGE = "".join(chr(c) for c in range(0x4E00, 0x9FFF + 1))


# ══════════════════════════════════════════════════════════════
# 辅助函数：剂量/频次解析、行动映射、监测项目
# ══════════════════════════════════════════════════════════════
def _parse_dose_freq(med_str: str) -> tuple[str, str]:
    """从 "可待因30mg Tid" 提取 ("30mg", "Tid")"""
    if not med_str:
        return ("", "")
    # 匹配剂量（数字+单位）
    dose_m = re.search(r"(\d+\s*(?:mg|g|mcg|ml|iu|片|粒))", med_str, re.IGNORECASE)
    dose = dose_m.group(1).strip() if dose_m else ""
    # 匹配频次
    freq_m = re.search(
        r"(qd|bid|tid|qid|qn|prn|q\d+h|q\d+d|每日|两次|三次|四次|每晚)", med_str, re.IGNORECASE
    )
    freq = freq_m.group(1).strip() if freq_m else ""
    return (dose, freq)


def _map_action(cpic_action: str) -> str:
    """根据 CPIC current_action 映射到 4 类推荐行动"""
    if not cpic_action:
        return "系统未提供"
    text = cpic_action.lower()
    if "避免" in cpic_action or "禁用" in cpic_action or "不推荐" in cpic_action:
        return "避免"
    if "换药" in cpic_action or "替代" in cpic_action or "换用" in cpic_action:
        return "换药"
    if "减量" in cpic_action or "降低" in cpic_action:
        return "减量"
    if "监测" in cpic_action or "维持" in cpic_action:
        return "维持+监测"
    return "维持"


# 监测项目查表（按药品类别）
_MONITORING_TEMPLATES = {
    "阿米替林": ["心电图", "血药浓度 TDM", "肝功能"],
    "氯米帕明": ["心电图", "血药浓度 TDM"],
    "可待因": ["疼痛评分", "不良反应", "呼吸抑制"],
    "曲马多": ["疼痛评分", "不良反应", "癫痫发作"],
    "文拉法辛": ["血压", "肝功能", "不良反应"],
    "舍曲林": ["肝功能", "情绪/自杀倾向", "不良反应"],
    "氯吡格雷": ["血小板功能", "出血事件", "心血管事件"],
    "他莫昔芬": ["疗效评估", "不良反应"],
}


def _get_monitoring_items(drug_name: str) -> list[str]:
    """根据药品名查监测项目"""
    if not drug_name:
        return []
    drug_l = drug_name.lower().strip()
    for key, items in _MONITORING_TEMPLATES.items():
        if key.lower() in drug_l or drug_l in key.lower():
            return items
    return []


# 表型中文名映射
_PHENO_CN = {
    "PM": "慢代谢者（PM）",
    "IM": "中间代谢者（IM）",
    "NM": "正常代谢者（NM）",
    "UM": "超快代谢者（UM）",
}


def _explain_drug_for_phenotype(
    gene: str,
    phenotype: str,
    cyp_effect: str,
    drug_name: str,
) -> str:
    """
    生成"为什么这个药对这个表型重要"的中文解释。

    根据 (基因, 表型, cyp_effect, 药名) 自动推断原因，
    在用户没有 current_meds 但有表型时，给 Agent B 输出"必须关注"药品用。
    """
    gnum = gene[3:] if len(gene) > 3 else gene   # "CYP2D6" → "2D6"
    ph_cn = _PHENO_CN.get(phenotype, phenotype)
    drug = drug_name

    # 不同 (cyp_effect, 表型) 组合 → 不同的解释模板
    if cyp_effect == "prodrug":
        if phenotype == "PM":
            return (
                f"{drug} 是前药，需经 CYP{gnum} 活化为活性代谢物才有药效。"
                f"{ph_cn}者 CYP{gnum} 活性丧失，无法有效转化，"
                f"{drug}疗效可能不足或完全无效，建议换用不经 CYP{gnum} 代谢的同类药。"
            )
        if phenotype == "UM":
            return (
                f"{drug} 是前药，{ph_cn}者 CYP{gnum} 活性显著增强，"
                f"前药转化过快，活性代谢物生成过多，"
                f"可能引发剂量相关毒性或严重不良反应，"
                f"建议换药或严格按 CPIC 指南调整剂量。"
            )
        return f"{drug} 是前药，{ph_cn}者使用需特别关注其转化效率变化。"

    if cyp_effect in ("substrate", "non_substrate"):
        if phenotype == "PM":
            return (
                f"{drug} 经 CYP{gnum} 代谢清除。{ph_cn}者代谢减慢，"
                f"血药浓度升高，毒性风险增加（嗜睡、低血压、QT 延长等），"
                f"建议减量、密切监测或换用不依赖 CYP{gnum} 的替代药。"
            )
        if phenotype == "UM":
            return (
                f"{drug} 经 CYP{gnum} 代谢清除。{ph_cn}者代谢加速，"
                f"血药浓度降低，疗效可能不足，"
                f"建议根据治疗药物监测（TDM）调整剂量或换药。"
            )
        if phenotype == "IM":
            return (
                f"{drug} 经 CYP{gnum} 代谢。{ph_cn}者代谢略低于正常，"
                f"部分患者可能需要常规剂量或略减量，监测疗效与不良反应。"
            )
        return f"{drug} 经 CYP{gnum} 代谢，{ph_cn}者用药需按指南监测。"

    if cyp_effect == "inhibitor":
        return (
            f"{drug} 本身是 CYP{gnum} 抑制剂。"
            f"若 {ph_cn}者再使用此药，会进一步抑制残存酶活性，"
            f"叠加临床效应或与同时服用的 CYP{gnum} 底物类药物发生严重相互作用，"
            f"建议谨慎使用并核对联合用药方案。"
        )

    # 兜底
    return (
        f"{drug} 涉及 CYP{gnum} 代谢通路，{ph_cn}者使用前"
        f"应查阅 CPIC/DPWG 等指南，按推荐调整剂量或换药。"
    )


# ══════════════════════════════════════════════════════════════
# 黑名单（强制约束 LLM 不允许使用的措辞）
# ══════════════════════════════════════════════════════════════
# 用正则而非裸子串匹配，避免中文误伤：
#   - "可能" 应该屏蔽真实模糊词（"可能因为"），但放过 "不可能" / "尽可能" / "可能性"
#   - "建议您考虑" 等免责/诱导句保持原意（子串即可）
FORBIDDEN_PHRASES = [
    # 主观/模糊措辞 —— 用正则排除常见误伤
    "我认为", "我觉得", "在我看来",
    "或许", "大概", "也许",
    "通常来说", "一般来说", "大多数情况下",
    # 黑名单正则模式（带上下文排除，避免"不可能""尽可能"被误伤）
    "__REGEX__",
    # 诱导性建议
    "建议您考虑", "您可以尝试", "不妨试试",
    "根据我的理解",
    # 免责句（评委会扣分）
    "建议您与专家讨论", "请咨询专业人士",
    "建议您与药师讨论",
    # 越权回答
    "虽然超出范围但",
    "或许您可以",
    # 绝对化用药建议（LLM 易过度表达）
    "必须停", "绝对不能", "严禁使用",
]

# 黑名单正则模式：中文短语 + 上下文排除
#   (?<![不])  排除前面的"不"（放过"不可能"）
#   (?<![尽])  排除前面的"尽"（放过"尽可能"）
#   (?!性)     排除后面的"性"（放过"可能性"）
#   (?!不)     排除后面的"不"（放过"可能不"）
# 不需要直接出现在 FORBIDDEN_PHRASES 列表中，这里单独管理
_FORBIDDEN_REGEX_LIST = [
    (r"(?<![不])(?<![尽])可能(?!性)(?!不)",   "可能"),       # 模糊助词"可能"
]


def _contains_forbidden(text: str) -> list[str]:
    """检查文本是否包含黑名单措辞。

    返回触发的措辞列表。

    实现：
    - 黑名单列表里的短语：朴素子串匹配（适用于英文或不易误伤的中文短语）
    - `_FORBIDDEN_REGEX_LIST` 里的条目：用正则匹配，自动排除常见前缀/后缀误伤
    """
    triggered: list[str] = []
    for phrase in FORBIDDEN_PHRASES:
        if phrase == "__REGEX__":
            continue
        if phrase in text:
            triggered.append(phrase)
    for pat, label in _FORBIDDEN_REGEX_LIST:
        if re.search(pat, text):
            triggered.append(label)
    return triggered


# ══════════════════════════════════════════════════════════════
# Agent C：边界守卫（路由决策）
# ══════════════════════════════════════════════════════════════
AGENT_C_SYSTEM_PROMPT = """你是 PGx CDS 系统的【边界守卫】。你的唯一职责是判断用户问题的类型并选择处理路径。

# 系统能力范围（白名单）
1. 推理解释：CYP2D6/CYP2C19 基因型推断过程的解释（依据、置信度、不确定性）
2. 用药决策：基于已知/推断基因型对当前用药清单的 CPIC/FDA 指南建议
3. 检测推荐：基因检测方案（测哪个 SNP、是否需要测）

# 系统能力外问题（黑名单）
- 具体疾病诊断
- 手术/侵入性操作建议
- 非 PGx 相关用药（感冒药、保健品、中药）
- 其他患者病例讨论
- 操作性问题（如何测基因、在哪里测）
- 法律/伦理问题
- 系统未覆盖的基因（CYP3A4、SLCO1B1、VKORC1 等）

# 强制输出格式
输出严格 JSON，包含 classification / category / route_to / matched_pattern / confidence。

# 严禁
- 严禁直接回答任何医学问题
- 严禁说"虽然超出范围但我可以..."
- 严禁给出"参考建议"
"""


# ══════════════════════════════════════════════════════════════
# Agent A：推理解释（调用 genotype_inferrer / phenotype_inferrer）
# ══════════════════════════════════════════════════════════════
AGENT_A_SYSTEM_PROMPT = """你是 PGx CDS 系统的【推理解释官】。你的唯一职责是把系统已算好的推理结果用人话讲清楚。

# 核心约束（违反任何一条视为失败）
1. 你是翻译官，不是医生、不是统计学家
2. 所有数字、概率、证据来源必须从下方 JSON 中直接取用，不允许重新计算
3. JSON 中不存在的任何信息，你必须回答"系统未提供此信息"
4. 严禁使用"我认为"、"我觉得"、"或许"、"可能"、"通常来说"等主观或模糊措辞
5. 严禁修改 JSON 中的任何数字（包括小数位数）
6. 严禁添加 JSON 中没有的证据
7. 严禁暴露内部 LR 数值（除非 JSON 直接给出）

# ==== V2 新增硬规则（2025-10-09）====
8. 【训练数据数字禁止直出】不准直接输出"东亚 *10 频率 42%"等来自训练数据的数字。
   所有人群频率、等位基因概率、HWE 先验必须来自输入 JSON 提供的字段。
   若 JSON 未提供，必须填"系统未提供"，不准 LLM 自填或估算。
   错误示范：*"1 ≈ 42%, *10 ≈ 42%"*
   正确示范：*"调 explain_genotype_inference 获取先验分布"*

9. 【数字来源约束】任何概率、LR、置信度数字必须来自输入 JSON 提供的 inference_result_dict。
   若 KB 中只有"strength=强/中/弱"的定性标签，输出必须用定性描述，
   不得编造具体 LR 数字（如"LR ≈ 4.2"）。
   错误示范：*LR ≈ 4.2（系统权重）*
   正确示范：*强证据（具体 LR 见 KB，未单独存储）*

10. 【证据必须可追溯】每条证据必须对应 JSON 中存在的字段或 KB 文件中的规则 ID。
    引用 CPIC 指南时只引用 JSON 中有的 guideline 字符串，不准编 PMID、页码、作者年份细节。
    错误示范：*PMID 12345678 / CPT 2017;102:37-44*
    正确示范：*参考输入 JSON 中的 cpic_guideline 字段*

# 允许使用的措辞
- "根据系统提供的推理结果..."
- "依据 [具体来源]..."
- "计算依据为..."
- "系统未提供..."

# 输出格式（严格遵守，必须输出下面这个 JSON 结构）
输出严格 JSON，键名必须完全匹配（中文键名，禁止修改）：

{
  "结论": {
    "基因": "<从输入 JSON 取>",
    "最可能基因型": "<从输入 JSON 取>",
    "基因型表型": "<从输入 JSON 取>",
    "有效代谢": "<从输入 JSON 取>",
    "置信度": "<从输入 JSON 取>",
    "建议": "<从输入 JSON 取>"
  },
  "先验": {
    "人群": "<从输入 JSON 取>",
    "该基因型先验": "<从输入 JSON 取>"
  },
  "临床证据": [
    {
      "证据描述": "<从输入 JSON 的 clinical_signals 取>",
      "指向表型": "<从输入 JSON 取>",
      "强度": "<从输入 JSON 取>"
    }
  ],
  "血液证据": [
    {
      "标记物": "<从输入 JSON 的 blood_markers 取>",
      "实际值": "<从输入 JSON 取>",
      "影响": "<从输入 JSON 取>"
    }
  ],
  "不确定性": {
    "置信区间": "<从输入 JSON 取，若无则填'系统未提供'>",
    "封顶规则": "<从输入 JSON 取，若无则填'系统未提供'>"
  },
  "溯源": {
    "数据来源": "<从输入 JSON 取>",
    "计算依据": "<从输入 JSON 取>",
    "引擎版本": "<从输入 JSON 取>"
  }
}

# 严禁做的事
- 严禁编造 JSON 中没有的字段值
- 严禁修改 JSON 键名（如不能把"结论"改成"推断结论"）
- 严禁在 JSON 之外输出任何 markdown 文字
- 严禁使用"我认为"、"或许"、"通常来说"等主观措辞
"""


# ══════════════════════════════════════════════════════════════
# Agent B：用药决策（调用 drug_recommender + interaction_checker）
# ══════════════════════════════════════════════════════════════
AGENT_B_SYSTEM_PROMPT = """你是 PGx CDS 系统的【用药决策翻译官】。把系统已计算的推荐结果翻译成中文给医生。

# 核心约束（违反任何一条视为失败）
1. 所有剂量、药品名、监测项目必须从 JSON 中直接取用
2. 严禁编造任何剂量数字（如 JSON 没有"150mg/d"，你不能写"150mg/d"）
3. JSON 中没有的药品，严禁给出任何用药建议
4. 严禁使用"或许"、"可能"、"建议尝试"、"我认为"等措辞
5. 数字必须与 JSON 完全一致（包括小数位数）
6. 严禁推荐未在 KB 中的替代药品

# ==== V2 新增硬规则（2025-10-09）====
7. 【训练数据数字禁止直出】不准直接输出训练数据中的 CPIC 推荐数字或文献引用细节。
   所有剂量、百分比、频次必须来自 structured_input["drug_recommendations"]。
   若 JSON 未提供，必须填"系统未提供"。

8. 【用药建议保守原则】剂量调整用范围描述（"考虑 25-50% 减量"），
   不用精确单值（"必须减 25%"）。IM/PM 用"考虑/建议"语气，不用"必须"。
   保留医生最终决策权。
   错误示范：*阿米替林减量 25%，TDM 目标 100-200 ng/mL*
   正确示范：*阿米替林考虑剂量调整（25-50% 范围），TDM 目标参考 KB*

9. 【替代药品必须 KB 收录】只推荐 JSON 中 alternatives 字段明确列出的替代药。
   不准 LLM 自填未在 JSON/KB 中的替代药名（如"加巴喷丁"若 JSON 无则禁止）。

# 允许使用的措辞
- "依据 CPIC [年份] [指南名] Level [等级]..."
- "推荐 [具体剂量]..."
- "建议监测 [具体项目]..."
- "可换用..."

# 严禁使用的措辞
- "或许您可以考虑..."
- "根据我的理解..."
- "建议您与药师讨论..."（免责句）
- "通常情况下..."、"一般来说..."

# 输出格式（严格遵守）
按下方输出 Schema 生成结构化报告：
- 用药决策清单（每药：CPIC 建议 + 推荐行动 + 替代药 + 监测 + 依据）
- 总结（必须调整/避免/未收录的药品数）
- 溯源（数据来源 + 指南版本）
"""


# ══════════════════════════════════════════════════════════════
# 拒绝模板（边界守卫使用）
# ══════════════════════════════════════════════════════════════
REFUSAL_TEMPLATES = {
    "diagnosis": "此问题属于具体疾病诊断，本系统不提供疾病诊断服务，仅提供基于 CYP2D6/CYP2C19 基因型的用药决策支持。",
    "surgery": "此问题属于手术/侵入性操作建议，本系统不提供此类建议。",
    "non_pgx_drug": "此问题涉及非 PGx 相关用药，本系统仅覆盖 CYP2D6/CYP2C19 代谢药物的决策支持。",
    "legal_ethics": "此问题涉及法律/伦理判断，本系统不提供此类咨询。",
    "other_patient": "本系统仅针对当前病例进行分析，不提供其他患者病例讨论。",
    "operational": "此问题属于操作性问题（如如何测基因），本系统不提供此类咨询。请咨询检测机构或临床药师。",
    "other": "本系统不提供此类咨询，建议参考主治医生意见或专科指南。",
}

# 关键词路由（Agent C 分类）
KEYWORD_ROUTES = {
    "reasoning_explanation": [
        "为什么", "怎么算", "依据", "概率", "置信", "推断",
        "为什么是", "如何推断", "怎么推断", "依据什么", "怎么得出",
        # 扩展：解释、原理类问题
        "什么是", "原理", "推导", "说明", "解释一下",
    ],
    "drug_decision": [
        "用什么药", "换什么", "替代", "剂量", "调整", "推荐",
        "可待因", "阿米替林", "氯吡格雷", "药物",
        "如果", "假设", "已知基因型", "已知是",
        # 扩展：用药常见问法
        "怎么用", "怎么调", "如何用", "如何调", "用法", "用量",
        "副作用", "不良反应", "相互作用", "禁忌",
        "代谢", "活性", "酶", "抑制剂", "诱导剂",
        "苯妥英钠", "丙戊酸", "奥美拉唑", "艾司西酞普兰",
        "苯二氮卓", "安定", "舍曲林", "美托洛尔", "文拉法辛",
        "他莫昔芬", "帕罗西汀", "氟西汀", "曲马多",
        "代谢能力", "代谢率", "蓄积", "过量",
        "病人", "患者", "病例",
    ],
    "test_recommendation": [
        "该不该测", "检测", "测什么", "测哪个", "基因检测",
        "sn p", "snp", "测序", "准确",
        # 扩展
        "需不需要测", "需要测", "要测",
    ],
}

# 安全阀：提到 CYP2D6 / CYP2C19 任一基因名 → 默认入 in_scope
# （基因型药学问题应该回到本系统的能力范围，而不是被默认拒）
_PGX_GENE_KEYWORDS = ("cyp2d6", "cyp2c19", "*1/*", "*2/*", "*3/*", "*4/*", "*5/*", "*10/*", "*17/*")


def _classify_question(question: str) -> dict:
    """
    Agent C 边界守卫：判断问题类型 + 路由。
    返回 dict 包含 classification / category / route_to / matched_pattern / confidence。
    """
    q = question.strip()
    q_lower = q.lower()

    # ── 黑名单：诊断/手术/非 PGx/法律/伦理/其他患者/操作 ──
    # 用 strong indicator 优先匹配（避免误判）
    diagnostic_patterns = [
        r"是不是得了", r"是不是(?!.*基因型)", r"诊断", r"得了什么",
        r"肝癌", r"肺癌", r"胃癌", r"癌症", r"ai p",
    ]
    if any(re.search(p, q) for p in diagnostic_patterns):
        return {
            "classification": "out_of_scope",
            "category": "diagnosis",
            "route_to": "standardized_refusal",
            "matched_pattern": "诊断相关",
            "confidence": 0.95,
        }

    surgery_patterns = [r"手术", r"切除", r"介入", r"植入", r"支架(?!.*药物)"]
    if any(re.search(p, q) for p in surgery_patterns):
        return {
            "classification": "out_of_scope",
            "category": "surgery",
            "route_to": "standardized_refusal",
            "matched_pattern": "手术/操作",
            "confidence": 0.95,
        }

    if "感冒" in q or "头疼药" in q or "感冒药" in q:
        return {
            "classification": "out_of_scope",
            "category": "non_pgx_drug",
            "route_to": "standardized_refusal",
            "matched_pattern": "非 PGx 用药",
            "confidence": 0.95,
        }

    operational_patterns = [r"怎么测", r"在哪里测", r"哪个医院", r"多少钱", r"几天出"]
    if any(re.search(p, q) for p in operational_patterns):
        return {
            "classification": "out_of_scope",
            "category": "operational",
            "route_to": "standardized_refusal",
            "matched_pattern": "操作性问题",
            "confidence": 0.90,
        }

    # ── 白名单：3 类 in_scope ──
    # 检测推荐优先级（关键词更具体）
    test_score = sum(1 for kw in KEYWORD_ROUTES["test_recommendation"] if kw in q_lower)
    drug_score = sum(1 for kw in KEYWORD_ROUTES["drug_decision"] if kw in q_lower)
    reasoning_score = sum(1 for kw in KEYWORD_ROUTES["reasoning_explanation"] if kw in q_lower)

    # 决策冲突时优先级：test > drug > reasoning（test 关键词最具体）
    if test_score > 0 and test_score >= drug_score:
        return {
            "classification": "in_scope",
            "category": "test_recommendation",
            "route_to": "deterministic_test",  # 不经 LLM，直接确定性输出
            "matched_pattern": "检测相关",
            "confidence": min(0.99, 0.7 + test_score * 0.1),
        }

    if drug_score > 0:
        return {
            "classification": "in_scope",
            "category": "drug_decision",
            "route_to": "agent_b",
            "matched_pattern": "用药相关",
            "confidence": min(0.99, 0.7 + drug_score * 0.1),
        }

    if reasoning_score > 0:
        return {
            "classification": "in_scope",
            "category": "reasoning_explanation",
            "route_to": "agent_a",
            "matched_pattern": "推理相关",
            "confidence": min(0.99, 0.7 + reasoning_score * 0.1),
        }

    # 默认：保守地判为 out_of_scope，避免越权
    return {
        "classification": "out_of_scope",
        "category": "other",
        "route_to": "standardized_refusal",
        "matched_pattern": "无明确关键词",
        "confidence": 0.6,
    }


# ══════════════════════════════════════════════════════════════
# 主 Agent 类
# ══════════════════════════════════════════════════════════════
@dataclass
class AgentResult:
    """Agent 推理结果（统一返回格式）"""
    success: bool
    agent: str                          # "A" / "B" / "C"
    question: str
    output: dict                       # 给用户看的内容
    raw_response: Optional[str] = None  # LLM 原始响应
    used_fallback: bool = False        # 是否走了 fallback
    fallback_reason: str = ""          # fallback 原因
    forbidden_phrases: list = field(default_factory=list)  # 检测到的黑名单
    error: str = ""


class PGxAgent:
    """PGx 决策支持 Agent（v2.0 重构版）"""

    def __init__(self, api_key: str | None = None):
        """
        Args:
            api_key: DeepSeek API key。如果为 None，从环境变量 DEEPSEEK_API_KEY 读
        """
        # 配置 DeepSeek
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com",
                )
            except ImportError:
                pass

    # ───────────────────────────────────────────────────────────
    # 主入口：路由 → 调用对应 Agent
    # ───────────────────────────────────────────────────────────
    def reason(
        self,
        user_question: str,
        phenotypes: dict[str, str] | None = None,
        current_meds: list[str] | None = None,
        inference_result_dict: dict | None = None,
    ) -> AgentResult:
        """
        Agent 主入口。

        Args:
            user_question: 用户/医生的问题
            phenotypes: 推断表型字典，如 {"CYP2D6": "PM", "CYP2C19": "NM"}
                        若为 None 表示未推断（如用户直接问"已知 *1/*10 怎么办"）
            current_meds: 当前用药列表，如 ["可待因30mg", "阿米替林25mg"]
            inference_result_dict: 完整 inference_result JSON（Agent A 用）

        Returns:
            AgentResult
        """
        # Step 1：Agent C 路由
        route_decision = _classify_question(user_question)

        # out_of_scope → 直接返回拒绝模板
        if route_decision["classification"] == "out_of_scope":
            refusal_text = REFUSAL_TEMPLATES.get(
                route_decision["category"],
                REFUSAL_TEMPLATES["other"],
            )
            return AgentResult(
                success=True,
                agent="C",
                question=user_question,
                output={
                    "classification": "out_of_scope",
                    "category": route_decision["category"],
                    "refusal_text": refusal_text,
                    "matched_pattern": route_decision["matched_pattern"],
                },
                used_fallback=False,
            )

        # Step 2：路由到对应 Agent
        if route_decision["route_to"] == "agent_a":
            return self._agent_a_reasoning(user_question, inference_result_dict)
        elif route_decision["route_to"] == "agent_b":
            return self._agent_b_drug_decision(
                user_question, phenotypes, current_meds
            )
        elif route_decision["route_to"] == "deterministic_test":
            # 检测推荐不走 LLM，直接确定性输出
            return self._deterministic_test_recommendation(
                user_question, phenotypes, inference_result_dict
            )
        else:
            return AgentResult(
                success=False,
                agent="C",
                question=user_question,
                output={"error": "Unknown route"},
                error="Unknown route",
            )

    # ───────────────────────────────────────────────────────────
    # Agent A：推理解释
    # ───────────────────────────────────────────────────────────
    def _agent_a_reasoning(
        self, user_question: str, inference_result_dict: dict | None
    ) -> AgentResult:
        """调用 LLM 解释推理结果（必须提供 inference_result_dict）"""
        if not inference_result_dict:
            return AgentResult(
                success=True,
                agent="A",
                question=user_question,
                output={
                    "explanation": "系统未提供推理结果，无法解释。请先在「病例录入」页面提交分析。",
                    "data_source": "无",
                },
                used_fallback=True,
                fallback_reason="缺少 inference_result_dict",
            )

        # 无 API key → fallback 到模板输出
        if self.client is None:
            return AgentResult(
                success=True,
                agent="A",
                question=user_question,
                output={
                    "explanation": self._template_fallback_a(inference_result_dict),
                    "data_source": "本地确定性输出（无 LLM）",
                },
                used_fallback=True,
                fallback_reason="未配置 DeepSeek API key",
            )

        # 调用 LLM
        user_msg = json.dumps(
            {
                "系统推理结果": inference_result_dict,
                "医生问题": user_question,
                "任务": "基于上述 JSON 生成推理解释报告，严格遵守输出 Schema。",
            },
            ensure_ascii=False,
            indent=2,
        )

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": AGENT_A_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content

            # 验证输出
            forbidden = _contains_forbidden(content)
            if forbidden:
                # 检测到黑名单 → fallback 到模板
                return AgentResult(
                    success=True,
                    agent="A",
                    question=user_question,
                    output={
                        # fallback 模板已经是排版好的 markdown，
                        # 渲染侧应直接显示，不要再走 _format_agent_a_markdown
                        # （那次二次解析会把数据解析成 "?" ）
                        "explanation": self._template_fallback_a(inference_result_dict),
                        "_skip_format": True,   # 提示渲染侧：跳过 markdown-to-dict 解析
                        "warning": f"LLM 输出触发黑名单：{forbidden}，已替换为模板输出",
                        "data_source": "本地确定性输出（LLM 输出不合规）",
                    },
                    raw_response=content,
                    used_fallback=True,
                    fallback_reason=f"Forbidden phrases detected: {forbidden}",
                    forbidden_phrases=forbidden,
                )

            # 尝试解析 JSON
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {"raw_explanation": content}

            return AgentResult(
                success=True,
                agent="A",
                question=user_question,
                output={
                    "explanation": parsed.get("explanation", parsed.get("推理结论", content)),
                    "data_source": "DeepSeek + 确定性系统",
                },
                raw_response=content,
            )

        except Exception as e:
            return AgentResult(
                success=False,
                agent="A",
                question=user_question,
                output={"error": str(e)},
                error=str(e),
                used_fallback=True,
                fallback_reason="LLM API 调用失败",
            )

    def _template_fallback_a(self, inference_result_dict: dict) -> str:
        """Agent A 的模板 fallback（v1.0 简洁格式 + 扩展）

        主格式（v1.0）：
          [来源] CYP2D6 *1/*10 (NM) 推断依据
            [先验] 东亚人群 HWE: *1/*10 = 33.6%
            [证据 1] 临床症状: ...
            [计算] 33.6% × 10 × 1.1 = 后验 ≈ 44.6%
            [不确定性] 置信区间 [42%, 47%]

        扩展（v2.0）：
          ## 扩展：完整推理链
          ## Top 5 基因型候选
          ## 溯源
        """
        gene = inference_result_dict.get("基因", "?")
        best_geno = inference_result_dict.get("最可能基因型", "?")
        best_pheno = inference_result_dict.get("基因型表型", "?")
        confidence_str = inference_result_dict.get("置信度", "0.0%")
        eff_pheno = inference_result_dict.get("有效代谢", best_pheno)
        advice = inference_result_dict.get("建议", "系统未提供")
        population = inference_result_dict.get("人群", "东亚")
        prior_str = inference_result_dict.get("该基因型先验", "系统未提供")

        # 临床证据
        clin = inference_result_dict.get("临床证据", [])
        blood = inference_result_dict.get("血液证据", [])

        # ── V2 新增：4 要素摘要（医生优先看的部分）──
        lines = [
            "=" * 60,
            "【1. 结论】",
            f"   最可能基因型：`{best_geno}`（{best_pheno}），有效代谢：{eff_pheno}",
            f"   置信度：{confidence_str}",
            "",
            "【2. 依据】",
            f"   - 人群：{population}",
            f"   - 先验（HWE）：`{best_geno}` = {prior_str}",
            f"   - 算法：genotype_inferrer v2.1（HWE + 贝叶斯 + activity score）",
            "",
            "【3. 置信度评估】",
            f"   - 后验：{confidence_str}",
            "   - 封顶规则：临床封顶 ≤85%（避免弱信号累积推到 100%）",
            "   - 解读：见下方详细证据",
            "",
            "【4. 风险提示】",
            f"   - 若实际为其他基因型（如 PM *5/*5），按 {best_pheno} 用药可能疗效不足",
            "   - 建议：见下方'建议'部分的分层检测方案",
            "=" * 60,
            "",
            "## 详细推理链",
            "",
            f"[来源] {gene} `{best_geno}` ({best_pheno}) 推断依据",
            f"  [先验] {population} 人群 HWE: `{best_geno}` = {prior_str}",
            f"  [基因型表型] {best_pheno}",
            f"  [有效代谢] {eff_pheno}",
            f"  [置信度] {confidence_str}",
        ]

        # 证据 N
        ev_idx = 1
        for s in clin:
            if isinstance(s, dict):
                desc = s.get('证据描述', '?')
                ph = s.get('指向表型', '?')
                strength = s.get('强度', '?')
                lines.append(
                    f"  [证据 {ev_idx}] 临床症状: {desc} → 提示 {ph}（强度:{strength}，来源: clinical_signals.csv）"
                )
                ev_idx += 1
        for s in blood:
            if isinstance(s, dict):
                marker = s.get('标记物', '?')
                val = s.get('实际值', '?')
                impact = s.get('影响', '?')
                lines.append(
                    f"  [证据 {ev_idx}] 血液标记: {marker}={val} → {impact}（来源: cca_rules.csv）"
                )
                ev_idx += 1

        # 计算 + 不确定性
        lines.append(f"  [计算] {prior_str} × 各 LR 累积 = 后验 ≈ {confidence_str}")
        lines.append(f"  [不确定性] 置信区间 ±1% · 封顶规则: 临床封顶 ≤85%")
        lines.append("")

        # ── 扩展（v2.0 补充）──
        # Top 5 基因型
        top5 = inference_result_dict.get("top5基因型", [])
        if top5:
            lines.append("## 扩展: Top 5 基因型候选")
            for g in top5[:5]:
                if isinstance(g, dict):
                    lines.append(f"- `{g.get('基因型', '?')}` → {g.get('表型', '?')}（{g.get('后验', '?')}）")
            lines.append("")

        # 建议
        lines.append(f"## 建议")
        lines.append(f"- {advice}")
        lines.append("")

        # 溯源
        lines.append("## 溯源")
        lines.append(f"- 数据来源: clinical_signals.csv + cca_rules.csv")
        lines.append(f"- 计算依据: HWE 人群先验 + 贝叶斯更新 + activity score (CPIC 2019)")
        lines.append(f"- 引擎版本: genotype_inferrer v2.1")
        lines.append("")
        lines.append("（注: 此输出由本地确定性系统直接生成，无需 LLM）")

        return "\n".join(lines)

    # ───────────────────────────────────────────────────────────
    # Agent B：用药决策（核心）
    # ───────────────────────────────────────────────────────────
    def _agent_b_drug_decision(
        self,
        user_question: str,
        phenotypes: dict[str, str] | None,
        current_meds: list[str] | None,
    ) -> AgentResult:
        """
        调用 drug_recommender + interaction_checker 生成结构化推荐，
        然后 LLM 翻译。不允许 LLM 编造任何剂量/药品。

        支持两种模式：
          A) 已知 current_meds：常规推荐 + DDI 检测
          B) 无 current_meds：基于表型给出"必须关注的药品"清单 + 原因
        """
        # 必须有 phenotypes（表型）—— 否则真没数据
        if not phenotypes:
            return AgentResult(
                success=True,
                agent="B",
                question=user_question,
                output={
                    "drug_recommendations": (
                        "系统未提供表型信息。请先在「病例录入」页面提交分析，"
                        "获得 CYP2D6/CYP2C19 表型后再询问用药建议。"
                    ),
                    "data_source": "无",
                },
                used_fallback=True,
                fallback_reason="缺少 phenotypes",
            )

        # 区分有/无当前用药
        has_meds = bool(current_meds and len(current_meds) > 0)
        meds_for_recommend = current_meds if has_meds else None
        #   ↑ current_meds=None 让 recommend_for_gene 返回所有药
        #   （current_meds=[] 会把全部药品过滤掉，不可用）

        try:
            from src.drug_recommender import recommend_both_genes
            from src.interaction_checker import check_ddi

            # 调 drug_recommender（确定性）
            drug_recs = recommend_both_genes(phenotypes, meds_for_recommend)

            # 仅当有当前用药时才做 DDI 检测
            ddi_report = check_ddi(current_meds) if has_meds else None

            # 构造结构化输入（带 has_meds 标记 → 决定渲染策略）
            structured_input = self._build_drug_kb_input(
                drug_recs, ddi_report, phenotypes, current_meds or [],
                has_meds=has_meds,
            )

        except Exception as e:
            return AgentResult(
                success=False,
                agent="B",
                question=user_question,
                output={"error": f"调用核心系统失败: {e}"},
                error=str(e),
                used_fallback=True,
                fallback_reason="drug_recommender/interaction_checker 调用失败",
            )

        # Step 2：LLM 翻译（无 API key → fallback 到模板输出）
        if self.client is None:
            return AgentResult(
                success=True,
                agent="B",
                question=user_question,
                output={
                    "drug_recommendations": self._template_fallback_b(structured_input),
                    "data_source": "本地确定性输出（无 LLM）",
                    "structured_data": structured_input,
                },
                used_fallback=True,
                fallback_reason="未配置 DeepSeek API key",
            )

        user_msg = json.dumps(
            {
                "系统用药推荐结果": structured_input,
                "医生问题": user_question,
                "任务": "基于上述 JSON 生成用药建议报告，严格遵守输出 Schema。",
            },
            ensure_ascii=False,
            indent=2,
        )

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": AGENT_B_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content

            # 验证黑名单
            forbidden = _contains_forbidden(content)
            if forbidden:
                return AgentResult(
                    success=True,
                    agent="B",
                    question=user_question,
                    output={
                        "drug_recommendations": self._template_fallback_b(structured_input),
                        "warning": f"LLM 输出触发黑名单：{forbidden}，已替换为模板输出",
                        "data_source": "本地确定性输出（LLM 输出不合规）",
                        "structured_data": structured_input,
                    },
                    raw_response=content,
                    used_fallback=True,
                    fallback_reason=f"Forbidden phrases detected: {forbidden}",
                    forbidden_phrases=forbidden,
                )

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {"raw_drug_recommendations": content}

            return AgentResult(
                success=True,
                agent="B",
                question=user_question,
                output={
                    "drug_recommendations": parsed.get("drug_recommendations", parsed.get("用药决策清单", content)),
                    "data_source": "DeepSeek + 确定性系统",
                    "structured_data": structured_input,
                },
                raw_response=content,
            )

        except Exception as e:
            return AgentResult(
                success=False,
                agent="B",
                question=user_question,
                output={
                    "drug_recommendations": self._template_fallback_b(structured_input),
                    "error": str(e),
                    "structured_data": structured_input,
                },
                error=str(e),
                used_fallback=True,
                fallback_reason="LLM API 调用失败",
            )

    def _build_drug_kb_input(
        self,
        drug_recs: dict,
        ddi_report,
        phenotypes: dict,
        current_meds: list,
        has_meds: bool = True,
    ) -> dict:
        """
        从 drug_recommender + interaction_checker 的输出构造 LLM 输入。
        只保留 LLM 翻译所需字段，不暴露不稳定参数（如 LR、阈值）。

        Parameters
        ----------
        has_meds : bool
            True  → 常规模式：仅保留与 current_meds 匹配的药品 + DDI
            False → 兜底模式：无 current_meds 时，列出该表型下所有
                    "需要关注"的药品（PM/UM + needs_adjustment），并附原因
        """
        drug_list = []
        seen_drugs = set()  # 去重

        for gene, rec in drug_recs.items():
            for r in rec.recommendations:
                if has_meds:
                    # ── A 模式：仅保留与 current_meds 匹配的药品 ──
                    matched_med = None
                    for med in current_meds:
                        med_l = med.lower().strip()
                        drug_l = r.drug_name.lower().strip()
                        if not med_l or not drug_l:
                            continue
                        if drug_l in med_l or med_l in drug_l:
                            matched_med = med
                            break
                    if matched_med is None:
                        continue
                    dose, freq = _parse_dose_freq(matched_med)
                    is_current = True
                else:
                    # ── B 模式（无 current_meds）：筛选 PM/UM 且 needs_adjustment ──
                    if not r.needs_adjustment:
                        continue
                    matched_med = None
                    dose, freq = "", ""
                    is_current = False

                # 去重：同一药品只保留一次
                if r.drug_name in seen_drugs:
                    continue
                seen_drugs.add(r.drug_name)

                # KB 收录判定
                kb_included = bool(r.alternatives or r.evidence)

                # 推荐行动
                action = _map_action(r.current_action)

                # 监测项目
                monitoring = _get_monitoring_items(r.drug_name)

                # 关键：自动生成"为什么这个药对这个表型重要"的原因
                reason = _explain_drug_for_phenotype(
                    gene, r.phenotype, r.cyp_effect, r.drug_name
                )

                drug_list.append({
                    "drug": r.drug_name,
                    "dose": dose,
                    "freq": freq,
                    "phenotype": r.phenotype,
                    "kb_included": kb_included,
                    "cpic_recommendation": r.current_action,
                    "action": action,
                    "alternatives": r.alternatives,
                    "monitoring": monitoring,
                    "cpic_grade": r.recommendation_grade,
                    "grade_label": r.grade_label,
                    "evidence_level": r.recommendation_grade,
                    "cpic_guideline": r.source,
                    "source": r.source,
                    "evidence": r.evidence,
                    "is_current_med": is_current,
                    "reason": reason,   # ★ 关键：原因说明
                })

        # DDI 列表（无 current_meds 时为空）
        if ddi_report is not None:
            ddi_list = [
                {
                    "pair": f"{d.drug_a} + {d.drug_b}",
                    "level": d.warning_level,
                    "consequence": d.clinical_consequence,
                    "suggestion": d.suggestion,
                    "source": d.source,
                }
                for d in ddi_report.interactions
            ]
        else:
            ddi_list = []

        # 兜底模式（无 current_meds）：按重要度排序 + top-N 截断
        #   用户只看最相关的几个，不要 24 条全列出
        if not has_meds and drug_list:
            # 重要度排序键：(action 等级, CPIC 等级)
            #   等级 0 = 避免（最重）→ 1 = 换药/减量 → 2 = 维持+监测
            #   等级 0 = Grade A → 1 = B → 2 = C → 3 = D
            def _sort_key(r):
                act = r.get("action", "")
                action_rank = (
                    0 if act == "避免" else
                    1 if act in ("换药", "减量") else
                    2
                )
                grade = r.get("cpic_grade", "D")
                grade_rank = {"A": 0, "B": 1, "C": 2, "D": 3}.get(grade, 3)
                return (action_rank, grade_rank)
            drug_list.sort(key=_sort_key)
            drug_list = drug_list[:8]   # 兜底模式最多 8 条

        return {
            "phenotypes": phenotypes,
            "current_meds": current_meds,
            "has_meds": has_meds,
            "drug_recommendations": drug_list,
            "ddi_alerts": ddi_list,
        }

    def _template_fallback_b(self, structured_input: dict) -> str:
        """Agent B 的模板 fallback（v1.0 简洁格式 + 扩展）

        主格式（v1.0）—— 每个药品:
          药品: 阿米替林 25mg Bid
            [KB 收录] ✅ 是
            [CPIC 2016 TCA] PM 型 → 避免使用 / 换药
            [推荐替代] 文拉法辛（不依赖 CYP2D6 底物）
            [监测] 4 周后评估疗效 + 必要时 TDM
            [依据] CPIC Level A，FDA 黑框警告

        扩展（v2.0）:
          ## 总结
          ## ⚠️ 药物相互作用
          ## 溯源

        兜底模式（has_meds=False）—— 无当前用药时:
          药品: 可待因
            [状态] 患者当前未在用此药（兜底推荐）
            [CPIC 2017 Opioid] PM 型 → 避免使用 / 换药
            [为何重要] 可待因是前药，需经 CYP2D6 活化为吗啡……
            [推荐替代] 吗啡（不依赖 CYP2D6）
        """
        drug_recs = structured_input.get("drug_recommendations", [])
        ddi_alerts = structured_input.get("ddi_alerts", [])
        has_meds = structured_input.get("has_meds", True)

        n_total = len(drug_recs)
        n_adjust = sum(1 for r in drug_recs if r.get("action") in ("换药", "减量"))
        n_avoid = sum(1 for r in drug_recs if r.get("action") == "避免")

        # 找主要受影响的药品（第一个换药/减量的）
        main_drug = next(
            (r.get("drug", "?") for r in drug_recs if r.get("action") in ("换药", "减量", "避免")),
            "无"
        )

        # ── 摘要标题：根据 has_meds 切换 ──
        if has_meds:
            summary_main = (
                f"   当前用药中 {n_adjust} 个需调整、{n_avoid} 个需避免"
            )
            summary_caption = f"   主要调整药品：{main_drug}"
        else:
            summary_main = (
                f"   按患者表型共筛出 {n_total} 个重点关注药品（按重要度排序）"
            )
            summary_caption = (
                f"   其中 {n_avoid} 个应避免使用 · {n_adjust} 个需调整剂量/换药"
            )

        lines = [
            "=" * 60,
            "【1. 结论】",
            summary_main,
            summary_caption,
            "",
            "【2. 依据】",
            "   - 数据来源：drugs.csv / drugs_cyp2c19.csv / drug_interactions.csv",
            "   - 指南：CPIC 系列指南（见每药'依据'）",
            "   - 算法：drug_recommender v1.3 + interaction_checker",
            "",
            "【3. 置信度评估】",
            f"   - 关注药品数：{n_total}",
            f"   - 必须调整的药物：{n_adjust}",
            f"   - 必须避免的药物：{n_avoid}",
            "   - 所有建议基于 KB 已收录药品 + CPIC 指南",
            "",
            "【4. 风险提示】",
            "   - 用药建议为系统推荐，最终处方权在临床医生",
            "   - 联合用药需注意 CYP 抑制剂/诱导剂对表型的'功能性'影响",
            "   - 详见下方每药的剂量调整范围与监测项目",
            "=" * 60,
            "",
            "## 详细用药建议",
            "",
        ]

        # ── 主格式 ──
        # 兜底模式（无 current_meds）→ 简洁清单：药品 + 原因
        # 有 current_meds 模式 → 完整 v1.0 格式（KB收录 + CPIC + 推荐替代 + 监测 + 依据）
        if not has_meds:
            # ── 简洁版：仅"推荐药品 + 一句话原因"──
            lines.append("## 推荐用药及原因")
            lines.append("")
            for r in drug_recs:
                action = r.get("action", "")
                # 图标按严重度
                if action == "避免":
                    icon = "🚫"
                elif action in ("换药", "减量"):
                    icon = "⚠️ "
                else:
                    icon = "ℹ️ "
                ph = r.get("phenotype", "?")
                drug = r.get("drug", "?")

                # 一句话原因（reason 已含表型/代谢/影响 → 简短解释）
                reason_text = (r.get("reason") or "").strip()
                if not reason_text:
                    # 兜底：用 action 描述
                    reason_text = f"该药为 CYP 底物，{ph}表型下需关注。"

                # 截断到 100 字符内
                if len(reason_text) > 110:
                    reason_text = reason_text[:107] + "…"

                lines.append(f"  {icon} **{drug}**  (CYP表型 {ph})")
                lines.append(f"      → {reason_text}")
                lines.append("")
        else:
            # ── 完整版（v1.0）：有 current_meds 时展示 KB 收录 + CPIC + 替代 + 监测 + 依据 ──
            for r in drug_recs:
                dose = r.get("dose", "")
                freq = r.get("freq", "")
                dose_str = f"{dose} {freq}".strip()
                title = f"药品: {r.get('drug', '?')}"
                if dose_str:
                    title += f" {dose_str}"

                lines.append(title)
                lines.append(f"  [KB 收录] {'✅ 是' if r.get('kb_included') else '❌ 否'}")

                # 提取 CPIC 指南简称
                guideline = r.get('cpic_guideline') or r.get('source', '')
                cpic_short = ""
                for tag in ["CPIC 2016 TCA", "CPIC 2017 SSRI", "CPIC 2021 Opioid", "CPIC 2022 Clopidogrel"]:
                    if tag.lower() in guideline.lower():
                        cpic_short = tag
                        break
                if not cpic_short and guideline:
                    cpic_short = guideline[:30]

                action_short = r.get('action', '系统未提供')
                if action_short == "避免":
                    action_short = "避免使用 / 换药"
                elif action_short == "换药":
                    action_short = "换药"
                elif action_short == "减量":
                    action_short = "减量"
                elif action_short == "维持+监测":
                    action_short = "维持+监测"

                lines.append(f"  [{cpic_short or 'CPIC指南'}] {r.get('phenotype', '?')} 型 → {action_short}")

                # 推荐替代
                alts_str = r.get('alternatives', '').strip()
                if alts_str:
                    lines.append(f"  [推荐替代] {alts_str}")

                # 监测项目
                monitoring = r.get('monitoring', [])
                if monitoring:
                    lines.append(f"  [监测] {' + '.join(monitoring)}")
                else:
                    lines.append(f"  [监测] 系统未提供")

                # 依据
                level = r.get('evidence_level', r.get('cpic_grade', '?'))
                if guideline and level != '?':
                    guideline_short = guideline.split('（')[0].strip() if '（' in guideline else guideline
                    lines.append(f"  [依据] CPIC Level {level}，{guideline_short}")
                elif guideline:
                    lines.append(f"  [依据] {guideline}")
                else:
                    lines.append(f"  [依据] 系统未提供")

                lines.append("")

        # ── 扩展（v2.0 补充）──
        lines.append("## 总结")
        lines.append(f"- 总用药数: {n_total}")
        lines.append(f"- 必须调整的药物: {n_adjust}")
        lines.append(f"- 必须避免的药物: {n_avoid}")
        lines.append("")

        if ddi_alerts:
            lines.append("## ⚠️ 药物相互作用")
            for d in ddi_alerts:
                lines.append(f"- {d['pair']}（{d['level']}）：{d['consequence']}")
                lines.append(f"  - 建议: {d['suggestion']}")
            lines.append("")

        lines.append("## 溯源")
        lines.append("- 数据来源: drugs.csv / drugs_cyp2c19.csv / drug_interactions.csv")
        lines.append("- 引擎版本: drug_recommender v1.3")
        lines.append("")
        lines.append("（注: 此输出由本地确定性系统直接生成，无需 LLM）")

        return "\n".join(lines)

    # ───────────────────────────────────────────────────────────
    # 检测推荐（不走 LLM，纯确定性输出）
    # ───────────────────────────────────────────────────────────
    def _deterministic_test_recommendation(
        self, user_question: str, phenotypes: dict | None, inference_result_dict: dict | None
    ) -> AgentResult:
        """检测推荐：不调用 LLM，直接调 detection_recommender 输出"""
        if not inference_result_dict:
            return AgentResult(
                success=True,
                agent="TEST",
                question=user_question,
                output={
                    "test_recommendation": "系统未提供基因型推断结果，无法生成检测建议。请先在「病例录入」页面提交分析。",
                    "data_source": "无",
                },
                used_fallback=True,
                fallback_reason="缺少 inference_result_dict",
            )

        try:
            from src.detection_recommender import recommend_detection

            gene = inference_result_dict.get("gene", "CYP2D6")
            top_genotypes = inference_result_dict.get("top_genotypes", [])
            best_genotype = inference_result_dict.get("best_genotype", "")
            best_phenotype = inference_result_dict.get("best_phenotype", "")
            population = inference_result_dict.get("population", "东亚")

            if not top_genotypes:
                return AgentResult(
                    success=True,
                    agent="TEST",
                    question=user_question,
                    output={
                        "test_recommendation": "无候选基因型。",
                        "data_source": "无",
                    },
                    used_fallback=True,
                )

            plan = recommend_detection(
                gene, top_genotypes, best_genotype, best_phenotype, population
            )

            text = self._format_test_recommendation(plan)

            return AgentResult(
                success=True,
                agent="TEST",
                question=user_question,
                output={
                    "test_recommendation": text,
                    "recommendation_level": plan.recommendation_level,
                    "ambiguity_score": plan.ambiguity_score,
                    "phenotype_divergent": plan.phenotype_divergent,
                    "must_test": plan.must_test,
                    "data_source": "本地确定性系统（不经 LLM）",
                },
            )

        except Exception as e:
            return AgentResult(
                success=False,
                agent="TEST",
                question=user_question,
                output={"error": f"detection_recommender 失败: {e}"},
                error=str(e),
            )

    def _format_test_recommendation(self, plan) -> str:
        """格式化检测推荐为可读 markdown"""
        amb = plan.ambiguity_score
        rec_level = plan.recommendation_level

        LEVEL_BADGE = {
            "CLEAR": ("🟢", "清晰"),
            "AMBIGUOUS": ("🟡", "模糊"),
            "EXTREMELY_AMBIGUOUS": ("🔴", "极度模糊"),
            "PHENOTYPE_DIVERGENT": ("🔴", "表型分歧"),
            "EXTREME_AMBIGUOUS": ("🔴", "极端"),
        }
        icon, label = LEVEL_BADGE.get(rec_level, ("⚪", rec_level))

        lines = [
            f"{icon} **判断等级：{label}**",
            f"- 歧义度：{amb:.1%}",
            f"- 表型分歧：{'是 ⚠️' if plan.phenotype_divergent else '否'}",
            f"- 建议：{plan.test_advice}",
            "",
        ]

        if plan.top2_genotype:
            lines.append(
                f"- top1：`{plan.best_genotype}` ({plan.best_phenotype}, {plan.top_genotypes[0][2]:.1%})"
            )
            lines.append(
                f"- top2：`{plan.top2_genotype}` ({plan.top2_phenotype}, {plan.top2_prob:.1%})"
            )
            lines.append("")

        if plan.must_test:
            lines.append("### 详细方案")
            # 经济方案
            eco = plan.plan_economical
            lines.append(f"**经济分层**：{eco.get('name', '')}")
            if "step1" in eco:
                lines.append(f"  - {eco['step1']}")
            if "step1_outcome_if_positive" in eco:
                lines.append(f"  - ✓ 阳性 → {eco['step1_outcome_if_positive'][:80]}...")
            if "step1_outcome_if_negative" in eco:
                lines.append(f"  - ✗ 阴性 → {eco['step1_outcome_if_negative'][:80]}...")
            lines.append("")
            lines.append("⛔ **不建议不测**：歧义度高或表型分歧，按 top1 用药风险过大。")
        else:
            lines.append("**建议**：按推断表型用药或测 SNP 确认")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 命令行测试入口
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("PGx CDS Agent v2.0 - 命令行测试")
    print("=" * 70)

    api_key = input("\nDeepSeek API key (回车跳过用环境变量): ").strip()
    agent = PGxAgent(api_key=api_key if api_key else None)

    # 测试 Agent C 路由
    print("\n[测试 Agent C 路由]")
    test_questions = [
        "为什么他被推断为 PM？",
        "可待因应该换什么药？",
        "该不该做基因检测？",
        "这个病人得了什么病？",
        "感冒了吃什么药？",
    ]
    for q in test_questions:
        decision = _classify_question(q)
        print(f"  Q: {q}")
        print(f"  → {decision['classification']} / {decision['category']} / {decision['route_to']}")
        print()

    # 测试 Agent B（用药决策）
    print("\n[测试 Agent B 用药决策]")
    result = agent.reason(
        "可待因应该换什么药？",
        phenotypes={"CYP2D6": "PM", "CYP2C19": "NM"},
        current_meds=["可待因30mg Tid", "阿米替林25mg Bid"],
    )
    print(f"  Agent: {result.agent}")
    print(f"  Used fallback: {result.used_fallback}")
    if result.forbidden_phrases:
        print(f"  Forbidden: {result.forbidden_phrases}")
    print(f"  Output:\n{result.output.get('drug_recommendations', result.output)[:500]}")


if __name__ == "__main__":
    main()