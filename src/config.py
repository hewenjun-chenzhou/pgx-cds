"""
config.py — 全局常量定义
所有字符串常量、列名映射、等级颜色等集中于此，避免散落在各模块中。
"""

import os
from pathlib import Path

# ── 路径常量（基于 __file__ 自动定位，跨机器/账号可移植）──
# src/config.py 的 parent.parent = 项目根目录
_BASE_FROM_FILE = Path(__file__).resolve().parent.parent

# 兼容策略：相对路径 > 旧硬编码 > 环境变量 > 默认 CherryStudio Files 目录
_LEGACY_BASE   = Path(r"C:\Users\Administrator\Desktop\ai人工智能竞赛\正式\pgx_cds")
_LEGACY_FALLBK = Path(r"C:\Users\Administrator\AppData\Roaming\CherryStudio\Data\Files")
_ENV_BASE      = Path(os.environ.get("PGX_BASE_DIR", "")) if os.environ.get("PGX_BASE_DIR") else None
_ENV_FALLBK    = Path(os.environ.get("PGX_FALLBACK_DATA", "")) if os.environ.get("PGX_FALLBACK_DATA") else None

def _first_existing(*candidates: Path) -> Path:
    """返回第一个存在的目录；都不存在则退回到第一个候选（用于后续诊断报错更清晰）"""
    for p in candidates:
        try:
            if p and p.exists():
                return p
        except (OSError, ValueError):
            continue
    return candidates[0]  # 兜底：返回相对路径定位

BASE_DIR = _first_existing(_BASE_FROM_FILE, _LEGACY_BASE, _ENV_BASE or _BASE_FROM_FILE)
DATA_DIR = BASE_DIR / "data"
_FALLBACK_DATA = _first_existing(
    _ENV_FALLBK,
    _LEGACY_FALLBK,
    Path.home() / "AppData" / "Roaming" / "CherryStudio" / "Data" / "Files",
)

# ── 表型 ────────────────────────────────────────────────────
PHENOTYPES = ("PM", "IM", "NM", "UM")
PHENOTYPE_LABELS = {
    "PM": "慢代谢型（Poor Metabolizer）",
    "IM": "中间代谢型（Intermediate）",
    "NM": "正常代谢型（Normal）",
    "UM": "超快代谢型（Ultra-Rapid）",
}

# ── 推荐等级 ────────────────────────────────────────────────
GRADE_LABELS = {
    "A": "强烈建议",
    "B": "建议",
    "C": "可选项",
    "D": "无需调整",
}
GRADE_COLORS = {
    "A": "#FF4B4B",
    "B": "#FFA500",
    "C": "#FFD700",
    "D": "#28A745",
}

# ── 警示级别 ────────────────────────────────────────────────
WARNING_LEVELS = ("高", "中", "低")
WARNING_COLORS = {
    "高": "#FF4B4B",
    "中": "#FFA500",
    "低": "#28A745",
}

# ── 信号强度 ────────────────────────────────────────────────
SIGNAL_STRENGTHS = ("强", "中", "弱")
SIGNAL_STRENGTH_COLORS = {
    "强": "#FF4B4B",
    "中": "#FFA500",
    "弱": "#90CAF9",
}

# ── 人群 ────────────────────────────────────────────────────
POPULATIONS = ("东亚", "欧洲", "非洲裔", "拉美裔")

# ── 基因 ────────────────────────────────────────────────────
GENES = ("CYP2D6", "CYP2C19")

# ── 药物作用类型 ────────────────────────────────────────────
CYP_EFFECTS = (
    "substrate",
    "prodrug",
    "substrate_and_inhibitor",
    "inhibitor",
    "non_substrate",
)

# ── CSV 文件名 → UUID 映射（优先用DATA_DIR，找不到则用_FALLBACK_DATA）─
CSV_FILES = {
    "clinical_signals":             "clinical_signals.csv",
    "clinical_signals_cyp2c19":     "clinical_signals_cyp2c19.csv",
    "drugs":                        "drugs.csv",
    "drugs_cyp2c19":                "drugs_cyp2c19.csv",
    "drug_interactions":            "drug_interactions.csv",
    "drugs_other_genes":            "drugs_other_genes.csv",
    "inhibitors_cyp2d6":            "inhibitors.csv",
    "inhibitors_cyp2c19":           "inhibitors_cyp2c19.csv",
    "phenotype_frequencies_cyp2d6": "population_frequencies.csv",
    "phenotype_frequencies_cyp2c19":"population_frequencies_cyp2c19.csv",
    "cyp_allele_freq":              "cyp_allele_freq.csv",
    "pgx_detection_panel":          "pgx_detection_panel.csv",
    "test_cases":                   "cases.csv",
}

# ── CSV 列名（中英对照）───────────────────────────────────────
COLUMN_LABELS = {
    "signal_name":            "信号名称",
    "target_phenotype":       "目标表型",
    "strength":               "强度",
    "context":                "情境",
    "interpretation":         "解释",
    "test_recommendation":    "检测建议",
    "citation":               "引用",
    "gene":                   "基因",
    "drug_name":              "药物名称",
    "drug_category":          "药物类别",
    "cyp2d6_effect":          "CYP2D6作用",
    "cyp2c19_effect":         "CYP2C19作用",
    "pm_action":              "PM行动",
    "um_action":              "UM行动",
    "recommendation_grade":   "推荐等级",
    "evidence":               "证据强度",
    "alternatives":           "替代药物",
    "source":                 "来源",
    "drug_a":                 "药物A",
    "drug_b":                 "药物B",
    "mechanism":              "机制",
    "warning_level":          "警示级别",
    "clinical_consequence":   "临床后果",
    "main_pathway":           "主要代谢途径",
    "therapeutic_class":      "治疗类别",
    "substrate_example":      "底物示例",
    "note":                   "备注",
    "category":               "类别",
    "gene_target":            "目标基因",
    "type":                   "类型",
    "population":             "人群",
    "pm_freq":                "PM频率",
    "im_freq":                "IM频率",
    "nm_freq":                "NM频率",
    "um_freq":                "UM频率",
    "case_id":                "病例ID",
    "diagnosis":              "诊断",
    "current_meds":           "当前用药",
    "past_signals":           "既往信号",
    "expected_output":        "预期输出",
    "notes":                  "备注",
}

# ── 证据等级 ────────────────────────────────────────────────
EVIDENCE_LEVELS = ("Strong", "Moderate", "Low", "None")

# ── 表型推断置信度徽章颜色 ────────────────────────────────────
CONFIDENCE_COLORS = {
    "强": "#FF4B4B",
    "中": "#FFA500",
    "弱": "#90CAF9",
    None: "#CCCCCC",
}
