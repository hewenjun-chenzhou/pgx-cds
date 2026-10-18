r"""
pharmacology_kb_loader.py
========================
加载并查询层级药理知识库（pharmacology_kb.json）。

用途：
    - 查询某个血液标记物的CCA规则
    - 查找基因型-药物关联
    - 构建推理链

使用：
    from src.pharmacology_kb_loader import PharmacologyKB
    kb = PharmacologyKB()
    rules = kb.get_cca_rules_for_marker("ALT", gene="CYP2D6")
    chains = kb.get_inference_chains()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(r"C:\Users\Administrator\Desktop\ai人工智能竞赛\正式\pgx_cds")
KB_FILE = BASE_DIR / "data" / "pharmacology_kb.json"


class PharmacologyKB:
    """层级药理知识库加载器"""

    def __init__(self, kb_file: Path | None = None) -> None:
        self.kb_file = kb_file or KB_FILE
        if not self.kb_file.exists():
            raise FileNotFoundError(f"知识库不存在: {self.kb_file}")
        with open(self.kb_file, encoding="utf-8") as f:
            self.data = json.load(f)
        print(f"[PharmacologyKB] 已加载: {self.kb_file.name}, 版本 {self.data['metadata']['version']}")

    # ────────────────────────────────────────────────────────
    # 血液标记物 × CCA 规则查询
    # ────────────────────────────────────────────────────────
    def get_cca_rules_for_marker(self, marker: str, gene: str | None = None) -> list[dict]:
        """
        获取某血液标记物的所有CCA规则。

        Args:
            marker: 标记物名，如 ALT, CrCl, INR
            gene:   可选，限定基因 CYP2D6 / CYP2C19
        Returns:
            [{"condition": "...", "gene": "...", "effect": "...", "action": "...", "evidence": "..."}, ...]
        """
        results = []
        for dim_key, dim_val in self.data.get("blood_marker_rules", {}).items():
            for marker_name, marker_info in dim_val.get("markers", {}).items():
                if marker_name.upper() == marker.upper():
                    for rule in marker_info.get("cca_rules", []):
                        if gene is None or rule["gene"] == gene:
                            results.append({**rule, "marker": marker_name, "dimension": dim_val["dimension"]})
        return results

    def get_all_markets_with_cca(self) -> dict[str, list[str]]:
        """返回所有有CCA规则的标记物及其维度"""
        result = {}
        for dim_key, dim_val in self.data.get("blood_marker_rules", {}).items():
            dim_name = dim_val["dimension"]
            markers = list(dim_val.get("markers", {}).keys())
            if markers:
                result[dim_name] = markers
        return result

    # ────────────────────────────────────────────────────────
    # 基因层面查询
    # ────────────────────────────────────────────────────────
    def get_gene_info(self, gene: str) -> dict:
        """获取某基因的完整信息（底物、抑制剂、表型频率、功能影响因子）"""
        return self.data.get("genes", {}).get(gene, {})

    def get_substrates(self, gene: str) -> list[str]:
        return self.data.get("genes", {}).get(gene, {}).get("substrates", [])

    def get_inhibitors(self, gene: str) -> list[str]:
        return self.data.get("genes", {}).get(gene, {}).get("inhibitors", [])

    def get_phenotype_info(self, gene: str, phenotype: str) -> dict:
        return self.data.get("genes", {}).get(gene, {}).get("phenotypes", {}).get(phenotype, {})

    def get_function_impact(self, gene: str, dimension: str, marker: str, condition: str) -> dict | None:
        """
        查询特定血液标记物异常对某基因代谢功能的实际影响。

        Args:
            gene:      CYP2D6 / CYP2C19
            dimension: liver / renal / coagulation / blood / electrolyte / inflammation
            marker:    ALT / CrCl / INR / ...
            condition: 具体条件字符串
        Returns:
            {"metabolism_reduction": "...", "evidence": "..."} 或 None
        """
        dim_data = self.data.get("genes", {}).get(gene, {}).get("function_impact_factors", {}).get(dimension, {})
        marker_data = dim_data.get(marker, {})
        if not marker_data:
            return None
        # 精确匹配
        for key, val in marker_data.items():
            if condition and (key in condition or condition in key):
                return val
        return None

    # ────────────────────────────────────────────────────────
    # TDM 查询
    # ────────────────────────────────────────────────────────
    def get_tdm_target(self, drug: str) -> dict | None:
        """获取某药物的TDM治疗窗信息"""
        for drug_key, info in self.data.get("tdm_targets", {}).items():
            if drug.lower() in drug_key.lower() or drug_key.lower() in drug.lower():
                return {**info, "drug": drug_key}
        return None

    # ────────────────────────────────────────────────────────
    # 推理链查询
    # ────────────────────────────────────────────────────────
    def get_inference_chains(self) -> list[dict]:
        return self.data.get("inference_chains", {}).get("chains", [])

    def build_chain_for_marker(self, marker: str, gene: str | None = None) -> str:
        """
        基于血液标记物构建推理链描述文本（供LLM使用）。
        """
        cca_rules = self.get_cca_rules_for_marker(marker, gene)
        if not cca_rules:
            return f"标记物【{marker}】暂无CCA规则"

        lines = [f"血液标记物异常【{marker}】的推理链："]
        for rule in cca_rules:
            lines.append(
                f"  → {rule.get('dimension', '')}【{marker}{rule['condition']}】"
                f" → {rule['gene']}代谢{rule['effect']}"
                f" → 建议行动：{rule['action']}"
                f"（证据：{rule['evidence']}）"
            )
        return "\n".join(lines)

    # ────────────────────────────────────────────────────────
    # 批量推理：对多个化验结果一次性查询
    # ────────────────────────────────────────────────────────
    def query_lab_panel(self, lab_results: dict[str, float]) -> list[dict]:
        """
        输入多个化验结果，返回所有触发的CCA规则。

        Args:
            lab_results: {"ALT": 150, "CrCl": 25, "INR": 1.8, ...}
        Returns:
            [{"marker": "ALT", "condition": ">3xULN", "gene": "CYP2D6",
              "effect": "代谢↓50%", "action": "减半剂量并监测", "evidence": "..."}, ...]
        """
        triggered = []
        for marker, value in lab_results.items():
            rules = self.get_cca_rules_for_marker(marker)
            for rule in rules:
                condition = rule.get("condition", "")
                if self._condition_matches(condition, value, marker):
                    triggered.append({**rule, "actual_value": value})
        return triggered

    def _condition_matches(self, condition: str, value: float | None, marker: str) -> bool:
        """判断化验值是否触发某条件"""
        if value is None:
            return False
        try:
            # 硬编码阈值（与CSV中的trigger_condition保持一致）
            thresholds = {
                "ALT": 120,    # 3xULN
                "AST": 105,    # 3xULN
                "CrCl": 30,
                "eGFR": 30,
                "INR": 1.5,
                "Alb": 2.5,
                "T.Bili": 2.0,
                "K": 5.5,
                "CRP": 50,
                "NH3": 67.5,
                "Plt": 50000,
            }
            if marker not in thresholds:
                return False
            threshold = thresholds[marker]
            # 解析条件
            if ">3xULN" in condition or ">3x" in condition:
                return value > threshold
            if ">1.5" in condition:
                return value > 1.5
            if "<30" in condition or "<30mL" in condition:
                return value < 30
            if "<2.5" in condition or "<2.5g" in condition:
                return value < 2.5
            if ">2" in condition or ">2mg" in condition:
                return value > 2.0
            if ">5.5" in condition:
                return value > 5.5
            if ">50" in condition:
                return value > 50
            if "HD" in condition:
                return False  # HD是状态，不是数值
            return False
        except (ValueError, TypeError):
            return False


def main() -> None:
    kb = PharmacologyKB()

    print("\n=== 测试：查询ALT的CCA规则 ===")
    rules = kb.get_cca_rules_for_marker("ALT")
    for r in rules:
        print(f"  {r}")

    print("\n=== 测试：查询CrCl<30的CCA规则 ===")
    rules = kb.get_cca_rules_for_marker("CrCl", gene="CYP2D6")
    for r in rules:
        print(f"  {r}")

    print("\n=== 测试：批量化验结果查询 ===")
    lab_panel = {"ALT": 150, "CrCl": 25, "INR": 1.8, "K": 5.8}
    triggered = kb.query_lab_panel(lab_panel)
    for t in triggered:
        print(f"  【{t['marker']}={t['actual_value']}】触发 → {t['gene']} {t['effect']}")

    print("\n=== 测试：构建推理链 ===")
    print(kb.build_chain_for_marker("ALT", gene="CYP2D6"))

    print("\n=== 测试：TDM查询 ===")
    tdm = kb.get_tdm_target("amitriptyline")
    print(f"  {tdm}")

    print("\n=== 测试：推理链列表 ===")
    chains = kb.get_inference_chains()
    print(f"  共 {len(chains)} 条推理链:")
    for c in chains:
        print(f"  - {c['chain_id']}: {c['description']}")


if __name__ == "__main__":
    main()
