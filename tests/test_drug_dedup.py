"""用药建议重复 bug 回归测试

背景：多塞平、丙米嗪、阿米替林等 9 个药同时是 CYP2D6 + CYP2C19 的底物，
原始代码遍历两个基因后会重复展示同一药品（甚至出现"避免 vs 安全"的矛盾归类）。

本测试验证 _merge_drug_items_by_name + 三档归类逻辑的修复正确性。

用法：python tests/test_drug_dedup.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pages._agent_section import _merge_drug_items_by_name  # noqa: E402
from src.drug_recommender import recommend_both_genes  # noqa: E402


def test_unit_all_safe_dedup():
    """NM/NM：双酶底物合并展示，无重复"""
    items = [
        {"drug": "多塞平", "gene": "CYP2D6", "phenotype": "NM", "grade": "C", "_severity": "safe"},
        {"drug": "多塞平", "gene": "CYP2C19", "phenotype": "NM", "grade": "C", "_severity": "safe"},
        {"drug": "奥美拉唑", "gene": "CYP2C19", "phenotype": "NM", "grade": "C", "_severity": "safe"},
    ]
    merged = _merge_drug_items_by_name(items)
    assert len(merged) == 2
    assert merged[0]["drug"] == "多塞平"
    assert "/" in merged[0]["gene"]   # 双基因合并
    assert merged[0]["phenotype"] == "NM"
    assert merged[1]["drug"] == "奥美拉唑"
    print("✓ test_unit_all_safe_dedup")


def test_unit_severity_pm_wins():
    """CYP2D6 PM 严重侧胜出"""
    items = [
        {"drug": "多塞平", "gene": "CYP2D6", "phenotype": "PM", "grade": "C",
         "_severity": "caution", "current_action": "避免或减量50%并TDM"},
        {"drug": "多塞平", "gene": "CYP2C19", "phenotype": "NM", "grade": "C",
         "_severity": "safe", "current_action": "建议常规剂量"},
    ]
    merged = _merge_drug_items_by_name(items)
    assert len(merged) == 1
    assert merged[0]["_severity"] == "caution"   # PM 严重侧胜出
    assert "减量" in merged[0]["current_action"]  # 保留 PM 侧的临床建议
    assert "PM" in merged[0]["phenotype"]
    print("✓ test_unit_severity_pm_wins")


def test_unit_severity_c19_pm_wins():
    """反向：CYP2C19 PM 时不能因 CYP2D6 NM 降级"""
    items = [
        {"drug": "多塞平", "gene": "CYP2D6", "phenotype": "NM", "grade": "C", "_severity": "safe"},
        {"drug": "多塞平", "gene": "CYP2C19", "phenotype": "PM", "grade": "C", "_severity": "caution"},
    ]
    merged = _merge_drug_items_by_name(items)
    assert merged[0]["_severity"] == "caution"
    print("✓ test_unit_severity_c19_pm_wins")


def test_e2e_no_duplicates():
    """端到端：所有 4 种典型表型组合都不应出现重复药品"""
    cases = [
        ("NM+NM", {"CYP2D6": "NM", "CYP2C19": "NM"}),
        ("PM+NM", {"CYP2D6": "PM", "CYP2C19": "NM"}),
        ("NM+PM", {"CYP2D6": "NM", "CYP2C19": "PM"}),
        ("PM+PM", {"CYP2D6": "PM", "CYP2C19": "PM"}),
    ]
    dual_substrate = {"多塞平", "丙米嗪", "阿米替林", "帕罗西汀", "文拉法辛",
                      "氟伏沙明", "氟西汀", "氯米帕明", "奥氮平"}

    for label, phenotypes in cases:
        recs = recommend_both_genes(phenotypes, current_meds=None)

        raw_items = []
        for gene, rec in recs.items():
            for r in rec.recommendations:
                ph = phenotypes[gene]
                grade = (r.recommendation_grade or "").upper()
                needs_adj = getattr(r, "needs_adjustment", False)
                if ph in ("PM", "UM") and grade in ("A", "B") and needs_adj:
                    sev = "avoid"
                elif (ph in ("PM", "UM") and grade in ("C", "D")) or (ph == "IM" and grade in ("A", "B")):
                    sev = "caution"
                else:
                    sev = "safe"
                raw_items.append({
                    "drug": r.drug_name, "gene": gene, "phenotype": ph,
                    "grade": grade, "_severity": sev,
                })

        merged = _merge_drug_items_by_name(raw_items)
        names = [x["drug"] for x in merged]
        dups = [n for n in set(names) if names.count(n) > 1]
        assert not dups, f"[{label}] 重复药品: {dups}"

        # 关键临床验证：多塞平在 PM 场景下应被分到 caution 或 avoid
        for x in merged:
            if x["drug"] == "多塞平" and "PM" in [phenotypes["CYP2D6"], phenotypes["CYP2C19"]]:
                assert x["_severity"] in ("caution", "avoid"), \
                    f"[{label}] 多塞平 PM 时严重度应为 caution/avoid，实际 {x['_severity']}"
        print(f"✓ test_e2e_no_duplicates [{label}]")


if __name__ == "__main__":
    test_unit_all_safe_dedup()
    test_unit_severity_pm_wins()
    test_unit_severity_c19_pm_wins()
    test_e2e_no_duplicates()
    print()
    print("✅ 所有回归测试通过")