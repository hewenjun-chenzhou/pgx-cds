"""
validate_data.py — Step 1 验证脚本
运行: python validate_data.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_loader import get_cache, DataLoadError
from src.config import CSV_FILES


def main():
    print("=" * 60)
    print("Step 1 验证: CSV加载 + 基础查询")
    print("=" * 60)

    try:
        cache = get_cache()
    except DataLoadError as e:
        print(f"[FAIL] 加载失败: {e}")
        return

    print(f"\n[OK] 共加载 {len(cache.list_all())} 个DataFrame:")
    for name, df in cache.list_all().items():
        print(f"  {name:40s} {len(df):3d} 行 × {len(df.columns):2d} 列")

    # ── 基础查询验证 ────────────────────────────────────────
    print("\n--- 基础查询验证 ---")

    # 1. 信号匹配（临床信号 → 表型）
    rows = cache.get_signal_rows("CYP2D6", "可待因镇痛完全无效")
    assert not rows.empty, "可待因镇痛无效信号未匹配"
    print(f"[OK] get_signal_rows CYP2D6/可待因镇痛完全无效 → {len(rows)} 条")
    print(f"     → 推断表型: {rows.iloc[0]['target_phenotype']}, 强度: {rows.iloc[0]['strength']}")

    rows2 = cache.get_signal_rows("CYP2C19", "QT延长")
    assert not rows2.empty, "CYP2C19 QT延长信号未匹配"
    print(f"[OK] get_signal_rows CYP2C19/QT延长 → {len(rows2)} 条")
    print(f"     → 推断表型: {rows2.iloc[0]['target_phenotype']}, 强度: {rows2.iloc[0]['strength']}")

    # 2. 药物查询（drug_name + drug_category 双列搜索）
    row = cache.get_drug_row("CYP2D6", "可待因")
    assert row is not None, "可待因未找到"
    print(f"[OK] get_drug_row CYP2D6/可待因 → pm_action: {row['pm_action']}")

    row = cache.get_drug_row("CYP2C19", "艾司西酞普兰")
    assert row is not None, "艾司西酞普兰未找到"
    print(f"[OK] get_drug_row CYP2C19/艾司西酞普兰 → pm_action: {row['pm_action']}")

    # 3. DDI查询（用具体药名在 drug_interactions 表中查）
    ddi = cache.get_ddi("氟西汀", "曲马多")
    assert not ddi.empty, "氟西汀+曲马多 DDI未找到"
    print(f"[OK] get_ddi 氟西汀+曲马多 → warning_level: {ddi.iloc[0]['warning_level']}")

    # 4. 人群频率
    freq = cache.get_population_freq("CYP2C19", "东亚")
    assert freq is not None, "东亚CYP2C19频率未找到"
    print(f"[OK] get_population_freq CYP2C19/东亚 → PM={freq['pm_freq']}, UM={freq['um_freq']}")

    freq2 = cache.get_population_freq("CYP2D6", "欧洲")
    assert freq2 is not None, "欧洲CYP2D6频率未找到"
    print(f"[OK] get_population_freq CYP2D6/欧洲 → PM={freq2['pm_freq']}, UM={freq2['um_freq']}")

    # 5. 替代药（drugs_other_genes.csv 存具体药名）
    alts = cache.get_alternative_drugs("阿片类镇痛药")
    assert not alts.empty, "阿片类替代药未找到"
    print(f"[OK] get_alternative_drugs 阿片类 → {list(alts['drug_name'])}")

    # 6. 抑制剂查询
    inh = cache.get_inhibitors("CYP2D6", "强")
    assert not inh.empty, "CYP2D6强抑制剂未找到"
    print(f"[OK] get_inhibitors CYP2D6/强 → {len(inh)} 条")

    # 7. 测试用例
    case = cache.get_test_case("PGx-001")
    assert case is not None, "PGx-001未找到"
    print(f"[OK] get_test_case PGx-001 → diagnosis: {case['diagnosis']}")
    print(f"     → current_meds: {case['current_meds']}")

    print("\n" + "=" * 60)
    print("[PASS] Step 1 所有验证通过，数据层已就绪")
    print("=" * 60)


if __name__ == "__main__":
    main()
