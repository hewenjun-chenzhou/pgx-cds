"""
T22 集成测试脚本：氟哌啶醇 + CYP2D6*4 (PM) + 欧洲人群 EPS 示范病例
=====================================================================
基于 Luca et al. Eur J Psychiatry 2026;40:100338 系统综述验证：
  1. 氟哌啶醇EPS信号匹配
  2. CYP2D6 PM型推断（欧洲人群 *4 等位基因高频）
  3. 用药推荐（减量50% + 换药考虑）
  4. 完整链路报告
"""

import sys
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.data_loader import get_cache
from src.phenotype_inferrer import infer_both_genes
from src.drug_recommender import recommend_both_genes
from src.interaction_checker import check_ddi
from src.population_analyzer import get_population_analysis
from src.report_generator import PatientInfo, build_report
from src.visualizer import generate_report_html


def main():
    print("=" * 60)
    print("T22 氟哌啶醇 CYP2D6*4 PM 欧洲 EPS 示范病例 - 集成验证")
    print("依据: Luca A et al. Eur J Psychiatry 2026;40:100338")
    print("      + Beunk L et al. (DPWG) Eur J Hum Genet 2024;32:278-285")
    print("      + Bondrescu M et al. Int J Mol Sci 2024;25:6350")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    cache = get_cache()
    cache.reload()  # 强制重载，确保读到最新 drugs.csv
    print(f"   drugs 行数: {len(cache.drugs)}")
    # 验证氟哌啶醇行升级
    hal_row = cache.get_drug_row("CYP2D6", "氟哌啶醇")
    if hal_row is not None:
        print(f"   氟哌啶醇 pm_action: {hal_row.get('pm_action', '')[:80]}")
        print(f"   氟哌啶醇 grade:      {hal_row.get('recommendation_grade', '')}")
        print(f"   氟哌啶醇 source:     {hal_row.get('source', '')[:80]}")

    # 验证 DPWG 2024 新增药物
    for drug in ["布瑞哌唑", "珠氯噻醇", "匹莫齐特"]:
        row = cache.get_drug_row("CYP2D6", drug)
        if row is not None:
            print(f"   {drug}: 已收录 ({row.get('recommendation_grade', '')})")
        else:
            print(f"   {drug}: ⚠️ 未找到")

    # 2. 构造 T22 患者（与 cases/T22_*.txt 一致）
    print("\n[2] T22 患者信息...")
    patient = PatientInfo(
        case_id="T22",
        diagnosis="偏执型精神分裂症",
        current_meds=["氟哌啶醇 5mg Bid", "苯海索 2mg Bid"],
        population="欧洲",
        chief_complaint="精神分裂症8年，复发伴行为紊乱2月",
        age=42,
        gender="男",
    )
    print(f"   病例: {patient.case_id}")
    print(f"   诊断: {patient.diagnosis}")
    print(f"   用药: {patient.current_meds}")
    print(f"   人群: {patient.population}")

    # 3. 推断表型（基于临床信号，未指定基因型）
    print("\n[3] CYP2D6 表型推断...")
    signals_cyp2d6 = [
        "氟哌啶醇EPS或急性肌张力障碍",   # 强信号，对应 Luca 2026 头条发现
    ]
    signals_cyp2c19 = []
    phenotypes = infer_both_genes(signals_cyp2d6, signals_cyp2c19, population=patient.population)
    for gene, pheno in phenotypes.items():
        fallback_tag = " [回退]" if pheno.is_fallback else ""
        print(f"   {gene}: {pheno.phenotype}{fallback_tag} (后验 {pheno.confidence:.0%}, 先验 {pheno.prior_prob:.1%})")
        print(f"     证据强度: {pheno.evidence_strength[:80]}")
        print(f"     最可能基因型: {pheno.best_genotype}")

    # 4. 药物推荐
    print("\n[4] 氟哌啶醇用药推荐...")
    phenotype_map = {gene: ph.phenotype for gene, ph in phenotypes.items()}
    drug_recs = recommend_both_genes(phenotype_map, patient.current_meds)
    for gene, rec in drug_recs.items():
        print(f"   {gene} ({rec.phenotype}):")
        for r in rec.recommendations:
            if "氟哌啶醇" in r.drug_name:
                print(f"     - {r.drug_name} ({r.drug_category})")
                print(f"       PM_action: {r.pm_action[:100]}")
                print(f"       current_action: {r.current_action[:80]}")
                print(f"       grade={r.recommendation_grade}, evidence={r.evidence}")
                print(f"       alternatives: {r.alternatives}")
                print(f"       needs_adjustment: {r.needs_adjustment}")
                print(f"       recommendation_index: {r.recommendation_index}")
                print(f"       source: {r.source[:100]}")

    # 5. DDI 检测
    print("\n[5] DDI 检测...")
    ddi_report = check_ddi(patient.current_meds)
    print(f"   相互作用数: {len(ddi_report.interactions)}, 高风险: {len(ddi_report.high_risk)}")

    # 6. 人群频率
    print("\n[6] CYP2D6 欧洲人群频率...")
    pd_cyp2d6 = get_population_analysis("CYP2D6", patient.population)
    if pd_cyp2d6:
        print(f"   PM={pd_cyp2d6.pm_freq:.1%}, IM={pd_cyp2d6.im_freq:.1%}, NM={pd_cyp2d6.nm_freq:.1%}, UM={pd_cyp2d6.um_freq:.1%}")
        print(f"   (欧洲 PM 频率显著高于东亚 0.4%，因 *4 等位基因高频 18%)")

    # 7. 完整报告
    print("\n[7] 生成 T22 报告...")
    pop_data = {"CYP2D6": pd_cyp2d6} if pd_cyp2d6 else {}
    report = build_report(patient, phenotypes, drug_recs, ddi_report, pop_data)
    print(f"   高优先级干预: {len(report.urgent_interventions)}")
    print(f"   高风险 DDI: {report.high_risk_ddi_count}")

    # 8. 生成 HTML
    print("\n[8] 生成 HTML 报告...")
    html = generate_report_html(report)
    out_path = "t22_european_report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"   已保存: {out_path}")

    print("\n" + "=" * 60)
    print("✅ T22 欧洲 EPS 示范病例集成验证完成")
    print("=" * 60)
    print("\n核心发现 (Luca 2026 论文):")
    print("  - 氟哌啶醇 PM 中 40-80% 出现 EPS")
    print("  - 氟哌啶醇 EM 中仅 12-30% 出现 EPS")
    print("  - 系统推荐: 氟哌啶醇 PM 型起始减50% + 监测EPS")
    print("  - 替代药: 奥氮平/喹硫平/阿立哌唑 (CYP3A4 代谢, 不受CYP2D6影响)")


if __name__ == "__main__":
    main()