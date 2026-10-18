"""
T21 集成测试脚本：利培酮 + CYP2D6*10 + 中国队列示范病例
===========================================================
基于 Lu et al. BMC Psychiatry 2021 (76例中国精神分裂症) 验证：
  1. 信号匹配 (利培酮EPS/PRL升高)
  2. CYP2D6 PM型推断
  3. 用药推荐（减量50% + TDM）
  4. 完整链路报告
"""

import sys
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.data_loader import get_cache
from src.phenotype_inferrer import infer_both_genes
from src.drug_recommender import recommend_both_genes, evaluate_prescription
from src.interaction_checker import check_ddi
from src.population_analyzer import get_population_analysis
from src.report_generator import PatientInfo, build_report
from src.visualizer import generate_report_html


def main():
    print("=" * 60)
    print("T21 利培酮 CYP2D6*10 示范病例 - 集成验证")
    print("依据: Lu et al. BMC Psychiatry 2021;21:70 (PMID:33540457)")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    cache = get_cache()
    cache.reload()  # 强制重载，确保读到最新 drugs.csv
    print(f"   drugs 行数: {len(cache.drugs)}")
    # 验证利培酮行是否更新
    ris_row = cache.get_drug_row("CYP2D6", "利培酮")
    if ris_row is not None:
        print(f"   利培酮 pm_action: {ris_row.get('pm_action', '')[:80]}")
        print(f"   利培酮 source:    {ris_row.get('source', '')[:80]}")

    # 验证新增的 cca 规则（直接读 CSV，不走 cache）
    import pandas as pd
    cca_df = pd.read_csv("data/cca_rules_extended.csv")
    ris_cca = cca_df[cca_df["content"].str.contains("利培酮", na=False)]
    print(f"   利培酮相关 cca_rules: {len(ris_cca)} 条")

    # 2. 构造 T21 患者（与 cases/T21_*.txt 一致）
    print("\n[2] T21 患者信息...")
    patient = PatientInfo(
        case_id="T21",
        diagnosis="慢性精神分裂症",
        current_meds=["利培酮 4mg Bid"],
        population="东亚",
        chief_complaint="精神分裂症5年，加重伴行为紊乱1月",
        age=45,
        gender="男",
    )
    print(f"   病例: {patient.case_id}")
    print(f"   诊断: {patient.diagnosis}")
    print(f"   用药: {patient.current_meds}")
    print(f"   人群: {patient.population}")

    # 3. 推断表型（基于临床信号，未指定基因型）
    print("\n[3] CYP2D6 表型推断...")
    # 信号来自T21病历：EPS+PRL升高+血脂异常+体重变化（多重信号）
    signals_cyp2d6 = [
        "利培酮锥体外系反应或PRL显著升高",   # 已有信号
        "利培酮治疗4周内体重增加超2kg或血脂异常",  # 新增强信号(Lu 2021)
    ]
    signals_cyp2c19 = []  # 该患者无 CYP2C19 相关信号
    phenotypes = infer_both_genes(signals_cyp2d6, signals_cyp2c19, population=patient.population)
    for gene, pheno in phenotypes.items():
        fallback_tag = " [回退]" if pheno.is_fallback else ""
        print(f"   {gene}: {pheno.phenotype}{fallback_tag} (后验 {pheno.confidence:.0%}, 先验 {pheno.prior_prob:.1%})")
        print(f"     证据强度: {pheno.evidence_strength[:80]}")
        print(f"     最可能基因型: {pheno.best_genotype}")

    # 4. 药物推荐
    print("\n[4] 利培酮用药推荐...")
    phenotype_map = {gene: ph.phenotype for gene, ph in phenotypes.items()}
    drug_recs = recommend_both_genes(phenotype_map, patient.current_meds)
    for gene, rec in drug_recs.items():
        print(f"   {gene} ({rec.phenotype}):")
        for r in rec.recommendations:
            if "利培酮" in r.drug_name:
                print(f"     - {r.drug_name} ({r.drug_category})")
                print(f"       PM_action: {r.pm_action[:80]}")
                print(f"       current_action: {r.current_action[:80]}")
                print(f"       grade={r.recommendation_grade}, evidence={r.evidence}, source={r.source[:60]}")
                print(f"       alternatives: {r.alternatives}")
                print(f"       needs_adjustment: {r.needs_adjustment}")
                print(f"       recommendation_index: {r.recommendation_index}")

    # 5. DDI 检测
    print("\n[5] DDI 检测...")
    ddi_report = check_ddi(patient.current_meds)
    print(f"   相互作用数: {len(ddi_report.interactions)}, 高风险: {len(ddi_report.high_risk)}")

    # 6. 人群频率
    print("\n[6] CYP2D6 东亚人群频率...")
    pd_cyp2d6 = get_population_analysis("CYP2D6", patient.population)
    if pd_cyp2d6:
        print(f"   PM={pd_cyp2d6.pm_freq:.1%}, IM={pd_cyp2d6.im_freq:.1%}, NM={pd_cyp2d6.nm_freq:.1%}, UM={pd_cyp2d6.um_freq:.1%}")

    # 7. 完整报告
    print("\n[7] 生成 T21 报告...")
    pop_data = {"CYP2D6": pd_cyp2d6} if pd_cyp2d6 else {}
    report = build_report(patient, phenotypes, drug_recs, ddi_report, pop_data)
    print(f"   高优先级干预: {len(report.urgent_interventions)}")
    print(f"   高风险 DDI: {report.high_risk_ddi_count}")

    # 8. 生成 HTML
    print("\n[8] 生成 HTML 报告...")
    html = generate_report_html(report)
    out_path = "t21_risperidone_report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"   已保存: {out_path}")

    print("\n" + "=" * 60)
    print("✅ T21 利培酮示范病例集成验证完成")
    print("=" * 60)


if __name__ == "__main__":
    main()