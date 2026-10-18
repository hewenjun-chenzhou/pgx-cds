"""临床准确性 + IM 场景回归测试"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.drug_recommender import recommend_both_genes, DrugRecommendation
from pages._agent_section import _merge_drug_items_by_name

print("=" * 90)
print("A. grade 修复验证")
print("=" * 90)
drugs_d6 = pd.read_csv("data/drugs.csv", encoding="utf-8-sig", dtype=str).fillna("")
drugs_c19 = pd.read_csv("data/drugs_cyp2c19.csv", encoding="utf-8-sig", dtype=str).fillna("")

EXPECTED = {
    ("CYP2D6", "地昔帕明"): "B",
    ("CYP2D6", "多塞平"):   "B",
    ("CYP2D6", "丙米嗪"):   "B",
    ("CYP2D6", "他莫昔芬"): "A",
    ("CYP2C19", "丙米嗪"):  "B",
    ("CYP2C19", "多塞平"):  "B",
}
all_pass = True
for (gene, drug), expected in EXPECTED.items():
    df = drugs_d6 if gene == "CYP2D6" else drugs_c19
    row = df[df.drug_name == drug]
    if row.empty:
        print(f"  ⚠️  找不到 {gene} {drug}")
        all_pass = False
        continue
    actual = str(row.iloc[0].get("recommendation_grade", "")).strip()
    status = "✓" if actual == expected else "❌"
    if actual != expected:
        all_pass = False
    print(f"  {status} {gene} {drug}: 实际={actual}, 期望={expected}")

print("\n" + "=" * 90)
print("B. alternatives 修复验证")
print("=" * 90)
ALT_EXPECTED = {
    ("CYP2D6", "曲马多"):   "吗啡/氢可酮/羟考酮",      # 不应再含布洛芬
    ("CYP2D6", "美托洛尔"): "阿替洛尔",                # 不应再含比索洛尔
    ("CYP2C19", "苯妥英"):  "左乙拉西坦",              # 不应再含丙戊酸
}
for (gene, drug), expected in ALT_EXPECTED.items():
    df = drugs_d6 if gene == "CYP2D6" else drugs_c19
    row = df[df.drug_name == drug]
    actual = str(row.iloc[0].get("alternatives", "")).strip()
    if actual == expected:
        print(f"  ✓ {gene} {drug}: alternatives = '{actual}'")
    else:
        print(f"  ❌ {gene} {drug}: 实际='{actual}', 期望='{expected}'")
        all_pass = False

print("\n" + "=" * 90)
print("C. P0 IM 场景回归（核心临床正确性）")
print("=" * 90)
# IM 阿米替林 → 应归 caution 而不是 safe
for gene, drug, expected_sev in [
    ("CYP2D6", "阿米替林", "caution"),  # IM + Grade B → 起始剂量低
    ("CYP2D6", "多塞平",   "caution"),  # IM + Grade B（新修）→ caution
    ("CYP2D6", "丙米嗪",   "caution"),  # IM + Grade B（新修）→ caution
    ("CYP2D6", "地昔帕明", "caution"),  # IM + Grade B（新修）→ caution
    ("CYP2D6", "奥氮平",   "safe"),     # IM + Grade D → safe（non_substrate）
    ("CYP2D6", "曲唑酮",   "caution"),  # IM + Grade C → caution（修复后）
    ("CYP2C19", "艾司西酞普兰", "caution"),  # IM + Grade B → caution
    ("CYP2C19", "舍曲林",  "caution"),  # IM + Grade B → caution
    ("CYP2C19", "奥氮平",  "safe"),     # IM + Grade D → safe
]:
    recs = recommend_both_genes({gene: "IM", "CYP2D6" if gene == "CYP2C19" else "CYP2D6": "IM"})
    target = [r for r in recs[gene].recommendations if r.drug_name == drug]
    if not target:
        print(f"  ⚠️  {gene} {drug}: 未找到该药")
        continue
    r = target[0]
    sev = "caution" if r.needs_adjustment else "safe"
    status = "✓" if sev == expected_sev else "❌"
    if sev != expected_sev:
        all_pass = False
    print(f"  {status} {gene} {drug} (IM): needs_adjustment={r.needs_adjustment} → {sev} (期望 {expected_sev})")

print("\n" + "=" * 90)
print("D. 端到端：IM 患者用 Grade B 阿米替林 → 不再误标安全")
print("=" * 90)
recs = recommend_both_genes({"CYP2D6": "IM", "CYP2C19": "NM"})
raw_items = []
for gene, rec in recs.items():
    for r in rec.recommendations:
        ph = "IM" if gene == "CYP2D6" else "NM"
        grade = (r.recommendation_grade or "").upper()
        needs_adj = r.needs_adjustment
        if ph in ("PM", "UM") and grade in ("A", "B") and needs_adj:
            sev = "avoid"
        elif (ph in ("PM", "UM") and grade in ("C", "D")) or (ph == "IM" and grade in ("A", "B", "C")):
            sev = "caution"
        else:
            sev = "safe"
        raw_items.append({
            "drug": r.drug_name, "gene": gene, "phenotype": ph,
            "grade": grade, "_severity": sev,
        })
dedup = _merge_drug_items_by_name(raw_items)
caution_drugs = [d["drug"] for d in dedup if d["_severity"] == "caution"]
key_drugs = ["阿米替林", "多塞平", "丙米嗪", "地昔帕明", "氯米帕明", "去甲替林"]
print(f"  IM 阿米替林类 TCA 出现在 caution 档:")
for d in key_drugs:
    print(f"    {'✓' if d in caution_drugs else '❌'} {d}")

# 三档分布
print(f"\n  IM 患者三档分布：")
counts = {"avoid": 0, "caution": 0, "safe": 0}
for d in dedup:
    counts[d["_severity"]] += 1
print(f"    ❌ 避免 {counts['avoid']} / ⚠️ 慎用 {counts['caution']} / ✅ 安全 {counts['safe']}")
if counts["caution"] == 0:
    print(f"  ❌ IM 患者没有任何 caution 档用药——说明修复未生效")
    all_pass = False

print("\n" + "=" * 90)
print("E. PM 场景回归（确保 PM 仍然正确）")
print("=" * 90)
recs = recommend_both_genes({"CYP2D6": "PM", "CYP2C19": "NM"})
raw_items = []
for gene, rec in recs.items():
    for r in rec.recommendations:
        ph = "PM" if gene == "CYP2D6" else "NM"
        grade = (r.recommendation_grade or "").upper()
        needs_adj = r.needs_adjustment
        if ph in ("PM", "UM") and grade in ("A", "B") and needs_adj:
            sev = "avoid"
        elif (ph in ("PM", "UM") and grade in ("C", "D")) or (ph == "IM" and grade in ("A", "B", "C")):
            sev = "caution"
        else:
            sev = "safe"
        raw_items.append({
            "drug": r.drug_name, "gene": gene, "phenotype": ph,
            "grade": grade, "_severity": sev,
        })
dedup = _merge_drug_items_by_name(raw_items)
counts = {"avoid": 0, "caution": 0, "safe": 0}
for d in dedup:
    counts[d["_severity"]] += 1
print(f"  PM 患者三档分布：❌ {counts['avoid']} / ⚠️ {counts['caution']} / ✅ {counts['safe']}")
# 关键：他莫昔芬（grade 现在是 A）应进 avoid
he_is_avoid = any(d["drug"] == "他莫昔芬" and d["_severity"] == "avoid" for d in dedup)
print(f"  {'✓' if he_is_avoid else '❌'} 他莫昔芬 (PM) → avoid（grade A 应触发 avoid 档）")
if not he_is_avoid:
    all_pass = False

print("\n" + "=" * 90)
print("总结：" + ("✅ 全部修复" if all_pass else "❌ 仍有失败"))
print("=" * 90)
sys.exit(0 if all_pass else 1)