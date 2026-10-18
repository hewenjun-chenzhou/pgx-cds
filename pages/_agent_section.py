r"""
pages/_agent_section.py
==========================
"🤖 Agent 推理" 区块（嵌入 pages/2_分析报告.py）

设计原则：
- API 失败时降级到规则检索，不卡 UI
- 缓存结果到 session_state，避免重复调用
- 展示 LLM 推理过程 + 知识库引用
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

# 提前导入，避免每个函数内重复 import
from src.data_loader import get_cache

# ✅ 补丁 ②：把 drugs.csv 的 source 字段压缩成短徽章
def _shorten_source(src: str) -> str:
    """把 drugs.csv 的 source 字段压缩成短徽章标签。

    返回示例：
      'CPIC 2023 SSRI/SNRI指南（Bousman et al., Clin Pharmacol Ther 2023;114:51-68）' → 'CPIC 2023'
      'DPWG 2024指南（Beunk et al., Eur J Hum Genet 2024;32:278-285） / Luca et al. Eur J Psychiatry 2026;40:100338' → 'Luca 2026'
    """
    if not src:
        return "—"
    s = src.upper()
    if "LUCA" in s and ("EUR J PSYCHIATRY" in s or "EJP" in s):
        return "Luca 2026"
    if "DPWG" in s and "CPIC" in s:
        return "CPIC+DPWG"
    if "DPWG" in s:
        return "DPWG 2024"
    if "FDA" in s and "DPWG" in s:
        return "FDA+DPWG"
    if "FDA" in s:
        return "FDA label"
    if "CPIC 2023" in s: return "CPIC 2023"
    if "CPIC 2022" in s: return "CPIC 2022"
    if "CPIC 2021" in s: return "CPIC 2021"
    if "CPIC 2020" in s: return "CPIC 2020"
    if "CPIC 2019" in s: return "CPIC 2019"
    if "CPIC 2018" in s: return "CPIC 2018"
    if "CPIC 2017" in s: return "CPIC 2017"
    if "CPIC 2016" in s: return "CPIC 2016"
    if "CPIC" in s: return "CPIC"
    if "PHARMGKB" in s: return "PharmGKB"
    return src[:12]

# 延迟导入 DeepSeek Agent（避免无 key 时也尝试连接）
def _get_agent():
    """懒加载 Agent，避免每次都尝试连接"""
    if "agent" in st.session_state:
        return st.session_state["agent"]

    # 按优先级读取 key
    api_key = None
    try:
        import os
        # 1. 环境变量
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        # 2. .env 文件
        if not api_key:
            env_file = ROOT / ".env"
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("DEEPSEEK_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    except Exception:
        pass

    if not api_key:
        st.session_state["agent"] = None
        return None

    try:
        from src.agent_reasoner import PGxAgent
        agent = PGxAgent(api_key=api_key)
        st.session_state["agent"] = agent
        return agent
    except Exception as e:
        st.session_state["agent"] = None
        st.session_state["agent_error"] = str(e)
        return None


def render_agent_section(report):
    """在分析报告页渲染 Agent 推理区块（v2.0 重构版）"""

    st.divider()

    # ── 1. 💊 当前用药速览（确定性输出，无需 LLM）──
    render_current_meds_summary(report)

    # ── 2. Agent 问答入口（核心）──
    render_agent_qa(report)

    # ── 3. 基因检测后 · 用药推荐已删除（与当前用药速览重复）──


# ══════════════════════════════════════════════════════════════
# 当前用药速览（确定性，无 LLM）
# v2.0 · 基于「最大概率表型」给出用药建议
# ══════════════════════════════════════════════════════════════

# 每个动作对应的 emoji / 颜色：决定卡片视觉
_PHE_ICON = {"PM": "🔴", "IM": "🟠", "NM": "🟢", "UM": "🔴"}
_PHE_LABEL = {
    "PM": "弱代谢",
    "IM": "中间代谢",
    "NM": "正常代谢",
    "UM": "超速代谢",
}

# 动作优先级：避免(0) > 换药(1) > 减量(2) > 维持+监测(3) > 维持(4)
_ACTION_RANK = {"避免": 0, "换药": 1, "减量": 2, "维持+监测": 3, "维持": 4, "": 5}
_GRADE_RANK = {"A": 0, "B": 1, "C": 2, "D": 3, "": 4}

# 三档清单的严重度等级（avoid > caution > safe）
_SEVERITY_RANK = {"avoid": 0, "caution": 1, "safe": 2}


def _merge_drug_items_by_name(items: list[dict]) -> list[dict]:
    """
    按药品名合并多基因条目（修复双酶底物重复 bug）。

    场景：多塞平同时是 CYP2D6 + CYP2C19 的底物，遍历两个基因时会重复出现。
    本函数按 drug_name 合并：
      - 取最严重的档位（避免 > 慎用 > 安全；PM/UM > IM > NM）
      - 同档时取最严格的 grade（A < B < C < D）
      - 表型字段合并展示：CYP2D6 NM + CYP2C19 PM → gene="CYP2D6/CYP2C19"，
        phenotype="NM+PM"，以便医生一眼看出"该药受双基因影响"
      - 当前 CPIC 建议 / 替代药：取最严重那条的（避免档位的建议最有临床意义）

    Parameters
    ----------
    items : list[dict]
        每条至少包含 "drug" 字段，可选 "gene"/"phenotype"/"grade"/
        "_severity"/"current_action"/"alternatives"/"emoji"/"action" 等。

    Returns
    -------
    合并后的 list[dict]，同一药名仅出现一次。
    """
    if not items:
        return []

    by_drug: dict[str, dict] = {}

    for it in items:
        drug = it.get("drug", "").strip()
        if not drug:
            continue
        if drug not in by_drug:
            # 首个出现 → 完整复制，初始化累积字段
            by_drug[drug] = dict(it)
            by_drug[drug]["_all_genes"] = [(it.get("gene", ""), it.get("phenotype", ""))]
            continue

        cur = by_drug[drug]
        # 累积基因信息
        cur.setdefault("_all_genes", [(cur.get("gene", ""), cur.get("phenotype", ""))])
        cur["_all_genes"].append((it.get("gene", ""), it.get("phenotype", "")))

        # 比较严重度，决定是否替换主字段
        cur_sev = _SEVERITY_RANK.get(cur.get("_severity", "safe"), 2)
        new_sev = _SEVERITY_RANK.get(it.get("_severity", "safe"), 2)
        cur_grade = _GRADE_RANK.get((cur.get("grade") or "").upper(), 4)
        new_grade = _GRADE_RANK.get((it.get("grade") or "").upper(), 4)

        # 替换条件：更严重 或 （同档但 grade 更严格）
        should_replace = (new_sev < cur_sev) or (new_sev == cur_sev and new_grade < cur_grade)
        if should_replace:
            # 替换主要字段，但保留累积的 _all_genes
            saved_all = cur["_all_genes"]
            for k, v in it.items():
                cur[k] = v
            cur["_all_genes"] = saved_all

    # 合并展示：把 gene / phenotype 改成 "CYP2D6 NM / CYP2C19 PM"
    result: list[dict] = []
    for drug, item in by_drug.items():
        all_genes = item.get("_all_genes") or [(item.get("gene", ""), item.get("phenotype", ""))]
        if len(all_genes) > 1:
            gene_str = "/".join(g for g, _ in all_genes if g)
            # 表型合并：去重保持出现顺序
            phenos: list[str] = []
            for _, p in all_genes:
                if p and p not in phenos:
                    phenos.append(p)
            item["gene"] = gene_str
            item["phenotype"] = "+".join(phenos)
            # 在合并条目中追加标记，便于追溯
            item["_merged_genes"] = all_genes
        result.append(item)

    return result


def _decide_action_for_phenotype(phenotype: str) -> tuple[str, str]:
    """根据表型返回 (动作文案, emoji)"""
    if phenotype in ("PM", "UM"):
        return "换药", _PHE_ICON[phenotype]
    if phenotype == "IM":
        return "减量监测", _PHE_ICON[phenotype]
    return "维持", _PHE_ICON[phenotype]


def _phenotype_prob(phen, target: str) -> float:
    """从 phenotype_posteriors 取出 target 表型的后验概率，找不到则退回 best_genotype_confidence"""
    pp = getattr(phen, "phenotype_posteriors", None) or {}
    if isinstance(pp, dict):
        v = pp.get(target)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return float(getattr(phen, "best_genotype_confidence", 0.0) or 0.0)


def _extract_current_meds_list(report) -> list[str]:
    """从 report.patient.current_meds 提取归一化的 list[str]"""
    raw = getattr(report.patient, "current_meds", None)
    if raw is None:
        return []
    if isinstance(raw, list):
        meds = [str(m).strip() for m in raw if str(m).strip()]
    else:
        meds = [
            m.strip()
            for m in re.split(r"[,，;；\n]", str(raw))
            if m.strip()
        ]
    return meds


def _high_risk_count(rec) -> int:
    """统计一个 GeneRecommendations 中"高风险药品"的数量（PM/UM 时 Grade A/B & 需要调整的）"""
    if not rec:
        return 0
    phenotype = getattr(rec, "phenotype", "NM")
    cnt = 0
    for r in (getattr(rec, "high_priority", []) or []):
        if phenotype in ("PM", "UM") and r.needs_adjustment:
            cnt += 1
        elif phenotype == "IM" and r.recommendation_grade in ("A", "B"):
            cnt += 1
    return cnt


def render_current_meds_summary(report):
    """当前用药建议（v2.1）

    自动识别两种场景：
      A) 复诊（有 current_meds）  → 当前用药检查 + 调整建议
      B) 首诊（无 current_meds） → 三档清单：避免 / 慎用 / 安全

    完全确定性，不调用 LLM。
    """
    st.divider()
    current_meds_list = _extract_current_meds_list(report)

    # ── 区分首诊 vs 复诊 ──
    if not current_meds_list:
        # 首诊：给出避免/慎用/安全三档清单
        st.subheader("💊 当前用药建议")
        st.caption(
            "💡 **首诊场景**（未提供当前用药）。基于**最大概率表型**输出三档用药清单："
            "❌ 避免 / ⚠️ 慎用 / ✅ 安全。具体首选药需结合诊断（请参考下方「询问 Agent」）。"
        )
        _render_first_visit_advice(report)
        return

    # 复诊：原有逻辑
    st.subheader("💊 当前用药建议")
    st.caption(
        "💡 **复诊场景**（当前在用药品）。基于**最大概率表型**给出用药建议："
        "左表型概览 → 右上当前用药检查 → 右下关键药品清单。"
        "具体剂量和随访细节 → 下方「询问 Agent」。"
    )

    drug_recs = getattr(report, "drug_recs", {}) or {}
    phenotypes = getattr(report, "phenotypes", {}) or {}
    current_meds_list = _extract_current_meds_list(report)

    # ───────────────── 兜底：补全 drug_recs.recommendations ────────────────
    # 场景：旧 session_state 里的 drug_recs 是在 meds_list=[] 触发的
    # recommend_for_gene 时生成的 → recommendations 被全部过滤掉。
    # 这里重新调用一次，强制传 None，让所有按最大概率表型计算的药品都回来。
    try:
        from src.drug_recommender import recommend_both_genes
        phenotype_map = {
            gene: phenotypes[gene].phenotype
            for gene in ("CYP2D6", "CYP2C19")
            if gene in phenotypes and phenotypes[gene].phenotype
        }
        for gene, rec in drug_recs.items():
            if not getattr(rec, "recommendations", None):
                rec.recommendations = recommend_both_genes(
                    phenotype_map, None
                ).get(gene, rec).recommendations
                # high_priority 也要重建（影响 _high_risk_count 计数）
                rec.high_priority = [
                    r for r in rec.recommendations
                    if r.recommendation_grade in ("A", "B") and r.needs_adjustment
                ]
    except Exception:
        # 数据加载失败时静默——下面会用空推荐退出，不影响其它模块
        pass

    # ───────────────── ① 表型概览（每基因一行）─────────────────
    phen_rows = []
    for gene in ["CYP2D6", "CYP2C19"]:
        phen = phenotypes.get(gene)
        rec = drug_recs.get(gene)
        if not phen or not phen.phenotype:
            continue
        prob = _phenotype_prob(phen, phen.phenotype)
        risk = _high_risk_count(rec)
        phen_rows.append({
            "gene": gene,
            "phenotype": phen.phenotype,
            "phen_label": _PHE_LABEL.get(phen.phenotype, phen.phenotype),
            "prob": prob,
            "risk": risk,
            "best_gt": getattr(phen, "best_genotype", "") or "?",
        })

    if phen_rows:
        cols = st.columns(len(phen_rows))
        for col, row in zip(cols, phen_rows):
            with col:
                icon = _PHE_ICON.get(row["phenotype"], "⚪")
                st.markdown(
                    f"### {icon} {row['gene']} {row['phenotype']}",
                    help=f"最大概率表型 · 后验 {row['prob']:.1%}",
                )
                st.caption(
                    f"_{row['phen_label']}_  · 后验 **{row['prob']:.1%}**"
                )
                st.caption(f"最可能基因型：`{row['best_gt']}`")

    st.markdown("---")

    # ───────────────── ② 当前用药检查（仅交集）─────────────────
    st.markdown("**✅ 当前用药检查**（仅与患者当前在用药品求交集）")
    raw_med_items = []
    for gene in ["CYP2D6", "CYP2C19"]:
        rec = drug_recs.get(gene)
        phen = phenotypes.get(gene)
        if not rec or not phen or not phen.phenotype:
            continue
        for r in rec.recommendations:
            if not _drug_in_current(r.drug_name, current_meds_list):
                continue
            action, emoji = _decide_action_for_phenotype(phen.phenotype)
            # 严重度：避免/换药 > 减量监测 > 维持+监测 > 维持
            sev = {"避免": "avoid", "换药": "avoid", "减量监测": "caution",
                   "维持+监测": "safe", "维持": "safe"}.get(action, "safe")
            raw_med_items.append({
                "drug": r.drug_name,
                "gene": gene,
                "phenotype": phen.phenotype,
                "action": action,
                "emoji": emoji,
                "grade": r.recommendation_grade,
                # ✅ 补丁 ⑥：注入 source 字段，给当前用药小卡片用
                "source": r.source or "",
                "source_short": _shorten_source(r.source or ""),
                "_severity": sev,
            })

    # 关键修复：按药品名 dedup，避免双酶底物重复展示
    med_items = _merge_drug_items_by_name(raw_med_items)

    if med_items:
        for item in med_items:
            c = st.columns([2, 1, 1, 1])
            with c[0]:
                st.markdown(f"**{item['drug']}**")
            with c[1]:
                st.caption(f"{item['gene']} {item['phenotype']}")
            with c[2]:
                st.markdown(f"{item['emoji']} {item['action']}")
            with c[3]:
                # ✅ 补丁 ⑥：当前用药小卡片用动态来源徽章
                st.caption(f"{item.get('source_short', 'CPIC')} {item['grade']}")
    else:
        if current_meds_list:
            st.info("ℹ️ 当前在用的药品未进入 CYP2D6/CYP2C19 推荐数据库，无需在最大概率表型下做特殊调整。")
        else:
            st.info("ℹ️ 未提供当前用药清单——请录入或参考下方「关键药品清单」。")

    st.markdown("---")

    # ───────────────── ③ 关键药品清单（最大概率表型）─────────────────
    st.markdown(
        "**🚨 关键药品清单（按最大概率表型）** "
        f"· 即使未在使用，医生也应知悉该表型下的高风险/常用药品。"
    )

    key_items: list[dict] = []
    for gene in ["CYP2D6", "CYP2C19"]:
        rec = drug_recs.get(gene)
        phen = phenotypes.get(gene)
        if not rec or not phen or not phen.phenotype:
            continue
        phenotype = phen.phenotype

        for r in rec.recommendations:
            action, emoji = _decide_action_for_phenotype(phenotype)
            # NM 表型下默认是"维持常规剂量"——只在 NM 时把"维持"压到列表底部
            if phenotype == "NM" and action == "维持":
                rank_action = _ACTION_RANK["维持+监测"]   # 仍然展示但靠后
            else:
                rank_action = _ACTION_RANK.get(action, 5)

            # 严重度（用于 dedup 时排序）
            sev = {"避免": "avoid", "换药": "avoid", "减量监测": "caution",
                   "维持+监测": "safe", "维持": "safe"}.get(action, "safe")

            key_items.append({
                "drug": r.drug_name,
                "gene": gene,
                "phenotype": phenotype,
                "action": action,
                "emoji": emoji,
                "grade": r.recommendation_grade or "",
                "alternatives": (r.alternatives or "").strip(),
                "current_action": r.current_action or "",
                "needs_adjustment": getattr(r, "needs_adjustment", False),
                "is_current": _drug_in_current(r.drug_name, current_meds_list),
                # ✅ 补丁 ②：把 source 字段从 DrugRecommendation 透传到 UI
                "source": r.source or "",
                "source_short": _shorten_source(r.source or ""),
                "_rank_action": rank_action,
                "_rank_grade": _GRADE_RANK.get((r.recommendation_grade or "").upper(), 4),
                "_rank_in_use": 0 if _drug_in_current(r.drug_name, current_meds_list) else 1,
                "_severity": sev,
            })

    # 关键修复：按药品名 dedup，避免双酶底物重复展示
    key_items = _merge_drug_items_by_name(key_items)

    if not key_items:
        st.success("✅ 当前用药在推断表型下无需特别调整")
        return

    # 排序：在用 > 避免/换药 > Grade A/B > NM 时维护性药品
    key_items.sort(key=lambda x: (
        x["_rank_in_use"],
        x["_rank_action"],
        x["_rank_grade"],
        x["drug"],
    ))

    # 限前 8 条（避免过长）
    MAX_KEY_ITEMS = 8
    show_items = key_items[:MAX_KEY_ITEMS]
    extra_count = len(key_items) - len(show_items)

    # ───────────────── ③-1 横向对比表格（紧凑版）─────────────────
    st.markdown(
        "**🚨 关键药品清单（按最大概率表型）** · 表格对比"
    )
    import pandas as pd
    table_rows = []
    for it in show_items:
        action_short = f"{it['emoji']} {it['action']}"
        # 化验叠加：当前推荐中如果 needs_adjustment=True 则提示化验
        lab_hint = "🔬 见详情" if it["needs_adjustment"] else "—"
        in_use = "✓" if it["is_current"] else ""
        table_rows.append({
            "药品": f"{in_use} {it['drug']}".strip(),
            "表型": f"{it['gene']} {it['phenotype']}",
            "动作": action_short,
            "来源": it.get("source_short", "—"),                              # ✅ 补丁 ③：动态来源徽章
            "等级": f"Grade {it['grade']}" if it["grade"] else "—",           # ✅ 补丁 ③：等级独立成列
            "替代": it["alternatives"] or "—",
            "化验": lab_hint,
        })
    df = pd.DataFrame(table_rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "药品": st.column_config.TextColumn("药品", width="medium"),
            "表型": st.column_config.TextColumn("表型", width="small"),
            "动作": st.column_config.TextColumn("动作", width="small"),
            "来源": st.column_config.TextColumn("来源", width="small"),        # ✅ 补丁 ③
            "等级": st.column_config.TextColumn("等级", width="small"),        # ✅ 补丁 ③
            "替代": st.column_config.TextColumn("替代", width="medium"),
            "化验": st.column_config.TextColumn("化验叠加", width="small"),
        },
    )

    st.markdown("---")
    st.markdown("**📋 详情卡片**（点击行 1 展开完整推理链）")

    # ───────────────── ③-2 详情卡片（精简版 + 默认折叠）─────────────────
    for idx, item in enumerate(show_items):
        in_use_badge = "🟢 当前在用" if item["is_current"] else ""
        # ✅ 补丁 ④：标题用动态来源徽章
        src_short = item.get("source_short", "—")
        grade_text = f"{src_short} · Grade {item['grade']}" if item["grade"] else src_short
        # 取该基因的最关键 organ_reason（化验修正依据）
        phen_for_item = phenotypes.get(item["gene"])
        top_reason = (
            phen_for_item.organ_reasons[0]
            if phen_for_item and getattr(phen_for_item, "organ_reasons", None)
            else None
        )
        with st.expander(
            f"💊 {item['drug']} · {item['gene']} {item['phenotype']} · "
            f"{item['emoji']} {item['action']} · {grade_text} {in_use_badge}",
            expanded=False,
        ):
            # 精简卡片：1 行核心 + 展开 CPIC 全文
            if item["current_action"]:
                # ✅ 补丁 ④：用动态来源徽章替换硬编码 "CPIC 建议"
                st.markdown(f"**📋 {src_short} 建议**：{item['current_action']}")
            # ✅ 补丁 ④：把 drugs.csv 的完整 source 字段（含 Luca 2026 卷号）展示出来
            if item.get("source"):
                st.caption(f"📚 **证据来源**：{item['source']}")
            if item["alternatives"]:
                st.markdown(f"**🔁 替代药物**：{item['alternatives']}")
            if item["needs_adjustment"]:
                if top_reason:
                    st.caption(f"🔬 **化验叠加**：{top_reason}")
                else:
                    # 无化验异常时，给出表型层面的原因
                    pheno_reason = {
                        "PM": "表型 PM → 代谢能力↓，母药/前药蓄积风险",
                        "UM": "表型 UM → 代谢能力↑，活性药物/毒性代谢物暴露↑",
                        "IM": "表型 IM → 代谢能力中等，部分药物需减量",
                    }.get(item["phenotype"], f"表型 {item['phenotype']} 需调整剂量")
                    st.caption(f"⚠️ **调整依据**：{pheno_reason}")

    if extra_count > 0:
        st.caption(
            f"… 还有 {extra_count} 条其余药品，"
            f"完整列表可点击下方「询问 Agent」按药品名提问。"
        )


def _render_first_visit_advice(report):
    """首诊场景：基于最大概率表型输出避免/慎用/安全三档清单

    分类规则（基于 CPIC Grade + needs_adjustment）：
        ❌ 避免:    PM/UM + Grade A/B + needs_adjustment
        ⚠️ 慎用:   PM/UM + Grade C/D   | IM + Grade A/B
        ✅ 安全:   NM + 无需调整        | IM/NM + Grade C/D
    """
    drug_recs = getattr(report, "drug_recs", {}) or {}
    phenotypes = getattr(report, "phenotypes", {}) or {}

    # ── 表型概览（顶部） ──
    phen_rows = []
    for gene in ["CYP2D6", "CYP2C19"]:
        phen = phenotypes.get(gene)
        if not phen or not phen.phenotype:
            continue
        prob = _phenotype_prob(phen, phen.phenotype)
        phen_rows.append({
            "gene": gene,
            "phenotype": phen.phenotype,
            "phen_label": _PHE_LABEL.get(phen.phenotype, phen.phenotype),
            "prob": prob,
            "best_gt": getattr(phen, "best_genotype", "") or "?",
        })

    if phen_rows:
        cols = st.columns(len(phen_rows))
        for col, row in zip(cols, phen_rows):
            with col:
                icon = _PHE_ICON.get(row["phenotype"], "⚪")
                st.markdown(
                    f"### {icon} {row['gene']} {row['phenotype']}",
                    help=f"最大概率表型 · 后验 {row['prob']:.1%}",
                )
                st.caption(
                    f"_{row['phen_label']}_  · 后验 **{row['prob']:.1%}**"
                )
                st.caption(f"最可能基因型：`{row['best_gt']}`")

    st.markdown("---")

    # ── 收集三档清单（先 dedup 再分档）──
    raw_items: list[dict] = []
    for gene in ["CYP2D6", "CYP2C19"]:
        rec = drug_recs.get(gene)
        phen = phenotypes.get(gene)
        if not rec or not phen or not phen.phenotype:
            continue
        for r in (getattr(rec, "recommendations", []) or []):
            phenotype = phen.phenotype
            grade = (r.recommendation_grade or "").upper()
            needs_adj = getattr(r, "needs_adjustment", False)

            # 三档分类（候选档位，dedup 后再以最严重档为准）
            # 修复前：IM + Grade C/D → safe（错！会把 IM 用 Grade B/C 的阿米替林归为安全）
            # 修复后：IM + Grade A/B/C → caution（CPIC 指南明确 IM 应"起始剂量低/密切监测"）
            if phenotype in ("PM", "UM") and grade in ("A", "B") and needs_adj:
                bucket, emoji = "avoid", "❌"
            elif (phenotype in ("PM", "UM") and grade in ("C", "D")) or \
                 (phenotype == "IM" and grade in ("A", "B", "C")):
                bucket, emoji = "caution", "⚠️"
            else:
                bucket, emoji = "safe", "✅"

            raw_items.append({
                "drug": r.drug_name,
                "gene": gene,
                "phenotype": phenotype,
                "grade": grade,
                "emoji": emoji,
                "current_action": (r.current_action or "").strip(),
                "alternatives": (r.alternatives or "").strip(),
                # ✅ 补丁 ⑦：注入 source 字段，给首诊三档表用
                "source": r.source or "",
                "source_short": _shorten_source(r.source or ""),
                "_severity": bucket,
            })

    # ── 关键修复：按药品名 dedup，取严重度最高者 ──
    # 临床意义：双酶底物（如多塞平、阿米替林等 9 个）在两个基因下都有条目，
    # 必须按"任一酶异常即视为需要警惕"原则合并，避免出现"同药半安全半危险"的混乱。
    dedup_items = _merge_drug_items_by_name(raw_items)

    # 重新按 dedup 后的 _severity 字段分桶
    avoid, caution, safe = [], [], []
    for it in dedup_items:
        bucket = it.get("_severity", "safe")
        {"avoid": avoid, "caution": caution, "safe": safe}[bucket].append(it)

    # ── 渲染三档（压缩表格版） ──
    import pandas as pd
    cols_header = st.columns(3)
    for col, items, title in [
        (cols_header[0], avoid, "❌ 严格避免"),
        (cols_header[1], caution, "⚠️ 慎用/减量"),
        (cols_header[2], safe, "✅ 相对安全"),
    ]:
        with col:
            st.markdown(f"#### {title}（{len(items)} 种）")

    # 合并三档到一张表（用「档位」列区分）
    table_rows = []
    for bucket_name, items, emoji in [
        ("避免", avoid, "❌"),
        ("慎用", caution, "⚠️"),
        ("安全", safe, "✅"),
    ]:
        items_sorted = sorted(
            items,
            key=lambda x: (_GRADE_RANK.get(x["grade"], 4), x["drug"]),
        )
        for it in items_sorted:
            # 截断 CPIC 建议到 40 字内（表格内只显示概要）
            action_brief = it["current_action"][:40] + ("…" if len(it["current_action"]) > 40 else "")
            table_rows.append({
                "档": f"{emoji} {bucket_name}",
                "药品": it["drug"],
                "表型": f"{it['gene']} {it['phenotype']}",
                "来源": it.get("source_short", "—"),                              # ✅ 补丁 ⑦
                "等级": f"Grade {it['grade']}" if it["grade"] else "—",           # ✅ 补丁 ⑦
                "建议摘要": action_brief or "—",
                "替代": it["alternatives"] or "—",
            })

    if table_rows:
        df = pd.DataFrame(table_rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "档": st.column_config.TextColumn("档位", width="small"),
                "药品": st.column_config.TextColumn("药品", width="small"),
                "表型": st.column_config.TextColumn("表型", width="small"),
                "来源": st.column_config.TextColumn("来源", width="small"),        # ✅ 补丁 ⑦
                "等级": st.column_config.TextColumn("等级", width="small"),        # ✅ 补丁 ⑦
                "建议摘要": st.column_config.TextColumn("建议摘要", width="large"),
                "替代": st.column_config.TextColumn("替代", width="medium"),
            },
        )

    st.markdown("---")
    st.caption(
        "💡 **重要提示**：以上清单基于最大概率表型（贝叶斯推断）。"
        "具体首选药需结合**临床诊断**——本系统未收录诊断→首选药映射，"
        "请参考您所在医院的临床路径指南，或通过下方「询问 Agent」按药品名提问。"
    )


def _drug_in_current(drug_name: str, current_meds_list: list[str]) -> bool:
    """检查药品是否在当前用药列表中（子串匹配，忽略大小写）"""
    drug_lower = drug_name.lower().strip()
    for med in current_meds_list:
        med_lower = med.lower().strip()
        if not med_lower:
            continue
        # 双向子串匹配
        if drug_lower in med_lower or med_lower in drug_lower:
            return True
    return False


def _format_agent_a_markdown(explanation: str) -> str:
    """把 Agent A 的输出格式化成 v1.0 风格的 markdown。

    输入可能是：
      - JSON 字符串（LLM 按 schema 返回）
      - JSON dict（已 parsed）
      - markdown 字符串（LLM 忽略 schema 返回的 markdown）

    输出（v1.0）:
      [来源] CYP2D6 *1/*10 (NM) 推断依据
        [先验] 东亚人群 HWE: *1/*10 = 33.6%
        [证据 1] 临床症状: 可待因镇痛无效（LR=10, 来源: clinical_signals.csv Row 4）
        [计算] 33.6% × 10 × 1.1 = 后验 ≈ 44.6%
        [不确定性] 置信区间 [42%, 47%]
    """
    import json as _json

    # ── Step 1：解析输入 → 统一为 dict 形式 ──
    data = None
    if isinstance(explanation, dict):
        data = explanation
    elif isinstance(explanation, str):
        text = explanation.strip()

        # 尝试 JSON 解析
        if text.startswith("{") or text.startswith("["):
            try:
                data = _json.loads(text)
            except (_json.JSONDecodeError, TypeError):
                pass

        # 否则当作 markdown 解析
        if data is None:
            data = _parse_markdown_to_dict(text)

    if data is None:
        return "（无输出）"

    # ── Step 2：提取字段（兼容 JSON keys 和 markdown 解析后的 keys）──
    # 支持的 key 映射（LLM 可能用的不同 keys）
    conclusion = (
        data.get("结论") or data.get("推断结论") or data.get("conclusion") or {}
    )
    if isinstance(conclusion, str):
        conclusion = {"基因型": conclusion}

    # 提取关键字段（兼容多种命名）
    gene = (
        conclusion.get("基因") or conclusion.get("gene") or "CYP2D6"
    ) if isinstance(conclusion, dict) else "CYP2D6"
    best_geno = (
        conclusion.get("最可能基因型") or conclusion.get("基因型")
        or conclusion.get("best_genotype") or "?"
    ) if isinstance(conclusion, dict) else "?"
    best_pheno = (
        conclusion.get("基因型表型") or conclusion.get("表型")
        or conclusion.get("phenotype") or "?"
    ) if isinstance(conclusion, dict) else "?"
    confidence = (
        conclusion.get("置信度") or conclusion.get("confidence") or "0.0%"
    ) if isinstance(conclusion, dict) else "0.0%"
    eff_pheno = (
        conclusion.get("有效代谢") or conclusion.get("effective_phenotype")
        or best_pheno
    ) if isinstance(conclusion, dict) else best_pheno
    advice = (
        conclusion.get("建议") or conclusion.get("advice") or "系统未提供"
    ) if isinstance(conclusion, dict) else "系统未提供"

    # 先验
    prior = data.get("先验") or data.get("prior") or {}
    if isinstance(prior, str):
        prior = {"人群": prior}
    population = (
        prior.get("人群") or prior.get("population") or "东亚"
    ) if isinstance(prior, dict) else "东亚"
    prior_str = (
        prior.get("该基因型先验") or prior.get("基因型先验")
        or prior.get("prior_prob") or "系统未提供"
    ) if isinstance(prior, dict) else "系统未提供"

    # 证据 + Top 5
    clin = (
        data.get("临床证据") or data.get("clinical_signals") or []
    )
    blood = (
        data.get("血液证据") or data.get("blood_markers") or []
    )
    top5 = (
        data.get("top5基因型") or data.get("top_genotypes") or []
    )

    # ── Step 3：输出干净格式 ──
    lines = [
        f"**{gene} `{best_geno}` ({best_pheno})**  · 后验 {confidence}",
        f"_有效代谢：{eff_pheno}_",
        "",
        "**📋 推断依据**",
        f"- **先验**：{population} 人群 HWE：`{best_geno}` = {prior_str}",
    ]

    ev_idx = 1
    for s in clin:
        if isinstance(s, dict):
            desc = s.get('证据描述') or s.get('description') or s.get('signal') or '?'
            ph = s.get('指向表型') or s.get('target_phenotype') or '?'
            strength = s.get('强度') or s.get('strength') or '?'
            lines.append(
                f"- **证据 {ev_idx}（临床）**：{desc} → 提示 {ph}（强度:{strength}）"
            )
            ev_idx += 1
    for s in blood:
        if isinstance(s, dict):
            marker = s.get('标记物') or s.get('marker') or s.get('name') or '?'
            val = s.get('实际值') or s.get('value') or '?'
            impact = s.get('影响') or s.get('impact') or '?'
            lines.append(
                f"- **证据 {ev_idx}（血液）**：{marker}={val} → {impact}"
            )
            ev_idx += 1

    lines.append("")
    lines.append("**🧮 计算**")
    lines.append(f"- {prior_str} × 各 LR 累积 = 后验 **{confidence}**")
    lines.append("")
    lines.append("**📐 不确定性**")
    lines.append(f"- 置信区间 ±1% · 封顶规则：临床封顶 ≤85%")

    # ── 扩展 ──
    if top5:
        lines.append("")
        lines.append("**🔬 Top 5 基因型候选**")
        for g in top5[:5]:
            if isinstance(g, dict):
                lines.append(
                    f"- `{g.get('基因型') or g.get('genotype') or '?'}` → "
                    f"{g.get('表型') or g.get('phenotype') or '?'}（{g.get('后验') or g.get('posterior') or '?'}）"
                )

    lines.append("")
    lines.append(f"**💡 建议**：{advice}")

    return "\n".join(lines)


def _strip_markdown_chars(s: str) -> str:
    """
    智能去掉 markdown 标记字符：
    - `code` → code（去掉反引号）
    - **bold** → bold（去掉成对 **）
    - *italic* → italic（去掉成对 *）
    - *1/*10 → *1/*10（保留内容里的 *）

    关键：只去掉**成对**的标记符，不破坏内容里的字符。
    """
    s = s.strip()

    # 反引号包裹的代码：保留内容
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        s = s[1:-1]

    # 成对的 ** **
    while s.startswith("**") and s.endswith("**") and len(s) >= 4:
        s = s[2:-2]

    # 成对的 * *（但要保护像 *1/*10 这种）
    # 检查是否真的是 italic 标记：起始* 后应是文本而非数字
    if s.startswith("*") and s.endswith("*") and len(s) >= 2:
        # 第二个字符如果是字母（italic），则去掉
        # 第二个字符如果是数字或 / （如 *1/*10），则保留
        if len(s) >= 3 and not s[1].isdigit() and s[1] != "/" and s[1] != " ":
            s = s[1:-1]

    return s.strip()


def _parse_markdown_to_dict(text: str) -> dict:
    """
    把 LLM 返回的 markdown 格式（## 推断结论 + - 基因: CYP2D6）
    解析回 dict 形式，便于重新格式化。
    """
    data = {}
    current_section = None

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # 章节标题：## 推断结论
        if line.startswith("## "):
            section_name = line[3:].strip()
            if section_name in ("推断结论", "结论"):
                current_section = "结论"
                data["结论"] = {}
            elif section_name in ("先验",):
                current_section = "先验"
                data["先验"] = {}
            elif "Top 5" in section_name or "Top" in section_name:
                current_section = "top5基因型"
                data["top5基因型"] = []
            elif "证据" in section_name and "临床" in section_name:
                current_section = "临床证据"
                data["临床证据"] = []
            elif "证据" in section_name and ("血液" in section_name or "CCA" in section_name):
                current_section = "血液证据"
                data["血液证据"] = []
            else:
                current_section = None
            continue

        # bullet list：- 基因: CYP2D6
        if line.startswith("- "):
            content = line[2:].strip()
            # 解析 key: value
            if ":" in content or "：" in content:
                # 用中英文冒号分割
                for sep in ["：", ":"]:
                    if sep in content:
                        key, val = content.split(sep, 1)
                        # 只去掉成对的 ** 和 * 标记（不破坏 *1/*10 这种内容）
                        key = _strip_markdown_chars(key)
                        val = _strip_markdown_chars(val)
                        if current_section == "结论":
                            data["结论"][key] = val
                        elif current_section == "先验":
                            data["先验"][key] = val
                        break
            else:
                # bullet with no key:value（Top 5 那类：- *1/*10 → NM（44.6%））
                if current_section == "top5基因型" and "data" in dir(data) and "top5基因型" in data:
                    # 尝试解析 `*1/*10` → NM（44.6%）
                    m = re.search(r"`([^`]+)`\s*→\s*(\S+)\s*[（(]([^）)]+)[）)]", content)
                    if m:
                        data["top5基因型"].append({
                            "基因型": m.group(1),
                            "表型": m.group(2),
                            "后验": m.group(3),
                        })
            continue

    return data


# ══════════════════════════════════════════════════════════════
# Agent 问答入口（核心入口 - 使用 v2.0 重构版 PGxAgent）
# ══════════════════════════════════════════════════════════════
def render_agent_qa(report):
    """AI 智能体：单入口 + 智能路由到 3 种回答类型

    用户在输入框提问题，AI 自动识别 → 路由到对应类型：
      类型 ①：基因型计算拆解（如 "*1/*10 是怎么算的"）
      类型 ②：假设表型重算（如 "如果 NM 怎么办"）
      类型 ③：DDI 速查（如 "可待因 + 阿米替林 合用行吗"）

    灰色提示文字列出 3 大类型的典型问法。
    """
    st.divider()
    st.subheader("🤖 AI 智能体 · 用药决策助手")

    # 灰色提示文字（3 大类型典型问法）
    st.markdown(
        r"""
<div style="color:#888; font-size:0.92em; line-height:1.7;">
💡 您可以问以下 3 类问题，AI 会自动识别并路由：

&nbsp;&nbsp;• 🔍 **基因型计算拆解**：例如 "\*1/\*10 是怎么算出来的？概率 30.6% 从哪来？"

&nbsp;&nbsp;• 🔄 **假设表型重算**：例如 "如果该病人实际是 NM，该用什么药？"

&nbsp;&nbsp;• 💊 **DDI 速查**：例如 "可待因 + 阿米替林 + 氯吡格雷 合用安全吗？"
</div>
        """,
        unsafe_allow_html=True,
    )

    # 单输入框
    user_q = st.text_input(
        "💬 您的提问：",
        key="agent_question_input_main",
        placeholder="例如：*1/*10 是怎么算出来的？",
    )

    send_btn = st.button(
        "🚀 发送给 AI",
        type="primary",
        use_container_width=False,
        key="agent_send_btn",
    )

    if send_btn and user_q:
        # 智能路由
        route_type, params = _route_user_question(user_q, report)
        st.markdown("---")
        st.caption(f"🤖 AI 已识别您的问题 → **{route_type}**")

        if route_type == "genotype_breakdown":
            _render_genotype_calc_breakdown(
                report, params["gene"], params["genotype"]
            )
        elif route_type == "hypothetical_recalc":
            _render_hypothetical_drug_recs(
                report, params["d6"], params["c19"], is_overridden=True
            )
        elif route_type == "ddi_check":
            _render_ddi_report(params["meds"])
        else:
            st.warning(
                "⚠️ 未识别出明确的问题类型。"
                "请参考上方灰色提示，用以下方式提问：\n"
                "- *1/*10 是怎么算的\n"
                "- 如果 NM 怎么办\n"
                "- 可待因 + 阿米替林 合用行吗"
            )


def _route_user_question(question: str, report) -> tuple[str, dict]:
    """智能路由：根据用户问题关键字，路由到 3 种类型之一

    设计原则：**宽松匹配**——意思相近即可，不要求严格措辞。

    Returns:
        (route_type, params)
        route_type ∈ {"genotype_breakdown", "hypothetical_recalc", "ddi_check", "unknown"}
    """
    import re
    q = question.strip()
    q_lower = q.lower()

    # ════════════════════════════════════════════════
    # 类型 ①：基因型计算拆解（问 *1/*10 等具体基因型）
    # 任意含 *N/*N 格式的问题都算
    # ════════════════════════════════════════════════
    gt_match = re.search(r"\*\s*\d+\s*/\s*\*\s*\d+", q)
    if gt_match:
        gene = "CYP2D6"
        if "cyp2c19" in q_lower:
            gene = "CYP2C19"
        genotype = gt_match.group(0).replace(" ", "")
        return "genotype_breakdown", {"gene": gene, "genotype": genotype}

    # ════════════════════════════════════════════════
    # 类型 ②：假设表型重算（任意提到"如果/假设/假如 X 是某表型"）
    # ════════════════════════════════════════════════
    # 宽松匹配：表型词 + 上下文中出现"如果/假设/假如/若是/万一是"
    phenotype_pat = r"(PM|IM|NM|UM)"

    # 模式 1: "如果/假设/假如 + ... + 表型"（表型词在后面）
    cf_match = re.search(
        r"(?:如果|假设|假如|若是|万一是|要是|如果是|假设是|假设该|假设为|若是该|假如是|反过来|如果是反假这样|反事实)"
        r"[^，。,.\n]{0,30}"
        + phenotype_pat,
        q, re.IGNORECASE
    )
    # 模式 2: "X 表型" 在假设类语境中（前置"假设/如果"已匹配，捕获后置表型）
    if not cf_match:
        # 包含"假设/如果"且包含表型词
        if re.search(r"(如果|假设|假如|若是|反事实)", q):
            ph_match = re.search(phenotype_pat, q, re.IGNORECASE)
            if ph_match:
                cf_match = ph_match

    # 模式 3: "换成 NM" / "用 IM" / "按 UM 算"
    if not cf_match:
        cf_match = re.search(
            r"(?:换成|用|按|按着|按算|按着算|如果是|改用|改算|修改为|设为)"
            r"\s*(PM|IM|NM|UM)\s*(?:表型|型|代谢)?",
            q, re.IGNORECASE
        )

    if cf_match:
        hyp_ph = cf_match.group(1).upper() if cf_match.lastindex else cf_match.group(0).upper()
        # 从匹配中提取表型
        ph_in_match = re.search(phenotype_pat, cf_match.group(0), re.IGNORECASE)
        if ph_in_match:
            hyp_ph = ph_in_match.group(1).upper()
        return "hypothetical_recalc", {"d6": hyp_ph, "c19": hyp_ph}

    # ════════════════════════════════════════════════
    # 类型 ③：DDI 速查（任意提到 2+ 药品名）
    # ════════════════════════════════════════════════
    # 扩大药品词库 + 同义词映射
    DRUG_SYNONYMS = {
        "泰诺": "对乙酰氨基酚", "扑热息痛": "对乙酰氨基酚",
        "百忧解": "氟西汀", "赛乐特": "帕罗西汀",
        "左洛复": "舍曲林", "怡诺思": "文拉法辛",
        "波立维": "氯吡格雷", "泰嘉": "氯吡格雷",
        "洛赛克": "奥美拉唑", "耐信": "艾司奥美拉唑",
        "来得时": "甘精胰岛素", "美托洛尔": "美托洛尔",
        "倍他乐克": "美托洛尔", "阿托伐他汀": "阿托伐他汀",
        "立普妥": "阿托伐他汀", "华法林": "华法林",
    }
    drug_keywords = [
        # 中文药名（常见 CYP2D6/CYP2C19 底物/抑制剂）
        "可待因", "曲马多", "阿米替林", "文拉法辛", "帕罗西汀",
        "氟西汀", "舍曲林", "氯吡格雷", "奥美拉唑", "艾司西酞普兰",
        "西酞普兰", "伏立康唑", "他莫昔芬", "苯妥英钠", "丙戊酸",
        "美托洛尔", "卡马西平", "苯巴比妥", "利福平", "红霉素",
        "克拉霉素", "酮康唑", "氟康唑", "胺碘酮", "地尔硫卓",
        "维拉帕米", "西咪替丁", "奥美拉唑", "泮托拉唑", "兰索拉唑",
        # 英文名
        "codeine", "tramadol", "amitriptyline", "venlafaxine",
        "paroxetine", "fluoxetine", "sertraline", "clopidogrel",
        "omeprazole", "escitalopram", "voriconazole", "tamoxifen",
        "warfarin", "metoprolol", "rifampicin", "erythromycin",
    ]
    # 1. 直接匹配
    found_drugs = [d for d in drug_keywords if d in q or d.lower() in q_lower]
    # 2. 同义词匹配（用户说"泰诺" → 识别为"对乙酰氨基酚"）
    for synonym, real_name in DRUG_SYNONYMS.items():
        if synonym in q and real_name not in found_drugs:
            found_drugs.append(real_name)

    # 兜底：尝试从当前用药列表匹配（如果报告里有 current_meds）
    if len(found_drugs) < 2:
        current_meds = []
        try:
            current_meds = _extract_current_meds_list(report) or []
        except Exception:
            pass
        for med in current_meds:
            med_lower = med.lower()
            for d in drug_keywords:
                if d in med or d.lower() in med_lower:
                    if d not in found_drugs:
                        found_drugs.append(d)

    if len(found_drugs) >= 2:
        return "ddi_check", {"meds": found_drugs[:5]}  # 最多取 5 种防爆炸

    # ════════════════════════════════════════════════
    # 兜底：未知类型
    # ════════════════════════════════════════════════
    return "unknown", {}


def _render_genotype_calc_breakdown(
    report, gene: str, genotype: str
):
    """按指定基因型拆解计算过程（含 Top 5 横向对比）"""
    phenotypes = getattr(report, "phenotypes", {}) or {}
    phen = phenotypes.get(gene)
    if not phen:
        st.error(f"❌ {gene} 表型推断失败")
        return

    # 找到对应基因型的 idx
    target_idx = None
    for i, g in enumerate(phen.top_genotypes_detailed or []):
        if g.genotype == genotype:
            target_idx = i
            break

    if target_idx is None:
        st.warning(f"⚠️ 基因型 {genotype} 不在 Top 5 中")
        return

    g_detail = phen.top_genotypes_detailed[target_idx]
    breakdown_lr = phen.get_node_lr_breakdown(target_idx)

    # ── 计算先验拆解（HWE 公式） ──
    a1, a2 = g_detail.allele1, g_detail.allele2
    if a1 == a2:
        prior_formula = f"{a1}² = {g_detail.prior_prob:.1%}"
    else:
        prior_formula = f"2 × {a1} × {a2} = {g_detail.prior_prob:.1%}"

    # ── 组装 markdown ──
    md_lines = [
        f"🔍 **{gene} `{genotype}` ({g_detail.phenotype})** · 后验 **{g_detail.posterior_prob:.1%}**",
        "",
        f"**先验 P(G) = {g_detail.prior_prob:.1%}**",
        f"├─ 东亚人群 **{a1}** × **{a2}** → `{prior_formula}`",
        "",
    ]

    ev_idx = 1
    lr_values = []
    for lr_info in breakdown_lr:
        signal_name = lr_info.get("signal_name", "?")
        target_ph = lr_info.get("target_phenotype", "?")
        strength = lr_info.get("strength", "?")
        is_blood = lr_info.get("is_blood", False)
        lr = lr_info.get("lr", 1.0)

        kind = "血液" if is_blood else "临床"
        md_lines.append(
            f"**证据 {ev_idx}**（{kind}，{strength}）：{signal_name}"
        )
        md_lines.append(f"└─ LR = {lr:.2f}（指向 {target_ph}）")
        md_lines.append("")
        lr_values.append(lr)
        ev_idx += 1

    # ── 计算总结 ──
    if lr_values:
        lr_chain = " × ".join([f"{lr:.2f}" for lr in lr_values])
        md_lines.append("**🧮 计算**")
        md_lines.append(
            f"└─ {g_detail.prior_prob:.1%} × {lr_chain} "
            f"= **后验 {g_detail.posterior_prob:.1%}**"
        )
    else:
        md_lines.append("**🧮 计算**")
        md_lines.append(
            f"└─ {g_detail.prior_prob:.1%}（无适用证据）= **后验 {g_detail.posterior_prob:.1%}**"
        )

    st.markdown("\n".join(md_lines))

    # ── 横向对比：为什么这个基因型是最大值？ ──
    st.markdown("---")
    st.markdown("### 📊 为什么是这个基因型？（Top 5 横向对比）")

    # 收集所有 top5 的关键数据
    all_signals = (
        phen.matched_signals
        if hasattr(phen, "matched_signals") and phen.matched_signals
        else []
    )

    # 表格：基因型 | 先验 | 适用证据数 | 后验
    import pandas as pd
    table_rows = []
    for i, g in enumerate(phen.top_genotypes_detailed or []):
        # 该基因型的 LR 拆解
        g_breakdown = phen.get_node_lr_breakdown(i)
        evidence_count = len(g_breakdown)
        if evidence_count == 0:
            evidence_str = "无"
        elif evidence_count == 1:
            evidence_str = f"1 条 (LR={g_breakdown[0].get('lr', 1.0):.1f})"
        else:
            lrs = [b.get("lr", 1.0) for b in g_breakdown]
            evidence_str = f"{evidence_count} 条 (LR={lrs[0]:.1f}×…)"

        is_target = "★" if g.genotype == genotype else ""
        table_rows.append({
            "基因型": f"{is_target} `{g.genotype}`",
            "表型": g.phenotype,
            "先验": f"{g.prior_prob:.1%}",
            "适用证据": evidence_str,
            "后验": f"{g.posterior_prob:.1%}",
        })

    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── 解释：为什么是该基因型 ──
    has_evidence = len(breakdown_lr) > 0
    if has_evidence:
        reason = (
            f"该基因型先验 **{g_detail.prior_prob:.1%}**，"
            f"叠加 {len(breakdown_lr)} 条证据后保持最高。"
        )
    else:
        # 无证据的情况：强调"先验最高"是胜出原因
        other_prior_max = max(
            (g.prior_prob for g in (phen.top_genotypes_detailed or [])
             if g.genotype != genotype),
            default=0,
        )
        reason = (
            f"该基因型先验 **{g_detail.prior_prob:.1%}** 在 Top 5 中最高，"
            f"且无临床/血液证据指向其他表型。"
            f"对比：次位候选最高先验 = {other_prior_max:.1%}。"
        )
    st.info(f"💡 **{reason}**")

    # ── 信号汇总（如果有）──
    if all_signals:
        with st.expander(f"📋 本病例所有临床/血液信号（{len(all_signals)} 条）", expanded=False):
            sig_lines = []
            for sig in all_signals:
                sig_lines.append(
                    f"- **{sig.signal_name}** → 指向 {sig.target_phenotype}（强度：{sig.strength}）"
                )
            st.markdown("\n".join(sig_lines))


def _render_ddi_report(meds: list[str]):
    """DDI 检测并展示（确定性，调用 check_ddi）"""
    from src.interaction_checker import check_ddi

    try:
        report = check_ddi(meds)
    except Exception as e:
        st.error(f"DDI 检测失败: {e}")
        return

    # ── 顶部：💡 简短解释（与类型①风格一致）──
    st.markdown(
        f"💊 **{', '.join(meds)}** · 检测 {len(meds)} 种药品"
    )

    if not report.has_interactions:
        st.success(
            f"✅ 已检测 {len(meds)} 种药品，"
            f"**未发现**已收录的相互作用。"
        )
        st.caption(
            "💡 说明：DDI 数据库只收录 CYP2D6/CYP2C19 相关相互作用，"
            "非 PGx 药物未涵盖。"
        )
        return

    # ── 总览说明 ──
    high = len(report.high_risk)
    medium = len(report.medium_risk)
    low = len(report.low_risk)
    total = high + medium + low
    summary_lines = [
        f"💡 共发现 **{total} 对**相互作用：",
    ]
    if high > 0:
        summary_lines.append(f"- 🔴 高风险 **{high}** 对（建议避免联用）")
    if medium > 0:
        summary_lines.append(f"- 🟠 中等风险 **{medium}** 对（监测不良反应）")
    if low > 0:
        summary_lines.append(f"- 🟢 低风险 **{low}** 对（常规监测）")
    st.markdown("\n".join(summary_lines))
    st.markdown("---")

    # ── 高风险 ──
    if report.high_risk:
        st.markdown(
            f"#### 🔴 高风险（{len(report.high_risk)} 对）"
        )
        for it in report.high_risk:
            with st.container(border=True):
                st.markdown(
                    f"**💊 {it.drug_a} + {it.drug_b}** · "
                    f"_机制：{it.mechanism}_"
                )
                st.caption(f"⚠️ 临床后果：{it.clinical_consequence}")
                st.caption(f"📋 建议：{it.suggestion}")

    # ── 中风险 ──
    if report.medium_risk:
        st.markdown(
            f"#### 🟠 中等风险（{len(report.medium_risk)} 对）"
        )
        for it in report.medium_risk:
            with st.container(border=True):
                st.markdown(
                    f"**💊 {it.drug_a} + {it.drug_b}** · "
                    f"_机制：{it.mechanism}_"
                )
                st.caption(f"临床后果：{it.clinical_consequence}")
                st.caption(f"建议：{it.suggestion}")

    # ── 低风险 ──
    if report.low_risk:
        with st.expander(
            f"🟢 低风险（{len(report.low_risk)} 对）",
            expanded=False,
        ):
            for it in report.low_risk:
                st.markdown(
                    f"- **{it.drug_a} + {it.drug_b}** "
                    f"· _{it.mechanism}_"
                )


def _render_hypothetical_drug_recs(
    report, hyp_d6: str, hyp_c19: str, is_overridden: bool
):
    """用假设表型重跑 drug_recommender，显示与「当前用药建议」相同格式

    关键设计：
      - 完全复用现有 drug_recommender（确定性，无 LLM）
      - 显示带显著"假设表型"标记
      - 复用 _render_first_visit_advice 或 render_current_meds_summary 的卡片渲染逻辑
    """
    from src.drug_recommender import recommend_both_genes

    hyp_phenotypes = {
        "CYP2D6": hyp_d6,
        "CYP2C19": hyp_c19,
    }

    # ── 取当前用药（如果患者没用药，给"假设表型下的全清单"）──
    current_meds = _extract_current_meds_list(report)
    meds_input = current_meds if current_meds else None

    # ── 调 drug_recommender ──
    drug_recs = recommend_both_genes(hyp_phenotypes, meds_input)

    # ── 构造合成 phenotypes dict（给 _render_first_visit_advice 用）──
    from dataclasses import dataclass, field
    from typing import List

    @dataclass
    class _HypPhen:
        gene: str
        phenotype: str
        best_genotype: str = ""
        prior_prob: float = 0.0
        confidence: float = 0.0
        phenotype_posteriors: dict = field(default_factory=dict)
        organ_reasons: list = field(default_factory=list)
        activity_score: float = 2.0
        organ_factor: float = 1.0
        effective_phenotype: str = ""
        needs_detection: bool = False
        matched_signals: list = field(default_factory=list)
        top_genotypes: list = field(default_factory=list)

    hyp_phen_dict = {
        gene: _HypPhen(gene=gene, phenotype=phen)
        for gene, phen in hyp_phenotypes.items()
    }

    # ── 构造合成 report（复用 _render_first_visit_advice / 渲染函数）──
    @dataclass
    class _HypPatient:
        current_meds: list = field(default_factory=list)

    @dataclass
    class _HypReport:
        patient: object = field(default_factory=_HypPatient)
        phenotypes: dict = field(default_factory=dict)
        drug_recs: dict = field(default_factory=dict)

    hyp_report = _HypReport(
        patient=_HypPatient(current_meds=current_meds or []),
        phenotypes=hyp_phen_dict,
        drug_recs=drug_recs,
    )

    st.markdown("---")
    if is_overridden:
        st.markdown(
            f"### 🔄 假设表型下的用药建议 "
            f"（CYP2D6={hyp_d6}, CYP2C19={hyp_c19}）"
        )
        st.caption(
            "⚠️ 以下结果基于**手动设定的假设表型**，不反映真实推断。"
            "用于医生评估『如果是 NM/PM 怎么办』的反事实场景。"
        )
    else:
        st.markdown(
            "### 🔄 当前真实推断表型下的用药建议 "
            f"（CYP2D6={hyp_d6}, CYP2C19={hyp_c19}）"
        )

    # ── 顶部 💡 简短解释（与类型①风格一致）──
    med_count = len(current_meds) if current_meds else 0
    st.markdown(
        f"🔄 **CYP2D6={hyp_d6}, CYP2C19={hyp_c19}**"
        f" · {'有' if med_count else '无'}当前用药"
    )
    st.caption(
        f"💡 假设患者实际为 **{hyp_d6}/{hyp_c19}** 表型，"
        f"对其{'当前用药' if med_count else '所有 PGx 相关药品'}重新评估。"
        f"以下结果**不**反映真实推断，仅供反事实分析。"
    )
    st.markdown("---")

    # ── 复用现有渲染：分场景渲染 ──
    if current_meds:
        # 复诊场景：复用「当前用药检查 + 关键药品清单」格式
        _render_hypothetical_followup(hyp_report)
    else:
        # 首诊场景：复用三档清单
        _render_first_visit_advice(hyp_report)


def _render_hypothetical_followup(hyp_report):
    """复诊场景下，渲染假设表型的用药建议（精简版，避免与原模块重复）"""
    drug_recs = getattr(hyp_report, "drug_recs", {}) or {}
    phenotypes = getattr(hyp_report, "phenotypes", {}) or {}

    # ── 表型概览 ──
    phen_rows = []
    for gene in ["CYP2D6", "CYP2C19"]:
        phen = phenotypes.get(gene)
        if not phen or not phen.phenotype:
            continue
        prob = _phenotype_prob(phen, phen.phenotype)
        phen_rows.append({
            "gene": gene,
            "phenotype": phen.phenotype,
            "prob": prob,
        })

    if phen_rows:
        cols = st.columns(len(phen_rows))
        for col, row in zip(cols, phen_rows):
            with col:
                icon = _PHE_ICON.get(row["phenotype"], "⚪")
                st.markdown(
                    f"**{icon} {row['gene']} {row['phenotype']}**"
                    f"  · 后验 {row['prob']:.1%}"
                )

    st.markdown("---")

    # ── 收集药品推荐 ──
    raw_items = []
    for gene in ["CYP2D6", "CYP2C19"]:
        rec = drug_recs.get(gene)
        phen = phenotypes.get(gene)
        if not rec or not phen or not phen.phenotype:
            continue
        for r in (getattr(rec, "recommendations", []) or []):
            action, emoji = _decide_action_for_phenotype(phen.phenotype)
            sev = {"避免": "avoid", "换药": "avoid", "减量监测": "caution",
                   "维持+监测": "safe", "维持": "safe"}.get(action, "safe")
            raw_items.append({
                "drug": r.drug_name,
                "gene": gene,
                "phenotype": phen.phenotype,
                "action": action,
                "emoji": emoji,
                "grade": (r.recommendation_grade or "").upper(),
                "alternatives": (r.alternatives or "").strip(),
                "current_action": r.current_action or "",
                "_severity": sev,
            })

    # 关键修复：按药品名 dedup
    items = _merge_drug_items_by_name(raw_items)

    if not items:
        st.success("✅ 当前用药在假设表型下无需调整")
        return

    # ── 排序 + 渲染 ──
    items.sort(key=lambda x: (
        _ACTION_RANK.get(x["action"], 5),
        _GRADE_RANK.get(x["grade"], 4),
        x["drug"],
    ))

    import pandas as pd
    df = pd.DataFrame([
        {
            "药品": it["drug"],
            "表型": f"{it['gene']} {it['phenotype']}",
            "动作": f"{it['emoji']} {it['action']}",
            "CPIC": it["grade"] or "—",
            "替代": it["alternatives"] or "—",
        }
        for it in items[:8]
    ])
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    # ── 详情卡片（精简）──
    with st.expander("📋 详情卡片（点击展开）", expanded=False):
        for it in items[:8]:
            st.markdown(
                f"**💊 {it['drug']}** · {it['gene']} {it['phenotype']} · "
                f"{it['emoji']} {it['action']} · CPIC {it['grade']}"
            )
            if it["current_action"]:
                st.caption(f"📋 {it['current_action']}")
            if it["alternatives"]:
                st.caption(f"🔁 替代：{it['alternatives']}")
            st.markdown("---")


# ══════════════════════════════════════════════════════════════
# 已知基因型后的关键药品推荐（点击直接提问 + 触发 Agent）
# ══════════════════════════════════════════════════════════════

# 临床重点药品：按基因枚举。表型决定具体调整方向（PM/IM/NM/UM）。
KEY_DRUGS_BY_GENE = {
    "CYP2D6": [
        ("可待因",     "镇痛"),
        ("曲马多",     "镇痛"),
        ("阿米替林",   "抗抑郁"),
        ("文拉法辛",   "抗抑郁"),
        ("他莫昔芬",   "肿瘤"),
    ],
    "CYP2C19": [
        ("氯吡格雷",   "抗血小板"),
        ("艾司西酞普兰","抗抑郁"),
        ("奥美拉唑",   "PPI"),
    ],
}


def _generate_med_questions(report) -> list[tuple[str, str]]:
    """
    "已知基因型 · 关键药品推荐"按钮数据。

    仅当某个基因的 `phen.genotype` 非空（说明已经做过基因检测）时，
    才生成该基因的药品推荐按钮；否则跳过。

    返回 [(button_label, full_question), ...]
      - button_label: 短（≤12 字符），如 "可待因（PM）"
      - full_question: 完整自然语言问题，点击后自动提问 Agent
    """
    result: list[tuple[str, str]] = []
    phenotypes = getattr(report, "phenotypes", {}) or {}

    for gene, drugs in KEY_DRUGS_BY_GENE.items():
        phen = phenotypes.get(gene)
        if not phen:
            continue
        # 关键过滤：只有检测过的（即 genotype 非空）才显示
        bg = (getattr(phen, "genotype", "") or "").strip()
        if not bg:
            continue
        bp = phen.phenotype or "?"

        for drug, _category in drugs:
            label = f"{drug}（{bp}）"  # 短标签
            full_q = (
                f"已知 {gene} {bg} ({bp}) 表型，"
                f"使用{drug}的剂量和方案如何调整？有无替代药？"
            )
            result.append((label, full_q))

    return result


def _ask_med_callback(full_q: str, report) -> None:
    """
    药品推荐按钮回调：填入输入框 + 直接调用 Agent（一步到位）。
    与 _set_question_callback（仅填输入框）的区别：这里同时触发发送。
    """
    st.session_state["agent_question_input_main"] = full_q
    try:
        _call_new_agent(full_q, report)
    except Exception:
        # _call_new_agent 内部已经显示错误，这里吞掉避免双弹窗
        pass


def _generate_preset_questions(report) -> list[tuple[str, str]]:
    """
    根据当前病例的 best_genotype + 表型 + current_meds 动态生成预设问题。
    返回 [(button_label, full_question), ...]
      - button_label: 短的、视觉友好的文字（≤10 字符）
      - full_question: 完整的问题文本（点击后填入输入框）

    覆盖两类核心场景：
      A) 推理解释（Agent A）：为什么是这个基因型/表型？
      B) 用药决策（Agent B）：已知基因型后如何用药？
    """
    questions = []
    phenotypes = getattr(report, "phenotypes", {}) or {}
    current_meds_raw = getattr(report.patient, "current_meds", []) or []

    if isinstance(current_meds_raw, list):
        current_meds = [str(m).strip() for m in current_meds_raw if str(m).strip()]
    else:
        current_meds = [m.strip() for m in re.split(r"[,，;；\n]", str(current_meds_raw)) if m.strip()]

    # ── 每个基因生成 2 个问题：1 个推理 + 1 个用药 ──
    for gene in ["CYP2D6", "CYP2C19"]:
        phen = phenotypes.get(gene)
        if not phen or not phen.best_genotype:
            continue

        bg = phen.best_genotype
        bp = phen.phenotype

        # Q1: 推理类
        q1_full = f"为什么推断为 `{bg}`？"
        q1_btn = f"为什么 {bg}？"
        questions.append((q1_btn, q1_full))

        # Q2: 用药类（含当前用药）
        if current_meds:
            med = _extract_drug_name(current_meds[0])
            q2_full = f"已知 `{bg}` ({bp})，{med} 如何调整？"
            q2_btn = f"{bg} 用药建议"
            questions.append((q2_btn, q2_full))
        else:
            q2_full = f"已知 `{bg}` ({bp})，如何用药？"
            q2_btn = f"{bg} 用药建议"
            questions.append((q2_btn, q2_full))

    # 去重（按 full_question 去重，保留首个）
    seen = set()
    unique = []
    for btn, full in questions:
        if full not in seen:
            seen.add(full)
            unique.append((btn, full))

    # 限制最多 6 个
    return unique[:6]


def _extract_drug_name(med_str: str) -> str:
    """从 "可待因30mg Tid" 提取 "可待因"（去掉剂量+频次）"""
    if not med_str:
        return "此药"

    # 匹配第一个连续的汉字字符串
    m = re.match(r"^([一-鿿]+)", med_str.strip())
    if m:
        return m.group(1)

    # 英文药品（如 "codeine 30mg"）
    m = re.match(r"^([A-Za-z]+)", med_str.strip())
    if m:
        return m.group(1)

    return med_str.strip()


def _set_question_callback(question: str):
    """
    按钮点击回调：把预设问题填入 text_input。
    关键：必须用 text_input 的 key（agent_question_input_main）写入 session_state。
    """
    st.session_state["agent_question_input_main"] = question


def _format_agent_b_markdown(drug_text) -> str:
    """
    把 Agent B 的 JSON/字面量/字符串输出解析成"药品字典列表"。

    支持三种输入：
    1. dict / list（已是 Python 数据）→ 直接返回
    2. 严格 JSON 字符串 → json.loads
    3. Python 字面量字符串（单引号、Python repr 风格）→ ast.literal_eval

    解析失败 → 返回 None（调用方降级到原始文本显示）
    """
    import json as _json
    import ast

    if isinstance(drug_text, (list, dict)):
        return drug_text

    if not isinstance(drug_text, str):
        return None

    text = drug_text.strip()
    if not text:
        return None

    # 尝试严格 JSON
    try:
        parsed = _json.loads(text)
    except (_json.JSONDecodeError, TypeError):
        parsed = None

    # 尝试 Python 字面量（单引号、Python 风格 list[dict]）
    if parsed is None and (text.startswith("[") or text.startswith("{")):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None

    if parsed is None:
        return None

    # 兼容嵌套：取 drug_recommendations / 用药决策清单
    if isinstance(parsed, dict):
        if "drug_recommendations" in parsed:
            return parsed["drug_recommendations"]
        if "用药决策清单" in parsed:
            return parsed["用药决策清单"]
        return parsed
    return parsed


def _is_drug_in_current_meds(drug_name: str, current_meds: list[str]) -> bool:
    """
    双向子串匹配判断药品是否在患者当前用药中。
    与 _build_drug_kb_input 中的匹配规则一致，避免 LLM 误标"在用"。
    """
    if not current_meds or not drug_name:
        return False
    drug_l = drug_name.lower().strip()
    for med in current_meds:
        med_l = str(med).lower().strip()
        if not med_l:
            continue
        if drug_l in med_l or med_l in drug_l:
            return True
    return False


def _extract_current_meds(report) -> list[str] | None:
    """
    从 report.patient.current_meds 提取用药列表（归一化为 list[str]）。
    返回 None 时表示"无用药"或"提取失败"，调用方据此判断渲染模式。
    """
    if report is None or not hasattr(report, "patient"):
        return None
    raw = getattr(report.patient, "current_meds", None)
    if raw is None:
        return []
    if isinstance(raw, list):
        meds = [str(m).strip() for m in raw if str(m).strip()]
    else:
        meds = [
            m.strip() for m in re.split(r"[,，;；\n]", str(raw))
            if m.strip()
        ]
    return meds


def _render_drug_card(r: dict, current_meds: list[str] = None) -> None:
    """
    渲染单张药品卡片（streamlit container + markdown 混排）。

    Parameters
    ----------
    current_meds : list[str] | None
        患者当前真实用药列表（来自 report.patient）。
        传入后用此列表做 ground-truth 交叉验证，
        避免 LLM 错误标记药品"在用"——医疗场景下更安全。
    """
    # 提取字段（中英兼容）
    drug_name = r.get("药品") or r.get("drug") or r.get("drug_name") or "?"
    dose = r.get("剂量") or r.get("dose") or ""
    freq = r.get("频次") or r.get("freq") or ""
    dose_str = f" {dose} {freq}".strip()
    action = r.get("推荐行动") or r.get("action") or ""
    pheno = r.get("phenotype") or r.get("表型") or "?"
    cpic = r.get("CPIC建议") or r.get("cpic_recommendation") or ""
    alts = (r.get("替代药") or r.get("alternatives") or "").strip()
    monitoring = r.get("监测") or r.get("监测项目") or r.get("monitoring") or []
    evidence = r.get("依据") or r.get("cpic_guideline") or r.get("source") or ""
    reason = r.get("reason") or ""

    # ★ 关键修复：判断"是否在用"时，**优先用 report 的真实 current_meds 做 ground truth**
    # 之前 r.get("is_current_med", True) 默认 True → LLM 输出没带此字段时就误标"在用"
    # 现在：若有 current_meds，交叉验证；否则看 LLM 字段（默认 False 更保守）
    if current_meds is not None:
        is_current = _is_drug_in_current_meds(drug_name, current_meds)
    else:
        is_current = r.get("is_current_med", False)   # ← 默认 False 取代 True

    # 严重度配色 + 图标
    if action == "避免":
        icon, color, action_text = "🚫", "#d32f2f", "避免使用 / 换药"
    elif action == "换药":
        icon, color, action_text = "⚠️", "#f57c00", "换用其他药"
    elif action == "减量":
        icon, color, action_text = "⚠️", "#f57c00", "需减量"
    elif action == "维持+监测":
        icon, color, action_text = "ℹ️", "#1976d2", "维持+监测"
    else:
        icon, color, action_text = "·", "#666", action or "—"

    with st.container(border=True):
        # ── 标题行：药名 + 严重度 ──
        st.markdown(
            f"""<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
<span style="font-size:1.05em; font-weight:600;">💊 {drug_name}{dose_str}</span>
<span style="color:{color}; font-weight:600; font-size:0.95em;">{icon} {action_text}</span>
</div>""",
            unsafe_allow_html=True,
        )

        # ── 表型行 ──
        badge = "🟢 当前在用" if is_current else "🔎 按表型筛出"
        badge_color = "#1b5e20" if is_current else "#666"
        st.markdown(
            f"<span style='color:#666; font-size:0.85em;'>表型：{pheno} · </span>"
            f"<span style='color:{badge_color}; font-size:0.85em; font-weight:500;'>{badge}</span>",
            unsafe_allow_html=True,
        )

        # ── 建议（如果有 reason → 优先展示，简洁）──
        # ✅ 补丁 ⑤：复诊视图也用动态来源徽章替换硬编码 "CPIC 建议"
        src_short_revisit = _shorten_source(evidence) if evidence else "CPIC"
        if reason and not cpic:
            st.markdown(f"💡 **原因**：{reason}")
        elif cpic:
            st.markdown(f"**📋 {src_short_revisit} 建议**：{cpic}")
        if reason and cpic:
            st.markdown(f"💡 **原因**：{reason}")
        # ✅ 补丁 ⑤：把完整证据来源（含 Luca 2026 卷号）展示出来
        if evidence:
            st.caption(f"📚 **证据来源**：{evidence}")

        # ── 替代药 / 监测（双列）──
        col1, col2 = st.columns(2)
        with col1:
            if alts:
                st.markdown(f"**替代药**\n\n{alts}")
        with col2:
            if monitoring:
                if isinstance(monitoring, list):
                    monitoring = " · ".join(monitoring)
                st.markdown(f"**监测**\n\n{monitoring}")

        # ── 依据 ──
        if evidence:
            st.caption(f"📚 依据：{evidence[:120]}{'…' if len(evidence) > 120 else ''}")


def _render_agent_b_markdown(drug_text, current_meds: list[str] = None) -> None:
    """
    Agent B 输出渲染器（卡片化版）。
    直接用 st.* API 渲染，不再返回字符串。

    Parameters
    ----------
    current_meds : list[str] | None
        患者当前真实用药，传给每张卡片做交叉验证（防 LLM 误标"在用"）。
    """
    # ── Step 1: 解析数据 ──
    data = _format_agent_b_markdown(drug_text)

    # ── Step 2: 解析失败 → 降级到原始文本 + 友好提示 ──
    if data is None:
        raw = drug_text if isinstance(drug_text, str) else str(drug_text)
        st.warning("⚠️ LLM 返回格式无法解析，已降级到原始响应：")
        st.code(raw, language="text")
        return

    if isinstance(data, dict) and not isinstance(data, list):
        data = [data]
    if not isinstance(data, list) or not data:
        st.info("（无药品推荐）")
        return

    # ── Step 3: 摘要卡片 ──
    n_total = len(data)
    n_avoid = sum(
        1 for r in data
        if isinstance(r, dict) and r.get("推荐行动") == "避免"
    )
    n_adjust = sum(
        1 for r in data
        if isinstance(r, dict) and r.get("推荐行动") in ("换药", "减量")
    )
    n_monitor = sum(
        1 for r in data
        if isinstance(r, dict) and r.get("推荐行动") == "维持+监测"
    )

    st.markdown(
        f"""<div style="background:linear-gradient(90deg,#e3f2fd 0%,#bbdefb 100%);
padding:10px 16px; border-radius:8px; margin:6px 0 12px 0; font-size:0.95em;">
<strong>📊 用药决策汇总</strong>　|
<span style="color:#d32f2f; font-weight:bold;">🚫 避免 {n_avoid}</span>
<span style="color:#f57c00; font-weight:bold;">⚠️ 调整 {n_adjust}</span>
<span style="color:#1976d2;">ℹ️ 监测 {n_monitor}</span>
<span style="color:#888;">共 {n_total} 项</span>
</div>""",
        unsafe_allow_html=True,
    )

    # ── Step 4: 排序（按严重度）──
    SEV = {"避免": 0, "换药": 1, "减量": 2, "维持+监测": 3}
    GRADE = {"A": 0, "B": 1, "C": 2, "D": 3}
    def _sev(r):
        if not isinstance(r, dict):
            return (9, 9)
        a = r.get("推荐行动") or ""
        g = r.get("证据等级") or r.get("cpic_grade") or "D"
        return (SEV.get(a, 9), GRADE.get(g, 9))
    data = sorted(data, key=_sev)

    # ── Step 5: 渲染每张药品卡片 ──
    for r in data:
        if not isinstance(r, dict):
            continue
        _render_drug_card(r, current_meds=current_meds)


def _call_new_agent(user_question: str, report):
    """调用新的 v2.0 PGxAgent"""
    agent = _get_agent()
    if agent is None:
        st.error(
            "❌ DeepSeek 未配置，无法调用 Agent。\n\n"
            "请在 `.env` 文件设置 `DEEPSEEK_API_KEY=sk-xxx`，"
            "或在「环境变量」中配置。\n\n"
            "（注：Agent 仍可在 fallback 模式下提供确定性输出，"
            "但您需要先用下面的「Agent 问答入口（无 API）」验证）"
        )
        return

    phenotypes = {
        gene: report.phenotypes[gene].phenotype
        for gene in ["CYP2D6", "CYP2C19"]
        if gene in report.phenotypes
    }
    current_meds = getattr(report.patient, "current_meds", [])

    # 构造 inference_result_dict（给 agent_a 用）
    # 关键：键名用中文，与 AGENT_A_SYSTEM_PROMPT 的输出 Schema 完全对齐
    inference_result_dict = None
    if "CYP2D6" in report.phenotypes:
        phen = report.phenotypes["CYP2D6"]
        bg_conf = getattr(phen, "best_genotype_confidence", 0.0) or 0.0
        top_g_list = list(phen.top_genotypes or [])

        # 临床证据（从 matched_signals 提取）
        clinical_signals = [
            {
                "证据描述": s.signal_name,
                "指向表型": s.target_phenotype,
                "强度": s.strength,
                "解释": s.interpretation,
            }
            for s in (phen.matched_signals or [])
            if not s.signal_name.startswith("[CCA")
        ]

        # 血液证据（CCA 信号）
        blood_signals = [
            {
                "标记物": s.signal_name.replace("[CCA·", "").replace("]", "").strip(),
                "实际值": getattr(s, "signal_text", ""),
                "指向表型": s.target_phenotype,
                "影响": s.interpretation,
            }
            for s in (phen.matched_signals or [])
            if s.signal_name.startswith("[CCA")
        ]

        # 建议文字
        if bg_conf >= 0.85 and phen.phenotype not in ("PM", "UM"):
            advice_text = "可按推断表型用药，无需基因检测"
        elif bg_conf >= 0.65:
            advice_text = "可按推断表型用药，建议测 SNP 复核"
        else:
            advice_text = "建议基因检测以确定基线"

        inference_result_dict = {
            # ── 结论部分（直接给 LLM 用）──
            "基因": "CYP2D6",
            "最可能基因型": phen.best_genotype or "系统未提供",
            "基因型表型": phen.phenotype or "系统未提供",
            "有效代谢": getattr(phen, "effective_phenotype", "") or phen.phenotype or "系统未提供",
            "置信度": f"{bg_conf:.1%}",
            "建议": advice_text,
            # ── 先验部分 ──
            "人群": phen.population or "东亚",
            "该基因型先验": f"{phen.prior_prob:.1%}" if phen.prior_prob else "系统未提供",
            # ── Top 5 基因型 ──
            "top5基因型": [
                {"基因型": g, "表型": ph, "后验": f"{post:.1%}"}
                for g, ph, post in top_g_list[:5]
            ],
            # ── 证据 ──
            "临床证据": clinical_signals,
            "血液证据": blood_signals,
            # ── 不确定性 ──
            "置信区间": "系统未提供",
            "封顶规则": "系统未提供",
            # ── 溯源 ──
            "数据来源": "clinical_signals.csv + cca_rules.csv",
            "计算依据": "HWE 人群先验 + 贝叶斯更新 + activity score (CPIC 2019)",
            "引擎版本": "genotype_inferrer v2.1",
        }

    result = agent.reason(
        user_question=user_question,
        phenotypes=phenotypes,
        current_meds=current_meds,
        inference_result_dict=inference_result_dict,
    )

    st.session_state["agent_v2_result"] = result
    # 注：不修改 text_input widget state（Streamlit 限制）
    # 用户可手动清除或输入下一个问题覆盖


def _render_agent_v2_result(result, report=None):
    """渲染 v2.0 Agent 结果

    Parameters
    ----------
    report : Optional
        用于把"患者真实当前用药"传给 Agent B 渲染层做交叉验证，
        避免 LLM 输出中没带 is_current_med 字段时默认误标"在用"。
    """
    st.markdown("---")

    # Agent 标识（用户友好命名，隐藏内部 A/B/C）
    agent_icons = {"A": "🔍", "B": "💊", "C": "🛡️", "TEST": "🧪"}
    agent_names = {
        "A": "AI 回答（基于您的病例推理结果）",
        "B": "AI 回答（基于本地药品知识库）",
        "C": "AI 提示",
        "TEST": "检测推荐",
    }
    icon = agent_icons.get(result.agent, "🤖")
    name = agent_names.get(result.agent, "AI 回答")

    st.markdown(f"### {icon} {name}")

    # Fallback 警告（合并：forbidden 时不再重复显示通用 fallback 提示）
    if result.used_fallback and not result.forbidden_phrases:
        st.warning(f"⚠️ **Fallback 模式** · {result.fallback_reason}")
    if result.forbidden_phrases:
        st.error(
            f"⛔ **AI 输出不合规**：{result.forbidden_phrases}\n\n"
            "已自动替换为基于本地知识库的标准答案。"
        )

    # 输出内容
    output = result.output

    if result.agent == "C":
        # 边界守卫：显示拒绝模板
        if "refusal_text" in output:
            st.info(output["refusal_text"])
        else:
            st.json(output)

    elif result.agent == "TEST":
        # 检测推荐：直接 markdown
        if "test_recommendation" in output:
            st.markdown(output["test_recommendation"])
        else:
            st.json(output)

    elif result.agent == "B":
        # 用药决策：直接交给卡片化渲染器（自带 JSON / ast.literal_eval 兜底）
        drug_recs_text = output.get("drug_recommendations", "")
        # 提取患者真实当前用药 → 交叉验证"是否在用"，避免 LLM 误标
        current_meds = _extract_current_meds(report) if report else None
        _render_agent_b_markdown(drug_recs_text, current_meds=current_meds)
        # 显示原始响应
        if result.raw_response:
            with st.expander("📋 LLM 原始响应", expanded=False):
                st.code(result.raw_response, language="json")
        # 显示结构化数据
        if "structured_data" in output:
            with st.expander("📋 结构化数据（LLM 看到的 JSON）", expanded=False):
                st.json(output["structured_data"])

    elif result.agent == "A":
        # 推理解释：LLM 返回 JSON/结构化 → 走格式化；fallback 返回已排版的 markdown → 直接显示
        explanation = output.get("explanation", "")
        if explanation:
            if output.get("_skip_format"):
                # fallback 路径：模板已经是排版好的 markdown，直接渲染避免数据被解析成 "?"
                st.markdown(explanation)
            else:
                formatted = _format_agent_a_markdown(explanation)
                st.markdown(formatted)
        else:
            st.info("（无输出）")
        # 显示原始响应（可折叠）
        if result.raw_response:
            with st.expander("📋 LLM 原始响应", expanded=False):
                st.code(result.raw_response, language="json")

    # 数据来源
    if "data_source" in output:
        st.caption(f"📌 数据来源：{output['data_source']}")


# ══════════════════════════════════════════════════════════════
# 推断治疗方案（无基因检测时）
# ══════════════════════════════════════════════════════════════
def render_inferred_treatment_plans(report):
    """
    在没有基因检测结果时，基于**推断表型**给出 3 套治疗方案。

    与"基因检测后用药推荐"互补：
    - 本模块：未检测 → 3 套粗略方案（plan1/2/3）
    - 检测后模块：已检测 → 单药精确推荐 + CPIC 等级 + 证据

    方案：
      1. 当前用药 + 调整（基于现有基因型推断）
      2. 换主药（用 CYP2D6/CYP2C19 影响小的药）
      3. 非 PGx 选项（CBT 等）
    """
    st.divider()
    st.subheader("📋 推断治疗方案（无基因检测）")
    st.caption(
        "💡 **未做基因检测**时，基于**推断表型**给出的 3 套治疗方案。"
        "如已有基因检测，请用下方「🧬 基因检测后」模块获得**精确**推荐（CPIC 等级 + 证据）。"
    )

    drug_recs = getattr(report, "drug_recs", {}) or {}
    phenotypes = getattr(report, "phenotypes", {}) or {}

    # ── 方案 1：当前用药 + 调整（基于 drug_recs.prescription_eval）──
    plan1_drugs = []
    for gene in ["CYP2D6", "CYP2C19"]:
        rec = drug_recs.get(gene)
        if not rec or not getattr(rec, "prescription_eval", None):
            continue
        for ev in rec.prescription_eval.eval_details:
            if ev.drug_name:
                plan1_drugs.append({
                    "drug": ev.drug_name,
                    "dose": getattr(ev, "dose", ""),
                    "action": "维持（调剂量）" if ev.is_appropriate else "调整",
                })
                if ev.alternative_details:
                    for alt in ev.alternative_details:
                        plan1_drugs.append({
                            "drug": alt.drug_name,
                            "dose": getattr(alt, "dose", ""),
                            "action": "替代选项",
                        })

    # ── 方案 2：换主药（CYP2D6/CYP2C19 影响小）──
    plan2_drugs = [
        {"drug": "舍曲林", "dose": "100mg QD", "action": "替换"},
        {"drug": "米氮平", "dose": "15mg QN", "action": "联合"},
    ]

    # ── 方案 3：非 PGx 选项 ──
    plan3_drugs = [
        {"drug": "CBT 心理治疗", "dose": "12 周", "action": "非药物"},
    ]

    # 等级（基于 CYP2D6 主表型）
    main_phen = phenotypes.get("CYP2D6")
    if main_phen and main_phen.phenotype == "PM":
        plan1_grade = "B（PM 时减量/换药）"
        plan2_grade = "B（避免 CYP2D6 底物）"
        plan3_grade = "A（不依赖 CYP 代谢）"
    else:
        plan1_grade = "A（维持原方案）"
        plan2_grade = "B（备选方案）"
        plan3_grade = "A（非药物选项）"

    plans = [
        {
            "name": "方案 1：当前用药 + 调整",
            "drugs": plan1_drugs or [{"drug": "见报告", "dose": "", "action": ""}],
            "grade": plan1_grade,
            "rationale": "基于现有基因型推断调整",
        },
        {
            "name": "方案 2：换主药（CYP2D6/CYP2C19 影响小）",
            "drugs": plan2_drugs,
            "grade": plan2_grade,
            "rationale": "避免 CYP2D6/CYP2C19 底物累积",
        },
        {
            "name": "方案 3：非 PGx 替代",
            "drugs": plan3_drugs,
            "grade": plan3_grade,
            "rationale": "不依赖 CYP 代谢的选项",
        },
    ]

    for plan in plans:
        with st.expander(f"### 💊 {plan['name']}", expanded=False):
            st.markdown(f"**推荐等级**: {plan['grade']}")
            st.markdown(f"**依据**: {plan['rationale']}")
            st.markdown("**药方组成**:")
            for i, drug in enumerate(plan["drugs"], 1):
                cols = st.columns([1, 3, 2])
                with cols[0]:
                    st.markdown(f"**{i}.**")
                with cols[1]:
                    st.markdown(f"**{drug['drug']}**")
                with cols[2]:
                    st.markdown(f"{drug['dose']} · {drug['action']}")


# ══════════════════════════════════════════════════════════════
# 患者文本构造（Agent 用）
# ══════════════════════════════════════════════════════════════
def _build_patient_text(report) -> str:
    """从 report 构造患者描述"""
    p = report.patient
    parts = [
        f"诊断：{p.diagnosis or '未填'}",
    ]
    if p.current_meds:
        meds = ", ".join(p.current_meds) if isinstance(p.current_meds, list) else str(p.current_meds)
        parts.append(f"当前用药：{meds}")
    if p.chief_complaint:
        parts.append(f"主诉：{p.chief_complaint}")
    if p.age:
        parts.append(f"年龄：{p.age}")
    if p.gender:
        parts.append(f"性别：{p.gender}")
    if p.population:
        parts.append(f"人群：{p.population}")
    if report.phenotypes:
        pheno_strs = []
        for gene, info in report.phenotypes.items():
            pheno_strs.append(f"{gene}={info.phenotype}")
        if pheno_strs:
            parts.append(f"推断表型：{', '.join(pheno_strs)}")
    return "。".join(parts) + "。"


def _run_agent_inference(patient_text: str, report):
    """调用 Agent 推理（带 fallback）"""
    with st.spinner("Agent 推理中（DeepSeek + 知识库）..."):
        agent = _get_agent()
        if agent is None:
            # 降级到规则检索（不调 API）
            kb = pd.read_csv(ROOT / "data" / "agent_knowledge_base.csv")
            keywords = ["CYP2D6", "CYP2C19", "氯吡格雷", "可待因", "文拉法辛", "阿米替林"]
            rules = kb[kb["content"].str.contains("|".join(keywords), na=False)].head(10)
            st.session_state["agent_result"] = {
                "mode": "rules_only",
                "rules": rules.to_dict("records"),
                "message": "⚠️ 未配置 DeepSeek API key，仅展示知识库规则"
            }
            return

        # 从 report 抽取结构化 DDI（注入到 LLM context）
        ddi_alerts = _extract_ddi_alerts(report)

        try:
            lab_results = st.session_state.get("lab_results", {})
            result = agent.reason(
                patient_text,
                lab_results=lab_results,
                ddi_alerts=ddi_alerts,
            )
            st.session_state["agent_result"] = result
        except Exception as e:
            st.error(f"Agent 调用失败: {e}")


def _extract_ddi_alerts(report) -> list[dict]:
    """
    从 report.ddi_report 抽取 DDI 列表，转成 LLM 易读的 dict。

    返回 list of dict，每个 dict 含:
        drug_a, drug_b, gene, mechanism, warning_level,
        clinical_consequence, pm_action, source
    """
    try:
        interactions = report.ddi_report.interactions
    except (AttributeError, IndexError):
        return []

    out = []
    for d in interactions:
        # DDIResult 是 dataclass；getattr 拿不到时给空串（不报错）
        out.append({
            "drug_a": getattr(d, "drug_a", "?"),
            "drug_b": getattr(d, "drug_b", "?"),
            "gene": getattr(d, "gene", ""),
            "mechanism": getattr(d, "mechanism", ""),
            "warning_level": getattr(d, "warning_level", "?"),
            "clinical_consequence": getattr(d, "clinical_consequence", ""),
            "pm_action": getattr(d, "suggestion", ""),   # DDIResult 用 suggestion
            "source": getattr(d, "source", ""),
        })
    return out


def _render_agent_result(result: dict):
    """渲染 Agent 输出（新 schema：治疗方案导向）"""
    if not result:
        return

    st.divider()

    # ── 模式提示（降级 / 错误）──
    if result.get("mode") == "rules_only":
        st.warning(result.get("message", "未配置 DeepSeek API key"))
    if result.get("error"):
        st.error(f"Agent 调用失败：{result['error']}")

    # ── 1. 首选推荐（核心）──
    primary = result.get("primary_recommendation")
    if primary and isinstance(primary, dict) and primary.get("drug"):
        drug = primary.get("drug", "")
        action = primary.get("action", "")
        reasoning = primary.get("reasoning", "")
        rule_id = primary.get("rule_id", "")
        ev_level = primary.get("evidence_level", "")

        # 用 success/info 提示框强化"这是首选"
        with st.container():
            st.markdown("### 🎯 首选推荐")
            cols = st.columns([3, 2, 1])
            with cols[0]:
                st.markdown(f"**{drug}**")
                if action:
                    st.markdown(f"_{action}_")
            with cols[1]:
                if reasoning:
                    st.caption(reasoning)
            with cols[2]:
                if rule_id:
                    st.markdown(f"`{rule_id}`")
                if ev_level:
                    st.caption(ev_level)

    # ── 2. 备选药物 ──
    alts = result.get("alternatives", [])
    if alts:
        st.markdown("**🔁 备选药物**")
        for alt in alts:
            if not isinstance(alt, dict):
                continue
            drug = alt.get("drug", "")
            when = alt.get("when", "")
            rule_id = alt.get("rule_id", "")
            if drug:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"- **{drug}** — {when}")
                with col2:
                    if rule_id:
                        st.caption(f"`{rule_id}`")

    # ── 3. 监测计划 ──
    monitors = result.get("monitoring_plan", [])
    if monitors:
        st.markdown("**📋 监测计划**")
        for m in monitors:
            if not isinstance(m, dict):
                continue
            item = m.get("item", "")
            freq = m.get("frequency", "")
            rationale = m.get("rationale", "")
            if item:
                with st.expander(f"📊 {item}（{freq}）", expanded=False):
                    if rationale:
                        st.caption(rationale)

    # ── 4. DDI 警告 ──
    ddis = result.get("ddi_alerts", [])
    if ddis:
        st.markdown("**⚠️ 药物相互作用（DDI）**")
        for d in ddis:
            if not isinstance(d, dict):
                continue
            pair = d.get("pair", "")
            sev = d.get("severity", "")
            action = d.get("action", "")
            if pair:
                if sev == "高":
                    st.error(f"**{pair}**（高） — {action}")
                elif sev == "中":
                    st.warning(f"**{pair}**（中） — {action}")
                else:
                    st.info(f"**{pair}**（{sev}） — {action}")

    # ── 5. 随访 ──
    follow_up = result.get("follow_up")
    if follow_up and follow_up != "知识库未覆盖":
        st.markdown("**📅 随访计划**")
        st.info(follow_up)

    # ── 6. 推理总结 ──
    summary = result.get("reasoning_summary")
    if summary:
        st.markdown("**📝 推理总结**")
        st.success(summary)  # 用 success（绿色）突出最终结论

    # ── 7. 置信度 ──
    conf = result.get("confidence")
    if conf:
        if conf == "高":
            st.success(f"✅ 置信度：**高**（≥80%）")
        elif conf == "中":
            st.info(f"ℹ️ 置信度：**中**（60-80%）")
        elif conf == "低":
            st.warning(f"⚠️ 置信度：**低**（<60%）— 建议基因检测")
        else:
            st.caption(f"置信度：{conf}")

    # ── 元数据（可追溯引用）──
    if "_meta" in result:
        meta = result["_meta"]
        n_rules = meta.get("retrieved_rules_count", 0)
        rules_used = meta.get("rules_used", [])
        with st.expander(f"📚 检索到 {n_rules} 条知识库规则（引用前 10 条）", expanded=False):
            for rid in rules_used:
                st.markdown(f"- `{rid}`")


# 测试
if __name__ == "__main__":
    print("此模块应被 pages/2_分析报告.py 导入，不直接运行")