"""
pages/1_病例录入.py — Step 7
病例信息录入页面
"""

import streamlit as st
import sys
import re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="病例录入 · PGx CDS", page_icon="📝", layout="wide")

# ── Session state 安全初始化 ──────────────────────────────────
for _k, _v in {"case_id": "", "diagnosis": "", "current_meds": [], "population": "东亚",
                "chief_complaint": "", "age": None, "gender": "", "signals_cyp2d6": "",
                "signals_cyp2c19": "", "report": None, "report_dict": None,
                "html_report": None, "history": [], "submitted": False,
                "_form_ver": 0}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── 页面标题 ─────────────────────────────────────────────────
st.title("📝 病例录入")
st.markdown("填写患者基本信息与临床检测信号，系统将自动推断代谢表型并生成 CDS 报告。")
st.divider()

# ── 辅助函数 ─────────────────────────────────────────────────
def _meds_to_list(raw: str) -> list[str]:
    """将逗号/换行分隔的药物字符串转为列表"""
    items = raw.replace("\n", ",").replace("，", ",").split(",")
    return [x.strip() for x in items if x.strip()]


# ── 病历文本自动解析 ─────────────────────────────────────────
def _parse_clinical_text(text: str) -> dict:
    """
    从自由文本病历中提取 PGx 相关字段。
    支持格式示例：
      "患者男性，58岁，诊断抑郁症，曾用舍曲林、可待因"
      "CYP2D6: *10/*10, PM型"
      "CYP2C19 *1/*1 NM型"
      "CYP2C19：*1/*17  ultra-rapid metabolizer"
    """
    if not text or not text.strip():
        return {}

    text_raw = text
    text = text.replace("\n", " ").strip()

    result: dict = {}

    # ── 病例 ID ─────────────────────────────────────────────────
    cid_m = re.search(r"病例\s*ID[:：]?\s*([A-Za-z0-9_\-]+)", text, re.IGNORECASE)
    if cid_m:
        result["case_id"] = cid_m.group(1).strip()

    # ── 诊断 ───────────────────────────────────────────────────
    diag_patterns = [
        r"诊断[为:：]?\s*([^\s,，。；]+)",
        r"患有?\s*([^\s,，。；]+)",
        r"主要诊断[为:：]?\s*([^\s,，。；]+)",
        r"印象[为:：]?\s*([^\s,，。；]+)",
        r"出院诊断[为:：]?\s*([^\s,，。；]+)",
    ]
    for pat in diag_patterns:
        m = re.search(pat, text)
        if m:
            result["diagnosis"] = m.group(1).strip()
            break

    # ── 用药 ───────────────────────────────────────────────────
    meds = []
    # 模式1：服用/使用/用药 后面跟药物名
    med_phrases = re.findall(
        r"(?:服用|使用|应用|用药|既往用|曾用|现用|服|给予)[药物]*(?:舍曲林|可待因|奥美拉唑|"
        r"文拉法辛|度洛西汀|阿米替林|帕罗西汀|氟西汀|艾司西酞普兰|"
        r"曲唑酮|米氮平|喹硫平|奥氮平|利培酮|阿普唑仑|地西泮|劳拉西泮|"
        r"可待因|吗啡|芬太尼|羟考酮|曲马多|布洛芬|萘普生|阿司匹林|"
        r"氯吡格雷|质子泵抑制剂|泮托拉唑|兰索拉唑|雷贝拉唑|艾司奥美拉唑|"
        r"[^\s,，。；]{2,10})",
        text,
    )
    # 模式2：直接列举的药物名列表（括号内或逗号分隔）
    known_drugs = [
        "舍曲林","可待因","奥美拉唑","文拉法辛","度洛西汀","阿米替林",
        "帕罗西汀","氟西汀","艾司西酞普兰","曲唑酮","米氮平","喹硫平",
        "奥氮平","利培酮","阿普唑仑","地西泮","劳拉西泮","吗啡","芬太尼",
        "羟考酮","曲马多","布洛芬","萘普生","阿司匹林","氯吡格雷",
        "泮托拉唑","兰索拉唑","雷贝拉唑","艾司奥美拉唑","西酞普兰",
        "托莫西汀","安非他酮","伏硫西汀","阿立哌唑","齐拉西酮",
    ]
    for drug in known_drugs:
        if drug in text and drug not in meds:
            meds.append(drug)
    result["meds"] = meds

    # ── CYP2D6 基因型 ─────────────────────────────────────────
    cyp2d6_patterns = [
        r"CYP2D6[：:\s]*[*]?(\d+[*/]\d+[,，]?\s*(?:PM|IM|NM|UM|ultra.?rapid|poor|intermediate|normal)?)",
        r"CYP2D6[*\s]+(\d+[*/]\d+)",
        r"(\*?\d+[*/]\d+\*?\d*)\s*,?\s*(?:PM|IM|NM|UM|慢代谢|中间代谢|正常代谢|超快代谢)",
        r"(?:PM|IM|NM|UM|慢代谢|中间代谢|正常代谢|超快代谢)[者型]*\s*,?\s*CYP2D6[*\s]*(\d+[*/]\d+)",
    ]
    for pat in cyp2d6_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw = m.group(0)
            # 提取基因型部分
            gt_m = re.search(r"\*?\d+[*/]\d+[*\d]*", raw, re.IGNORECASE)
            if gt_m:
                result["signals_cyp2d6"] = f"CYP2D6: {gt_m.group(0)}"
            # 提取表型
            ph_m = re.search(
                r"(?:PM|IM|NM|UM|ultra.?rapid|poor|intermediate|normal|慢代谢|中间代谢|正常代谢|超快代谢)",
                raw, re.IGNORECASE
            )
            if ph_m and "signals_cyp2d6" in result:
                result["signals_cyp2d6"] += f", {ph_m.group(0)}"
            break

    # ── CYP2C19 基因型 ─────────────────────────────────────────
    cyp2c19_patterns = [
        r"CYP2C19[：:\s]*[*]?(\d+[*/]\d+[,，]?\s*(?:PM|IM|NM|UM|ultra.?rapid|poor|intermediate|normal)?)",
        r"CYP2C19[*\s]+(\d+[*/]\d+)",
        r"(\*?\d+[*/]\d+\*?\d*)\s*,?\s*(?:PM|IM|NM|UM|ultra.?rapid|poor|intermediate|normal|超快代谢|慢代谢)",
        r"(?:PM|IM|NM|UM|慢代谢|中间代谢|正常代谢|超快代谢)[者型]*\s*,?\s*CYP2C19[*\s]*(\d+[*/]\d+)",
    ]
    for pat in cyp2c19_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw = m.group(0)
            gt_m = re.search(r"\*?\d+[*/]\d+[*\d]*", raw, re.IGNORECASE)
            if gt_m:
                result["signals_cyp2c19"] = f"CYP2C19: {gt_m.group(0)}"
            ph_m = re.search(
                r"(?:PM|IM|NM|UM|ultra.?rapid|poor|intermediate|normal|慢代谢|中间代谢|正常代谢|超快代谢)",
                raw, re.IGNORECASE
            )
            if ph_m and "signals_cyp2c19" in result:
                result["signals_cyp2c19"] += f", {ph_m.group(0)}"
            break

    # ── 人群 ───────────────────────────────────────────────────
    pop_map = {
        "东亚": ["东亚", "亚裔", "中国", "汉族", "华人", "Asian", "East Asian"],
        "欧洲": ["欧洲", "白人", "高加索", "European", "Caucasian"],
        "非洲裔": ["非洲", "非裔", "African", "Black"],
        "拉美裔": ["拉美", "拉丁裔", "Hispanic", "Latino"],
        "南亚": ["南亚", "印度", "South Asian"],
    }
    for pop, keywords in pop_map.items():
        for kw in keywords:
            if kw.lower() in text.lower():
                result["population"] = pop
                break
        if "population" in result:
            break

    # ✅ 新增：自动从病历正文抽取临床信号短语
    # 之前：解析器只抽基因型，临床信号短语（如"氟哌啶醇EPS或急性肌张力障碍"）
    #       需要用户手动复制粘贴到 "CYP2D6 信号" 文本框，否则表型推断拿不到 PM/IM 证据
    # 现在：扫描 clinical_signals.csv 的 signal_name 列，凡在病历正文里出现过的就
    #       自动追加到 signals_cyp2d6 / signals_cyp2c19（换行分隔）
    try:
        from src.data_loader import get_cache
        _cache = get_cache()
        _sig_tables = {
            "CYP2D6": getattr(_cache, "clinical_signals", None),
            "CYP2C19": getattr(_cache, "clinical_signals_cyp2c19", None),
        }
        for gene, _sig_df in _sig_tables.items():
            if _sig_df is None or _sig_df.empty:
                continue
            key = f"signals_{gene.lower()}"
            existing = result.get(key, "") or ""
            seen = set(line.strip() for line in existing.split("\n") if line.strip())
            for _, sig_row in _sig_df.iterrows():
                sig_name = str(sig_row.get("signal_name", "")).strip()
                if not sig_name or sig_name in seen:
                    continue
                if sig_name in text:
                    if existing:
                        existing += "\n" + sig_name
                    else:
                        existing = sig_name
                    seen.add(sig_name)
            if existing:
                result[key] = existing
    except Exception:
        # 解析器必须鲁棒：cache 未就绪时不影响其他字段
        pass

    # ── 性别 ───────────────────────────────────────────────────
    if re.search(r"性别[:：]?\s*([男女])\b", text):
        result["gender"] = re.search(r"性别[:：]?\s*([男女])\b", text).group(1)
    elif re.search(r"男性|男\s*性|先生|♂", text):
        result["gender"] = "男"
    elif re.search(r"女性|女\s*性|女士|♀", text):
        result["gender"] = "女"

    # ── 年龄 ───────────────────────────────────────────────────
    # 支持多种格式：65岁、65 岁、年龄65、年龄: 65
    age_m = re.search(r"年龄[:：]?\s*(\d+)", text)
    if not age_m:
        age_m = re.search(r"(\d+)\s*(?:岁|年龄|years?\s*old)", text)
    if age_m:
        result["age"] = int(age_m.group(1))

    # ── 化验结果（新增）───────────────────────────────────────
    # ALT = X U/L  （U/L 可选）
    alt_m = re.search(
        r"ALT[=\s:：]*\s*(\d+(?:\.\d+)?)(?:\s*U/?/?L?/?)?",
        text, re.IGNORECASE,
    )
    if alt_m:
        result["alt"] = float(alt_m.group(1))

    # AST = X U/L
    ast_m = re.search(
        r"AST[=\s:：]*\s*(\d+(?:\.\d+)?)(?:\s*U/?/?L?/?)?",
        text, re.IGNORECASE,
    )
    if ast_m:
        result["ast"] = float(ast_m.group(1))

    # Cr = X mg/dL
    cr_m = re.search(
        r"Cr[=\s:：]*\s*(\d+(?:\.\d+)?)(?:\s*mg/?/?d?L?/?)?",
        text, re.IGNORECASE,
    )
    if cr_m:
        result["cr"] = float(cr_m.group(1))

    # CrCl = X mL/min
    crcl_m = re.search(
        r"CrCl[=\s:：]*\s*(\d+(?:\.\d+)?)(?:\s*mL/?/?min?/?)?",
        text, re.IGNORECASE,
    )
    if crcl_m:
        result["crcl"] = float(crcl_m.group(1))

    # eGFR = X mL/min/1.73m²
    egfr_m = re.search(
        r"eGFR[=\s:：]*\s*(\d+(?:\.\d+)?)(?:\s*mL/?/?min?/?1\.?73m?[²2]?/?)?",
        text, re.IGNORECASE,
    )
    if egfr_m:
        result["egfr"] = float(egfr_m.group(1))

    # Alb = X g/dL
    alb_m = re.search(
        r"Alb[=\s:：]*\s*(\d+(?:\.\d+)?)(?:\s*g/?/?d?L?/?)?",
        text, re.IGNORECASE,
    )
    if alb_m:
        result["alb"] = float(alb_m.group(1))

    # T.Bili = X mg/dL
    tbili_m = re.search(
        r"T\.?Bili[=\s:：]*\s*(\d+(?:\.\d+)?)(?:\s*mg/?/?d?L?/?)?",
        text, re.IGNORECASE,
    )
    if tbili_m:
        result["t_bili"] = float(tbili_m.group(1))

    # NH3 = X μg/dL  (支持 μ 或 u)
    nh3_m = re.search(
        r"NH3[=\s:：]*\s*(\d+(?:\.\d+)?)(?:\s*(?:μ|u)g/?/?d?L?/?)?",
        text, re.IGNORECASE,
    )
    if nh3_m:
        result["nh3"] = float(nh3_m.group(1))

    # INR = X
    inr_m = re.search(
        r"INR[=\s:：]*\s*(\d+(?:\.\d+)?)",
        text, re.IGNORECASE,
    )
    if inr_m:
        result["inr"] = float(inr_m.group(1))

    # PT = X 秒/seconds
    pt_m = re.search(
        r"PT[=\s:：]*\s*(\d+(?:\.\d+)?)(?:\s*秒|seconds?)?",
        text, re.IGNORECASE,
    )
    if pt_m:
        result["pt"] = float(pt_m.group(1))

    # Plt = X /μL 或 /μl
    plt_m = re.search(
        r"Plt[=\s:：]*\s*(\d+)(?:\s*/?[μu]L?/?)?",
        text, re.IGNORECASE,
    )
    if plt_m:
        result["plt"] = float(plt_m.group(1))

    # K = X mEq/L
    k_m = re.search(
        r"\bK[=\s:：]*\s*(\d+(?:\.\d+)?)(?:\s*mEq/?/?L?/?)?",
        text, re.IGNORECASE,
    )
    if k_m:
        result["k"] = float(k_m.group(1))

    # CRP = X mg/L
    crp_m = re.search(
        r"CRP[=\s:：]*\s*(\d+(?:\.\d+)?)(?:\s*mg/?/?L?/?)?",
        text, re.IGNORECASE,
    )
    if crp_m:
        result["crp"] = float(crp_m.group(1))

    # ── 主诉（新增）───────────────────────────────────────────
    # 如果文本里有"主诉"标签，则提取其后的内容
    cc_m = re.search(r"主诉[为:：]?\s*([^\n]+)", text_raw)
    if cc_m:
        result["chief_complaint"] = cc_m.group(1).strip()
    else:
        # 否则把整个文本作为主诉（去除已知结构化字段后）
        # 但保留完整文本，方便 Agent 识别化验等
        result["chief_complaint"] = text_raw[:500].strip()  # 截断到 500 字符

    # ── 症状信号识别（新增）─────────────────────────────────
    # 从主诉/既往史中识别药物反应信号（如"可待因镇痛完全无效"）
    # 关键：signal_desc 必须是 CSV signal_name 的子串，才能被 _match_clinical_signals 匹配
    symptom_signals = {
        "CYP2D6": [
            # 只识别"症状/疗效/不良反应"信号，不识别"在用什么药"
            (r"可待因.{0,5}镇痛.{0,5}完全无效", "可待因镇痛完全无效"),
            (r"可待因.{0,5}镇痛.{0,5}无效", "可待因镇痛无效"),
            (r"可待因.{0,5}无效", "可待因无效"),
            (r"曲马多.{0,5}镇痛.{0,5}完全无效", "曲马多镇痛完全无效"),
            (r"曲马多.{0,5}镇痛.{0,5}无效", "曲马多镇痛无效"),
            (r"曲马多.{0,5}无效", "曲马多无效"),
            (r"文拉法辛.{0,15}效(果)?不佳", "文拉法辛疗效不佳"),
            (r"文拉法辛.{0,15}无效", "文拉法辛无效"),
            (r"文拉法辛.{0,15}疗效差", "文拉法辛疗效差"),
            (r"阿米替林.{0,15}疗效不佳", "阿米替林疗效不佳"),
            (r"阿米替林.{0,15}副作用", "阿米替林副作用"),
            (r"阿米替林.{0,30}(口干|便秘|尿潴留|视力模糊|抗胆碱能)", "阿米替林ADR(低剂量TCA副作用)"),
            (r"低剂量.{0,15}(口干|便秘|尿潴留|视力模糊|抗胆碱能)", "低剂量TCA副作用"),
            (r"低剂量TCA.{0,15}(副作用|ADR)", "低剂量TCA副作用"),
            (r"氟西汀.{0,15}严重.{0,5}副作用", "氟西汀严重副作用"),
            (r"他莫昔芬.{0,15}疗效.{0,5}(差|不佳|不足)", "他莫昔芬疗效不足"),
            (r"他莫昔芬.{0,15}复发", "他莫昔芬复发"),
            (r"帕罗西汀.{0,15}副作用", "帕罗西汀副作用"),
            # 中毒/UM 信号
            (r"可待因.{0,30}中毒", "可待因中毒"),
            (r"可待因.{0,30}严重呼吸抑制", "可待因中毒"),
            (r"曲马多.{0,30}癫痫", "曲马多癫痫"),
        ],
        "CYP2C19": [
            (r"氯吡格雷.{0,15}抵抗", "氯吡格雷抵抗"),
            (r"氯吡格雷.{0,15}无反应", "氯吡格雷无反应"),
            (r"氯吡格雷.{0,15}疗效差", "氯吡格雷疗效差"),
            (r"氯吡格雷.{0,15}血栓", "氯吡格雷血栓"),
            (r"伏立康唑.{0,15}浓度过高", "伏立康唑浓度过高"),
            (r"奥美拉唑.{0,15}疗效差", "奥美拉唑疗效差"),
            (r"PPI.{0,15}根除失败", "PPI根除失败"),
            (r"艾司西酞普兰.{0,15}无效", "艾司西酞普兰无效"),
            (r"地西泮.{0,15}过度镇静", "地西泮过度镇静"),
        ],
    }

    # 去除剂量（如 30mg、100mg、25mg/d）让症状描述能匹配 CSV signal_name
    def _strip_dose(t: str) -> str:
        t = re.sub(r"\d+\s*(?:mg|g|mcg|ml|iu|片|粒)(?:/[a-z])?", "", t, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", t).strip()

    for gene, sig_triggers in symptom_signals.items():
        for trigger_re, signal_desc in sig_triggers:
            m = re.search(trigger_re, text)
            if m:
                # 用匹配到的完整文本（去除剂量），而非硬编码截断版
                # 这样可保证是 CSV signal_name 的子串
                matched_text = _strip_dose(m.group(0))
                # 如果 matched_text 比 signal_desc 更具体，用 matched_text
                # 否则用 signal_desc（更标准）
                final_desc = matched_text if len(matched_text) >= len(signal_desc) else signal_desc
                if gene == "CYP2D6":
                    existing = result.get("signals_cyp2d6", "")
                    if not existing:
                        result["signals_cyp2d6"] = final_desc
                elif gene == "CYP2C19":
                    existing = result.get("signals_cyp2c19", "")
                    if not existing:
                        result["signals_cyp2c19"] = final_desc
                break  # 每基因只填第一个信号

    return result


# ── Session 同步 ─────────────────────────────────────────────
def _sync():
    """从 session_state 恢复表单值（跨页面状态保持）"""
    return {
        "case_id": st.session_state.get("case_id", ""),
        "diagnosis": st.session_state.get("diagnosis", ""),
        "current_meds_raw": ",".join(st.session_state.get("current_meds", [])),
        "population": st.session_state.get("population", "东亚"),
        "chief_complaint": st.session_state.get("chief_complaint", ""),
        "age": st.session_state.get("age", None),
        "gender": st.session_state.get("gender", ""),
        "signals_cyp2d6": st.session_state.get("signals_cyp2d6", ""),
        "signals_cyp2c19": st.session_state.get("signals_cyp2c19", ""),
    }

# ── 导入病例（从 cases/ 文件夹读电子版病历）────────────────
with st.expander("📁 导入病例（从 cases/ 文件夹选择）", expanded=False):
    st.caption(
        "💡 把电子版病历（.txt/.md/.csv）放到项目根目录的 `cases/` 文件夹，"
        "这里会自动列出，点击后内容会填入下方文本框"
    )
    _cases_dir = ROOT / "cases"
    if not _cases_dir.exists():
        st.warning(f"未找到 `cases/` 文件夹（{_cases_dir}）")
        case_files: list[str] = []
    else:
        case_files = sorted(
            [f.name for f in _cases_dir.iterdir()
             if f.is_file() and f.suffix.lower() in (".txt", ".md", ".csv")],
            key=str.lower,
        )
    if not case_files:
        st.info("`cases/` 文件夹是空的。把 .txt 病例文件放进去就会在这里显示。")
    else:
        col_pick, col_load = st.columns([3, 1])
        with col_pick:
            selected_case = st.selectbox(
                "选择病例文件",
                options=["— 请选择 —"] + case_files,
                key="case_file_picker",
                label_visibility="collapsed",
            )
        with col_load:
            load_clicked = st.button(
                "📥 加载到下方",
                use_container_width=True,
                disabled=(selected_case == "— 请选择 —"),
            )
        if load_clicked and selected_case != "— 请选择 —":
            try:
                file_path = _cases_dir / selected_case
                case_text = file_path.read_text(encoding="utf-8")
                # 只把文本塞到粘贴框；字段清零+填充留给"解析"按钮做
                st.session_state["raw_clinical_text"] = case_text
                st.success(f"✅ 已加载 {selected_case}（{len(case_text)} 字符），点下方「🔍 解析病历文本」填充")
            except Exception as e:
                st.error(f"读取失败: {e}")


# ── 解析后写入 session 的逻辑（必须在 button 之前定义）──
def _apply_parsed_to_session(raw_text: str, source_name: str = "") -> None:
    """
    解析病历文本 → 写入 session_state → bump form_ver（强制 widget 重建）
    关键设计：
    1. 先把化验字段全部重置为 None（避免旧病例残留）
    2. 覆盖解析到的字段
    3. bump _form_ver，让所有带 key=f"..._{form_ver}" 的 widget 重建显示新值
    """
    parsed = _parse_clinical_text(raw_text)
    if not parsed:
        st.warning("未能从文本中识别到 PGx 相关字段，请手动填写下方表单。")
        return

    # 病例 ID：解析结果优先，否则保留旧值（兜底自动生成）
    if parsed.get("case_id"):
        st.session_state["case_id"] = parsed["case_id"]

    # 这些字段直接覆盖（新文本没有就给空）
    st.session_state["diagnosis"] = parsed.get("diagnosis", "")
    st.session_state["current_meds"] = parsed.get("meds", [])
    st.session_state["signals_cyp2d6"] = parsed.get("signals_cyp2d6", "")
    st.session_state["signals_cyp2c19"] = parsed.get("signals_cyp2c19", "")
    st.session_state["population"] = parsed.get("population", "东亚")
    st.session_state["gender"] = parsed.get("gender", "未知")
    st.session_state["age"] = parsed.get("age", 18) or 18
    st.session_state["chief_complaint"] = parsed.get("chief_complaint", "") or ""

    # 化验字段：先全部重置为 None（清零旧值），再覆盖
    _LAB_FIELDS = ["alt", "ast", "cr", "crcl", "egfr", "alb",
                   "t_bili", "nh3", "inr", "pt", "plt", "k", "crp"]
    for f in _LAB_FIELDS:
        st.session_state[f] = None
    for f in _LAB_FIELDS:
        if parsed.get(f) is not None:
            st.session_state[f] = parsed[f]

    # bump form_ver → 所有 widget 重新创建显示最新值
    st.session_state["_form_ver"] = st.session_state.get("_form_ver", 0) + 1

    st.success(
        f"✅ 已解析：诊断=`{parsed.get('diagnosis','未识别')}` "
        f"用药={len(parsed.get('meds',[]))}种 "
        f"CYP2D6=`{parsed.get('signals_cyp2d6','无')}` "
        f"CYP2C19=`{parsed.get('signals_cyp2c19','无')}` "
        f"人群=`{parsed.get('population','未识别')}`"
    )
    st.rerun()


# ── 病历文本粘贴区（自动解析）────────────────────────────────
with st.expander("📋 粘贴完整病历文本（自动解析所有字段）", expanded=False):
    raw_text = st.text_area(
        "将完整病历文本粘贴至此，系统自动识别诊断/用药/基因型等字段：",
        placeholder=(
            "示例：\n"
            "患者男性，58岁，诊断抑郁症，曾服用舍曲林50mg qd，伴有失眠。\n"
            "CYP2D6: *10/*10 (PM型)，CYP2C19: *1/*1 (正常代谢型)。\n"
            "既往服用可待因效果不佳。家族史无特殊。汉族，东亚人群。"
        ),
        height=140,
        key="raw_clinical_text",
    )
    if raw_text and st.button("🔍 解析病历文本 → 填充下方表单", use_container_width=True):
        _apply_parsed_to_session(raw_text, source_name="粘贴文本")

st.divider()

# ── 表单布局 ─────────────────────────────────────────────────
with st.form("case_form", clear_on_submit=False):
    # form_ver 用于强制 widget 重建（key 变化时 streamlit 重建 widget，
    # 解决"导入新病例时旧病例的字段不归零"问题）
    form_ver = st.session_state.get("_form_ver", 0)
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("👤 患者基本信息")
        case_id = st.text_input(
            "病例ID",
            value=st.session_state.get("case_id", f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
            key=f"case_id_{form_ver}",
            help="唯一标识符，支持自动生成",
        )
        diagnosis = st.text_input(
            "主要诊断",
            placeholder='如：抑郁症、焦虑症、精神分裂症',
            value=st.session_state.get("diagnosis", ""),
            key=f"diagnosis_{form_ver}",
        )
        population = st.selectbox(
            "人群",
            ["东亚", "欧洲", "非洲裔", "拉美裔", "南亚", "其他"],
            index=["东亚", "欧洲", "非洲裔", "拉美裔", "南亚", "其他"].index(
                st.session_state.get("population", "东亚")
            ),
            key=f"population_{form_ver}",
        )
        age = st.number_input("年龄", min_value=0, max_value=120,
                             value=st.session_state.get("age", 18) or 18, step=1,
                             key=f"age_{form_ver}")
        gender = st.radio("性别", ["男", "女", "未知"],
                         horizontal=True,
                         index=["男", "女", "未知"].index(st.session_state.get("gender", "未知") or "未知"),
                         key=f"gender_{form_ver}")

    with col_right:
        st.subheader("💊 当前用药")
        current_meds_raw = st.text_area(
            "已使用药物（用逗号分隔）",
            placeholder="如：舍曲林, 可待因, 奥美拉唑",
            value=",".join(st.session_state.get("current_meds", [])),
            help="系统将基于这些药物检测 DDI 风险并给出调整建议",
            key=f"current_meds_raw_{form_ver}",
            height=120,
        )
        chief_complaint = st.text_area(
            "主诉",
            placeholder="如：情绪低落伴失眠2周",
            value=st.session_state.get("chief_complaint", ""),
            key=f"chief_complaint_{form_ver}",
            height=80,
        )

    st.divider()
    st.subheader("🧪 化验结果（肝/肾指标）")
    st.caption("⚠️ 化验用于评估肝/肾状态 → 调整剂量，不是推断基因型的依据")

    lab_col1, lab_col2, lab_col3, lab_col4 = st.columns(4)
    with lab_col1:
        alt = st.number_input(
            "ALT (U/L)",
            min_value=0, max_value=2000,
            value=int(st.session_state.get("alt", 0) or 0),
            step=1,
            key=f"alt_{form_ver}",
            help="正常 7-40 U/L；>120 (3×ULN) 提示肝功能异常",
        )
    with lab_col2:
        ast = st.number_input(
            "AST (U/L)",
            min_value=0, max_value=2000,
            value=int(st.session_state.get("ast", 0) or 0),
            step=1,
            key=f"ast_{form_ver}",
            help="正常 13-35 U/L；>105 (3×ULN) 提示肝功能异常",
        )
    with lab_col3:
        cr = st.number_input(
            "Cr (mg/dL)",
            min_value=0.0, max_value=20.0,
            value=float(st.session_state.get("cr", 0.0) or 0.0),
            step=0.1,
            key=f"cr_{form_ver}",
            help="正常 0.6-1.2 mg/dL；>2.0 提示肾功能不全",
        )
    with lab_col4:
        crcl = st.number_input(
            "CrCl (mL/min)",
            min_value=0, max_value=200,
            value=int(st.session_state.get("crcl", 0) or 0),
            step=1,
            key=f"crcl_{form_ver}",
            help="正常 ≥60；<30 提示肾脏清除能力下降",
        )

    # ── 扩展化验：肝/肾/凝血/血常规/电解质/炎症 ─────────────
    lab_col5, lab_col6, lab_col7, lab_col8 = st.columns(4)
    with lab_col5:
        egfr = st.number_input(
            "eGFR (mL/min/1.73m²)",
            min_value=0, max_value=200,
            value=int(st.session_state.get("egfr", 0) or 0),
            step=1,
            key=f"egfr_{form_ver}",
            help="正常 ≥90；<30 提示肾功能严重减退",
        )
    with lab_col6:
        alb = st.number_input(
            "Alb (g/dL)",
            min_value=0.0, max_value=8.0,
            value=float(st.session_state.get("alb", 0.0) or 0.0),
            step=0.1,
            key=f"alb_{form_ver}",
            help="正常 3.5-5.0 g/dL；<2.5 提示低蛋白血症，游离药物↑",
        )
    with lab_col7:
        t_bili = st.number_input(
            "T.Bili (mg/dL)",
            min_value=0.0, max_value=30.0,
            value=float(st.session_state.get("t_bili", 0.0) or 0.0),
            step=0.1,
            key=f"t_bili_{form_ver}",
            help="正常 0.2-1.2 mg/dL；>2.0 提示肝功能减退",
        )
    with lab_col8:
        nh3 = st.number_input(
            "NH3 (μg/dL)",
            min_value=0, max_value=500,
            value=int(st.session_state.get("nh3", 0) or 0),
            step=1,
            key=f"nh3_{form_ver}",
            help="正常 15-45 μg/dL；>67.5 (>1.5×ULN) 提示肝性脑病",
        )

    lab_col9, lab_col10, lab_col11, lab_col12 = st.columns(4)
    with lab_col9:
        inr = st.number_input(
            "INR",
            min_value=0.0, max_value=10.0,
            value=float(st.session_state.get("inr", 0.0) or 0.0),
            step=0.1,
            key=f"inr_{form_ver}",
            help="正常 0.8-1.2；>1.5 提示凝血功能障碍，出血风险↑",
        )
    with lab_col10:
        pt = st.number_input(
            "PT (秒)",
            min_value=0.0, max_value=60.0,
            value=float(st.session_state.get("pt", 0.0) or 0.0),
            step=0.1,
            key=f"pt_{form_ver}",
            help="正常 11-13.5 秒；>1.5×control 提示肝脏合成功能↓",
        )
    with lab_col11:
        plt = st.number_input(
            "Plt (/μL)",
            min_value=0, max_value=800000,
            value=int(st.session_state.get("plt", 0) or 0),
            step=1000,
            key=f"plt_{form_ver}",
            help="正常 150000-400000/μL；<50000 提示严重血小板减少",
        )
    with lab_col12:
        k = st.number_input(
            "K (mEq/L)",
            min_value=0.0, max_value=10.0,
            value=float(st.session_state.get("k", 0.0) or 0.0),
            step=0.1,
            key=f"k_{form_ver}",
            help="正常 3.5-5.0 mEq/L；>5.5 提示高钾血症，反映肾功能损害",
        )

    crp = st.number_input(
        "CRP (mg/L)",
        min_value=0, max_value=500,
        value=int(st.session_state.get("crp", 0) or 0),
        step=1,
        key=f"crp_{form_ver}",
        help="正常 <10 mg/L；>50 mg/L 提示严重炎症，可抑制CYP酶活性",
    )

    lab_results_dict = {
        "ALT": alt if alt > 0 else None,
        "AST": ast if ast > 0 else None,
        "Cr": cr if cr > 0 else None,
        "CrCl": crcl if crcl > 0 else None,
        "eGFR": egfr if egfr > 0 else None,
        "Alb": alb if alb > 0 else None,
        "T.Bili": t_bili if t_bili > 0 else None,
        "NH3": nh3 if nh3 > 0 else None,
        "INR": inr if inr > 0 else None,
        "PT": pt if pt > 0 else None,
        "Plt": plt if plt > 0 else None,
        "K": k if k > 0 else None,
        "CRP": crp if crp > 0 else None,
    }

    st.divider()
    st.subheader("🧬 临床信号（不知道基因型时填，用于推断表型）")
    st.caption("💡 提示：已知基因型请在主诉中注明，跳过此栏；不知道基因型才填以下信号")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**CYP2D6 信号**（推断 PM/IM）")
        st.caption("根据用药史推断：可待因无效→PM，曲马多无效→PM，SSRI疗效差→PM/IM")
        signals_cyp2d6 = st.text_area(
            "CYP2D6 临床信号",
            placeholder="如：服用可待因镇痛无效（推断PM），或曲马多无效，或文拉法辛疗效不佳",
            value=st.session_state.get("signals_cyp2d6", ""),
            key=f"signals_cyp2d6_{form_ver}",
            height=80,
            label_visibility="collapsed",
        )
    with c2:
        st.markdown("**CYP2C19 信号**（推断 PM/IM）")
        st.caption("根据用药史推断：氯吡格雷抵抗→PM，奥美拉唑无效→PM，伏立康唑反应强→PM")
        signals_cyp2c19 = st.text_area(
            "CYP2C19 临床信号",
            placeholder="如：氯吡格雷抵抗（推断PM），或伏立康唑副作用明显",
            value=st.session_state.get("signals_cyp2c19", ""),
            key=f"signals_cyp2c19_{form_ver}",
            height=80,
            label_visibility="collapsed",
        )

    st.divider()
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    with col_btn1:
        submitted = st.form_submit_button("🚀 开始分析", use_container_width=True, type="primary")
    with col_btn2:
        cleared = st.form_submit_button("🗑️ 清空", use_container_width=True)
    with col_btn3:
        st.write("")

# ── 处理提交 ─────────────────────────────────────────────────
if submitted:
    # 解析用药列表
    meds_list = _meds_to_list(current_meds_raw)

    # 保存到 session_state
    st.session_state["case_id"] = case_id
    st.session_state["diagnosis"] = diagnosis
    st.session_state["current_meds"] = meds_list
    st.session_state["population"] = population
    st.session_state["chief_complaint"] = chief_complaint
    st.session_state["age"] = age
    st.session_state["gender"] = gender
    st.session_state["signals_cyp2d6"] = signals_cyp2d6
    st.session_state["signals_cyp2c19"] = signals_cyp2c19
    # 化验结果
    st.session_state["alt"] = alt
    st.session_state["ast"] = ast
    st.session_state["cr"] = cr
    st.session_state["crcl"] = crcl
    st.session_state["lab_results"] = lab_results_dict
    st.session_state["submitted"] = True

    # 基本验证
    if not signals_cyp2d6.strip() and not signals_cyp2c19.strip():
        st.warning("⚠️ 请至少填写一个基因的临床信号，否则系统将使用人群频率作为 fallback 表型。")

    with st.spinner("🔬 正在运行 PGx 分析引擎，请稍候..."):
        import time; time.sleep(0.5)

        try:
            from src.data_loader import get_cache
            from src.phenotype_inferrer import infer_phenotype
            from src.drug_recommender import recommend_both_genes
            from src.interaction_checker import check_ddi
            from src.population_analyzer import get_population_analysis
            from src.report_generator import (
                PatientInfo, build_report,
            )
            from src.visualizer import generate_report_html
            from src.report_generator import report_to_dict

            cache = get_cache()

            # ── 表型推断 ───────────────────────────────────────
            # ✅ 关键修复：把整段字符串按换行/分号切成多个独立信号
            # 否则下游 infer_phenotype 会把整段当一个 text，匹配不到任何 signal_name
            def _split_signals(raw: str) -> list[str]:
                if not raw or not raw.strip():
                    return []
                import re as _re
                parts = _re.split(r'[\n;；]', raw)
                return [p.strip() for p in parts if p.strip()]

            signals_d = _split_signals(signals_cyp2d6)
            signals_c = _split_signals(signals_cyp2c19)

            phenotypes = {}
            for gene, sigs in [("CYP2D6", signals_d), ("CYP2C19", signals_c)]:
                if sigs:
                    phenotypes[gene] = infer_phenotype(
                        gene, sigs, population=population, lab_results=lab_results_dict
                    )
                else:
                    # 即使无临床症状，血液标记超阈值也会生成 CCA 弱证据
                    phenotypes[gene] = infer_phenotype(
                        gene, [], population=population, lab_results=lab_results_dict
                    )

            phenotype_map = {g: p.phenotype for g, p in phenotypes.items()}

            # ── 药物推荐 ─────────────────────────────────────────
            # 关键修复：当 meds_list 为空时必须传 None，否则
            # recommend_for_gene 会进入"仅保留 current_meds 命中药品"分支
            # → 把所有药品全部过滤掉 → drug_recs.recommendations 为空，
            # → 下游"当前用药速览"无法按最大概率表型给出任何推荐
            meds_for_rec = meds_list if meds_list else None
            drug_recs = recommend_both_genes(phenotype_map, meds_for_rec)

            # ── DDI 检测 ───────────────────────────────────────
            ddi_report = check_ddi(meds_list)

            # ── 处方合理性评估（综合DDI）─────────────────────────
            from src.drug_recommender import evaluate_prescription
            eval_results = evaluate_prescription(drug_recs, meds_list, ddi_report)
            for gene, ev in eval_results.items():
                drug_recs[gene].prescription_eval = ev

            # ── 人群频率 ────────────────────────────────────────
            pop_data = {}
            for gene in ["CYP2D6", "CYP2C19"]:
                pd_ = get_population_analysis(gene, population)
                if pd_:
                    pop_data[gene] = pd_

            # ── 构建报告 ────────────────────────────────────────
            patient = PatientInfo(
                case_id=case_id,
                diagnosis=diagnosis,
                current_meds=meds_list,
                population=population,
                chief_complaint=chief_complaint,
                age=age,
                gender=gender,
            )
            report = build_report(patient, phenotypes, drug_recs, ddi_report, pop_data)
            report_dict = report_to_dict(report)
            html_report = generate_report_html(report)

            # 保存结果
            st.session_state["report"] = report
            st.session_state["report_dict"] = report_dict
            st.session_state["html_report"] = html_report

            # 加入历史
            history_entry = {
                "case_id": case_id,
                "diagnosis": diagnosis,
                "population": population,
                "meds": meds_list,
                "phenotypes": phenotype_map,
                "generated_at": report.generated_at,
                "high_risk_ddi": report.high_risk_ddi_count,
                "urgent_count": len(report.urgent_interventions),
            }
            st.session_state["history"].insert(0, history_entry)
            if len(st.session_state["history"]) > 50:
                st.session_state["history"] = st.session_state["history"][:50]

            st.success("✅ 分析完成！请前往左侧 **📊 分析报告** 页面查看完整CDS报告。")

        except Exception as e:
            import traceback
            st.error(f"❌ 分析过程出错: {e}")
            with st.expander("详细错误信息"):
                st.code(traceback.format_exc())

elif cleared:
    for k in list(st.session_state.keys()):
        if k not in ("history",):
            st.session_state[k] = "" if k != "submitted" else False
    st.session_state["current_meds"] = []
    st.rerun()
