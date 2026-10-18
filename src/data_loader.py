"""
data_loader.py — CSV统一加载器
所有CSV通过UUID定位，避免文件名混淆；提供统一缓存+查找接口。
优先从DATA_DIR加载，找不到则回退到_FALLBACK_DATA。
"""

import warnings
import pandas as pd
from pathlib import Path
from typing import Optional
from .config import DATA_DIR, _FALLBACK_DATA, CSV_FILES


class DataLoadError(Exception):
    """CSV加载失败时抛出"""
    pass


class PGxDataCache:
    """
    单例缓存：所有DataFrame只加载一次。
    用法：
        cache = PGxDataCache()          # 首次调用加载全部
        df = cache.signals_cyp2d6       # 直接访问
    或：
        cache = PGxDataCache.get_instance()
    """

    _instance: Optional["PGxDataCache"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
            cls._instance._mtime_signature = {}
        return cls._instance

    @classmethod
    def get_instance(cls) -> "PGxDataCache":
        return cls()

    def _find_csv(self, uuid_name: str) -> Path:
        """依次在 DATA_DIR、_FALLBACK_DATA 查找CSV路径"""
        for base in (DATA_DIR, _FALLBACK_DATA):
            p = base / uuid_name
            if p.exists():
                return p
        raise DataLoadError(
            f"CSV文件不存在（已搜索 DATA_DIR 和 _FALLBACK_DATA）: {uuid_name}"
        )

    def _load(self) -> None:
        """实际执行CSV加载（仅调用一次）"""
        self._cache: dict[str, pd.DataFrame] = {}
        for key, uuid_name in CSV_FILES.items():
            path = self._find_csv(uuid_name)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)
                    df = pd.read_csv(
                        path, dtype=str, encoding="utf-8-sig",
                        on_bad_lines="skip", index_col=False,
                    )
            except UnicodeDecodeError:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)
                    df = pd.read_csv(
                        path, dtype=str, encoding="utf-8",
                        on_bad_lines="skip", index_col=False,
                    )
            # 清理列名首尾空格
            df.columns = df.columns.str.strip()
            # 清理字符串单元格空格
            df = df.apply(
                lambda col: col.str.strip() if col.dtype == "object" else col
            )
            self._cache[key] = df
        self._loaded = True

    def __getattr__(self, name: str) -> pd.DataFrame:
        if not self._loaded:
            self._load()
        if name == "_cache" or name == "_loaded":
            return object.__getattribute__(self, name)
        if name not in self._cache:
            raise AttributeError(
                f"PGxDataCache没有 '{name}'，可用: {list(self._cache.keys())}"
            )
        return self._cache[name]

    # ── 便捷查找方法 ─────────────────────────────────────────

    def get_drug_row(self, gene: str, drug_name: str) -> Optional[pd.Series]:
        """
        在 drugs.csv（CYP2D6）或 drugs_cyp2c19.csv（CYP2C19）中查找药物。
        gene: "CYP2D6" 或 "CYP2C19"
        drug_name: 药物通用名（子串匹配）
        """
        if gene == "CYP2D6":
            df = self.drugs
        elif gene == "CYP2C19":
            df = self.drugs_cyp2c19
        else:
            return None
        # 同时搜索 drug_name 和 drug_category（drugs_cyp2c19没有drug_category列）
        name_mask = df["drug_name"].str.contains(drug_name, case=False, na=False)
        if "drug_category" in df.columns:
            cat_mask = df["drug_category"].str.contains(drug_name, case=False, na=False)
            mask = name_mask | cat_mask
        else:
            mask = name_mask
        rows = df[mask]
        if rows.empty:
            return None
        return rows.iloc[0]

    def get_signal_rows(self, gene: str, signal_text: str) -> pd.DataFrame:
        """
        在 clinical_signals 表中模糊匹配信号。
        gene: "CYP2D6" 或 "CYP2C19"
        signal_text: 既往信号文本（子串匹配 interpretation 或 signal_name）
        """
        if gene == "CYP2D6":
            df = self.clinical_signals
        elif gene == "CYP2C19":
            df = self.clinical_signals_cyp2c19
        else:
            return pd.DataFrame()
        mask = (
            df["signal_name"].str.contains(signal_text, case=False, na=False)
            | df["context"].str.contains(signal_text, case=False, na=False)
            | df["interpretation"].str.contains(signal_text, case=False, na=False)
        )
        return df[mask]

    def get_ddi(self, drug_a: str, drug_b: str) -> pd.DataFrame:
        """
        查询两药相互作用（双向查询）。
        """
        df = self.drug_interactions
        mask = (
            (
                df["drug_a"].str.contains(drug_a, case=False, na=False)
                & df["drug_b"].str.contains(drug_b, case=False, na=False)
            )
            | (
                df["drug_a"].str.contains(drug_b, case=False, na=False)
                & df["drug_b"].str.contains(drug_a, case=False, na=False)
            )
        )
        return df[mask]

    def get_alternative_drugs(self, therapeutic_class: str) -> pd.DataFrame:
        """
        按治疗类别查询替代药物（drugs_other_genes.csv）。
        """
        df = self.drugs_other_genes
        mask = df["therapeutic_class"].str.contains(
            therapeutic_class, case=False, na=False
        )
        return df[mask]

    def get_other_gene_drug(self, drug_name: str) -> Optional[pd.Series]:
        """
        在 drugs_other_genes.csv 中按药名子串匹配查找药物。
        返回匹配到的行（pd.Series），未找到返回 None。
        """
        df = self.drugs_other_genes
        name_lower = drug_name.strip().lower()
        for _, row in df.iterrows():
            drug_nm = str(row.get("drug_name", "")).strip().lower()
            if name_lower in drug_nm or drug_nm in name_lower:
                return row
        return None

    def get_inhibitors(
        self, gene: str, strength: Optional[str] = None
    ) -> pd.DataFrame:
        """
        查询某基因的抑制剂列表。
        gene: "CYP2D6" 或 "CYP2C19"
        strength: 可选，筛选 "强"/"中"/"弱"
        """
        if gene == "CYP2D6":
            df = self.inhibitors_cyp2d6
        elif gene == "CYP2C19":
            df = self.inhibitors_cyp2c19
        else:
            return pd.DataFrame()
        if strength:
            df = df[df["strength"].str.contains(strength, case=False, na=False)]
        return df

    def get_population_freq(
        self, gene: str, population: str
    ) -> Optional[pd.Series]:
        """
        查询某人群的表型频率。
        """
        if gene == "CYP2D6":
            df = self.phenotype_frequencies_cyp2d6
        elif gene == "CYP2C19":
            df = self.phenotype_frequencies_cyp2c19
        else:
            return None
        mask = df["population"].str.contains(population, case=False, na=False)
        rows = df[mask]
        if rows.empty:
            return None
        return rows.iloc[0]

    def get_test_case(self, case_id: str) -> Optional[pd.Series]:
        """
        按 case_id 查测试用例。
        """
        df = self.test_cases
        mask = df["case_id"].str.strip() == case_id.strip()
        rows = df[mask]
        if rows.empty:
            return None
        return rows.iloc[0]

    def list_all(self) -> dict[str, pd.DataFrame]:
        """返回所有 DataFrame 的字典（调试用）"""
        if not self._loaded:
            self._load()
        return dict(self._cache)

    def reload(self) -> None:
        """强制重新加载所有CSV（调试/热更新用）"""
        self._loaded = False
        self._cache = {}
        self._load()


# ── 快捷单例访问函数 ──────────────────────────────────────────

_cache: Optional[PGxDataCache] = None


def get_cache() -> PGxDataCache:
    global _cache
    if _cache is None:
        _cache = PGxDataCache()
        _cache._mtime_signature = _compute_mtime_signature(_cache)
    else:
        # 自动检测 CSV 文件修改，若有变化则重载
        current_sig = _compute_mtime_signature(_cache)
        if current_sig != _cache._mtime_signature:
            _cache.reload()
            _cache._mtime_signature = current_sig
    return _cache


def _compute_mtime_signature(cache: PGxDataCache) -> dict:
    """计算所有已加载 CSV 的 mtime 签名，用于检测文件修改"""
    sig = {}
    for key in cache._cache.keys():
        try:
            # 从 cache 拿到 DataFrame，再从路径拿 mtime
            # 但 DataFrame 不带路径，所以我们用 CSV_FILES 反查
            uuid_name = CSV_FILES.get(key, "")
            for base in (DATA_DIR, _FALLBACK_DATA):
                p = base / uuid_name
                if p.exists():
                    sig[key] = p.stat().st_mtime
                    break
        except Exception:
            sig[key] = 0.0
    return sig
