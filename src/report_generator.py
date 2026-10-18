"""
report_generator.py — Step 6
生成结构化 CDS（临床决策支持）报告

将表型推断 + 药物推荐 + DDI检测 + 人群频率分析
整合为统一的诊断报告，供可视化模块使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from .phenotype_inferrer import InferredPhenotype
from .drug_recommender import GeneRecommendations, DrugRecommendation
from .interaction_checker import DDICheckReport, DDIResult
from .population_analyzer import PopulationPhenotypeFreq


@dataclass
class PatientInfo:
    """患者基本信息"""
    case_id: str
    diagnosis: str
    current_meds: list[str]
    population: str
    chief_complaint: str = ""           # 主诉
    age: Optional[int] = None
    gender: str = ""


@dataclass
class CDSReport:
    """
    完整 CDS 报告结构。
    包含：患者信息、推断表型、药物推荐、DDI、人群频率、生成时间。
    """
    patient: PatientInfo
    phenotypes: dict[str, InferredPhenotype]      # CYP2D6 / CYP2C19
    drug_recs: dict[str, GeneRecommendations]
    ddi_report: DDICheckReport
    population_data: dict[str, PopulationPhenotypeFreq]
    generated_at: str = field(default_factory=lambda:
                              datetime.now().strftime("%Y-%m-%d %H:%M"))

    # ── 汇总属性 ───────────────────────────────────────────────

    @property
    def cyp2d6_phenotype(self) -> InferredPhenotype:
        return self.phenotypes["CYP2D6"]

    @property
    def cyp2c19_phenotype(self) -> InferredPhenotype:
        return self.phenotypes["CYP2C19"]

    @property
    def high_risk_ddi_count(self) -> int:
        return len(self.ddi_report.high_risk)

    @property
    def total_ddi_count(self) -> int:
        return len(self.ddi_report.interactions)

    @property
    def urgent_interventions(self) -> list[DrugRecommendation]:
        """需要紧急干预的高优先级药物推荐"""
        interventions = []
        for gene_rec in self.drug_recs.values():
            interventions.extend(gene_rec.high_priority)
        return interventions

    @property
    def has_clinically_significant_interactions(self) -> bool:
        return self.ddi_report.has_high_risk or any(
            r.needs_adjustment for r in self.drug_recs["CYP2D6"].recommendations
        ) or any(
            r.needs_adjustment for r in self.drug_recs["CYP2C19"].recommendations
        )


def build_report(
    patient: PatientInfo,
    phenotypes: dict[str, InferredPhenotype],
    drug_recs: dict[str, GeneRecommendations],
    ddi_report: DDICheckReport,
    population_data: dict[str, PopulationPhenotypeFreq],
) -> CDSReport:
    """构造 CDSReport 对象"""
    return CDSReport(
        patient=patient,
        phenotypes=phenotypes,
        drug_recs=drug_recs,
        ddi_report=ddi_report,
        population_data=population_data,
    )


def report_to_dict(report: CDSReport) -> dict:
    """
    将 CDSReport 转为纯 dict（便于 JSON 序列化 / 前端渲染）。
    """
    def _pheno(p: InferredPhenotype) -> dict:
        return {
            "gene": p.gene,
            "phenotype": p.phenotype,
            "genotype": p.genotype,
            "confidence": p.confidence,
            "posterior_prob": p.posterior_prob,
            "prior_prob": p.prior_prob,
            "evidence_strength": p.evidence_strength,
            "population": p.population,
            "is_fallback": p.is_fallback,
            "matched_signals": [
                {
                    "signal_name": m.signal_name,
                    "signal_text": m.signal_text,
                    "target_phenotype": m.target_phenotype,
                    "strength": m.strength,
                    "confidence": m.confidence,
                    "interpretation": m.interpretation,
                }
                for m in p.matched_signals
            ],
        }

    def _rec(r: DrugRecommendation) -> dict:
        return {
            "gene": r.gene,
            "drug_name": r.drug_name,
            "phenotype": r.phenotype,
            "current_action": r.current_action,
            "recommendation_grade": r.recommendation_grade,
            "grade_label": r.grade_label,
            "needs_adjustment": r.needs_adjustment,
            "evidence": r.evidence,
            "alternatives": r.alternatives,
        }

    def _ddi(d: DDIResult) -> dict:
        return {
            "drug_a": d.drug_a,
            "drug_b": d.drug_b,
            "mechanism": d.mechanism,
            "warning_level": d.warning_level,
            "clinical_consequence": d.clinical_consequence,
            "suggestion": d.suggestion,
        }

    def _pop(p: PopulationPhenotypeFreq) -> dict:
        return p.to_dict()

    high_prio_recs = report.urgent_interventions

    return {
        "generated_at": report.generated_at,
        "patient": {
            "case_id": report.patient.case_id,
            "diagnosis": report.patient.diagnosis,
            "current_meds": report.patient.current_meds,
            "population": report.patient.population,
            "chief_complaint": report.patient.chief_complaint,
            "age": report.patient.age,
            "gender": report.patient.gender,
        },
        "phenotypes": {
            "CYP2D6": _pheno(report.cyp2d6_phenotype),
            "CYP2C19": _pheno(report.cyp2c19_phenotype),
        },
        "drug_recommendations": {
            gene: {
                "phenotype": rec.phenotype,
                "high_priority": [_rec(r) for r in rec.high_priority],
                "all": [_rec(r) for r in rec.recommendations],
            }
            for gene, rec in report.drug_recs.items()
        },
        "ddi": {
            "total_count": report.total_ddi_count,
            "high_risk_count": report.high_risk_ddi_count,
            "interactions": [_ddi(d) for d in report.ddi_report.interactions],
            "high_risk": [_ddi(d) for d in report.ddi_report.high_risk],
        },
        "population_data": {
            gene: _pop(p) for gene, p in report.population_data.items()
        },
        "summary": {
            "urgent_intervention_count": len(high_prio_recs),
            "has_clinically_significant_interactions": report.has_clinically_significant_interactions,
            "high_risk_ddi": report.high_risk_ddi_count > 0,
        },
        "urgent_interventions": [_rec(r) for r in high_prio_recs],
    }
