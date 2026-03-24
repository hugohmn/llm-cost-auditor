"""Final report schema — combines analysis and recommendations."""

from datetime import datetime

from pydantic import BaseModel, Field

from src.models.analysis import AnalysisResult
from src.models.quality import QualityAssessment
from src.models.recommendation import OptimizationPlan


class AuditReport(BaseModel):
    """The complete audit report — final output of the pipeline."""

    generated_at: datetime = Field(default_factory=datetime.now)
    log_source: str
    total_entries_analyzed: int

    analysis: AnalysisResult
    optimization: OptimizationPlan

    executive_summary: str = Field(
        description="LLM-generated executive summary",
    )
    quality_assessment: QualityAssessment | None = Field(
        default=None,
        description="Quality evaluation results (3 layers), when computed.",
    )

    @property
    def headline_savings(self) -> str:
        savings = self.optimization.total_potential_savings_usd
        pct = self.optimization.total_potential_savings_pct * 100
        return f"${savings:,.0f}/month ({pct:.0f}% reduction)"
