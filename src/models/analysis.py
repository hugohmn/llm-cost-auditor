"""Analysis results schema — output of the analysis agent."""

from pydantic import BaseModel, Field

from src.models.features import AuditConfig


class ModelCostBreakdown(BaseModel):
    """Cost breakdown for a single model."""

    model: str
    total_calls: int
    total_tokens: int
    total_cost_usd: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_latency_ms: float | None = None
    error_rate: float = Field(ge=0.0, le=1.0)
    pct_of_total_cost: float = Field(ge=0.0, le=1.0)


class WastePattern(BaseModel):
    """A detected pattern of waste in LLM usage."""

    pattern_type: str = Field(
        description=("bloated_prompt | wrong_model | excessive_retries | cacheable_request")
    )
    description: str
    affected_calls: int
    estimated_waste_usd: float
    methodology: str = Field(
        default="",
        description="How the waste was computed",
    )
    confidence: str = Field(
        default="medium",
        description="high = directly observed, medium = heuristic, low = rough proxy",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Key assumptions that drive this detection",
    )


class FeatureCostBreakdown(BaseModel):
    """Cost breakdown by application feature."""

    feature: str
    total_calls: int
    total_cost_usd: float
    primary_model: str
    avg_tokens_per_call: float


class AnalysisResult(BaseModel):
    """Complete analysis output."""

    total_cost_usd: float
    total_calls: int
    total_tokens: int
    date_range_days: int
    monthly_projected_cost_usd: float

    cost_by_model: list[ModelCostBreakdown]
    cost_by_feature: list[FeatureCostBreakdown]
    waste_patterns: list[WastePattern]

    total_waste_usd: float = Field(
        description="Sum of all detected waste patterns (may overlap)",
    )
    deduplicated_waste_usd: float = Field(
        description="Maximum recoverable waste with no double-counting",
    )
    waste_pct: float = Field(
        ge=0.0,
        le=1.0,
        description="Deduplicated waste as % of total cost",
    )

    # Configuration used — stored so the report can show all assumptions
    audit_config: AuditConfig = Field(default_factory=AuditConfig)

    # Agent-provided insights (populated when agentic analysis runs)
    agent_findings_summary: str = Field(
        default="",
        description="Agent-generated summary of key findings",
    )
    agent_key_insights: list[str] = Field(
        default_factory=list,
        description="Agent-identified insights beyond standard detectors",
    )
