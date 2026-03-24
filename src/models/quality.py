"""Quality evaluation models — proxy signals, LLM-as-Judge, and eval plans.

Three layers of quality assessment for routing decisions:
  Layer 1: Statistical proxy signals from existing cross-model data
  Layer 2: LLM-as-Judge blind A/B evaluation (opt-in)
  Layer 3: Concrete evaluation plans for pre-routing validation
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# ── Layer 1: Proxy Signals ────────────────────────────────────


class ConfidenceLevel(StrEnum):
    """Statistical confidence based on sample size per model tier."""

    INSUFFICIENT = "insufficient"  # n < proxy_min_samples
    MODERATE = "moderate"  # proxy_min_samples <= n < 50
    STRONG = "strong"  # n >= 50


class OutputTokenStats(BaseModel):
    """Output token distribution for one model tier on one feature."""

    model_config = ConfigDict(frozen=True)

    sample_size: int = Field(ge=0)
    mean: float = Field(ge=0.0)
    median: float = Field(ge=0.0)
    p25: float = Field(ge=0.0)
    p75: float = Field(ge=0.0)


class FeatureProxySignal(BaseModel):
    """Cross-model proxy comparison for a single feature."""

    model_config = ConfigDict(frozen=True)

    feature: str
    frontier_output: OutputTokenStats
    light_output: OutputTokenStats
    output_ratio: float = Field(
        description="light_mean / frontier_mean (1.0 = same length)",
    )
    frontier_error_rate: float = Field(ge=0.0, le=1.0)
    light_error_rate: float = Field(ge=0.0, le=1.0)
    confidence: ConfidenceLevel
    interpretation: str = Field(
        description="Human-readable assessment of the proxy signals",
    )


class ProxySignalResult(BaseModel):
    """Complete Layer 1 result."""

    model_config = ConfigDict(frozen=True)

    feature_signals: list[FeatureProxySignal]
    features_without_comparison: list[str] = Field(
        description="Routable features with no light model calls in the dataset",
    )
    methodology: str
    limitations: list[str]


# ── Layer 2: LLM-as-Judge ─────────────────────────────────────

# Criterion weights for composite scoring
ACCURACY_WEIGHT = 0.35
COMPLETENESS_WEIGHT = 0.25
COHERENCE_WEIGHT = 0.15
INSTRUCTION_FOLLOWING_WEIGHT = 0.25


class CriterionScores(BaseModel):
    """Judge scores for a single response on all criteria."""

    model_config = ConfigDict(frozen=True)

    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    coherence: int = Field(ge=1, le=5)
    instruction_following: int = Field(ge=1, le=5)

    @property
    def weighted_composite(self) -> float:
        """Weighted composite score (1.0-5.0 range)."""
        return (
            ACCURACY_WEIGHT * self.accuracy
            + COMPLETENESS_WEIGHT * self.completeness
            + COHERENCE_WEIGHT * self.coherence
            + INSTRUCTION_FOLLOWING_WEIGHT * self.instruction_following
        )


class JudgePairResult(BaseModel):
    """Result of one blind A/B evaluation."""

    model_config = ConfigDict(frozen=True)

    frontier_scores: CriterionScores
    light_scores: CriterionScores
    preferred: str = Field(
        description="'frontier' | 'light' | 'tie'",
    )
    quality_delta: float = Field(
        description="light_composite - frontier_composite",
    )
    judge_rationale: str
    frontier_was_a: bool = Field(
        description="True if frontier was shown as Response A (for bias auditing)",
    )


class RoutingVerdict(StrEnum):
    """Quality-based verdict for routing a feature to the light model."""

    SAFE_TO_ROUTE = "safe_to_route"
    ROUTE_WITH_MONITORING = "route_with_monitoring"
    ROUTE_WITH_CAUTION = "route_with_caution"
    DO_NOT_ROUTE = "do_not_route"


class FeatureJudgeResult(BaseModel):
    """Aggregated judge results for one feature."""

    model_config = ConfigDict(frozen=True)

    feature: str
    sample_size: int
    mean_quality_delta: float
    std_quality_delta: float
    ci_lower_95: float = Field(description="95% CI lower bound on mean delta")
    ci_upper_95: float = Field(description="95% CI upper bound on mean delta")
    win_rate: float = Field(ge=0.0, le=1.0, description="Fraction where light was preferred")
    tie_rate: float = Field(ge=0.0, le=1.0)
    loss_rate: float = Field(ge=0.0, le=1.0, description="Fraction where frontier was preferred")
    verdict: RoutingVerdict
    per_criterion_deltas: dict[str, float] = Field(
        description="Mean delta per criterion (light minus frontier)",
    )


class JudgeEvalResult(BaseModel):
    """Complete Layer 2 result."""

    model_config = ConfigDict(frozen=True)

    feature_results: list[FeatureJudgeResult]
    skipped_features: list[str] = Field(
        description="Features excluded (insufficient samples or no content)",
    )
    total_samples: int
    total_eval_cost_usd: float = Field(ge=0.0)
    judge_model: str
    methodology: str


class JudgeEvalNotRun(BaseModel):
    """Placeholder when Layer 2 is skipped."""

    model_config = ConfigDict(frozen=True)

    reason: str = Field(
        description=("'disabled_by_config' | 'no_prompt_content_available'"),
    )


# ── Layer 3: Eval Plans ───────────────────────────────────────


class EvalCriterion(BaseModel):
    """One scoring criterion for a feature's evaluation plan."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    minimum_acceptable: int = Field(ge=1, le=5)


class FeatureEvalPlan(BaseModel):
    """Concrete evaluation plan for one feature's routing decision."""

    model_config = ConfigDict(frozen=True)

    feature: str
    current_model: str
    proposed_model: str
    recommended_sample_size: int = Field(ge=5)
    sample_size_rationale: str
    criteria: list[EvalCriterion]
    minimum_composite_score: float = Field(ge=1.0, le=5.0)
    estimated_eval_cost_usd: float = Field(ge=0.0)
    steps: list[str] = Field(description="Step-by-step evaluation instructions")


class EvalPlanResult(BaseModel):
    """Complete Layer 3 result."""

    model_config = ConfigDict(frozen=True)

    feature_plans: list[FeatureEvalPlan]
    total_estimated_cost_usd: float = Field(ge=0.0)
    methodology: str


# ── Top-level composite ───────────────────────────────────────


class QualityAssessment(BaseModel):
    """Complete quality evaluation across all 3 layers."""

    model_config = ConfigDict(frozen=True)

    proxy_signals: ProxySignalResult
    judge_eval: JudgeEvalResult | JudgeEvalNotRun
    eval_plans: EvalPlanResult
