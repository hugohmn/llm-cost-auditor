"""Recommendation schema — output of the routing simulation agent."""

from pydantic import BaseModel, Field


class RoutingSimResult(BaseModel):
    """Result of simulating multi-model routing."""

    current_cost_usd: float
    optimized_cost_usd: float
    savings_usd: float
    savings_pct: float = Field(ge=0.0, le=1.0)
    calls_routed_to_light: int
    calls_kept_on_frontier: int
    light_model: str
    frontier_model: str
    light_model_error_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Observed API error rate for light model in the dataset",
    )
    frontier_error_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Observed API error rate for frontier models in the dataset",
    )
    quality_caveat: str = Field(
        default=(
            "Error rates measure API failures (timeouts, rate limits), "
            "not output quality. Output quality assessment requires "
            "an eval set comparing model responses side-by-side."
        ),
        description="Caveat about what error rates do and do not measure",
    )


class Recommendation(BaseModel):
    """A single actionable recommendation."""

    priority: int = Field(ge=1, le=5, description="1 = highest priority")
    category: str = Field(
        description=("routing | prompt_optimization | caching | model_switch | architecture")
    )
    title: str
    description: str
    estimated_monthly_savings_usd: float
    implementation_effort: str = Field(description="low | medium | high")
    details: str = Field(default="", description="Technical implementation details")


class OptimizationPlan(BaseModel):
    """Complete optimization plan with all recommendations."""

    routing_simulation: RoutingSimResult | None = None
    recommendations: list[Recommendation]
    total_potential_savings_usd: float
    total_potential_savings_pct: float = Field(ge=0.0, le=1.0)
