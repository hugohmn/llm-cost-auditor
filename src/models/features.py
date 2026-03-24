"""Feature complexity classification and audit configuration.

AuditConfig is the single source of truth for all tunable thresholds.
Every magic number in the pipeline lives here — nothing is hardcoded
in detection or simulation logic.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TaskComplexity(StrEnum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


# Sensible defaults — override via AuditConfig for production use.
DEFAULT_FEATURE_COMPLEXITY: dict[str, TaskComplexity] = {
    "email_drafting": TaskComplexity.SIMPLE,
    "data_extraction": TaskComplexity.SIMPLE,
    "customer_support_agent": TaskComplexity.MODERATE,
    "doc_qa": TaskComplexity.MODERATE,
    "code_review": TaskComplexity.COMPLEX,
}


class AuditConfig(BaseModel):
    """All tunable parameters for the audit pipeline.

    Every threshold, model name, and feature classification used by
    waste detectors and routing simulation is defined here. The report
    renders these so the reader can audit every assumption.
    """

    model_config = ConfigDict(frozen=True)

    # ── Feature classification ───────────────────────────────────
    feature_complexity: dict[str, TaskComplexity] = Field(
        default_factory=lambda: dict(DEFAULT_FEATURE_COMPLEXITY),
        description=(
            "Maps feature names to complexity tiers. Features not listed "
            "default to COMPLEX (conservative — no routing, no waste flagging)."
        ),
    )

    # ── Routing thresholds ───────────────────────────────────────
    moderate_input_threshold: int = Field(
        default=4000,
        description=(
            "MODERATE features with input tokens below this are treated "
            "as SIMPLE for routing and wrong-model detection."
        ),
    )
    light_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Target model for routed calls.",
    )
    frontier_model: str = Field(
        default="claude-sonnet-4-6",
        description="Default frontier model kept for complex tasks.",
    )

    # ── Bloated prompt detection ─────────────────────────────────
    bloated_prompt_multiplier: float = Field(
        default=2.0,
        description=(
            "Flag entries with input tokens exceeding this multiple of "
            "the per-feature median. 2.0 = more than double the typical call."
        ),
    )
    bloated_prompt_min_tokens: int = Field(
        default=8000,
        description=(
            "Minimum absolute token threshold — entries below this are "
            "never flagged regardless of the multiplier."
        ),
    )
    bloated_prompt_min_feature_entries: int = Field(
        default=10,
        description=(
            "Features with fewer entries than this are excluded from "
            "bloat detection (insufficient data for a reliable median)."
        ),
    )

    # ── Duplicate detection ──────────────────────────────────────
    duplicate_window_seconds: int = Field(
        default=120,
        description="Time window to consider calls as potential duplicates.",
    )
    duplicate_token_tolerance: float = Field(
        default=0.05,
        description=(
            "Maximum relative difference in input token count for two "
            "calls to be considered near-duplicates. 0.05 = within 5%%."
        ),
    )

    # ── Retry detection ──────────────────────────────────────────
    retry_window_seconds: int = Field(
        default=60,
        description=(
            "Time window after a failed call to look for a retry "
            "(a subsequent success on the same feature+model)."
        ),
    )

    # ── Quality evaluation ────────────────────────────────────────
    enable_judge_eval: bool = Field(
        default=False,
        description="Run LLM-as-Judge evaluation (costs money for replay + judge calls).",
    )
    judge_eval_sample_size: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Samples per feature for LLM-as-Judge.",
    )
    judge_model: str = Field(
        default="claude-sonnet-4-6",
        description="Model used as judge in Layer 2 evaluation.",
    )
    proxy_min_samples: int = Field(
        default=10,
        description="Min calls per model tier per feature for proxy signal comparison.",
    )
    judge_max_budget_usd: float = Field(
        default=5.0,
        ge=0.01,
        description="Hard cap on total Layer 2 evaluation spend.",
    )

    @property
    def routable_features(self) -> frozenset[str]:
        """Features eligible for routing and bloat detection."""
        return frozenset(
            f
            for f, c in self.feature_complexity.items()
            if c in {TaskComplexity.SIMPLE, TaskComplexity.MODERATE}
        )
