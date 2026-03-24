"""Optimization tools — simulation and estimation tools for the optimization agent.

Bound to dataset and analysis results via closure at registration time.
"""

import json
import logging
from collections import defaultdict

from src.models.analysis import AnalysisResult
from src.models.log_entry import LogDataset
from src.models.recommendation import RoutingSimResult
from src.tools.registry import ToolRegistry
from src.utils.llm_client import estimate_cost

logger = logging.getLogger(__name__)


def _register_summary_tools(
    registry: ToolRegistry,
    dataset: LogDataset,
    analysis: AnalysisResult,
    routing_sim: RoutingSimResult | None,
) -> None:
    """Register analysis summary and routing simulation tools."""
    registry.register(
        name="get_analysis_summary",
        description=(
            "Get the complete analysis results: cost breakdowns by model and "
            "feature, waste patterns with methodology, and monthly projections."
        ),
        handler=lambda _: analysis.model_dump_json(indent=2),
    )

    registry.register(
        name="simulate_routing",
        description=(
            "Simulate multi-model routing: route simple/moderate tasks to Haiku, "
            "keep complex ones on Sonnet. Returns current vs optimized cost, "
            "savings percentage, calls routed, and observed error rates per model tier."
        ),
        handler=lambda _: _get_routing_sim(dataset, routing_sim),
    )


def _register_detail_tools(registry: ToolRegistry, dataset: LogDataset) -> None:
    """Register model-switch estimation and feature detail tools."""
    registry.register(
        name="estimate_model_switch_savings",
        description=(
            "Calculate savings if a specific feature switched to a different model. "
            "Returns current cost, projected cost, and savings for that feature."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "feature": {"type": "string", "description": "Feature name to analyze"},
                "target_model": {"type": "string", "description": "Model to switch to"},
            },
            "required": ["feature", "target_model"],
        },
        handler=lambda params: _estimate_switch(
            dataset, str(params["feature"]), str(params["target_model"])
        ),
    )

    registry.register(
        name="get_feature_detail",
        description=(
            "Get detailed statistics for a specific feature: call volume, "
            "model distribution, token percentiles (p25/p50/p75/p90), "
            "error rate, and cost breakdown."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "feature": {"type": "string", "description": "Feature name"},
            },
            "required": ["feature"],
        },
        handler=lambda params: _feature_detail(dataset, str(params["feature"])),
    )


def build_optimization_registry(
    dataset: LogDataset,
    analysis: AnalysisResult,
    routing_sim: RoutingSimResult | None = None,
) -> ToolRegistry:
    """Create tool registry for the optimization agent."""
    registry = ToolRegistry()
    _register_summary_tools(registry, dataset, analysis, routing_sim)
    _register_detail_tools(registry, dataset)
    return registry


def _estimate_switch(
    dataset: LogDataset,
    feature: str,
    target_model: str,
) -> str:
    """Estimate savings from switching a feature to a different model."""
    entries = [e for e in dataset.entries if e.feature == feature]
    if not entries:
        return json.dumps({"error": f"No entries for feature: {feature}"})

    current_cost = sum(e.cost_usd for e in entries)
    projected_cost = sum(
        estimate_cost(target_model, e.input_tokens, e.output_tokens) for e in entries
    )

    return json.dumps(
        {
            "feature": feature,
            "target_model": target_model,
            "total_calls": len(entries),
            "current_cost_usd": round(current_cost, 4),
            "projected_cost_usd": round(projected_cost, 4),
            "savings_usd": round(current_cost - projected_cost, 4),
            "savings_pct": (
                round((current_cost - projected_cost) / current_cost, 4) if current_cost > 0 else 0
            ),
        },
        indent=2,
    )


def _feature_detail(dataset: LogDataset, feature: str) -> str:
    """Get detailed statistics for a feature."""
    entries = [e for e in dataset.entries if e.feature == feature]
    if not entries:
        return json.dumps({"error": f"No entries for feature: {feature}"})

    model_dist: dict[str, int] = defaultdict(int)
    for e in entries:
        model_dist[e.model] += 1

    input_tokens = sorted(e.input_tokens for e in entries)
    errors = sum(1 for e in entries if e.status != "success")

    return json.dumps(
        {
            "feature": feature,
            "total_calls": len(entries),
            "total_cost_usd": round(sum(e.cost_usd for e in entries), 4),
            "model_distribution": dict(model_dist),
            "input_token_percentiles": _percentiles(input_tokens),
            "error_rate": round(errors / len(entries), 4),
        },
        indent=2,
    )


def _get_routing_sim(
    dataset: LogDataset,
    cached: RoutingSimResult | None,
) -> str:
    """Return cached routing sim or compute fresh."""
    if cached is not None:
        return cached.model_dump_json(indent=2)
    from src.agents.routing_sim import simulate_routing

    return simulate_routing(dataset).model_dump_json(indent=2)


def _percentiles(values: list[int]) -> dict[str, int]:
    """Compute percentiles for a sorted list of values."""
    n = len(values)
    if n == 0:
        return {}
    return {
        "p25": values[n // 4],
        "p50": values[n // 2],
        "p75": values[3 * n // 4],
        "p90": values[int(n * 0.9)],
    }
