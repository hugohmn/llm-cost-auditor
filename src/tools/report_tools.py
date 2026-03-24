"""Report tools — data-fetching tools for the report agent.

The report agent uses these to pull specific data points
for constructing the executive summary.
"""

import json
import logging

from src.models.analysis import AnalysisResult
from src.models.log_entry import LogDataset
from src.models.quality import JudgeEvalNotRun, QualityAssessment
from src.models.recommendation import OptimizationPlan
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _register_overview_tools(
    registry: ToolRegistry,
    analysis: AnalysisResult,
    optimization: OptimizationPlan,
) -> None:
    """Register cost overview, recommendations, and waste pattern tools."""
    registry.register(
        name="get_cost_overview",
        description=(
            "Get overall cost metrics: total spend, monthly projection, "
            "waste percentage, date range, and call volume."
        ),
        handler=lambda _: _cost_overview(analysis),
    )

    registry.register(
        name="get_top_recommendations",
        description=(
            "Get the prioritized list of optimization recommendations "
            "with estimated monthly savings and implementation effort."
        ),
        handler=lambda _: json.dumps(
            [r.model_dump() for r in optimization.recommendations],
            indent=2,
        ),
    )

    registry.register(
        name="get_waste_patterns",
        description=(
            "Get all detected waste patterns with descriptions, affected "
            "call counts, waste amounts, and detection methodology."
        ),
        handler=lambda _: json.dumps(
            [w.model_dump() for w in analysis.waste_patterns],
            indent=2,
        ),
    )


def _register_detail_report_tools(
    registry: ToolRegistry,
    analysis: AnalysisResult,
    optimization: OptimizationPlan,
) -> None:
    """Register routing simulation and cost driver tools."""
    registry.register(
        name="get_routing_simulation",
        description=(
            "Get routing simulation results: current vs optimized cost, "
            "savings percentage, calls routed to light model, and observed error rates."
        ),
        handler=lambda _: (
            optimization.routing_simulation.model_dump_json(indent=2)
            if optimization.routing_simulation
            else json.dumps({"result": None})
        ),
    )

    registry.register(
        name="get_top_cost_drivers",
        description=(
            "Get the top 5 models and features by cost with detailed "
            "breakdowns including call counts and percentage of total spend."
        ),
        handler=lambda _: _top_drivers(analysis),
    )


def build_report_registry(
    dataset: LogDataset,
    analysis: AnalysisResult,
    optimization: OptimizationPlan,
    quality: QualityAssessment | None = None,
) -> ToolRegistry:
    """Create tool registry for the report agent."""
    registry = ToolRegistry()
    _register_overview_tools(registry, analysis, optimization)
    _register_detail_report_tools(registry, analysis, optimization)
    if quality:
        _register_quality_tool(registry, quality)
    return registry


def _register_quality_tool(
    registry: ToolRegistry,
    quality: QualityAssessment,
) -> None:
    """Register quality evaluation tool for the report agent."""
    registry.register(
        name="get_quality_evaluation",
        description=(
            "Get quality evaluation results: proxy signal comparisons "
            "between frontier and light models per feature, LLM-as-Judge "
            "results if available, and evaluation plans for routing changes."
        ),
        handler=lambda _: _quality_summary(quality),
    )


def _quality_summary(quality: QualityAssessment) -> str:
    """Build compact quality evaluation summary for the report agent."""
    sections: list[str] = []

    # Proxy signals
    proxy = quality.proxy_signals
    if proxy.feature_signals:
        signals = [
            {
                "feature": s.feature,
                "output_ratio": s.output_ratio,
                "confidence": s.confidence.value,
                "interpretation": s.interpretation,
            }
            for s in proxy.feature_signals
        ]
        sections.append(json.dumps({"proxy_signals": signals}, indent=2))

    if proxy.features_without_comparison:
        sections.append(
            f"Features without light model data: {', '.join(proxy.features_without_comparison)}"
        )

    # Judge eval
    judge = quality.judge_eval
    if isinstance(judge, JudgeEvalNotRun):
        sections.append(f"Judge evaluation: not run ({judge.reason})")
    else:
        results = [
            {
                "feature": r.feature,
                "verdict": r.verdict.value,
                "mean_delta": r.mean_quality_delta,
                "sample_size": r.sample_size,
            }
            for r in judge.feature_results
        ]
        sections.append(
            json.dumps(
                {"judge_eval": results, "cost": judge.total_eval_cost_usd},
                indent=2,
            )
        )

    # Eval plans count
    plans = quality.eval_plans
    sections.append(
        f"Eval plans: {len(plans.feature_plans)} features, "
        f"${plans.total_estimated_cost_usd:.2f} estimated cost"
    )

    return "\n\n".join(sections)


def _cost_overview(analysis: AnalysisResult) -> str:
    """Build cost overview JSON."""
    return json.dumps(
        {
            "total_cost_usd": round(analysis.total_cost_usd, 2),
            "monthly_projected_cost_usd": round(analysis.monthly_projected_cost_usd, 2),
            "total_calls": analysis.total_calls,
            "total_tokens": analysis.total_tokens,
            "date_range_days": analysis.date_range_days,
            "gross_waste_usd": round(analysis.total_waste_usd, 2),
            "deduplicated_waste_usd": round(analysis.deduplicated_waste_usd, 2),
            "waste_pct": round(analysis.waste_pct, 4),
            "num_waste_patterns": len(analysis.waste_patterns),
        },
        indent=2,
    )


def _top_drivers(analysis: AnalysisResult) -> str:
    """Build top cost drivers JSON."""
    return json.dumps(
        {
            "top_models": [
                {
                    "model": m.model,
                    "cost_usd": round(m.total_cost_usd, 2),
                    "calls": m.total_calls,
                    "pct_of_total": round(m.pct_of_total_cost, 3),
                }
                for m in analysis.cost_by_model[:5]
            ],
            "top_features": [
                {
                    "feature": f.feature,
                    "cost_usd": round(f.total_cost_usd, 2),
                    "calls": f.total_calls,
                    "primary_model": f.primary_model,
                }
                for f in analysis.cost_by_feature[:5]
            ],
        },
        indent=2,
    )
