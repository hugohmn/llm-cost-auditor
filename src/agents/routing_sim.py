"""Routing Simulation Agent — simulates multi-model routing and estimates savings.

Uses feature-based complexity classification to determine routing.
LLM is used only for generating actionable recommendations.
"""

import json
import logging

from pydantic import ValidationError

from src.models.analysis import AnalysisResult
from src.models.features import AuditConfig, TaskComplexity
from src.models.log_entry import LogDataset
from src.models.recommendation import (
    OptimizationPlan,
    Recommendation,
    RoutingSimResult,
)
from src.utils.json_extract import parse_json_response
from src.utils.llm_client import UnifiedLLMClient, estimate_cost
from src.utils.model_utils import is_light_model as _is_light_model
from src.utils.retry import with_retry

logger = logging.getLogger(__name__)

ROUTING_SYSTEM_PROMPT = """\
You are an LLM cost optimization expert.

Given analysis results and computed waste patterns from an LLM deployment, \
generate a prioritized list of actionable optimization recommendations.

The waste pattern numbers below are computed deterministically from the data. \
Use these exact figures in your recommendations. Do not invent alternative numbers.

Respond ONLY with a JSON array. Each item must have these fields:
{
  "priority": 1-5 (1 = highest ROI),
  "category": "routing | prompt_optimization | caching | model_switch | architecture",
  "title": "Short actionable title",
  "description": "Clear explanation",
  "estimated_monthly_savings_usd": <number>,
  "implementation_effort": "low | medium | high",
  "details": "Technical implementation steps"
}

RULES:
- Provide 3-7 recommendations where data supports them
- If routing_simulation.savings is 0, do not include a routing recommendation
- NEVER claim "quality retention" percentages — error rates measure API failures \
only, not output quality. Do NOT fabricate quality scores
- When recommending model switches, state that quality validation is required"""


def _compute_error_rates(
    dataset: LogDataset,
) -> tuple[float, float]:
    """Compute observed error rates for light and frontier models."""
    light_total = 0
    light_errors = 0
    frontier_total = 0
    frontier_errors = 0

    for entry in dataset.entries:
        if _is_light_model(entry.model):
            light_total += 1
            if entry.status != "success":
                light_errors += 1
        else:
            frontier_total += 1
            if entry.status != "success":
                frontier_errors += 1

    light_rate = light_errors / light_total if light_total > 0 else 0.0
    frontier_rate = frontier_errors / frontier_total if frontier_total > 0 else 0.0
    return light_rate, frontier_rate


def simulate_routing(
    dataset: LogDataset,
    config: AuditConfig | None = None,
) -> RoutingSimResult:
    """Simulate routing based on feature complexity tiers.

    SIMPLE features always route to light model.
    MODERATE features route to light model if input < threshold.
    COMPLEX features always stay on frontier.
    """
    if config is None:
        config = AuditConfig()

    current_cost = 0.0
    optimized_cost = 0.0
    routed_light = 0
    kept_frontier = 0
    already_light = 0

    for entry in dataset.entries:
        current_cost += entry.cost_usd

        if _is_light_model(entry.model):
            optimized_cost += entry.cost_usd
            already_light += 1
            continue

        complexity = config.feature_complexity.get(
            entry.feature or "",
            TaskComplexity.COMPLEX,
        )
        should_route = complexity == TaskComplexity.SIMPLE or (
            complexity == TaskComplexity.MODERATE
            and entry.input_tokens < config.moderate_input_threshold
        )

        if should_route and entry.status == "success":
            optimized_cost += estimate_cost(
                config.light_model,
                entry.input_tokens,
                entry.output_tokens,
            )
            routed_light += 1
        else:
            optimized_cost += entry.cost_usd
            kept_frontier += 1

    savings = current_cost - optimized_cost
    savings_pct = max(0.0, min(savings / current_cost, 1.0)) if current_cost > 0 else 0.0

    light_err, frontier_err = _compute_error_rates(dataset)

    return RoutingSimResult(
        current_cost_usd=current_cost,
        optimized_cost_usd=optimized_cost,
        savings_usd=savings,
        savings_pct=savings_pct,
        calls_routed_to_light=routed_light,
        calls_kept_on_frontier=kept_frontier,
        light_model=config.light_model,
        frontier_model=config.frontier_model,
        light_model_error_rate=round(light_err, 4),
        frontier_error_rate=round(frontier_err, 4),
    )


def _build_recommendation_context(
    analysis: AnalysisResult,
    routing_sim: RoutingSimResult,
) -> str:
    """Build the JSON context string for the recommendations prompt."""
    context = {
        "total_monthly_cost_usd": round(analysis.monthly_projected_cost_usd, 2),
        "total_calls": analysis.total_calls,
        "waste_patterns": [
            {
                "type": w.pattern_type,
                "description": w.description,
                "waste_usd": w.estimated_waste_usd,
                "methodology": w.methodology,
            }
            for w in analysis.waste_patterns
        ],
        "routing_simulation": {
            "current_cost": round(routing_sim.current_cost_usd, 2),
            "optimized_cost": round(routing_sim.optimized_cost_usd, 2),
            "savings": round(routing_sim.savings_usd, 2),
            "calls_routable": routing_sim.calls_routed_to_light,
        },
        "top_models_by_cost": [
            {
                "model": m.model,
                "cost": round(m.total_cost_usd, 2),
                "calls": m.total_calls,
            }
            for m in analysis.cost_by_model[:5]
        ],
    }
    return json.dumps(context, indent=2)


def _parse_recommendations(raw_content: str) -> list[Recommendation]:
    """Parse LLM response text into validated Recommendation objects."""
    try:
        recs_raw = parse_json_response(raw_content)
        if isinstance(recs_raw, dict):
            recs_raw = [recs_raw]
        results: list[Recommendation] = []
        for r in recs_raw:
            try:
                if "priority" in r:
                    r["priority"] = max(1, min(5, int(r["priority"])))
                results.append(Recommendation(**r))
            except (TypeError, ValueError, ValidationError) as e:
                logger.debug("Skipping invalid recommendation: %s", e)
        return results
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("Failed to parse recommendations: %s", e)
        return []


async def generate_recommendations(
    analysis: AnalysisResult,
    routing_sim: RoutingSimResult,
    client: UnifiedLLMClient,
) -> list[Recommendation]:
    """Use LLM to generate optimization recommendations."""
    context_json = _build_recommendation_context(analysis, routing_sim)
    response = await with_retry(
        client.complete,
        system=ROUTING_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Generate optimization recommendations "
                    f"based on this analysis:\n\n{context_json}"
                ),
            }
        ],
        temperature=0.0,
        agent_name="routing_sim_agent",
    )
    return _parse_recommendations(response.content)


async def optimize(
    dataset: LogDataset,
    analysis: AnalysisResult,
    client: UnifiedLLMClient,
) -> OptimizationPlan:
    """Run routing simulation and generate recommendations."""
    routing_sim = simulate_routing(dataset, analysis.audit_config)

    logger.info(
        "Routing simulation: %.0f%% savings potential ($%.2f), %d calls routable to %s",
        routing_sim.savings_pct * 100,
        routing_sim.savings_usd,
        routing_sim.calls_routed_to_light,
        routing_sim.light_model,
    )

    recommendations = await generate_recommendations(analysis, routing_sim, client)
    raw_total = sum(r.estimated_monthly_savings_usd for r in recommendations)

    # Cap at deduplicated waste ceiling (projected to monthly)
    days = max(analysis.date_range_days, 1)
    monthly_waste_ceiling = (analysis.deduplicated_waste_usd / days) * 30
    total_savings = min(raw_total, monthly_waste_ceiling)

    monthly = analysis.monthly_projected_cost_usd
    savings_pct = min(total_savings / monthly, 1.0) if monthly > 0 else 0.0

    return OptimizationPlan(
        routing_simulation=routing_sim,
        recommendations=sorted(recommendations, key=lambda r: r.priority),
        total_potential_savings_usd=round(total_savings, 2),
        total_potential_savings_pct=savings_pct,
    )
