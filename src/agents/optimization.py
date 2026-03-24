"""Optimization Agent — generates actionable recommendations via tool-use loop.

Uses simulation tools to quantify savings and produces prioritized
recommendations. Falls back to deterministic path on failure.
"""

import json
import logging

import anthropic
from pydantic import ValidationError

from src.agents.base import AgentLoopExhaustedError, run_agent_loop
from src.agents.routing_sim import optimize as deterministic_optimize
from src.agents.routing_sim import simulate_routing
from src.models.analysis import AnalysisResult
from src.models.log_entry import LogDataset
from src.models.recommendation import (
    OptimizationPlan,
    Recommendation,
    RoutingSimResult,
)
from src.tools.optimization_tools import build_optimization_registry
from src.utils.json_extract import parse_json_response
from src.utils.llm_client import UnifiedLLMClient

logger = logging.getLogger(__name__)

OPTIMIZATION_SYSTEM_PROMPT = """\
You are an LLM cost optimization strategist. You have analysis results \
from a cost audit and access to simulation tools.

Your goal: produce a prioritized list of actionable optimization \
recommendations with concrete savings estimates.

WORKFLOW:
1. Call get_analysis_summary to review waste patterns and cost breakdowns
2. Call simulate_routing to quantify multi-model routing savings
3. Use estimate_model_switch_savings to validate specific feature switches
4. Use get_feature_detail to investigate high-cost features
5. Synthesize into prioritized recommendations

OUTPUT FORMAT: After investigation, output ONLY a JSON array of recommendations:
[
  {
    "priority": 1,
    "category": "routing | prompt_optimization | caching | model_switch | architecture",
    "title": "Short actionable title",
    "description": "Clear explanation of the recommendation",
    "estimated_monthly_savings_usd": 42.50,
    "implementation_effort": "low | medium | high",
    "details": "Technical implementation steps"
  }
]

RULES:
- Base ALL savings numbers on tool results — never invent figures
- Each recommendation must be specific and actionable (not "consider optimizing")
- Priority 1 = highest impact, easiest to implement
- Provide 3-7 recommendations, covering routing, caching, and prompt optimization
- Use exact dollar amounts from simulations"""


async def run_optimization_agent(
    dataset: LogDataset,
    analysis: AnalysisResult,
    client: UnifiedLLMClient,
) -> OptimizationPlan:
    """Run agentic optimization with tool-use loop.

    Falls back to deterministic optimize() on failure.
    """
    try:
        return await _run_agentic_optimization(dataset, analysis, client)
    except (AgentLoopExhaustedError, anthropic.APIError, ValueError) as e:
        logger.warning("Optimization agent failed, using deterministic fallback: %s", e)
        return await deterministic_optimize(dataset, analysis, client)


async def _run_agentic_optimization(
    dataset: LogDataset,
    analysis: AnalysisResult,
    client: UnifiedLLMClient,
) -> OptimizationPlan:
    """Internal: run the agentic optimization loop."""
    # Compute routing sim once — used both as a tool result and in final output
    routing_result = simulate_routing(dataset, analysis.audit_config)
    registry = build_optimization_registry(dataset, analysis, routing_result)

    initial_message = (
        "Analyze the cost audit results and generate optimization "
        "recommendations.\n\n"
        f"Summary: {analysis.total_calls} calls analyzed, "
        f"${analysis.monthly_projected_cost_usd:.2f}/month projected, "
        f"${analysis.total_waste_usd:.2f} waste detected "
        f"({analysis.waste_pct:.0%}).\n\n"
        "Use your tools to investigate and produce actionable recommendations."
    )

    agent_result = await run_agent_loop(
        client=client,
        system_prompt=OPTIMIZATION_SYSTEM_PROMPT,
        initial_message=initial_message,
        registry=registry,
        max_iterations=10,
        agent_name="optimization_agent",
    )

    return _build_plan_from_agent(
        routing_result,
        analysis,
        agent_result.final_content,
    )


def _build_plan_from_agent(
    routing_result: RoutingSimResult,
    analysis: AnalysisResult,
    agent_output: str,
) -> OptimizationPlan:
    """Build OptimizationPlan from agent output + pre-computed routing sim.

    Caps total savings at the deduplicated waste ceiling — LLM-generated
    recommendations may double-count overlapping optimizations.
    """
    recommendations = _parse_recommendations(agent_output)

    raw_total = sum(r.estimated_monthly_savings_usd for r in recommendations)
    # Scale to monthly: waste is period-total, project to monthly
    days = max(analysis.date_range_days, 1)
    monthly_waste_ceiling = (analysis.deduplicated_waste_usd / days) * 30
    total_savings = min(raw_total, monthly_waste_ceiling)

    monthly = analysis.monthly_projected_cost_usd
    savings_pct = min(total_savings / monthly, 1.0) if monthly > 0 else 0.0

    return OptimizationPlan(
        routing_simulation=routing_result,
        recommendations=sorted(recommendations, key=lambda r: r.priority),
        total_potential_savings_usd=round(total_savings, 2),
        total_potential_savings_pct=savings_pct,
    )


def _parse_recommendations(agent_output: str) -> list[Recommendation]:
    """Parse recommendations from agent JSON output.

    Validates each recommendation individually — skips invalid ones
    rather than failing the entire batch.
    """
    try:
        recs_raw = parse_json_response(agent_output)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("Failed to parse agent recommendations JSON: %s", e)
        return []

    if isinstance(recs_raw, dict):
        recs_raw = [recs_raw]

    results: list[Recommendation] = []
    for raw in recs_raw:
        try:
            # Clamp priority to valid range
            if "priority" in raw:
                raw["priority"] = max(1, min(5, int(raw["priority"])))
            results.append(Recommendation(**raw))
        except (TypeError, ValueError, ValidationError) as e:
            logger.debug("Skipping invalid recommendation: %s", e)
    return results
