"""Report Agent — generates the final audit report with executive summary.

Provides both:
  - Agentic report generation via run_report_agent() (tool-use loop)
  - Deterministic fallback via build_report() (single LLM call)
"""

import logging

import anthropic

from src.agents.base import AgentLoopExhaustedError, run_agent_loop
from src.models.analysis import AnalysisResult
from src.models.log_entry import LogDataset
from src.models.recommendation import OptimizationPlan
from src.models.report import AuditReport
from src.tools.report_tools import build_report_registry
from src.utils.llm_client import UnifiedLLMClient
from src.utils.retry import with_retry

logger = logging.getLogger(__name__)

REPORT_SYSTEM_PROMPT = """\
You are a senior AI infrastructure consultant writing an executive summary \
for a CTO or VP Engineering.

You have access to tools that retrieve specific data from a completed \
LLM cost audit. Use them to gather the numbers you need.

After investigation, write a 3-4 paragraph executive summary. Be direct, \
specific, and actionable. Use exact numbers from the tools. No fluff.

Structure:
1. Current state (spend, volume, key observation)
2. Main findings (biggest waste sources with methodology)
3. Recommended actions with expected impact
4. Bottom line (total potential savings, implementation priority)

IMPORTANT RULES:
- NEVER claim "quality retention" percentages — error rates in the data measure \
API failures (timeouts, rate limits), not output quality. You may cite observed \
error rates, but do NOT fabricate quality scores or retention metrics.
- State that output quality validation requires an eval set when recommending \
model switches.

OUTPUT: Write the summary as plain text (not JSON). This will be read \
directly by the CTO."""

# Fallback prompt for single-call mode (no tools)
_FALLBACK_SYSTEM_PROMPT = """\
You are a senior AI infrastructure consultant writing an executive summary \
for a CTO or VP Engineering.

Write a 3-4 paragraph executive summary of an LLM cost audit. Be direct, \
specific, and actionable. Use exact numbers. No fluff.

Structure:
1. Current state (spend, volume, key observation)
2. Main findings (biggest waste sources)
3. Recommended actions with expected impact
4. Bottom line (total potential savings)"""


async def run_report_agent(
    dataset: LogDataset,
    analysis: AnalysisResult,
    optimization: OptimizationPlan,
    client: UnifiedLLMClient,
) -> AuditReport:
    """Run agentic report generation with tool-use loop.

    Falls back to single LLM call on failure.
    """
    try:
        summary = await _run_agentic_report(dataset, analysis, optimization, client)
    except (AgentLoopExhaustedError, anthropic.APIError, ValueError) as e:
        logger.warning("Report agent failed, using fallback: %s", e)
        summary = await _generate_fallback_summary(analysis, optimization, client)

    return _assemble_report(dataset, analysis, optimization, summary)


async def _run_agentic_report(
    dataset: LogDataset,
    analysis: AnalysisResult,
    optimization: OptimizationPlan,
    client: UnifiedLLMClient,
) -> str:
    """Internal: run the agentic report loop."""
    registry = build_report_registry(dataset, analysis, optimization)

    initial_message = (
        "Write an executive summary for this LLM cost audit.\n\n"
        f"Quick context: {analysis.total_calls} calls analyzed over "
        f"{analysis.date_range_days} days, "
        f"${analysis.monthly_projected_cost_usd:.2f}/month projected.\n\n"
        "Use your tools to gather the specific numbers you need, then write the summary."
    )

    agent_result = await run_agent_loop(
        client=client,
        system_prompt=REPORT_SYSTEM_PROMPT,
        initial_message=initial_message,
        registry=registry,
        max_iterations=8,
        agent_name="report_agent",
        temperature=0.2,
    )

    return agent_result.final_content


def _build_summary_context(
    analysis: AnalysisResult,
    optimization: OptimizationPlan,
) -> str:
    """Build the context string for the fallback executive summary."""
    lines = [
        f"Monthly projected cost: ${analysis.monthly_projected_cost_usd:,.2f}",
        f"Total calls analyzed: {analysis.total_calls:,}",
        f"Date range: {analysis.date_range_days} days",
        f"Detected waste: ${analysis.total_waste_usd:,.2f} ({analysis.waste_pct:.0%})",
    ]
    if analysis.cost_by_model:
        top = analysis.cost_by_model[0]
        lines.append(
            f"Top model by cost: {top.model} "
            f"(${top.total_cost_usd:,.2f}, {top.pct_of_total_cost:.0%} of total)"
        )

    if optimization.routing_simulation:
        rs = optimization.routing_simulation
        lines.append(
            f"\nRouting simulation: {rs.savings_pct:.0%} savings "
            f"(${rs.savings_usd:,.2f}) by routing "
            f"{rs.calls_routed_to_light:,} simple calls to {rs.light_model}"
        )

    savings = optimization.total_potential_savings_usd
    lines.append(f"\nTotal potential monthly savings: ${savings:,.2f}")
    lines.append("\nTop recommendations:")
    for rec in optimization.recommendations[:5]:
        lines.append(
            f"- [{rec.category}] {rec.title}: "
            f"${rec.estimated_monthly_savings_usd:,.0f}/mo "
            f"({rec.implementation_effort} effort)"
        )
    return "\n".join(lines)


async def _generate_fallback_summary(
    analysis: AnalysisResult,
    optimization: OptimizationPlan,
    client: UnifiedLLMClient,
) -> str:
    """Generate executive summary with a single LLM call (no tools)."""
    context = _build_summary_context(analysis, optimization)
    response = await with_retry(
        client.complete,
        system=_FALLBACK_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
        temperature=0.2,
        agent_name="report_agent_fallback",
    )
    return response.content


def _assemble_report(
    dataset: LogDataset,
    analysis: AnalysisResult,
    optimization: OptimizationPlan,
    summary: str,
) -> AuditReport:
    """Assemble the final AuditReport."""
    report = AuditReport(
        log_source=dataset.source_format,
        total_entries_analyzed=dataset.total_entries,
        analysis=analysis,
        optimization=optimization,
        executive_summary=summary,
    )
    logger.info("Report generated: %s potential savings", report.headline_savings)
    return report
