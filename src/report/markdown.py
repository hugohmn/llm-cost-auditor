"""Markdown report renderer."""

from src.models.analysis import AnalysisResult
from src.models.features import AuditConfig
from src.models.recommendation import OptimizationPlan
from src.models.report import AuditReport
from src.utils.llm_client import MODEL_PRICING


def _fmt_cost(cost: float) -> str:
    """Format a dollar amount, using more decimals for small values."""
    if cost >= 1.0:
        return f"${cost:,.2f}"
    return f"${cost:.4f}"


def _monthly(period_cost: float, days: int) -> float:
    """Project a period cost to monthly."""
    return (period_cost / max(days, 1)) * 30


def _render_header(report: AuditReport) -> list[str]:
    """Render the report header and executive summary."""
    return [
        "# LLM Cost Audit Report",
        "",
        f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M')}",
        f"**Source:** {report.log_source} ({report.total_entries_analyzed:,} entries)",
        f"**Potential savings:** {report.headline_savings}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        report.executive_summary,
        "",
        "---",
        "",
    ]


def _render_cost_overview(analysis: AnalysisResult) -> list[str]:
    """Render cost overview and model/feature breakdown tables."""
    days = analysis.date_range_days
    dedup_monthly = _monthly(analysis.deduplicated_waste_usd, days)
    lines = [
        "## Cost Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Period analyzed | {days} days |",
        f"| Total cost (period) | {_fmt_cost(analysis.total_cost_usd)} |",
        f"| Monthly projected | {_fmt_cost(analysis.monthly_projected_cost_usd)} |",
        f"| Total API calls | {analysis.total_calls:,} |",
        f"| Total tokens | {analysis.total_tokens:,} |",
        f"| Recoverable waste | {_fmt_cost(dedup_monthly)}/month ({analysis.waste_pct:.0%}) |",
        "",
        "## Cost by Model",
        "",
        "| Model | Calls | Cost | % of Total | Avg Input Tk | Error Rate |",
        "|-------|-------|------|------------|-------------|------------|",
    ]
    for m in analysis.cost_by_model:
        lines.append(
            f"| {m.model} | {m.total_calls:,} "
            f"| {_fmt_cost(m.total_cost_usd)} "
            f"| {m.pct_of_total_cost:.0%} "
            f"| {m.avg_input_tokens:,.0f} "
            f"| {m.error_rate:.1%} |"
        )
    lines.append("")

    if analysis.cost_by_feature:
        lines.extend(_render_feature_table(analysis))

    return lines


def _render_feature_table(analysis: AnalysisResult) -> list[str]:
    """Render cost-by-feature table."""
    lines = [
        "## Cost by Feature",
        "",
        "| Feature | Calls | Cost | Primary Model | Avg Tokens |",
        "|---------|-------|------|--------------|------------|",
    ]
    for fb in analysis.cost_by_feature[:10]:
        lines.append(
            f"| {fb.feature} | {fb.total_calls:,} "
            f"| {_fmt_cost(fb.total_cost_usd)} "
            f"| {fb.primary_model} "
            f"| {fb.avg_tokens_per_call:,.0f} |"
        )
    lines.append("")
    return lines


def _render_agent_insights(analysis: AnalysisResult) -> list[str]:
    """Render agent-generated insights section."""
    if not analysis.agent_findings_summary and not analysis.agent_key_insights:
        return []

    lines = ["## Agent Analysis Insights", ""]
    if analysis.agent_findings_summary:
        lines.extend([analysis.agent_findings_summary, ""])
    if analysis.agent_key_insights:
        for insight in analysis.agent_key_insights:
            lines.append(f"- {insight}")
        lines.append("")
    return lines


def _render_waste_patterns(analysis: AnalysisResult) -> list[str]:
    """Render detected waste patterns with confidence and assumptions."""
    if not analysis.waste_patterns:
        return []

    days = analysis.date_range_days
    confidence_label = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}

    lines = ["## Detected Waste Patterns", ""]
    for w in analysis.waste_patterns:
        title = w.pattern_type.replace("_", " ").title()
        monthly_waste = _monthly(w.estimated_waste_usd, days)
        conf = confidence_label.get(w.confidence, w.confidence.upper())
        lines.extend(
            [
                f"### {title}",
                "",
                w.description,
                "",
                f"- **Affected calls:** {w.affected_calls:,}",
                f"- **Estimated waste:** {_fmt_cost(w.estimated_waste_usd)} "
                f"over {days} days ({_fmt_cost(monthly_waste)}/month projected)",
                f"- **Confidence:** {conf}",
                f"- **Methodology:** {w.methodology}",
            ]
        )
        if w.assumptions:
            lines.append("- **Assumptions:**")
            for assumption in w.assumptions:
                lines.append(f"  - {assumption}")
        lines.append("")
    return lines


def _render_routing_sim(optimization: OptimizationPlan) -> list[str]:
    """Render routing simulation results."""
    if not optimization.routing_simulation:
        return []

    rs = optimization.routing_simulation
    lines = [
        "## Routing Simulation",
        "",
        f"Simulated routing simple requests to **{rs.light_model}** "
        f"while keeping complex requests on **{rs.frontier_model}**.",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Current cost | {_fmt_cost(rs.current_cost_usd)} |",
        f"| Optimized cost | {_fmt_cost(rs.optimized_cost_usd)} |",
        f"| Savings | {_fmt_cost(rs.savings_usd)} ({rs.savings_pct:.0%}) |",
        f"| Calls routed to light model | {rs.calls_routed_to_light:,} |",
        f"| Calls kept on frontier | {rs.calls_kept_on_frontier:,} |",
        f"| Light model API error rate | {rs.light_model_error_rate:.2%} |",
        f"| Frontier API error rate | {rs.frontier_error_rate:.2%} |",
        "",
        (
            "**Note:** Error rates measure API-level failures "
            "(timeouts, rate limits, malformed requests), not output "
            "quality. Validating output quality requires an eval set "
            "comparing model responses on representative tasks."
        ),
        "",
    ]
    return lines


def _render_recommendations(
    optimization: OptimizationPlan,
) -> list[str]:
    """Render recommendations section."""
    lines = ["## Recommendations", ""]
    effort_map = {"low": "🟢", "medium": "🟡", "high": "🔴"}

    for i, rec in enumerate(optimization.recommendations, 1):
        emoji = effort_map.get(rec.implementation_effort, "⚪")
        savings = _fmt_cost(rec.estimated_monthly_savings_usd)
        lines.extend(
            [
                f"### {i}. {rec.title}",
                "",
                f"**Category:** {rec.category} | "
                f"**Effort:** {emoji} {rec.implementation_effort} | "
                f"**Savings:** {savings}/month",
                "",
                rec.description,
            ]
        )
        if rec.details:
            lines.extend(["", f"*Implementation:* {rec.details}"])
        lines.append("")
    return lines


def _render_methodology(
    report: AuditReport,
    config: AuditConfig,
) -> list[str]:
    """Render comprehensive methodology section."""
    lines = [
        "---",
        "",
        f"**Total potential savings: {report.headline_savings}**",
        "",
        "## Methodology",
        "",
        "### Cost computation",
        "",
        "Costs are taken from the source data when available "
        "(e.g., LangFuse `calculatedTotalCost`). When missing, "
        "costs are estimated using published per-token pricing:",
        "",
        "| Model | Input (per 1M tokens) | Output (per 1M tokens) |",
        "|-------|-----------------------|------------------------|",
    ]
    for model, (inp, out) in sorted(MODEL_PRICING.items()):
        lines.append(f"| {model} | ${inp:.2f} | ${out:.2f} |")
    lines.extend(
        [
            "",
            "Counterfactual savings (e.g., 'what if we used Haiku?') "
            "use the same pricing table. If your contract includes "
            "volume discounts or custom rates, actual savings may differ.",
            "",
        ]
    )

    lines.extend(_render_feature_classification(config))
    lines.extend(_render_detection_thresholds(config))
    lines.extend(
        [
            "### What error rates measure",
            "",
            "Error rates in this report count API-level failures "
            "(HTTP errors, timeouts, rate limits) — entries where "
            "`status != 'success'`. They do **not** measure output "
            "quality, accuracy, or task completion. A call that returns "
            "a low-quality response with HTTP 200 is counted as 'success'.",
            "",
            "*Report generated by LLM Cost Auditor*",
        ]
    )
    return lines


def _render_feature_classification(config: AuditConfig) -> list[str]:
    """Render the feature complexity table."""
    lines = [
        "### Feature complexity classification",
        "",
        "The wrong-model detector and routing simulation depend on "
        "this classification. Features not listed default to COMPLEX "
        "(conservative — never flagged as waste).",
        "",
        "| Feature | Classification |",
        "|---------|---------------|",
    ]
    for feature, complexity in sorted(config.feature_complexity.items()):
        lines.append(f"| {feature} | {complexity.value} |")
    lines.extend(
        [
            "",
            "This classification is a **configurable input**, not "
            "derived from the data. Adjust it to match your domain.",
            "",
        ]
    )
    return lines


def _render_detection_thresholds(config: AuditConfig) -> list[str]:
    """Render the detection thresholds table."""
    pct = config.duplicate_token_tolerance * 100
    return [
        "### Detection thresholds",
        "",
        "| Parameter | Value | Used by |",
        "|-----------|-------|---------|",
        (
            f"| Bloated prompt multiplier | {config.bloated_prompt_multiplier:.1f}x median "
            f"| Bloated prompt detector |"
        ),
        (
            f"| Bloated prompt minimum | {config.bloated_prompt_min_tokens:,} tokens "
            f"| Bloated prompt detector |"
        ),
        (
            f"| Moderate input threshold | {config.moderate_input_threshold:,} tokens "
            f"| Wrong model + routing sim |"
        ),
        (
            f"| Duplicate time window | {config.duplicate_window_seconds}s "
            f"| Cacheable duplicate detector |"
        ),
        (f"| Duplicate token tolerance | {pct:.0f}% | Cacheable duplicate detector |"),
        (f"| Retry time window | {config.retry_window_seconds}s | Excessive retry detector |"),
        "",
        "These thresholds are configurable defaults. Tighter values "
        "reduce false positives but may miss real waste.",
        "",
    ]


def render_markdown(report: AuditReport) -> str:
    """Render an AuditReport as a Markdown document."""
    config = report.analysis.audit_config
    lines: list[str] = []
    lines.extend(_render_header(report))
    lines.extend(_render_cost_overview(report.analysis))
    lines.extend(_render_agent_insights(report.analysis))
    lines.extend(_render_waste_patterns(report.analysis))
    lines.extend(_render_routing_sim(report.optimization))
    lines.extend(_render_recommendations(report.optimization))
    lines.extend(_render_methodology(report, config))
    return "\n".join(lines)
