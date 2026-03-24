"""Analysis Agent — analyzes normalized logs to detect cost patterns and waste.

Provides both:
  - Deterministic analysis functions (used as tool handlers + fallback)
  - Agentic analysis via run_analysis_agent() using tool-use loops

All detectors accept an AuditConfig so every threshold is explicit
and configurable — no hardcoded magic numbers.
"""

import json
import logging
import statistics
from collections import defaultdict

from src.models.analysis import (
    AnalysisResult,
    FeatureCostBreakdown,
    ModelCostBreakdown,
    WastePattern,
)
from src.models.features import AuditConfig, TaskComplexity
from src.models.log_entry import LogDataset, LogEntry
from src.utils.llm_client import UnifiedLLMClient, estimate_cost

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """\
You are an LLM cost analysis expert. You have access to tools that analyze \
a dataset of LLM API call logs.

Your goal: produce a comprehensive cost analysis identifying waste patterns \
and quantifying their impact.

WORKFLOW:
1. Start with get_dataset_summary to understand the data
2. Run compute_cost_by_model and compute_cost_by_feature for breakdowns
3. Run each waste detector to find specific waste patterns
4. Optionally run simulate_routing to quantify routing savings
5. Synthesize all findings into a structured JSON analysis

OUTPUT FORMAT: After gathering all data, output a JSON object:
{
  "findings_summary": "2-3 sentence overview of key findings",
  "key_insights": ["insight1", "insight2", ...],
  "additional_observations": "Any patterns not captured by standard detectors"
}

RULES:
- Always call get_dataset_summary first
- Use EXACT numbers from tool results — do not estimate or round differently
- If a waste detector returns null, note it as "not detected" — do not invent waste
- Be specific: name models, features, and dollar amounts
- NEVER claim "quality retention" percentages — error rates measure API failures, \
not output quality. You may compare observed error rates between models, but do NOT \
extrapolate them into a quality retention score"""


def _is_light_model(model: str) -> bool:
    """Check if a model is a light/cheap model."""
    lower = model.lower()
    return "haiku" in lower or "mini" in lower


def compute_cost_by_model(
    dataset: LogDataset,
) -> list[ModelCostBreakdown]:
    """Compute cost breakdown by model — pure computation."""
    groups: dict[str, list[LogEntry]] = defaultdict(list)
    for entry in dataset.entries:
        groups[entry.model].append(entry)

    total_cost = dataset.total_cost_usd
    breakdowns = []

    for model, entries in sorted(groups.items(), key=lambda x: -sum(e.cost_usd for e in x[1])):
        cost = sum(e.cost_usd for e in entries)
        errors = sum(1 for e in entries if e.status != "success")
        latencies = [e.latency_ms for e in entries if e.latency_ms is not None]

        breakdowns.append(
            ModelCostBreakdown(
                model=model,
                total_calls=len(entries),
                total_tokens=sum(e.total_tokens for e in entries),
                total_cost_usd=cost,
                avg_input_tokens=(sum(e.input_tokens for e in entries) / len(entries)),
                avg_output_tokens=(sum(e.output_tokens for e in entries) / len(entries)),
                avg_latency_ms=(sum(latencies) / len(latencies) if latencies else None),
                error_rate=errors / len(entries) if entries else 0,
                pct_of_total_cost=(cost / total_cost if total_cost > 0 else 0),
            )
        )
    return breakdowns


def compute_cost_by_feature(
    dataset: LogDataset,
) -> list[FeatureCostBreakdown]:
    """Compute cost breakdown by feature — pure computation."""
    groups: dict[str, list[LogEntry]] = defaultdict(list)
    for entry in dataset.entries:
        feature = entry.feature or "unknown"
        groups[feature].append(entry)

    breakdowns = []
    for feature, entries in sorted(groups.items(), key=lambda x: -sum(e.cost_usd for e in x[1])):
        model_counts: dict[str, int] = defaultdict(int)
        for e in entries:
            model_counts[e.model] += 1
        primary_model = max(
            model_counts,
            key=model_counts.get,  # type: ignore[arg-type]
        )

        breakdowns.append(
            FeatureCostBreakdown(
                feature=feature,
                total_calls=len(entries),
                total_cost_usd=sum(e.cost_usd for e in entries),
                primary_model=primary_model,
                avg_tokens_per_call=(sum(e.total_tokens for e in entries) / len(entries)),
            )
        )
    return breakdowns


# ── Deterministic waste detectors ───────────────────────────────────────


def detect_bloated_prompts(
    dataset: LogDataset,
    config: AuditConfig,
) -> WastePattern | None:
    """Detect prompts with excessive input tokens on routable features."""
    routable = config.routable_features
    multiplier = config.bloated_prompt_multiplier
    min_tokens = config.bloated_prompt_min_tokens
    min_entries = config.bloated_prompt_min_feature_entries

    feature_tokens: dict[str, list[int]] = defaultdict(list)
    for e in dataset.entries:
        if e.feature in routable and e.status == "success":
            feature_tokens[e.feature].append(e.input_tokens)

    bloated_count = 0
    total_waste = 0.0

    for feature, token_list in feature_tokens.items():
        if len(token_list) < min_entries:
            continue
        median = statistics.median(token_list)
        threshold = max(median * multiplier, min_tokens)

        for e in dataset.entries:
            if e.feature != feature or e.status != "success":
                continue
            if e.input_tokens > threshold:
                excess = e.input_tokens - int(median)
                total_waste += estimate_cost(e.model, excess, 0)
                bloated_count += 1

    if bloated_count == 0:
        return None

    return WastePattern(
        pattern_type="bloated_prompt",
        description=(
            f"{bloated_count} calls on routable features have input "
            f"tokens >{multiplier:.0f}x the feature median. Excess tokens "
            f"cost ${total_waste:.2f} at published model pricing."
        ),
        affected_calls=bloated_count,
        estimated_waste_usd=round(total_waste, 2),
        methodology=(
            f"Per-feature median input tokens computed. "
            f"Entries >{multiplier:.0f}x median (min {min_tokens:,}) flagged. "
            f"Waste = excess tokens x model input price."
        ),
        confidence="medium",
        assumptions=[
            f"Threshold: >{multiplier:.0f}x feature median or >{min_tokens:,} tokens",
            "Median represents a 'normal' prompt size for the feature",
            "Excess tokens can be trimmed without degrading output quality",
            "Only input token cost counted (output length assumed unchanged)",
        ],
    )


def detect_wrong_model(
    dataset: LogDataset,
    config: AuditConfig,
) -> WastePattern | None:
    """Detect frontier models used for tasks a light model could handle."""
    wrong_count = 0
    total_waste = 0.0
    threshold = config.moderate_input_threshold

    for e in dataset.entries:
        if e.status != "success" or _is_light_model(e.model):
            continue

        complexity = config.feature_complexity.get(e.feature or "")
        if complexity is None or complexity == TaskComplexity.COMPLEX:
            continue

        is_wasteful = complexity == TaskComplexity.SIMPLE or (
            complexity == TaskComplexity.MODERATE and e.input_tokens < threshold
        )
        if not is_wasteful:
            continue

        haiku_cost = estimate_cost(config.light_model, e.input_tokens, e.output_tokens)
        savings = e.cost_usd - haiku_cost
        if savings > 0:
            total_waste += savings
            wrong_count += 1

    if wrong_count == 0:
        return None

    classified = {
        f: c.value for f, c in config.feature_complexity.items() if c != TaskComplexity.COMPLEX
    }

    return WastePattern(
        pattern_type="wrong_model",
        description=(
            f"{wrong_count} calls use frontier models for tasks "
            f"classified as simple/moderate. Switching to "
            f"{config.light_model} would save ${total_waste:.2f} "
            f"(requires output quality validation)."
        ),
        affected_calls=wrong_count,
        estimated_waste_usd=round(total_waste, 2),
        methodology=(
            "SIMPLE features always flagged. MODERATE features "
            f"flagged when input < {threshold:,} tokens. "
            f"Waste = actual cost - {config.light_model} equivalent cost."
        ),
        confidence="medium",
        assumptions=[
            f"Feature classification (user-configured): {classified}",
            "Features not classified default to COMPLEX (not flagged)",
            f"Light model ({config.light_model}) can handle these tasks — "
            f"NOT validated by output quality eval",
            "Savings based on published per-token pricing",
        ],
    )


def detect_excessive_retries(
    dataset: LogDataset,
    config: AuditConfig,
) -> WastePattern | None:
    """Detect error-then-success retry chains."""
    window = config.retry_window_seconds
    groups: dict[tuple[str, str], list[LogEntry]] = defaultdict(list)
    for e in dataset.entries:
        key = (e.feature or "unknown", e.model)
        groups[key].append(e)

    retry_errors = 0
    total_waste = 0.0

    for entries in groups.values():
        sorted_entries = sorted(entries, key=lambda x: x.timestamp)
        for i, entry in enumerate(sorted_entries):
            if entry.status != "error":
                continue
            for j in range(i + 1, len(sorted_entries)):
                delta = (sorted_entries[j].timestamp - entry.timestamp).total_seconds()
                if delta > window:
                    break
                if sorted_entries[j].status == "success":
                    total_waste += entry.cost_usd
                    retry_errors += 1
                    break

    if retry_errors == 0:
        return None

    return WastePattern(
        pattern_type="excessive_retries",
        description=(
            f"{retry_errors} failed calls were retried within {window}s. "
            f"The failed attempts cost ${total_waste:.2f}."
        ),
        affected_calls=retry_errors,
        estimated_waste_usd=round(total_waste, 2),
        methodology=(
            "Entries grouped by (feature, model). Error entries "
            f"followed by a success within {window}s are retry chains. "
            "Waste = cost of the failed attempts."
        ),
        confidence="high",
        assumptions=[
            "Error followed by success on same feature+model = retry",
            f"Window: {window}s (may miss slow retries or catch coincidences)",
        ],
    )


def detect_cacheable_duplicates(
    dataset: LogDataset,
    config: AuditConfig,
) -> WastePattern | None:
    """Detect near-duplicate calls within configurable time windows."""
    window = config.duplicate_window_seconds
    tolerance = config.duplicate_token_tolerance
    entries = sorted(dataset.entries, key=lambda e: e.timestamp)
    duplicate_indices: set[int] = set()

    for i, e1 in enumerate(entries):
        if i in duplicate_indices:
            continue
        for j in range(i + 1, len(entries)):
            e2 = entries[j]
            delta = (e2.timestamp - e1.timestamp).total_seconds()
            if delta > window:
                break
            if (
                e1.feature == e2.feature
                and e1.model == e2.model
                and e1.feature is not None
                and _tokens_similar(e1.input_tokens, e2.input_tokens, tolerance)
            ):
                duplicate_indices.add(j)

    if not duplicate_indices:
        return None

    total_waste = sum(entries[j].cost_usd for j in duplicate_indices)
    pct = tolerance * 100

    return WastePattern(
        pattern_type="cacheable_request",
        description=(
            f"{len(duplicate_indices)} calls are near-duplicates of earlier "
            f"calls (same feature, model, similar tokens within "
            f"{window}s). Caching would save ${total_waste:.2f}."
        ),
        affected_calls=len(duplicate_indices),
        estimated_waste_usd=round(total_waste, 2),
        methodology=(
            f"Entries sorted by time. Pairs with same feature, "
            f"model, and input_tokens within {pct:.0f}% occurring within "
            f"{window}s are duplicates. Waste = cost of redundant calls."
        ),
        confidence="low",
        assumptions=[
            "Similar token count is used as a PROXY for similar prompt content",
            "Without prompt text hashing, false positives are possible "
            "(two different questions with similar length would match)",
            f"Token tolerance: {pct:.0f}% — tighter values reduce false positives",
            f"Time window: {window}s — only near-immediate duplicates caught",
        ],
    )


def _tokens_similar(a: int, b: int, tolerance: float) -> bool:
    """Check if two token counts are within tolerance of each other."""
    if a == 0 and b == 0:
        return True
    return abs(a - b) / max(a, b) < tolerance


def detect_waste_patterns(
    dataset: LogDataset,
    config: AuditConfig,
) -> list[WastePattern]:
    """Run all deterministic waste detectors."""
    detectors = [
        detect_bloated_prompts,
        detect_wrong_model,
        detect_excessive_retries,
        detect_cacheable_duplicates,
    ]
    return [r for d in detectors if (r := d(dataset, config)) is not None]


# ── Deduplication ────────────────────────────────────────────────────


def _compute_feature_medians(
    entries: list[LogEntry],
    config: AuditConfig,
) -> dict[str, float]:
    """Compute per-feature median input tokens for routable features."""
    routable = config.routable_features
    min_entries = config.bloated_prompt_min_feature_entries
    feature_tokens: dict[str, list[int]] = defaultdict(list)
    for e in entries:
        if e.feature in routable and e.status == "success":
            feature_tokens[e.feature].append(e.input_tokens)

    medians: dict[str, float] = {}
    for feature, tokens in feature_tokens.items():
        if len(tokens) >= min_entries:
            medians[feature] = statistics.median(tokens)
    return medians


def _find_duplicate_indices(
    entries: list[LogEntry],
    config: AuditConfig,
) -> set[int]:
    """Identify near-duplicate entry indices."""
    window = config.duplicate_window_seconds
    tolerance = config.duplicate_token_tolerance
    duplicate_indices: set[int] = set()
    for i, e1 in enumerate(entries):
        if i in duplicate_indices:
            continue
        for j in range(i + 1, len(entries)):
            e2 = entries[j]
            if (e2.timestamp - e1.timestamp).total_seconds() > window:
                break
            if (
                e1.feature == e2.feature
                and e1.model == e2.model
                and e1.feature is not None
                and _tokens_similar(e1.input_tokens, e2.input_tokens, tolerance)
            ):
                duplicate_indices.add(j)
    return duplicate_indices


def _find_retry_waste_indices(
    entries: list[LogEntry],
    config: AuditConfig,
) -> set[int]:
    """Identify failed retry entry indices."""
    window = config.retry_window_seconds
    groups: dict[tuple[str, str], list[tuple[int, LogEntry]]] = defaultdict(list)
    for i, e in enumerate(entries):
        groups[(e.feature or "unknown", e.model)].append((i, e))

    retry_indices: set[int] = set()
    for group in groups.values():
        sorted_group = sorted(group, key=lambda x: x[1].timestamp)
        for gi, (idx, entry) in enumerate(sorted_group):
            if entry.status != "error":
                continue
            for _, (_, later) in enumerate(sorted_group[gi + 1 :]):
                delta = (later.timestamp - entry.timestamp).total_seconds()
                if delta > window:
                    break
                if later.status == "success":
                    retry_indices.add(idx)
                    break
    return retry_indices


def _compute_deduplicated_waste(
    dataset: LogDataset,
    config: AuditConfig,
) -> float:
    """Compute maximum recoverable waste with no double-counting.

    For each entry, computes the minimum achievable cost if ALL
    applicable optimizations were applied simultaneously, then
    sums the differences. This gives a true savings ceiling.
    """
    entries = sorted(dataset.entries, key=lambda e: e.timestamp)
    if not entries:
        return 0.0

    duplicate_indices = _find_duplicate_indices(entries, config)
    retry_indices = _find_retry_waste_indices(entries, config)
    feature_medians = _compute_feature_medians(entries, config)

    total_waste = 0.0
    for i, entry in enumerate(entries):
        total_waste += _entry_max_waste(
            i,
            entry,
            duplicate_indices,
            retry_indices,
            feature_medians,
            config,
        )
    return round(total_waste, 2)


def _entry_max_waste(
    idx: int,
    entry: LogEntry,
    duplicate_indices: set[int],
    retry_indices: set[int],
    feature_medians: dict[str, float],
    config: AuditConfig,
) -> float:
    """Compute maximum recoverable waste for a single entry."""
    if idx in duplicate_indices or idx in retry_indices:
        return entry.cost_usd

    optimal_model = entry.model
    if not _is_light_model(entry.model) and entry.status == "success":
        complexity = config.feature_complexity.get(entry.feature or "")
        if complexity == TaskComplexity.SIMPLE or (
            complexity == TaskComplexity.MODERATE
            and entry.input_tokens < config.moderate_input_threshold
        ):
            optimal_model = config.light_model

    optimal_input = entry.input_tokens
    if entry.feature in feature_medians and entry.status == "success":
        median = feature_medians[entry.feature]
        threshold = max(
            median * config.bloated_prompt_multiplier,
            config.bloated_prompt_min_tokens,
        )
        if entry.input_tokens > threshold:
            optimal_input = int(median)

    min_cost = estimate_cost(optimal_model, optimal_input, entry.output_tokens)
    return max(entry.cost_usd - min_cost, 0.0)


# ── Main analysis ────────────────────────────────────────────────────


def analyze(
    dataset: LogDataset,
    config: AuditConfig | None = None,
) -> AnalysisResult:
    """Run full analysis on a normalized dataset. Pure computation.

    This is the deterministic fallback — always produces valid results.
    """
    if config is None:
        config = AuditConfig()

    cost_by_model = compute_cost_by_model(dataset)
    cost_by_feature = compute_cost_by_feature(dataset)
    waste_patterns = detect_waste_patterns(dataset, config)

    gross_waste = sum(w.estimated_waste_usd for w in waste_patterns)
    dedup_waste = _compute_deduplicated_waste(dataset, config)

    days = 1
    if dataset.date_range_start and dataset.date_range_end:
        delta = dataset.date_range_end - dataset.date_range_start
        days = max(delta.days, 1)

    monthly_projected = (dataset.total_cost_usd / days) * 30

    return AnalysisResult(
        total_cost_usd=dataset.total_cost_usd,
        total_calls=dataset.total_entries,
        total_tokens=dataset.total_tokens,
        date_range_days=days,
        monthly_projected_cost_usd=monthly_projected,
        cost_by_model=cost_by_model,
        cost_by_feature=cost_by_feature,
        waste_patterns=waste_patterns,
        total_waste_usd=gross_waste,
        deduplicated_waste_usd=dedup_waste,
        waste_pct=(
            min(dedup_waste / dataset.total_cost_usd, 1.0) if dataset.total_cost_usd > 0 else 0
        ),
        audit_config=config,
    )


# ── Agentic analysis ──────────────────────────────────────────────────


async def run_analysis_agent(
    dataset: LogDataset,
    client: UnifiedLLMClient,
    config: AuditConfig | None = None,
) -> AnalysisResult:
    """Run agentic analysis with tool-use loop.

    Falls back to deterministic analyze() on failure.
    """
    import anthropic as _anthropic

    from src.agents.base import AgentLoopExhaustedError

    if config is None:
        config = AuditConfig()

    try:
        return await _run_agentic_analysis(dataset, client, config)
    except (AgentLoopExhaustedError, _anthropic.APIError, ValueError) as e:
        logger.warning("Analysis agent failed, using deterministic fallback: %s", e)
        return analyze(dataset, config)


async def _run_agentic_analysis(
    dataset: LogDataset,
    client: UnifiedLLMClient,
    config: AuditConfig,
) -> AnalysisResult:
    """Internal: run the agentic analysis loop."""
    from src.agents.base import run_agent_loop
    from src.tools.analysis_tools import build_analysis_registry
    from src.utils.dataset_summary import summarize_dataset

    registry = build_analysis_registry(dataset, config)
    summary = summarize_dataset(dataset)

    initial_message = (
        "Analyze this LLM usage dataset and identify all waste patterns.\n\n"
        f"Dataset overview:\n{json.dumps(summary, indent=2)}"
    )

    agent_result = await run_agent_loop(
        client=client,
        system_prompt=ANALYSIS_SYSTEM_PROMPT,
        initial_message=initial_message,
        registry=registry,
        max_iterations=15,
        agent_name="analysis_agent",
    )

    return _build_analysis_from_agent(dataset, config, agent_result.final_content)


def _build_analysis_from_agent(
    dataset: LogDataset,
    config: AuditConfig,
    agent_output: str,
) -> AnalysisResult:
    """Build AnalysisResult from deterministic data + agent insights."""
    from src.utils.json_extract import parse_json_response

    result = analyze(dataset, config)
    findings, insights = _parse_agent_insights(agent_output, parse_json_response)

    if findings:
        result.agent_findings_summary = findings
    if insights:
        result.agent_key_insights = insights

    return result


def _parse_agent_insights(
    agent_output: str,
    parser: object,
) -> tuple[str, list[str]]:
    """Extract findings_summary and key_insights from agent output."""
    try:
        insights = parser(agent_output)  # type: ignore[operator]
        if isinstance(insights, dict):
            summary = str(insights.get("findings_summary", ""))
            keys = insights.get("key_insights", [])
            key_list = [str(k) for k in keys] if isinstance(keys, list) else []
            logger.info("Agent analysis insights: %s", summary[:200])
            return summary, key_list
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.debug("Could not parse agent insights (non-fatal)")
    return "", []
