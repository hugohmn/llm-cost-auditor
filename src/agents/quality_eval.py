"""Quality Evaluation — 3-layer quality assessment for routing decisions.

Layer 1: Proxy signals from existing cross-model data (always runs, zero cost)
Layer 2: LLM-as-Judge blind A/B evaluation (opt-in, costs money)
Layer 3: Eval plan generation (always runs, zero cost)

Not an agent loop — deterministic computation + optional direct LLM calls.
"""

import logging
import math
import random
import statistics

from src.models.analysis import AnalysisResult
from src.models.features import AuditConfig
from src.models.log_entry import LogDataset, LogEntry
from src.models.quality import (
    ConfidenceLevel,
    CriterionScores,
    EvalCriterion,
    EvalPlanResult,
    FeatureEvalPlan,
    FeatureJudgeResult,
    FeatureProxySignal,
    JudgeEvalNotRun,
    JudgeEvalResult,
    JudgePairResult,
    OutputTokenStats,
    ProxySignalResult,
    QualityAssessment,
    RoutingVerdict,
)
from src.models.recommendation import OptimizationPlan
from src.utils.json_extract import parse_json_response
from src.utils.llm_client import UnifiedLLMClient, estimate_cost
from src.utils.model_utils import is_light_model
from src.utils.retry import with_retry

logger = logging.getLogger(__name__)

# ── Layer 1: Proxy Signals ────────────────────────────────────

_PROXY_METHODOLOGY = (
    "For each routable feature with calls to both frontier and light models, "
    "we compare output token distributions as a behavioral proxy. "
    "Output ratio = light model mean output tokens / frontier model mean output tokens. "
    "Confidence levels: INSUFFICIENT (n < {min_samples}), "
    "MODERATE ({min_samples}-49), STRONG (50+) per model tier."
)

_PROXY_LIMITATIONS = [
    "Output token count is a length proxy, not a quality measure.",
    "Similar output length does not guarantee similar content quality.",
    "Error rates measure API failures, not output correctness.",
    "Features without light model data cannot be compared.",
]


def compute_proxy_signals(
    dataset: LogDataset,
    config: AuditConfig,
) -> ProxySignalResult:
    """Layer 1: compute cross-model proxy quality signals."""
    routable = config.routable_features
    feature_groups = _group_by_feature_and_tier(dataset, routable)

    signals: list[FeatureProxySignal] = []
    without_comparison: list[str] = []

    for feature in sorted(routable):
        frontier = feature_groups.get((feature, "frontier"), [])
        light = feature_groups.get((feature, "light"), [])

        if not light:
            without_comparison.append(feature)
            continue

        if not frontier:
            continue

        signals.append(_compute_feature_signal(feature, frontier, light, config))

    return ProxySignalResult(
        feature_signals=signals,
        features_without_comparison=without_comparison,
        methodology=_PROXY_METHODOLOGY.format(
            min_samples=config.proxy_min_samples,
        ),
        limitations=list(_PROXY_LIMITATIONS),
    )


def _group_by_feature_and_tier(
    dataset: LogDataset,
    routable: frozenset[str],
) -> dict[tuple[str, str], list[LogEntry]]:
    """Group entries by (feature, 'frontier'|'light') for routable features."""
    groups: dict[tuple[str, str], list[LogEntry]] = {}
    for entry in dataset.entries:
        if entry.feature not in routable:
            continue
        tier = "light" if is_light_model(entry.model) else "frontier"
        key = (entry.feature, tier)
        groups.setdefault(key, []).append(entry)
    return groups


def _compute_token_stats(entries: list[LogEntry]) -> OutputTokenStats:
    """Compute output token distribution statistics."""
    tokens = [e.output_tokens for e in entries]
    if not tokens:
        return OutputTokenStats(sample_size=0, mean=0.0, median=0.0, p25=0.0, p75=0.0)
    sorted_tokens = sorted(tokens)
    n = len(sorted_tokens)
    return OutputTokenStats(
        sample_size=n,
        mean=statistics.mean(tokens),
        median=statistics.median(tokens),
        p25=sorted_tokens[n // 4] if n >= 4 else sorted_tokens[0],
        p75=sorted_tokens[(3 * n) // 4] if n >= 4 else sorted_tokens[-1],
    )


def _compute_error_rate(entries: list[LogEntry]) -> float:
    """Compute fraction of entries with non-success status."""
    if not entries:
        return 0.0
    errors = sum(1 for e in entries if e.status != "success")
    return errors / len(entries)


def _determine_confidence(
    frontier_n: int,
    light_n: int,
    min_samples: int,
) -> ConfidenceLevel:
    """Determine confidence from the smaller sample size."""
    min_n = min(frontier_n, light_n)
    if min_n < min_samples:
        return ConfidenceLevel.INSUFFICIENT
    if min_n < 50:
        return ConfidenceLevel.MODERATE
    return ConfidenceLevel.STRONG


def _interpret_signal(
    output_ratio: float,
    error_delta: float,
    confidence: ConfidenceLevel,
) -> str:
    """Generate human-readable interpretation of proxy signals."""
    if confidence == ConfidenceLevel.INSUFFICIENT:
        return "Insufficient data for reliable comparison — evaluate before routing."

    parts: list[str] = []
    if 0.85 <= output_ratio <= 1.15:
        parts.append("Output length comparable between models, suggesting similar behavior.")
    elif output_ratio < 0.85:
        parts.append(
            "Light model produces notably shorter outputs — "
            "potential quality gap, evaluate before routing."
        )
    else:
        parts.append("Light model produces longer outputs — different behavior observed.")

    if error_delta > 0.05:
        parts.append("Light model has significantly higher API error rate.")
    elif error_delta < -0.02:
        parts.append("Light model has lower API error rate.")

    return " ".join(parts)


def _compute_feature_signal(
    feature: str,
    frontier_entries: list[LogEntry],
    light_entries: list[LogEntry],
    config: AuditConfig,
) -> FeatureProxySignal:
    """Build proxy signal comparison for a single feature."""
    frontier_stats = _compute_token_stats(frontier_entries)
    light_stats = _compute_token_stats(light_entries)

    # Default to 1.0 if frontier has no output tokens (data anomaly)
    ratio = 1.0 if frontier_stats.mean == 0 else light_stats.mean / frontier_stats.mean
    frontier_err = _compute_error_rate(frontier_entries)
    light_err = _compute_error_rate(light_entries)
    confidence = _determine_confidence(
        frontier_stats.sample_size,
        light_stats.sample_size,
        config.proxy_min_samples,
    )

    return FeatureProxySignal(
        feature=feature,
        frontier_output=frontier_stats,
        light_output=light_stats,
        output_ratio=round(ratio, 3),
        frontier_error_rate=round(frontier_err, 4),
        light_error_rate=round(light_err, 4),
        confidence=confidence,
        interpretation=_interpret_signal(ratio, light_err - frontier_err, confidence),
    )


# ── Layer 3: Eval Plans ───────────────────────────────────────

FEATURE_EVAL_CRITERIA: dict[str, list[tuple[str, str, int]]] = {
    "email_drafting": [
        ("tone", "Professional, appropriate tone for context", 4),
        ("completeness", "All required elements (greeting, body, sign-off)", 4),
        ("accuracy", "Factual accuracy of referenced information", 4),
    ],
    "data_extraction": [
        ("accuracy", "Correctly extracts all requested fields", 5),
        ("completeness", "No fields missing from extraction", 4),
        ("format_compliance", "Output matches expected schema/format", 5),
    ],
    "customer_support_agent": [
        ("helpfulness", "Addresses the customer's actual question", 4),
        ("accuracy", "Information provided is correct", 4),
        ("tone", "Empathetic, professional tone", 4),
    ],
    "doc_qa": [
        ("accuracy", "Answer correctly reflects source document", 5),
        ("relevance", "Answer addresses the specific question asked", 4),
        ("completeness", "All relevant information from docs included", 4),
    ],
}

_DEFAULT_EVAL_CRITERIA: list[tuple[str, str, int]] = [
    ("task_completion", "The output fulfills the stated task", 4),
    ("accuracy", "Output is factually correct and consistent", 4),
    ("completeness", "No important information is missing", 3),
]

_EVAL_PLAN_METHODOLOGY = (
    "Sample sizes based on proxy signal confidence: "
    "50 (insufficient data), 30 (moderate), 20 (strong). "
    "Larger samples when less is known about the quality gap. "
    "Cost estimates use published per-token pricing via estimate_cost(). "
    "Evaluation criteria are feature-specific where possible, "
    "with generic fallbacks for unknown features."
)


def generate_eval_plans(
    optimization: OptimizationPlan,
    config: AuditConfig,
    proxy_signals: ProxySignalResult,
) -> EvalPlanResult:
    """Layer 3: generate concrete evaluation plans for routable features."""
    proxy_map = {s.feature: s for s in proxy_signals.feature_signals}
    plans: list[FeatureEvalPlan] = []

    if not optimization.routing_simulation:
        return EvalPlanResult(
            feature_plans=[],
            total_estimated_cost_usd=0.0,
            methodology=_EVAL_PLAN_METHODOLOGY,
        )

    light_model = optimization.routing_simulation.light_model
    frontier_model = optimization.routing_simulation.frontier_model

    for feature in sorted(config.routable_features):
        proxy = proxy_map.get(feature)
        plans.append(_build_feature_plan(feature, frontier_model, light_model, config, proxy))

    total_cost = sum(p.estimated_eval_cost_usd for p in plans)
    return EvalPlanResult(
        feature_plans=plans,
        total_estimated_cost_usd=round(total_cost, 2),
        methodology=_EVAL_PLAN_METHODOLOGY,
    )


def _recommend_sample_size(
    proxy: FeatureProxySignal | None,
) -> tuple[int, str]:
    """Recommend sample size based on proxy confidence."""
    if proxy is None or proxy.confidence == ConfidenceLevel.INSUFFICIENT:
        return 50, "No reliable proxy data — larger sample needed"
    if proxy.confidence == ConfidenceLevel.MODERATE:
        return 30, "Moderate proxy data — standard sample size"
    return 20, "Strong proxy data — smaller sample sufficient"


def _get_criteria(feature: str) -> list[EvalCriterion]:
    """Get feature-specific or default evaluation criteria."""
    raw = FEATURE_EVAL_CRITERIA.get(feature, _DEFAULT_EVAL_CRITERIA)
    return [
        EvalCriterion(name=name, description=desc, minimum_acceptable=min_score)
        for name, desc, min_score in raw
    ]


def _estimate_feature_eval_cost(
    sample_size: int,
    light_model: str,
    judge_model: str,
) -> float:
    """Estimate total eval cost for one feature."""
    avg_input = 5000
    avg_output = 1000
    judge_prompt_overhead = 400

    replay_cost = sample_size * estimate_cost(light_model, avg_input, avg_output)
    judge_input = judge_prompt_overhead + avg_input + avg_output * 2
    judge_cost = sample_size * estimate_cost(judge_model, judge_input, 200)
    return round(replay_cost + judge_cost, 2)


def _build_feature_plan(
    feature: str,
    frontier_model: str,
    light_model: str,
    config: AuditConfig,
    proxy: FeatureProxySignal | None,
) -> FeatureEvalPlan:
    """Build evaluation plan for a single feature."""
    sample_size, rationale = _recommend_sample_size(proxy)
    criteria = _get_criteria(feature)
    cost = _estimate_feature_eval_cost(sample_size, light_model, config.judge_model)

    min_scores = [c.minimum_acceptable for c in criteria]
    min_composite = round(sum(min_scores) / len(min_scores), 1) if min_scores else 3.5

    return FeatureEvalPlan(
        feature=feature,
        current_model=frontier_model,
        proposed_model=light_model,
        recommended_sample_size=sample_size,
        sample_size_rationale=rationale,
        criteria=criteria,
        minimum_composite_score=min_composite,
        estimated_eval_cost_usd=cost,
        steps=_build_eval_steps(
            feature, light_model, config.judge_model, sample_size, min_composite
        ),
    )


def _build_eval_steps(
    feature: str,
    light_model: str,
    judge_model: str,
    sample_size: int,
    min_composite: float,
) -> list[str]:
    """Build step-by-step evaluation instructions."""
    return [
        f"Export {sample_size} representative {feature} prompts "
        f"(stratified by input length: short/medium/long)",
        f"Run each prompt through {light_model}",
        f"Blind-evaluate both outputs using {judge_model} as judge "
        f"(randomize A/B order to control position bias)",
        "Score each response on the criteria above (1-5 scale)",
        "Compute mean weighted composite score per model",
        f"If light model composite >= {min_composite}, approve routing; "
        f"if delta > -5%, routing is safe",
    ]


# ── Layer 2: LLM-as-Judge ─────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are an impartial quality evaluator comparing two AI model responses \
to the same prompt. You will see the original user prompt and two responses \
labeled "Response A" and "Response B". You do not know which model produced \
which response. Do not let the order of responses influence your evaluation.

Evaluate BOTH responses independently on these criteria. \
All scores must be whole integers (1, 2, 3, 4, or 5) — no decimals:

1. ACCURACY: Factual correctness and faithfulness to the prompt's intent.
   1 = Contains critical errors or fabrications
   3 = Mostly correct with minor inaccuracies
   5 = Fully correct and faithful to prompt intent

2. COMPLETENESS: Whether the response addresses all parts of the request.
   1 = Misses the main request entirely
   3 = Addresses the core request but misses secondary aspects
   5 = Thoroughly covers every aspect of the request

3. COHERENCE: Logical structure, clarity, and readability.
   1 = Incoherent or contradictory
   3 = Understandable but could be clearer
   5 = Exceptionally clear and well-organized

4. INSTRUCTION_FOLLOWING: Adherence to explicit format, length, tone, \
or constraint instructions in the prompt.
   1 = Ignores instructions entirely
   3 = Follows most instructions with minor deviations
   5 = Follows all instructions precisely

Respond ONLY with a JSON object in this exact format:
{
  "response_a": {
    "accuracy": <int 1-5>,
    "completeness": <int 1-5>,
    "coherence": <int 1-5>,
    "instruction_following": <int 1-5>
  },
  "response_b": {
    "accuracy": <int 1-5>,
    "completeness": <int 1-5>,
    "coherence": <int 1-5>,
    "instruction_following": <int 1-5>
  },
  "preferred": "a" | "b" | "tie",
  "confidence": "high" | "medium" | "low",
  "rationale": "<1-2 sentences explaining the key differentiator>"
}"""


_JUDGE_METHODOLOGY = (
    "Blind A/B comparison with randomized response order. "
    "Judge scores on accuracy, completeness, coherence, "
    "instruction_following (1-5). Weighted composite: "
    "accuracy 35%, completeness 25%, coherence 15%, "
    "instruction_following 25%. "
    "CI computed via t-distribution at 95% confidence."
)


def _filter_judgeable_entries(
    dataset: LogDataset,
    config: AuditConfig,
) -> dict[str, list[LogEntry]]:
    """Filter to frontier success entries with content, grouped by feature."""
    by_feature: dict[str, list[LogEntry]] = {}
    for e in dataset.entries:
        if (
            e.input_text
            and e.output_text
            and not is_light_model(e.model)
            and e.status == "success"
            and e.feature in config.routable_features
        ):
            by_feature.setdefault(e.feature or "", []).append(e)
    return by_feature


async def run_judge_eval(
    dataset: LogDataset,
    config: AuditConfig,
    client: UnifiedLLMClient,
) -> JudgeEvalResult | JudgeEvalNotRun:
    """Layer 2: run LLM-as-Judge evaluation if enabled and content available."""
    if not config.enable_judge_eval:
        return JudgeEvalNotRun(reason="disabled_by_config")

    by_feature = _filter_judgeable_entries(dataset, config)
    if not by_feature:
        return JudgeEvalNotRun(reason="no_prompt_content_available")

    feature_results: list[FeatureJudgeResult] = []
    skipped: list[str] = []
    total_cost = 0.0
    total_samples = 0

    for feature in sorted(by_feature):
        entries = by_feature[feature]
        if len(entries) < 5:
            skipped.append(feature)
            continue
        if total_cost >= config.judge_max_budget_usd:
            logger.warning("Budget exhausted at $%.2f", total_cost)
            skipped.append(feature)
            continue

        remaining = config.judge_max_budget_usd - total_cost
        result, cost = await _evaluate_feature(
            feature, entries, config, client, remaining_budget=remaining
        )
        if result:
            feature_results.append(result)
            total_cost += cost
            total_samples += result.sample_size

    return JudgeEvalResult(
        feature_results=feature_results,
        skipped_features=skipped,
        total_samples=total_samples,
        total_eval_cost_usd=round(total_cost, 4),
        judge_model=config.judge_model,
        methodology=_JUDGE_METHODOLOGY,
    )


async def _evaluate_feature(
    feature: str,
    entries: list[LogEntry],
    config: AuditConfig,
    client: UnifiedLLMClient,
    remaining_budget: float = 5.0,
) -> tuple[FeatureJudgeResult | None, float]:
    """Evaluate a single feature: sample, replay, judge, aggregate."""
    samples = _select_samples(entries, config.judge_eval_sample_size, seed=hash(feature))

    pairs: list[JudgePairResult] = []
    cost = 0.0
    replay_failures = 0

    for sample in samples:
        if cost >= remaining_budget:
            break

        pair_result, pair_cost = await _replay_and_judge_one(sample, config, client)
        cost += pair_cost
        if pair_result:
            pairs.append(pair_result)
        else:
            replay_failures += 1

    if not pairs:
        logger.warning("No successful evaluations for feature %s", feature)
        return None, cost

    result = _aggregate_feature_results(feature, pairs, replay_failures)
    return result, cost


def _select_samples(
    entries: list[LogEntry],
    target_size: int,
    seed: int,
) -> list[LogEntry]:
    """Stratified sampling by input length buckets."""
    if not entries:
        return []

    buckets: dict[str, list[LogEntry]] = {
        "short": [],
        "medium": [],
        "long": [],
    }
    for e in entries:
        if e.input_tokens < 2000:
            buckets["short"].append(e)
        elif e.input_tokens < 8000:
            buckets["medium"].append(e)
        else:
            buckets["long"].append(e)

    rng = random.Random(seed)
    target = min(target_size, len(entries))
    selected: list[LogEntry] = []

    for bucket_entries in buckets.values():
        if not bucket_entries:
            continue
        share = max(1, round(target * len(bucket_entries) / len(entries)))
        share = min(share, len(bucket_entries))
        selected.extend(rng.sample(bucket_entries, share))

    # Trim to exact target if oversampled
    if len(selected) > target:
        selected = rng.sample(selected, target)

    return selected


async def _replay_and_judge_one(
    sample: LogEntry,
    config: AuditConfig,
    client: UnifiedLLMClient,
) -> tuple[JudgePairResult | None, float]:
    """Replay one prompt through light model and judge both outputs."""
    cost = 0.0

    # Replay through light model
    try:
        replay_response = await with_retry(
            client.complete,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": sample.input_text or ""}],
            model=config.light_model,
            temperature=0.0,
            agent_name="quality_eval_replay",
        )
        cost += replay_response.cost_usd
    except Exception as exc:
        logger.warning("Replay failed for entry on %s: %s", sample.feature, exc)
        return None, cost

    # Judge the pair
    frontier_output = sample.output_text or ""
    light_output = replay_response.content

    try:
        pair, judge_cost = await _call_judge(
            sample.input_text or "",
            frontier_output,
            light_output,
            config,
            client,
        )
        cost += judge_cost
        return pair, cost
    except Exception as exc:
        logger.warning("Judge call failed for entry on %s: %s", sample.feature, exc)
        return None, cost


async def _call_judge(
    input_text: str,
    frontier_output: str,
    light_output: str,
    config: AuditConfig,
    client: UnifiedLLMClient,
) -> tuple[JudgePairResult, float]:
    """Call the judge model with blind A/B comparison."""
    # Randomize order to prevent position bias
    frontier_is_a = random.choice([True, False])
    if frontier_is_a:
        response_a, response_b = frontier_output, light_output
    else:
        response_a, response_b = light_output, frontier_output

    user_message = (
        f"## Original Prompt\n\n{input_text}\n\n"
        f"## Response A\n\n{response_a}\n\n"
        f"## Response B\n\n{response_b}"
    )

    response = await with_retry(
        client.complete,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        model=config.judge_model,
        temperature=0.0,
        agent_name="quality_eval_judge",
    )

    parsed = parse_json_response(response.content)
    return _parse_judge_response(parsed, frontier_is_a), response.cost_usd


def _parse_judge_response(
    parsed: dict[str, object] | list[dict[str, object]],
    frontier_was_a: bool,
) -> JudgePairResult:
    """Parse judge JSON response into typed result."""
    if isinstance(parsed, list):
        parsed = parsed[0]

    if "response_a" not in parsed or "response_b" not in parsed:
        raise ValueError(f"Judge response missing required keys. Got: {list(parsed.keys())}")

    scores_a = CriterionScores(**parsed["response_a"])  # type: ignore[arg-type]
    scores_b = CriterionScores(**parsed["response_b"])  # type: ignore[arg-type]

    if frontier_was_a:
        frontier_scores, light_scores = scores_a, scores_b
    else:
        frontier_scores, light_scores = scores_b, scores_a

    # Map preference from a/b to frontier/light
    raw_pref = str(parsed.get("preferred", "tie")).lower()
    if raw_pref == "a":
        preferred = "frontier" if frontier_was_a else "light"
    elif raw_pref == "b":
        preferred = "light" if frontier_was_a else "frontier"
    else:
        preferred = "tie"

    delta = light_scores.weighted_composite - frontier_scores.weighted_composite

    return JudgePairResult(
        frontier_scores=frontier_scores,
        light_scores=light_scores,
        preferred=preferred,
        quality_delta=round(delta, 4),
        judge_rationale=str(parsed.get("rationale", "")),
        frontier_was_a=frontier_was_a,
    )


def _compute_ci_margin(deltas: list[float]) -> float:
    """Compute 95% CI margin of error using t-distribution."""
    n = len(deltas)
    if n <= 1:
        return 0.0
    std = statistics.stdev(deltas)
    t_crit = 1.96 if n >= 30 else _t_critical_approx(n - 1)
    return t_crit * (std / math.sqrt(n))


def _compute_per_criterion_deltas(
    pairs: list[JudgePairResult],
) -> dict[str, float]:
    """Compute mean delta per scoring criterion (light minus frontier)."""
    criteria = ["accuracy", "completeness", "coherence", "instruction_following"]
    result: dict[str, float] = {}
    for crit in criteria:
        crit_deltas = [
            getattr(p.light_scores, crit) - getattr(p.frontier_scores, crit) for p in pairs
        ]
        result[crit] = round(statistics.mean(crit_deltas), 3)
    return result


def _aggregate_feature_results(
    feature: str,
    pairs: list[JudgePairResult],
    replay_failures: int,
) -> FeatureJudgeResult:
    """Aggregate judge pair results into feature-level statistics."""
    deltas = [p.quality_delta for p in pairs]
    n = len(deltas)
    mean_delta = statistics.mean(deltas)
    margin = _compute_ci_margin(deltas)
    per_criterion = _compute_per_criterion_deltas(pairs)

    wins = sum(1 for p in pairs if p.preferred == "light")
    ties = sum(1 for p in pairs if p.preferred == "tie")
    losses = sum(1 for p in pairs if p.preferred == "frontier")

    high_failure_rate = replay_failures > 0.3 * (n + replay_failures)
    win_tie_rate = (wins + ties) / n if n > 0 else 0.0
    verdict = _compute_verdict(
        mean_delta,
        mean_delta + margin,
        win_tie_rate,
        per_criterion,
        high_failure_rate,
    )

    return FeatureJudgeResult(
        feature=feature,
        sample_size=n,
        mean_quality_delta=round(mean_delta, 4),
        std_quality_delta=round(statistics.stdev(deltas) if n > 1 else 0.0, 4),
        ci_lower_95=round(mean_delta - margin, 4),
        ci_upper_95=round(mean_delta + margin, 4),
        win_rate=round(wins / n, 3) if n > 0 else 0.0,
        tie_rate=round(ties / n, 3) if n > 0 else 0.0,
        loss_rate=round(losses / n, 3) if n > 0 else 0.0,
        verdict=verdict,
        per_criterion_deltas=per_criterion,
    )


def _t_critical_approx(df: int) -> float:
    """Approximate t-critical value for 95% CI (two-tailed)."""
    # Good enough for df >= 4; for smaller df, conservative
    if df <= 1:
        return 12.706
    if df <= 2:
        return 4.303
    if df <= 5:
        return 2.571
    if df <= 10:
        return 2.228
    if df <= 20:
        return 2.086
    if df <= 30:
        return 2.042
    return 1.96


def _compute_verdict(
    mean_delta: float,
    ci_upper: float,
    win_tie_rate: float,
    per_criterion: dict[str, float],
    high_failure_rate: bool,
) -> RoutingVerdict:
    """Map quality delta + conditions to routing verdict."""
    if high_failure_rate:
        return RoutingVerdict.ROUTE_WITH_CAUTION

    worst_criterion = min(per_criterion.values()) if per_criterion else 0.0

    # Safe: delta > -0.15, CI not too negative, high win+tie, no criterion tanks
    if mean_delta > -0.15 and ci_upper > -0.30 and win_tie_rate >= 0.70 and worst_criterion > -0.50:
        return RoutingVerdict.SAFE_TO_ROUTE

    if mean_delta > -0.50:
        return RoutingVerdict.ROUTE_WITH_MONITORING

    if mean_delta > -1.00:
        return RoutingVerdict.ROUTE_WITH_CAUTION

    return RoutingVerdict.DO_NOT_ROUTE


# ── Orchestrator ──────────────────────────────────────────────


async def run_quality_evaluation(
    dataset: LogDataset,
    analysis: AnalysisResult,
    optimization: OptimizationPlan,
    client: UnifiedLLMClient,
    config: AuditConfig | None = None,
) -> QualityAssessment:
    """Run all 3 quality evaluation layers."""
    if config is None:
        config = analysis.audit_config

    logger.info("Layer 1: Computing proxy quality signals...")
    proxy_signals = compute_proxy_signals(dataset, config)
    logger.info(
        "Proxy signals: %d features compared, %d without light model data",
        len(proxy_signals.feature_signals),
        len(proxy_signals.features_without_comparison),
    )

    logger.info("Layer 2: LLM-as-Judge evaluation...")
    judge_eval = await run_judge_eval(dataset, config, client)
    if isinstance(judge_eval, JudgeEvalNotRun):
        logger.info("Judge eval skipped: %s", judge_eval.reason)
    else:
        logger.info(
            "Judge eval: %d samples, $%.4f cost",
            judge_eval.total_samples,
            judge_eval.total_eval_cost_usd,
        )

    logger.info("Layer 3: Generating evaluation plans...")
    eval_plans = generate_eval_plans(optimization, config, proxy_signals)
    logger.info(
        "Eval plans: %d features, $%.2f estimated cost",
        len(eval_plans.feature_plans),
        eval_plans.total_estimated_cost_usd,
    )

    return QualityAssessment(
        proxy_signals=proxy_signals,
        judge_eval=judge_eval,
        eval_plans=eval_plans,
    )
