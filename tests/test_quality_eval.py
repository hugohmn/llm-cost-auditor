"""Tests for quality evaluation — proxy signals, eval plans, and judge logic.

Layer 1 and Layer 3 tests are deterministic (no mocks needed).
Layer 2 tests use mock data for verdict/aggregation logic.
"""

from datetime import UTC, datetime, timedelta

import pytest

from src.agents.quality_eval import (
    _aggregate_feature_results,
    _build_feature_plan,
    _compute_error_rate,
    _compute_token_stats,
    _compute_verdict,
    _determine_confidence,
    _get_criteria,
    _interpret_signal,
    _recommend_sample_size,
    _select_samples,
    compute_proxy_signals,
    generate_eval_plans,
)
from src.models.features import AuditConfig
from src.models.log_entry import LogDataset, LogEntry
from src.models.quality import (
    ConfidenceLevel,
    CriterionScores,
    FeatureProxySignal,
    JudgeEvalNotRun,
    JudgePairResult,
    OutputTokenStats,
    RoutingVerdict,
)
from src.models.recommendation import OptimizationPlan, RoutingSimResult
from src.utils.llm_client import estimate_cost

CONFIG = AuditConfig()
BASE_TIME = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)


def _entry(
    *,
    model: str = "claude-sonnet-4-6",
    feature: str = "email_drafting",
    input_tokens: int = 1000,
    output_tokens: int = 500,
    status: str = "success",
    offset_seconds: int = 0,
    input_text: str | None = None,
    output_text: str | None = None,
) -> LogEntry:
    """Helper to build a LogEntry with computed cost."""
    cost = estimate_cost(model, input_tokens, output_tokens)
    return LogEntry(
        timestamp=BASE_TIME + timedelta(seconds=offset_seconds),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=cost,
        feature=feature,
        status=status,
        input_text=input_text,
        output_text=output_text,
    )


def _dataset(entries: list[LogEntry]) -> LogDataset:
    """Helper to build a LogDataset."""
    return LogDataset(
        entries=entries,
        source_format="test",
        date_range_start=entries[0].timestamp if entries else None,
        date_range_end=entries[-1].timestamp if entries else None,
    )


# ── Layer 1: Proxy Signal Tests ───────────────────────────────


class TestComputeTokenStats:
    def test_basic_stats(self) -> None:
        entries = [
            _entry(output_tokens=100),
            _entry(output_tokens=200),
            _entry(output_tokens=300),
            _entry(output_tokens=400),
        ]
        stats = _compute_token_stats(entries)
        assert stats.sample_size == 4
        assert stats.mean == 250.0
        assert stats.median == 250.0  # median of [100,200,300,400]

    def test_empty_entries(self) -> None:
        stats = _compute_token_stats([])
        assert stats.sample_size == 0
        assert stats.mean == 0.0


class TestComputeErrorRate:
    def test_all_success(self) -> None:
        entries = [_entry(status="success") for _ in range(5)]
        assert _compute_error_rate(entries) == 0.0

    def test_some_errors(self) -> None:
        entries = [
            _entry(status="success"),
            _entry(status="error"),
            _entry(status="success"),
            _entry(status="error"),
        ]
        assert _compute_error_rate(entries) == 0.5

    def test_empty(self) -> None:
        assert _compute_error_rate([]) == 0.0


class TestDetermineConfidence:
    def test_insufficient(self) -> None:
        assert _determine_confidence(5, 3, 10) == ConfidenceLevel.INSUFFICIENT

    def test_moderate(self) -> None:
        assert _determine_confidence(30, 20, 10) == ConfidenceLevel.MODERATE

    def test_strong(self) -> None:
        assert _determine_confidence(100, 60, 10) == ConfidenceLevel.STRONG

    def test_uses_smaller_sample(self) -> None:
        # frontier has 100 but light only has 5 → insufficient
        assert _determine_confidence(100, 5, 10) == ConfidenceLevel.INSUFFICIENT


class TestInterpretSignal:
    def test_comparable_output(self) -> None:
        result = _interpret_signal(0.95, 0.0, ConfidenceLevel.STRONG)
        assert "comparable" in result.lower()

    def test_shorter_output(self) -> None:
        result = _interpret_signal(0.60, 0.0, ConfidenceLevel.MODERATE)
        assert "shorter" in result.lower()

    def test_longer_output(self) -> None:
        result = _interpret_signal(1.30, 0.0, ConfidenceLevel.MODERATE)
        assert "longer" in result.lower()

    def test_high_error_delta(self) -> None:
        result = _interpret_signal(1.0, 0.10, ConfidenceLevel.STRONG)
        assert "higher" in result.lower() and "error" in result.lower()

    def test_insufficient_data(self) -> None:
        result = _interpret_signal(0.5, 0.0, ConfidenceLevel.INSUFFICIENT)
        assert "insufficient" in result.lower()


class TestComputeProxySignals:
    def test_cross_model_comparison(self) -> None:
        """Dataset with both frontier and light calls produces comparison."""
        entries = [
            _entry(
                model="claude-sonnet-4-6",
                feature="email_drafting",
                output_tokens=500,
                offset_seconds=i,
            )
            for i in range(15)
        ] + [
            _entry(
                model="claude-haiku-4-5-20251001",
                feature="email_drafting",
                output_tokens=450,
                offset_seconds=i + 100,
            )
            for i in range(15)
        ]
        ds = _dataset(entries)
        result = compute_proxy_signals(ds, CONFIG)

        assert len(result.feature_signals) == 1
        signal = result.feature_signals[0]
        assert signal.feature == "email_drafting"
        assert signal.output_ratio == pytest.approx(0.9, abs=0.01)
        assert signal.confidence == ConfidenceLevel.MODERATE

    def test_features_without_light_data(self) -> None:
        """Frontier-only features listed in features_without_comparison."""
        entries = [
            _entry(
                model="claude-sonnet-4-6",
                feature="email_drafting",
                offset_seconds=i,
            )
            for i in range(10)
        ]
        ds = _dataset(entries)
        result = compute_proxy_signals(ds, CONFIG)

        assert "email_drafting" in result.features_without_comparison
        assert len(result.feature_signals) == 0

    def test_empty_dataset(self) -> None:
        """Empty dataset returns empty result."""
        ds = LogDataset(
            entries=[],
            source_format="test",
            date_range_start=None,
            date_range_end=None,
        )
        result = compute_proxy_signals(ds, CONFIG)
        assert len(result.feature_signals) == 0

    def test_non_routable_features_excluded(self) -> None:
        """Complex features don't appear in signals; routable ones without data are listed."""
        entries = [
            _entry(
                model="claude-sonnet-4-6",
                feature="code_review",
                offset_seconds=i,
            )
            for i in range(20)
        ]
        ds = _dataset(entries)
        result = compute_proxy_signals(ds, CONFIG)

        assert len(result.feature_signals) == 0
        # code_review is COMPLEX (not routable) → not in features_without_comparison
        assert "code_review" not in result.features_without_comparison
        # But routable features with no data ARE listed
        assert "email_drafting" in result.features_without_comparison

    def test_multiple_routable_features(self) -> None:
        """Multiple features each get their own signal."""
        entries = []
        for i in range(15):
            entries.append(
                _entry(
                    model="claude-sonnet-4-6",
                    feature="email_drafting",
                    output_tokens=500,
                    offset_seconds=i,
                )
            )
            entries.append(
                _entry(
                    model="claude-haiku-4-5-20251001",
                    feature="email_drafting",
                    output_tokens=480,
                    offset_seconds=i + 100,
                )
            )
            entries.append(
                _entry(
                    model="claude-sonnet-4-6",
                    feature="data_extraction",
                    output_tokens=200,
                    offset_seconds=i + 200,
                )
            )
            entries.append(
                _entry(
                    model="gpt-4o-mini",
                    feature="data_extraction",
                    output_tokens=180,
                    offset_seconds=i + 300,
                )
            )

        ds = _dataset(entries)
        result = compute_proxy_signals(ds, CONFIG)

        features = {s.feature for s in result.feature_signals}
        assert "email_drafting" in features
        assert "data_extraction" in features


# ── Layer 3: Eval Plan Tests ──────────────────────────────────


class TestRecommendSampleSize:
    def test_no_proxy(self) -> None:
        size, _ = _recommend_sample_size(None)
        assert size == 50

    def test_insufficient_proxy(self) -> None:
        proxy = FeatureProxySignal(
            feature="test",
            frontier_output=OutputTokenStats(
                sample_size=3,
                mean=100,
                median=100,
                p25=80,
                p75=120,
            ),
            light_output=OutputTokenStats(
                sample_size=3,
                mean=90,
                median=90,
                p25=70,
                p75=110,
            ),
            output_ratio=0.9,
            frontier_error_rate=0.0,
            light_error_rate=0.0,
            confidence=ConfidenceLevel.INSUFFICIENT,
            interpretation="test",
        )
        size, _ = _recommend_sample_size(proxy)
        assert size == 50

    def test_strong_proxy(self) -> None:
        proxy = FeatureProxySignal(
            feature="test",
            frontier_output=OutputTokenStats(
                sample_size=100,
                mean=100,
                median=100,
                p25=80,
                p75=120,
            ),
            light_output=OutputTokenStats(
                sample_size=60,
                mean=90,
                median=90,
                p25=70,
                p75=110,
            ),
            output_ratio=0.9,
            frontier_error_rate=0.0,
            light_error_rate=0.0,
            confidence=ConfidenceLevel.STRONG,
            interpretation="test",
        )
        size, _ = _recommend_sample_size(proxy)
        assert size == 20


class TestGetCriteria:
    def test_known_feature(self) -> None:
        criteria = _get_criteria("email_drafting")
        names = [c.name for c in criteria]
        assert "tone" in names
        assert "completeness" in names

    def test_unknown_feature_gets_defaults(self) -> None:
        criteria = _get_criteria("unknown_feature_xyz")
        names = [c.name for c in criteria]
        assert "task_completion" in names
        assert "accuracy" in names

    def test_data_extraction_criteria(self) -> None:
        criteria = _get_criteria("data_extraction")
        names = [c.name for c in criteria]
        assert "format_compliance" in names
        # Accuracy for extraction should require 5/5
        accuracy = next(c for c in criteria if c.name == "accuracy")
        assert accuracy.minimum_acceptable == 5


class TestGenerateEvalPlans:
    def test_generates_plan_per_routable_feature(self) -> None:
        proxy = compute_proxy_signals(
            _dataset([_entry()]),
            CONFIG,
        )
        routing_sim = RoutingSimResult(
            current_cost_usd=100.0,
            optimized_cost_usd=70.0,
            savings_usd=30.0,
            savings_pct=0.30,
            calls_routed_to_light=100,
            calls_kept_on_frontier=50,
            light_model="claude-haiku-4-5-20251001",
            frontier_model="claude-sonnet-4-6",
            light_model_error_rate=0.01,
            frontier_error_rate=0.02,
        )
        optimization = OptimizationPlan(
            routing_simulation=routing_sim,
            recommendations=[],
            total_potential_savings_usd=30.0,
            total_potential_savings_pct=0.30,
        )

        result = generate_eval_plans(optimization, CONFIG, proxy)

        features = {p.feature for p in result.feature_plans}
        assert "email_drafting" in features
        assert "data_extraction" in features
        assert "customer_support_agent" in features
        assert "doc_qa" in features
        # code_review is COMPLEX → not routable
        assert "code_review" not in features

    def test_plan_has_concrete_steps(self) -> None:
        plan = _build_feature_plan(
            "email_drafting",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            CONFIG,
            None,
        )
        assert len(plan.steps) >= 4
        assert plan.estimated_eval_cost_usd > 0
        assert plan.recommended_sample_size >= 5

    def test_cost_estimation_positive(self) -> None:
        proxy = compute_proxy_signals(
            _dataset([_entry()]),
            CONFIG,
        )
        routing_sim = RoutingSimResult(
            current_cost_usd=100.0,
            optimized_cost_usd=70.0,
            savings_usd=30.0,
            savings_pct=0.30,
            calls_routed_to_light=100,
            calls_kept_on_frontier=50,
            light_model="claude-haiku-4-5-20251001",
            frontier_model="claude-sonnet-4-6",
            light_model_error_rate=0.01,
            frontier_error_rate=0.02,
        )
        optimization = OptimizationPlan(
            routing_simulation=routing_sim,
            recommendations=[],
            total_potential_savings_usd=30.0,
            total_potential_savings_pct=0.30,
        )

        result = generate_eval_plans(optimization, CONFIG, proxy)
        assert result.total_estimated_cost_usd > 0

    def test_no_routing_sim_returns_empty(self) -> None:
        proxy = compute_proxy_signals(
            _dataset([_entry()]),
            CONFIG,
        )
        optimization = OptimizationPlan(
            routing_simulation=None,
            recommendations=[],
            total_potential_savings_usd=0.0,
            total_potential_savings_pct=0.0,
        )
        result = generate_eval_plans(optimization, CONFIG, proxy)
        assert len(result.feature_plans) == 0

    def test_sample_size_scales_with_confidence(self) -> None:
        """No proxy → 50 samples; strong proxy → 20 samples."""
        plan_no_proxy = _build_feature_plan(
            "email_drafting",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            CONFIG,
            None,
        )
        assert plan_no_proxy.recommended_sample_size == 50

        strong_proxy = FeatureProxySignal(
            feature="email_drafting",
            frontier_output=OutputTokenStats(
                sample_size=100,
                mean=500,
                median=500,
                p25=400,
                p75=600,
            ),
            light_output=OutputTokenStats(
                sample_size=60,
                mean=480,
                median=480,
                p25=380,
                p75=580,
            ),
            output_ratio=0.96,
            frontier_error_rate=0.01,
            light_error_rate=0.01,
            confidence=ConfidenceLevel.STRONG,
            interpretation="test",
        )
        plan_strong = _build_feature_plan(
            "email_drafting",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            CONFIG,
            strong_proxy,
        )
        assert plan_strong.recommended_sample_size == 20


# ── Layer 2: Judge Logic Tests (no LLM calls) ─────────────────


class TestSelectSamples:
    def test_stratified_selection(self) -> None:
        """Samples come from different length buckets."""
        entries = (
            [_entry(input_tokens=500, offset_seconds=i) for i in range(10)]
            + [_entry(input_tokens=5000, offset_seconds=i + 100) for i in range(10)]
            + [_entry(input_tokens=15000, offset_seconds=i + 200) for i in range(10)]
        )
        samples = _select_samples(entries, target_size=15, seed=42)
        assert len(samples) <= 15

        short = sum(1 for s in samples if s.input_tokens < 2000)
        medium = sum(1 for s in samples if 2000 <= s.input_tokens < 8000)
        long = sum(1 for s in samples if s.input_tokens >= 8000)
        # All buckets should be represented
        assert short > 0
        assert medium > 0
        assert long > 0

    def test_respects_target_size(self) -> None:
        entries = [_entry(offset_seconds=i) for i in range(50)]
        samples = _select_samples(entries, target_size=10, seed=42)
        assert len(samples) <= 10

    def test_small_dataset(self) -> None:
        entries = [_entry(offset_seconds=i) for i in range(3)]
        samples = _select_samples(entries, target_size=10, seed=42)
        assert len(samples) <= 3


class TestComputeVerdict:
    def test_safe_to_route(self) -> None:
        verdict = _compute_verdict(
            mean_delta=-0.05,
            ci_upper=0.10,
            win_tie_rate=0.85,
            per_criterion={"accuracy": -0.1, "completeness": 0.0},
            high_failure_rate=False,
        )
        assert verdict == RoutingVerdict.SAFE_TO_ROUTE

    def test_route_with_monitoring(self) -> None:
        verdict = _compute_verdict(
            mean_delta=-0.30,
            ci_upper=-0.10,
            win_tie_rate=0.60,
            per_criterion={"accuracy": -0.3, "completeness": -0.2},
            high_failure_rate=False,
        )
        assert verdict == RoutingVerdict.ROUTE_WITH_MONITORING

    def test_route_with_caution(self) -> None:
        verdict = _compute_verdict(
            mean_delta=-0.70,
            ci_upper=-0.40,
            win_tie_rate=0.40,
            per_criterion={"accuracy": -0.8, "completeness": -0.5},
            high_failure_rate=False,
        )
        assert verdict == RoutingVerdict.ROUTE_WITH_CAUTION

    def test_do_not_route(self) -> None:
        verdict = _compute_verdict(
            mean_delta=-1.50,
            ci_upper=-1.00,
            win_tie_rate=0.20,
            per_criterion={"accuracy": -1.5, "completeness": -1.0},
            high_failure_rate=False,
        )
        assert verdict == RoutingVerdict.DO_NOT_ROUTE

    def test_high_failure_forces_caution(self) -> None:
        verdict = _compute_verdict(
            mean_delta=0.0,
            ci_upper=0.5,
            win_tie_rate=1.0,
            per_criterion={"accuracy": 0.0},
            high_failure_rate=True,
        )
        assert verdict == RoutingVerdict.ROUTE_WITH_CAUTION

    def test_bad_criterion_blocks_safe(self) -> None:
        """Even with good overall delta, a tanked criterion blocks safe."""
        verdict = _compute_verdict(
            mean_delta=-0.05,
            ci_upper=0.10,
            win_tie_rate=0.85,
            per_criterion={"accuracy": -0.60, "completeness": 0.3},
            high_failure_rate=False,
        )
        assert verdict != RoutingVerdict.SAFE_TO_ROUTE


class TestAggregateFeatureResults:
    def _make_pair(
        self,
        frontier_acc: int = 4,
        light_acc: int = 4,
        preferred: str = "tie",
    ) -> JudgePairResult:
        frontier = CriterionScores(
            accuracy=frontier_acc,
            completeness=4,
            coherence=4,
            instruction_following=4,
        )
        light = CriterionScores(
            accuracy=light_acc,
            completeness=4,
            coherence=4,
            instruction_following=4,
        )
        delta = light.weighted_composite - frontier.weighted_composite
        return JudgePairResult(
            frontier_scores=frontier,
            light_scores=light,
            preferred=preferred,
            quality_delta=round(delta, 4),
            judge_rationale="test",
            frontier_was_a=True,
        )

    def test_equal_scores_near_zero_delta(self) -> None:
        pairs = [self._make_pair() for _ in range(10)]
        result = _aggregate_feature_results("test_feature", pairs, 0)
        assert result.mean_quality_delta == pytest.approx(0.0, abs=0.01)
        assert result.tie_rate == 1.0

    def test_light_worse_negative_delta(self) -> None:
        pairs = [
            self._make_pair(frontier_acc=5, light_acc=3, preferred="frontier") for _ in range(10)
        ]
        result = _aggregate_feature_results("test_feature", pairs, 0)
        assert result.mean_quality_delta < 0
        assert result.loss_rate == 1.0

    def test_confidence_interval_computed(self) -> None:
        pairs = [self._make_pair() for _ in range(20)]
        result = _aggregate_feature_results("test_feature", pairs, 0)
        assert result.ci_lower_95 <= result.mean_quality_delta
        assert result.ci_upper_95 >= result.mean_quality_delta

    def test_per_criterion_deltas(self) -> None:
        pairs = [self._make_pair(frontier_acc=5, light_acc=4) for _ in range(10)]
        result = _aggregate_feature_results("test_feature", pairs, 0)
        assert "accuracy" in result.per_criterion_deltas
        assert result.per_criterion_deltas["accuracy"] < 0


# ── Model Validation Tests ────────────────────────────────────


class TestCriterionScores:
    def test_weighted_composite(self) -> None:
        scores = CriterionScores(
            accuracy=5,
            completeness=4,
            coherence=3,
            instruction_following=4,
        )
        # 0.35*5 + 0.25*4 + 0.15*3 + 0.25*4 = 1.75+1.0+0.45+1.0 = 4.2
        assert scores.weighted_composite == pytest.approx(4.2)

    def test_min_scores(self) -> None:
        scores = CriterionScores(
            accuracy=1,
            completeness=1,
            coherence=1,
            instruction_following=1,
        )
        assert scores.weighted_composite == pytest.approx(1.0)

    def test_max_scores(self) -> None:
        scores = CriterionScores(
            accuracy=5,
            completeness=5,
            coherence=5,
            instruction_following=5,
        )
        assert scores.weighted_composite == pytest.approx(5.0)

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            CriterionScores(
                accuracy=6,
                completeness=4,
                coherence=4,
                instruction_following=4,
            )


class TestJudgeEvalNotRun:
    def test_disabled_reason(self) -> None:
        result = JudgeEvalNotRun(reason="disabled_by_config")
        assert result.reason == "disabled_by_config"

    def test_no_content_reason(self) -> None:
        result = JudgeEvalNotRun(reason="no_prompt_content_available")
        assert result.reason == "no_prompt_content_available"
