"""Tests for analysis agent — deterministic waste detection.

Uses hand-crafted LogDatasets, no LLM calls needed.
"""

from datetime import UTC, datetime, timedelta

import pytest

from src.agents.analysis import (
    analyze,
    compute_cost_by_feature,
    compute_cost_by_model,
    detect_bloated_prompts,
    detect_cacheable_duplicates,
    detect_excessive_retries,
    detect_wrong_model,
)
from src.models.features import AuditConfig
from src.models.log_entry import LogDataset, LogEntry
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
    )


def _dataset(entries: list[LogEntry]) -> LogDataset:
    """Helper to build a LogDataset."""
    return LogDataset(
        entries=entries,
        source_format="test",
        date_range_start=entries[0].timestamp if entries else None,
        date_range_end=entries[-1].timestamp if entries else None,
    )


class TestComputeCostByModel:
    def test_single_model(self) -> None:
        entries = [_entry(model="gpt-4o") for _ in range(5)]
        ds = _dataset(entries)
        result = compute_cost_by_model(ds)
        assert len(result) == 1
        assert result[0].model == "gpt-4o"
        assert result[0].total_calls == 5

    def test_multiple_models_sorted_by_cost(self) -> None:
        entries = [
            _entry(model="claude-sonnet-4-6", input_tokens=10000),
            _entry(model="claude-haiku-4-5-20251001", input_tokens=1000),
        ]
        ds = _dataset(entries)
        result = compute_cost_by_model(ds)
        assert result[0].model == "claude-sonnet-4-6"
        assert result[0].total_cost_usd > result[1].total_cost_usd

    def test_error_rate(self) -> None:
        entries = [
            _entry(status="success"),
            _entry(status="success"),
            _entry(status="error"),
        ]
        ds = _dataset(entries)
        result = compute_cost_by_model(ds)
        assert result[0].error_rate == pytest.approx(1 / 3)


class TestComputeCostByFeature:
    def test_groups_by_feature(self) -> None:
        entries = [
            _entry(feature="email_drafting"),
            _entry(feature="email_drafting"),
            _entry(feature="code_review"),
        ]
        ds = _dataset(entries)
        result = compute_cost_by_feature(ds)
        features = {r.feature for r in result}
        assert features == {"email_drafting", "code_review"}

    def test_primary_model(self) -> None:
        entries = [
            _entry(feature="test", model="claude-sonnet-4-6"),
            _entry(feature="test", model="claude-sonnet-4-6"),
            _entry(feature="test", model="gpt-4o"),
        ]
        ds = _dataset(entries)
        result = compute_cost_by_feature(ds)
        assert result[0].primary_model == "claude-sonnet-4-6"


class TestDetectBloatedPrompts:
    def test_detects_bloated(self) -> None:
        """Entries >2x median AND >8K should be flagged."""
        normal = [_entry(input_tokens=1000) for _ in range(20)]
        bloated = [_entry(input_tokens=12000) for _ in range(5)]
        ds = _dataset(normal + bloated)
        result = detect_bloated_prompts(ds, CONFIG)
        assert result is not None
        assert result.affected_calls == 5
        assert result.estimated_waste_usd > 0

    def test_no_bloat_when_similar(self) -> None:
        """All entries similar — no bloat detected."""
        entries = [_entry(input_tokens=1500) for _ in range(20)]
        ds = _dataset(entries)
        result = detect_bloated_prompts(ds, CONFIG)
        assert result is None

    def test_complex_features_not_flagged(self) -> None:
        """code_review is COMPLEX — not checked for bloat."""
        entries = [_entry(feature="code_review", input_tokens=50000) for _ in range(20)]
        ds = _dataset(entries)
        result = detect_bloated_prompts(ds, CONFIG)
        assert result is None


class TestDetectWrongModel:
    def test_detects_wrong_model(self) -> None:
        """Frontier model on simple feature with low tokens."""
        entries = [
            _entry(
                model="claude-sonnet-4-6",
                feature="email_drafting",
                input_tokens=1000,
                output_tokens=300,
            )
            for _ in range(5)
        ]
        ds = _dataset(entries)
        result = detect_wrong_model(ds, CONFIG)
        assert result is not None
        assert result.affected_calls == 5
        assert result.estimated_waste_usd > 0

    def test_haiku_not_flagged(self) -> None:
        """Haiku on simple feature is correct — no waste."""
        entries = [
            _entry(
                model="claude-haiku-4-5-20251001",
                feature="email_drafting",
                input_tokens=1000,
            )
            for _ in range(5)
        ]
        ds = _dataset(entries)
        result = detect_wrong_model(ds, CONFIG)
        assert result is None

    def test_complex_feature_not_flagged(self) -> None:
        """Frontier on code_review is correct — no waste."""
        entries = [
            _entry(
                model="claude-sonnet-4-6",
                feature="code_review",
                input_tokens=1000,
            )
            for _ in range(5)
        ]
        ds = _dataset(entries)
        result = detect_wrong_model(ds, CONFIG)
        assert result is None

    def test_moderate_high_tokens_not_flagged(self) -> None:
        """MODERATE feature with high tokens should NOT be flagged."""
        entries = [
            _entry(
                model="claude-sonnet-4-6",
                feature="customer_support_agent",
                input_tokens=5000,
            )
            for _ in range(5)
        ]
        ds = _dataset(entries)
        result = detect_wrong_model(ds, CONFIG)
        assert result is None

    def test_moderate_low_tokens_flagged(self) -> None:
        """MODERATE feature with low tokens SHOULD be flagged."""
        entries = [
            _entry(
                model="claude-sonnet-4-6",
                feature="customer_support_agent",
                input_tokens=2000,
                output_tokens=500,
            )
            for _ in range(5)
        ]
        ds = _dataset(entries)
        result = detect_wrong_model(ds, CONFIG)
        assert result is not None
        assert result.affected_calls == 5


class TestDetectExcessiveRetries:
    def test_detects_retry_chain(self) -> None:
        """Error followed by success within 60s = retry."""
        entries = [
            _entry(status="error", offset_seconds=0),
            _entry(status="success", offset_seconds=5),
        ]
        ds = _dataset(entries)
        result = detect_excessive_retries(ds, CONFIG)
        assert result is not None
        assert result.affected_calls == 1

    def test_no_retry_when_only_errors(self) -> None:
        """Errors without a following success are not retries."""
        entries = [
            _entry(status="error", offset_seconds=0),
            _entry(status="error", offset_seconds=5),
        ]
        ds = _dataset(entries)
        result = detect_excessive_retries(ds, CONFIG)
        assert result is None

    def test_no_retry_when_too_far_apart(self) -> None:
        """Error + success > 60s apart is not a retry."""
        entries = [
            _entry(status="error", offset_seconds=0),
            _entry(status="success", offset_seconds=120),
        ]
        ds = _dataset(entries)
        result = detect_excessive_retries(ds, CONFIG)
        assert result is None


class TestDetectCacheableDuplicates:
    def test_detects_duplicates(self) -> None:
        """Same feature+model+tokens within 2 min = cacheable."""
        entries = [
            _entry(input_tokens=1000, offset_seconds=0),
            _entry(input_tokens=1000, offset_seconds=30),
            _entry(input_tokens=1000, offset_seconds=60),
        ]
        ds = _dataset(entries)
        result = detect_cacheable_duplicates(ds, CONFIG)
        assert result is not None
        assert result.affected_calls == 2  # first is original

    def test_no_duplicates_different_features(self) -> None:
        """Different features are not duplicates."""
        entries = [
            _entry(feature="email_drafting", offset_seconds=0),
            _entry(feature="code_review", offset_seconds=30),
        ]
        ds = _dataset(entries)
        result = detect_cacheable_duplicates(ds, CONFIG)
        assert result is None

    def test_no_duplicates_when_far_apart(self) -> None:
        """Same call >2 min apart is not cacheable."""
        entries = [
            _entry(offset_seconds=0),
            _entry(offset_seconds=300),
        ]
        ds = _dataset(entries)
        result = detect_cacheable_duplicates(ds, CONFIG)
        assert result is None


class TestAnalyze:
    def test_full_analysis(self) -> None:
        """End-to-end analysis produces valid AnalysisResult."""
        entries = [_entry(input_tokens=i * 100, offset_seconds=i) for i in range(1, 21)]
        ds = _dataset(entries)
        result = analyze(ds, CONFIG)
        assert result.total_calls == 20
        assert result.total_cost_usd > 0
        assert len(result.cost_by_model) >= 1
        assert len(result.cost_by_feature) >= 1
        assert result.date_range_days >= 0

    def test_empty_waste_on_clean_data(self) -> None:
        """Clean data with no waste patterns."""
        entries = [
            _entry(
                model="claude-haiku-4-5-20251001",
                feature="code_review",
                input_tokens=10000,
                offset_seconds=i * 300,
            )
            for i in range(20)
        ]
        ds = _dataset(entries)
        result = analyze(ds, CONFIG)
        assert result.total_waste_usd == 0.0
        assert len(result.waste_patterns) == 0
