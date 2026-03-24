"""Tests for Pydantic model validation."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.models.analysis import (
    AnalysisResult,
    WastePattern,
)
from src.models.log_entry import LogDataset, LogEntry
from src.models.recommendation import (
    OptimizationPlan,
    Recommendation,
)
from src.models.report import AuditReport


class TestLogEntry:
    def test_valid_entry(self) -> None:
        entry = LogEntry(
            timestamp=datetime(2026, 3, 1, tzinfo=UTC),
            model="claude-sonnet-4-6",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            cost_usd=0.0105,
        )
        assert entry.model == "claude-sonnet-4-6"
        assert entry.status == "success"

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LogEntry(
                timestamp=datetime(2026, 3, 1, tzinfo=UTC),
                model="test",
                input_tokens=-1,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
            )

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LogEntry(
                timestamp=datetime(2026, 3, 1, tzinfo=UTC),
                model="test",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=-1.0,
            )

    def test_metadata_typed(self) -> None:
        entry = LogEntry(
            timestamp=datetime(2026, 3, 1, tzinfo=UTC),
            model="test",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            metadata={"traceId": "abc-123", "tags": ["test"]},
        )
        assert entry.metadata["traceId"] == "abc-123"


class TestLogDataset:
    def test_properties(self) -> None:
        entries = [
            LogEntry(
                timestamp=datetime(2026, 3, 1, tzinfo=UTC),
                model="test",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cost_usd=0.01,
            ),
            LogEntry(
                timestamp=datetime(2026, 3, 2, tzinfo=UTC),
                model="test",
                input_tokens=200,
                output_tokens=100,
                total_tokens=300,
                cost_usd=0.02,
            ),
        ]
        ds = LogDataset(entries=entries, source_format="test")
        assert ds.total_entries == 2
        assert ds.total_cost_usd == pytest.approx(0.03)
        assert ds.total_tokens == 450


class TestWastePattern:
    def test_valid_pattern(self) -> None:
        wp = WastePattern(
            pattern_type="bloated_prompt",
            description="Test waste",
            affected_calls=10,
            estimated_waste_usd=5.50,
            methodology="Per-feature median analysis",
        )
        assert wp.pattern_type == "bloated_prompt"
        assert wp.methodology == "Per-feature median analysis"


class TestRecommendation:
    def test_priority_range(self) -> None:
        with pytest.raises(ValidationError):
            Recommendation(
                priority=0,
                category="routing",
                title="Test",
                description="Test",
                estimated_monthly_savings_usd=10.0,
                implementation_effort="low",
            )

    def test_valid_recommendation(self) -> None:
        rec = Recommendation(
            priority=1,
            category="routing",
            title="Route to Haiku",
            description="Switch simple calls",
            estimated_monthly_savings_usd=10.0,
            implementation_effort="low",
        )
        assert rec.priority == 1


class TestAuditReport:
    def test_headline_savings(self) -> None:
        report = AuditReport(
            log_source="test",
            total_entries_analyzed=100,
            analysis=AnalysisResult(
                total_cost_usd=100.0,
                total_calls=100,
                total_tokens=10000,
                date_range_days=30,
                monthly_projected_cost_usd=100.0,
                cost_by_model=[],
                cost_by_feature=[],
                waste_patterns=[],
                total_waste_usd=0.0,
                deduplicated_waste_usd=0.0,
                waste_pct=0.0,
            ),
            optimization=OptimizationPlan(
                recommendations=[],
                total_potential_savings_usd=25.0,
                total_potential_savings_pct=0.25,
            ),
            executive_summary="Test summary",
        )
        assert "$25" in report.headline_savings
        assert "25%" in report.headline_savings
