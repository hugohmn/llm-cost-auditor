"""Tests for agent tool handlers — no LLM calls, fast, deterministic."""

import json
from datetime import UTC, datetime, timedelta

from src.models.log_entry import LogDataset, LogEntry
from src.tools.analysis_tools import build_analysis_registry
from src.tools.optimization_tools import build_optimization_registry
from src.tools.report_tools import build_report_registry
from src.utils.llm_client import estimate_cost

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
    return LogDataset(
        entries=entries,
        source_format="test",
        date_range_start=entries[0].timestamp if entries else None,
        date_range_end=entries[-1].timestamp if entries else None,
    )


class TestAnalysisTools:
    def _registry(self) -> tuple:
        entries = [
            _entry(model="claude-sonnet-4-6", input_tokens=i * 100, offset_seconds=i)
            for i in range(1, 21)
        ]
        ds = _dataset(entries)
        return build_analysis_registry(ds), ds

    def test_get_dataset_summary_returns_json(self) -> None:
        registry, _ = self._registry()
        result = registry.handlers["get_dataset_summary"]({})
        data = json.loads(result)
        assert data["total_entries"] == 20
        assert data["total_cost_usd"] > 0
        assert "models" in data
        assert "features" in data

    def test_compute_cost_by_model_returns_json(self) -> None:
        registry, _ = self._registry()
        result = registry.handlers["compute_cost_by_model"]({})
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["model"] == "claude-sonnet-4-6"

    def test_compute_cost_by_feature_returns_json(self) -> None:
        registry, _ = self._registry()
        result = registry.handlers["compute_cost_by_feature"]({})
        data = json.loads(result)
        assert isinstance(data, list)
        assert data[0]["feature"] == "email_drafting"

    def test_detect_bloated_prompts_no_bloat(self) -> None:
        registry, _ = self._registry()
        result = registry.handlers["detect_bloated_prompts"]({})
        data = json.loads(result)
        assert data["result"] is None

    def test_detect_wrong_model_detects(self) -> None:
        """Frontier model on simple feature should be flagged."""
        entries = [_entry(model="claude-sonnet-4-6", feature="email_drafting") for _ in range(5)]
        ds = _dataset(entries)
        registry = build_analysis_registry(ds)
        result = registry.handlers["detect_wrong_model"]({})
        data = json.loads(result)
        assert data["affected_calls"] == 5

    def test_simulate_routing_returns_json(self) -> None:
        registry, _ = self._registry()
        result = registry.handlers["simulate_routing"]({})
        data = json.loads(result)
        assert "savings_usd" in data
        assert "calls_routed_to_light" in data

    def test_registry_has_all_tools(self) -> None:
        registry, _ = self._registry()
        names = {d.name for d in registry.definitions}
        expected = {
            "get_dataset_summary",
            "compute_cost_by_model",
            "compute_cost_by_feature",
            "detect_bloated_prompts",
            "detect_wrong_model",
            "detect_excessive_retries",
            "detect_cacheable_duplicates",
            "simulate_routing",
        }
        assert names == expected

    def test_to_api_format(self) -> None:
        registry, _ = self._registry()
        api_tools = registry.to_api_format()
        assert len(api_tools) == 8
        for tool in api_tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool


class TestRecommendationParsing:
    """Test recommendation parsing resilience."""

    def test_invalid_priority_clamped(self) -> None:
        """Priority out of range should be clamped, not crash."""
        from src.agents.optimization import _parse_recommendations

        raw_json = '[{"priority": 8, "category": "routing", "title": "Test", '
        raw_json += '"description": "Test", "estimated_monthly_savings_usd": 10.0, '
        raw_json += '"implementation_effort": "low"}]'
        recs = _parse_recommendations(raw_json)
        assert len(recs) == 1
        assert recs[0].priority == 5  # clamped to max

    def test_missing_field_skipped(self) -> None:
        """Recommendation missing required field should be skipped."""
        from src.agents.optimization import _parse_recommendations

        raw_json = '[{"priority": 1, "category": "routing"}]'
        recs = _parse_recommendations(raw_json)
        assert len(recs) == 0  # skipped due to missing fields

    def test_invalid_json_returns_empty(self) -> None:
        """Unparseable JSON should return empty list."""
        from src.agents.optimization import _parse_recommendations

        recs = _parse_recommendations("not json at all")
        assert recs == []


class TestOptimizationTools:
    def test_estimate_model_switch_savings(self) -> None:
        entries = [_entry(model="claude-sonnet-4-6", feature="email_drafting") for _ in range(10)]
        ds = _dataset(entries)
        from src.agents.analysis import analyze

        analysis = analyze(ds)
        registry = build_optimization_registry(ds, analysis)

        result = registry.handlers["estimate_model_switch_savings"](
            {"feature": "email_drafting", "target_model": "claude-haiku-4-5-20251001"}
        )
        data = json.loads(result)
        assert data["savings_usd"] > 0
        assert data["total_calls"] == 10

    def test_get_feature_detail(self) -> None:
        entries = [
            _entry(model="claude-sonnet-4-6", feature="code_review", input_tokens=5000)
            for _ in range(10)
        ]
        ds = _dataset(entries)
        from src.agents.analysis import analyze

        analysis = analyze(ds)
        registry = build_optimization_registry(ds, analysis)

        result = registry.handlers["get_feature_detail"]({"feature": "code_review"})
        data = json.loads(result)
        assert data["total_calls"] == 10
        assert "input_token_percentiles" in data

    def test_unknown_feature_returns_error(self) -> None:
        entries = [_entry() for _ in range(5)]
        ds = _dataset(entries)
        from src.agents.analysis import analyze

        analysis = analyze(ds)
        registry = build_optimization_registry(ds, analysis)

        result = registry.handlers["get_feature_detail"]({"feature": "nonexistent"})
        data = json.loads(result)
        assert "error" in data


class TestReportTools:
    def test_get_cost_overview(self) -> None:
        entries = [_entry() for _ in range(10)]
        ds = _dataset(entries)
        from src.agents.analysis import analyze
        from src.models.recommendation import OptimizationPlan

        analysis = analyze(ds)
        optimization = OptimizationPlan(
            recommendations=[],
            total_potential_savings_usd=0,
            total_potential_savings_pct=0,
        )
        registry = build_report_registry(ds, analysis, optimization)

        result = registry.handlers["get_cost_overview"]({})
        data = json.loads(result)
        assert data["total_calls"] == 10
        assert data["total_cost_usd"] > 0

    def test_get_top_cost_drivers(self) -> None:
        entries = [_entry() for _ in range(10)]
        ds = _dataset(entries)
        from src.agents.analysis import analyze
        from src.models.recommendation import OptimizationPlan

        analysis = analyze(ds)
        optimization = OptimizationPlan(
            recommendations=[],
            total_potential_savings_usd=0,
            total_potential_savings_pct=0,
        )
        registry = build_report_registry(ds, analysis, optimization)

        result = registry.handlers["get_top_cost_drivers"]({})
        data = json.loads(result)
        assert "top_models" in data
        assert "top_features" in data
