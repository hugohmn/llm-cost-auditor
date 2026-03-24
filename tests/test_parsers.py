"""Tests for all three parsers against sample data."""

from pathlib import Path

import pytest

from src.parsers.generic import parse_generic_csv
from src.parsers.langfuse import parse_langfuse_export
from src.parsers.openai_csv import parse_openai_csv

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample-data"


class TestLangFuseParser:
    def test_entry_count(self) -> None:
        ds = parse_langfuse_export(SAMPLE_DIR / "example-logs.json")
        assert ds.total_entries == 5084

    def test_source_format(self) -> None:
        ds = parse_langfuse_export(SAMPLE_DIR / "example-logs.json")
        assert ds.source_format == "langfuse"

    def test_total_cost_range(self) -> None:
        ds = parse_langfuse_export(SAMPLE_DIR / "example-logs.json")
        assert 200.0 < ds.total_cost_usd < 300.0

    def test_date_range(self) -> None:
        ds = parse_langfuse_export(SAMPLE_DIR / "example-logs.json")
        assert ds.date_range_start is not None
        assert ds.date_range_end is not None
        delta = ds.date_range_end - ds.date_range_start
        assert 28 <= delta.days <= 31

    def test_models_parsed(self) -> None:
        ds = parse_langfuse_export(SAMPLE_DIR / "example-logs.json")
        models = {e.model for e in ds.entries}
        assert "claude-sonnet-4-6" in models
        assert "gpt-4o" in models
        assert "claude-haiku-4-5-20251001" in models

    def test_features_parsed(self) -> None:
        ds = parse_langfuse_export(SAMPLE_DIR / "example-logs.json")
        features = {e.feature for e in ds.entries}
        assert "customer_support_agent" in features
        assert "code_review" in features

    def test_entries_sorted_by_timestamp(self) -> None:
        ds = parse_langfuse_export(SAMPLE_DIR / "example-logs.json")
        timestamps = [e.timestamp for e in ds.entries]
        assert timestamps == sorted(timestamps)

    def test_cost_matches_tokens(self) -> None:
        """Verify calculatedTotalCost = tokens x pricing."""
        from src.utils.llm_client import estimate_cost

        ds = parse_langfuse_export(SAMPLE_DIR / "example-logs.json")
        for entry in ds.entries[:50]:
            expected = estimate_cost(entry.model, entry.input_tokens, entry.output_tokens)
            assert abs(entry.cost_usd - expected) < 0.0001, (
                f"{entry.model}: {entry.cost_usd} != {expected}"
            )


class TestOpenAIParser:
    def test_entry_count(self) -> None:
        ds = parse_openai_csv(SAMPLE_DIR / "example-openai.csv")
        assert ds.total_entries == 2000

    def test_source_format(self) -> None:
        ds = parse_openai_csv(SAMPLE_DIR / "example-openai.csv")
        assert ds.source_format == "openai_csv"

    def test_models_parsed(self) -> None:
        ds = parse_openai_csv(SAMPLE_DIR / "example-openai.csv")
        models = {e.model for e in ds.entries}
        assert "gpt-4o" in models

    def test_total_cost_positive(self) -> None:
        ds = parse_openai_csv(SAMPLE_DIR / "example-openai.csv")
        assert ds.total_cost_usd > 0


class TestGenericParser:
    def test_entry_count(self) -> None:
        ds = parse_generic_csv(SAMPLE_DIR / "example-generic.csv")
        assert ds.total_entries == 1000

    def test_source_format(self) -> None:
        ds = parse_generic_csv(SAMPLE_DIR / "example-generic.csv")
        assert ds.source_format == "generic"

    def test_models_parsed(self) -> None:
        ds = parse_generic_csv(SAMPLE_DIR / "example-generic.csv")
        models = {e.model for e in ds.entries}
        assert len(models) >= 2

    def test_features_parsed(self) -> None:
        ds = parse_generic_csv(SAMPLE_DIR / "example-generic.csv")
        features = {e.feature for e in ds.entries if e.feature}
        assert len(features) >= 3

    def test_missing_required_columns_raises(self, tmp_path: Path) -> None:
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("a,b,c\n1,2,3\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            parse_generic_csv(bad_csv)
