"""LangFuse JSON export parser.

Handles the export format from LangFuse's generation list export.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from src.models.log_entry import LogDataset, LogEntry
from src.utils.llm_client import estimate_cost

logger = logging.getLogger(__name__)


def _parse_langfuse_entry(rec: dict[str, object]) -> LogEntry:
    """Parse a single LangFuse record into a LogEntry.

    Raises ValueError/KeyError/TypeError on invalid data.
    """
    model = rec.get("model", "unknown")
    input_tokens = int(rec.get("promptTokens", 0) or rec.get("input_tokens", 0))
    output_tokens = int(rec.get("completionTokens", 0) or rec.get("output_tokens", 0))
    total_tokens = input_tokens + output_tokens

    # LangFuse may provide cost or we estimate it
    cost = float(rec.get("calculatedTotalCost", 0) or 0)
    if cost == 0:
        cost = estimate_cost(str(model), input_tokens, output_tokens)

    # Parse timestamp
    ts_raw = rec.get("startTime") or rec.get("createdAt", "")
    timestamp = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))

    # Latency
    latency = _parse_latency(rec)

    return LogEntry(
        timestamp=timestamp,
        model=str(model),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost,
        latency_ms=latency,
        feature=rec.get("traceName") or rec.get("name"),  # type: ignore[arg-type]
        status="error" if rec.get("level") == "ERROR" else "success",
        metadata={
            k: v for k, v in rec.items() if k in ("traceId", "observationId", "userId", "tags")
        },
    )


def _parse_latency(rec: dict[str, object]) -> float | None:
    """Extract latency in milliseconds from a LangFuse record."""
    if rec.get("latency") is not None:
        return float(rec["latency"])  # type: ignore[arg-type]
    if rec.get("endTime") and rec.get("startTime"):
        start = datetime.fromisoformat(str(rec["startTime"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(rec["endTime"]).replace("Z", "+00:00"))
        return (end - start).total_seconds() * 1000
    return None


def parse_langfuse_export(filepath: Path) -> LogDataset:
    """Parse a LangFuse JSON export into a normalized LogDataset."""
    with open(filepath) as f:
        raw = json.load(f)

    # LangFuse exports can be a list of generations or wrapped in {"data": [...]}
    if isinstance(raw, dict) and "data" in raw:
        records = raw["data"]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError(f"Unexpected LangFuse export format in {filepath}")

    entries: list[LogEntry] = []
    for i, rec in enumerate(records):
        try:
            entries.append(_parse_langfuse_entry(rec))
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("Skipping LangFuse record %d: %s", i, e)
            continue

    if not entries:
        raise ValueError(f"No valid entries parsed from {filepath}")

    entries.sort(key=lambda e: e.timestamp)

    return LogDataset(
        entries=entries,
        source_format="langfuse",
        date_range_start=entries[0].timestamp,
        date_range_end=entries[-1].timestamp,
    )
