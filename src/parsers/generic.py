"""Generic CSV parser — handles the simplest common format.

Expected columns: timestamp, model, input_tokens, output_tokens, cost_usd
Optional columns: latency_ms, feature, status, metadata
"""

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.models.log_entry import LogDataset, LogEntry

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"timestamp", "model", "input_tokens", "output_tokens"}


def _parse_generic_row(row: dict[str, str]) -> LogEntry:
    """Parse a single generic CSV row into a LogEntry.

    Raises ValueError/KeyError on invalid data.
    """
    input_tokens = int(row["input_tokens"])
    output_tokens = int(row["output_tokens"])
    total_tokens = input_tokens + output_tokens

    ts = datetime.fromisoformat(row["timestamp"])
    timestamp = ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)

    return LogEntry(
        timestamp=timestamp,
        model=row["model"].strip(),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=float(row.get("cost_usd", 0.0)),
        latency_ms=float(row["latency_ms"]) if row.get("latency_ms") else None,
        feature=row.get("feature", "").strip() or None,
        status=row.get("status", "success").strip(),
        input_text=row.get("input_text", "").strip() or None,
        output_text=row.get("output_text", "").strip() or None,
    )


def parse_generic_csv(filepath: Path) -> LogDataset:
    """Parse a generic CSV file into a normalized LogDataset."""
    entries: list[LogEntry] = []

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV file: {filepath}")

        fields = set(reader.fieldnames)
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for i, row in enumerate(reader):
            try:
                entries.append(_parse_generic_row(row))
            except (ValueError, KeyError) as e:
                logger.warning("Skipping row %d: %s", i, e)
                continue

    if not entries:
        raise ValueError(f"No valid entries parsed from {filepath}")

    entries.sort(key=lambda e: e.timestamp)

    return LogDataset(
        entries=entries,
        source_format="generic",
        date_range_start=entries[0].timestamp,
        date_range_end=entries[-1].timestamp,
    )
