"""OpenAI usage CSV export parser.

Handles the format from OpenAI's usage dashboard export.
"""

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.models.log_entry import LogDataset, LogEntry
from src.utils.llm_client import estimate_cost

logger = logging.getLogger(__name__)


def _parse_openai_row(row: dict[str, str]) -> LogEntry:
    """Parse a single OpenAI CSV row into a LogEntry.

    Raises ValueError/KeyError on invalid data.
    """
    model = row.get("model", row.get("model_id", "unknown")).strip()
    input_tokens = int(row.get("prompt_tokens", row.get("n_context_tokens_total", 0)))
    output_tokens = int(row.get("completion_tokens", row.get("n_generated_tokens_total", 0)))
    total_tokens = input_tokens + output_tokens

    cost = float(row.get("cost", 0) or 0)
    if cost == 0:
        cost = estimate_cost(model, input_tokens, output_tokens)

    ts_raw = row.get("timestamp", row.get("date", row.get("created_at", "")))
    ts = datetime.fromisoformat(ts_raw)
    timestamp = ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)

    return LogEntry(
        timestamp=timestamp,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost,
        feature=row.get("api_key_name") or row.get("organization"),
        status="success",
    )


def parse_openai_csv(filepath: Path) -> LogDataset:
    """Parse an OpenAI usage CSV export into a normalized LogDataset."""
    entries: list[LogEntry] = []

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                entries.append(_parse_openai_row(row))
            except (ValueError, KeyError) as e:
                logger.warning("Skipping OpenAI row %d: %s", i, e)
                continue

    if not entries:
        raise ValueError(f"No valid entries parsed from {filepath}")

    entries.sort(key=lambda e: e.timestamp)

    return LogDataset(
        entries=entries,
        source_format="openai_csv",
        date_range_start=entries[0].timestamp,
        date_range_end=entries[-1].timestamp,
    )
