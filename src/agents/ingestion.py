"""Ingestion Agent — detects log format and parses into normalized LogDataset.

This is a pure data pipeline step (no LLM call needed).
"""

import logging
from pathlib import Path

from src.models.log_entry import LogDataset
from src.parsers.generic import parse_generic_csv
from src.parsers.langfuse import parse_langfuse_export
from src.parsers.openai_csv import parse_openai_csv

logger = logging.getLogger(__name__)


def detect_format(filepath: Path) -> str:
    """Auto-detect log file format from extension and content."""
    suffix = filepath.suffix.lower()

    if suffix == ".json":
        return "langfuse"
    elif suffix == ".csv":
        # Peek at headers to distinguish OpenAI from generic
        with open(filepath) as f:
            header = f.readline().lower()
        if "prompt_tokens" in header or "n_context_tokens" in header:
            return "openai_csv"
        return "generic"
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def ingest(filepath: Path) -> LogDataset:
    """Parse a log file into a normalized LogDataset.

    Auto-detects the format and delegates to the appropriate parser.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    fmt = detect_format(path)
    logger.info("Detected format: %s for %s", fmt, path.name)

    match fmt:
        case "langfuse":
            dataset = parse_langfuse_export(path)
        case "openai_csv":
            dataset = parse_openai_csv(path)
        case "generic":
            dataset = parse_generic_csv(path)
        case _:
            raise ValueError(f"Unknown format: {fmt}")

    logger.info(
        "Ingested %d entries, total cost: $%.2f, total tokens: %d",
        dataset.total_entries,
        dataset.total_cost_usd,
        dataset.total_tokens,
    )
    return dataset
