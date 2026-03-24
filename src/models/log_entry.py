"""Normalized log entry schema — the common format all parsers produce."""

from datetime import datetime

from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    """A single normalized LLM API call record."""

    timestamp: datetime
    model: str = Field(description="Model identifier (e.g. gpt-4o, claude-sonnet-4-6)")
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0, description="Cost in USD for this call")
    latency_ms: float | None = Field(default=None, ge=0.0)
    feature: str | None = Field(default=None, description="Application feature or endpoint")
    status: str = Field(default="success", description="success | error | timeout")
    metadata: dict[str, str | int | list[str]] = Field(default_factory=dict)
    input_text: str | None = Field(
        default=None,
        description="Raw prompt text, when available in source data.",
    )
    output_text: str | None = Field(
        default=None,
        description="Raw completion text, when available in source data.",
    )


class LogDataset(BaseModel):
    """A collection of normalized log entries with summary stats."""

    entries: list[LogEntry]
    source_format: str = Field(description="Original format: langfuse | openai_csv | generic")
    date_range_start: datetime | None = None
    date_range_end: datetime | None = None

    @property
    def total_entries(self) -> int:
        return len(self.entries)

    @property
    def total_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    @property
    def total_tokens(self) -> int:
        return sum(e.total_tokens for e in self.entries)
