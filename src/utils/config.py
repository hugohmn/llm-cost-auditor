"""Environment configuration loader."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment variables."""

    anthropic_api_key: str
    openai_api_key: str | None = None

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"

    log_level: str = "INFO"
    default_model: str = "claude-sonnet-4-6"


def load_config() -> Config:
    """Load configuration from .env file and environment variables."""
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required")

    return Config(
        anthropic_api_key=api_key,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        langfuse_host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        default_model=os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-6"),
    )
