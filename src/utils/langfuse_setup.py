"""LangFuse tracing setup — initialized once at startup."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langfuse import Langfuse

    from src.utils.config import Config

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency
_langfuse_client: Langfuse | None = None


def init_langfuse(config: Config) -> bool:
    """Initialize LangFuse client for observability.

    Returns True if LangFuse was initialized, False otherwise.
    The app runs fine without LangFuse — it just won't trace.
    """
    global _langfuse_client  # noqa: PLW0603

    if not config.langfuse_public_key or not config.langfuse_secret_key:
        logger.warning("LangFuse credentials not set — running without tracing")
        return False

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=config.langfuse_host,
        )
        logger.info("LangFuse initialized: %s", config.langfuse_host)
        return True
    except Exception as e:
        logger.warning("LangFuse init failed (non-fatal): %s", e)
        return False


def get_langfuse() -> Langfuse | None:
    """Get the LangFuse client, or None if not initialized."""
    return _langfuse_client


def flush_langfuse() -> None:
    """Flush pending LangFuse events."""
    if _langfuse_client is not None:
        _langfuse_client.flush()
