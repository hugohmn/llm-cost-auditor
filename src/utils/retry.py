"""Retry and fallback utilities for LLM calls."""

import asyncio
import logging
from collections.abc import Awaitable, Callable

import anthropic

logger = logging.getLogger(__name__)

# Errors that should never be retried — immediate re-raise
_NON_RETRYABLE = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
)


async def with_retry[T](
    fn: Callable[..., Awaitable[T]],
    *args: object,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs: object,
) -> T:
    """Execute an async function with exponential backoff retry.

    Non-retryable errors (auth, permission, 404) are raised immediately.
    Transient errors (rate limits, timeouts, server errors) are retried.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            return await fn(*args, **kwargs)
        except _NON_RETRYABLE:
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "All %d attempts failed: %s",
                    max_retries,
                    e,
                )

    raise last_error  # type: ignore[misc]
