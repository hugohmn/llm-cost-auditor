"""Unified LLM client — all API calls go through here.

Provides: centralized error handling, retries, LangFuse tracing, token tracking.
Supports Anthropic (default) and OpenAI as fallback.
Supports tool-use for agentic loops via complete_with_tools().
"""

import logging
import time
from dataclasses import dataclass

import anthropic
from pydantic import BaseModel

from src.utils.config import Config
from src.utils.langfuse_setup import get_langfuse

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


@dataclass(frozen=True)
class ToolCallRequest:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    input: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class LLMToolResponse:
    """Response from a tool-use completion.

    Contains both structured metadata and raw content blocks
    for building the next API message in an agent loop.
    """

    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens"
    text_content: str
    tool_calls: tuple[ToolCallRequest, ...]
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    raw_content: tuple[object, ...]  # SDK content blocks for message forwarding

    def to_assistant_message(self) -> dict[str, str | list[object]]:
        """Build an assistant message for the conversation history."""
        return {"role": "assistant", "content": list(self.raw_content)}


# Approximate pricing per 1M tokens (input/output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "claude-opus-4-6": (15.0, 75.0),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD based on token counts and model pricing."""
    pricing = MODEL_PRICING.get(model, (5.0, 15.0))
    input_cost = (input_tokens / 1_000_000) * pricing[0]
    output_cost = (output_tokens / 1_000_000) * pricing[1]
    return input_cost + output_cost


class UnifiedLLMClient:
    """Single entry point for all LLM API calls.

    Usage:
        client = UnifiedLLMClient(config)
        response = await client.complete(
            system="You are an expert...",
            messages=[{"role": "user", "content": "..."}],
        )
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.anthropic = anthropic.AsyncAnthropic(
            api_key=config.anthropic_api_key,
        )
        self.default_model = config.default_model

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        agent_name: str = "",
    ) -> LLMResponse:
        """Send a completion request to Anthropic.

        All agent calls should use this method.
        LangFuse tracing is automatic when configured.
        """
        target_model = model or self.default_model

        start = time.monotonic()
        try:
            response = await self.anthropic.messages.create(
                model=target_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )
        except anthropic.APIError as e:
            logger.error("Anthropic API error: %s", e)
            raise

        elapsed_ms = (time.monotonic() - start) * 1000
        return self._build_llm_response(
            response,
            target_model,
            elapsed_ms,
            agent_name,
            system,
            messages,
        )

    def _build_llm_response(
        self,
        response: anthropic.types.Message,
        model: str,
        latency_ms: float,
        agent_name: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        """Extract content from response and build an LLMResponse."""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = estimate_cost(model, input_tokens, output_tokens)

        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        self._trace_generation(
            agent_name=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            system_prompt=system_prompt,
            messages=messages,
            output=content,
        )

        logger.info(
            "LLM response: model=%s tokens=%d+%d cost=$%.4f latency=%.0fms",
            model,
            input_tokens,
            output_tokens,
            cost,
            latency_ms,
        )

        return LLMResponse(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

    async def complete_with_tools(
        self,
        *,
        system: str,
        messages: list[dict[str, str | list[object]]],
        tools: list[dict[str, object]],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        agent_name: str = "",
    ) -> LLMToolResponse:
        """Send a completion request with tool definitions.

        Used by the agent loop for tool-use interactions.
        Returns structured response with tool call requests.
        """
        target_model = model or self.default_model

        start = time.monotonic()
        try:
            response = await self.anthropic.messages.create(
                model=target_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
            )
        except anthropic.APIError as e:
            logger.error("Anthropic API error (tool-use): %s", e)
            raise

        elapsed_ms = (time.monotonic() - start) * 1000
        return self._parse_tool_response(response, target_model, elapsed_ms, agent_name)

    @staticmethod
    def _extract_content_blocks(
        response: anthropic.types.Message,
    ) -> tuple[list[str], list[ToolCallRequest]]:
        """Split response content blocks into text parts and tool calls."""
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCallRequest(id=block.id, name=block.name, input=block.input))
        return text_parts, tool_calls

    def _parse_tool_response(
        self,
        response: anthropic.types.Message,
        model: str,
        latency_ms: float,
        agent_name: str,
    ) -> LLMToolResponse:
        """Parse an Anthropic response into a structured LLMToolResponse."""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = estimate_cost(model, input_tokens, output_tokens)

        text_parts, tool_calls = self._extract_content_blocks(response)

        self._trace_generation(
            agent_name=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            system_prompt="(tool-use call)",
            messages=[],
            output="".join(text_parts),
        )

        logger.info(
            "LLM tool response: model=%s stop=%s tools=%d tokens=%d+%d cost=$%.4f",
            model,
            response.stop_reason,
            len(tool_calls),
            input_tokens,
            output_tokens,
            cost,
        )

        return LLMToolResponse(
            stop_reason=response.stop_reason or "end_turn",
            text_content="".join(text_parts),
            tool_calls=tuple(tool_calls),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            raw_content=tuple(response.content),
        )

    def _trace_generation(
        self,
        *,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency_ms: float,
        system_prompt: str,
        messages: list[dict[str, str | list[object]]],
        output: str,
    ) -> None:
        """Record a generation span in LangFuse."""
        lf = get_langfuse()
        if lf is None:
            return

        try:
            trace = lf.trace(
                name=agent_name or "llm_call",
                metadata={"model": model},
            )
            trace.generation(
                name=f"{agent_name or 'generation'}",
                model=model,
                input={"system": system_prompt, "messages": messages},
                output=output,
                usage={
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                },
                metadata={
                    "cost_usd": cost,
                    "latency_ms": round(latency_ms, 1),
                },
            )
        except Exception as e:
            logger.debug("LangFuse trace failed (non-fatal): %s", e)
