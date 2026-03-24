"""Tests for the agent loop — uses mock LLM client, no real API calls."""

from datetime import UTC, datetime, timedelta

import pytest

from src.agents.base import AgentLoopExhaustedError, run_agent_loop
from src.models.features import AuditConfig
from src.models.log_entry import LogDataset, LogEntry
from src.tools.registry import ToolRegistry
from src.utils.llm_client import LLMToolResponse, ToolCallRequest, estimate_cost

CONFIG = AuditConfig()


class MockLLMClient:
    """Mock client that returns predetermined responses."""

    def __init__(self, responses: list[LLMToolResponse]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    async def complete_with_tools(self, **kwargs: object) -> LLMToolResponse:
        if self.call_count >= len(self.responses):
            return self.responses[-1]
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


def _end_turn_response(text: str = "Done.") -> LLMToolResponse:
    """Create a simple end_turn response."""
    return LLMToolResponse(
        stop_reason="end_turn",
        text_content=text,
        tool_calls=(),
        model="test-model",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_ms=500.0,
        raw_content=(),
    )


def _tool_use_response(
    tool_name: str,
    tool_input: dict[str, str | int | float | bool] | None = None,
) -> LLMToolResponse:
    """Create a tool_use response."""
    return LLMToolResponse(
        stop_reason="tool_use",
        text_content="Let me check...",
        tool_calls=(
            ToolCallRequest(
                id="toolu_test_123",
                name=tool_name,
                input=tool_input or {},
            ),
        ),
        model="test-model",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_ms=500.0,
        raw_content=({"type": "text", "text": "Let me check..."},),
    )


def _simple_registry() -> ToolRegistry:
    """Create a registry with one simple tool."""
    registry = ToolRegistry()
    registry.register(
        name="get_info",
        description="Get info",
        handler=lambda _: '{"answer": 42}',
    )
    return registry


@pytest.mark.asyncio
async def test_agent_loop_immediate_end_turn() -> None:
    """Agent produces final answer immediately."""
    client = MockLLMClient([_end_turn_response("The answer is 42.")])
    result = await run_agent_loop(
        client=client,
        system_prompt="You are helpful.",
        initial_message="What is the answer?",
        registry=_simple_registry(),
        max_iterations=5,
        agent_name="test_agent",
    )
    assert result.final_content == "The answer is 42."
    assert result.total_iterations == 1
    assert result.total_cost_usd == 0.001


@pytest.mark.asyncio
async def test_agent_loop_calls_tool_then_ends() -> None:
    """Agent calls a tool, then produces final answer."""
    client = MockLLMClient(
        [
            _tool_use_response("get_info"),
            _end_turn_response("Based on the info: 42."),
        ]
    )
    result = await run_agent_loop(
        client=client,
        system_prompt="You are helpful.",
        initial_message="Find the answer.",
        registry=_simple_registry(),
        max_iterations=5,
        agent_name="test_agent",
    )
    assert result.final_content == "Based on the info: 42."
    assert result.total_iterations == 2
    assert len(result.tool_call_history) == 1
    assert result.tool_call_history[0].tool_name == "get_info"


@pytest.mark.asyncio
async def test_agent_loop_max_iterations_exhausted() -> None:
    """Agent should raise when exceeding max iterations."""
    # Always returns tool_use — will never finish
    client = MockLLMClient([_tool_use_response("get_info")] * 10)
    with pytest.raises(AgentLoopExhaustedError):
        await run_agent_loop(
            client=client,
            system_prompt="You are helpful.",
            initial_message="Keep going.",
            registry=_simple_registry(),
            max_iterations=3,
            agent_name="test_agent",
        )


@pytest.mark.asyncio
async def test_agent_loop_unknown_tool_handled() -> None:
    """Unknown tool call should not crash — returns error message."""
    client = MockLLMClient(
        [
            _tool_use_response("nonexistent_tool"),
            _end_turn_response("OK."),
        ]
    )
    result = await run_agent_loop(
        client=client,
        system_prompt="You are helpful.",
        initial_message="Try something.",
        registry=_simple_registry(),
        max_iterations=5,
        agent_name="test_agent",
    )
    assert result.total_iterations == 2
    assert "nonexistent_tool" in result.tool_call_history[0].tool_name


@pytest.mark.asyncio
async def test_agent_loop_tracks_tokens() -> None:
    """Token and cost tracking should accumulate across iterations."""
    client = MockLLMClient(
        [
            _tool_use_response("get_info"),
            _tool_use_response("get_info"),
            _end_turn_response("Final answer."),
        ]
    )
    result = await run_agent_loop(
        client=client,
        system_prompt="You are helpful.",
        initial_message="Investigate.",
        registry=_simple_registry(),
        max_iterations=5,
        agent_name="test_agent",
    )
    assert result.total_iterations == 3
    assert result.total_input_tokens == 300  # 100 * 3
    assert result.total_output_tokens == 150  # 50 * 3
    assert result.total_cost_usd == pytest.approx(0.003)


# ── Agent fallback tests ──────────────────────────────────────────────


BASE_TIME = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)


def _make_dataset() -> LogDataset:
    """Create a small test dataset."""
    entries = []
    for i in range(20):
        cost = estimate_cost("claude-sonnet-4-6", 1000, 500)
        entries.append(
            LogEntry(
                timestamp=BASE_TIME + timedelta(seconds=i * 300),
                model="claude-sonnet-4-6",
                input_tokens=1000,
                output_tokens=500,
                total_tokens=1500,
                cost_usd=cost,
                feature="email_drafting",
                status="success",
            )
        )
    return LogDataset(
        entries=entries,
        source_format="test",
        date_range_start=entries[0].timestamp,
        date_range_end=entries[-1].timestamp,
    )


@pytest.mark.asyncio
async def test_analysis_agent_fallback_on_api_error() -> None:
    """Analysis agent should fall back to deterministic on API failure."""
    from src.agents.analysis import run_analysis_agent

    class FailingClient:
        async def complete_with_tools(self, **kwargs: object) -> LLMToolResponse:
            import anthropic

            raise anthropic.APIConnectionError(request=None)  # type: ignore[arg-type]

    dataset = _make_dataset()
    result = await run_analysis_agent(dataset, FailingClient())  # type: ignore[arg-type]

    # Should have valid results from deterministic fallback
    assert result.total_calls == 20
    assert result.total_cost_usd > 0
    assert len(result.cost_by_model) >= 1


@pytest.mark.asyncio
async def test_analysis_agent_insights_populated() -> None:
    """Agent insights should be stored on AnalysisResult."""
    from src.agents.analysis import _build_analysis_from_agent

    dataset = _make_dataset()
    agent_output = '{"findings_summary": "Test finding", "key_insights": ["insight1", "insight2"]}'
    result = _build_analysis_from_agent(dataset, CONFIG, agent_output)

    assert result.agent_findings_summary == "Test finding"
    assert result.agent_key_insights == ["insight1", "insight2"]
