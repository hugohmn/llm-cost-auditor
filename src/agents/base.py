"""Agent loop — ReAct-style tool-use loop using Anthropic's native API.

All agents share this loop. The LLM receives a system prompt,
initial context, and tool definitions. It reasons about which
tools to call, executes them, and iterates until producing a
final answer or hitting the iteration limit.
"""

import json
import logging

from src.models.agent_result import AgentResult, ToolCallRecord
from src.tools.registry import ToolHandler, ToolRegistry
from src.utils.langfuse_setup import get_langfuse
from src.utils.llm_client import LLMToolResponse, UnifiedLLMClient

logger = logging.getLogger(__name__)


class AgentLoopExhaustedError(Exception):
    """Raised when the agent exceeds its max iteration limit."""


async def run_agent_loop(
    *,
    client: UnifiedLLMClient,
    system_prompt: str,
    initial_message: str,
    registry: ToolRegistry,
    max_iterations: int = 15,
    agent_name: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> AgentResult:
    """Execute a tool-use agent loop.

    The agent receives a system prompt and initial user message,
    then iteratively calls tools until it produces a final answer.
    """
    messages: list[dict[str, str | list[object]]] = [
        {"role": "user", "content": initial_message},
    ]
    tool_api = registry.to_api_format()
    handlers = registry.handlers
    history: list[ToolCallRecord] = []
    totals = _TokenTotals()

    _trace_agent_start(agent_name, max_iterations, len(tool_api))

    for iteration in range(max_iterations):
        response = await _call_llm(
            client,
            system_prompt,
            messages,
            tool_api,
            agent_name,
            iteration,
            temperature,
            max_tokens,
        )
        totals.add(response)

        if response.stop_reason == "end_turn" or not response.tool_calls:
            return _build_result(agent_name, response, history, iteration + 1, totals)

        messages.append(response.to_assistant_message())
        tool_results = _execute_tool_calls(response, handlers, history)
        messages.append({"role": "user", "content": tool_results})

    raise AgentLoopExhaustedError(f"Agent {agent_name} exhausted {max_iterations} iterations")


class _TokenTotals:
    """Accumulates token counts and cost across iterations."""

    __slots__ = ("input_tokens", "output_tokens", "cost_usd")

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

    def add(self, response: LLMToolResponse) -> None:
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.cost_usd += response.cost_usd


async def _call_llm(
    client: UnifiedLLMClient,
    system_prompt: str,
    messages: list[dict[str, str | list[object]]],
    tools: list[dict[str, object]],
    agent_name: str,
    iteration: int,
    temperature: float,
    max_tokens: int,
) -> LLMToolResponse:
    """Send one iteration to the LLM."""
    return await client.complete_with_tools(
        system=system_prompt,
        messages=messages,
        tools=tools,
        agent_name=f"{agent_name}_iter_{iteration}",
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _build_result(
    agent_name: str,
    response: LLMToolResponse,
    history: list[ToolCallRecord],
    iterations: int,
    totals: _TokenTotals,
) -> AgentResult:
    """Build the final AgentResult and log completion."""
    logger.info(
        "Agent %s finished in %d iterations (tokens=%d+%d, cost=$%.4f)",
        agent_name,
        iterations,
        totals.input_tokens,
        totals.output_tokens,
        totals.cost_usd,
    )
    return AgentResult(
        agent_name=agent_name,
        final_content=response.text_content,
        tool_call_history=history,
        total_iterations=iterations,
        total_input_tokens=totals.input_tokens,
        total_output_tokens=totals.output_tokens,
        total_cost_usd=totals.cost_usd,
    )


def _execute_tool_calls(
    response: LLMToolResponse,
    handlers: dict[str, ToolHandler],
    history: list[ToolCallRecord],
) -> list[dict[str, str]]:
    """Execute tool calls and return results for the API."""
    results: list[dict[str, str]] = []

    for call in response.tool_calls:
        output = _run_single_tool(call.name, call.input, handlers)
        history.append(
            ToolCallRecord(
                tool_name=call.name,
                input_params=call.input,
                output_preview=output[:200],
            )
        )
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": output,
            }
        )
        logger.debug("Tool %s → %d chars", call.name, len(output))

    return results


def _run_single_tool(
    name: str,
    tool_input: dict[str, str | int | float | bool],
    handlers: dict[str, ToolHandler],
) -> str:
    """Execute a single tool handler, returning JSON output."""
    handler = handlers.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return handler(tool_input)
    except Exception as e:
        logger.warning("Tool %s failed: %s", name, e)
        return json.dumps({"error": str(e)})


def _trace_agent_start(agent_name: str, max_iter: int, num_tools: int) -> None:
    """Record agent start in LangFuse."""
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.trace(
            name=agent_name,
            metadata={
                "max_iterations": max_iter,
                "num_tools": num_tools,
                "type": "agent_loop",
            },
        )
    except Exception as e:
        logger.debug("LangFuse agent trace failed (non-fatal): %s", e)
