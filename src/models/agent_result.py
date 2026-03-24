"""Agent execution result models — output of agentic tool-use loops."""

from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    """Record of a single tool invocation within an agent loop."""

    tool_name: str
    input_params: dict[str, str | int | float | bool] = Field(default_factory=dict)
    output_preview: str = Field(
        default="",
        description="First 200 chars of tool output",
    )


class AgentResult(BaseModel):
    """Final result from an agent loop execution."""

    agent_name: str
    final_content: str
    tool_call_history: list[ToolCallRecord] = Field(default_factory=list)
    total_iterations: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
