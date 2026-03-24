"""Tool registry — defines and collects tools for agent loops.

Each tool has a name, description, JSON Schema input, and a synchronous
handler function. Tools are registered per-agent and bound to data
via closures at creation time.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

# Tool handlers receive parsed input dict, return JSON string
type ToolHandler = Callable[[dict[str, str | int | float | bool]], str]


@dataclass(frozen=True)
class ToolDefinition:
    """A tool the agent can invoke via the Anthropic tool-use API."""

    name: str
    description: str
    input_schema: dict[str, object]


def schema_from_model(model_class: type[BaseModel]) -> dict[str, object]:
    """Convert a Pydantic model to a JSON Schema dict for Anthropic's tools API."""
    schema = model_class.model_json_schema()
    return {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }


EMPTY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {},
    "required": [],
}


@dataclass
class ToolRegistry:
    """Collects tool definitions and handlers for an agent."""

    _definitions: list[ToolDefinition] = field(default_factory=list)
    _handlers: dict[str, ToolHandler] = field(default_factory=dict)

    def register(
        self,
        name: str,
        description: str,
        handler: ToolHandler,
        input_schema: dict[str, object] | None = None,
    ) -> None:
        """Register a tool with its handler."""
        schema = input_schema or EMPTY_SCHEMA
        defn = ToolDefinition(name=name, description=description, input_schema=schema)
        self._definitions.append(defn)
        self._handlers[name] = handler

    @property
    def definitions(self) -> list[ToolDefinition]:
        """All registered tool definitions."""
        return list(self._definitions)

    @property
    def handlers(self) -> dict[str, ToolHandler]:
        """Map of tool name to handler function."""
        return dict(self._handlers)

    def to_api_format(self) -> list[dict[str, object]]:
        """Convert all tools to Anthropic API format."""
        return [
            {
                "name": d.name,
                "description": d.description,
                "input_schema": d.input_schema,
            }
            for d in self._definitions
        ]
