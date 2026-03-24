"""Analysis tools — deterministic functions callable by the analysis agent.

Each tool is bound to a LogDataset and AuditConfig via closure at registration time.
The agent never sees raw entries — only computed results.
"""

import json
import logging

from pydantic import BaseModel

from src.agents.analysis import (
    compute_cost_by_feature,
    compute_cost_by_model,
    detect_bloated_prompts,
    detect_cacheable_duplicates,
    detect_excessive_retries,
    detect_wrong_model,
)
from src.agents.routing_sim import simulate_routing
from src.models.features import AuditConfig
from src.models.log_entry import LogDataset
from src.tools.registry import ToolRegistry
from src.utils.dataset_summary import summarize_dataset

logger = logging.getLogger(__name__)


def _serialize_model(obj: BaseModel | list[BaseModel]) -> str:
    """Serialize a Pydantic model or list of models to JSON."""
    if isinstance(obj, list):
        return json.dumps([o.model_dump() for o in obj], indent=2)
    return obj.model_dump_json(indent=2)


def _serialize_optional(obj: BaseModel | None) -> str:
    """Serialize a Pydantic model or return 'null' if None."""
    if obj is None:
        return json.dumps({"result": None, "message": "No pattern detected"})
    return obj.model_dump_json(indent=2)


def _register_stat_tools(registry: ToolRegistry, dataset: LogDataset) -> None:
    """Register dataset summary and cost breakdown tools."""
    registry.register(
        name="get_dataset_summary",
        description=(
            "Get a high-level summary of the log dataset: total entries, cost, "
            "tokens, date range, models, features, error rates. "
            "Call this first to understand the data."
        ),
        handler=lambda _: json.dumps(summarize_dataset(dataset), indent=2),
    )

    registry.register(
        name="compute_cost_by_model",
        description=(
            "Break down costs by LLM model. Returns each model's total calls, "
            "tokens, cost, error rate, and percentage of total spend. "
            "Sorted by cost descending."
        ),
        handler=lambda _: _serialize_model(compute_cost_by_model(dataset)),
    )

    registry.register(
        name="compute_cost_by_feature",
        description=(
            "Break down costs by application feature. Returns each feature's "
            "call count, cost, primary model, and average tokens per call. "
            "Sorted by cost descending."
        ),
        handler=lambda _: _serialize_model(compute_cost_by_feature(dataset)),
    )


def _register_waste_tools(
    registry: ToolRegistry,
    dataset: LogDataset,
    config: AuditConfig,
) -> None:
    """Register waste-detection tools."""
    registry.register(
        name="detect_bloated_prompts",
        description=(
            "Find prompts with excessive input tokens (>2x median for the feature, "
            "minimum 8000 tokens). Returns affected call count and estimated waste "
            "in USD, or null if no bloat detected."
        ),
        handler=lambda _: _serialize_optional(detect_bloated_prompts(dataset, config)),
    )

    registry.register(
        name="detect_wrong_model",
        description=(
            "Find calls using expensive frontier models for tasks a cheaper model "
            "could handle. SIMPLE features always flagged, MODERATE features flagged "
            "when input < 4000 tokens. Returns affected count and savings potential."
        ),
        handler=lambda _: _serialize_optional(detect_wrong_model(dataset, config)),
    )

    registry.register(
        name="detect_excessive_retries",
        description=(
            "Find error-then-success retry chains within 60 seconds. Returns count "
            "of wasted retry calls and their cost, or null if none found."
        ),
        handler=lambda _: _serialize_optional(detect_excessive_retries(dataset, config)),
    )

    registry.register(
        name="detect_cacheable_duplicates",
        description=(
            "Find near-duplicate calls (same feature, model, similar token count) "
            "within 2-minute windows that could be cached. Returns count and savings."
        ),
        handler=lambda _: _serialize_optional(detect_cacheable_duplicates(dataset, config)),
    )


def _register_routing_tools(
    registry: ToolRegistry,
    dataset: LogDataset,
    config: AuditConfig,
) -> None:
    """Register routing simulation tool."""
    registry.register(
        name="simulate_routing",
        description=(
            "Simulate routing simple/moderate requests to a light model (Haiku) "
            "while keeping complex ones on a frontier model (Sonnet). Returns "
            "current vs optimized cost, savings percentage, and observed error rates."
        ),
        handler=lambda _: _serialize_model(simulate_routing(dataset, config)),
    )


def build_analysis_registry(
    dataset: LogDataset,
    config: AuditConfig | None = None,
) -> ToolRegistry:
    """Create tool registry with all analysis tools bound to a dataset."""
    if config is None:
        config = AuditConfig()
    registry = ToolRegistry()
    _register_stat_tools(registry, dataset)
    _register_waste_tools(registry, dataset, config)
    _register_routing_tools(registry, dataset, config)
    return registry
