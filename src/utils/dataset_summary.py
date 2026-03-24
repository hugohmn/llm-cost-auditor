"""Dataset summarization — compact stats for LLM agent context.

Produces a ~500-800 token summary of a LogDataset so the agent
understands the data without seeing 5K+ raw entries.
"""

from collections import Counter

from src.models.log_entry import LogDataset


def summarize_dataset(dataset: LogDataset) -> dict[str, object]:
    """Create a compact statistical summary of a LogDataset."""
    model_counts: Counter[str] = Counter()
    feature_counts: Counter[str] = Counter()
    model_costs: dict[str, float] = {}
    feature_costs: dict[str, float] = {}
    status_counts: Counter[str] = Counter()

    for e in dataset.entries:
        model_counts[e.model] += 1
        model_costs[e.model] = model_costs.get(e.model, 0.0) + e.cost_usd
        feat = e.feature or "unknown"
        feature_counts[feat] += 1
        feature_costs[feat] = feature_costs.get(feat, 0.0) + e.cost_usd
        status_counts[e.status] += 1

    total_cost = dataset.total_cost_usd or 1.0
    days = _compute_days(dataset)

    return {
        "total_entries": dataset.total_entries,
        "total_cost_usd": round(dataset.total_cost_usd, 2),
        "total_tokens": dataset.total_tokens,
        "monthly_projected_cost_usd": round((dataset.total_cost_usd / max(days, 1)) * 30, 2),
        "date_range": {
            "start": dataset.date_range_start.isoformat() if dataset.date_range_start else None,
            "end": dataset.date_range_end.isoformat() if dataset.date_range_end else None,
            "days": days,
        },
        "models": [
            {
                "model": m,
                "calls": model_counts[m],
                "cost_usd": round(model_costs[m], 2),
                "pct_of_cost": round(model_costs[m] / total_cost, 3),
            }
            for m in sorted(model_costs, key=model_costs.get, reverse=True)  # type: ignore[arg-type]
        ],
        "features": [
            {
                "feature": f,
                "calls": feature_counts[f],
                "cost_usd": round(feature_costs[f], 2),
                "pct_of_cost": round(feature_costs[f] / total_cost, 3),
            }
            for f in sorted(feature_costs, key=feature_costs.get, reverse=True)  # type: ignore[arg-type]
        ],
        "status_distribution": dict(status_counts),
    }


def _compute_days(dataset: LogDataset) -> int:
    """Compute date range in days."""
    if dataset.date_range_start and dataset.date_range_end:
        return max((dataset.date_range_end - dataset.date_range_start).days, 1)
    return 1
