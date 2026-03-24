"""Generate realistic sample data for the LLM Cost Auditor.

Produces three files in sample-data/:
  - example-logs.json  (LangFuse-style, ~5,000 entries)
  - example-openai.csv (OpenAI-style, ~2,000 entries)
  - example-generic.csv (generic CSV, ~1,000 entries)

All outputs are deterministic (seeded random).
Costs are computed from token counts x published model pricing.
"""

import csv
import json
import random
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

SEED = 42
rng = random.Random(SEED)

OUT_DIR = Path(__file__).resolve().parent.parent / "sample-data"

END_DATE = datetime(2026, 3, 23, 23, 59, 0, tzinfo=UTC)
START_DATE = END_DATE - timedelta(days=30)

# ── Model pricing (per 1M tokens: input, output) ───────────────────────
MODELS = {
    "claude-sonnet-4-6": {
        "weight": 0.60,
        "input_price": 3.0,
        "output_price": 15.0,
    },
    "gpt-4o": {
        "weight": 0.25,
        "input_price": 2.50,
        "output_price": 10.0,
    },
    "claude-haiku-4-5-20251001": {
        "weight": 0.15,
        "input_price": 0.80,
        "output_price": 4.0,
    },
}

ALL_PRICING = {
    **{k: (v["input_price"], v["output_price"]) for k, v in MODELS.items()},
    "gpt-4o-mini": (0.15, 0.60),
}

# ── Features with realistic token profiles ──────────────────────────────
FEATURES = {
    "customer_support_agent": {
        "weight": 0.30,
        "input_range": (2_000, 6_000),
        "output_range": (500, 2_000),
    },
    "doc_qa": {
        "weight": 0.25,
        "input_range": (4_000, 25_000),
        "output_range": (1_000, 4_000),
    },
    "code_review": {
        "weight": 0.20,
        "input_range": (6_000, 40_000),
        "output_range": (2_000, 8_000),
    },
    "email_drafting": {
        "weight": 0.15,
        "input_range": (500, 2_500),
        "output_range": (200, 1_000),
    },
    "data_extraction": {
        "weight": 0.10,
        "input_range": (1_500, 5_000),
        "output_range": (300, 1_500),
    },
}


def pick_weighted(options: dict[str, dict]) -> str:
    keys = list(options.keys())
    weights = [options[k]["weight"] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def random_timestamp() -> datetime:
    delta = (END_DATE - START_DATE).total_seconds()
    return START_DATE + timedelta(seconds=rng.random() * delta)


def make_id() -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute cost from tokens x published pricing."""
    p = ALL_PRICING.get(model, (3.0, 15.0))
    return (input_tokens / 1_000_000) * p[0] + (output_tokens / 1_000_000) * p[1]


def _build_record(
    *,
    model: str,
    feature: str,
    input_tokens: int,
    output_tokens: int,
    ts: datetime,
    latency_ms: float,
    status: str,
) -> dict:
    cost = calc_cost(model, input_tokens, output_tokens)
    trace_id = make_id()
    end_time = ts + timedelta(milliseconds=latency_ms)
    return {
        "id": make_id(),
        "traceId": trace_id,
        "observationId": make_id(),
        "name": f"{feature}_generation",
        "traceName": feature,
        "model": model,
        "startTime": ts.isoformat(),
        "endTime": end_time.isoformat(),
        "latency": round(latency_ms, 1),
        "promptTokens": input_tokens,
        "completionTokens": output_tokens,
        "totalTokens": input_tokens + output_tokens,
        "calculatedTotalCost": round(cost, 6),
        "level": "ERROR" if status == "error" else "DEFAULT",
        "userId": f"user-{rng.randint(1, 50)}",
        "tags": [feature],
    }


# ── Entry generators ────────────────────────────────────────────────────


def gen_normal() -> dict:
    model = pick_weighted(MODELS)
    feature = pick_weighted(FEATURES)
    fp = FEATURES[feature]
    return _build_record(
        model=model,
        feature=feature,
        input_tokens=rng.randint(*fp["input_range"]),
        output_tokens=rng.randint(*fp["output_range"]),
        ts=random_timestamp(),
        latency_ms=rng.uniform(800, 12000),
        status="success",
    )


def gen_bloated() -> dict:
    """Bloated prompt: >8K input tokens for simple tasks."""
    model = rng.choice(["claude-sonnet-4-6", "gpt-4o"])
    feature = rng.choice(
        [
            "email_drafting",
            "customer_support_agent",
            "data_extraction",
        ]
    )
    return _build_record(
        model=model,
        feature=feature,
        input_tokens=rng.randint(8_000, 25_000),
        output_tokens=rng.randint(200, 1_000),
        ts=random_timestamp(),
        latency_ms=rng.uniform(3000, 15000),
        status="success",
    )


def gen_cacheable_cluster(base_ts: datetime) -> list[dict]:
    """Near-duplicate requests within 2 minutes."""
    model = pick_weighted(MODELS)
    feature = rng.choice(["doc_qa", "data_extraction"])
    fp = FEATURES[feature]
    input_tokens = rng.randint(*fp["input_range"])
    output_tokens = rng.randint(*fp["output_range"])

    entries = []
    for _ in range(rng.randint(2, 4)):
        ts = base_ts + timedelta(seconds=rng.randint(1, 90))
        entries.append(
            _build_record(
                model=model,
                feature=feature,
                input_tokens=input_tokens + rng.randint(-30, 30),
                output_tokens=output_tokens + rng.randint(-10, 10),
                ts=ts,
                latency_ms=rng.uniform(800, 6000),
                status="success",
            )
        )
    return entries


def gen_error_retries(base_ts: datetime) -> list[dict]:
    """Error followed by retry within 60 seconds."""
    model = pick_weighted(MODELS)
    feature = pick_weighted(FEATURES)
    fp = FEATURES[feature]
    input_tokens = rng.randint(*fp["input_range"])
    output_tokens = rng.randint(*fp["output_range"])

    entries = []
    num_errors = rng.randint(1, 2)
    for i in range(num_errors + 1):
        ts = base_ts + timedelta(seconds=i * rng.randint(2, 10))
        is_error = i < num_errors
        entries.append(
            _build_record(
                model=model,
                feature=feature,
                input_tokens=input_tokens,
                output_tokens=0 if is_error else output_tokens,
                ts=ts,
                latency_ms=(rng.uniform(100, 500) if is_error else rng.uniform(800, 6000)),
                status="error" if is_error else "success",
            )
        )
    return entries


def gen_wrong_model() -> dict:
    """Frontier model for trivially simple queries."""
    model = rng.choice(["claude-sonnet-4-6", "gpt-4o"])
    feature = rng.choice(["email_drafting", "customer_support_agent"])
    return _build_record(
        model=model,
        feature=feature,
        input_tokens=rng.randint(300, 1_500),
        output_tokens=rng.randint(100, 500),
        ts=random_timestamp(),
        latency_ms=rng.uniform(300, 2000),
        status="success",
    )


# ── Main generation ─────────────────────────────────────────────────────


def generate_langfuse_data() -> list[dict]:
    entries: list[dict] = []

    for _ in range(3200):
        entries.append(gen_normal())
    for _ in range(800):
        entries.append(gen_bloated())
    for _ in range(150):
        entries.extend(gen_cacheable_cluster(random_timestamp()))
    for _ in range(120):
        entries.extend(gen_error_retries(random_timestamp()))
    for _ in range(300):
        entries.append(gen_wrong_model())

    entries.sort(key=lambda e: e["startTime"])
    return entries


def generate_openai_csv(count: int = 2000) -> list[dict]:
    rows = []
    for _ in range(count):
        model = rng.choice(["gpt-4o", "gpt-4o-mini", "gpt-4o"])
        feature = pick_weighted(FEATURES)
        fp = FEATURES[feature]
        input_tokens = rng.randint(*fp["input_range"])
        output_tokens = rng.randint(*fp["output_range"])
        cost = calc_cost(model, input_tokens, output_tokens)
        rows.append(
            {
                "timestamp": random_timestamp().isoformat(),
                "model": model,
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "cost": round(cost, 6),
                "api_key_name": feature,
            }
        )
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def generate_generic_csv(count: int = 1000) -> list[dict]:
    rows = []
    all_models = list(MODELS.keys()) + ["gpt-4o-mini"]
    for _ in range(count):
        model = rng.choice(all_models)
        feature = pick_weighted(FEATURES)
        fp = FEATURES[feature]
        input_tokens = rng.randint(*fp["input_range"])
        output_tokens = rng.randint(*fp["output_range"])
        cost = calc_cost(model, input_tokens, output_tokens)
        rows.append(
            {
                "timestamp": random_timestamp().isoformat(),
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(cost, 6),
                "latency_ms": round(rng.uniform(500, 15000), 1),
                "feature": feature,
                "status": rng.choice(["success"] * 19 + ["error"]),
            }
        )
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def write_langfuse_json(entries: list[dict]) -> Path:
    path = OUT_DIR / "example-logs.json"
    with open(path, "w") as f:
        json.dump({"data": entries}, f, indent=2)
    return path


def write_openai_csv(rows: list[dict]) -> Path:
    path = OUT_DIR / "example-openai.csv"
    fields = [
        "timestamp",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "cost",
        "api_key_name",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def write_generic_csv(rows: list[dict]) -> Path:
    path = OUT_DIR / "example-generic.csv"
    fields = [
        "timestamp",
        "model",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_ms",
        "feature",
        "status",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lf = generate_langfuse_data()
    p = write_langfuse_json(lf)
    cost = sum(e["calculatedTotalCost"] for e in lf)
    errors = sum(1 for e in lf if e["level"] == "ERROR")
    bloated = sum(
        1
        for e in lf
        if e["traceName"] in {"email_drafting", "customer_support_agent", "data_extraction"}
        and e["promptTokens"] > 8000
    )
    print(f"LangFuse: {len(lf)} entries, ${cost:,.2f} total -> {p}")
    print(f"  Errors: {errors}, Bloated prompts: {bloated}")

    oai = generate_openai_csv()
    p = write_openai_csv(oai)
    print(f"OpenAI:   {len(oai)} entries, ${sum(r['cost'] for r in oai):,.2f} total -> {p}")

    gen = generate_generic_csv()
    p = write_generic_csv(gen)
    print(f"Generic:  {len(gen)} entries, ${sum(r['cost_usd'] for r in gen):,.2f} total -> {p}")


if __name__ == "__main__":
    main()
