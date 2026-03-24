# LLM Cost Auditor

> **Prototype / Portfolio Project** — This is a technical demonstration of production-grade AI agent engineering, not a production-ready product. The included sample data is synthetically generated to showcase the system's capabilities. Waste detection thresholds, feature complexity classifications, and model pricing are illustrative defaults that would need calibration for a real deployment.

An autonomous multi-agent system that audits LLM API usage logs and delivers actionable cost optimization recommendations with estimated monthly savings.

Feed it your LangFuse exports, OpenAI usage CSVs, or any generic log format. Four AI agents autonomously investigate your data, evaluate quality risks, simulate optimizations, and produce a full audit report.

## Features

- **Agentic tool-use architecture** — three autonomous agents (analysis, optimization, report) using Anthropic's native tool-use API with ReAct-style reasoning loops
- **Typed tool system** — deterministic computation functions exposed as agent-callable tools with Pydantic schemas
- **Multi-format ingestion** — parses LangFuse JSON exports, OpenAI usage CSVs, and generic CSV formats
- **Deterministic waste detection** — four algorithmic detectors (bloated prompts, wrong model, excessive retries, cacheable duplicates)
- **Routing simulation** — simulates multi-model routing using feature-based complexity classification
- **3-layer quality evaluation** — proxy signals from cross-model data, opt-in LLM-as-Judge blind A/B eval, and concrete per-feature eval plans with criteria and cost estimates
- **Prioritized recommendations** — agents generate actionable optimization advice with estimated savings
- **Deterministic fallbacks** — every agent falls back to pure computation if the LLM call fails
- **Full observability** — every agent iteration and tool call traced in LangFuse
- **Self-contained deployment** — ships as Docker Compose (app + LangFuse + PostgreSQL)

## How It Works

```
Input: LLM usage logs (JSON / CSV)
         │
         ▼
┌──────────────────┐
│  1. Ingestion    │  Auto-detect format, parse, normalize to typed schema
└────────┬─────────┘  (pure computation)
         │
         ▼
┌──────────────────┐  ┌─────────────────────────────────────────┐
│  2. Analysis     │──│  Agent loop: 3-5 iterations             │
│     Agent        │  │  Tools: dataset summary, cost breakdowns │
└────────┬─────────┘  │  waste detectors, routing simulation    │
         │            └─────────────────────────────────────────┘
         ▼
┌──────────────────┐  ┌─────────────────────────────────────────┐
│  3. Optimization │──│  Agent loop: 3-5 iterations             │
│     Agent        │  │  Tools: analysis summary, routing sim,   │
└────────┬─────────┘  │  model switch estimator, feature detail │
         │            └─────────────────────────────────────────┘
         ▼
┌──────────────────┐  Layer 1: Proxy signals (cross-model output comparison)
│  4. Quality      │  Layer 2: LLM-as-Judge blind A/B eval (opt-in, --eval)
│     Evaluation   │  Layer 3: Per-feature eval plans with criteria + costs
└────────┬─────────┘  (deterministic + optional LLM calls)
         │
         ▼
┌──────────────────┐  ┌─────────────────────────────────────────┐
│  5. Report       │──│  Agent loop: 2-3 iterations             │
│     Agent        │  │  Tools: cost overview, recommendations,  │
└──────────────────┘  │  waste patterns, routing sim, quality   │
                      └─────────────────────────────────────────┘
Output: Markdown audit report with methodology
```

**Architecture details:**
- Each agent runs a **ReAct loop** using Anthropic's native tool-use API — no framework
- Tools are deterministic computation functions (cost aggregation, waste detection, routing simulation) that agents call to investigate the data
- The agent decides **which tools to call, in what order, and how to interpret results**
- Numbers always come from tool computations (deterministic) — the LLM adds reasoning and prioritization
- Every agent has a **deterministic fallback** — if the LLM fails, pure computation produces the result
- Total: **~9 LLM iterations** across 3 agents + quality evaluation (~$0.15-0.30 per audit without `--eval`)

### Agent Tools

| Agent | Tools | What it investigates |
|-------|-------|---------------------|
| **Analysis** | `get_dataset_summary`, `compute_cost_by_model`, `compute_cost_by_feature`, `detect_bloated_prompts`, `detect_wrong_model`, `detect_excessive_retries`, `detect_cacheable_duplicates`, `simulate_routing` | Cost patterns, waste sources, routing potential |
| **Optimization** | `get_analysis_summary`, `simulate_routing`, `estimate_model_switch_savings`, `get_feature_detail` | Savings scenarios, feature-specific optimizations |
| **Quality Eval** | *(deterministic + optional LLM-as-Judge)* | Proxy signals, blind A/B quality comparison, per-feature eval plans |
| **Report** | `get_cost_overview`, `get_top_recommendations`, `get_waste_patterns`, `get_routing_simulation`, `get_top_cost_drivers`, `get_quality_evaluation` | Data points for executive summary |

### Waste Detection Algorithms

| Pattern | Method | What it catches |
|---------|--------|-----------------|
| **Bloated prompts** | Per-feature median analysis, flags >2x median (min 8K tokens) | Oversized prompts on simple tasks |
| **Wrong model** | Feature complexity classification + token threshold | Frontier models where Haiku suffices |
| **Excessive retries** | Error-then-success chains within 60s window | Wasted spend on failed calls |
| **Cacheable duplicates** | Same (feature, model, tokens within 5%) in 2-min window | Repeated identical requests |

### Routing Simulation

Features are classified by complexity:

| Tier | Features | Routing rule |
|------|----------|--------------|
| SIMPLE | email_drafting, data_extraction | Always route to light model |
| MODERATE | customer_support_agent, doc_qa | Route to light model if input < 4K tokens |
| COMPLEX | code_review | Keep on frontier model |

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An Anthropic API key

### Installation

```bash
git clone https://github.com/hhamon/llm-cost-auditor.git
cd llm-cost-auditor

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Run an audit

```bash
# Against sample data
uv run python -m src.main --input sample-data/example-logs.json --output reports/

# With LLM-as-Judge quality evaluation (requires prompt content in logs, costs extra)
uv run python -m src.main --input logs-with-content.json --output reports/ --eval

# Against your own logs
uv run python -m src.main --input /path/to/your/logs.json --output reports/
```

### Run with Docker Compose

```bash
# Full stack: app + LangFuse + PostgreSQL
docker compose up -d

# View LangFuse dashboard at http://localhost:3000
```

## Supported Log Formats

### LangFuse JSON export

```json
{
  "data": [
    {
      "model": "claude-sonnet-4-6",
      "promptTokens": 4500,
      "completionTokens": 1200,
      "calculatedTotalCost": 0.0315,
      "startTime": "2026-03-15T10:30:00Z",
      "traceName": "customer_support_agent",
      "level": "DEFAULT"
    }
  ]
}
```

### OpenAI usage CSV

```csv
timestamp,model,prompt_tokens,completion_tokens,cost,api_key_name
2026-03-15T10:30:00Z,gpt-4o,4500,1200,0.0233,customer_support
```

### Generic CSV

```csv
timestamp,model,input_tokens,output_tokens,cost_usd,feature,status
2026-03-15T10:30:00Z,gpt-4o,4500,1200,0.0233,customer_support,success
```

## Configuration

Environment variables (set in `.env`):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | API key for agent LLM calls |
| `LANGFUSE_PUBLIC_KEY` | No | — | LangFuse public key (tracing) |
| `LANGFUSE_SECRET_KEY` | No | — | LangFuse secret key (tracing) |
| `LANGFUSE_HOST` | No | `http://localhost:3000` | LangFuse server URL |
| `DEFAULT_MODEL` | No | `claude-sonnet-4-6` | Model for agent LLM calls |
| `LOG_LEVEL` | No | `INFO` | Logging level |

The app runs fine without LangFuse credentials — tracing is optional.

## Project Structure

```
llm-cost-auditor/
├── src/
│   ├── main.py               # CLI entry point + pipeline orchestrator
│   ├── agents/
│   │   ├── base.py            # Agent loop framework (ReAct + tool use)
│   │   ├── ingestion.py       # Log parsing and normalization
│   │   ├── analysis.py        # Analysis agent + deterministic fallback
│   │   ├── optimization.py    # Optimization agent + deterministic fallback
│   │   ├── routing_sim.py     # Routing simulation (deterministic)
│   │   ├── quality_eval.py    # Quality evaluation (3 layers)
│   │   └── report.py          # Report agent + single-call fallback
│   ├── tools/
│   │   ├── registry.py        # Tool registry + schema generation
│   │   ├── analysis_tools.py  # Tools for the analysis agent
│   │   ├── optimization_tools.py # Tools for the optimization agent
│   │   └── report_tools.py    # Tools for the report agent
│   ├── models/
│   │   ├── log_entry.py       # Normalized log entry schema
│   │   ├── analysis.py        # Analysis results schema
│   │   ├── recommendation.py  # Recommendation schema
│   │   ├── report.py          # Final report schema
│   │   ├── features.py        # Feature complexity + audit config
│   │   ├── quality.py         # Quality evaluation models (3 layers)
│   │   └── agent_result.py    # Agent execution result schema
│   ├── parsers/
│   │   ├── langfuse.py        # LangFuse JSON export parser
│   │   ├── openai_csv.py      # OpenAI usage CSV parser
│   │   └── generic.py         # Generic CSV parser
│   ├── report/
│   │   └── markdown.py        # Markdown report renderer
│   └── utils/
│       ├── llm_client.py      # Unified LLM client (text + tool use)
│       ├── langfuse_setup.py  # LangFuse initialization
│       ├── dataset_summary.py # Dataset summarization for agent context
│       ├── model_utils.py     # Shared model classification
│       ├── retry.py           # Exponential backoff retry
│       ├── json_extract.py    # JSON extraction from LLM responses
│       └── config.py          # Environment config loader
├── tests/
│   ├── test_agent_loop.py     # Agent loop tests (mock LLM)
│   ├── test_tools.py          # Tool handler tests (no LLM)
│   ├── test_analysis.py       # Waste detection algorithm tests
│   ├── test_parsers.py        # Parser tests against sample data
│   ├── test_models.py         # Pydantic model validation tests
│   └── test_quality_eval.py   # Quality evaluation tests (49 tests)
├── scripts/
│   └── generate_sample_data.py # Deterministic sample data generator
├── sample-data/
│   ├── example-logs.json       # 5,084 LangFuse-style entries
│   ├── example-openai.csv      # 2,000 OpenAI-style entries
│   ├── example-generic.csv     # 1,000 generic entries
│   └── example-report.md       # Example audit output
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Run tests (119 tests, no API key needed)
uv run pytest tests/ -v

# Lint and format
uv run ruff check src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/

# Generate fresh sample data
python3 scripts/generate_sample_data.py
```

## Example Output

See [`sample-data/example-report.md`](sample-data/example-report.md) for a full audit report.

**Summary from a 5,084-entry synthetic dataset:**
- Total cost: $240 over 29 days ($248/month projected)
- 4 agents investigated autonomously (~9 tool-use iterations)
- 4 waste patterns identified with confidence levels and methodology
- 3-layer quality evaluation (proxy signals, judge eval, eval plans)
- Prioritized recommendations with implementation steps
- Potential savings: ~$78/month (32% reduction)

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Runtime | Python 3.12+ | Industry standard for ML/AI |
| LLM clients | Anthropic SDK (tool use) | Direct SDK calls, native tool-use API |
| Data validation | Pydantic v2 | Type safety, self-documenting schemas |
| Observability | LangFuse (self-hosted) | Open-source, full trace capture |
| Containerization | Docker Compose | Portable, no cloud vendor lock-in |
| Testing | pytest | Standard, fast |
| Linting | ruff | Fast Python linter + formatter |
| Package manager | uv | Fast, modern Python package manager |

## Design Decisions

1. **Agentic tool-use architecture** — agents use Anthropic's native tool-use API in ReAct loops. No LangChain, no framework. The agent loop is a single async function. Debuggable, testable, easy to hand off.

2. **Tools are deterministic, agents add reasoning** — waste detectors, cost aggregations, and routing simulations are pure computation. Agents decide which tools to call and interpret results. Numbers are always reproducible.

3. **Deterministic fallbacks** — every agent has a fallback path. If the LLM fails, the system still produces a valid report using pure computation. Production reliability without sacrificing agentic capability.

4. **Pydantic as the backbone** — every piece of data has a schema. Tool inputs/outputs are validated. The pipeline is self-documenting and catches integration errors at boundaries.

5. **LangFuse from day one** — every agent iteration and tool call is traced. Cost, latency, token usage captured automatically.

6. **Docker Compose as delivery format** — the entire system (app + LangFuse) starts with one command.

## License

MIT
