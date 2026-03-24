"""LLM Cost Auditor — CLI entry point and pipeline orchestrator.

Usage:
    python -m src.main --input logs.json --output reports/
"""

import asyncio
import logging
from pathlib import Path

import click

from src.agents.analysis import run_analysis_agent
from src.agents.ingestion import ingest
from src.agents.optimization import run_optimization_agent
from src.agents.quality_eval import run_quality_evaluation
from src.agents.report import run_report_agent
from src.models.features import AuditConfig
from src.report.markdown import render_markdown
from src.utils.config import load_config
from src.utils.langfuse_setup import flush_langfuse, init_langfuse
from src.utils.llm_client import UnifiedLLMClient


def setup_logging(level: str) -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def run_audit(
    input_path: Path,
    output_dir: Path,
    run_eval: bool = False,
) -> None:
    """Execute the full agentic audit pipeline."""
    config = load_config()
    setup_logging(config.log_level)
    logger = logging.getLogger("auditor")

    # Initialize services
    init_langfuse(config)
    client = UnifiedLLMClient(config)

    # Build audit config, enabling judge eval if --eval flag is passed
    audit_config = AuditConfig(enable_judge_eval=run_eval) if run_eval else None

    logger.info("Starting audit: %s", input_path)

    # Pipeline: Ingest → Analysis → Optimization → Quality Eval → Report
    logger.info("[1/5] Ingesting logs...")
    dataset = ingest(input_path)

    logger.info("[2/5] Analysis agent investigating cost patterns...")
    analysis = await run_analysis_agent(dataset, client, config=audit_config)

    logger.info("[3/5] Optimization agent generating recommendations...")
    optimization = await run_optimization_agent(dataset, analysis, client)

    logger.info("[4/5] Evaluating quality signals...")
    quality = await run_quality_evaluation(
        dataset, analysis, optimization, client, config=audit_config
    )

    logger.info("[5/5] Report agent writing executive summary...")
    report = await run_report_agent(dataset, analysis, optimization, client, quality=quality)

    # Render and save
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"audit-{input_path.stem}.md"
    markdown = render_markdown(report)
    report_path.write_text(markdown)

    logger.info("Audit complete! Report saved to %s", report_path)
    logger.info("Headline: %s", report.headline_savings)

    # Flush LangFuse traces
    flush_langfuse()


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to log file",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    default="reports/",
    type=click.Path(),
    help="Output directory",
)
@click.option(
    "--eval",
    "run_eval",
    is_flag=True,
    default=False,
    help="Enable LLM-as-Judge quality evaluation (costs extra API calls)",
)
def main(input_path: str, output_dir: str, run_eval: bool) -> None:
    """LLM Cost Auditor — Audit your LLM usage and find savings."""
    asyncio.run(run_audit(Path(input_path), Path(output_dir), run_eval=run_eval))


if __name__ == "__main__":
    main()
