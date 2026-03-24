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
from src.agents.report import run_report_agent
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


async def run_audit(input_path: Path, output_dir: Path) -> None:
    """Execute the full agentic audit pipeline."""
    config = load_config()
    setup_logging(config.log_level)
    logger = logging.getLogger("auditor")

    # Initialize services
    init_langfuse(config)
    client = UnifiedLLMClient(config)

    logger.info("Starting audit: %s", input_path)

    # Pipeline: Ingest → Analysis Agent → Optimization Agent → Report Agent
    logger.info("[1/4] Ingesting logs...")
    dataset = ingest(input_path)

    logger.info("[2/4] Analysis agent investigating cost patterns...")
    analysis = await run_analysis_agent(dataset, client)

    logger.info("[3/4] Optimization agent generating recommendations...")
    optimization = await run_optimization_agent(dataset, analysis, client)

    logger.info("[4/4] Report agent writing executive summary...")
    report = await run_report_agent(dataset, analysis, optimization, client)

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
def main(input_path: str, output_dir: str) -> None:
    """LLM Cost Auditor — Audit your LLM usage and find savings."""
    asyncio.run(run_audit(Path(input_path), Path(output_dir)))


if __name__ == "__main__":
    main()
