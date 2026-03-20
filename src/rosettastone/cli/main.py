from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(name="rosettastone", help="Automated LLM model migration")
console = Console()


@app.command()
def migrate(
    data: Path = typer.Option(..., "--data", "-d", help="Path to JSONL file"),
    source: str = typer.Option(..., "--from", help="Source model (e.g. openai/gpt-4o)"),
    target: str = typer.Option(..., "--to", help="Target model (e.g. anthropic/claude-sonnet-4)"),
    output: Path = typer.Option("./migration_output", "--output", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost without running"),
    auto_mode: str = typer.Option("light", "--auto", help="GEPA intensity: light/medium/heavy"),
    # Phase 2 flags
    local_only: bool = typer.Option(
        False, "--local-only", help="Skip external API calls for evaluation"
    ),
    redis_url: str | None = typer.Option(  # noqa: UP007
        None, "--redis-url", help="Redis URL for cache ingestion"
    ),
    judge_model: str = typer.Option(
        "openai/gpt-4o", "--judge-model", help="Model for LLM-as-judge evaluation"
    ),
    no_pii_scan: bool = typer.Option(False, "--no-pii-scan", help="Disable PII scanning"),
    no_prompt_audit: bool = typer.Option(False, "--no-prompt-audit", help="Disable prompt audit"),
    optimizer: str = typer.Option("gepa", "--optimizer", help="Optimizer: gepa or mipro"),
    mipro_auto: str | None = typer.Option(  # noqa: UP007
        None, "--mipro-auto", help="MIPROv2 auto preset: light/medium/heavy"
    ),
) -> None:
    """Run a full migration: preflight -> optimize -> evaluate -> report."""
    from rosettastone.config import MigrationConfig
    from rosettastone.core.migrator import Migrator

    config = MigrationConfig(
        source_model=source,
        target_model=target,
        data_path=data,
        output_dir=output,
        dry_run=dry_run,
        gepa_auto=auto_mode,  # type: ignore[arg-type]
        local_only=local_only,
        redis_url=redis_url,
        judge_model=judge_model,
        pii_scan=not no_pii_scan,
        prompt_audit=not no_prompt_audit,
        mipro_auto=mipro_auto if optimizer == "mipro" else None,  # type: ignore[arg-type]
    )
    migrator = Migrator(config)
    result = migrator.run()

    # Phase 2: Use Rich display for output
    from rosettastone.cli.display import MigrationDisplay

    display = MigrationDisplay(console=console)

    console.print("\n[bold green]Migration complete![/bold green]")
    console.print(f"Confidence score: {result.confidence_score:.0%}")
    console.print(f"Baseline score:   {result.baseline_score:.0%}")
    console.print(f"Improvement:      +{result.improvement:.0%}")

    if result.recommendation:
        display.show_recommendation(result.recommendation, result.recommendation_reasoning or "")

    if result.per_type_scores:
        display.show_summary_table(result.validation_results, result.per_type_scores)

    if result.safety_warnings:
        display.show_safety_warnings(result.safety_warnings)

    display.show_cost_summary(
        result.config.get("costs", {}) if isinstance(result.config, dict) else {}
    )

    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in result.warnings:
            console.print(f"  - {w}")


@app.command()
def preflight(
    data: Path = typer.Option(..., "--data", "-d"),
    source: str = typer.Option(..., "--from"),
    target: str = typer.Option(..., "--to"),
) -> None:
    """Run pre-flight checks only."""
    from rosettastone.config import MigrationConfig
    from rosettastone.core.migrator import Migrator

    config = MigrationConfig(
        source_model=source,
        target_model=target,
        data_path=data,
        dry_run=True,
    )
    migrator = Migrator(config)
    result = migrator.run()

    console.print("[bold]Pre-flight report:[/bold]")
    for w in result.warnings:
        console.print(f"  - {w}")


@app.command()
def evaluate(
    data: Path = typer.Option(..., "--data", "-d"),
    source: str = typer.Option(..., "--from"),
    target: str = typer.Option(..., "--to"),
) -> None:
    """Run evaluation only (no optimization)."""
    from rosettastone.config import MigrationConfig
    from rosettastone.core.pipeline import evaluate_baseline, load_and_split_data

    config = MigrationConfig(
        source_model=source,
        target_model=target,
        data_path=data,
        skip_preflight=True,
    )
    _, _, test = load_and_split_data(config)
    results = evaluate_baseline(test, config)

    wins = sum(1 for r in results if r.is_win)
    console.print(f"Win rate: {wins}/{len(results)} ({wins / max(len(results), 1):.0%})")


if __name__ == "__main__":
    app()
