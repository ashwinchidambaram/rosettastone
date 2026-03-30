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

    if result.cost_usd > 0:
        display.show_cost_summary({"total": result.cost_usd})

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


@app.command()
def batch(
    manifest: Path = typer.Option(..., "--manifest", "-m", help="Path to batch YAML manifest"),
    output: Path = typer.Option("./batch_output", "--output", "-o", help="Base output directory"),
) -> None:
    """Run multiple migrations from a YAML manifest."""
    from rosettastone.batch import format_batch_summary, load_manifest, run_batch

    console.print(f"[bold]Loading manifest:[/bold] {manifest}")
    batch_manifest = load_manifest(manifest)
    console.print(f"Found {len(batch_manifest.migrations)} migration(s)\n")

    results = run_batch(batch_manifest, output)

    summary = format_batch_summary(results)
    console.print(summary)


@app.command(name="ci-report")
def ci_report(
    result_path: Path = typer.Option(
        ..., "--result", "-r", help="Path to migration result JSON file"
    ),
    format: str = typer.Option(
        "json", "--format", "-f", help="Output format: json, pr-comment, or quality-diff"
    ),
    source: str | None = typer.Option(  # noqa: UP007
        None, "--from", help="Source model (required for pr-comment format)"
    ),
    target: str | None = typer.Option(  # noqa: UP007
        None, "--to", help="Target model (required for pr-comment format)"
    ),
    output: Path | None = typer.Option(  # noqa: UP007
        None, "--output", "-o", help="Write output to file (default: stdout)"
    ),
) -> None:
    """Generate CI/CD-friendly output from a migration result."""
    import json

    from rosettastone.core.types import MigrationResult

    # Load result from JSON file
    try:
        raw = json.loads(result_path.read_text())
        result = MigrationResult(**raw)
    except FileNotFoundError:
        console.print(f"[red]Error: Result file not found: {result_path}[/red]")
        raise typer.Exit(code=1)
    except json.JSONDecodeError:
        console.print(f"[red]Error: Invalid JSON in {result_path}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error loading result: {e}[/red]")
        raise typer.Exit(code=1)

    from rosettastone.cli.ci_output import (
        format_ci_json,
        format_pr_comment,
        format_quality_diff,
    )

    try:
        if format == "json":
            formatted = format_ci_json(result)
        elif format == "pr-comment":
            if not source or not target:
                console.print(
                    "[red]Error: pr-comment format requires --from and --to flags[/red]"
                )
                raise typer.Exit(code=1)
            formatted = format_pr_comment(result, source, target)
        elif format == "quality-diff":
            formatted = format_quality_diff(result)
        else:
            console.print(
                "[red]Unknown format: "
                f"{format}. Use: json, pr-comment, or quality-diff[/red]"
            )
            raise typer.Exit(code=1)

        if output:
            output.write_text(formatted)
            console.print(f"[green]Written to {output}[/green]")
        else:
            console.print(formatted, highlight=False)
    except Exception as e:
        console.print(f"[red]Error formatting output: {e}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
