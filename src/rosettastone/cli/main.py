from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(name="rosettastone", help="Automated LLM model migration")
console = Console()


@app.command()
def migrate(
    data: Path | None = typer.Option(  # noqa: UP007
        None, "--data", "-d", help="Path to data file (JSONL/CSV/OTel)"
    ),
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
    # Phase 4 flags
    adapter: str = typer.Option(
        "jsonl", "--adapter", help="Data adapter: jsonl, redis, csv, braintrust, langsmith, otel"
    ),
    pii_engine: str = typer.Option("regex", "--pii-engine", help="PII scanner: regex or presidio"),
    cluster_prompts: bool = typer.Option(
        False, "--cluster-prompts", help="Cluster prompts before optimization"
    ),
    # Adapter-specific options
    csv_delimiter: str | None = typer.Option(  # noqa: UP007
        None, "--csv-delimiter", help="CSV delimiter"
    ),
    csv_prompt_col: str | None = typer.Option(  # noqa: UP007
        None, "--csv-prompt-col", help="CSV prompt column name"
    ),
    csv_response_col: str | None = typer.Option(  # noqa: UP007
        None, "--csv-response-col", help="CSV response column name"
    ),
    braintrust_project: str | None = typer.Option(  # noqa: UP007
        None, "--braintrust-project", help="Braintrust project name"
    ),
    langsmith_project: str | None = typer.Option(  # noqa: UP007
        None, "--langsmith-project", help="LangSmith project name"
    ),
    langsmith_start: str | None = typer.Option(  # noqa: UP007
        None, "--langsmith-start", help="LangSmith start date (ISO-8601)"
    ),
    langsmith_end: str | None = typer.Option(  # noqa: UP007
        None, "--langsmith-end", help="LangSmith end date (ISO-8601)"
    ),
    otel_path: Path | None = typer.Option(  # noqa: UP007
        None, "--otel-path", help="Path to OTel JSON export"
    ),
    improvement_objectives: str | None = typer.Option(  # noqa: UP007
        None,
        "--improvement-objectives",
        help='JSON array of objectives, e.g. \'[{"description": "be more concise"}]\'',
    ),
    pipeline: Annotated[
        Path | None,
        typer.Option(
            "--pipeline",
            help="Path to pipeline YAML config for multi-step migration.",
        ),
    ] = None,
) -> None:
    """Run a full migration: preflight -> optimize -> evaluate -> report."""
    import json as json_mod

    from rosettastone.config import MigrationConfig
    from rosettastone.core.migrator import Migrator

    # Handle pipeline mode
    if pipeline:
        from rosettastone.optimize.pipeline_config import load_pipeline_config

        pipeline_config = load_pipeline_config(pipeline)
        num_modules = len(pipeline_config.modules)
        typer.echo(f"Pipeline mode: {pipeline_config.name} ({num_modules} modules)")
        typer.echo(
            "Pipeline optimization is handled server-side via POST /api/v1/pipelines/migrate"
        )
        return

    # Parse improvement objectives from JSON string
    parsed_objectives = None
    if improvement_objectives:
        try:
            parsed_objectives = json_mod.loads(improvement_objectives)
        except json_mod.JSONDecodeError:
            console.print("[red]Error: --improvement-objectives must be valid JSON[/red]")
            raise typer.Exit(code=1)

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
        # Phase 4
        adapter=adapter,  # type: ignore[arg-type]
        pii_engine=pii_engine,  # type: ignore[arg-type]
        cluster_prompts=cluster_prompts,
        csv_delimiter=csv_delimiter,
        csv_prompt_column=csv_prompt_col,
        csv_response_column=csv_response_col,
        braintrust_project=braintrust_project,
        langsmith_project=langsmith_project,
        langsmith_start_date=langsmith_start,
        langsmith_end_date=langsmith_end,
        otel_path=otel_path,
        improvement_objectives=parsed_objectives,
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
    output_format: str = typer.Option(
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
        if output_format == "json":
            formatted = format_ci_json(result)
        elif output_format == "pr-comment":
            if not source or not target:
                console.print("[red]Error: pr-comment format requires --from and --to flags[/red]")
                raise typer.Exit(code=1)
            formatted = format_pr_comment(result, source, target)
        elif output_format == "quality-diff":
            formatted = format_quality_diff(result)
        else:
            console.print(
                f"[red]Unknown format: {output_format}. "
                f"Use: json, pr-comment, or quality-diff[/red]"
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
