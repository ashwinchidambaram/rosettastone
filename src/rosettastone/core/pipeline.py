"""Pipeline step definitions — each function is a thin wrapper delegating to subsystems."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.types import EvalResult, MigrationResult, PromptPair


class PreflightReport:
    def __init__(self, warnings: list[str], blockers: list[str]) -> None:
        self.warnings = warnings
        self.blockers = blockers

    @property
    def has_blockers(self) -> bool:
        return len(self.blockers) > 0

    def as_dry_run_result(self) -> Any:
        from rosettastone.core.types import MigrationResult

        return MigrationResult(
            config={},
            optimized_prompt="",
            baseline_results=[],
            validation_results=[],
            confidence_score=0.0,
            baseline_score=0.0,
            improvement=0.0,
            cost_usd=0.0,
            duration_seconds=0.0,
            warnings=self.warnings + ["DRY RUN — no migration performed"],
        )


def run_preflight(config: MigrationConfig) -> PreflightReport:
    from rosettastone.preflight.checks import run_all_checks

    return run_all_checks(config)


def load_and_split_data(
    config: MigrationConfig,
) -> tuple[list[PromptPair], list[PromptPair], list[PromptPair]]:
    from rosettastone.ingest.jsonl import JSONLAdapter
    from rosettastone.ingest.splitter import split_data

    adapter = JSONLAdapter(config.data_path)
    pairs = adapter.load()
    return split_data(pairs, config.train_split, config.val_split)


def optimize_prompt(train: list[PromptPair], val: list[PromptPair], config: MigrationConfig) -> str:
    from rosettastone.optimize.gepa import GEPAOptimizer

    optimizer = GEPAOptimizer()
    return optimizer.optimize(train, val, config)


def evaluate_baseline(test: list[PromptPair], config: MigrationConfig) -> list[EvalResult]:
    from rosettastone.evaluate.composite import CompositeEvaluator

    evaluator = CompositeEvaluator(config)
    return evaluator.evaluate(test)


def evaluate_optimized(
    test: list[PromptPair], optimized_prompt: str, config: MigrationConfig
) -> list[EvalResult]:
    from rosettastone.evaluate.composite import CompositeEvaluator

    evaluator = CompositeEvaluator(config)
    return evaluator.evaluate(test, optimized_prompt=optimized_prompt)


def build_result(
    config: MigrationConfig,
    optimized_prompt: str,
    baseline: list[EvalResult],
    validation: list[EvalResult],
    duration: float,
) -> MigrationResult:
    from rosettastone.core.types import MigrationResult

    baseline_wins = sum(1 for r in baseline if r.is_win)
    validation_wins = sum(1 for r in validation if r.is_win)
    total = len(validation) if validation else 1

    baseline_score = baseline_wins / max(len(baseline), 1)
    confidence_score = validation_wins / total

    warnings: list[str] = []
    if not validation:
        warnings.append(
            "All validation pairs were skipped — check target model configuration and API keys."
        )
    if not baseline:
        warnings.append(
            "All baseline pairs were skipped — check target model configuration and API keys."
        )

    return MigrationResult(
        config=config.model_dump(mode="json"),
        optimized_prompt=optimized_prompt,
        baseline_results=baseline,
        validation_results=validation,
        confidence_score=confidence_score,
        baseline_score=baseline_score,
        improvement=confidence_score - baseline_score,
        cost_usd=0.0,  # TODO: track actual cost via LiteLLM callbacks
        duration_seconds=duration,
        warnings=warnings,
    )


def generate_report(result: MigrationResult, output_dir: Path) -> None:
    from rosettastone.report.markdown import generate_markdown_report

    output_dir.mkdir(parents=True, exist_ok=True)
    generate_markdown_report(result, output_dir)
