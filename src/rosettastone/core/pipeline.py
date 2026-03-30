"""Pipeline step definitions — each function is a thin wrapper delegating to subsystems."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.context import PipelineContext
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


def _build_adapter(config: MigrationConfig) -> Any:
    """Build the appropriate data adapter based on config.adapter."""
    from rosettastone.config import AdapterChoice

    adapter_type = config.adapter

    # Legacy compat: if redis_url is set and adapter is still default, use Redis
    if config.redis_url and adapter_type == AdapterChoice.JSONL:
        adapter_type = AdapterChoice.REDIS

    if adapter_type == AdapterChoice.REDIS:
        from rosettastone.ingest.redis_adapter import RedisAdapter

        if not config.redis_url:
            raise ValueError("redis_url is required for the Redis adapter")
        return RedisAdapter(config.redis_url, config.source_model)

    if adapter_type == AdapterChoice.CSV:
        from rosettastone.ingest.csv_adapter import CSVAdapter

        if not config.data_path:
            raise ValueError("data_path is required for the CSV adapter")
        kwargs: dict[str, Any] = {}
        if config.csv_delimiter:
            kwargs["delimiter"] = config.csv_delimiter
        if config.csv_prompt_column or config.csv_response_column:
            from rosettastone.ingest.csv_adapter import CSVColumnMapping

            kwargs["column_mapping"] = CSVColumnMapping(
                prompt_col=config.csv_prompt_column or "prompt",
                response_col=config.csv_response_column or "response",
            )
        return CSVAdapter(config.data_path, **kwargs)

    if adapter_type == AdapterChoice.BRAINTRUST:
        from rosettastone.ingest.braintrust_adapter import BraintrustAdapter

        return BraintrustAdapter(
            project_name=config.braintrust_project or "",
            source_model=config.source_model,
        )

    if adapter_type == AdapterChoice.LANGSMITH:
        from rosettastone.ingest.langsmith_adapter import LangSmithAdapter

        if not config.langsmith_project:
            raise ValueError("langsmith_project is required for the LangSmith adapter")
        kwargs_ls: dict[str, Any] = {"project_name": config.langsmith_project}
        if config.langsmith_start_date:
            kwargs_ls["start_date"] = config.langsmith_start_date
        if config.langsmith_end_date:
            kwargs_ls["end_date"] = config.langsmith_end_date
        return LangSmithAdapter(**kwargs_ls)

    if adapter_type == AdapterChoice.OTEL:
        from rosettastone.ingest.otel_adapter import OTelAdapter

        otel_path = config.otel_path or config.data_path
        if not otel_path:
            raise ValueError("otel_path or data_path is required for the OTel adapter")
        return OTelAdapter(export_path=otel_path, source_model=config.source_model)

    # Default: JSONL
    from rosettastone.ingest.jsonl import JSONLAdapter

    if not config.data_path:
        raise ValueError("data_path is required for the JSONL adapter")
    return JSONLAdapter(config.data_path)


def load_and_split_data(
    config: MigrationConfig,
) -> tuple[list[PromptPair], list[PromptPair], list[PromptPair]]:
    adapter = _build_adapter(config)

    from rosettastone.ingest.splitter import split_data

    pairs = adapter.load()
    return split_data(pairs, config.train_split, config.val_split)


def optimize_prompt(train: list[PromptPair], val: list[PromptPair], config: MigrationConfig) -> str:
    if config.mipro_auto is not None:
        from rosettastone.optimize.mipro import MIPROv2Optimizer

        optimizer = MIPROv2Optimizer()
    else:
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


def run_pii_scan(pairs: list[PromptPair], ctx: PipelineContext, config: MigrationConfig) -> None:
    """Scan prompt pairs for PII and add warnings to context."""
    if not config.pii_scan:
        return

    from rosettastone.config import PIIEngine
    from rosettastone.core.context import SafetySeverity, SafetyWarning

    if config.pii_engine == PIIEngine.PRESIDIO:
        from rosettastone.safety.presidio_engine import scan_pairs_presidio

        pii_warnings = scan_pairs_presidio(pairs)
    else:
        from rosettastone.safety.pii_scanner import scan_pairs

        pii_warnings = scan_pairs(pairs)
    for pw in pii_warnings:
        ctx.safety_warnings.append(
            SafetyWarning(
                warning_type="pii",
                severity=SafetySeverity(pw.severity),
                message=f"PII detected: {pw.pii_type} (count: {pw.count})",
                details={"pair_index": pw.pair_index, "pii_type": pw.pii_type},
            )
        )


def run_prompt_audit(
    optimized_prompt: str,
    train: list[PromptPair],
    ctx: PipelineContext,
    config: MigrationConfig,
) -> None:
    """Audit optimized prompt for training data leakage."""
    if not config.prompt_audit:
        return

    from rosettastone.core.context import SafetySeverity, SafetyWarning
    from rosettastone.safety.prompt_auditor import audit_prompt

    findings = audit_prompt(optimized_prompt, train)
    for finding in findings:
        ctx.safety_warnings.append(
            SafetyWarning(
                warning_type="prompt_audit",
                severity=SafetySeverity.MEDIUM,
                message=(
                    f"Training data leakage: {len(finding.substring)} chars "
                    f"from {finding.source_count} source(s)"
                ),
                details={"source_count": finding.source_count},
            )
        )


def run_pii_scan_text(
    text: str, ctx: PipelineContext, config: MigrationConfig | None = None
) -> None:
    """Scan optimized prompt text for HIGH-severity PII (blocker)."""
    from rosettastone.config import PIIEngine
    from rosettastone.core.context import SafetySeverity, SafetyWarning

    use_presidio = config and config.pii_engine == PIIEngine.PRESIDIO
    if use_presidio:
        from rosettastone.safety.presidio_engine import scan_text_presidio

        findings = scan_text_presidio(text)
    else:
        from rosettastone.safety.pii_scanner import scan_text

        findings = scan_text(text)
    for pii_type, severity in findings:
        if severity == "HIGH":
            ctx.safety_warnings.append(
                SafetyWarning(
                    warning_type="pii_in_prompt",
                    severity=SafetySeverity.HIGH,
                    message=f"HIGH-severity PII ({pii_type}) found in optimized prompt",
                    details={"pii_type": pii_type},
                )
            )


def make_recommendation(
    validation: list[EvalResult],
    ctx: PipelineContext,
    config: MigrationConfig,
) -> tuple[str, str, dict[str, Any]]:
    """Run recommendation engine and return (recommendation, reasoning, per_type_scores)."""
    from rosettastone.decision.recommendation import make_recommendation as _make_rec

    safety_dicts = [
        {"severity": str(w.severity), "message": w.message} for w in ctx.safety_warnings
    ]
    rec_result = _make_rec(validation, safety_dicts, config.win_thresholds)

    import dataclasses

    per_type_scores = {k: dataclasses.asdict(v) for k, v in rec_result.per_type_details.items()}

    return str(rec_result.recommendation), rec_result.reasoning, per_type_scores


def build_result(
    config: MigrationConfig,
    optimized_prompt: str,
    baseline: list[EvalResult],
    validation: list[EvalResult],
    duration: float,
    ctx: PipelineContext | None = None,
) -> MigrationResult:
    from rosettastone.core.types import MigrationResult

    baseline_wins = sum(1 for r in baseline if r.is_win)
    validation_wins = sum(1 for r in validation if r.is_win)
    total = len(validation) if validation else 1

    baseline_score = baseline_wins / max(len(baseline), 1)
    confidence_score = validation_wins / total

    warnings: list[str] = []
    if ctx:
        warnings.extend(ctx.warnings)
    if not validation:
        warnings.append(
            "All validation pairs were skipped — check target model configuration and API keys."
        )
    if not baseline:
        warnings.append(
            "All baseline pairs were skipped — check target model configuration and API keys."
        )

    # Safety warnings as dicts for serialization
    safety_warnings: list[Any] = []
    if ctx:
        safety_warnings = [
            {
                "warning_type": w.warning_type,
                "severity": str(w.severity),
                "message": w.message,
            }
            for w in ctx.safety_warnings
        ]

    # Recommendation and per-type scores
    recommendation = None
    recommendation_reasoning = None
    per_type_scores: dict[str, Any] = {}
    if ctx and ctx.recommendation is not None:
        recommendation, recommendation_reasoning, per_type_scores = ctx.recommendation

    total_cost = sum(ctx.costs.values()) if ctx else 0.0

    return MigrationResult(
        config=config.model_dump(mode="json"),
        optimized_prompt=optimized_prompt,
        baseline_results=baseline,
        validation_results=validation,
        confidence_score=confidence_score,
        baseline_score=baseline_score,
        improvement=confidence_score - baseline_score,
        cost_usd=total_cost,
        duration_seconds=duration,
        warnings=warnings,
        safety_warnings=safety_warnings,
        recommendation=recommendation,
        recommendation_reasoning=recommendation_reasoning,
        per_type_scores=per_type_scores,
    )


def generate_report(result: MigrationResult, output_dir: Path) -> None:
    from rosettastone.report.markdown import generate_markdown_report

    output_dir.mkdir(parents=True, exist_ok=True)
    generate_markdown_report(result, output_dir)
