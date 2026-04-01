from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.types import MigrationResult


class MigrationBlockedError(Exception):
    def __init__(self, preflight_report: object) -> None:
        self.preflight_report = preflight_report
        super().__init__(f"Migration blocked by pre-flight checks: {preflight_report}")


class Migrator:
    def __init__(
        self,
        config: MigrationConfig,
        progress_callback: Callable[[str, float, float], None] | None = None,
    ) -> None:
        self.config = config
        self._progress_callback = progress_callback

    def _emit(self, stage: str, stage_pct: float, overall_pct: float) -> None:
        """Invoke progress_callback if set; swallow any exception it raises."""
        if self._progress_callback is not None:
            try:
                self._progress_callback(stage, stage_pct, overall_pct)
            except Exception:
                pass

    def run(self) -> MigrationResult:
        from rosettastone.core.context import PipelineContext
        from rosettastone.core.pipeline import (
            build_result,
            evaluate_baseline,
            evaluate_optimized,
            generate_report,
            load_and_split_data,
            make_recommendation,
            optimize_prompt,
            run_pii_scan,
            run_pii_scan_text,
            run_preflight,
            run_prompt_audit,
        )
        from rosettastone.core.types import MigrationResult

        # Respect dry_run regardless of skip_preflight
        if self.config.dry_run and self.config.skip_preflight:
            return MigrationResult(
                config=self.config.model_dump(),
                optimized_prompt="",
                baseline_results=[],
                validation_results=[],
                confidence_score=0.0,
                baseline_score=0.0,
                improvement=0.0,
                cost_usd=0.0,
                duration_seconds=0.0,
                warnings=[],
                recommendation="NO_GO",
                recommendation_reasoning="Dry run — no migration performed.",
            )

        start = time.time()
        ctx = PipelineContext()

        # Step 0: Pre-flight
        if not self.config.skip_preflight:
            preflight_report = run_preflight(self.config)
            ctx.warnings.extend(preflight_report.warnings)
            if preflight_report.has_blockers:
                raise MigrationBlockedError(preflight_report)
            if self.config.dry_run:
                return preflight_report.as_dry_run_result()  # type: ignore[no-any-return]
            self._emit("preflight", 1.0, 0.12)

        # Step 1: Ingest
        t0 = time.time()
        train, val, test = load_and_split_data(self.config, ctx)
        ctx.timing["ingest"] = time.time() - t0
        self._emit("data_load", 1.0, 0.25)

        # Step 1.5: PII scan on ingested data
        t0 = time.time()
        run_pii_scan(train + val + test, ctx, self.config)
        ctx.timing["pii_scan"] = time.time() - t0
        self._emit("pii_scan", 1.0, 0.37)

        # Step 2: Baseline
        t0 = time.time()
        baseline = evaluate_baseline(test, self.config)
        ctx.timing["baseline_eval"] = time.time() - t0
        self._emit("baseline_eval", 1.0, 0.50)

        # Step 3: Optimize
        t0 = time.time()
        optimized_prompt = optimize_prompt(train, val, self.config)
        ctx.timing["optimize"] = time.time() - t0
        self._emit("optimize", 1.0, 0.75)

        # Step 3.5: Safety checks on optimized prompt
        t0 = time.time()
        run_pii_scan_text(optimized_prompt, ctx, self.config)
        self._emit("pii_scan_text", 1.0, 0.80)
        run_prompt_audit(optimized_prompt, train, ctx, self.config)
        ctx.timing["prompt_safety"] = time.time() - t0
        self._emit("prompt_audit", 1.0, 0.85)

        # Step 4: Validate
        t0 = time.time()
        validation = evaluate_optimized(test, optimized_prompt, self.config)
        ctx.timing["validation_eval"] = time.time() - t0
        self._emit("validation_eval", 1.0, 0.95)

        # Step 4.5: Recommendation
        t0 = time.time()
        rec, reasoning, per_type = make_recommendation(validation, ctx, self.config)
        ctx.recommendation = (rec, reasoning, per_type)
        ctx.timing["recommendation"] = time.time() - t0
        self._emit("recommendation", 1.0, 0.98)

        # Step 5: Report
        duration = time.time() - start
        result = build_result(self.config, optimized_prompt, baseline, validation, duration, ctx)
        generate_report(result, self.config.output_dir)
        self._emit("report", 1.0, 1.0)

        return result
