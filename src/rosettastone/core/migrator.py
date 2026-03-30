from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.types import MigrationResult


class MigrationBlockedError(Exception):
    def __init__(self, preflight_report: object) -> None:
        self.preflight_report = preflight_report
        super().__init__(f"Migration blocked by pre-flight checks: {preflight_report}")


class Migrator:
    def __init__(self, config: MigrationConfig) -> None:
        self.config = config

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

        # Step 1: Ingest
        t0 = time.time()
        train, val, test = load_and_split_data(self.config)
        ctx.timing["ingest"] = time.time() - t0

        # Step 1.5: PII scan on ingested data
        t0 = time.time()
        run_pii_scan(train + val + test, ctx, self.config)
        ctx.timing["pii_scan"] = time.time() - t0

        # Step 2: Baseline
        t0 = time.time()
        baseline = evaluate_baseline(test, self.config)
        ctx.timing["baseline_eval"] = time.time() - t0

        # Step 3: Optimize
        t0 = time.time()
        optimized_prompt = optimize_prompt(train, val, self.config)
        ctx.timing["optimize"] = time.time() - t0

        # Step 3.5: Safety checks on optimized prompt
        t0 = time.time()
        run_pii_scan_text(optimized_prompt, ctx, self.config)
        run_prompt_audit(optimized_prompt, train, ctx, self.config)
        ctx.timing["prompt_safety"] = time.time() - t0

        # Step 4: Validate
        t0 = time.time()
        validation = evaluate_optimized(test, optimized_prompt, self.config)
        ctx.timing["validation_eval"] = time.time() - t0

        # Step 4.5: Recommendation
        t0 = time.time()
        rec, reasoning, per_type = make_recommendation(validation, ctx, self.config)
        ctx.recommendation = (rec, reasoning, per_type)
        ctx.timing["recommendation"] = time.time() - t0

        # Step 5: Report
        duration = time.time() - start
        result = build_result(self.config, optimized_prompt, baseline, validation, duration, ctx)
        generate_report(result, self.config.output_dir)

        return result
