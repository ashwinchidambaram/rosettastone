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
        from rosettastone.core.pipeline import (
            build_result,
            evaluate_baseline,
            evaluate_optimized,
            generate_report,
            load_and_split_data,
            optimize_prompt,
            run_preflight,
        )

        start = time.time()

        # Step 0: Pre-flight
        if not self.config.skip_preflight:
            preflight_report = run_preflight(self.config)
            if preflight_report.has_blockers:
                raise MigrationBlockedError(preflight_report)
            if self.config.dry_run:
                return preflight_report.as_dry_run_result()

        # Step 1: Ingest
        train, val, test = load_and_split_data(self.config)

        # Step 2: Baseline
        baseline = evaluate_baseline(test, self.config)

        # Step 3: Optimize
        optimized_prompt = optimize_prompt(train, val, self.config)

        # Step 4: Validate
        validation = evaluate_optimized(test, optimized_prompt, self.config)

        # Step 5: Report
        duration = time.time() - start
        result = build_result(self.config, optimized_prompt, baseline, validation, duration)
        generate_report(result, self.config.output_dir)

        return result
