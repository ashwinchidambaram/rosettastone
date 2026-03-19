"""Run all pre-flight checks and return a PreflightReport."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.pipeline import PreflightReport


def run_all_checks(config: MigrationConfig) -> PreflightReport:
    from rosettastone.core.pipeline import PreflightReport
    from rosettastone.preflight.capabilities import check_capabilities
    from rosettastone.preflight.cost_estimator import estimate_cost
    from rosettastone.preflight.token_budget import check_token_budget

    warnings: list[str] = []
    blockers: list[str] = []

    # Capability checks
    cap_warnings, cap_blockers = check_capabilities(config)
    warnings.extend(cap_warnings)
    blockers.extend(cap_blockers)

    # Token budget checks
    tok_warnings, tok_blockers = check_token_budget(config)
    warnings.extend(tok_warnings)
    blockers.extend(tok_blockers)

    # Cost estimation
    cost_warnings = estimate_cost(config)
    warnings.extend(cost_warnings)

    return PreflightReport(warnings=warnings, blockers=blockers)
