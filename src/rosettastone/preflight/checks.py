"""Run all pre-flight checks and return a PreflightReport."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.pipeline import PreflightReport


def run_all_checks(config: MigrationConfig) -> PreflightReport:
    from rosettastone.core.deprecations import check_model_deprecation
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
    cost_warnings, estimated_cost_usd = estimate_cost(config)
    warnings.extend(cost_warnings)

    # Deprecation checks
    for model_id in [config.source_model, config.target_model]:
        dep = check_model_deprecation(model_id)
        if not dep:
            continue

        if dep["already_retired"]:
            # Target model already retired is a blocker
            if model_id == config.target_model:
                blockers.append(
                    f"Target model '{model_id}' was already retired on "
                    f"{dep['retirement_date']}. Use '{dep['replacement']}' instead."
                )
            # Source model already retired is a warning
            else:
                warnings.append(
                    f"Source model '{model_id}' was already retired on "
                    f"{dep['retirement_date']}. Consider migrating to '{dep['replacement']}'."
                )
        elif dep["days_until_retirement"] <= 30:
            # Within 30 days is critical
            warnings.append(
                f"Model '{model_id}' will retire in {dep['days_until_retirement']} days "
                f"(on {dep['retirement_date']}). Recommended replacement: '{dep['replacement']}'."
            )
        else:
            # Beyond 30 days is a warning
            warnings.append(
                f"Model '{model_id}' will retire in {dep['days_until_retirement']} days "
                f"(on {dep['retirement_date']}). Recommended replacement: '{dep['replacement']}'."
            )

    return PreflightReport(
        warnings=warnings, blockers=blockers, estimated_cost_usd=estimated_cost_usd
    )
