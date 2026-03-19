"""Estimate API spend based on dataset size and model pricing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig


# Approximate metric calls per GEPA auto mode
GEPA_METRIC_CALLS = {
    "light": 560,
    "medium": 2000,
    "heavy": 5000,
}


def estimate_cost(config: MigrationConfig) -> list[str]:
    """Estimate total API cost for a migration run.

    Returns list of cost-related warnings.
    """
    import litellm

    warnings: list[str] = []

    try:
        target_info = litellm.get_model_info(config.target_model)
        input_cost = target_info.get("input_cost_per_token", 0) or 0
        output_cost = target_info.get("output_cost_per_token", 0) or 0
    except Exception:
        warnings.append("Could not estimate cost — model pricing not available.")
        return warnings

    metric_calls = GEPA_METRIC_CALLS.get(config.gepa_auto, 560)

    # Rough estimate: avg 500 input tokens + 500 output tokens per call
    avg_input_tokens = 500
    avg_output_tokens = 500
    estimated_cost = metric_calls * (
        avg_input_tokens * input_cost + avg_output_tokens * output_cost
    )

    if estimated_cost > 0:
        warnings.append(
            f"Estimated API cost: ${estimated_cost:.2f} "
            f"(~{metric_calls} calls, {config.gepa_auto} mode)"
        )

    if estimated_cost > 20:
        warnings.append(
            f"Estimated cost exceeds $20. Consider using --auto light or reducing dataset size."
        )

    return warnings
