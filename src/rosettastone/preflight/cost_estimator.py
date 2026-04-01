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

# MIPROv2 call estimates (zero-shot mode — fewer than GEPA)
MIPRO_METRIC_CALLS = {
    "light": 300,
    "medium": 1200,
    "heavy": 3000,
}


def estimate_cost(config: MigrationConfig) -> tuple[list[str], float]:
    """Estimate total API cost for a migration run.

    Returns tuple of (cost-related warnings list, estimated_cost_usd float).
    """
    import litellm

    warnings: list[str] = []
    estimated_cost_usd = 0.0

    try:
        target_info = litellm.get_model_info(config.target_model)
        input_cost = target_info.get("input_cost_per_token", 0) or 0
        output_cost = target_info.get("output_cost_per_token", 0) or 0
    except Exception:
        warnings.append("Could not estimate cost — model pricing not available.")
        return warnings, 0.0

    # Optimizer cost
    if config.mipro_auto is not None:
        metric_calls = MIPRO_METRIC_CALLS.get(config.mipro_auto, 300)
        opt_label = f"MIPROv2 {config.mipro_auto}"
    else:
        metric_calls = GEPA_METRIC_CALLS.get(config.gepa_auto, 560)
        opt_label = f"GEPA {config.gepa_auto}"

    # Rough estimate: avg 500 input tokens + 500 output tokens per call
    avg_input_tokens = 500
    avg_output_tokens = 500
    estimated_cost = metric_calls * (
        avg_input_tokens * input_cost + avg_output_tokens * output_cost
    )

    # Judge cost: 4N calls (2 bidirectional × 2 phases) × avg tokens
    judge_cost = 0.0
    if not config.local_only:
        try:
            judge_info = litellm.get_model_info(config.judge_model)
            judge_input = judge_info.get("input_cost_per_token", 0) or 0
            judge_output = judge_info.get("output_cost_per_token", 0) or 0
            # Estimate 4 judge calls per pair, 2 phases
            judge_calls = 4 * config.recommended_pairs
            judge_cost = judge_calls * (
                avg_input_tokens * judge_input + avg_output_tokens * judge_output
            )
        except Exception:
            pass

    total_cost = estimated_cost + judge_cost
    estimated_cost_usd = total_cost

    if total_cost > 0:
        cost_parts = [f"optimizer: ${estimated_cost:.2f} (~{metric_calls} calls, {opt_label})"]
        if judge_cost > 0:
            cost_parts.append(f"LLM judge: ${judge_cost:.2f}")
        warnings.append(f"Estimated API cost: ${total_cost:.2f} ({', '.join(cost_parts)})")

    if total_cost > 20:
        warnings.append(
            "Estimated cost exceeds $20. Consider using --auto light or reducing dataset size."
        )

    # Redis dependency check
    if config.redis_url:
        try:
            import redis  # noqa: F401
        except ImportError:
            warnings.append(
                "redis-url specified but 'redis' package not installed. Install with: uv add redis"
            )

    return warnings, estimated_cost_usd
