"""Tests for preflight.cost_estimator.estimate_cost.

This module proves that cost estimation is accurate for each GEPA auto mode,
that the correct warning thresholds ($0 and $20) fire when expected, and that
missing pricing information is handled gracefully.

Mock strategy: cost_estimator.py uses `import litellm` inside the function body
(lazy import). We patch `litellm.get_model_info` directly on the litellm package.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from rosettastone.config import MigrationConfig
from rosettastone.preflight.cost_estimator import GEPA_METRIC_CALLS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_MODEL_INFO_WITH_PRICING = {
    "max_input_tokens": 128000,
    "max_output_tokens": 4096,
    "input_cost_per_token": 0.000005,
    "output_cost_per_token": 0.000015,
    "supports_function_calling": True,
    "supports_vision": True,
    "supports_response_schema": True,
}


def _config(gepa_auto: str = "light") -> MigrationConfig:
    return MigrationConfig(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        data_path=Path("/fake/data.jsonl"),
        gepa_auto=gepa_auto,  # type: ignore[arg-type]
    )


def _expected_cost(gepa_auto: str, input_cost: float, output_cost: float) -> float:
    """Replicate the formula from cost_estimator.py for assertion purposes."""
    metric_calls = GEPA_METRIC_CALLS[gepa_auto]
    avg_input_tokens = 500
    avg_output_tokens = 500
    return metric_calls * (avg_input_tokens * input_cost + avg_output_tokens * output_cost)


# ---------------------------------------------------------------------------
# GEPA_METRIC_CALLS constants
# ---------------------------------------------------------------------------


def test_gepa_metric_calls_light_is_560():
    """This test proves that the light mode constant hasn't silently changed."""
    assert GEPA_METRIC_CALLS["light"] == 560, (
        f"Expected GEPA_METRIC_CALLS['light'] == 560, got {GEPA_METRIC_CALLS['light']}"
    )


def test_gepa_metric_calls_medium_is_2000():
    """This test proves that the medium mode constant hasn't silently changed."""
    assert GEPA_METRIC_CALLS["medium"] == 2000, (
        f"Expected GEPA_METRIC_CALLS['medium'] == 2000, got {GEPA_METRIC_CALLS['medium']}"
    )


def test_gepa_metric_calls_heavy_is_5000():
    """This test proves that the heavy mode constant hasn't silently changed."""
    assert GEPA_METRIC_CALLS["heavy"] == 5000, (
        f"Expected GEPA_METRIC_CALLS['heavy'] == 5000, got {GEPA_METRIC_CALLS['heavy']}"
    )


# ---------------------------------------------------------------------------
# Cost formula correctness per mode
# ---------------------------------------------------------------------------


def test_light_mode_cost_matches_formula():
    """This test proves that light mode cost = 560 * (500 * input_cost + 500 * output_cost)."""
    config = _config(gepa_auto="light")
    input_cost = 0.000005
    output_cost = 0.000015
    model_info = {
        **_BASE_MODEL_INFO_WITH_PRICING,
        "input_cost_per_token": input_cost,
        "output_cost_per_token": output_cost,
    }

    with patch("litellm.get_model_info", return_value=model_info):
        from rosettastone.preflight.cost_estimator import estimate_cost

        warnings = estimate_cost(config)

    expected = _expected_cost("light", input_cost, output_cost)
    cost_warning = next((w for w in warnings if "$" in w), None)
    assert cost_warning is not None, (
        f"Expected a cost warning containing '$', got warnings: {warnings}"
    )
    assert f"${expected:.2f}" in cost_warning, (
        f"Expected cost ${expected:.2f} in warning, got: {cost_warning!r}"
    )
    assert "light" in cost_warning, (
        f"Expected mode 'light' mentioned in warning, got: {cost_warning!r}"
    )
    assert "560" in cost_warning, f"Expected call count 560 in warning, got: {cost_warning!r}"


def test_medium_mode_cost_matches_formula():
    """This test proves that medium mode cost = 2000 * (500 * input_cost + 500 * output_cost)."""
    config = _config(gepa_auto="medium")
    input_cost = 0.000005
    output_cost = 0.000015
    model_info = {
        **_BASE_MODEL_INFO_WITH_PRICING,
        "input_cost_per_token": input_cost,
        "output_cost_per_token": output_cost,
    }

    with patch("litellm.get_model_info", return_value=model_info):
        from rosettastone.preflight.cost_estimator import estimate_cost

        warnings = estimate_cost(config)

    expected = _expected_cost("medium", input_cost, output_cost)
    cost_warning = next((w for w in warnings if "$" in w and f"${expected:.2f}" in w), None)
    assert cost_warning is not None, (
        f"Expected cost warning with ${expected:.2f} for medium mode, got warnings: {warnings}"
    )
    assert "medium" in cost_warning, (
        f"Expected mode 'medium' mentioned in warning, got: {cost_warning!r}"
    )
    assert "2000" in cost_warning, f"Expected call count 2000 in warning, got: {cost_warning!r}"


def test_heavy_mode_cost_matches_formula():
    """This test proves that heavy mode cost = 5000 * (500 * input_cost + 500 * output_cost)."""
    config = _config(gepa_auto="heavy")
    input_cost = 0.000005
    output_cost = 0.000015
    model_info = {
        **_BASE_MODEL_INFO_WITH_PRICING,
        "input_cost_per_token": input_cost,
        "output_cost_per_token": output_cost,
    }

    with patch("litellm.get_model_info", return_value=model_info):
        from rosettastone.preflight.cost_estimator import estimate_cost

        warnings = estimate_cost(config)

    expected = _expected_cost("heavy", input_cost, output_cost)
    cost_warning = next((w for w in warnings if "$" in w and f"${expected:.2f}" in w), None)
    assert cost_warning is not None, (
        f"Expected cost warning with ${expected:.2f} for heavy mode, got warnings: {warnings}"
    )
    assert "heavy" in cost_warning, (
        f"Expected mode 'heavy' mentioned in warning, got: {cost_warning!r}"
    )
    assert "5000" in cost_warning, f"Expected call count 5000 in warning, got: {cost_warning!r}"


# ---------------------------------------------------------------------------
# $20 threshold warning
# ---------------------------------------------------------------------------


def test_cost_above_20_dollars_produces_additional_threshold_warning():
    """This test proves that estimated cost > $20 produces an extra threshold warning."""
    config = _config(gepa_auto="heavy")
    # heavy mode: 5000 * (500 * 0.01 + 500 * 0.01) = $50,000 — well above $20
    expensive_info = {
        **_BASE_MODEL_INFO_WITH_PRICING,
        "input_cost_per_token": 0.01,
        "output_cost_per_token": 0.01,
    }

    with patch("litellm.get_model_info", return_value=expensive_info):
        from rosettastone.preflight.cost_estimator import estimate_cost

        warnings = estimate_cost(config)

    # Should have 2 warnings: one with cost amount, one about exceeding $20
    assert len(warnings) >= 2, (
        f"Expected at least 2 warnings when cost exceeds $20, got {len(warnings)}: {warnings}"
    )
    threshold_warning = next(
        (
            w
            for w in warnings
            if "exceeds $20" in w or "light" in w.lower() or "reduce" in w.lower()
        ),
        None,
    )
    assert threshold_warning is not None, (
        f"Expected a warning about exceeding $20 / reducing dataset, got warnings: {warnings}"
    )


def test_cost_under_20_dollars_does_not_produce_threshold_warning():
    """This test proves that cost < $20 does NOT produce the threshold warning."""
    config = _config(gepa_auto="light")
    # light: 560 * (500 * 0.000001 + 500 * 0.000001) = 560 * 0.001 = $0.56 — under $20
    cheap_info = {
        **_BASE_MODEL_INFO_WITH_PRICING,
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000001,
    }

    with patch("litellm.get_model_info", return_value=cheap_info):
        from rosettastone.preflight.cost_estimator import estimate_cost

        warnings = estimate_cost(config)

    threshold_warning = next(
        (w for w in warnings if "exceeds $20" in w or "Exceeds $20" in w), None
    )
    assert threshold_warning is None, (
        f"Expected no $20-threshold warning for cheap model, got: {threshold_warning!r}"
    )


def test_cost_just_below_20_does_not_trigger_threshold():
    """This test proves that cost just under $20 does NOT trigger the extra threshold warning.

    We avoid landing exactly on $20.00 because floating-point division makes that
    boundary non-representable exactly in IEEE-754. Instead we use a cost of ~$19.99
    which is safely below the threshold.
    """
    config = _config(gepa_auto="light")
    calls = GEPA_METRIC_CALLS["light"]  # 560
    # Target ~$19.99: cost = calls * 500 * (in + out) ≈ 19.99
    # → unit_cost = 19.99 / (560 * 1000) ≈ 3.569e-05
    target_cost = 19.99
    unit_cost = target_cost / (calls * 1000)
    boundary_info = {
        **_BASE_MODEL_INFO_WITH_PRICING,
        "input_cost_per_token": unit_cost,
        "output_cost_per_token": unit_cost,
    }

    with patch("litellm.get_model_info", return_value=boundary_info):
        from rosettastone.preflight.cost_estimator import estimate_cost

        warnings = estimate_cost(config)

    threshold_warning = next(
        (w for w in warnings if "exceeds $20" in w or "Exceeds $20" in w), None
    )
    assert threshold_warning is None, (
        f"Expected no threshold warning when cost ≈ $19.99, got: {threshold_warning!r}"
    )


# ---------------------------------------------------------------------------
# Zero-cost pricing (tokens cost nothing)
# ---------------------------------------------------------------------------


def test_zero_cost_tokens_produce_no_cost_warning():
    """This test proves that when both in/out costs are 0.0, no cost warning is emitted."""
    config = _config(gepa_auto="light")
    free_info = {
        **_BASE_MODEL_INFO_WITH_PRICING,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }

    with patch("litellm.get_model_info", return_value=free_info):
        from rosettastone.preflight.cost_estimator import estimate_cost

        warnings = estimate_cost(config)

    assert warnings == [], f"Expected no warnings for zero-cost pricing, got: {warnings}"


def test_null_cost_fields_treated_as_zero_no_cost_warning():
    """This test proves that None pricing values are coerced to 0 and produce no cost warning."""
    config = _config(gepa_auto="light")
    null_pricing_info = {
        **_BASE_MODEL_INFO_WITH_PRICING,
        "input_cost_per_token": None,
        "output_cost_per_token": None,
    }

    with patch("litellm.get_model_info", return_value=null_pricing_info):
        from rosettastone.preflight.cost_estimator import estimate_cost

        warnings = estimate_cost(config)

    # None pricing should be treated as zero → no cost warning (not a crash)
    assert all("$" not in w for w in warnings), (
        f"Expected no cost-amount warning when pricing is None, got: {warnings}"
    )


# ---------------------------------------------------------------------------
# Pricing unavailable (get_model_info raises)
# ---------------------------------------------------------------------------


def test_get_model_info_raises_returns_unavailable_warning():
    """This test proves that when get_model_info() raises, a 'not available' warning is returned."""
    config = _config()

    with patch("litellm.get_model_info", side_effect=Exception("Connection refused")):
        from rosettastone.preflight.cost_estimator import estimate_cost

        warnings = estimate_cost(config)

    assert len(warnings) >= 1, "Expected at least one warning when get_model_info raises, got none"
    unavailable_warning = next(
        (
            w
            for w in warnings
            if "pricing" in w.lower() or "cost" in w.lower() or "not available" in w.lower()
        ),
        None,
    )
    assert unavailable_warning is not None, (
        f"Expected a warning about pricing being unavailable, got: {warnings}"
    )


def test_get_model_info_raises_returns_list_not_crash():
    """This test proves that a get_model_info() failure returns a list, not an exception."""
    config = _config()

    with patch("litellm.get_model_info", side_effect=RuntimeError("Model unknown")):
        from rosettastone.preflight.cost_estimator import estimate_cost

        result = estimate_cost(config)

    assert isinstance(result, list), (
        f"Expected estimate_cost to return a list even on failure, got: {type(result)}"
    )


# ---------------------------------------------------------------------------
# Return type contract
# ---------------------------------------------------------------------------


def test_estimate_cost_always_returns_list():
    """This test proves that estimate_cost always returns a list of strings."""
    config = _config()

    with patch("litellm.get_model_info", return_value=_BASE_MODEL_INFO_WITH_PRICING.copy()):
        from rosettastone.preflight.cost_estimator import estimate_cost

        result = estimate_cost(config)

    assert isinstance(result, list), f"Expected list return type, got: {type(result)}"
    for item in result:
        assert isinstance(item, str), (
            f"Expected all warnings to be strings, got: {type(item)} — {item!r}"
        )
