"""Tests for preflight.capabilities.check_capabilities.

This module proves that capability gap detection between source and target models
works correctly — emitting warnings for missing features/smaller context windows,
and returning gracefully when litellm.get_model_info() fails.

Mock strategy: capabilities.py uses `import litellm` inside the function body
(lazy import). Because the module-level name `litellm` is not bound in
capabilities.py's namespace at import time, we must patch `litellm.get_model_info`
directly on the litellm package itself.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from rosettastone.config import MigrationConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_MODEL_INFO = {
    "max_input_tokens": 128000,
    "max_output_tokens": 4096,
    "input_cost_per_token": 0.000005,
    "output_cost_per_token": 0.000015,
    "supports_function_calling": True,
    "supports_vision": True,
    "supports_response_schema": True,
}


def _config(
    source: str = "openai/gpt-4o", target: str = "anthropic/claude-sonnet-4"
) -> MigrationConfig:
    return MigrationConfig(
        source_model=source,
        target_model=target,
        data_path=Path("/fake/data.jsonl"),
    )


def _source_info(**overrides) -> dict:
    return {**_BASE_MODEL_INFO, **overrides}


def _target_info(**overrides) -> dict:
    return {**_BASE_MODEL_INFO, **overrides}


# ---------------------------------------------------------------------------
# Happy path: identical capabilities — no warnings, no blockers
# ---------------------------------------------------------------------------


def test_identical_models_produce_no_warnings_or_blockers():
    """This test proves that two models with identical capabilities emit nothing."""
    config = _config()
    mock_info = _BASE_MODEL_INFO.copy()

    with patch("litellm.get_model_info", return_value=mock_info):
        from rosettastone.preflight.capabilities import check_capabilities

        warnings, blockers = check_capabilities(config)

    assert warnings == [], f"Expected no warnings for identical models, got: {warnings}"
    assert blockers == [], f"Expected no blockers for identical models, got: {blockers}"


# ---------------------------------------------------------------------------
# Feature capability gaps → warnings
# ---------------------------------------------------------------------------


def test_source_supports_function_calling_target_does_not_emits_warning():
    """This test proves that a function-calling gap between source and target produces a warning."""
    config = _config()
    source_info = _source_info(supports_function_calling=True)
    target_info = _target_info(supports_function_calling=False)

    with patch("litellm.get_model_info", side_effect=[source_info, target_info]):
        from rosettastone.preflight.capabilities import check_capabilities

        warnings, blockers = check_capabilities(config)

    assert len(warnings) >= 1, (
        "Expected at least one warning when target lacks function calling support"
    )
    function_warning = next(
        (w for w in warnings if "function" in w.lower() or "tool" in w.lower()), None
    )
    assert function_warning is not None, (
        f"Expected a warning mentioning function/tool calling, got warnings: {warnings}"
    )
    assert blockers == [], (
        f"Function calling gap should produce a warning, not a blocker, got: {blockers}"
    )


def test_source_supports_vision_target_does_not_emits_warning():
    """This test proves that a vision capability gap produces a warning."""
    config = _config()
    source_info = _source_info(supports_vision=True)
    target_info = _target_info(supports_vision=False)

    with patch("litellm.get_model_info", side_effect=[source_info, target_info]):
        from rosettastone.preflight.capabilities import check_capabilities

        warnings, blockers = check_capabilities(config)

    vision_warning = next(
        (w for w in warnings if "vision" in w.lower() or "image" in w.lower()), None
    )
    assert vision_warning is not None, (
        f"Expected a warning mentioning vision/image input, got warnings: {warnings}"
    )
    assert blockers == [], f"Vision gap should produce a warning, not a blocker, got: {blockers}"


def test_source_supports_response_schema_target_does_not_emits_warning():
    """This test proves that a structured-output/JSON-mode gap produces a warning."""
    config = _config()
    source_info = _source_info(supports_response_schema=True)
    target_info = _target_info(supports_response_schema=False)

    with patch("litellm.get_model_info", side_effect=[source_info, target_info]):
        from rosettastone.preflight.capabilities import check_capabilities

        warnings, blockers = check_capabilities(config)

    schema_warning = next(
        (
            w
            for w in warnings
            if "structured" in w.lower() or "json" in w.lower() or "schema" in w.lower()
        ),
        None,
    )
    assert schema_warning is not None, (
        f"Expected a warning mentioning structured output/JSON mode, got warnings: {warnings}"
    )
    assert blockers == [], (
        f"Schema support gap should produce a warning, not a blocker, got: {blockers}"
    )


def test_target_has_more_capabilities_than_source_emits_no_warnings():
    """This test proves that the target having MORE features than source is fine — no warnings."""
    config = _config()
    # Source lacks features, target has them — no warning should fire (only downgrade is checked)
    source_info = _source_info(
        supports_function_calling=False,
        supports_vision=False,
        supports_response_schema=False,
    )
    target_info = _target_info(
        supports_function_calling=True,
        supports_vision=True,
        supports_response_schema=True,
    )

    with patch("litellm.get_model_info", side_effect=[source_info, target_info]):
        from rosettastone.preflight.capabilities import check_capabilities

        warnings, blockers = check_capabilities(config)

    # Context windows are equal so no context warning either
    assert warnings == [], (
        f"Expected no warnings when target has more capabilities than source, got: {warnings}"
    )
    assert blockers == [], f"Expected no blockers, got: {blockers}"


def test_multiple_capability_gaps_produce_multiple_warnings():
    """This test proves that each missing feature produces its own distinct warning."""
    config = _config()
    source_info = _source_info(
        supports_function_calling=True,
        supports_vision=True,
        supports_response_schema=True,
    )
    target_info = _target_info(
        supports_function_calling=False,
        supports_vision=False,
        supports_response_schema=False,
    )

    with patch("litellm.get_model_info", side_effect=[source_info, target_info]):
        from rosettastone.preflight.capabilities import check_capabilities

        warnings, blockers = check_capabilities(config)

    # There are 3 capability gaps so we expect at least 3 warnings
    assert len(warnings) >= 3, (
        f"Expected at least 3 warnings for 3 capability gaps, got {len(warnings)}: {warnings}"
    )
    assert blockers == [], f"Capability gaps produce warnings, not blockers, got: {blockers}"


# ---------------------------------------------------------------------------
# Context window comparison → warning
# ---------------------------------------------------------------------------


def test_target_smaller_context_window_emits_warning():
    """This test proves that a smaller target context window produces a warning."""
    config = _config()
    source_info = _source_info(max_input_tokens=128000)
    target_info = _target_info(max_input_tokens=8192)  # much smaller

    with patch("litellm.get_model_info", side_effect=[source_info, target_info]):
        from rosettastone.preflight.capabilities import check_capabilities

        warnings, blockers = check_capabilities(config)

    context_warning = next(
        (
            w
            for w in warnings
            if "context" in w.lower() or "window" in w.lower() or "truncat" in w.lower()
        ),
        None,
    )
    assert context_warning is not None, (
        f"Expected a warning about smaller target context window, got warnings: {warnings}"
    )
    # The warning should contain both token counts so the user can understand the gap
    assert "8,192" in context_warning or "8192" in context_warning, (
        f"Expected target token count in warning, got: {context_warning!r}"
    )
    assert "128,000" in context_warning or "128000" in context_warning, (
        f"Expected source token count in warning, got: {context_warning!r}"
    )
    assert blockers == [], (
        f"Smaller context window should produce a warning, not a blocker, got: {blockers}"
    )


def test_target_larger_context_window_emits_no_context_warning():
    """This test proves that a larger target context window is fine — no warning."""
    config = _config()
    source_info = _source_info(max_input_tokens=8192)
    target_info = _target_info(max_input_tokens=200000)  # bigger

    with patch("litellm.get_model_info", side_effect=[source_info, target_info]):
        from rosettastone.preflight.capabilities import check_capabilities

        warnings, blockers = check_capabilities(config)

    context_warning = next(
        (
            w
            for w in warnings
            if "context" in w.lower() or "window" in w.lower() or "truncat" in w.lower()
        ),
        None,
    )
    assert context_warning is None, (
        f"Expected no context window warning when target is larger, got: {context_warning!r}"
    )


def test_equal_context_windows_emit_no_context_warning():
    """This test proves that equal context windows don't produce a warning."""
    config = _config()
    same_info = _BASE_MODEL_INFO.copy()

    with patch("litellm.get_model_info", return_value=same_info):
        from rosettastone.preflight.capabilities import check_capabilities

        warnings, blockers = check_capabilities(config)

    context_warning = next(
        (w for w in warnings if "context" in w.lower() or "window" in w.lower()), None
    )
    assert context_warning is None, (
        f"Expected no context window warning for equal windows, got: {context_warning!r}"
    )


# ---------------------------------------------------------------------------
# Error handling: get_model_info raises
# ---------------------------------------------------------------------------


def test_get_model_info_raises_exception_returns_warning_not_crash():
    """This test proves that when get_model_info() fails, a warning is returned instead of raising."""
    config = _config()

    with patch("litellm.get_model_info", side_effect=Exception("Network timeout")):
        from rosettastone.preflight.capabilities import check_capabilities

        warnings, blockers = check_capabilities(config)

    assert len(warnings) >= 1, "Expected at least one warning when get_model_info raises, got none"
    skip_warning = next(
        (
            w
            for w in warnings
            if "skip" in w.lower() or "could not" in w.lower() or "fetch" in w.lower()
        ),
        None,
    )
    assert skip_warning is not None, (
        f"Expected a warning about skipping capability checks, got warnings: {warnings}"
    )
    assert blockers == [], f"A get_model_info failure should not produce blockers, got: {blockers}"


def test_get_model_info_raises_does_not_produce_capability_warnings():
    """This test proves that when get_model_info() fails, no false capability warnings are generated."""
    config = _config()

    with patch("litellm.get_model_info", side_effect=RuntimeError("Model not found")):
        from rosettastone.preflight.capabilities import check_capabilities

        warnings, blockers = check_capabilities(config)

    # There should be exactly one warning — the "skipping" message — no capability-specific ones
    assert len(warnings) == 1, (
        f"Expected exactly 1 warning (the skip message) when get_model_info fails, got: {warnings}"
    )


# ---------------------------------------------------------------------------
# Return type contract
# ---------------------------------------------------------------------------


def test_check_capabilities_always_returns_two_lists():
    """This test proves that check_capabilities always returns a 2-tuple of lists."""
    config = _config()

    with patch("litellm.get_model_info", return_value=_BASE_MODEL_INFO.copy()):
        from rosettastone.preflight.capabilities import check_capabilities

        result = check_capabilities(config)

    assert isinstance(result, tuple), f"Expected tuple return, got: {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple (warnings, blockers), got {len(result)}-tuple"
    warnings, blockers = result
    assert isinstance(warnings, list), f"Expected warnings to be a list, got: {type(warnings)}"
    assert isinstance(blockers, list), f"Expected blockers to be a list, got: {type(blockers)}"
