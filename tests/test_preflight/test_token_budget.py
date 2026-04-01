"""Tests for preflight.token_budget.check_token_budget.

This module proves that token budget checks correctly classify prompts as blockers
(exceeds context window) or warnings (exceeds max_context_usage threshold), that
failures in get_model_info/token_counter are handled gracefully, and that pairs
that can't be counted are silently skipped.

Mock strategy: token_budget.py uses `import litellm` inside the function body
(lazy import). We patch `litellm.get_model_info` and `litellm.token_counter`
directly on the litellm package. Data loading is tested via real temp JSONL files
(avoids over-mocking the ingest layer).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from rosettastone.config import MigrationConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_INFO_128K = {
    "max_input_tokens": 128000,
    "max_output_tokens": 4096,
    "input_cost_per_token": 0.000005,
    "output_cost_per_token": 0.000015,
    "supports_function_calling": True,
    "supports_vision": True,
    "supports_response_schema": True,
}


def _make_jsonl_file(*prompts: str) -> Path:
    """Write a temp JSONL file containing the given prompts and return its Path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for prompt in prompts:
            line = {"prompt": prompt, "response": "ok", "source_model": "openai/gpt-4o"}
            f.write(json.dumps(line) + "\n")
        return Path(f.name)


def _config(data_path: Path, max_context_usage: float = 0.75) -> MigrationConfig:
    return MigrationConfig(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        data_path=data_path,
        max_context_usage=max_context_usage,
    )


# ---------------------------------------------------------------------------
# Tokens well within budget — no warnings
# ---------------------------------------------------------------------------


def test_tokens_well_within_budget_produces_no_warnings_or_blockers():
    """This test proves that prompts using a small fraction of the context window are fine."""
    data_path = _make_jsonl_file("Short prompt.", "Another short prompt.")

    with (
        patch("litellm.get_model_info", return_value=_MODEL_INFO_128K.copy()),
        patch("litellm.token_counter", return_value=50),
    ):
        # 50 tokens out of 128000 = 0.04% — well under 75%
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(_config(data_path))

    assert warnings == [], f"Expected no warnings for prompts well within budget, got: {warnings}"
    assert blockers == [], f"Expected no blockers for prompts well within budget, got: {blockers}"


# ---------------------------------------------------------------------------
# Tokens exceed context window → blocker
# ---------------------------------------------------------------------------


def test_tokens_exceeding_context_window_produces_blocker():
    """This test proves that a prompt exceeding max_input_tokens produces a blocker, not a warning."""
    data_path = _make_jsonl_file("This prompt is way too long.")

    with (
        patch("litellm.get_model_info", return_value=_MODEL_INFO_128K.copy()),
        patch("litellm.token_counter", return_value=200000),
    ):
        # 200000 tokens out of 128000 — exceeds the context window
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(_config(data_path))

    assert len(blockers) >= 1, (
        f"Expected at least one blocker when tokens exceed context window, got: {blockers}"
    )
    blocker = blockers[0]
    assert "200,000" in blocker or "200000" in blocker, (
        f"Expected actual token count in blocker message, got: {blocker!r}"
    )
    assert "128,000" in blocker or "128000" in blocker, (
        f"Expected max token count in blocker message, got: {blocker!r}"
    )


def test_tokens_exceeding_context_window_is_blocker_not_warning():
    """This test proves the distinction: context overflow is a blocker, not just a warning."""
    data_path = _make_jsonl_file("Oversized prompt text here.")

    with (
        patch("litellm.get_model_info", return_value=_MODEL_INFO_128K.copy()),
        patch("litellm.token_counter", return_value=200000),
    ):
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(_config(data_path))

    assert len(blockers) >= 1, (
        f"Expected at least one blocker for context overflow, got: {blockers}"
    )
    overflow_in_blockers = any("200,000" in b or "200000" in b for b in blockers)
    overflow_in_warnings = any("200,000" in w or "200000" in w for w in warnings)
    assert overflow_in_blockers, (
        f"Expected overflow token count in blockers, got blockers: {blockers}"
    )
    assert not overflow_in_warnings, (
        f"Overflow should be a blocker, not a warning, got warnings: {warnings}"
    )


def test_blocker_message_references_correct_prompt_index():
    """This test proves that the blocker message identifies which prompt (by index) is oversized."""
    data_path = _make_jsonl_file("Short prompt.", "This enormous prompt exceeds the window.")

    def token_counter_side_effect(model, text):
        if "enormous" in text:
            return 200000  # this one exceeds the window
        return 10  # short prompt is fine

    with (
        patch("litellm.get_model_info", return_value=_MODEL_INFO_128K.copy()),
        patch("litellm.token_counter", side_effect=token_counter_side_effect),
    ):
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(_config(data_path))

    assert len(blockers) == 1, (
        f"Expected exactly 1 blocker (for the oversized prompt), got: {blockers}"
    )
    # The oversized prompt is at index 1 (0-indexed), so "1" appears in the message
    assert "1" in blockers[0], (
        f"Expected prompt index 1 referenced in blocker message, got: {blockers[0]!r}"
    )


# ---------------------------------------------------------------------------
# Tokens exceed 75% threshold → warning
# ---------------------------------------------------------------------------


def test_tokens_above_75_percent_threshold_produces_warning():
    """This test proves that a prompt using more than 75% of the context window gets a warning."""
    data_path = _make_jsonl_file("A long-ish prompt that fills most of the context.")
    max_input = _MODEL_INFO_128K["max_input_tokens"]  # 128000
    token_count = int(max_input * 0.80)  # 80% — above the 75% default threshold

    with (
        patch("litellm.get_model_info", return_value=_MODEL_INFO_128K.copy()),
        patch("litellm.token_counter", return_value=token_count),
    ):
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(_config(data_path))

    assert len(warnings) >= 1, (
        f"Expected at least one warning at 80% context usage, got: {warnings}"
    )
    usage_warning = next((w for w in warnings if "%" in w or "token" in w.lower()), None)
    assert usage_warning is not None, (
        f"Expected a warning mentioning token usage percentage, got: {warnings}"
    )
    assert blockers == [], f"80% usage should be a warning, not a blocker, got: {blockers}"


def test_tokens_at_exactly_75_percent_does_not_produce_warning():
    """This test proves that exactly 75% usage does NOT produce a warning (threshold is > not >=)."""
    data_path = _make_jsonl_file("A prompt right at the threshold.")
    max_input = _MODEL_INFO_128K["max_input_tokens"]
    # Exactly 75% → usage_pct == 0.75, which does NOT satisfy > 0.75
    token_count = int(max_input * 0.75)

    with (
        patch("litellm.get_model_info", return_value=_MODEL_INFO_128K.copy()),
        patch("litellm.token_counter", return_value=token_count),
    ):
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(_config(data_path))

    assert warnings == [], (
        f"Expected no warning at exactly 75% context usage (> threshold, not >=), got: {warnings}"
    )
    assert blockers == [], f"Expected no blockers at 75% usage, got: {blockers}"


def test_tokens_at_74_percent_produces_no_warning():
    """This test proves that usage just below the threshold is fine."""
    data_path = _make_jsonl_file("A prompt just under the threshold.")
    max_input = _MODEL_INFO_128K["max_input_tokens"]
    token_count = int(max_input * 0.74)

    with (
        patch("litellm.get_model_info", return_value=_MODEL_INFO_128K.copy()),
        patch("litellm.token_counter", return_value=token_count),
    ):
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(_config(data_path))

    assert warnings == [], f"Expected no warning at 74% context usage, got: {warnings}"
    assert blockers == [], f"Expected no blockers at 74% usage, got: {blockers}"


def test_custom_max_context_usage_threshold_respected():
    """This test proves that config.max_context_usage overrides the default 75% threshold."""
    data_path = _make_jsonl_file("A moderately long prompt.")
    max_input = _MODEL_INFO_128K["max_input_tokens"]
    # Use 60% of max_input — above a custom 50% threshold but below default 75%
    token_count = int(max_input * 0.60)
    config = _config(data_path, max_context_usage=0.50)

    with (
        patch("litellm.get_model_info", return_value=_MODEL_INFO_128K.copy()),
        patch("litellm.token_counter", return_value=token_count),
    ):
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(config)

    assert len(warnings) >= 1, (
        f"Expected warning at 60% usage with a 50% threshold, got: {warnings}"
    )
    assert blockers == [], f"Expected no blockers, got: {blockers}"


# ---------------------------------------------------------------------------
# Multiple prompts: correct per-prompt checking
# ---------------------------------------------------------------------------


def test_multiple_prompts_each_checked_independently():
    """This test proves that each prompt in the sample is checked, not just the first."""
    data_path = _make_jsonl_file(
        "Short 1.", "Short 2.", "Short 3.", "Oversized prompt here.", "Short 5."
    )
    max_input = _MODEL_INFO_128K["max_input_tokens"]

    def token_counter_side_effect(model, text):
        if "Oversized" in text:
            return int(max_input * 0.90)  # above 75% → warning
        return 50

    with (
        patch("litellm.get_model_info", return_value=_MODEL_INFO_128K.copy()),
        patch("litellm.token_counter", side_effect=token_counter_side_effect),
    ):
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(_config(data_path))

    assert len(warnings) == 1, (
        f"Expected exactly 1 warning (for the oversized prompt), got {len(warnings)}: {warnings}"
    )
    assert blockers == [], f"Expected no blockers, got: {blockers}"


def test_oversized_prompt_beyond_sample_limit_not_caught():
    """Token budget checks a random sample of up to 20 pairs; prompts beyond position 20 are skipped.

    Patches random.sample to deterministically return the first k items in order, so the
    21st (oversized) prompt is reliably excluded from the sample and the assertion is stable.
    """
    # 21 prompts; only the 21st is oversized
    data_path = _make_jsonl_file(*[f"P{i}." for i in range(20)], "P21 oversized beyond limit.")
    max_input = _MODEL_INFO_128K["max_input_tokens"]

    oversized_caught = False

    def token_counter_side_effect(model, text):
        if "oversized" in text.lower():
            nonlocal oversized_caught
            oversized_caught = True
            return max_input * 2
        return 50

    with (
        patch("litellm.get_model_info", return_value=_MODEL_INFO_128K.copy()),
        patch("litellm.token_counter", side_effect=token_counter_side_effect),
        # Deterministic sample: always return first k items in order, keeping the oversized
        # 21st prompt (index 20) outside the 20-item sample window.
        patch("random.sample", side_effect=lambda population, k: list(population)[:k]),
    ):
        from rosettastone.preflight.token_budget import check_token_budget

        check_token_budget(_config(data_path))

    assert not oversized_caught, (
        "Expected prompt beyond 20-item sample limit not to be checked"
    )


# ---------------------------------------------------------------------------
# Error handling: get_model_info raises
# ---------------------------------------------------------------------------


def test_get_model_info_raises_returns_warning_not_crash():
    """This test proves that get_model_info() failure returns a warning instead of raising."""
    data_path = _make_jsonl_file("A prompt.")

    with patch("litellm.get_model_info", side_effect=Exception("Network unreachable")):
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(_config(data_path))

    assert len(warnings) >= 1, "Expected at least one warning when get_model_info raises, got none"
    assert blockers == [], f"A get_model_info failure should not produce blockers, got: {blockers}"


def test_get_model_info_raises_warning_mentions_context_window():
    """This test proves that the failure warning is about the context window, not a generic error."""
    data_path = _make_jsonl_file("A prompt.")

    with patch("litellm.get_model_info", side_effect=RuntimeError("Model not found")):
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(_config(data_path))

    context_warning = next(
        (
            w
            for w in warnings
            if "context" in w.lower() or "window" in w.lower() or "determine" in w.lower()
        ),
        None,
    )
    assert context_warning is not None, (
        f"Expected a warning about context window determination, got: {warnings}"
    )


def test_get_model_info_returns_zero_max_input_exits_cleanly():
    """This test proves that max_input_tokens=0 causes early return with no warnings or blockers."""
    data_path = _make_jsonl_file("A prompt.")
    zero_info = {**_MODEL_INFO_128K, "max_input_tokens": 0}

    with patch("litellm.get_model_info", return_value=zero_info):
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(_config(data_path))

    assert warnings == [], f"Expected no warnings when max_input_tokens is 0, got: {warnings}"
    assert blockers == [], f"Expected no blockers when max_input_tokens is 0, got: {blockers}"


def test_get_model_info_missing_max_input_key_exits_cleanly():
    """This test proves that a missing max_input_tokens key (defaults to 0) causes early return."""
    data_path = _make_jsonl_file("A prompt.")
    no_ctx_info = {k: v for k, v in _MODEL_INFO_128K.items() if k != "max_input_tokens"}

    with patch("litellm.get_model_info", return_value=no_ctx_info):
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(_config(data_path))

    assert warnings == [], (
        f"Expected no warnings when max_input_tokens key is missing, got: {warnings}"
    )
    assert blockers == [], (
        f"Expected no blockers when max_input_tokens key is missing, got: {blockers}"
    )


# ---------------------------------------------------------------------------
# Error handling: token_counter raises for a pair → pair silently skipped
# ---------------------------------------------------------------------------


def test_token_counter_raises_for_one_pair_that_pair_is_skipped():
    """This test proves that a token_counter failure for one pair doesn't affect other pairs."""
    data_path = _make_jsonl_file("Good prompt.", "Bad prompt that causes token_counter to fail.")

    def token_counter_side_effect(model, text):
        if "Bad" in text:
            raise Exception("Tokenizer failure")
        return 50  # good prompt is fine

    with (
        patch("litellm.get_model_info", return_value=_MODEL_INFO_128K.copy()),
        patch("litellm.token_counter", side_effect=token_counter_side_effect),
    ):
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(_config(data_path))

    # The bad pair is silently skipped — no warning or blocker for it
    assert warnings == [], (
        f"Expected no warnings when token_counter fails for one pair, got: {warnings}"
    )
    assert blockers == [], (
        f"Expected no blockers when token_counter fails for one pair, got: {blockers}"
    )


def test_token_counter_raises_for_all_pairs_produces_no_crash():
    """This test proves that token_counter failing for every pair still returns cleanly."""
    data_path = _make_jsonl_file("Prompt A.", "Prompt B.", "Prompt C.")

    with (
        patch("litellm.get_model_info", return_value=_MODEL_INFO_128K.copy()),
        patch("litellm.token_counter", side_effect=Exception("Always fails")),
    ):
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(_config(data_path))

    # All pairs are skipped; no crash, no warnings, no blockers
    assert isinstance(warnings, list), f"Expected list, got {type(warnings)}"
    assert isinstance(blockers, list), f"Expected list, got {type(blockers)}"
    assert blockers == [], (
        f"Expected no blockers when all token_counter calls fail, got: {blockers}"
    )


# ---------------------------------------------------------------------------
# Data loading failure
# ---------------------------------------------------------------------------


def test_nonexistent_data_file_produces_warning_not_crash():
    """This test proves that a missing data file produces a warning instead of an unhandled exception."""
    config = _config(Path("/nonexistent/fake_data.jsonl"))

    with patch("litellm.get_model_info", return_value=_MODEL_INFO_128K.copy()):
        from rosettastone.preflight.token_budget import check_token_budget

        warnings, blockers = check_token_budget(config)

    assert len(warnings) >= 1, (
        "Expected at least one warning when data file doesn't exist, got none"
    )
    data_warning = next(
        (
            w
            for w in warnings
            if "budget" in w.lower()
            or "check" in w.lower()
            or "load" in w.lower()
            or "file" in w.lower()
        ),
        None,
    )
    assert data_warning is not None, (
        f"Expected a warning about data loading failure, got: {warnings}"
    )


# ---------------------------------------------------------------------------
# Return type contract
# ---------------------------------------------------------------------------


def test_check_token_budget_always_returns_two_lists():
    """This test proves that check_token_budget always returns a 2-tuple of lists."""
    data_path = _make_jsonl_file("A prompt.")

    with (
        patch("litellm.get_model_info", return_value=_MODEL_INFO_128K.copy()),
        patch("litellm.token_counter", return_value=50),
    ):
        from rosettastone.preflight.token_budget import check_token_budget

        result = check_token_budget(_config(data_path))

    assert isinstance(result, tuple), f"Expected tuple return, got: {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple (warnings, blockers), got {len(result)}-tuple"
    warnings, blockers = result
    assert isinstance(warnings, list), f"Expected warnings to be a list, got: {type(warnings)}"
    assert isinstance(blockers, list), f"Expected blockers to be a list, got: {type(blockers)}"
