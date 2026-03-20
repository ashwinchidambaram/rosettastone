"""Tests for feedback utilities in optimize/feedback.py.

Tests cover build_feedback_map key encoding, feedback skipping for None values,
and prepend_feedback with/without a known issue string.
"""

from __future__ import annotations

import json

from rosettastone.core.types import PromptPair
from rosettastone.optimize.feedback import build_feedback_map, prepend_feedback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _str_pair(prompt: str, feedback: str | None = None) -> PromptPair:
    return PromptPair(
        prompt=prompt,
        response="response",
        source_model="openai/gpt-4o",
        feedback=feedback,
    )


def _list_pair(prompt: list, feedback: str | None = None) -> PromptPair:
    return PromptPair(
        prompt=prompt,
        response="response",
        source_model="openai/gpt-4o",
        feedback=feedback,
    )


# ---------------------------------------------------------------------------
# build_feedback_map — basic behavior
# ---------------------------------------------------------------------------


class TestBuildFeedbackMapBasic:
    """build_feedback_map must return a dict with correct keys and values."""

    def test_returns_dict(self) -> None:
        """Return type must be dict."""
        result = build_feedback_map([])
        assert isinstance(result, dict)

    def test_empty_train_set_returns_empty_dict(self) -> None:
        """An empty train_set must yield an empty feedback map."""
        result = build_feedback_map([])
        assert result == {}

    def test_single_pair_with_feedback(self) -> None:
        """A pair with feedback must appear in the map."""
        pair = _str_pair("hello", feedback="watch for tone")
        result = build_feedback_map([pair])

        assert len(result) == 1
        assert result["hello"] == "watch for tone"

    def test_multiple_pairs_all_with_feedback(self) -> None:
        """All pairs with feedback must appear in the map."""
        pairs = [
            _str_pair("p1", feedback="issue 1"),
            _str_pair("p2", feedback="issue 2"),
        ]
        result = build_feedback_map(pairs)

        assert result == {"p1": "issue 1", "p2": "issue 2"}


# ---------------------------------------------------------------------------
# build_feedback_map — None feedback filtering
# ---------------------------------------------------------------------------


class TestBuildFeedbackMapNoneFiltering:
    """Pairs with feedback=None must be excluded from the map."""

    def test_skips_pair_with_none_feedback(self) -> None:
        """A pair with feedback=None must not appear in the feedback map."""
        pair = _str_pair("prompt_without_feedback", feedback=None)
        result = build_feedback_map([pair])

        assert result == {}

    def test_mixed_none_and_non_none(self) -> None:
        """Only pairs with non-None feedback must appear in the map."""
        pairs = [
            _str_pair("has feedback", feedback="some issue"),
            _str_pair("no feedback", feedback=None),
        ]
        result = build_feedback_map(pairs)

        assert "has feedback" in result
        assert "no feedback" not in result
        assert len(result) == 1

    def test_all_none_feedback_returns_empty(self) -> None:
        """When all pairs have feedback=None the result must be an empty dict."""
        pairs = [_str_pair(f"p{i}", feedback=None) for i in range(5)]
        result = build_feedback_map(pairs)

        assert result == {}


# ---------------------------------------------------------------------------
# build_feedback_map — key encoding
# ---------------------------------------------------------------------------


class TestBuildFeedbackMapKeyEncoding:
    """Key encoding must differ by prompt type."""

    def test_str_prompt_used_as_key_directly(self) -> None:
        """A str prompt must be used as the dict key without modification."""
        pair = _str_pair("exact string key", feedback="fb")
        result = build_feedback_map([pair])

        assert "exact string key" in result

    def test_list_prompt_encoded_as_json(self) -> None:
        """A list prompt must be encoded via json.dumps(sort_keys=True)."""
        prompt_list = [{"role": "user", "content": "hi"}]
        pair = _list_pair(prompt_list, feedback="list feedback")
        result = build_feedback_map([pair])

        expected_key = json.dumps(prompt_list, sort_keys=True)
        assert expected_key in result
        assert result[expected_key] == "list feedback"

    def test_list_prompt_key_is_sort_keys_stable(self) -> None:
        """json.dumps with sort_keys=True must produce a stable key regardless of insertion order."""
        prompt_a = [{"role": "user", "content": "hi", "extra": 1}]
        prompt_b = [{"extra": 1, "content": "hi", "role": "user"}]

        pair_a = _list_pair(prompt_a, feedback="fb_a")
        pair_b = _list_pair(prompt_b, feedback="fb_b")

        map_a = build_feedback_map([pair_a])
        map_b = build_feedback_map([pair_b])

        # Both should produce the same key
        assert list(map_a.keys()) == list(map_b.keys()), (
            "List prompts with same fields in different order must produce identical keys"
        )


# ---------------------------------------------------------------------------
# prepend_feedback — without known issue
# ---------------------------------------------------------------------------


class TestPrependFeedbackNoIssue:
    """When known_issue is None, base_feedback must be returned unchanged."""

    def test_returns_base_feedback_unchanged(self) -> None:
        """No known issue → base_feedback is returned as-is."""
        result = prepend_feedback("base feedback text", None)
        assert result == "base feedback text"

    def test_returns_empty_string_unchanged(self) -> None:
        """An empty base_feedback string with no known issue returns empty string."""
        result = prepend_feedback("", None)
        assert result == ""

    def test_return_type_is_str(self) -> None:
        """Return type must always be str."""
        result = prepend_feedback("feedback", None)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# prepend_feedback — with known issue
# ---------------------------------------------------------------------------


class TestPrependFeedbackWithIssue:
    """When known_issue is provided, it must be prepended with a KNOWN ISSUE: prefix."""

    def test_prepends_known_issue_prefix(self) -> None:
        """known_issue must be prepended with 'KNOWN ISSUE: ' prefix."""
        result = prepend_feedback("base", "tone mismatch")
        assert result.startswith("KNOWN ISSUE: tone mismatch")

    def test_base_feedback_present_after_newline(self) -> None:
        """The base_feedback must appear after a newline following the known issue."""
        result = prepend_feedback("base feedback", "known problem")
        assert "base feedback" in result
        # Must be separated by a newline
        assert result == "KNOWN ISSUE: known problem\nbase feedback"

    def test_exact_format(self) -> None:
        """Full format: 'KNOWN ISSUE: {known_issue}\\n{base_feedback}'."""
        known = "response too short"
        base = "similarity: 0.65"
        expected = f"KNOWN ISSUE: {known}\n{base}"
        assert prepend_feedback(base, known) == expected

    def test_does_not_alter_score_semantics(self) -> None:
        """prepend_feedback only touches text — scores are not modified."""
        # This is a structural test: the function must not reference any numeric values
        result = prepend_feedback("score: 0.9", "issue")
        assert "0.9" in result, "Base feedback content must be preserved verbatim"

    def test_empty_base_with_known_issue(self) -> None:
        """An empty base_feedback with a known issue still produces valid output."""
        result = prepend_feedback("", "critical issue")
        assert result == "KNOWN ISSUE: critical issue\n"

    def test_empty_known_issue_string_is_still_prepended(self) -> None:
        """An empty string (not None) for known_issue is still treated as a provided value."""
        result = prepend_feedback("base", "")
        assert result == "KNOWN ISSUE: \nbase"

    def test_return_type_is_str(self) -> None:
        """Return type must always be str."""
        result = prepend_feedback("feedback", "issue")
        assert isinstance(result, str)
