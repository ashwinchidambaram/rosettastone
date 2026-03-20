"""Tests for output type detection logic."""

from __future__ import annotations

from rosettastone.core.types import OutputType
from rosettastone.evaluate.types import detect_output_type


class TestDetectOutputType:
    def test_json_dict_string(self) -> None:
        assert detect_output_type('{"key": "val"}') == OutputType.JSON

    def test_json_numeric_string(self) -> None:
        # json.loads("42") succeeds, so numeric strings are JSON (without classify prompt)
        assert detect_output_type("42") == OutputType.JSON

    def test_json_array_string(self) -> None:
        assert detect_output_type("[1, 2, 3]") == OutputType.JSON

    def test_json_boolean_string(self) -> None:
        assert detect_output_type("true") == OutputType.JSON

    def test_empty_string_is_short_text(self) -> None:
        # Empty string hits the guard clause: `if not response: return SHORT_TEXT`
        result = detect_output_type("")
        assert result == OutputType.SHORT_TEXT

    def test_whitespace_only_is_short_text(self) -> None:
        assert detect_output_type("   ") == OutputType.SHORT_TEXT

    def test_single_word_is_classification(self) -> None:
        assert detect_output_type("Paris") == OutputType.CLASSIFICATION

    def test_five_word_text_is_classification(self) -> None:
        assert detect_output_type("one two three four five") == OutputType.CLASSIFICATION

    def test_six_word_text_is_short_text(self) -> None:
        assert detect_output_type("one two three four five six") == OutputType.SHORT_TEXT

    def test_fifty_word_text_is_short_text(self) -> None:
        words = " ".join(["word"] * 50)
        assert detect_output_type(words) == OutputType.SHORT_TEXT

    def test_fifty_one_word_text_is_long_text(self) -> None:
        words = " ".join(["word"] * 51)
        assert detect_output_type(words) == OutputType.LONG_TEXT

    def test_newline_prevents_classification(self) -> None:
        # Text with a newline and <=5 words should NOT be CLASSIFICATION
        result = detect_output_type("yes\nno")
        assert result != OutputType.CLASSIFICATION
        assert result == OutputType.SHORT_TEXT

    def test_classification_whitespace_stripped(self) -> None:
        assert detect_output_type("  Paris  ") == OutputType.CLASSIFICATION

    def test_invalid_json_falls_through(self) -> None:
        result = detect_output_type("{not valid json}")
        assert result != OutputType.JSON

    def test_multiline_long_text(self) -> None:
        words = " ".join(["word"] * 51)
        long_with_newline = words + "\nextra"
        assert detect_output_type(long_with_newline) == OutputType.LONG_TEXT

    def test_short_text_no_newline_six_words(self) -> None:
        result = detect_output_type("The quick brown fox jumps over")
        assert result == OutputType.SHORT_TEXT


# ---------------------------------------------------------------------------
# New: markdown code fence stripping
# ---------------------------------------------------------------------------


class TestCodeFenceStripping:
    def test_json_in_backtick_fence_detected_as_json(self) -> None:
        response = '```json\n{"key": "value"}\n```'
        assert detect_output_type(response) == OutputType.JSON

    def test_json_in_plain_fence_detected_as_json(self) -> None:
        response = '```\n{"key": "value"}\n```'
        assert detect_output_type(response) == OutputType.JSON

    def test_json_array_in_fence(self) -> None:
        response = "```json\n[1, 2, 3]\n```"
        assert detect_output_type(response) == OutputType.JSON

    def test_non_json_in_fence_falls_through(self) -> None:
        # Fenced block with non-JSON content should NOT be classified as JSON
        response = "```\nhello world\n```"
        result = detect_output_type(response)
        assert result != OutputType.JSON

    def test_fence_stripped_before_type_detection(self) -> None:
        # A fenced JSON dict should be JSON, not CLASSIFICATION despite being short
        response = "```json\n{}\n```"
        assert detect_output_type(response) == OutputType.JSON

    def test_case_insensitive_fence_language(self) -> None:
        response = '```JSON\n{"a": 1}\n```'
        assert detect_output_type(response) == OutputType.JSON

    def test_no_fence_json_still_works(self) -> None:
        # Existing behaviour: unfenced JSON still detected
        assert detect_output_type('{"a": 1}') == OutputType.JSON


# ---------------------------------------------------------------------------
# New: numeric JSON + classify/label prompt override → CLASSIFICATION
# ---------------------------------------------------------------------------


class TestClassifyLabelOverride:
    def test_numeric_json_with_classify_prompt(self) -> None:
        assert (
            detect_output_type("42", prompt="Please classify the sentiment")
            == OutputType.CLASSIFICATION
        )

    def test_numeric_json_with_label_prompt(self) -> None:
        assert (
            detect_output_type("3", prompt="Label the category of this document")
            == OutputType.CLASSIFICATION
        )

    def test_numeric_json_without_classify_prompt_is_json(self) -> None:
        # Without a classify/label prompt, numeric JSON stays JSON
        assert detect_output_type("42") == OutputType.JSON

    def test_numeric_json_with_unrelated_prompt_is_json(self) -> None:
        assert detect_output_type("42", prompt="What is 6 times 7?") == OutputType.JSON

    def test_numeric_float_with_classify_prompt(self) -> None:
        assert detect_output_type("3.14", prompt="classify this value") == OutputType.CLASSIFICATION

    def test_categorize_keyword_triggers_override(self) -> None:
        assert (
            detect_output_type("2", prompt="Categorize the following text")
            == OutputType.CLASSIFICATION
        )

    def test_tag_keyword_triggers_override(self) -> None:
        assert (
            detect_output_type("1", prompt="Tag the input with a number")
            == OutputType.CLASSIFICATION
        )

    def test_dict_json_with_classify_prompt_stays_json(self) -> None:
        # The override only applies to scalar numerics, not dicts
        assert detect_output_type('{"label": 1}', prompt="classify this") == OutputType.JSON

    def test_prompt_none_does_not_override(self) -> None:
        # prompt=None (default) → no override
        assert detect_output_type("5") == OutputType.JSON

    def test_keyword_case_insensitive(self) -> None:
        assert detect_output_type("1", prompt="CLASSIFY the following") == OutputType.CLASSIFICATION
