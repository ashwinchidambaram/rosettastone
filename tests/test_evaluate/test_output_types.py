"""Tests for output type detection logic."""

from __future__ import annotations

from rosettastone.core.types import OutputType
from rosettastone.evaluate.types import detect_output_type


class TestDetectOutputType:
    def test_json_dict_string(self) -> None:
        assert detect_output_type('{"key": "val"}') == OutputType.JSON

    def test_json_numeric_string(self) -> None:
        # json.loads("42") succeeds, so numeric strings are JSON
        assert detect_output_type("42") == OutputType.JSON

    def test_json_array_string(self) -> None:
        assert detect_output_type("[1, 2, 3]") == OutputType.JSON

    def test_json_boolean_string(self) -> None:
        assert detect_output_type("true") == OutputType.JSON

    def test_empty_string_is_short_text(self) -> None:
        # Empty string hits the guard clause: `if not response: return SHORT_TEXT`
        # This was fixed to avoid classifying empty responses as CLASSIFICATION
        result = detect_output_type("")
        assert result == OutputType.SHORT_TEXT

    def test_whitespace_only_is_short_text(self) -> None:
        # Whitespace-only is stripped to empty, then hits the guard clause
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
        # because the "\n" check fails
        result = detect_output_type("yes\nno")
        assert result != OutputType.CLASSIFICATION
        assert result == OutputType.SHORT_TEXT

    def test_classification_whitespace_stripped(self) -> None:
        # Leading/trailing whitespace stripped before evaluation
        assert detect_output_type("  Paris  ") == OutputType.CLASSIFICATION

    def test_invalid_json_falls_through(self) -> None:
        # Malformed JSON should not be detected as JSON
        result = detect_output_type("{not valid json}")
        assert result != OutputType.JSON

    def test_multiline_long_text(self) -> None:
        # Many words with newlines → LONG_TEXT (fails classification newline check,
        # then checks word count)
        words = " ".join(["word"] * 51)
        long_with_newline = words + "\nextra"
        assert detect_output_type(long_with_newline) == OutputType.LONG_TEXT

    def test_short_text_no_newline_six_words(self) -> None:
        result = detect_output_type("The quick brown fox jumps over")
        assert result == OutputType.SHORT_TEXT
