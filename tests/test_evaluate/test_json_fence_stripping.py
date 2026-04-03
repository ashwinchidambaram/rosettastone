"""Tests for markdown fence stripping in JSONEvaluator."""

from __future__ import annotations

from rosettastone.evaluate.json_validator import JSONEvaluator


class TestJSONFenceStripping:
    def setup_method(self) -> None:
        self.evaluator = JSONEvaluator()

    # --- Fenced JSON with language tag ---

    def test_fenced_json_with_language_tag(self) -> None:
        """Fenced JSON (```json...) should parse correctly."""
        expected = '{"name": "Alice", "age": 30}'
        actual = """```json
{"name": "Alice", "age": 30}
```"""
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0

    def test_both_fenced_with_language_tag(self) -> None:
        """Both expected and actual fenced should still match."""
        expected = """```json
{"name": "Alice", "age": 30}
```"""
        actual = """```json
{"name": "Alice", "age": 30}
```"""
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0

    # --- Plain code fences (no language tag) ---

    def test_plain_fences_no_language_tag(self) -> None:
        """Plain ``` fences (no language) should also work."""
        expected = '{"x": 1}'
        actual = """```
{"x": 1}
```"""
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0

    def test_both_plain_fences(self) -> None:
        """Both with plain fences should match."""
        expected = """```
{"x": 1}
```"""
        actual = """```
{"x": 1}
```"""
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0

    # --- Unfenced JSON (regression test) ---

    def test_unfenced_json_still_works(self) -> None:
        """Unfenced JSON should still parse without regression."""
        expected = '{"name": "Bob", "age": 25}'
        actual = '{"name": "Bob", "age": 25}'
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0

    def test_unfenced_array_still_works(self) -> None:
        """Unfenced arrays should still work."""
        expected = "[1, 2, 3]"
        actual = "[1, 2, 3]"
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0

    # --- Mixed fencing ---

    def test_expected_fenced_actual_not(self) -> None:
        """Expected fenced, actual not."""
        expected = """```json
{"key": "value"}
```"""
        actual = '{"key": "value"}'
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0

    def test_expected_not_actual_fenced(self) -> None:
        """Expected not fenced, actual fenced."""
        expected = '{"key": "value"}'
        actual = """```json
{"key": "value"}
```"""
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0

    # --- Non-JSON content (still fails) ---

    def test_fenced_non_json_still_fails(self) -> None:
        """Fenced non-JSON content should still score 0.0."""
        expected = '{"a": 1}'
        actual = """```
not valid json at all
```"""
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 0.0
        assert scores["json_field_match"] == 0.0

    def test_fenced_with_different_json_values(self) -> None:
        """Fenced JSON with different values should score accordingly."""
        expected = '{"x": 1}'
        actual = """```json
{"x": 2}
```"""
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        # Same key, different value → key_overlap=1.0, value_score=0.0 → field_match=0.5
        assert scores["json_field_match"] == 0.5

    # --- Whitespace handling ---

    def test_fenced_with_extra_whitespace(self) -> None:
        """Fenced JSON with extra whitespace should parse."""
        expected = '{"a": 1}'
        actual = """```json

{"a": 1}

```"""
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0

    def test_fenced_complex_json(self) -> None:
        """Fenced complex JSON should work."""
        expected = '{"name": "Alice", "data": [1, 2, 3], "nested": {"key": "val"}}'
        actual = """```json
{"name": "Alice", "data": [1, 2, 3], "nested": {"key": "val"}}
```"""
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0
