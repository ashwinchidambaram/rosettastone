"""Tests for JSONEvaluator."""

from __future__ import annotations

import pytest

from rosettastone.evaluate.json_validator import JSONEvaluator


class TestJSONEvaluator:
    def setup_method(self) -> None:
        self.evaluator = JSONEvaluator()

    # --- Valid JSON dicts ---

    def test_identical_dicts(self) -> None:
        expected = '{"name": "Alice", "age": 30}'
        actual = '{"name": "Alice", "age": 30}'
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0

    def test_same_keys_different_values(self) -> None:
        expected = '{"name": "Alice", "age": 30}'
        actual = '{"name": "Bob", "age": 25}'
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        # All keys overlap but no values match → key_overlap=1.0, value_score=0.0 → field_match=0.5
        assert scores["json_field_match"] < 1.0

    def test_partial_key_overlap(self) -> None:
        expected = '{"name": "Alice", "age": 30, "city": "NY"}'
        actual = '{"name": "Alice", "age": 30}'
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        # expected has 3 keys, actual has 2; overlap = 2/3; shared values match 2/2
        # key_overlap = 2/3, value_score = 2/2 = 1.0, field_match = (2/3 + 1.0) / 2
        assert 0.0 < scores["json_field_match"] < 1.0

    def test_disjoint_keys(self) -> None:
        expected = '{"name": "Alice"}'
        actual = '{"city": "NY"}'
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        # key_overlap = 0/1 = 0.0, shared = empty, value_score = 0/max(0,1) = 0.0
        # field_match = (0 + 0) / 2 = 0.0
        assert scores["json_field_match"] == 0.0

    def test_expected_invalid_json(self) -> None:
        scores = self.evaluator.score("{not valid}", '{"key": "val"}')
        assert scores["json_valid"] == 0.0
        assert scores["json_field_match"] == 0.0

    def test_actual_invalid_json(self) -> None:
        scores = self.evaluator.score('{"key": "val"}', "{not valid}")
        assert scores["json_valid"] == 0.0
        assert scores["json_field_match"] == 0.0

    def test_both_invalid_json(self) -> None:
        scores = self.evaluator.score("{bad}", "{also bad}")
        assert scores["json_valid"] == 0.0
        assert scores["json_field_match"] == 0.0

    # --- Arrays ---

    def test_equal_arrays(self) -> None:
        expected = "[1, 2, 3]"
        actual = "[1, 2, 3]"
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0

    def test_different_arrays(self) -> None:
        expected = "[1, 2, 3]"
        actual = "[4, 5, 6]"
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 0.0

    # --- Primitives ---

    def test_equal_string_primitives(self) -> None:
        scores = self.evaluator.score('"hello"', '"hello"')
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0

    def test_different_string_primitives(self) -> None:
        scores = self.evaluator.score('"hello"', '"world"')
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 0.0

    def test_equal_number_primitives(self) -> None:
        scores = self.evaluator.score("42", "42")
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0

    def test_different_number_primitives(self) -> None:
        scores = self.evaluator.score("42", "99")
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 0.0

    # --- Edge cases ---

    def test_returns_correct_keys(self) -> None:
        scores = self.evaluator.score('{"a": 1}', '{"a": 1}')
        assert "json_valid" in scores
        assert "json_field_match" in scores

    def test_empty_expected_dict(self) -> None:
        # expected has no keys: expected_keys is empty → field_match = 1.0 if actual_keys empty
        scores = self.evaluator.score("{}", "{}")
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0

    def test_empty_expected_dict_nonempty_actual(self) -> None:
        # expected empty, actual has keys → field_match = 0.5 (per code)
        scores = self.evaluator.score("{}", '{"key": "val"}')
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 0.5

    def test_nested_dict_key_match(self) -> None:
        # Nested values are compared by equality (shallow)
        expected = '{"data": {"nested": 1}}'
        actual = '{"data": {"nested": 1}}'
        scores = self.evaluator.score(expected, actual)
        assert scores["json_valid"] == 1.0
        assert scores["json_field_match"] == 1.0

    def test_scores_in_valid_range(self) -> None:
        scores = self.evaluator.score('{"a": 1, "b": 2}', '{"a": 1, "c": 3}')
        assert 0.0 <= scores["json_valid"] <= 1.0
        assert 0.0 <= scores["json_field_match"] <= 1.0
