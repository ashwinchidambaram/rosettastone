"""Tests for JSONStructuralEvaluator."""

from __future__ import annotations

import pytest

from rosettastone.evaluate.json_structural import (
    JSONStructuralEvaluator,
    _coerce_match,
    _lcs_length,
)

# ---------------------------------------------------------------------------
# _coerce_match
# ---------------------------------------------------------------------------


class TestCoerceMatch:
    def test_equal_values(self) -> None:
        assert _coerce_match(5, 5) == 1.0

    def test_equal_strings(self) -> None:
        assert _coerce_match("hello", "hello") == 1.0

    def test_int_string_coerce(self) -> None:
        # 5 vs "5" → partial match
        assert _coerce_match(5, "5") == 0.5

    def test_string_int_coerce(self) -> None:
        assert _coerce_match("5", 5) == 0.5

    def test_completely_different(self) -> None:
        assert _coerce_match("foo", "bar") == 0.0

    def test_different_numbers(self) -> None:
        assert _coerce_match(1, 2) == 0.0

    def test_none_equal(self) -> None:
        assert _coerce_match(None, None) == 1.0

    def test_bool_true_vs_int_one(self) -> None:
        # In Python, True == 1 so this is a full match
        assert _coerce_match(True, 1) == 1.0


# ---------------------------------------------------------------------------
# _lcs_length
# ---------------------------------------------------------------------------


class TestLCSLength:
    def test_empty_sequences(self) -> None:
        assert _lcs_length([], []) == 0

    def test_one_empty(self) -> None:
        assert _lcs_length([1, 2, 3], []) == 0

    def test_identical_sequences(self) -> None:
        assert _lcs_length([1, 2, 3], [1, 2, 3]) == 3

    def test_no_common_elements(self) -> None:
        assert _lcs_length([1, 2, 3], [4, 5, 6]) == 0

    def test_partial_overlap(self) -> None:
        assert _lcs_length([1, 2, 3, 4], [2, 4]) == 2

    def test_single_element_match(self) -> None:
        assert _lcs_length([1], [1]) == 1

    def test_single_element_no_match(self) -> None:
        assert _lcs_length([1], [2]) == 0


# ---------------------------------------------------------------------------
# JSONStructuralEvaluator — basic interface
# ---------------------------------------------------------------------------


class TestJSONStructuralEvaluatorInterface:
    def setup_method(self) -> None:
        self.evaluator = JSONStructuralEvaluator()

    def test_returns_correct_keys(self) -> None:
        scores = self.evaluator.score('{"a": 1}', '{"a": 1}')
        assert "json_structural_sim" in scores
        assert "json_schema_match" in scores

    def test_scores_are_floats(self) -> None:
        scores = self.evaluator.score('{"a": 1}', '{"a": 1}')
        assert isinstance(scores["json_structural_sim"], float)
        assert isinstance(scores["json_schema_match"], float)

    def test_scores_in_range(self) -> None:
        scores = self.evaluator.score('{"a": 1, "b": 2}', '{"a": 1, "c": 3}')
        assert 0.0 <= scores["json_structural_sim"] <= 1.0
        assert 0.0 <= scores["json_schema_match"] <= 1.0

    def test_non_json_expected_returns_zeros(self) -> None:
        scores = self.evaluator.score("not json", '{"a": 1}')
        assert scores == {"json_structural_sim": 0.0, "json_schema_match": 0.0}

    def test_non_json_actual_returns_zeros(self) -> None:
        scores = self.evaluator.score('{"a": 1}', "not json")
        assert scores == {"json_structural_sim": 0.0, "json_schema_match": 0.0}

    def test_both_non_json_returns_zeros(self) -> None:
        scores = self.evaluator.score("hello world", "goodbye world")
        assert scores == {"json_structural_sim": 0.0, "json_schema_match": 0.0}

    def test_empty_strings_return_zeros(self) -> None:
        scores = self.evaluator.score("", "")
        assert scores == {"json_structural_sim": 0.0, "json_schema_match": 0.0}

    def test_malformed_json_returns_zeros(self) -> None:
        scores = self.evaluator.score("{not: valid}", "{also: bad}")
        assert scores == {"json_structural_sim": 0.0, "json_schema_match": 0.0}


# ---------------------------------------------------------------------------
# Identical structures
# ---------------------------------------------------------------------------


class TestIdenticalStructures:
    def setup_method(self) -> None:
        self.evaluator = JSONStructuralEvaluator()

    def test_identical_flat_dict(self) -> None:
        data = '{"name": "Alice", "age": 30}'
        scores = self.evaluator.score(data, data)
        assert scores["json_structural_sim"] == pytest.approx(1.0)
        assert scores["json_schema_match"] == pytest.approx(1.0)

    def test_identical_nested_dict(self) -> None:
        data = '{"user": {"name": "Bob", "prefs": {"theme": "dark"}}}'
        scores = self.evaluator.score(data, data)
        assert scores["json_structural_sim"] == pytest.approx(1.0)
        assert scores["json_schema_match"] == pytest.approx(1.0)

    def test_identical_array(self) -> None:
        data = "[1, 2, 3]"
        scores = self.evaluator.score(data, data)
        assert scores["json_structural_sim"] == pytest.approx(1.0)
        assert scores["json_schema_match"] == pytest.approx(1.0)

    def test_identical_empty_dict(self) -> None:
        scores = self.evaluator.score("{}", "{}")
        assert scores["json_structural_sim"] == pytest.approx(1.0)
        assert scores["json_schema_match"] == pytest.approx(1.0)

    def test_identical_empty_array(self) -> None:
        scores = self.evaluator.score("[]", "[]")
        assert scores["json_structural_sim"] == pytest.approx(1.0)
        assert scores["json_schema_match"] == pytest.approx(1.0)

    def test_identical_scalar(self) -> None:
        scores = self.evaluator.score('"hello"', '"hello"')
        assert scores["json_structural_sim"] == pytest.approx(1.0)

    def test_identical_number(self) -> None:
        scores = self.evaluator.score("42", "42")
        assert scores["json_structural_sim"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Schema match (key-path overlap)
# ---------------------------------------------------------------------------


class TestSchemaMatch:
    def setup_method(self) -> None:
        self.evaluator = JSONStructuralEvaluator()

    def test_same_keys_different_values(self) -> None:
        expected = '{"name": "Alice", "age": 30}'
        actual = '{"name": "Bob", "age": 25}'
        scores = self.evaluator.score(expected, actual)
        # All keys match → schema_match = 1.0
        assert scores["json_schema_match"] == pytest.approx(1.0)

    def test_partial_key_overlap(self) -> None:
        expected = '{"a": 1, "b": 2, "c": 3}'
        actual = '{"a": 1, "b": 2}'
        scores = self.evaluator.score(expected, actual)
        # union={a,b,c} intersection={a,b} → 2/3
        assert scores["json_schema_match"] == pytest.approx(2 / 3)

    def test_disjoint_keys(self) -> None:
        expected = '{"x": 1}'
        actual = '{"y": 2}'
        scores = self.evaluator.score(expected, actual)
        assert scores["json_schema_match"] == pytest.approx(0.0)

    def test_superset_keys(self) -> None:
        expected = '{"a": 1}'
        actual = '{"a": 1, "b": 2, "c": 3}'
        scores = self.evaluator.score(expected, actual)
        # union={a,b,c} intersection={a} → 1/3
        assert scores["json_schema_match"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Structural similarity (value matching)
# ---------------------------------------------------------------------------


class TestStructuralSimilarity:
    def setup_method(self) -> None:
        self.evaluator = JSONStructuralEvaluator()

    def test_same_keys_same_values(self) -> None:
        scores = self.evaluator.score('{"a": 1}', '{"a": 1}')
        assert scores["json_structural_sim"] == pytest.approx(1.0)

    def test_same_keys_different_values(self) -> None:
        scores = self.evaluator.score('{"a": 1}', '{"a": 2}')
        # keys match (schema=1) but values differ → structural < 1
        assert scores["json_structural_sim"] < 1.0
        assert scores["json_structural_sim"] >= 0.0

    def test_completely_different_keys_and_values(self) -> None:
        scores = self.evaluator.score('{"a": 1}', '{"b": 2}')
        assert scores["json_structural_sim"] == pytest.approx(0.0)

    def test_type_coercion_partial_score(self) -> None:
        # int 5 vs string "5" → coerce match = 0.5
        scores = self.evaluator.score('{"x": 5}', '{"x": "5"}')
        # structural_sim = 0.5 / 1 key = 0.5
        assert scores["json_structural_sim"] == pytest.approx(0.5)

    def test_nested_dict_full_match(self) -> None:
        expected = '{"user": {"name": "Alice", "age": 30}}'
        actual = '{"user": {"name": "Alice", "age": 30}}'
        scores = self.evaluator.score(expected, actual)
        assert scores["json_structural_sim"] == pytest.approx(1.0)

    def test_nested_dict_value_mismatch(self) -> None:
        expected = '{"user": {"name": "Alice", "age": 30}}'
        actual = '{"user": {"name": "Bob", "age": 30}}'
        scores = self.evaluator.score(expected, actual)
        # inner: name mismatch, age match → sub-sim < 1, > 0
        assert 0.0 < scores["json_structural_sim"] < 1.0


# ---------------------------------------------------------------------------
# Array handling
# ---------------------------------------------------------------------------


class TestArrayHandling:
    def setup_method(self) -> None:
        self.evaluator = JSONStructuralEvaluator()

    def test_identical_arrays(self) -> None:
        scores = self.evaluator.score("[1, 2, 3]", "[1, 2, 3]")
        assert scores["json_structural_sim"] == pytest.approx(1.0)
        assert scores["json_schema_match"] == pytest.approx(1.0)

    def test_completely_different_arrays(self) -> None:
        scores = self.evaluator.score("[1, 2, 3]", "[4, 5, 6]")
        assert scores["json_structural_sim"] == pytest.approx(0.0)

    def test_partial_array_overlap(self) -> None:
        scores = self.evaluator.score("[1, 2, 3]", "[1, 2, 9]")
        # Two matches out of three max-length items
        assert 0.0 < scores["json_structural_sim"] < 1.0

    def test_empty_arrays(self) -> None:
        scores = self.evaluator.score("[]", "[]")
        assert scores["json_structural_sim"] == pytest.approx(1.0)

    def test_one_empty_one_nonempty_array(self) -> None:
        scores = self.evaluator.score("[1, 2, 3]", "[]")
        assert scores["json_structural_sim"] == pytest.approx(0.0)

    def test_arrays_different_lengths(self) -> None:
        scores = self.evaluator.score("[1, 2]", "[1, 2, 3]")
        # 2 matches out of max(2,3)=3
        assert scores["json_structural_sim"] == pytest.approx(2 / 3)

    def test_array_schema_match_uses_lcs(self) -> None:
        # [1, 2, 3] vs [2, 3] → LCS=2, total=3 → schema_match=2/3
        scores = self.evaluator.score("[1, 2, 3]", "[2, 3]")
        assert scores["json_schema_match"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def setup_method(self) -> None:
        self.evaluator = JSONStructuralEvaluator()

    def test_whitespace_stripped_before_parse(self) -> None:
        scores = self.evaluator.score('  {"a": 1}  ', '  {"a": 1}  ')
        assert scores["json_structural_sim"] == pytest.approx(1.0)

    def test_scalar_vs_dict_returns_zeros_structural(self) -> None:
        scores = self.evaluator.score('{"a": 1}', '"hello"')
        # Type mismatch at top level
        assert scores["json_structural_sim"] == pytest.approx(0.0)

    def test_dict_vs_array_returns_zeros(self) -> None:
        scores = self.evaluator.score('{"a": 1}', "[1, 2]")
        assert scores["json_structural_sim"] == pytest.approx(0.0)

    def test_no_config_required(self) -> None:
        evaluator = JSONStructuralEvaluator(config=None)
        scores = evaluator.score('{"a": 1}', '{"a": 1}')
        assert scores["json_structural_sim"] == pytest.approx(1.0)

    def test_deeply_nested_identical(self) -> None:
        data = '{"a": {"b": {"c": {"d": 42}}}}'
        scores = self.evaluator.score(data, data)
        assert scores["json_structural_sim"] == pytest.approx(1.0)
        assert scores["json_schema_match"] == pytest.approx(1.0)

    def test_boolean_json(self) -> None:
        scores = self.evaluator.score("true", "true")
        assert scores["json_structural_sim"] == pytest.approx(1.0)

    def test_null_json(self) -> None:
        scores = self.evaluator.score("null", "null")
        assert scores["json_structural_sim"] == pytest.approx(1.0)
