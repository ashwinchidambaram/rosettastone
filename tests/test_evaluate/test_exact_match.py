"""Tests for ExactMatchEvaluator and string_similarity."""

from __future__ import annotations

from rosettastone.evaluate.exact_match import ExactMatchEvaluator, string_similarity


class TestStringSimilarity:
    def test_identical_strings(self) -> None:
        assert string_similarity("hello", "hello") == 1.0

    def test_completely_different_strings(self) -> None:
        score = string_similarity("abc", "xyz")
        assert score == 0.0

    def test_similar_strings(self) -> None:
        score = string_similarity("hello world", "hello there")
        assert 0.0 < score < 1.0

    def test_case_sensitive(self) -> None:
        # string_similarity is case-sensitive (uses raw strings)
        score = string_similarity("Hello", "hello")
        assert score < 1.0

    def test_whitespace_stripped(self) -> None:
        # string_similarity strips leading/trailing whitespace
        assert string_similarity("  hello  ", "hello") == 1.0

    def test_empty_strings(self) -> None:
        assert string_similarity("", "") == 1.0

    def test_one_empty_string(self) -> None:
        score = string_similarity("hello", "")
        assert score == 0.0


class TestExactMatchEvaluator:
    def setup_method(self) -> None:
        self.evaluator = ExactMatchEvaluator()

    def test_identical_strings_returns_exact_match_one(self) -> None:
        scores = self.evaluator.score("Paris", "Paris")
        assert scores["exact_match"] == 1.0

    def test_identical_strings_returns_string_similarity_one(self) -> None:
        scores = self.evaluator.score("Paris", "Paris")
        assert scores["string_similarity"] == 1.0

    def test_case_insensitive_exact_match(self) -> None:
        # "paris" vs "PARIS" should match (case-insensitive)
        scores = self.evaluator.score("PARIS", "paris")
        assert scores["exact_match"] == 1.0

    def test_case_difference_reduces_string_similarity(self) -> None:
        # string_similarity is case-sensitive, so "PARIS" vs "paris" < 1.0
        scores = self.evaluator.score("PARIS", "paris")
        assert scores["string_similarity"] < 1.0

    def test_completely_different_strings(self) -> None:
        scores = self.evaluator.score("Paris", "London")
        assert scores["exact_match"] == 0.0

    def test_completely_different_string_similarity(self) -> None:
        scores = self.evaluator.score("Paris", "London")
        assert 0.0 <= scores["string_similarity"] < 1.0

    def test_whitespace_differences_exact_match(self) -> None:
        # Leading/trailing whitespace should be stripped
        scores = self.evaluator.score("  Paris  ", "Paris")
        assert scores["exact_match"] == 1.0

    def test_returns_correct_keys(self) -> None:
        scores = self.evaluator.score("yes", "no")
        assert "exact_match" in scores
        assert "string_similarity" in scores

    def test_scores_are_floats(self) -> None:
        scores = self.evaluator.score("yes", "no")
        assert isinstance(scores["exact_match"], float)
        assert isinstance(scores["string_similarity"], float)

    def test_scores_in_valid_range(self) -> None:
        scores = self.evaluator.score("some text here", "some other text")
        assert 0.0 <= scores["exact_match"] <= 1.0
        assert 0.0 <= scores["string_similarity"] <= 1.0

    def test_partial_match_no_exact_match(self) -> None:
        scores = self.evaluator.score("positive", "positive sentiment")
        assert scores["exact_match"] == 0.0
        assert scores["string_similarity"] > 0.0

    def test_mixed_case_with_whitespace(self) -> None:
        scores = self.evaluator.score("  YES  ", "yes")
        assert scores["exact_match"] == 1.0
