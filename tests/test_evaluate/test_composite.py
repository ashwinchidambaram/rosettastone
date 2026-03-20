"""Tests for CompositeEvaluator."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from rosettastone.config import MigrationConfig
from rosettastone.core.types import EvalResult, OutputType, PromptPair
from rosettastone.evaluate.composite import WIN_THRESHOLD, CompositeEvaluator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config() -> MigrationConfig:
    return MigrationConfig(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        data_path=Path("/tmp/fake.jsonl"),
    )


def make_pair(
    prompt: str,
    response: str,
    output_type: OutputType | None = None,
) -> PromptPair:
    return PromptPair(
        prompt=prompt,
        response=response,
        source_model="openai/gpt-4o",
        output_type=output_type,
    )


def make_litellm_response(content: str) -> MagicMock:
    """Return a mock object shaped like a litellm.completion() response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# _composite_score tests (pure unit — no mocks needed)
# ---------------------------------------------------------------------------


class TestCompositeScore:
    def setup_method(self) -> None:
        self.evaluator = CompositeEvaluator(make_config())

    def test_empty_scores_returns_zero(self) -> None:
        assert self.evaluator._composite_score({}) == 0.0

    def test_average_of_two_scores(self) -> None:
        result = self.evaluator._composite_score({"a": 0.8, "b": 0.9})
        assert abs(result - 0.85) < 1e-9

    def test_single_score_returns_itself(self) -> None:
        assert self.evaluator._composite_score({"x": 0.75}) == 0.75

    def test_all_zeros(self) -> None:
        assert self.evaluator._composite_score({"a": 0.0, "b": 0.0}) == 0.0

    def test_all_ones(self) -> None:
        assert self.evaluator._composite_score({"a": 1.0, "b": 1.0}) == 1.0

    def test_average_of_three_scores(self) -> None:
        result = self.evaluator._composite_score({"a": 0.6, "b": 0.9, "c": 0.3})
        assert abs(result - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# WIN_THRESHOLD
# ---------------------------------------------------------------------------


class TestWinThreshold:
    def test_win_threshold_value(self) -> None:
        assert WIN_THRESHOLD == 0.8

    def test_composite_above_threshold_is_win(self) -> None:
        evaluator = CompositeEvaluator(make_config())
        # composite of 0.81 → True
        scores = {"a": 0.81}
        composite = evaluator._composite_score(scores)
        assert composite >= WIN_THRESHOLD

    def test_composite_below_threshold_is_not_win(self) -> None:
        evaluator = CompositeEvaluator(make_config())
        # composite of 0.79 → False
        scores = {"a": 0.79}
        composite = evaluator._composite_score(scores)
        assert composite < WIN_THRESHOLD

    def test_composite_exactly_at_threshold_is_win(self) -> None:
        evaluator = CompositeEvaluator(make_config())
        scores = {"a": 0.8}
        composite = evaluator._composite_score(scores)
        assert composite >= WIN_THRESHOLD


# ---------------------------------------------------------------------------
# _score routing tests
# ---------------------------------------------------------------------------


class TestScoreRouting:
    def setup_method(self) -> None:
        self.evaluator = CompositeEvaluator(make_config())

    def test_json_output_type_uses_json_evaluator(self) -> None:
        scores = self.evaluator._score('{"name": "Alice"}', '{"name": "Alice"}', OutputType.JSON)
        assert "json_valid" in scores
        assert "json_field_match" in scores

    def test_classification_output_type_uses_exact_match(self) -> None:
        scores = self.evaluator._score("positive", "positive", OutputType.CLASSIFICATION)
        assert "exact_match" in scores
        assert "string_similarity" in scores

    def test_short_text_falls_back_without_optional_deps(self) -> None:
        # With no bert_score or sentence_transformers installed,
        # falls back to ExactMatchEvaluator
        with (
            patch.dict("sys.modules", {"bert_score": None}),
            patch.dict("sys.modules", {"sentence_transformers": None}),
        ):
            for key in list(sys.modules.keys()):
                if (
                    "rosettastone.evaluate.bertscore" in key
                    or "rosettastone.evaluate.embedding" in key
                ):
                    del sys.modules[key]
            scores = self.evaluator._score(
                "This is a short text.", "This is similar text.", OutputType.SHORT_TEXT
            )
        # Should have some metric key
        assert len(scores) > 0

    def test_none_output_type_auto_detected(self) -> None:
        # JSON response → auto-detected as JSON → json_valid key present
        scores = self.evaluator._score('{"key": "val"}', '{"key": "val"}', None)
        assert "json_valid" in scores

    def test_long_text_falls_back_without_optional_deps(self) -> None:
        with (
            patch.dict("sys.modules", {"bert_score": None}),
            patch.dict("sys.modules", {"sentence_transformers": None}),
        ):
            for key in list(sys.modules.keys()):
                if (
                    "rosettastone.evaluate.bertscore" in key
                    or "rosettastone.evaluate.embedding" in key
                ):
                    del sys.modules[key]
            long_text = " ".join(["word"] * 60)
            scores = self.evaluator._score(long_text, long_text, OutputType.LONG_TEXT)
        assert len(scores) > 0


# ---------------------------------------------------------------------------
# evaluate() integration tests (mock litellm)
# ---------------------------------------------------------------------------


class TestCompositeEvaluatorEvaluate:
    def setup_method(self) -> None:
        self.config = make_config()
        self.evaluator = CompositeEvaluator(self.config)

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_json_input_produces_json_evaluator_scores(self, mock_completion: MagicMock) -> None:
        """JSON PromptPair → EvalResult scores contain json_valid and json_field_match."""
        mock_completion.return_value = make_litellm_response('{"name": "Alice", "age": 30}')

        pair = make_pair(
            prompt='Return JSON: {"name": "Alice", "age": 30}',
            response='{"name": "Alice", "age": 30}',
            output_type=OutputType.JSON,
        )
        results = self.evaluator.evaluate([pair])

        assert len(results) == 1
        scores = results[0].scores
        assert "json_valid" in scores
        assert "json_field_match" in scores

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_classification_input_produces_exact_match_scores(
        self, mock_completion: MagicMock
    ) -> None:
        """CLASSIFICATION PromptPair → EvalResult scores contain exact_match and string_similarity."""
        mock_completion.return_value = make_litellm_response("positive")

        pair = make_pair(
            prompt="Is this positive or negative?",
            response="positive",
            output_type=OutputType.CLASSIFICATION,
        )
        results = self.evaluator.evaluate([pair])

        assert len(results) == 1
        scores = results[0].scores
        assert "exact_match" in scores
        assert "string_similarity" in scores

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_is_win_true_when_composite_at_threshold(self, mock_completion: MagicMock) -> None:
        """is_win is True when composite score >= WIN_THRESHOLD."""
        # Perfect classification match → composite = 1.0
        mock_completion.return_value = make_litellm_response("positive")
        pair = make_pair(
            prompt="Classify sentiment",
            response="positive",
            output_type=OutputType.CLASSIFICATION,
        )
        results = self.evaluator.evaluate([pair])
        assert results[0].is_win is True

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_is_win_false_when_composite_below_threshold(self, mock_completion: MagicMock) -> None:
        """is_win is False when composite score < WIN_THRESHOLD."""
        # Completely wrong classification → composite = 0.0
        mock_completion.return_value = make_litellm_response("XYZ_VERY_DIFFERENT_RESPONSE")
        pair = make_pair(
            prompt="Classify sentiment",
            response="positive",
            output_type=OutputType.CLASSIFICATION,
        )
        results = self.evaluator.evaluate([pair])
        # exact_match=0.0, string_similarity likely very low → composite < 0.8
        assert results[0].composite_score < WIN_THRESHOLD
        assert results[0].is_win is False

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_eval_result_has_correct_structure(self, mock_completion: MagicMock) -> None:
        """EvalResult contains all required fields."""
        mock_completion.return_value = make_litellm_response("Paris")
        pair = make_pair(
            prompt="Capital of France?",
            response="Paris",
            output_type=OutputType.CLASSIFICATION,
        )
        results = self.evaluator.evaluate([pair])

        result = results[0]
        assert isinstance(result, EvalResult)
        assert result.new_response == "Paris"
        assert isinstance(result.composite_score, float)
        assert isinstance(result.is_win, bool)
        assert isinstance(result.scores, dict)

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_optimized_prompt_prepended_as_system_message(self, mock_completion: MagicMock) -> None:
        """optimized_prompt is added as a system message before user messages."""
        mock_completion.return_value = make_litellm_response("positive")
        pair = make_pair(
            prompt="Classify sentiment",
            response="positive",
            output_type=OutputType.CLASSIFICATION,
        )
        self.evaluator.evaluate([pair], optimized_prompt="You are a classifier.")

        call_args = mock_completion.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][1]
        # First message should be the system message
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a classifier."

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_multiple_pairs_all_evaluated(self, mock_completion: MagicMock) -> None:
        """All pairs in test_set are evaluated."""
        mock_completion.return_value = make_litellm_response("positive")
        pairs = [
            make_pair("Q1", "positive", OutputType.CLASSIFICATION),
            make_pair("Q2", "negative", OutputType.CLASSIFICATION),
            make_pair("Q3", "neutral", OutputType.CLASSIFICATION),
        ]
        results = self.evaluator.evaluate(pairs)
        assert len(results) == 3

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_empty_test_set_returns_empty_list(self, mock_completion: MagicMock) -> None:
        results = self.evaluator.evaluate([])
        assert results == []
        mock_completion.assert_not_called()

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_litellm_returns_none_content_handled(self, mock_completion: MagicMock) -> None:
        """When litellm returns None content, new_response is empty string."""
        choice = MagicMock()
        choice.message.content = None
        response = MagicMock()
        response.choices = [choice]
        mock_completion.return_value = response

        pair = make_pair(
            prompt="Question",
            response="positive",
            output_type=OutputType.CLASSIFICATION,
        )
        results = self.evaluator.evaluate([pair])
        assert results[0].new_response == ""

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_json_field_match_perfect_when_identical(self, mock_completion: MagicMock) -> None:
        """Identical JSON → json_field_match == 1.0."""
        mock_completion.return_value = make_litellm_response('{"status": "ok"}')
        pair = make_pair(
            prompt="Give me JSON",
            response='{"status": "ok"}',
            output_type=OutputType.JSON,
        )
        results = self.evaluator.evaluate([pair])
        assert results[0].scores["json_field_match"] == 1.0

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_composite_score_is_average_of_sub_scores(self, mock_completion: MagicMock) -> None:
        """composite_score == mean of all score values."""
        mock_completion.return_value = make_litellm_response('{"k": "v"}')
        pair = make_pair(
            prompt="JSON?",
            response='{"k": "v"}',
            output_type=OutputType.JSON,
        )
        results = self.evaluator.evaluate([pair])
        result = results[0]
        expected_composite = sum(result.scores.values()) / len(result.scores)
        assert abs(result.composite_score - expected_composite) < 1e-9
