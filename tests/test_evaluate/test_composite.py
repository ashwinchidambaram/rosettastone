"""Tests for CompositeEvaluator."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from rosettastone.config import MigrationConfig
from rosettastone.core.types import EvalResult, OutputType, PromptPair
from rosettastone.evaluate.composite import DEFAULT_WIN_THRESHOLDS, CompositeEvaluator

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
        assert self.evaluator._composite_score({}, OutputType.SHORT_TEXT) == 0.0

    def test_single_score_returns_itself(self) -> None:
        # A metric with default weight 1.0
        assert self.evaluator._composite_score({"custom": 0.75}, OutputType.SHORT_TEXT) == 0.75

    def test_all_zeros(self) -> None:
        assert self.evaluator._composite_score({"a": 0.0, "b": 0.0}, OutputType.SHORT_TEXT) == 0.0

    def test_all_ones(self) -> None:
        assert self.evaluator._composite_score({"a": 1.0, "b": 1.0}, OutputType.SHORT_TEXT) == 1.0

    def test_json_gating_invalid_json_returns_zero(self) -> None:
        """json_valid == 0.0 gates the composite score to 0.0 for JSON output."""
        scores = {"json_valid": 0.0, "json_field_match": 0.8}
        assert self.evaluator._composite_score(scores, OutputType.JSON) == 0.0

    def test_json_valid_excluded_from_weighted_average(self) -> None:
        """json_valid is a gating metric — not included in the weighted average."""
        scores = {"json_valid": 1.0, "json_field_match": 0.6}
        composite = self.evaluator._composite_score(scores, OutputType.JSON)
        # json_field_match has weight 0.4, so composite = 0.6 * 0.4 / 0.4 = 0.6
        assert abs(composite - 0.6) < 1e-9

    def test_weighted_average_with_known_metrics(self) -> None:
        """exact_match (weight 0.7) and string_similarity (weight 0.3) give weighted composite."""
        scores = {"exact_match": 1.0, "string_similarity": 0.5}
        composite = self.evaluator._composite_score(scores, OutputType.CLASSIFICATION)
        expected = (1.0 * 0.7 + 0.5 * 0.3) / (0.7 + 0.3)
        assert abs(composite - expected) < 1e-9


# ---------------------------------------------------------------------------
# Win thresholds
# ---------------------------------------------------------------------------


class TestWinThreshold:
    def test_default_thresholds_exist(self) -> None:
        assert "json" in DEFAULT_WIN_THRESHOLDS
        assert "classification" in DEFAULT_WIN_THRESHOLDS
        assert "short_text" in DEFAULT_WIN_THRESHOLDS
        assert "long_text" in DEFAULT_WIN_THRESHOLDS

    def test_json_threshold_is_strictest(self) -> None:
        assert DEFAULT_WIN_THRESHOLDS["json"] > DEFAULT_WIN_THRESHOLDS["long_text"]

    def test_evaluator_uses_config_thresholds(self) -> None:
        config = make_config()
        config_dict = config.model_dump()
        config_dict["win_thresholds"] = {"classification": 0.99}
        config2 = MigrationConfig(**config_dict)
        evaluator = CompositeEvaluator(config2)
        threshold = evaluator._get_threshold(OutputType.CLASSIFICATION)
        assert threshold == 0.99


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
        """JSON PromptPair -> EvalResult scores contain json_valid and json_field_match."""
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
        """CLASSIFICATION PromptPair -> EvalResult scores contain exact_match and string_similarity."""
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
        """is_win is True when composite score >= threshold for output type."""
        # Perfect classification match -> composite = 1.0
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
        """is_win is False when composite score < threshold."""
        # Completely wrong classification -> composite well below 0.9
        mock_completion.return_value = make_litellm_response("XYZ_VERY_DIFFERENT_RESPONSE")
        pair = make_pair(
            prompt="Classify sentiment",
            response="positive",
            output_type=OutputType.CLASSIFICATION,
        )
        results = self.evaluator.evaluate([pair])
        threshold = DEFAULT_WIN_THRESHOLDS["classification"]
        assert results[0].composite_score < threshold
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
        """Identical JSON -> json_field_match == 1.0."""
        mock_completion.return_value = make_litellm_response('{"status": "ok"}')
        pair = make_pair(
            prompt="Give me JSON",
            response='{"status": "ok"}',
            output_type=OutputType.JSON,
        )
        results = self.evaluator.evaluate([pair])
        assert results[0].scores["json_field_match"] == 1.0

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_details_contain_output_type(self, mock_completion: MagicMock) -> None:
        """EvalResult details include output_type and threshold."""
        mock_completion.return_value = make_litellm_response('{"k": "v"}')
        pair = make_pair(
            prompt="JSON?",
            response='{"k": "v"}',
            output_type=OutputType.JSON,
        )
        results = self.evaluator.evaluate([pair])
        result = results[0]
        assert result.details["output_type"] == "json"
        assert "threshold" in result.details

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_progress_callback_called(self, mock_completion: MagicMock) -> None:
        """on_progress callback is called for each pair."""
        mock_completion.return_value = make_litellm_response("positive")
        progress_calls: list[tuple[int, int]] = []

        def on_progress(current: int, total: int) -> None:
            progress_calls.append((current, total))

        evaluator = CompositeEvaluator(make_config(), on_progress=on_progress)
        pairs = [
            make_pair("Q1", "positive", OutputType.CLASSIFICATION),
            make_pair("Q2", "negative", OutputType.CLASSIFICATION),
        ]
        evaluator.evaluate(pairs)
        assert progress_calls == [(1, 2), (2, 2)]
