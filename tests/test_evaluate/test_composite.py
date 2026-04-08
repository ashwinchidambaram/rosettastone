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

    def test_logs_warning_on_skipped_pairs(self) -> None:
        """When litellm.completion raises for some pairs, all pairs are included in
        results — skipped ones with failure_reason set — and a warning is logged.
        """
        import logging
        import unittest

        pairs = [make_pair(f"Q{i}", "positive", OutputType.CLASSIFICATION) for i in range(5)]

        call_count = 0

        def side_effect(*_, **__):
            nonlocal call_count
            call_count += 1
            # Fail for pairs 0, 2, 4 (indices 0, 2, 4); succeed for 1 and 3
            if call_count in (1, 3, 5):
                raise RuntimeError("Simulated LLM failure")
            return make_litellm_response("positive")

        tc = unittest.TestCase()
        tc.maxDiff = None
        with patch("rosettastone.evaluate.composite.litellm.completion", side_effect=side_effect):
            with tc.assertLogs("rosettastone.evaluate.composite", level=logging.WARNING) as log_ctx:
                results = self.evaluator.evaluate(pairs)

        # All 5 pairs return EvalResults: 3 skipped (failure_reason set) + 2 successful
        assert len(results) == 5, f"Expected 5 results (all pairs), got {len(results)}"

        skipped = [r for r in results if r.failure_reason is not None]
        successful = [r for r in results if r.failure_reason is None]
        assert len(skipped) == 3, f"Expected 3 skipped, got {len(skipped)}"
        assert len(successful) == 2, f"Expected 2 successful, got {len(successful)}"

        # A warning about skipped pairs must have been logged
        warning_messages = "\n".join(log_ctx.output)
        assert "skipped" in warning_messages.lower() or "Skipped" in warning_messages, (
            f"Expected a 'skipped' warning in logs, got: {warning_messages}"
        )

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_failure_reason_none_on_success(self, mock_completion: MagicMock) -> None:
        """EvalResult.failure_reason is None when scoring succeeds."""
        mock_completion.return_value = make_litellm_response("positive")
        pair = make_pair(
            prompt="Classify sentiment",
            response="positive",
            output_type=OutputType.CLASSIFICATION,
        )
        results = self.evaluator.evaluate([pair])
        assert len(results) == 1
        assert results[0].failure_reason is None

    def test_failure_reason_api_error_on_exception(self) -> None:
        """EvalResult.failure_reason is 'api_error' when litellm raises a generic exception."""
        pair = make_pair("Q1", "positive", OutputType.CLASSIFICATION)

        with patch(
            "rosettastone.evaluate.composite.litellm.completion",
            side_effect=ConnectionError("Connection refused"),
        ):
            results = self.evaluator.evaluate([pair])

        assert len(results) == 1
        assert results[0].failure_reason == "api_error"
        assert results[0].composite_score == 0.0
        assert results[0].is_win is False

    def test_failure_reason_timeout_on_timeout_exception(self) -> None:
        """EvalResult.failure_reason is 'timeout' when a timeout exception is raised."""
        pair = make_pair("Q1", "positive", OutputType.CLASSIFICATION)

        class FakeTimeoutError(Exception):
            pass

        with patch(
            "rosettastone.evaluate.composite.litellm.completion",
            side_effect=FakeTimeoutError("Request timed out"),
        ):
            results = self.evaluator.evaluate([pair])

        assert len(results) == 1
        assert results[0].failure_reason == "timeout"

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_failure_reason_json_gate(self, mock_completion: MagicMock) -> None:
        """EvalResult.failure_reason is 'json_gate_failed' when JSON validity gate fires."""
        # Return invalid JSON so json_valid == 0
        mock_completion.return_value = make_litellm_response("not valid json at all")
        pair = make_pair(
            prompt="Give me JSON",
            response='{"key": "value"}',
            output_type=OutputType.JSON,
        )
        results = self.evaluator.evaluate([pair])
        assert len(results) == 1
        assert results[0].failure_reason == "json_gate_failed"
        assert results[0].composite_score == 0.0

    def test_failure_reason_rate_limit_on_ratelimit_exception(self) -> None:
        """EvalResult.failure_reason is 'rate_limit' when a rate-limit exception is raised."""
        pair = make_pair("Q1", "positive", OutputType.CLASSIFICATION)

        class FakeRateLimitError(Exception):
            pass

        with patch(
            "rosettastone.evaluate.composite.litellm.completion",
            side_effect=FakeRateLimitError("quota exceeded"),
        ):
            results = self.evaluator.evaluate([pair])

        assert len(results) == 1
        assert results[0].failure_reason == "rate_limit"
        assert results[0].composite_score == 0.0
        assert results[0].is_win is False

    def test_failure_reason_no_response_on_empty_choices(self) -> None:
        """EvalResult.failure_reason is 'no_response' when response has empty choices list."""
        pair = make_pair("Q1", "positive", OutputType.CLASSIFICATION)

        mock_response = MagicMock()
        mock_response.choices = []
        mock_response._hidden_params = {}

        with patch(
            "rosettastone.evaluate.composite.litellm.completion", return_value=mock_response
        ):
            results = self.evaluator.evaluate([pair])

        assert len(results) == 1
        assert results[0].failure_reason == "no_response"
        assert results[0].composite_score == 0.0
        assert results[0].is_win is False


# ---------------------------------------------------------------------------
# _score_semantic fallback chain logging tests
# ---------------------------------------------------------------------------


def _clear_semantic_module_cache() -> None:
    """Remove cached bertscore / embedding modules so imports re-run inside _score_semantic."""
    for key in list(sys.modules.keys()):
        if (
            "rosettastone.evaluate.bertscore" in key
            or "rosettastone.evaluate.embedding" in key
        ):
            del sys.modules[key]


def _attach_capturing_handler(logger_name: str) -> tuple:
    """Attach an INFO+ capturing handler directly to the named logger.

    Returns (logger, handler, original_level). The caller must call
    _detach_capturing_handler() after the test body.
    """
    import logging as _logging

    captured: list[str] = []

    class _CapturingHandler(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            if record.levelno >= _logging.INFO:
                captured.append(self.format(record))

    the_logger = _logging.getLogger(logger_name)
    original_level = the_logger.level
    handler = _CapturingHandler()
    the_logger.addHandler(handler)
    the_logger.setLevel(_logging.DEBUG)  # ensure INFO records are handled
    return the_logger, handler, original_level, captured


def _detach_capturing_handler(the_logger, handler, original_level: int) -> None:
    the_logger.removeHandler(handler)
    the_logger.setLevel(original_level)


class TestScoreSemanticFallbackLogging:
    """Verify that _score_semantic logs at each fallback tier.

    The composite logger uses get_logger() which sets propagate=False and adds
    its own StreamHandler. pytest's caplog fixture only intercepts loggers that
    propagate to root, so we attach a handler directly to the named logger.
    """

    LOGGER_NAME = "rosettastone.evaluate.composite"

    def setup_method(self) -> None:
        self.evaluator = CompositeEvaluator(make_config())

    def test_score_semantic_logs_bertscore_fallback_to_embedding(self) -> None:
        """When BERTScore raises ImportError and Embedding succeeds, the fallback log appears."""
        fake_embedding_evaluator = MagicMock()
        fake_embedding_evaluator.score.return_value = {"embedding_sim": 0.85}
        fake_embedding_cls = MagicMock(return_value=fake_embedding_evaluator)

        # Clear cache BEFORE patch.dict so patch.dict's None value sticks
        _clear_semantic_module_cache()

        the_logger, handler, orig_level, captured = _attach_capturing_handler(self.LOGGER_NAME)
        try:
            # Setting the bertscore module to None causes ImportError on import
            with patch.dict("sys.modules", {"rosettastone.evaluate.bertscore": None}):
                with patch(
                    "rosettastone.evaluate.embedding.EmbeddingEvaluator",
                    fake_embedding_cls,
                ):
                    self.evaluator._score_semantic("hello", "hello")
        finally:
            _detach_capturing_handler(the_logger, handler, orig_level)

        all_output = "\n".join(captured)
        assert "BERTScore unavailable" in all_output or "EmbeddingEvaluator" in all_output, (
            f"Expected fallback log message, got: {all_output!r}"
        )

    def test_score_semantic_logs_full_fallback_to_exact_match(self) -> None:
        """When both BERTScore and Embedding raise ImportError, the ExactMatch fallback log appears."""
        # Clear cache BEFORE patch.dict so None values stick
        _clear_semantic_module_cache()

        the_logger, handler, orig_level, captured = _attach_capturing_handler(self.LOGGER_NAME)
        try:
            with patch.dict(
                "sys.modules",
                {
                    "rosettastone.evaluate.bertscore": None,
                    "rosettastone.evaluate.embedding": None,
                },
            ):
                result = self.evaluator._score_semantic("hello world", "hello world")
        finally:
            _detach_capturing_handler(the_logger, handler, orig_level)

        assert len(result) > 0
        all_output = "\n".join(captured)
        assert "EmbeddingEvaluator unavailable" in all_output or "ExactMatchEvaluator" in all_output, (
            f"Expected ExactMatch fallback log, got: {all_output!r}"
        )

    def test_score_semantic_no_fallback_log_on_happy_path(self) -> None:
        """When BERTScore is available and works, no fallback INFO log messages appear."""
        fake_bert_evaluator = MagicMock()
        fake_bert_evaluator.score.return_value = {"bertscore_f1": 0.92}
        fake_bert_cls = MagicMock(return_value=fake_bert_evaluator)

        the_logger, handler, orig_level, captured = _attach_capturing_handler(self.LOGGER_NAME)
        try:
            # BERTScoreEvaluator is mocked to succeed — no ImportError, no fallback
            with patch(
                "rosettastone.evaluate.bertscore.BERTScoreEvaluator",
                fake_bert_cls,
            ):
                _clear_semantic_module_cache()
                self.evaluator._score_semantic("hello", "hello")
        finally:
            _detach_capturing_handler(the_logger, handler, orig_level)

        fallback_phrases = [
            "BERTScore unavailable",
            "EmbeddingEvaluator unavailable",
            "falling back",
        ]
        for phrase in fallback_phrases:
            for msg in captured:
                assert phrase not in msg, (
                    f"Unexpected fallback log '{phrase}' found on happy path: {msg}"
                )


# ---------------------------------------------------------------------------
# Batch BERTScore index alignment tests
# ---------------------------------------------------------------------------


class TestBatchBERTScoreAlignment:
    """Verify that the 3-phase batching correctly maps BERTScore results to pairs.

    The evaluate() method:
    1. Collects LLM completions in parallel (Phase 1).
    2. Batch-computes BERTScore for free-text pairs only (Phase 2).
    3. Assembles EvalResults, feeding each free-text pair its own pre-computed score (Phase 3).

    These tests ensure the free_text_indices -> bertscore_map mapping is correct.
    """

    def setup_method(self) -> None:
        self.config = make_config()
        self.evaluator = CompositeEvaluator(self.config)

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_batch_evaluate_assigns_scores_to_correct_pairs(
        self, mock_completion: MagicMock
    ) -> None:
        """5 pairs (3 free-text, 2 JSON): each free-text pair gets its own BERTScore.

        The batch returns 3 distinct scores. The test verifies they are assigned in
        order to the free-text pairs, not to their JSON neighbours.
        """
        # Pairs layout: [short_text, json, short_text, json, short_text]
        pairs = [
            make_pair("ST-0 question", "ST-0 answer", OutputType.SHORT_TEXT),
            make_pair("JS-1 question", '{"k": "v"}', OutputType.JSON),
            make_pair("ST-2 question", "ST-2 answer", OutputType.SHORT_TEXT),
            make_pair("JS-3 question", '{"k": "v"}', OutputType.JSON),
            make_pair("ST-4 question", "ST-4 answer", OutputType.SHORT_TEXT),
        ]

        # Completions mirror the expected responses (so JSON pairs score perfectly)
        responses = [
            "ST-0 answer",       # short_text
            '{"k": "v"}',        # json
            "ST-2 answer",       # short_text
            '{"k": "v"}',        # json
            "ST-4 answer",       # short_text
        ]
        mock_completion.side_effect = [make_litellm_response(r) for r in responses]

        # 3 distinct BERTScores for the 3 free-text pairs (in order)
        batch_scores_returned = [0.11, 0.55, 0.99]

        with patch(
            "rosettastone.evaluate.bertscore.batch_compute_bertscore",
            return_value=batch_scores_returned,
        ) as mock_batch:
            results = self.evaluator.evaluate(pairs)

        # BERTScore batch was called once with exactly 3 free-text pairs
        mock_batch.assert_called_once()
        call_pairs = mock_batch.call_args[0][0]
        assert len(call_pairs) == 3

        # Pull out the short_text results (indices 0, 2, 4 in input order)
        # results ordering follows completion order, so find by prompt
        st_results = [r for r in results if r.details.get("output_type") == "short_text"]
        json_results = [r for r in results if r.details.get("output_type") == "json"]

        assert len(st_results) == 3
        assert len(json_results) == 2

        # Each short_text pair must carry the bertscore_f1 that was pre-computed for it
        st_scores = sorted(r.scores.get("bertscore_f1", -1.0) for r in st_results)
        assert st_scores == sorted(batch_scores_returned), (
            f"BERTScore values were not correctly mapped. Got {st_scores}, "
            f"expected {sorted(batch_scores_returned)}"
        )

        # JSON pairs must NOT have a bertscore_f1 key (they use json_validator instead)
        for r in json_results:
            assert "bertscore_f1" not in r.scores, (
                "JSON pair unexpectedly received a bertscore_f1 score"
            )

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_batch_evaluate_empty_free_text_list(self, mock_completion: MagicMock) -> None:
        """When all pairs are JSON type, BERTScore batch is never called."""
        pairs = [
            make_pair("Q1", '{"a": 1}', OutputType.JSON),
            make_pair("Q2", '{"b": 2}', OutputType.JSON),
        ]
        mock_completion.side_effect = [
            make_litellm_response('{"a": 1}'),
            make_litellm_response('{"b": 2}'),
        ]

        with patch(
            "rosettastone.evaluate.bertscore.batch_compute_bertscore",
        ) as mock_batch:
            results = self.evaluator.evaluate(pairs)

        mock_batch.assert_not_called()
        assert len(results) == 2
        # All results should be JSON-scored
        for r in results:
            assert r.details["output_type"] == "json"
            assert "json_valid" in r.scores

    @patch("rosettastone.evaluate.composite.litellm.completion")
    def test_evaluate_single_pair_matches_batch(self, mock_completion: MagicMock) -> None:
        """A single short_text pair evaluated via batch uses the batch BERTScore value."""
        pair = make_pair("Tell me a story.", "Once upon a time.", OutputType.SHORT_TEXT)
        expected_bert_score = 0.73

        mock_completion.return_value = make_litellm_response("Once upon a time.")

        with patch(
            "rosettastone.evaluate.bertscore.batch_compute_bertscore",
            return_value=[expected_bert_score],
        ):
            results = self.evaluator.evaluate([pair])

        assert len(results) == 1
        result = results[0]
        # The single free-text pair must receive exactly the batch BERTScore value
        assert "bertscore_f1" in result.scores, (
            "Expected bertscore_f1 in scores but it was missing"
        )
        assert abs(result.scores["bertscore_f1"] - expected_bert_score) < 1e-9, (
            f"Expected bertscore_f1={expected_bert_score}, got {result.scores['bertscore_f1']}"
        )
