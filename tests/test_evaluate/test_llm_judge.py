"""Tests for LLMJudgeEvaluator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rosettastone.config import MigrationConfig
from rosettastone.evaluate.llm_judge import (
    LLMJudgeEvaluator,
    _build_messages,
    _normalize,
    _parse_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(judge_model: str = "openai/gpt-4o") -> MigrationConfig:
    return MigrationConfig(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        data_path=Path("/tmp/fake.jsonl"),
        judge_model=judge_model,
    )


def _mock_completion(scores: list[int]) -> MagicMock:
    """Build a mock litellm.completion that returns the given score strings in order."""
    responses = []
    for s in scores:
        choice = MagicMock()
        choice.message.content = str(s)
        resp = MagicMock()
        resp.choices = [choice]
        responses.append(resp)
    mock = MagicMock(side_effect=responses)
    return mock


# ---------------------------------------------------------------------------
# _parse_score
# ---------------------------------------------------------------------------


class TestParseScore:
    def test_bare_digit(self) -> None:
        assert _parse_score("3") == 3.0

    def test_digit_with_whitespace(self) -> None:
        assert _parse_score("  5  ") == 5.0

    def test_digit_in_sentence(self) -> None:
        assert _parse_score("I would rate this a 4 out of 5.") == 4.0

    def test_first_valid_digit_used(self) -> None:
        # "2" appears before "4"
        assert _parse_score("Score: 2 (but could be 4)") == 2.0

    def test_returns_none_for_no_digit(self) -> None:
        assert _parse_score("no score here") is None

    def test_returns_none_for_out_of_range(self) -> None:
        # "6" is not in 1-5 range
        assert _parse_score("6") is None

    def test_returns_none_for_zero(self) -> None:
        assert _parse_score("0") is None

    def test_all_valid_digits(self) -> None:
        for d in range(1, 6):
            assert _parse_score(str(d)) == float(d)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_score_one_is_zero(self) -> None:
        assert _normalize(1.0) == 0.0

    def test_score_five_is_one(self) -> None:
        assert _normalize(5.0) == 1.0

    def test_score_three_is_half(self) -> None:
        assert _normalize(3.0) == pytest.approx(0.5)

    def test_score_two(self) -> None:
        assert _normalize(2.0) == pytest.approx(0.25)

    def test_score_four(self) -> None:
        assert _normalize(4.0) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_default_order_contains_expected_label(self) -> None:
        msgs = _build_messages("exp_text", "act_text", None, flip=False)
        user_content = msgs[1]["content"]
        assert "Expected" in user_content
        assert "exp_text" in user_content

    def test_flipped_order_contains_response_first(self) -> None:
        msgs = _build_messages("exp_text", "act_text", None, flip=True)
        user_content = msgs[1]["content"]
        assert "Response" in user_content

    def test_prompt_included_when_provided(self) -> None:
        msgs = _build_messages("exp", "act", "What is 2+2?", flip=False)
        user_content = msgs[1]["content"]
        assert "What is 2+2?" in user_content

    def test_prompt_absent_when_none(self) -> None:
        msgs = _build_messages("exp", "act", None, flip=False)
        user_content = msgs[1]["content"]
        assert "Original prompt" not in user_content

    def test_system_message_present(self) -> None:
        msgs = _build_messages("e", "a", None)
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_rubric_present(self) -> None:
        msgs = _build_messages("e", "a", None)
        user_content = msgs[1]["content"]
        assert "1-5" in user_content


# ---------------------------------------------------------------------------
# LLMJudgeEvaluator.score
# ---------------------------------------------------------------------------


class TestLLMJudgeEvaluatorScore:
    def test_perfect_equivalence_both_fives(self) -> None:
        evaluator = LLMJudgeEvaluator()
        with patch("litellm.completion", _mock_completion([5, 5])):
            scores = evaluator.score("hello", "hello")
        assert "llm_judge_score" in scores
        assert scores["llm_judge_score"] == pytest.approx(1.0)

    def test_completely_different_both_ones(self) -> None:
        evaluator = LLMJudgeEvaluator()
        with patch("litellm.completion", _mock_completion([1, 1])):
            scores = evaluator.score("hello", "world")
        assert scores["llm_judge_score"] == pytest.approx(0.0)

    def test_middle_score_averaged(self) -> None:
        evaluator = LLMJudgeEvaluator()
        # First call returns 4, second returns 2; avg raw = 3, normalized = 0.5
        with patch("litellm.completion", _mock_completion([4, 2])):
            scores = evaluator.score("a", "b")
        assert scores["llm_judge_score"] == pytest.approx(0.5)

    def test_uses_config_judge_model(self) -> None:
        config = _make_config("anthropic/claude-3-haiku-20240307")
        evaluator = LLMJudgeEvaluator(config=config)
        assert evaluator._judge_model == "anthropic/claude-3-haiku-20240307"

    def test_default_judge_model_when_no_config(self) -> None:
        evaluator = LLMJudgeEvaluator()
        assert evaluator._judge_model == "openai/gpt-4o"

    def test_returns_empty_dict_on_api_error(self) -> None:
        evaluator = LLMJudgeEvaluator()
        mock = MagicMock(side_effect=RuntimeError("API down"))
        with patch("litellm.completion", mock):
            scores = evaluator.score("a", "b")
        assert scores == {}

    def test_returns_empty_dict_on_parse_failure(self) -> None:
        evaluator = LLMJudgeEvaluator()
        choice = MagicMock()
        choice.message.content = "I cannot determine a score."
        resp = MagicMock()
        resp.choices = [choice]
        mock = MagicMock(return_value=resp)
        with patch("litellm.completion", mock):
            scores = evaluator.score("a", "b")
        assert scores == {}

    def test_empty_expected_and_actual(self) -> None:
        evaluator = LLMJudgeEvaluator()
        with patch("litellm.completion", _mock_completion([3, 3])):
            scores = evaluator.score("", "")
        assert "llm_judge_score" in scores
        assert 0.0 <= scores["llm_judge_score"] <= 1.0

    def test_score_in_valid_range(self) -> None:
        evaluator = LLMJudgeEvaluator()
        with patch("litellm.completion", _mock_completion([3, 4])):
            scores = evaluator.score("some text", "similar text")
        assert 0.0 <= scores["llm_judge_score"] <= 1.0

    def test_prompt_kwarg_forwarded(self) -> None:
        """Ensure that the prompt kwarg is included in both judge calls."""
        evaluator = LLMJudgeEvaluator()
        captured_messages: list[list[dict[str, str]]] = []

        def capturing_completion(**kwargs: object) -> MagicMock:
            captured_messages.append(kwargs.get("messages", []))  # type: ignore[arg-type]
            choice = MagicMock()
            choice.message.content = "4"
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        with patch("litellm.completion", capturing_completion):
            evaluator.score("expected", "actual", prompt="What is the capital?")

        assert len(captured_messages) == 2
        for msgs in captured_messages:
            user_msg = next(m for m in msgs if m["role"] == "user")
            assert "What is the capital?" in user_msg["content"]

    def test_bidirectional_calls_made(self) -> None:
        """Verify the judge is called exactly twice."""
        evaluator = LLMJudgeEvaluator()
        mock = _mock_completion([3, 3])
        with patch("litellm.completion", mock):
            evaluator.score("a", "b")
        assert mock.call_count == 2

    def test_returns_dict_with_single_key(self) -> None:
        evaluator = LLMJudgeEvaluator()
        with patch("litellm.completion", _mock_completion([5, 5])):
            scores = evaluator.score("x", "x")
        assert list(scores.keys()) == ["llm_judge_score"]

    def test_second_api_error_returns_empty(self) -> None:
        """If the second call fails, return empty (not a partial result)."""
        evaluator = LLMJudgeEvaluator()
        choice = MagicMock()
        choice.message.content = "4"
        resp = MagicMock()
        resp.choices = [choice]
        mock = MagicMock(side_effect=[resp, RuntimeError("timeout")])
        with patch("litellm.completion", mock):
            scores = evaluator.score("a", "b")
        assert scores == {}
