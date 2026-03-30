"""Tests for the improvement module — behavioral cloning + improvement mode.

The improvement module provides:
- build_improvement_scorer: LLM-as-judge scoring against improvement objectives
- build_improvement_feedback: merge improvement feedback into GEPA's metric feedback
- compute_blended_score: blend equivalence and improvement scores
- ImprovementObjective / ImprovementScore dataclasses

All scorer tests mock litellm.completion — no real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rosettastone.optimize.improvement import (
    ImprovementObjective,
    ImprovementScore,
    build_improvement_feedback,
    build_improvement_scorer,
    compute_blended_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_LLM_RESPONSE_TEMPLATE = "Score: {score}\nFeedback: {feedback}"


def _make_litellm_response(text: str) -> MagicMock:
    """Create a mock litellm completion response with the given text content."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    return response


# ---------------------------------------------------------------------------
# Scorer tests (10)
# ---------------------------------------------------------------------------


class TestBuildImprovementScorer:
    """build_improvement_scorer must return a callable that produces ImprovementScore list."""

    def test_returns_list_of_improvement_score(self) -> None:
        """Scorer callable must return a list of ImprovementScore objects."""
        mock_resp = _make_litellm_response("Score: 4\nFeedback: Good conciseness.")
        with patch(
            "rosettastone.optimize.improvement._call_litellm_completion",
            return_value=mock_resp,
        ):
            scorer = build_improvement_scorer(["be more concise"])
            results = scorer("prompt", "expected", "actual")

        assert isinstance(results, list)
        assert all(isinstance(r, ImprovementScore) for r in results)

    def test_one_score_per_objective(self) -> None:
        """Scorer must return exactly one ImprovementScore per objective."""
        objectives = ["be concise", "add examples", "improve tone"]
        mock_resp = _make_litellm_response("Score: 3\nFeedback: Decent.")
        with patch(
            "rosettastone.optimize.improvement._call_litellm_completion",
            return_value=mock_resp,
        ):
            scorer = build_improvement_scorer(objectives)
            results = scorer("prompt", "expected", "actual")

        assert len(results) == len(objectives)

    def test_scores_in_unit_interval(self) -> None:
        """All scores must be in [0.0, 1.0] after normalization."""
        mock_resp = _make_litellm_response("Score: 4\nFeedback: Good.")
        with patch(
            "rosettastone.optimize.improvement._call_litellm_completion",
            return_value=mock_resp,
        ):
            scorer = build_improvement_scorer(["be concise"])
            results = scorer("prompt", "expected", "actual")

        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} outside [0, 1]"

    def test_non_empty_feedback_strings(self) -> None:
        """Every ImprovementScore must have non-empty feedback."""
        mock_resp = _make_litellm_response("Score: 3\nFeedback: Needs more examples.")
        with patch(
            "rosettastone.optimize.improvement._call_litellm_completion",
            return_value=mock_resp,
        ):
            scorer = build_improvement_scorer(["add examples"])
            results = scorer("prompt", "expected", "actual")

        for r in results:
            assert isinstance(r.feedback, str)
            assert len(r.feedback) > 0

    def test_litellm_called_with_correct_model(self) -> None:
        """litellm.completion must be called with the judge_model."""
        mock_resp = _make_litellm_response("Score: 4\nFeedback: Good.")
        with patch(
            "rosettastone.optimize.improvement._call_litellm_completion",
            return_value=mock_resp,
        ) as mock_completion:
            scorer = build_improvement_scorer(
                ["be concise"], judge_model="anthropic/claude-sonnet-4"
            )
            scorer("prompt", "expected", "actual")

        # Every call should use the specified model
        for c in mock_completion.call_args_list:
            assert c.kwargs.get("model") == "anthropic/claude-sonnet-4"

    def test_handles_litellm_failure_gracefully(self) -> None:
        """If litellm.completion raises, scorer returns score=0.5 with error feedback."""
        with patch(
            "rosettastone.optimize.improvement._call_litellm_completion",
            side_effect=Exception("API connection failed"),
        ):
            scorer = build_improvement_scorer(["be concise"])
            results = scorer("prompt", "expected", "actual")

        assert len(results) == 1
        assert results[0].score == pytest.approx(0.5)
        assert "error" in results[0].feedback.lower() or "fail" in results[0].feedback.lower()

    def test_handles_unparseable_llm_response(self) -> None:
        """If the LLM response can't be parsed, return score=0.5 with error feedback."""
        mock_resp = _make_litellm_response("I cannot rate this.")
        with patch(
            "rosettastone.optimize.improvement._call_litellm_completion",
            return_value=mock_resp,
        ):
            scorer = build_improvement_scorer(["be concise"])
            results = scorer("prompt", "expected", "actual")

        assert len(results) == 1
        assert results[0].score == pytest.approx(0.5)

    def test_single_objective_works(self) -> None:
        """Scorer works correctly with a single objective."""
        mock_resp = _make_litellm_response("Score: 5\nFeedback: Excellent.")
        with patch(
            "rosettastone.optimize.improvement._call_litellm_completion",
            return_value=mock_resp,
        ):
            scorer = build_improvement_scorer(["be concise"])
            results = scorer("prompt", "expected", "actual")

        assert len(results) == 1
        assert results[0].objective == "be concise"
        # Score 5 -> (5-1)/4 = 1.0
        assert results[0].score == pytest.approx(1.0)

    def test_multiple_objectives_work(self) -> None:
        """Scorer handles 3+ objectives correctly."""
        objectives = [
            "be more concise",
            "add examples",
            "improve tone",
            "use simpler words",
        ]
        mock_resp = _make_litellm_response("Score: 3\nFeedback: Acceptable.")
        with patch(
            "rosettastone.optimize.improvement._call_litellm_completion",
            return_value=mock_resp,
        ):
            scorer = build_improvement_scorer(objectives)
            results = scorer("prompt", "expected", "actual")

        assert len(results) == 4
        result_objectives = [r.objective for r in results]
        for obj in objectives:
            assert obj in result_objectives

    def test_prompt_expected_actual_passed_to_llm(self) -> None:
        """The prompt, expected, and actual must appear in the LLM call messages."""
        mock_resp = _make_litellm_response("Score: 4\nFeedback: Good.")
        with patch(
            "rosettastone.optimize.improvement._call_litellm_completion",
            return_value=mock_resp,
        ) as mock_completion:
            scorer = build_improvement_scorer(["be concise"])
            scorer(
                "What is Python?",
                "Python is a programming language.",
                "Python is a snake and a language.",
            )

        # Check that the actual response appears in the messages
        call_args = mock_completion.call_args_list[0]
        messages = call_args.kwargs.get("messages", [])
        messages_text = " ".join(m.get("content", "") for m in messages)
        assert "Python is a snake and a language." in messages_text


# ---------------------------------------------------------------------------
# Feedback tests (6)
# ---------------------------------------------------------------------------


class TestBuildImprovementFeedback:
    """build_improvement_feedback merges improvement feedback into GEPA's base feedback."""

    def test_combined_feedback_includes_base_and_improvement(self) -> None:
        """Output must contain both the base feedback and improvement section."""
        scores = [ImprovementScore(objective="be concise", score=0.75, feedback="Good brevity.")]
        result = build_improvement_feedback("Base: Good match (0.9)", scores)

        assert "Base: Good match (0.9)" in result
        assert "improvement" in result.lower() or "objective" in result.lower()

    def test_empty_improvement_scores_returns_base_unchanged(self) -> None:
        """With no improvement scores, return the base feedback as-is."""
        result = build_improvement_feedback("Base feedback only.", [])
        assert result == "Base feedback only."

    def test_each_objective_description_appears(self) -> None:
        """Every objective description must appear in the output."""
        scores = [
            ImprovementScore(objective="be concise", score=0.8, feedback="Good."),
            ImprovementScore(objective="add examples", score=0.6, feedback="Needs work."),
        ]
        result = build_improvement_feedback("Base feedback.", scores)

        assert "be concise" in result
        assert "add examples" in result

    def test_per_objective_score_value_appears(self) -> None:
        """The numeric score for each objective must appear in the output."""
        scores = [
            ImprovementScore(objective="be concise", score=0.75, feedback="Good."),
        ]
        result = build_improvement_feedback("Base.", scores)

        assert "0.75" in result

    def test_improvement_suggestions_included(self) -> None:
        """The feedback text from each score must appear in the output."""
        scores = [
            ImprovementScore(
                objective="be concise",
                score=0.6,
                feedback="Response is too verbose. Cut filler words.",
            ),
        ]
        result = build_improvement_feedback("Base.", scores)

        assert "Response is too verbose" in result

    def test_output_has_gepa_readable_structure(self) -> None:
        """Output must have clear section markers for GEPA's reflective optimization."""
        scores = [
            ImprovementScore(objective="be concise", score=0.8, feedback="Good brevity."),
            ImprovementScore(objective="add examples", score=0.5, feedback="Missing examples."),
        ]
        result = build_improvement_feedback("Base feedback.", scores)

        # Should have some structure — section headers, separators, etc.
        assert "\n" in result, "Multi-section feedback should span multiple lines"
        # The improvement section should be visually distinct
        lines = result.strip().split("\n")
        assert len(lines) >= 3, "Expected at least 3 lines for structured GEPA feedback"


# ---------------------------------------------------------------------------
# Blended score tests (8)
# ---------------------------------------------------------------------------


class TestComputeBlendedScore:
    """compute_blended_score blends equivalence and improvement scores."""

    def test_default_weight_30_percent(self) -> None:
        """Default weight=0.3: 70% equivalence + 30% improvement."""
        scores = [ImprovementScore(objective="obj", score=0.8, feedback="Good.")]
        result = compute_blended_score(1.0, scores)
        # (1 - 0.3) * 1.0 + 0.3 * 0.8 = 0.7 + 0.24 = 0.94
        assert result == pytest.approx(0.94, abs=0.001)

    def test_custom_weight(self) -> None:
        """Custom weight=0.5 gives 50/50 blend."""
        scores = [ImprovementScore(objective="obj", score=0.6, feedback="Ok.")]
        result = compute_blended_score(0.8, scores, improvement_weight=0.5)
        # 0.5 * 0.8 + 0.5 * 0.6 = 0.4 + 0.3 = 0.7
        assert result == pytest.approx(0.7, abs=0.001)

    def test_weight_zero_returns_pure_equivalence(self) -> None:
        """improvement_weight=0.0 returns the equivalence score unmodified."""
        scores = [ImprovementScore(objective="obj", score=0.2, feedback="Bad.")]
        result = compute_blended_score(0.9, scores, improvement_weight=0.0)
        assert result == pytest.approx(0.9, abs=0.001)

    def test_weight_one_returns_pure_improvement(self) -> None:
        """improvement_weight=1.0 returns the average improvement score."""
        scores = [ImprovementScore(objective="obj", score=0.6, feedback="Ok.")]
        result = compute_blended_score(0.9, scores, improvement_weight=1.0)
        assert result == pytest.approx(0.6, abs=0.001)

    def test_result_always_in_unit_interval(self) -> None:
        """Result must always be in [0.0, 1.0] even with extreme inputs."""
        scores = [ImprovementScore(objective="obj", score=1.0, feedback="Perfect.")]
        # Even with edge values
        result = compute_blended_score(1.0, scores, improvement_weight=0.5)
        assert 0.0 <= result <= 1.0

        result = compute_blended_score(0.0, scores, improvement_weight=0.5)
        assert 0.0 <= result <= 1.0

    def test_perfect_equivalence_zero_improvement(self) -> None:
        """Perfect equivalence (1.0) + zero improvement (0.0)."""
        scores = [ImprovementScore(objective="obj", score=0.0, feedback="Failed.")]
        result = compute_blended_score(1.0, scores)
        # (1 - 0.3) * 1.0 + 0.3 * 0.0 = 0.7
        assert result == pytest.approx(0.7, abs=0.001)

    def test_zero_equivalence_perfect_improvement(self) -> None:
        """Zero equivalence (0.0) + perfect improvement (1.0)."""
        scores = [ImprovementScore(objective="obj", score=1.0, feedback="Great.")]
        result = compute_blended_score(0.0, scores)
        # (1 - 0.3) * 0.0 + 0.3 * 1.0 = 0.3
        assert result == pytest.approx(0.3, abs=0.001)

    def test_multiple_improvement_scores_averaged(self) -> None:
        """Multiple improvement scores should be averaged before blending."""
        scores = [
            ImprovementScore(objective="concise", score=0.8, feedback="Good."),
            ImprovementScore(objective="examples", score=0.4, feedback="Needs work."),
            ImprovementScore(objective="tone", score=0.6, feedback="Ok."),
        ]
        result = compute_blended_score(1.0, scores)
        # avg improvement = (0.8 + 0.4 + 0.6) / 3 = 0.6
        # (1 - 0.3) * 1.0 + 0.3 * 0.6 = 0.7 + 0.18 = 0.88
        assert result == pytest.approx(0.88, abs=0.001)


# ---------------------------------------------------------------------------
# Objective dataclass tests (6)
# ---------------------------------------------------------------------------


class TestImprovementObjective:
    """ImprovementObjective dataclass."""

    def test_default_weight_is_0_3(self) -> None:
        """Default weight should be 0.3."""
        obj = ImprovementObjective(description="be concise")
        assert obj.weight == pytest.approx(0.3)

    def test_custom_weight_preserved(self) -> None:
        """Custom weight value is preserved."""
        obj = ImprovementObjective(description="be concise", weight=0.7)
        assert obj.weight == pytest.approx(0.7)

    def test_common_objectives(self) -> None:
        """Common improvement objectives can be created."""
        objectives = [
            ImprovementObjective(description="be more concise"),
            ImprovementObjective(description="add examples"),
            ImprovementObjective(description="improve tone"),
        ]
        assert len(objectives) == 3
        assert objectives[0].description == "be more concise"
        assert objectives[1].description == "add examples"
        assert objectives[2].description == "improve tone"

    def test_empty_objectives_list_scorer_returns_empty(self) -> None:
        """Scorer with empty objectives list returns empty list."""
        scorer = build_improvement_scorer([])
        results = scorer("prompt", "expected", "actual")
        assert results == []

    def test_description_preserved(self) -> None:
        """Description field is preserved on the dataclass."""
        obj = ImprovementObjective(description="reduce verbosity")
        assert obj.description == "reduce verbosity"

    def test_can_create_without_optional_params(self) -> None:
        """Only description is required — weight is optional."""
        obj = ImprovementObjective(description="test objective")
        assert obj.description == "test objective"
        assert hasattr(obj, "weight")
