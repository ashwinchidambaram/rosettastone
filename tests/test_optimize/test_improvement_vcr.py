"""VCR-enabled tests for ImprovementScorer — replays recorded API responses."""

from __future__ import annotations

import pytest


@pytest.mark.vcr
def test_improvement_scorer_returns_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """ImprovementScorer correctly parses a score from a recorded API response.

    The cassette returns "Score: 5\\nFeedback: The response is extremely concise..."
    so the expected normalized score is (5-1)/4 = 1.0.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")

    from rosettastone.optimize.improvement import build_improvement_scorer

    scorer = build_improvement_scorer(
        objectives=["Be concise and clear"],
        judge_model="openai/gpt-4o",
    )
    results = scorer(
        prompt="What is 2 + 2?",
        expected_response="The answer is 4.",
        actual_response="4",
    )

    assert len(results) == 1
    result = results[0]
    assert result.objective == "Be concise and clear"
    assert 0.0 <= result.score <= 1.0
    # Cassette returns "Score: 5", normalized = (5-1)/4 = 1.0
    assert result.score == pytest.approx(1.0)
    assert len(result.feedback) > 0
