"""VCR-enabled tests for LLM judge — replays recorded API responses."""

from __future__ import annotations

import pytest


@pytest.mark.vcr
def test_llm_judge_score_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM judge correctly parses a score from a recorded API response.

    The cassette returns score "4" for both bidirectional calls, so the
    expected llm_judge_score is _normalize(4.0) = 0.75.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")

    from rosettastone.evaluate.llm_judge import LLMJudgeEvaluator

    evaluator = LLMJudgeEvaluator()
    scores = evaluator.score("Paris is the capital of France.", "The capital of France is Paris.")

    assert "llm_judge_score" in scores
    assert 0.0 <= scores["llm_judge_score"] <= 1.0
    # Both cassette responses return "4"; avg raw = 4.0, normalized = (4-1)/4 = 0.75
    assert scores["llm_judge_score"] == pytest.approx(0.75)
