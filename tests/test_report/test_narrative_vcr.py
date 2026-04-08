"""VCR-enabled tests for executive narrative — replays recorded API responses."""

from __future__ import annotations

import pytest

from rosettastone.core.types import EvalResult, MigrationResult, OutputType, PromptPair


def _make_migration_result() -> MigrationResult:
    """Build a minimal MigrationResult for narrative generation."""
    pair = PromptPair(
        prompt="What is the capital of France?",
        response="Paris",
        source_model="openai/gpt-4o",
        output_type=OutputType.CLASSIFICATION,
    )
    eval_result = EvalResult(
        prompt_pair=pair,
        new_response="Paris",
        scores={"embedding_similarity": 0.95},
        composite_score=0.95,
        is_win=True,
    )
    return MigrationResult(
        config={"source_model": "openai/gpt-4o", "target_model": "anthropic/claude-sonnet-4"},
        optimized_prompt="What is the capital of France?",
        baseline_results=[eval_result],
        validation_results=[eval_result],
        confidence_score=0.85,
        baseline_score=0.73,
        improvement=0.12,
        cost_usd=0.0023,
        duration_seconds=45.2,
        warnings=[],
        recommendation="GO",
    )


@pytest.mark.vcr
def test_narrative_generates_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """generate_executive_narrative returns a non-empty string from a recorded response."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")

    from rosettastone.report.narrative import generate_executive_narrative

    result = _make_migration_result()
    narrative = generate_executive_narrative(result)

    assert isinstance(narrative, str)
    assert len(narrative) > 50
