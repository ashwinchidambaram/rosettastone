"""Property-based tests for improvement.py — blended score and score parsing."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from rosettastone.optimize.improvement import (
    ImprovementScore,
    _parse_score_and_feedback,
    compute_blended_score,
)

# ---------------------------------------------------------------------------
# compute_blended_score properties
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    eq=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_blended_score_always_in_unit_interval(eq: float, weight: float) -> None:
    """compute_blended_score must always return a value in [0, 1]."""
    result = compute_blended_score(
        equivalence_score=eq,
        improvement_scores=[],
        improvement_weight=weight,
    )
    assert 0.0 <= result <= 1.0, f"Expected result in [0, 1], got {result}"


@settings(max_examples=100)
@given(
    eq=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_blended_score_weight_zero_equals_equivalence(eq: float) -> None:
    """With improvement_weight=0, blended score equals equivalence_score (within float tolerance)."""
    improvement_scores = [
        ImprovementScore(objective="test obj", score=0.5, feedback="ok"),
        ImprovementScore(objective="other obj", score=0.8, feedback="good"),
    ]
    result = compute_blended_score(
        equivalence_score=eq,
        improvement_scores=improvement_scores,
        improvement_weight=0.0,
    )
    # With weight=0: (1 - 0) * eq + 0 * avg = eq, clamped to [0, 1]
    expected = max(0.0, min(1.0, eq))
    assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result} for eq={eq}, weight=0"


@settings(max_examples=100)
@given(
    eq=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_blended_score_empty_improvements_equals_equivalence(eq: float, weight: float) -> None:
    """With empty improvement_scores list, blended score equals clamped equivalence_score."""
    result = compute_blended_score(
        equivalence_score=eq,
        improvement_scores=[],
        improvement_weight=weight,
    )
    # avg_improvement = 0.0 when empty, so:
    # blended = (1 - weight) * eq + weight * 0.0 = (1 - weight) * eq
    expected = max(0.0, min(1.0, (1.0 - weight) * eq))
    assert abs(result - expected) < 1e-9, (
        f"Expected {expected}, got {result} for eq={eq}, weight={weight}"
    )


# ---------------------------------------------------------------------------
# _parse_score_and_feedback properties
# ---------------------------------------------------------------------------


def test_parse_score_valid_scores() -> None:
    """'Score: N' for N in 1–5 must produce a normalized value in [0, 1]."""
    for raw in range(1, 6):
        text = f"Score: {raw}\nFeedback: Some feedback here."
        score, _feedback = _parse_score_and_feedback(text)
        assert score is not None, f"Expected a score for 'Score: {raw}', got None"
        assert 0.0 <= score <= 1.0, f"Expected score in [0, 1], got {score} for raw={raw}"
    # Spot-check boundary normalization: Score: 1 → 0.0, Score: 5 → 1.0
    score_min, _ = _parse_score_and_feedback("Score: 1")
    score_max, _ = _parse_score_and_feedback("Score: 5")
    assert score_min is not None and abs(score_min - 0.0) < 1e-9, (
        f"Score: 1 should normalize to 0.0, got {score_min}"
    )
    assert score_max is not None and abs(score_max - 1.0) < 1e-9, (
        f"Score: 5 should normalize to 1.0, got {score_max}"
    )


@settings(max_examples=100)
@given(
    text=st.text(max_size=100).filter(
        lambda t: "Score:" not in t and "score:" not in t and not t.strip().isdigit()
    )
)
def test_parse_score_invalid_returns_none(text: str) -> None:
    """Text without a recognizable score pattern must return score=None."""
    # Also filter out lone digits 1-5 on their own line (the fallback pattern)
    import re

    if re.search(r"^([1-5])\s*$", text, re.MULTILINE):
        return  # skip — the fallback regex would match
    score, _feedback = _parse_score_and_feedback(text)
    assert score is None, f"Expected score=None for text={text!r}, got {score}"
