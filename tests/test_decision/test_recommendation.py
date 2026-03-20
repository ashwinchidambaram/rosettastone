"""Tests for the recommendation engine: GO, NO_GO, CONDITIONAL paths."""

from __future__ import annotations

from rosettastone.core.types import EvalResult, PromptPair
from rosettastone.decision.recommendation import (
    DEFAULT_THRESHOLDS,
    MIN_RELIABLE_SAMPLES,
    Recommendation,
    RecommendationResult,
    make_recommendation,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_result(composite_score: float, output_type: str) -> EvalResult:
    return EvalResult(
        prompt_pair=PromptPair(prompt="q", response="a", source_model="m"),
        new_response="r",
        scores={"composite": composite_score},
        composite_score=composite_score,
        is_win=composite_score >= 0.8,
        details={"output_type": output_type},
    )


def _enough_results(
    output_type: str,
    win_rate: float,
    n: int = MIN_RELIABLE_SAMPLES,
) -> list[EvalResult]:
    """Return n results for output_type with win_rate fraction above threshold."""
    threshold = DEFAULT_THRESHOLDS.get(output_type, 0.80)
    wins = round(win_rate * n)
    results = []
    for i in range(n):
        score = threshold + 0.01 if i < wins else threshold - 0.01
        results.append(_make_result(score, output_type))
    return results


# ── RecommendationResult type ────────────────────────────────────────────────


def test_recommendation_result_is_dataclass():
    r = RecommendationResult(recommendation=Recommendation.GO, reasoning="ok")
    assert r.recommendation == Recommendation.GO
    assert r.per_type_details == {}


# ── NO_GO: high-severity safety warning ─────────────────────────────────────


def test_no_go_on_high_severity_warning():
    """A single HIGH-severity warning must produce NO_GO."""
    results = _enough_results("json", win_rate=1.0)
    warnings = [{"severity": "HIGH", "message": "PII leak detected"}]
    rec = make_recommendation(results, warnings, {})
    assert rec.recommendation == Recommendation.NO_GO
    assert "PII leak detected" in rec.reasoning


def test_no_go_on_multiple_high_severity_warnings():
    warnings = [
        {"severity": "HIGH", "message": "Security issue"},
        {"severity": "HIGH", "message": "Data exposure"},
    ]
    results = _enough_results("short_text", win_rate=1.0)
    rec = make_recommendation(results, warnings, {})
    assert rec.recommendation == Recommendation.NO_GO
    assert "Security issue" in rec.reasoning
    assert "Data exposure" in rec.reasoning


def test_no_go_high_severity_case_insensitive():
    """Severity check should be case-insensitive."""
    warnings = [{"severity": "high", "message": "Critical issue"}]
    results = _enough_results("json", win_rate=1.0)
    rec = make_recommendation(results, warnings, {})
    assert rec.recommendation == Recommendation.NO_GO


def test_medium_severity_does_not_trigger_no_go():
    """MEDIUM severity warnings do not block migration."""
    warnings = [{"severity": "MEDIUM", "message": "Minor concern"}]
    results = _enough_results("json", win_rate=1.0)
    rec = make_recommendation(results, warnings, {})
    assert rec.recommendation != Recommendation.NO_GO


def test_plain_string_warning_does_not_trigger_no_go():
    """Plain string warnings have no severity → should not produce NO_GO."""
    results = _enough_results("json", win_rate=1.0)
    rec = make_recommendation(results, ["some warning string"], {})
    assert rec.recommendation != Recommendation.NO_GO


# ── CONDITIONAL: below-threshold ────────────────────────────────────────────


def test_conditional_when_win_rate_below_threshold():
    """Output type below its threshold triggers CONDITIONAL."""
    # json threshold is 0.95 — send win_rate=0.5
    results = _enough_results("json", win_rate=0.5)
    rec = make_recommendation(results, [], {})
    assert rec.recommendation == Recommendation.CONDITIONAL
    assert "json" in rec.reasoning


def test_conditional_contains_threshold_info():
    results = _enough_results("json", win_rate=0.5)
    rec = make_recommendation(results, [], {})
    # Should mention the threshold or win rate
    assert "win rate" in rec.reasoning.lower() or "threshold" in rec.reasoning.lower()


def test_conditional_when_custom_threshold_not_met():
    """Custom threshold overrides default."""
    results = _enough_results("short_text", win_rate=0.85, n=20)
    # Override short_text threshold to something extremely high
    rec = make_recommendation(results, [], {"short_text": 0.99})
    assert rec.recommendation == Recommendation.CONDITIONAL


# ── CONDITIONAL: insufficient samples ────────────────────────────────────────


def test_conditional_on_insufficient_samples():
    """Fewer than MIN_RELIABLE_SAMPLES triggers CONDITIONAL with caveat."""
    results = [_make_result(1.0, "json") for _ in range(MIN_RELIABLE_SAMPLES - 1)]
    rec = make_recommendation(results, [], {})
    assert rec.recommendation == Recommendation.CONDITIONAL
    assert "insufficient" in rec.reasoning.lower() or "sample" in rec.reasoning.lower()


def test_conditional_single_sample():
    """Single sample is definitely insufficient."""
    rec = make_recommendation([_make_result(1.0, "classification")], [], {})
    assert rec.recommendation == Recommendation.CONDITIONAL


# ── GO: all types pass ───────────────────────────────────────────────────────


def test_go_when_all_types_pass():
    """All output types at 100% win rate with enough samples → GO."""
    results = _enough_results("json", win_rate=1.0) + _enough_results("short_text", win_rate=1.0)
    rec = make_recommendation(results, [], {})
    assert rec.recommendation == Recommendation.GO


def test_go_reasoning_mentions_types():
    results = _enough_results("classification", win_rate=1.0)
    rec = make_recommendation(results, [], {})
    assert rec.recommendation == Recommendation.GO
    assert "classification" in rec.reasoning


def test_go_with_non_high_warnings_still_goes():
    """Low-severity warnings should not block GO."""
    results = _enough_results("long_text", win_rate=1.0)
    warnings = [{"severity": "LOW", "message": "Minor note"}]
    rec = make_recommendation(results, warnings, {})
    assert rec.recommendation == Recommendation.GO


# ── per_type_details ─────────────────────────────────────────────────────────


def test_per_type_details_populated():
    """per_type_details is always populated from validation results."""
    results = _enough_results("json", win_rate=1.0)
    rec = make_recommendation(results, [], {})
    assert "json" in rec.per_type_details


def test_per_type_details_sample_count_correct():
    n = 15
    results = [_make_result(1.0, "classification") for _ in range(n)]
    rec = make_recommendation(results, [], {})
    assert rec.per_type_details["classification"].sample_count == n


# ── edge cases ────────────────────────────────────────────────────────────────


def test_no_results_returns_conditional():
    """No validation results → CONDITIONAL (cannot recommend GO)."""
    rec = make_recommendation([], [], {})
    assert rec.recommendation == Recommendation.CONDITIONAL


def test_priority_no_go_beats_threshold_failure():
    """NO_GO (high severity) takes priority over threshold failures."""
    # json results with low win rate AND a HIGH safety warning
    results = _enough_results("json", win_rate=0.1)
    warnings = [{"severity": "HIGH", "message": "Critical"}]
    rec = make_recommendation(results, warnings, {})
    assert rec.recommendation == Recommendation.NO_GO


def test_returns_recommendation_result_type():
    rec = make_recommendation([], [], {})
    assert isinstance(rec, RecommendationResult)
