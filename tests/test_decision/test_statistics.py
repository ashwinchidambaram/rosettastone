"""Tests for the statistics module: wilson_interval and compute_type_stats."""

from __future__ import annotations

import pytest

from rosettastone.core.context import TypeStats
from rosettastone.core.types import EvalResult, PromptPair
from rosettastone.decision.statistics import compute_type_stats, wilson_interval

# ── wilson_interval ──────────────────────────────────────────────────────────


def test_wilson_interval_zero_total():
    """Edge case: zero total returns (0.0, 0.0)."""
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_all_wins():
    """All wins: upper bound is 1.0, lower bound is high."""
    lo, hi = wilson_interval(10, 10)
    assert hi == pytest.approx(1.0, abs=1e-9)
    assert lo > 0.65  # Wilson pulls back from extreme certainty


def test_wilson_interval_no_wins():
    """Zero wins: lower bound is 0.0."""
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi > 0.0  # Some uncertainty remains


def test_wilson_interval_bounds_are_valid_probabilities():
    lo, hi = wilson_interval(7, 10)
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
    assert lo <= hi


def test_wilson_interval_midpoint_50pct():
    """50% win rate should produce a symmetric interval around 0.5."""
    lo, hi = wilson_interval(50, 100)
    centre = (lo + hi) / 2
    assert centre == pytest.approx(0.5, abs=0.02)


def test_wilson_interval_larger_z_widens_interval():
    """Higher z-score should produce a wider interval."""
    lo_95, hi_95 = wilson_interval(5, 10, z=1.96)
    lo_99, hi_99 = wilson_interval(5, 10, z=2.576)
    assert (hi_99 - lo_99) > (hi_95 - lo_95)


def test_wilson_interval_single_trial():
    """Single trial: still a valid probability interval."""
    lo, hi = wilson_interval(1, 1)
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_eval_result(composite_score: float, output_type: str) -> EvalResult:
    return EvalResult(
        prompt_pair=PromptPair(prompt="q", response="a", source_model="m"),
        new_response="r",
        scores={"composite": composite_score},
        composite_score=composite_score,
        is_win=composite_score >= 0.8,
        details={"output_type": output_type},
    )


# ── compute_type_stats ───────────────────────────────────────────────────────


def test_compute_type_stats_empty_results():
    """Empty input returns a zeroed TypeStats with sample_count=0."""
    stats = compute_type_stats([], "json")
    assert stats.sample_count == 0
    assert stats.win_rate == 0.0
    assert stats.confidence_interval == (0.0, 0.0)


def test_compute_type_stats_filters_by_output_type():
    """Only results matching output_type are included."""
    results = [
        _make_eval_result(0.9, "json"),
        _make_eval_result(0.5, "short_text"),
        _make_eval_result(0.8, "json"),
    ]
    stats = compute_type_stats(results, "json")
    assert stats.sample_count == 2


def test_compute_type_stats_no_matching_type():
    """No results for the requested type → zeroed TypeStats."""
    results = [_make_eval_result(0.9, "json")]
    stats = compute_type_stats(results, "long_text")
    assert stats.sample_count == 0


def test_compute_type_stats_win_rate():
    """Win rate computed correctly against default threshold 0.8."""
    results = [
        _make_eval_result(0.9, "json"),  # win
        _make_eval_result(0.7, "json"),  # not a win (below 0.8 threshold)
        _make_eval_result(0.85, "json"),  # win
    ]
    stats = compute_type_stats(results, "json", threshold=0.8)
    assert stats.win_rate == pytest.approx(2 / 3)
    assert stats.sample_count == 3


def test_compute_type_stats_all_wins():
    results = [_make_eval_result(1.0, "classification") for _ in range(5)]
    stats = compute_type_stats(results, "classification", threshold=0.9)
    assert stats.win_rate == 1.0


def test_compute_type_stats_percentiles_single_sample():
    """Single sample: p10, p50, p90 all equal the single score."""
    results = [_make_eval_result(0.75, "long_text")]
    stats = compute_type_stats(results, "long_text")
    assert stats.p10 == pytest.approx(0.75)
    assert stats.p50 == pytest.approx(0.75)
    assert stats.p90 == pytest.approx(0.75)
    assert stats.min_score == pytest.approx(0.75)
    assert stats.max_score == pytest.approx(0.75)


def test_compute_type_stats_percentiles_ordering():
    """P10 ≤ P50 ≤ P90."""
    results = [_make_eval_result(float(i) / 10, "short_text") for i in range(1, 11)]
    stats = compute_type_stats(results, "short_text")
    assert stats.p10 <= stats.p50 <= stats.p90


def test_compute_type_stats_mean_and_median():
    results = [
        _make_eval_result(0.6, "json"),
        _make_eval_result(0.8, "json"),
        _make_eval_result(1.0, "json"),
    ]
    stats = compute_type_stats(results, "json")
    assert stats.mean == pytest.approx((0.6 + 0.8 + 1.0) / 3)
    assert stats.median == pytest.approx(0.8)


def test_compute_type_stats_confidence_interval_valid():
    """Confidence interval should be a valid (lo, hi) pair within [0, 1]."""
    results = [_make_eval_result(0.9, "json") for _ in range(15)]
    stats = compute_type_stats(results, "json", threshold=0.8)
    lo, hi = stats.confidence_interval
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
    assert lo <= hi


def test_compute_type_stats_returns_type_stats_instance():
    results = [_make_eval_result(0.85, "json")]
    stats = compute_type_stats(results, "json")
    assert isinstance(stats, TypeStats)
