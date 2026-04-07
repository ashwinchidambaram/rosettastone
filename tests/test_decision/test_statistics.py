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


# ── Comprehensive Wilson interval tests ──────────────────────────────────────


@pytest.mark.parametrize("wins", list(range(10, 100, 10)))
def test_wilson_interval_monotonicity(wins: int):
    """For fixed total=100, lower bound strictly increases as wins increases.

    We verify the monotonicity property pairwise: each successive wins value
    must yield a strictly higher lower bound than the previous one.
    """
    total = 100
    lo, _ = wilson_interval(wins, total)
    prev_lo, _ = wilson_interval(wins - 10, total)
    assert lo > prev_lo, (
        f"Lower bound did not strictly increase when wins went from {wins - 10} "
        f"to {wins} (total={total}): {prev_lo:.6f} -> {lo:.6f}"
    )


@pytest.mark.parametrize("total", [10, 50, 100, 500, 1000])
def test_wilson_interval_narrows_with_samples(total: int):
    """For fixed win_rate=0.7, interval width strictly decreases as total grows.

    Each successive (larger) total should yield a narrower Wilson interval than
    the previous total in the list.
    """
    wins = round(0.7 * total)
    lo, hi = wilson_interval(wins, total)
    width = hi - lo

    # Build the previous-total entry to compare against.
    totals = [10, 50, 100, 500, 1000]
    idx = totals.index(total)
    if idx == 0:
        # Nothing to compare against for the smallest total — just assert the
        # interval is a valid range.
        assert width >= 0.0
        return

    prev_total = totals[idx - 1]
    prev_wins = round(0.7 * prev_total)
    prev_lo, prev_hi = wilson_interval(prev_wins, prev_total)
    prev_width = prev_hi - prev_lo

    assert width < prev_width, (
        f"Interval did not narrow when total increased from {prev_total} to {total}: "
        f"width {prev_width:.6f} -> {width:.6f}"
    )


@pytest.mark.parametrize(
    "wins,total,z,expected_lo,expected_hi",
    [
        # Hand-computed using the standard Wilson score interval formula:
        #   centre = (p_hat + z²/2n) / (1 + z²/n)
        #   margin = z * sqrt(p_hat*(1-p_hat)/n + z²/4n²) / (1 + z²/n)
        # wins=7, total=10, z=1.96 -> lo≈0.3968, hi≈0.8922
        (7, 10, 1.96, 0.3968, 0.8922),
        # wins=50, total=100, z=1.96 -> lo≈0.4038, hi≈0.5962 (symmetric around 0.5)
        (50, 100, 1.96, 0.4038, 0.5962),
    ],
)
def test_wilson_interval_known_values(
    wins: int, total: int, z: float, expected_lo: float, expected_hi: float
):
    """Verify against hand-computed Wilson interval values."""
    lo, hi = wilson_interval(wins, total, z=z)
    assert lo == pytest.approx(expected_lo, abs=0.01), (
        f"Lower bound mismatch for wins={wins}, total={total}: "
        f"got {lo:.4f}, expected ~{expected_lo:.4f}"
    )
    assert hi == pytest.approx(expected_hi, abs=0.01), (
        f"Upper bound mismatch for wins={wins}, total={total}: "
        f"got {hi:.4f}, expected ~{expected_hi:.4f}"
    )


def test_wilson_interval_large_n():
    """For large n at 50% win rate, both bounds should be within 0.02 of 0.5."""
    lo, hi = wilson_interval(5000, 10000)
    assert abs(lo - 0.5) < 0.02, f"Lower bound {lo:.4f} is more than 0.02 from 0.5"
    assert abs(hi - 0.5) < 0.02, f"Upper bound {hi:.4f} is more than 0.02 from 0.5"


def test_wilson_interval_extreme_proportions():
    """Near-zero and near-one win rates produce sensible extreme intervals."""
    # Very few wins in many trials
    lo_low, hi_low = wilson_interval(1, 1000)
    assert lo_low > 0.0, f"Lower bound {lo_low} should be > 0 even for 1/1000"
    assert hi_low < 0.01, f"Upper bound {hi_low} should be < 0.01 for 1/1000"

    # Very many wins in many trials
    lo_high, hi_high = wilson_interval(999, 1000)
    assert lo_high > 0.99, f"Lower bound {lo_high} should be > 0.99 for 999/1000"
    assert hi_high <= 1.0, f"Upper bound {hi_high} must not exceed 1.0"


def test_wilson_interval_custom_z_scores():
    """For the same wins/total, wider z produces a wider interval.

    Verify: width(z=1.645) < width(z=1.96) < width(z=2.576).
    """
    wins, total = 50, 100

    lo_90, hi_90 = wilson_interval(wins, total, z=1.645)
    lo_95, hi_95 = wilson_interval(wins, total, z=1.96)
    lo_99, hi_99 = wilson_interval(wins, total, z=2.576)

    width_90 = hi_90 - lo_90
    width_95 = hi_95 - lo_95
    width_99 = hi_99 - lo_99

    assert width_90 < width_95, (
        f"z=1.645 interval ({width_90:.6f}) should be narrower than "
        f"z=1.96 interval ({width_95:.6f})"
    )
    assert width_95 < width_99, (
        f"z=1.96 interval ({width_95:.6f}) should be narrower than "
        f"z=2.576 interval ({width_99:.6f})"
    )


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
