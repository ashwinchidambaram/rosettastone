"""Property-based tests for ab_stats.py — chi2 approximation and bootstrap CI."""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from rosettastone.decision.ab_stats import (
    _chi2_survival_approx,
    bootstrap_ci,
    chi_squared_test,
)

# ---------------------------------------------------------------------------
# _chi2_survival_approx properties
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False))
def test_chi2_survival_always_in_unit_interval(x: float) -> None:
    """Survival function output must always lie in [0, 1]."""
    result = _chi2_survival_approx(x)
    assert 0.0 <= result <= 1.0, f"Expected result in [0, 1], got {result} for x={x}"


@settings(max_examples=100)
@given(
    x1=st.floats(min_value=1e-6, max_value=50, allow_nan=False, allow_infinity=False),
    x2=st.floats(min_value=1e-6, max_value=50, allow_nan=False, allow_infinity=False),
)
def test_chi2_survival_monotonically_decreasing(x1: float, x2: float) -> None:
    """Survival function must be monotonically non-increasing: x1 < x2 → approx(x1) >= approx(x2)."""
    if x1 >= x2:
        x1, x2 = x2, x1  # ensure x1 < x2
    if math.isclose(x1, x2, rel_tol=1e-9):
        return  # skip degenerate case
    p1 = _chi2_survival_approx(x1)
    p2 = _chi2_survival_approx(x2)
    assert p1 >= p2 - 1e-9, f"Expected approx({x1}) >= approx({x2}), got {p1} < {p2}"


def test_chi2_survival_zero_returns_one() -> None:
    """Survival function at x=0 must return 1.0 (full probability mass above zero)."""
    result = _chi2_survival_approx(0.0)
    assert result == 1.0, f"Expected 1.0 for x=0.0, got {result}"


# ---------------------------------------------------------------------------
# bootstrap_ci properties
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    scores_a=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=50,
    ),
    scores_b=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=50,
    ),
)
def test_bootstrap_ci_contains_mean_diff(scores_a: list[float], scores_b: list[float]) -> None:
    """Bootstrap CI must contain the observed mean difference."""
    mean_diff, ci_lower, ci_upper = bootstrap_ci(scores_a, scores_b, n_bootstrap=200, seed=42)
    assert ci_lower <= mean_diff <= ci_upper, (
        f"mean_diff={mean_diff} not in [{ci_lower}, {ci_upper}]"
    )


@settings(max_examples=100)
@given(
    scores_a=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=50,
    ),
    scores_b=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=50,
    ),
)
def test_bootstrap_ci_deterministic_with_seed(scores_a: list[float], scores_b: list[float]) -> None:
    """Same inputs and seed=42 must produce identical output."""
    result1 = bootstrap_ci(scores_a, scores_b, n_bootstrap=200, seed=42)
    result2 = bootstrap_ci(scores_a, scores_b, n_bootstrap=200, seed=42)
    assert result1 == result2, (
        f"Expected identical results with same seed, got {result1} vs {result2}"
    )


# ---------------------------------------------------------------------------
# chi_squared_test p-value properties
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    wins_a=st.integers(min_value=0, max_value=100),
    total_a=st.integers(min_value=1, max_value=100),
    wins_b=st.integers(min_value=0, max_value=100),
    total_b=st.integers(min_value=1, max_value=100),
)
def test_chi_squared_test_p_value_bounded(
    wins_a: int, total_a: int, wins_b: int, total_b: int
) -> None:
    """p_value from chi_squared_test must always be in [0, 1]."""
    # Clamp wins to not exceed totals
    wins_a = min(wins_a, total_a)
    wins_b = min(wins_b, total_b)

    _chi2_stat, p_value, _significant = chi_squared_test(wins_a, total_a, wins_b, total_b)
    assert 0.0 <= p_value <= 1.0, (
        f"p_value={p_value} not in [0, 1] for "
        f"wins_a={wins_a}, total_a={total_a}, wins_b={wins_b}, total_b={total_b}"
    )
