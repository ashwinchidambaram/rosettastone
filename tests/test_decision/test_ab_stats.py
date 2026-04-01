"""Tests for the A/B testing statistical significance engine."""

from __future__ import annotations

from rosettastone.decision.ab_stats import (
    bootstrap_ci,
    chi_squared_test,
    compute_ab_significance,
)


class TestChiSquaredTest:
    def test_significant_difference(self):
        """Clearly different win rates should be significant."""
        chi2, p, sig = chi_squared_test(80, 100, 50, 100)
        assert sig == True  # noqa: E712 — scipy may return np.bool_
        assert p < 0.05
        assert chi2 > 0

    def test_identical_rates(self):
        """Identical win rates should not be significant."""
        chi2, p, sig = chi_squared_test(50, 100, 50, 100)
        assert sig == False  # noqa: E712 — scipy may return np.bool_

    def test_empty_groups(self):
        """Zero-sized groups return non-significant."""
        chi2, p, sig = chi_squared_test(0, 0, 0, 0)
        assert sig is False
        assert p == 1.0

    def test_all_wins(self):
        """All wins on both sides is not significant."""
        chi2, p, sig = chi_squared_test(100, 100, 100, 100)
        assert sig is False

    def test_all_losses(self):
        """All losses on both sides is not significant."""
        chi2, p, sig = chi_squared_test(0, 100, 0, 100)
        assert sig is False

    def test_one_sided_zero(self):
        """One side has zero total."""
        chi2, p, sig = chi_squared_test(50, 100, 0, 0)
        assert sig is False
        assert p == 1.0


class TestBootstrapCI:
    def test_clearly_different(self):
        """Bootstrap CI should not contain zero for clearly different scores."""
        scores_a = [0.9, 0.85, 0.92, 0.88, 0.87, 0.91]
        scores_b = [0.6, 0.65, 0.62, 0.58, 0.61, 0.63]
        diff, lo, hi = bootstrap_ci(scores_a, scores_b)
        assert diff > 0
        assert lo > 0  # CI doesn't cross zero
        assert hi > lo

    def test_similar_scores(self):
        """Bootstrap CI should contain zero for similar scores."""
        scores_a = [0.8, 0.82, 0.79, 0.81]
        scores_b = [0.79, 0.81, 0.80, 0.82]
        diff, lo, hi = bootstrap_ci(scores_a, scores_b)
        assert lo <= 0 <= hi  # CI crosses zero

    def test_empty_lists(self):
        """Empty score lists return zeros."""
        diff, lo, hi = bootstrap_ci([], [])
        assert diff == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_deterministic_with_seed(self):
        """Same seed produces same results."""
        a = [0.9, 0.85, 0.8]
        b = [0.7, 0.65, 0.6]
        r1 = bootstrap_ci(a, b, seed=42)
        r2 = bootstrap_ci(a, b, seed=42)
        assert r1 == r2


class TestComputeABSignificance:
    def test_full_analysis(self):
        """compute_ab_significance produces complete result."""
        results = [
            {"assigned_version": "a", "score_a": 0.9, "score_b": 0.7, "winner": "a"},
            {"assigned_version": "a", "score_a": 0.85, "score_b": 0.65, "winner": "a"},
            {"assigned_version": "b", "score_a": 0.8, "score_b": 0.6, "winner": "a"},
            {"assigned_version": "b", "score_a": 0.9, "score_b": 0.8, "winner": "a"},
        ] * 25  # 100 results for statistical power

        sig = compute_ab_significance(results)
        assert sig.sample_size_a == 50
        assert sig.sample_size_b == 50
        assert sig.mean_diff != 0

    def test_empty_results(self):
        """Empty results return non-significant."""
        sig = compute_ab_significance([])
        assert sig.significant is False
        assert sig.sample_size_a == 0
        assert sig.sample_size_b == 0
