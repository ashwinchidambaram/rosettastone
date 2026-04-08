"""Statistical significance engine for A/B testing.

Provides chi-squared test for win-rate comparison and bootstrap confidence
intervals for score differences. Uses scipy when available, falls back to
a pure-Python chi-squared approximation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass
class ABSignificanceResult:
    """Result of statistical significance analysis."""

    chi2: float
    p_value: float
    significant: bool  # p < 0.05
    mean_diff: float  # mean(scores_a) - mean(scores_b)
    ci_lower: float
    ci_upper: float
    sample_size_a: int
    sample_size_b: int


def _chi2_survival_approx(x: float, df: int = 1) -> float:
    """Pure-Python approximation of chi-squared survival function (1 - CDF).

    Uses the Wilson-Hilferty normal approximation for df=1:
        Z = ((x/df)^(1/3) - (1 - 2/(9*df))) / sqrt(2/(9*df))
    Then converts to p-value via the complementary error function.

    Accurate to ~0.01 for typical test values. Sufficient for A/B testing
    where exact p-values aren't needed — the decision is significant vs not.
    """
    if x <= 0:
        return 1.0
    if df <= 0:
        return 0.0

    # Wilson-Hilferty approximation
    k = df
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))

    # Approximate P(Z > z) using erfc
    # erfc(x) = 1 - erf(x), and P(Z > z) = 0.5 * erfc(z / sqrt(2))
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return max(0.0, min(1.0, p))


def chi_squared_test(
    wins_a: int,
    total_a: int,
    wins_b: int,
    total_b: int,
) -> tuple[float, float, bool]:
    """Chi-squared test for comparing two win rates.

    Returns (chi2_statistic, p_value, is_significant).
    Uses scipy.stats.chi2_contingency when available, else pure-Python fallback.
    """
    if total_a == 0 or total_b == 0:
        return 0.0, 1.0, False

    losses_a = total_a - wins_a
    losses_b = total_b - wins_b

    # Observed counts
    observed = [[wins_a, losses_a], [wins_b, losses_b]]
    grand_total = total_a + total_b
    total_wins = wins_a + wins_b
    total_losses = losses_a + losses_b

    # Avoid division by zero when all wins or all losses
    if total_wins == 0 or total_losses == 0:
        return 0.0, 1.0, False

    try:
        from scipy.stats import chi2_contingency  # type: ignore[import-untyped]

        chi2, p, _, _ = chi2_contingency(observed, correction=True)
        return float(chi2), float(p), p < 0.05
    except ImportError:
        pass

    # Pure-Python: manual chi-squared with Yates' correction
    expected = [
        [total_a * total_wins / grand_total, total_a * total_losses / grand_total],
        [total_b * total_wins / grand_total, total_b * total_losses / grand_total],
    ]

    chi2 = 0.0
    for i in range(2):
        for j in range(2):
            e = expected[i][j]
            if e > 0:
                # Yates' continuity correction
                chi2 += (abs(observed[i][j] - e) - 0.5) ** 2 / e

    p = _chi2_survival_approx(chi2, df=1)
    return chi2, p, p < 0.05


def bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the difference in means.

    Returns (mean_diff, ci_lower, ci_upper).
    mean_diff = mean(scores_a) - mean(scores_b).
    """
    if not scores_a or not scores_b:
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    mean_a = sum(scores_a) / len(scores_a)
    mean_b = sum(scores_b) / len(scores_b)
    observed_diff = mean_a - mean_b

    diffs: list[float] = []
    for _ in range(n_bootstrap):
        sample_a = rng.choices(scores_a, k=len(scores_a))
        sample_b = rng.choices(scores_b, k=len(scores_b))
        diff = sum(sample_a) / len(sample_a) - sum(sample_b) / len(sample_b)
        diffs.append(diff)

    diffs.sort()
    alpha = 1.0 - confidence
    lower_idx = max(0, int(math.floor(alpha / 2 * n_bootstrap)))
    upper_idx = min(n_bootstrap - 1, int(math.ceil((1.0 - alpha / 2) * n_bootstrap)) - 1)

    return observed_diff, diffs[lower_idx], diffs[upper_idx]


def compute_ab_significance(
    results: list[dict[str, Any]],
) -> ABSignificanceResult:
    """Compute full significance analysis from A/B test result rows.

    Each result dict must have: assigned_version ("a"/"b"), score_a (float|None),
    score_b (float|None), winner ("a"/"b"/"tie"|None).
    """
    wins_a = 0
    wins_b = 0
    total_a = 0
    total_b = 0
    scores_a: list[float] = []
    scores_b: list[float] = []

    for r in results:
        version = r.get("assigned_version", "")
        if version == "a":
            total_a += 1
            if r.get("winner") == "a":
                wins_a += 1
        elif version == "b":
            total_b += 1
            if r.get("winner") == "b":
                wins_b += 1

        if r.get("score_a") is not None:
            scores_a.append(float(r["score_a"]))
        if r.get("score_b") is not None:
            scores_b.append(float(r["score_b"]))

    chi2, p_value, significant = chi_squared_test(wins_a, total_a, wins_b, total_b)
    mean_diff, ci_lower, ci_upper = bootstrap_ci(scores_a, scores_b)

    return ABSignificanceResult(
        chi2=chi2,
        p_value=p_value,
        significant=significant,
        mean_diff=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        sample_size_a=total_a,
        sample_size_b=total_b,
    )
