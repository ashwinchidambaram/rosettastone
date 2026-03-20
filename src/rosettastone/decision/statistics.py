"""Statistical analysis of evaluation results."""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING

from rosettastone.core.context import TypeStats

if TYPE_CHECKING:
    from rosettastone.core.types import EvalResult


def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute the Wilson score confidence interval for a proportion.

    Args:
        wins: Number of successes.
        total: Total number of trials.
        z: Z-score for the desired confidence level (default 1.96 for 95%).

    Returns:
        (lower, upper) bounds of the confidence interval, both in [0, 1].
    """
    if total == 0:
        return (0.0, 0.0)

    p_hat = wins / total
    z2 = z * z
    denominator = 1 + z2 / total
    centre = (p_hat + z2 / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(p_hat * (1 - p_hat) / total + z2 / (4 * total * total))

    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return (lower, upper)


def compute_type_stats(
    results: list[EvalResult],
    output_type: str,
    threshold: float = 0.8,
) -> TypeStats:
    """Compute aggregate statistics for a specific output type.

    Results are filtered by matching ``result.details["output_type"]`` to *output_type*.
    Win rate is determined by comparing ``composite_score`` against *threshold*.

    Args:
        results: Full list of evaluation results (all types).
        output_type: The output type string to filter on (e.g. "json", "short_text").
        threshold: Score threshold above which a result is counted as a win.

    Returns:
        A :class:`TypeStats` dataclass populated with the computed statistics.
    """
    filtered = [r for r in results if r.details.get("output_type") == output_type]

    if not filtered:
        return TypeStats(sample_count=0)

    scores = [r.composite_score for r in filtered]
    sample_count = len(scores)

    wins = sum(1 for s in scores if s >= threshold)
    win_rate = wins / sample_count

    mean = statistics.mean(scores)
    median = statistics.median(scores)

    sorted_scores = sorted(scores)

    def percentile(data: list[float], p: float) -> float:
        """Linear interpolation percentile (same as numpy default)."""
        n = len(data)
        if n == 1:
            return data[0]
        idx = p / 100 * (n - 1)
        lo = int(idx)
        hi = lo + 1
        if hi >= n:
            return data[-1]
        frac = idx - lo
        return data[lo] + frac * (data[hi] - data[lo])

    p10 = percentile(sorted_scores, 10)
    p50 = percentile(sorted_scores, 50)
    p90 = percentile(sorted_scores, 90)

    min_score = sorted_scores[0]
    max_score = sorted_scores[-1]

    ci = wilson_interval(wins, sample_count)

    return TypeStats(
        win_rate=win_rate,
        mean=mean,
        median=median,
        p10=p10,
        p50=p50,
        p90=p90,
        min_score=min_score,
        max_score=max_score,
        sample_count=sample_count,
        confidence_interval=ci,
    )
