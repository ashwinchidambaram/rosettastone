"""Recommendation engine: derives GO / NO_GO / CONDITIONAL from evaluation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rosettastone.core.context import TypeStats
from rosettastone.decision.statistics import compute_type_stats

if TYPE_CHECKING:
    from rosettastone.core.types import EvalResult

# Default win-rate thresholds per output type.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "json": 0.95,
    "classification": 0.90,
    "short_text": 0.80,
    "long_text": 0.75,
}

# Minimum sample count before we consider statistics reliable enough for GO.
MIN_RELIABLE_SAMPLES = 30
# Minimum sample count for CONDITIONAL (insufficient data caveat).
MIN_SAMPLES_FOR_CONDITIONAL = 10


class Recommendation(StrEnum):
    GO = "GO"
    NO_GO = "NO_GO"
    CONDITIONAL = "CONDITIONAL"


@dataclass
class RecommendationResult:
    """Result of the recommendation engine."""

    recommendation: Recommendation
    reasoning: str
    per_type_details: dict[str, TypeStats] = field(default_factory=dict)


def _has_high_severity(warning: Any) -> bool:
    """Return True if a safety warning has severity HIGH."""
    if isinstance(warning, dict):
        return str(warning.get("severity", "")).upper() == "HIGH"
    # Plain strings are not HIGH severity by definition.
    return False


def make_recommendation(
    validation_results: list[EvalResult],
    safety_warnings: list[Any],
    thresholds: dict[str, float],
) -> RecommendationResult:
    """Derive a migration recommendation from evaluation results and safety findings.

    Rules (evaluated in priority order):

    1. Any safety warning with severity HIGH → NO_GO.
    2. Any output type whose Wilson CI lower bound falls below its threshold → CONDITIONAL
       (statistically uncertain results are CONDITIONAL even if the point estimate passes).
    3. Any output type with fewer than MIN_SAMPLES_FOR_CONDITIONAL samples → CONDITIONAL
       (insufficient data caveat).
    4. All types have ≥ MIN_RELIABLE_SAMPLES samples and CI lower bounds meet thresholds → GO.

    Args:
        validation_results: Evaluation results from the validation phase.
        safety_warnings: List of warning strings or dicts (dicts may carry a ``severity`` key).
        thresholds: Per-type win-rate thresholds.  Missing types fall back to
            :data:`DEFAULT_THRESHOLDS` and then to 0.80.

    Returns:
        A :class:`RecommendationResult` with the recommendation, human-readable reasoning,
        and per-type statistics.
    """
    effective_thresholds = {**DEFAULT_THRESHOLDS, **thresholds}

    # Collect the distinct output types present in results.
    output_types: set[str] = set()
    for result in validation_results:
        ot = result.details.get("output_type")
        if ot:
            output_types.add(str(ot))

    # If no output_type annotations exist, treat all results as one anonymous group.
    if not output_types and validation_results:
        output_types = {"unknown"}

    # Compute per-type stats.
    per_type: dict[str, TypeStats] = {}
    for ot in sorted(output_types):
        threshold = effective_thresholds.get(ot, 0.80)
        if ot == "unknown":
            # No output_type annotation — score all results.
            stats = compute_type_stats(
                # Temporarily inject "unknown" into details for filtering.
                [
                    r.model_copy(update={"details": {**r.details, "output_type": "unknown"}})
                    for r in validation_results
                ],
                "unknown",
                threshold,
            )
        else:
            stats = compute_type_stats(validation_results, ot, threshold)
        per_type[ot] = stats

    # ── Rule 1: HIGH-severity safety blocker ────────────────────────────────
    high_warnings = [w for w in safety_warnings if _has_high_severity(w)]
    if high_warnings:
        descriptions = []
        for w in high_warnings:
            if isinstance(w, dict):
                msg = w.get("message") or w.get("msg") or str(w)
            else:
                msg = str(w)
            descriptions.append(msg)
        reasoning = (
            "Migration blocked due to HIGH-severity safety findings: "
            + "; ".join(descriptions)
            + ". Resolve these issues before proceeding."
        )
        return RecommendationResult(
            recommendation=Recommendation.NO_GO,
            reasoning=reasoning,
            per_type_details=per_type,
        )

    # ── Rules 2 & 3: threshold / sample-size checks ─────────────────────────
    below_threshold: list[str] = []
    insufficient_samples: list[str] = []

    for ot, stats in per_type.items():
        threshold = effective_thresholds.get(ot, 0.80)
        if stats.sample_count < MIN_SAMPLES_FOR_CONDITIONAL:
            insufficient_samples.append(
                f"{ot} ({stats.sample_count} sample{'s' if stats.sample_count != 1 else ''})"
            )
        elif stats.confidence_interval[0] < threshold:
            # Use Wilson CI lower bound: statistically uncertain results are CONDITIONAL
            # even when the point-estimate win_rate clears the threshold.
            below_threshold.append(
                f"{ot} (win rate {stats.win_rate:.1%}, CI lower bound "
                f"{stats.confidence_interval[0]:.1%} < threshold {threshold:.1%})"
            )

    if below_threshold or insufficient_samples:
        parts: list[str] = []
        if below_threshold:
            parts.append(
                "The following output types did not meet their win-rate thresholds: "
                + ", ".join(below_threshold)
                + "."
            )
        if insufficient_samples:
            parts.append(
                "Insufficient samples for reliable statistics in: "
                + ", ".join(insufficient_samples)
                + ". Collect more data before making a final decision."
            )
        reasoning = " ".join(parts)
        return RecommendationResult(
            recommendation=Recommendation.CONDITIONAL,
            reasoning=reasoning,
            per_type_details=per_type,
        )

    # ── Rule 4: All types pass ───────────────────────────────────────────────
    if not per_type:
        reasoning = "No evaluation results found. Cannot make a recommendation."
        return RecommendationResult(
            recommendation=Recommendation.CONDITIONAL,
            reasoning=reasoning,
            per_type_details=per_type,
        )

    # GO requires MIN_RELIABLE_SAMPLES (30) for each type; fewer → CONDITIONAL.
    low_sample_for_go = [
        f"{ot} ({stats.sample_count} sample{'s' if stats.sample_count != 1 else ''})"
        for ot, stats in per_type.items()
        if stats.sample_count < MIN_RELIABLE_SAMPLES
    ]
    if low_sample_for_go:
        reasoning = (
            "Insufficient samples for a GO recommendation in: "
            + ", ".join(low_sample_for_go)
            + f". At least {MIN_RELIABLE_SAMPLES} samples per type are required for GO."
        )
        return RecommendationResult(
            recommendation=Recommendation.CONDITIONAL,
            reasoning=reasoning,
            per_type_details=per_type,
        )

    type_summaries = [
        f"{ot} (win rate {stats.win_rate:.1%}, n={stats.sample_count})"
        for ot, stats in sorted(per_type.items())
    ]
    reasoning = (
        "All output types meet their win-rate thresholds with adequate sample sizes: "
        + ", ".join(type_summaries)
        + ". Migration is recommended."
    )
    return RecommendationResult(
        recommendation=Recommendation.GO,
        reasoning=reasoning,
        per_type_details=per_type,
    )
