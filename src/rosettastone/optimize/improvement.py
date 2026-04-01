"""Improvement mode — LLM-as-judge scoring for behavioral improvement objectives.

Provides helper functions that produce improvement-specific feedback to be
merged into GEPA's optimization metric. The integration phase will wire
these into ``build_migration_metric()`` in ``metric.py``.

Public API:
- ImprovementObjective: dataclass describing an improvement goal + weight.
- ImprovementScore: dataclass with per-objective score and feedback.
- build_improvement_scorer: build a callable that judges responses against objectives.
- build_improvement_feedback: merge improvement feedback into GEPA's base feedback.
- compute_blended_score: blend equivalence and improvement scores.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ImprovementObjective:
    """An improvement objective with a blending weight.

    Attributes:
        description: Human-readable description of the improvement goal.
        weight: Blending weight for this objective (default 0.3).
    """

    description: str
    weight: float = 0.3


@dataclass
class ImprovementScore:
    """Result of evaluating a response against one improvement objective.

    Attributes:
        objective: The objective description that was evaluated.
        score: Normalized score in [0.0, 1.0].
        feedback: Textual feedback from the judge.
    """

    objective: str
    score: float  # 0.0-1.0
    feedback: str


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def _call_litellm_completion(**kwargs: Any) -> Any:
    """Call litellm.completion with lazy import.

    Follows the project's lazy import convention. Importing litellm here
    keeps it out of the module namespace until first use while remaining
    patchable at ``rosettastone.optimize.improvement._call_litellm_completion``.
    """
    import litellm

    return litellm.completion(**kwargs)


def _parse_score_and_feedback(text: str) -> tuple[float | None, str]:
    """Extract a numeric score (1-5) and feedback from LLM judge output.

    Returns (normalized_score, feedback_text).  normalized_score is None when
    the text cannot be parsed.
    """
    # Try to find "Score: N" or just a standalone digit 1-5
    score_match = re.search(r"[Ss]core\s*[:\-]?\s*(\d(?:\.\d+)?)", text)
    if score_match is None:
        # Fallback: look for a bare digit at start of line
        score_match = re.search(r"^(\d(?:\.\d+)?)", text, re.MULTILINE)

    raw_score: float | None = None
    if score_match:
        try:
            raw_score = float(score_match.group(1))
        except ValueError:
            raw_score = None

    # Validate range
    if raw_score is not None and not (1.0 <= raw_score <= 5.0):
        raw_score = None

    normalized: float | None = None
    if raw_score is not None:
        normalized = (raw_score - 1.0) / 4.0

    # Extract feedback text
    feedback_match = re.search(r"[Ff]eedback\s*[:\-]?\s*(.+)", text, re.DOTALL)
    feedback_text = feedback_match.group(1).strip() if feedback_match else text.strip()

    return normalized, feedback_text


def _score_objective(
    objective: str,
    prompt: str,
    actual_response: str,
    model: str,
) -> ImprovementScore:
    """Score a single objective via LLM-as-judge.

    On failure returns score=0.5 with an error-level feedback message.
    """
    messages = [
        {
            "role": "user",
            "content": (
                "Rate on a scale of 1-5 how well the following response "
                "achieves this objective:\n"
                f"Objective: {objective}\n"
                f"Prompt: {prompt}\n"
                f"Response: {actual_response}\n\n"
                "Score (1-5):\nFeedback:"
            ),
        }
    ]

    try:
        response = _call_litellm_completion(model=model, messages=messages)
        text = response.choices[0].message.content or ""
    except Exception:
        logger.debug(
            "LLM judge call failed for objective scoring",
            exc_info=True,
        )
        return ImprovementScore(
            objective=objective,
            score=0.5,
            feedback="Error: LLM judge call failed. Using default score.",
        )

    normalized, feedback_text = _parse_score_and_feedback(text)

    if normalized is None:
        logger.debug("Could not parse LLM judge response for objective")
        return ImprovementScore(
            objective=objective,
            score=0.5,
            feedback=f"Error: could not parse judge response. Raw: {text[:200]}",
        )

    return ImprovementScore(
        objective=objective,
        score=normalized,
        feedback=feedback_text,
    )


def build_improvement_scorer(
    objectives: list[str],
    judge_model: str = "openai/gpt-4o",
) -> Callable[[str, str, str], list[ImprovementScore]]:
    """Build a scorer function that evaluates responses against improvement objectives.

    The returned callable takes (prompt, expected_response, actual_response) and
    returns a list of ImprovementScore, one per objective.

    Uses LiteLLM for LLM-as-judge scoring with a structured rubric:
    - "Rate 1-5 how well the actual response achieves the improvement objective"
    - Normalizes to [0,1]: (score - 1) / 4
    - Handles failures gracefully (returns 0.5 score on error)

    Args:
        objectives: List of objective description strings.
        judge_model: LiteLLM model identifier for the judge.

    Returns:
        A callable (prompt, expected_response, actual_response) -> list[ImprovementScore].
    """

    def scorer(
        prompt: str,
        expected_response: str,
        actual_response: str,
    ) -> list[ImprovementScore]:
        if not objectives:
            return []

        results: list[ImprovementScore] = []
        for objective in objectives:
            result = _score_objective(
                objective=objective,
                prompt=prompt,
                actual_response=actual_response,
                model=judge_model,
            )
            results.append(result)
        return results

    return scorer


# ---------------------------------------------------------------------------
# Feedback merging
# ---------------------------------------------------------------------------


def build_improvement_feedback(
    base_feedback: str,
    improvement_scores: list[ImprovementScore],
) -> str:
    """Merge improvement feedback into GEPA's metric feedback string.

    Returns combined feedback: base_feedback + improvement section with
    per-objective details.  When improvement_scores is empty, returns
    base_feedback unchanged.

    Args:
        base_feedback: The base metric feedback (e.g. from semantic similarity).
        improvement_scores: List of per-objective scores from the improvement scorer.

    Returns:
        Combined feedback string with clear structure for GEPA's reflection step.
    """
    if not improvement_scores:
        return base_feedback

    lines: list[str] = [base_feedback, "", "--- IMPROVEMENT OBJECTIVES ---"]
    for score in improvement_scores:
        lines.append(f"  Objective: {score.objective} | Score: {score.score:.2f}")
        lines.append(f"  Feedback: {score.feedback}")
        lines.append("")

    avg = sum(s.score for s in improvement_scores) / len(improvement_scores)
    lines.append(f"Average improvement score: {avg:.2f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Blended scoring
# ---------------------------------------------------------------------------


def compute_blended_score(
    equivalence_score: float,
    improvement_scores: list[ImprovementScore],
    improvement_weight: float = 0.3,
) -> float:
    """Blend equivalence score with improvement scores.

    Formula::

        (1 - improvement_weight) * equivalence_score
        + improvement_weight * avg(improvement_scores)

    Result is clamped to [0.0, 1.0].

    Args:
        equivalence_score: Semantic equivalence score in [0, 1].
        improvement_scores: Per-objective improvement scores.
        improvement_weight: How much weight to give improvement (default 0.3).

    Returns:
        Blended score in [0.0, 1.0].
    """
    if not improvement_scores:
        avg_improvement = 0.0
    else:
        avg_improvement = sum(s.score for s in improvement_scores) / len(improvement_scores)

    blended = (1.0 - improvement_weight) * equivalence_score + improvement_weight * avg_improvement
    return max(0.0, min(1.0, blended))
