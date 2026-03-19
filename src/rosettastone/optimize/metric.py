"""DSPy metric function for behavioral equivalence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig


def build_migration_metric(config: MigrationConfig) -> Any:
    """Build a DSPy-compatible metric function for GEPA.

    Returns dspy.Prediction(score=..., feedback=...) for GEPA's reflective optimization.
    """

    def migration_metric(
        gold: Any,
        pred: Any,
        trace: Any = None,
        pred_name: Any = None,
        pred_trace: Any = None,
    ) -> dspy.Prediction:
        expected = gold.expected_response
        actual = pred.response
        feedback_parts: list[str] = []

        # Semantic similarity — try BERTScore first, fall back to basic similarity
        try:
            from rosettastone.evaluate.bertscore import compute_bertscore

            sem_score = compute_bertscore(expected, actual)
        except ImportError:
            try:
                from rosettastone.evaluate.embedding import compute_embedding_sim

                sem_score = compute_embedding_sim(expected, actual)
            except ImportError:
                from rosettastone.evaluate.exact_match import string_similarity

                sem_score = string_similarity(expected, actual)

        score = sem_score

        if sem_score < 0.7:
            feedback_parts.append(
                f"Response diverges significantly from expected (similarity: {sem_score:.2f}). "
                f"Expected style/content: '{expected[:200]}...'"
            )
        elif sem_score < 0.85:
            feedback_parts.append(
                f"Response partially matches (similarity: {sem_score:.2f}). "
                f"Check formatting and completeness."
            )
        else:
            feedback_parts.append(f"Good match (similarity: {sem_score:.2f})")

        feedback = "\n".join(feedback_parts) if feedback_parts else ""
        return dspy.Prediction(score=min(score, 1.0), feedback=feedback)

    return migration_metric
