"""DSPy metric function for behavioral equivalence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import dspy

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.types import PromptPair


def build_migration_metric(
    config: MigrationConfig,
    train_set: list[PromptPair] | None = None,
) -> Any:
    """Build a DSPy-compatible metric function for GEPA/MIPROv2.

    Returns dspy.Prediction(score=..., feedback=...) for GEPA's reflective optimization.

    If train_set is provided and contains pairs with feedback, known issues are
    prepended to the metric feedback for pairs that match.
    """
    # Build feedback map if train_set provided
    feedback_map: dict[str, str] = {}
    if train_set:
        from rosettastone.optimize.feedback import build_feedback_map

        feedback_map = build_feedback_map(train_set)

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
                f"Expected ~{len(expected)} chars. Check structure and intent."
            )
        elif sem_score < 0.85:
            feedback_parts.append(
                f"Response partially matches (similarity: {sem_score:.2f}). "
                f"Check formatting and completeness."
            )
        else:
            feedback_parts.append(f"Good match (similarity: {sem_score:.2f})")

        feedback = "\n".join(feedback_parts) if feedback_parts else ""

        # Prepend known-issue feedback if available
        if feedback_map:
            from rosettastone.optimize.feedback import prepend_feedback

            prompt = gold.prompt
            if isinstance(prompt, str):
                key = prompt
            else:
                key = json.dumps(prompt, sort_keys=True)
            known_issue = feedback_map.get(key)
            feedback = prepend_feedback(feedback, known_issue)

        return dspy.Prediction(score=min(score, 1.0), feedback=feedback)

    return migration_metric
