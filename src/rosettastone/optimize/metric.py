"""DSPy metric function for behavioral equivalence."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import UTC
from datetime import datetime as _dt
from typing import TYPE_CHECKING, Any

import dspy

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.types import PromptPair


class IterationTracker:
    """Thread-safe counter that fires a callback after each full trainset sweep.

    This class is importable without DSPy — it only depends on stdlib threading.
    If trainset_size is 0 the tracker is a no-op (wrap returns the original function).
    """

    def __init__(
        self,
        trainset_size: int,
        total_iterations: int,
        callback: Callable[[int, int, float], None],
    ) -> None:
        self._lock = threading.Lock()
        self._call_count = 0
        self._trainset_size = trainset_size
        self._total_iterations = total_iterations
        self._iteration = 0
        self._scores: list[float] = []
        self._callback = callback  # (iteration, total_iterations, running_mean_score) -> None
        self._iteration_history: list[dict] = []

    def wrap(self, metric_fn: Any) -> Any:
        """Wrap a DSPy metric function to track iteration progress.

        The wrapper calls the original metric, records the score, and fires the
        callback after every ``trainset_size`` calls (= one full GEPA iteration).
        The callback is invoked *outside* the lock to avoid deadlocks.
        """
        if self._trainset_size == 0:
            return metric_fn

        tracker = self  # avoid closure capture of self in nested def

        def wrapped(
            example: Any,
            pred: Any,
            trace: Any = None,
            pred_name: Any = None,
            pred_trace: Any = None,
        ) -> Any:
            result = metric_fn(example, pred, trace, pred_name, pred_trace)
            score = result.score if hasattr(result, "score") else float(result or 0)
            fire_callback: tuple[int, int, float] | None = None
            with tracker._lock:
                tracker._call_count += 1
                tracker._scores.append(score)
                if tracker._call_count % tracker._trainset_size == 0:
                    tracker._iteration += 1
                    mean = sum(tracker._scores) / len(tracker._scores)
                    fire_callback = (tracker._iteration, tracker._total_iterations, mean)
                    tracker._iteration_history.append({
                        "iteration_num": tracker._iteration,
                        "mean_score": round(mean, 4),
                        "timestamp_iso": _dt.now(UTC).isoformat(),
                    })
            if fire_callback is not None:
                tracker._callback(*fire_callback)
            return result

        return wrapped

    def get_history(self) -> list[dict]:
        """Return a shallow copy of the iteration history under lock."""
        with self._lock:
            return list(self._iteration_history)


def build_migration_metric(
    config: MigrationConfig,
    train_set: list[PromptPair] | None = None,
) -> Any:
    """Build a DSPy-compatible metric function for GEPA/MIPROv2.

    Returns dspy.Prediction(score=..., feedback=...) for GEPA's reflective optimization.

    If train_set is provided and contains pairs with feedback, known issues are
    prepended to the metric feedback for pairs that match.

    If config.improvement_objectives is set, the metric blends equivalence scoring
    with LLM-as-judge improvement scoring using compute_blended_score.
    """
    # Build feedback map if train_set provided
    feedback_map: dict[str, str] = {}
    if train_set:
        from rosettastone.optimize.feedback import build_feedback_map

        feedback_map = build_feedback_map(train_set)

    # Build improvement scorer if objectives are specified
    improvement_scorer = None
    if config.improvement_objectives:
        from rosettastone.optimize.improvement import build_improvement_scorer

        objective_descriptions = [str(obj["description"]) for obj in config.improvement_objectives]
        improvement_scorer = build_improvement_scorer(objective_descriptions, config.judge_model)

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

        # Blend with improvement scores and merge feedback if objectives are specified
        if improvement_scorer is not None:
            from rosettastone.optimize.improvement import (
                build_improvement_feedback,
                compute_blended_score,
            )

            improvement_scores = improvement_scorer(gold.prompt, gold.expected_response, actual)
            score = compute_blended_score(sem_score, improvement_scores)
            feedback = build_improvement_feedback(feedback, improvement_scores)

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
            if known_issue is not None:
                score = score / config.known_issue_weight

        return dspy.Prediction(score=max(0.0, min(score, 1.0)), feedback=feedback)

    return migration_metric
