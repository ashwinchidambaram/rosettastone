"""BERTScore wrapper (optional dependency)."""

from __future__ import annotations

from typing import Any

from rosettastone.evaluate.base import Evaluator


def compute_bertscore(expected: str, actual: str) -> float:
    """Compute BERTScore F1 between expected and actual text.

    Raises ImportError if bert-score is not installed.
    """
    from bert_score import score as bert_score

    _p, _r, f1 = bert_score(
        [actual],
        [expected],
        model_type="distilbert-base-uncased",
        verbose=False,
    )
    return float(f1[0])


class BERTScoreEvaluator(Evaluator):
    def score(self, expected: str, actual: str, **kwargs: Any) -> dict[str, float]:
        return {"bertscore_f1": compute_bertscore(expected, actual)}
