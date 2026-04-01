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


def batch_compute_bertscore(
    pairs: list[tuple[str, str]],
    lang: str = "en",
    model_type: str = "distilbert-base-uncased",
) -> list[float]:
    """Compute BERTScore F1 for a list of (actual, expected) pairs in a single batch call.

    Raises ImportError if bert-score is not installed.

    Args:
        pairs: List of (actual, expected) string tuples.
        lang: Language code passed to bert_score (default "en").
        model_type: BERTScore model type (default "distilbert-base-uncased").

    Returns:
        List of F1 scores, one per pair, in the same order as input.
    """
    if not pairs:
        return []

    from bert_score import score as bert_score

    actuals = [a for a, _ in pairs]
    expecteds = [e for _, e in pairs]

    _p, _r, f1 = bert_score(
        actuals,
        expecteds,
        model_type=model_type,
        verbose=False,
    )
    return [float(v) for v in f1]


class BERTScoreEvaluator(Evaluator):
    def score(self, expected: str, actual: str, **kwargs: Any) -> dict[str, float]:
        return {"bertscore_f1": compute_bertscore(expected, actual)}
