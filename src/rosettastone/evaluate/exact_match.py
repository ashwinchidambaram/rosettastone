"""Exact match and string similarity for classifications."""

from __future__ import annotations

from difflib import SequenceMatcher

from rosettastone.evaluate.base import Evaluator


def string_similarity(expected: str, actual: str) -> float:
    """Basic string similarity using SequenceMatcher (no external deps)."""
    return SequenceMatcher(None, expected.strip(), actual.strip()).ratio()


class ExactMatchEvaluator(Evaluator):
    def score(self, expected: str, actual: str) -> dict[str, float]:
        is_match = expected.strip().lower() == actual.strip().lower()
        return {
            "exact_match": 1.0 if is_match else 0.0,
            "string_similarity": string_similarity(expected, actual),
        }
