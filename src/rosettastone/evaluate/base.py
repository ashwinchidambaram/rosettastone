"""Abstract base class for evaluators."""

from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    def score(self, expected: str, actual: str) -> dict[str, float]:
        """Score a single expected/actual pair. Returns metric name -> score mapping."""
        ...
