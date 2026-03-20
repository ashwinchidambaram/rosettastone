"""Abstract base class for evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig


class Evaluator(ABC):
    def __init__(self, config: MigrationConfig | None = None) -> None:
        self.config = config

    @abstractmethod
    def score(self, expected: str, actual: str, **kwargs: Any) -> dict[str, float]:
        """Score a single expected/actual pair. Returns metric name -> score mapping."""
        ...
