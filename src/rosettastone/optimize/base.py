"""Abstract base class for optimizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.types import PromptPair


class Optimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        train_set: list[PromptPair],
        val_set: list[PromptPair],
        config: MigrationConfig,
    ) -> str:
        """Run optimization and return the optimized prompt text."""
        ...
