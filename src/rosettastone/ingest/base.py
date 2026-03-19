"""Abstract base class for data adapters."""

from abc import ABC, abstractmethod

from rosettastone.core.types import PromptPair


class DataAdapter(ABC):
    @abstractmethod
    def load(self) -> list[PromptPair]:
        """Load prompt/response pairs from the data source."""
        ...
