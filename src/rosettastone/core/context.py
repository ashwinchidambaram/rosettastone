"""Shared context dataclasses used across pipeline steps and the decision layer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TypeStats:
    """Aggregate statistics for a single output type."""

    win_rate: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    p10: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    sample_count: int = 0
    confidence_interval: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
