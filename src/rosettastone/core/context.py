"""Pipeline context: accumulates warnings, costs, timing, and per-type stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from rosettastone.core.types import OutputType


class SafetySeverity(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class SafetyWarning:
    warning_type: str
    severity: SafetySeverity
    message: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class TypeStats:
    win_rate: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    p10: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    sample_count: int = 0
    confidence_interval: tuple[float, float] = (0.0, 0.0)


@dataclass
class PipelineContext:
    warnings: list[str] = field(default_factory=list)
    safety_warnings: list[SafetyWarning] = field(default_factory=list)
    costs: dict[str, float] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    per_type_stats: dict[OutputType, TypeStats] = field(default_factory=dict)
