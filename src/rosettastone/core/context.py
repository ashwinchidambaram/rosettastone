"""Pipeline context: accumulates warnings, costs, timing, and per-type stats."""

from __future__ import annotations

import threading
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
    confidence_interval: tuple[float, float] = (0.0, 0.0)


@dataclass
class PipelineContext:
    warnings: list[str] = field(default_factory=list)
    safety_warnings: list[SafetyWarning] = field(default_factory=list)
    costs: dict[str, float] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    per_type_stats: dict[OutputType, TypeStats] = field(default_factory=dict)
    _current_phase: str = ""
    _cost_lock: threading.Lock = field(default_factory=threading.Lock)

    def register_cost_callback(self, phase: str) -> None:
        """Register a LiteLLM success callback to accumulate costs for the given phase."""
        import litellm

        self._current_phase = phase

        def _track_cost(kwargs: dict, completion_response: object, **cb_kwargs: object) -> None:
            cost = getattr(completion_response, "_hidden_params", {}).get("response_cost", 0)
            if cost and cost > 0:
                with self._cost_lock:
                    self.costs[self._current_phase] = (
                        self.costs.get(self._current_phase, 0.0) + cost
                    )

        if _track_cost not in litellm.success_callback:
            litellm.success_callback.append(_track_cost)  # type: ignore[arg-type]
