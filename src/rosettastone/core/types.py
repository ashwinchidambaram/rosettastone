from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel


class CostLimitExceeded(Exception):  # noqa: N818
    """Raised when actual LLM spend exceeds the configured max_cost_usd cap."""

    def __init__(self, actual: float, limit: float) -> None:
        super().__init__(f"Cost cap exceeded: ${actual:.4f} > ${limit:.4f}")
        self.actual = actual
        self.limit = limit


class OutputType(StrEnum):
    JSON = "json"
    CLASSIFICATION = "classification"
    SHORT_TEXT = "short_text"
    LONG_TEXT = "long_text"


class PromptPair(BaseModel):
    prompt: str | list[dict[str, Any]]
    response: str
    source_model: str
    metadata: dict[str, Any] = {}
    feedback: str | None = None
    output_type: OutputType | None = None


class EvalResult(BaseModel):
    prompt_pair: PromptPair
    new_response: str
    scores: dict[str, float]
    composite_score: float
    is_win: bool
    details: dict[str, Any] = {}

    # T4: Multi-run variance tracking
    run_scores: list[float] = []
    score_std: float = 0.0
    is_non_deterministic: bool = False

    # F6: Failure reason taxonomy (categorical only, no PII)
    failure_reason: str | None = None


class PromptRegression(BaseModel):
    prompt_index: int
    output_type: str
    baseline_score: float
    optimized_score: float
    delta: float  # optimized_score - baseline_score
    baseline_is_win: bool
    optimized_is_win: bool
    status: Literal["improved", "stable", "regressed", "at_risk"]
    metric_deltas: dict[str, float] = {}  # per-metric delta


class MigrationResult(BaseModel):
    config: dict[str, Any]
    optimized_prompt: str
    baseline_results: list[EvalResult]
    validation_results: list[EvalResult]
    confidence_score: float
    baseline_score: float
    improvement: float
    cost_usd: float
    duration_seconds: float
    warnings: list[str]

    # Phase 2: Safety, decision, per-type breakdown
    safety_warnings: list[Any] = []
    recommendation: str | None = None
    recommendation_reasoning: str | None = None
    per_type_scores: dict[str, Any] = {}

    # Cost tracking
    cost_breakdown: dict[str, float] = {}
    estimated_cost_usd: float = 0.0

    # T3: Per-prompt regression analysis
    prompt_regressions: list[PromptRegression] = []
    regression_count: int = 0
    at_risk_count: int = 0

    # T4: Multi-run metadata
    non_deterministic_count: int = 0
    eval_runs: int = 1

    # F5: Token tracking
    total_tokens: int = 0
    token_breakdown: dict[str, int] = {}
    stage_timing: dict[str, float] = {}

    # F2: GEPA iteration telemetry
    optimization_iterations: list[dict[str, Any]] = []
