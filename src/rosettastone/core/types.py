from enum import StrEnum
from typing import Any

from pydantic import BaseModel


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
