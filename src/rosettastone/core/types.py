from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class OutputType(str, Enum):
    JSON = "json"
    CLASSIFICATION = "classification"
    SHORT_TEXT = "short_text"
    LONG_TEXT = "long_text"


class PromptPair(BaseModel):
    prompt: str | list[dict[str, Any]]
    response: str
    source_model: str
    metadata: dict[str, Any] = {}
    feedback: Optional[str] = None
    output_type: Optional[OutputType] = None


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
