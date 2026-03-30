"""Pydantic API response schemas for the web server."""

from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class MigrationSummary(BaseModel):
    id: int
    source_model: str
    target_model: str
    recommendation: str | None
    confidence_score: float | None
    status: str
    cost_usd: float
    created_at: datetime


class TestCaseSummary(BaseModel):
    id: int
    phase: str
    output_type: str
    composite_score: float
    is_win: bool
    scores: dict[str, float]
    response_length: int
    new_response_length: int
    token_count: int | None
    new_token_count: int | None
    evaluators_used: str | None
    fallback_triggered: bool


class TestCaseDetail(BaseModel):
    id: int
    phase: str
    output_type: str
    composite_score: float
    is_win: bool
    scores: dict[str, float]
    details: dict[str, object]
    response_length: int
    new_response_length: int
    token_count: int | None
    new_token_count: int | None
    evaluators_used: str | None
    fallback_triggered: bool
    diff: DiffData | None = None


class DiffData(BaseModel):
    prompt: str | None = None
    source_response: str | None = None
    target_response: str | None = None
    available: bool = False


class WarningSchema(BaseModel):
    id: int
    warning_type: str
    severity: str | None
    message: str


class TypeScoreStats(BaseModel):
    win_rate: float
    mean: float
    median: float
    p10: float
    p50: float
    p90: float
    min_score: float
    max_score: float
    sample_count: int
    confidence_interval: tuple[float, float]


class ScoreDistribution(BaseModel):
    output_type: str
    stats: TypeScoreStats
    histogram: list[int] = []  # bucket counts for chart rendering


class MigrationDetail(BaseModel):
    id: int
    source_model: str
    target_model: str
    status: str
    created_at: datetime

    optimized_prompt: str | None
    confidence_score: float | None
    baseline_score: float | None
    improvement: float | None
    cost_usd: float
    duration_seconds: float

    recommendation: str | None
    recommendation_reasoning: str | None

    # Latency metrics
    source_latency_p50: float | None = None
    source_latency_p95: float | None = None
    target_latency_p50: float | None = None
    target_latency_p95: float | None = None

    # Cost projection
    projected_source_cost_per_call: float | None = None
    projected_target_cost_per_call: float | None = None

    config: dict[str, object]
    per_type_scores: dict[str, TypeScoreStats]
    warnings: list[str]
    safety_warnings: list[WarningSchema]

    test_cases: list[TestCaseSummary]


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    per_page: int


class MigrationVersionSummary(BaseModel):
    id: int
    migration_id: int
    version_number: int
    confidence_score: float | None
    created_at: datetime
    created_by: str | None


class MigrationVersionDetail(BaseModel):
    id: int
    migration_id: int
    version_number: int
    snapshot: dict[str, object]
    optimized_prompt: str | None
    confidence_score: float | None
    created_at: datetime
    created_by: str | None


class AuditLogEntry(BaseModel):
    id: int
    resource_type: str
    resource_id: int | None
    action: str
    user_id: int | None
    details: dict[str, object]
    created_at: datetime
