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
    failure_reason: str | None = None


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
    failure_reason: str | None = None
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

    cluster_summary: dict[str, object] | None = None

    # Phase A observability fields
    stage_timing: dict[str, float] = {}
    non_deterministic_count: int = 0
    eval_runs: int = 1

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


class ABTestCreate(BaseModel):
    migration_id: int
    version_a_id: int
    version_b_id: int
    name: str = ""
    traffic_split: float = 0.5


class ABTestSummary(BaseModel):
    id: int
    migration_id: int
    name: str
    status: str
    traffic_split: float
    winner: str | None
    created_at: datetime


class ABTestMetrics(BaseModel):
    total_results: int
    wins_a: int
    wins_b: int
    ties: int
    win_rate_a: float
    win_rate_b: float
    mean_score_a: float
    mean_score_b: float
    chi2: float | None = None
    p_value: float | None = None
    significant: bool | None = None
    mean_diff: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None


class ABTestDetail(BaseModel):
    id: int
    migration_id: int
    version_a_id: int
    version_b_id: int
    name: str
    status: str
    traffic_split: float
    start_time: datetime | None
    end_time: datetime | None
    winner: str | None
    created_at: datetime
    metrics: ABTestMetrics | None = None


class ABTestResultEntry(BaseModel):
    id: int
    test_case_id: int | None
    assigned_version: str
    score_a: float | None
    score_b: float | None
    winner: str | None
    created_at: datetime


class PipelineCreate(BaseModel):
    config_yaml: str
    source_model: str
    target_model: str


class PipelineStageSummary(BaseModel):
    module_name: str
    status: str
    optimized_prompt: str | None
    score: float | None
    duration_seconds: float | None


class PipelineSummary(BaseModel):
    id: int
    name: str
    source_model: str
    target_model: str
    status: str
    created_at: datetime


class PipelineDetail(PipelineSummary):
    stages: list[PipelineStageSummary]
    config_yaml: str


# ---------------------------------------------------------------------------
# Task 5.5.6 — Team management schemas
# ---------------------------------------------------------------------------


class TeamCreate(BaseModel):
    name: str


class TeamSummary(BaseModel):
    id: int
    name: str
    created_at: datetime


class TeamMemberSummary(BaseModel):
    user_id: int
    team_id: int
    role: str


class AddTeamMember(BaseModel):
    user_id: int
    role: str = "member"


# ---------------------------------------------------------------------------
# Task 5.5.3c — Multi-user auth schemas
# ---------------------------------------------------------------------------


class UserLogin(BaseModel):
    username: str
    password: str


class UserRegister(BaseModel):
    username: str
    password: str
    email: str | None = None
    role: str = "viewer"  # Default role for new users


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    role: str


class UserMe(BaseModel):
    id: int
    username: str
    email: str | None
    role: str
    is_active: bool


class UserCreate(BaseModel):
    username: str
    password: str
    email: str | None = None
    role: str = "viewer"


class UserUpdate(BaseModel):
    role: str | None = None
    password: str | None = None
    email: str | None = None
    is_active: bool | None = None


# ---------------------------------------------------------------------------
# Task 5.5.7 — Annotation schemas
# ---------------------------------------------------------------------------


class AnnotationCreate(BaseModel):
    migration_id: int
    test_case_id: int | None = None
    annotation_type: str
    text: str


class AnnotationSummary(BaseModel):
    id: int
    migration_id: int
    test_case_id: int | None
    annotator_id: int | None
    annotation_type: str
    text: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Task 5.5.8 — Approval workflow schemas
# ---------------------------------------------------------------------------


class ApprovalWorkflowCreate(BaseModel):
    required_approvals: int = 1


class ApprovalWorkflowSummary(BaseModel):
    id: int
    migration_id: int
    required_approvals: int
    status: str
    current_approvals: int


class ApprovalCreate(BaseModel):
    decision: str  # "approve" / "reject"
    comment: str | None = None


class ApprovalSummary(BaseModel):
    id: int
    workflow_id: int
    user_id: int | None
    decision: str
    comment: str | None
    created_at: datetime


class GEPAIterationOut(BaseModel):
    iteration: int
    total_iterations: int
    mean_score: float
    recorded_at: datetime
