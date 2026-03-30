"""SQLModel table definitions for persisting migration results."""

from datetime import UTC, datetime
from typing import Optional

try:
    from sqlmodel import Field, Relationship, SQLModel
except ImportError:
    raise ImportError("Web dependencies required. Install with: uv pip install 'rosettastone[web]'")


class RegisteredModel(SQLModel, table=True):
    __tablename__ = "registered_models"

    id: int | None = Field(default=None, primary_key=True)
    model_id: str = Field(unique=True, index=True)  # LiteLLM identifier e.g. "openai/gpt-4o"
    added_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = Field(default=True)


class MigrationRecord(SQLModel, table=True):
    __tablename__ = "migrations"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: str = Field(default="pending")  # pending / running / complete / failed

    source_model: str
    target_model: str

    optimized_prompt: str | None = None
    confidence_score: float | None = None
    baseline_score: float | None = None
    improvement: float | None = None
    cost_usd: float = 0.0
    duration_seconds: float = 0.0

    recommendation: str | None = None  # GO / NO_GO / CONDITIONAL
    recommendation_reasoning: str | None = None

    # Latency metrics (p50/p95 in seconds, sampled after migration completes)
    source_latency_p50: float | None = None
    source_latency_p95: float | None = None
    target_latency_p50: float | None = None
    target_latency_p95: float | None = None

    # Cost projection (per-call cost in USD)
    projected_source_cost_per_call: float | None = None
    projected_target_cost_per_call: float | None = None

    config_json: str = "{}"  # serialized MigrationConfig
    per_type_scores_json: str = "{}"  # serialized TypeStats dict
    warnings_json: str = "[]"  # serialized warnings list
    safety_warnings_json: str = "[]"  # serialized SafetyWarning list

    test_cases: list["TestCaseRecord"] = Relationship(back_populates="migration")
    warning_records: list["WarningRecord"] = Relationship(back_populates="migration")


class TestCaseRecord(SQLModel, table=True):
    __tablename__ = "test_cases"

    id: int | None = Field(default=None, primary_key=True)
    migration_id: int = Field(foreign_key="migrations.id")
    phase: str  # "baseline" or "validation"
    output_type: str  # json / classification / short_text / long_text

    composite_score: float
    is_win: bool
    scores_json: str = "{}"  # serialized metric scores dict
    details_json: str = "{}"  # serialized details dict

    # Structural metadata (always populated, no PII)
    response_length: int = 0  # char count of source response
    new_response_length: int = 0  # char count of target response
    token_count: int | None = None
    new_token_count: int | None = None
    evaluators_used: str | None = None  # comma-separated evaluator names
    fallback_triggered: bool = False

    # Content columns — NULL unless --store-prompt-content
    prompt_text: str | None = None
    response_text: str | None = None
    new_response_text: str | None = None

    migration: Optional["MigrationRecord"] = Relationship(back_populates="test_cases")


class WarningRecord(SQLModel, table=True):
    __tablename__ = "warnings"

    id: int | None = Field(default=None, primary_key=True)
    migration_id: int = Field(foreign_key="migrations.id")
    warning_type: str  # pipeline / safety / preflight
    severity: str | None = None  # HIGH / MEDIUM / LOW
    message: str

    migration: Optional["MigrationRecord"] = Relationship(back_populates="warning_records")


class Alert(SQLModel, table=True):
    __tablename__ = "alerts"

    id: int | None = Field(default=None, primary_key=True)
    alert_type: str  # "deprecation" | "price_change" | "new_model" | "migration_complete" | "migration_failed"  # noqa: E501
    model_id: str | None = None  # related model identifier
    migration_id: int | None = Field(default=None, foreign_key="migrations.id")
    title: str  # short title
    message: str  # detail message
    action: str | None = None  # suggested action
    severity: str = Field(default="info")  # "critical" | "warning" | "info"
    is_read: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata_json: str = Field(default="{}")  # extra data (dates, prices, etc.)


class MigrationVersion(SQLModel, table=True):
    __tablename__ = "migration_versions"

    id: int | None = Field(default=None, primary_key=True)
    migration_id: int = Field(foreign_key="migrations.id", index=True)
    version_number: int
    snapshot_json: str  # Full MigrationRecord state as JSON
    optimized_prompt: str | None = None
    confidence_score: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_by: str | None = None  # user identifier or "system"


class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_log"

    id: int | None = Field(default=None, primary_key=True)
    resource_type: str  # "migration", "model", "ab_test", etc.
    resource_id: int | None = None
    action: str  # "create", "complete", "approve", "rollback", "delete"
    user_id: int | None = None  # null for system actions, FK to users later
    details_json: str = "{}"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ABTest(SQLModel, table=True):
    __tablename__ = "ab_tests"

    id: int | None = Field(default=None, primary_key=True)
    migration_id: int = Field(foreign_key="migrations.id", index=True)
    version_a_id: int = Field(foreign_key="migration_versions.id")
    version_b_id: int = Field(foreign_key="migration_versions.id")
    name: str = ""
    traffic_split: float = 0.5  # fraction assigned to version_a
    status: str = "draft"  # draft / running / concluded
    start_time: datetime | None = None
    end_time: datetime | None = None
    winner: str | None = None  # "a", "b", or "inconclusive"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ABTestResult(SQLModel, table=True):
    __tablename__ = "ab_test_results"

    id: int | None = Field(default=None, primary_key=True)
    ab_test_id: int = Field(foreign_key="ab_tests.id", index=True)
    test_case_id: int | None = None
    assigned_version: str  # "a" or "b"
    score_a: float | None = None
    score_b: float | None = None
    winner: str | None = None  # "a", "b", or "tie"
    details_json: str = "{}"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PipelineRecord(SQLModel, table=True):
    __tablename__ = "pipelines"

    id: int | None = Field(default=None, primary_key=True)
    name: str
    config_yaml: str  # Raw YAML pipeline config
    source_model: str
    target_model: str
    status: str = "pending"  # pending / running / complete / failed
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PipelineStageRecord(SQLModel, table=True):
    __tablename__ = "pipeline_stages"

    id: int | None = Field(default=None, primary_key=True)
    pipeline_id: int = Field(foreign_key="pipelines.id", index=True)
    module_name: str
    status: str = "pending"  # pending / running / complete / failed
    optimized_prompt: str | None = None
    score: float | None = None
    duration_seconds: float = 0.0
