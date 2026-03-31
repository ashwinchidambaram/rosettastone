"""Migration CRUD and trigger endpoints."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import Session, func, select

from rosettastone.server.api.utils import _get_migration_or_404
from rosettastone.server.database import get_session
from rosettastone.server.models import MigrationRecord, TestCaseRecord, WarningRecord
from rosettastone.server.rbac import require_role
from rosettastone.server.schemas import (
    MigrationDetail,
    MigrationSummary,
    PaginatedResponse,
    TestCaseDetail,
    TestCaseSummary,
    TypeScoreStats,
    WarningSchema,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dummy data for UI shell
# ---------------------------------------------------------------------------

DUMMY_MODELS = [
    {
        "id": "openai/gpt-4o",
        "provider": "OpenAI",
        "status": "active",
        "context": "128K",
        "cost_per_1m": "$2.50",
    },
    {
        "id": "anthropic/claude-sonnet-4",
        "provider": "Anthropic",
        "status": "active",
        "context": "200K",
        "cost_per_1m": "$3.00",
    },
    {
        "id": "openai/gpt-4o-mini",
        "provider": "OpenAI",
        "status": "active",
        "context": "128K",
        "cost_per_1m": "$0.15",
    },
    {
        "id": "openai/gpt-4o-0613",
        "provider": "OpenAI",
        "status": "deprecated",
        "context": "8K",
        "cost_per_1m": "$5.00",
        "retirement_date": "Apr 15, 2026",
        "replacement": "openai/gpt-4o",
    },
]

DUMMY_MIGRATIONS = [
    {
        "id": 1,
        "source": "gpt-4o",
        "target": "claude-sonnet-4",
        "recommendation": "Safe to ship",
        "confidence": 92,
        "test_cases": 156,
        "cost": "$2.34",
        "time_ago": "2 minutes ago",
        "status": "complete",
        "baseline": 85,
        "improvement": 7,
        "reasoning": (
            "All output types meet or exceed quality thresholds. JSON validation "
            "passes 100%. Minimal regressions detected in edge cases only."
        ),
        "per_type": [
            {
                "type": "JSON",
                "wins": 48,
                "total": 48,
                "badge": "PASS",
                "desc": "All fields match",
            },
            {
                "type": "Text",
                "wins": 89,
                "total": 96,
                "badge": "PASS",
                "desc": "Strong semantic match",
            },
            {
                "type": "Code",
                "wins": 5,
                "total": 6,
                "badge": "WARN",
                "desc": "Low sample size (6)",
            },
            {
                "type": "Classification",
                "wins": 4,
                "total": 6,
                "badge": "WARN",
                "desc": "Low sample size (6)",
            },
        ],
        "regressions": [
            {
                "tc_id": 42,
                "score": 0.31,
                "title": "Priority classification mismatch",
                "expected": "urgent",
                "got": "high_priority",
            },
            {
                "tc_id": 87,
                "score": 0.45,
                "title": "Truncated JSON response",
                "expected": '{"status": "complete", ...}',
                "got": '{"status": "done"}',
            },
            {
                "tc_id": 103,
                "score": 0.52,
                "title": "Different code formatting",
                "expected": "def foo():",
                "got": "def foo() ->None:",
            },
        ],
    },
    {
        "id": 2,
        "source": "gpt-4o-0613",
        "target": "gpt-4o",
        "recommendation": "Needs review",
        "confidence": 78,
        "test_cases": 43,
        "cost": "$1.12",
        "time_ago": "1 day ago",
        "status": "complete",
        "baseline": 72,
        "improvement": 6,
        "reasoning": (
            "Most output types pass, but classification accuracy dropped 8%. "
            "Recommend human review of edge cases before deployment."
        ),
        "per_type": [],
        "regressions": [],
    },
    {
        "id": 3,
        "source": "gpt-3.5-turbo",
        "target": "gpt-4o-mini",
        "recommendation": "Do not ship",
        "confidence": 61,
        "test_cases": 89,
        "cost": "$0.87",
        "time_ago": "3 days ago",
        "status": "failed",
        "baseline": 58,
        "improvement": 3,
        "reasoning": (
            "Critical schema violations in 16% of JSON outputs. Code generation "
            "quality dropped significantly. Multiple regressions across all output types."
        ),
        "per_type": [
            {
                "type": "JSON",
                "wins": 30,
                "total": 42,
                "badge": "FAIL",
                "desc": "Schema violations in 12 cases",
            },
            {
                "type": "Text",
                "wins": 22,
                "total": 28,
                "badge": "WARN",
                "desc": "Semantic drift detected",
            },
            {
                "type": "Code",
                "wins": 3,
                "total": 11,
                "badge": "FAIL",
                "desc": "Syntax errors in 8 outputs",
            },
            {
                "type": "Classification",
                "wins": 5,
                "total": 8,
                "badge": "FAIL",
                "desc": "37% accuracy drop",
            },
        ],
        "regressions": [
            {
                "tc_id": 7,
                "score": 0.12,
                "title": "Schema Violation: Missing required key 'metadata'",
                "expected": '{"data": ..., "metadata": ...}',
                "got": '{"data": ...}',
            },
            {
                "tc_id": 19,
                "score": 0.22,
                "title": "Syntax error in generated Python",
                "expected": "valid Python function",
                "got": "SyntaxError on line 3",
            },
            {
                "tc_id": 31,
                "score": 0.28,
                "title": "Wrong classification category",
                "expected": "billing_dispute",
                "got": "general_inquiry",
            },
        ],
    },
]

DUMMY_ALERTS = [
    {
        "type": "deprecation",
        "model": "gpt-4o-0613",
        "date": "Apr 15, 2026",
        "days_left": 26,
        "message": "Model retiring in 26 days",
        "action": "Start migration to gpt-4o",
    },
    {
        "type": "price_change",
        "model": "claude-sonnet-4",
        "old_price": "$3.00",
        "new_price": "$2.50",
        "message": "Price decreased 17%",
        "action": "No action needed",
    },
    {
        "type": "new_model",
        "model": "claude-opus-4.6",
        "date": "Feb 5, 2026",
        "message": "New model available",
        "action": "+12% reasoning improvement over claude-opus-4",
    },
]

DUMMY_COSTS = {
    "total_month": "$1,247",
    "potential_savings": "$312",
    "after_optimization": "$935",
    "by_model": [
        {"model": "gpt-4o", "cost": "$823", "pct": 66},
        {"model": "claude-sonnet-4", "cost": "$312", "pct": 25},
        {"model": "gpt-4o-mini", "cost": "$112", "pct": 9},
    ],
    "opportunities": [
        {
            "title": "Switch gpt-4o classification to gpt-4o-mini",
            "savings": "$187/mo",
            "confidence": "94% parity",
        },
        {
            "title": "Batch non-urgent gpt-4o requests",
            "savings": "$125/mo",
            "confidence": "No quality impact",
        },
    ],
}

DUMMY_TEST_CASES: dict[int, dict] = {
    42: {
        "tc_id": 42,
        "is_win": False,
        "composite_score": 0.31,
        "output_type": "Classification",
        "phase": "validation",
        "scores": {"bertscore": 0.54, "embedding": 0.48, "composite": 0.31},
        "prompt": "Content not stored (run with --store-prompt-content)",
        "source_response": '{"priority": "urgent", "category": "billing", "confidence": 0.94}',
        "target_response": (
            '{"priority": "high_priority", "category": "billing", "confidence": 0.91}'
        ),
        "evaluators_used": "bertscore,embedding_similarity",
        "fallback_triggered": False,
    },
    87: {
        "tc_id": 87,
        "is_win": False,
        "composite_score": 0.45,
        "output_type": "JSON",
        "phase": "validation",
        "scores": {"bertscore": 0.61, "embedding": 0.55, "composite": 0.45},
        "prompt": "Content not stored (run with --store-prompt-content)",
        "source_response": '{"status": "complete", "result": {"items": [1, 2, 3], "total": 3}}',
        "target_response": '{"status": "done"}',
        "evaluators_used": "bertscore,json_validator",
        "fallback_triggered": False,
    },
    103: {
        "tc_id": 103,
        "is_win": False,
        "composite_score": 0.52,
        "output_type": "Code",
        "phase": "validation",
        "scores": {"bertscore": 0.68, "embedding": 0.71, "composite": 0.52},
        "prompt": "Content not stored (run with --store-prompt-content)",
        "source_response": "def foo():\n    return 42",
        "target_response": "def foo() ->None:\n    return 42",
        "evaluators_used": "bertscore,embedding_similarity",
        "fallback_triggered": False,
    },
}


# ---------------------------------------------------------------------------
# Template data formatting helpers
# ---------------------------------------------------------------------------


def _format_recommendation(rec: str | None) -> str:
    """Map DB recommendation to human label."""
    mapping = {"GO": "Safe to ship", "CONDITIONAL": "Needs review", "NO_GO": "Do not ship"}
    return mapping.get(rec or "", rec or "Unknown")


def _format_time_ago(dt: datetime) -> str:
    """Format datetime as relative time string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    now = datetime.now(UTC)
    diff_seconds = (now - dt).total_seconds()

    if diff_seconds < 0:
        return "just now"
    elif diff_seconds < 60:
        return "just now"
    elif diff_seconds < 3600:
        minutes = int(diff_seconds // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif diff_seconds < 86400:
        hours = int(diff_seconds // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(diff_seconds // 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"


def _format_cost(cost_usd: float) -> str:
    """Format cost as dollar string."""
    return f"${cost_usd:.2f}"


def _migration_to_template_dict(record: MigrationRecord, session: Session) -> dict:
    """Convert MigrationRecord to the dict shape templates expect."""
    result: dict = {
        "id": record.id,
        "source": (
            record.source_model.split("/")[-1]
            if "/" in record.source_model
            else record.source_model
        ),
        "target": (
            record.target_model.split("/")[-1]
            if "/" in record.target_model
            else record.target_model
        ),
        "source_full": record.source_model,
        "target_full": record.target_model,
        "recommendation": _format_recommendation(record.recommendation),
        "confidence": round((record.confidence_score or 0) * 100),
        "cost": _format_cost(record.cost_usd),
        "time_ago": _format_time_ago(record.created_at),
        "status": record.status,
        "baseline": round((record.baseline_score or 0) * 100),
        "improvement": round((record.improvement or 0) * 100),
        "reasoning": record.recommendation_reasoning or "",
    }

    # Latency/cost metrics
    result["source_latency_p50"] = record.source_latency_p50
    result["source_latency_p95"] = record.source_latency_p95
    result["target_latency_p50"] = record.target_latency_p50
    result["target_latency_p95"] = record.target_latency_p95
    result["projected_source_cost_per_call"] = record.projected_source_cost_per_call
    result["projected_target_cost_per_call"] = record.projected_target_cost_per_call

    # Test case count
    count_stmt = select(func.count()).select_from(TestCaseRecord).where(
        TestCaseRecord.migration_id == record.id
    )
    result["test_cases"] = session.exec(count_stmt).one()

    # Per-type scores
    per_type_raw = json.loads(record.per_type_scores_json)
    per_type = []
    for type_name, stats in per_type_raw.items():
        win_rate = stats.get("win_rate", 0)
        sample_count = stats.get("sample_count", 0)
        wins = round(win_rate * sample_count)
        if win_rate >= 0.9:
            badge = "PASS"
        elif win_rate >= 0.7:
            badge = "WARN"
        else:
            badge = "FAIL"
        if win_rate >= 1.0:
            desc = "All fields match"
        elif sample_count < 10:
            desc = f"Low sample size ({sample_count})"
        elif win_rate < 0.7:
            desc = f"Failures in {sample_count - wins} cases"
        else:
            desc = "Strong semantic match"

        per_type.append(
            {
                "type": type_name.replace("_", " ").title(),
                "wins": wins,
                "total": sample_count,
                "badge": badge,
                "desc": desc,
            }
        )
    result["per_type"] = per_type

    # Regressions: worst non-winning test cases (up to 5)
    tc_stmt = (
        select(TestCaseRecord)
        .where(
            TestCaseRecord.migration_id == record.id,
            TestCaseRecord.is_win == False,  # noqa: E712
        )
        .order_by(TestCaseRecord.composite_score.asc())  # type: ignore[union-attr]
        .limit(5)
    )
    regressions = []
    for tc in session.exec(tc_stmt).all():
        reg: dict = {
            "tc_id": tc.id,
            "score": round(tc.composite_score, 2),
            "title": (
                f"{tc.output_type.replace('_', ' ').title()} regression"
                f" (score: {tc.composite_score:.2f})"
            ),
        }
        if tc.response_text and tc.new_response_text:
            reg["expected"] = tc.response_text[:80]
            reg["got"] = tc.new_response_text[:80]
        else:
            reg["expected"] = "Content not stored"
            reg["got"] = "Content not stored"
        regressions.append(reg)
    result["regressions"] = regressions

    return result


def _test_case_to_diff_dict(tc: TestCaseRecord, migration: MigrationRecord) -> dict:
    """Convert TestCaseRecord to the diff dict shape the template expects."""
    scores = json.loads(tc.scores_json)
    return {
        "tc_id": tc.id,
        "is_win": tc.is_win,
        "composite_score": round(tc.composite_score, 2),
        "output_type": tc.output_type.replace("_", " ").title(),
        "scores": {
            "bertscore": round(
                scores.get("bertscore_f1", scores.get("bertscore", scores.get("bert_score", 0))),
                2,
            ),
            "embedding": round(
                scores.get(
                    "embedding_sim",
                    scores.get("embedding_similarity", scores.get("embedding", 0)),
                ),
                2,
            ),
            "composite": round(tc.composite_score, 2),
        },
        "source_model": (
            migration.source_model.split("/")[-1]
            if "/" in migration.source_model
            else migration.source_model
        ),
        "target_model": (
            migration.target_model.split("/")[-1]
            if "/" in migration.target_model
            else migration.target_model
        ),
        "expected": (
            tc.response_text or "Content not stored (run with --store-prompt-content)"
        ),
        "actual": (
            tc.new_response_text or "Content not stored (run with --store-prompt-content)"
        ),
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _migration_to_summary(record: MigrationRecord) -> MigrationSummary:
    """Convert a MigrationRecord to a MigrationSummary response."""
    return MigrationSummary(
        id=record.id,  # type: ignore[arg-type]
        source_model=record.source_model,
        target_model=record.target_model,
        recommendation=record.recommendation,
        confidence_score=record.confidence_score,
        status=record.status,
        cost_usd=record.cost_usd,
        created_at=record.created_at,
    )


def _test_case_to_summary(tc: TestCaseRecord) -> TestCaseSummary:
    """Convert a TestCaseRecord to a TestCaseSummary response."""
    return TestCaseSummary(
        id=tc.id,  # type: ignore[arg-type]
        phase=tc.phase,
        output_type=tc.output_type,
        composite_score=tc.composite_score,
        is_win=tc.is_win,
        scores=json.loads(tc.scores_json),
        response_length=tc.response_length,
        new_response_length=tc.new_response_length,
        token_count=tc.token_count,
        new_token_count=tc.new_token_count,
        evaluators_used=tc.evaluators_used,
        fallback_triggered=tc.fallback_triggered,
    )


def _test_case_to_detail(tc: TestCaseRecord) -> TestCaseDetail:
    """Convert a TestCaseRecord to a TestCaseDetail response."""
    from rosettastone.server.schemas import DiffData

    diff = None
    if tc.prompt_text or tc.response_text or tc.new_response_text:
        diff = DiffData(
            prompt=tc.prompt_text,
            source_response=tc.response_text,
            target_response=tc.new_response_text,
            available=True,
        )

    return TestCaseDetail(
        id=tc.id,  # type: ignore[arg-type]
        phase=tc.phase,
        output_type=tc.output_type,
        composite_score=tc.composite_score,
        is_win=tc.is_win,
        scores=json.loads(tc.scores_json),
        details=json.loads(tc.details_json),
        response_length=tc.response_length,
        new_response_length=tc.new_response_length,
        token_count=tc.token_count,
        new_token_count=tc.new_token_count,
        evaluators_used=tc.evaluators_used,
        fallback_triggered=tc.fallback_triggered,
        diff=diff,
    )


def _migration_to_detail(record: MigrationRecord, session: Session) -> MigrationDetail:
    """Convert a MigrationRecord to a MigrationDetail response with nested data."""
    # Parse JSON columns
    config = json.loads(record.config_json)
    per_type_scores_raw = json.loads(record.per_type_scores_json)
    warnings = json.loads(record.warnings_json)
    safety_warnings_raw = json.loads(record.safety_warnings_json)

    # Convert per-type scores to TypeScoreStats
    per_type_scores: dict[str, TypeScoreStats] = {}
    for output_type, stats in per_type_scores_raw.items():
        ci = stats.get("confidence_interval", [0.0, 0.0])
        per_type_scores[output_type] = TypeScoreStats(
            win_rate=stats.get("win_rate", 0.0),
            mean=stats.get("mean", 0.0),
            median=stats.get("median", 0.0),
            p10=stats.get("p10", 0.0),
            p50=stats.get("p50", 0.0),
            p90=stats.get("p90", 0.0),
            min_score=stats.get("min_score", 0.0),
            max_score=stats.get("max_score", 0.0),
            sample_count=stats.get("sample_count", 0),
            confidence_interval=(ci[0], ci[1]) if len(ci) >= 2 else (0.0, 0.0),
        )

    # Load safety warnings from WarningRecord table
    safety_stmt = select(WarningRecord).where(
        WarningRecord.migration_id == record.id,
        WarningRecord.warning_type == "safety",
    )
    safety_records = list(session.exec(safety_stmt).all())
    safety_warnings = [
        WarningSchema(
            id=w.id,  # type: ignore[arg-type]
            warning_type=w.warning_type,
            severity=w.severity,
            message=w.message,
        )
        for w in safety_records
    ]

    # If no WarningRecords, fall back to serialized safety_warnings_json
    if not safety_warnings and safety_warnings_raw:
        for i, sw in enumerate(safety_warnings_raw):
            if isinstance(sw, dict):
                safety_warnings.append(
                    WarningSchema(
                        id=-(i + 1),  # negative sentinel for non-persisted warnings
                        warning_type=sw.get("warning_type", "safety"),
                        severity=sw.get("severity"),
                        message=sw.get("message", str(sw)),
                    )
                )

    # Load test cases
    tc_stmt = select(TestCaseRecord).where(TestCaseRecord.migration_id == record.id)
    test_cases = list(session.exec(tc_stmt).all())
    test_case_summaries = [_test_case_to_summary(tc) for tc in test_cases]

    # Extract cluster summary if clustering was enabled
    cluster_summary = None
    if config.get("cluster_prompts"):
        cluster_info = config.get("cluster_summary")
        if cluster_info:
            cluster_summary = {
                "n_clusters": cluster_info.get("n_clusters"),
                "silhouette_score": cluster_info.get("silhouette_score"),
                "original_pairs": cluster_info.get("original_pairs"),
                "representative_pairs": cluster_info.get("representative_pairs"),
            }

    return MigrationDetail(
        id=record.id,  # type: ignore[arg-type]
        source_model=record.source_model,
        target_model=record.target_model,
        status=record.status,
        created_at=record.created_at,
        optimized_prompt=record.optimized_prompt,
        confidence_score=record.confidence_score,
        baseline_score=record.baseline_score,
        improvement=record.improvement,
        cost_usd=record.cost_usd,
        duration_seconds=record.duration_seconds,
        recommendation=record.recommendation,
        recommendation_reasoning=record.recommendation_reasoning,
        source_latency_p50=record.source_latency_p50,
        source_latency_p95=record.source_latency_p95,
        target_latency_p50=record.target_latency_p50,
        target_latency_p95=record.target_latency_p95,
        projected_source_cost_per_call=record.projected_source_cost_per_call,
        projected_target_cost_per_call=record.projected_target_cost_per_call,
        config=config,
        per_type_scores=per_type_scores,
        warnings=warnings,
        safety_warnings=safety_warnings,
        cluster_summary=cluster_summary,
        test_cases=test_case_summaries,
    )


# ---------------------------------------------------------------------------
# JSON API endpoints
# ---------------------------------------------------------------------------


@router.get("/api/v1/migrations", response_model=PaginatedResponse[MigrationSummary])
async def list_migrations(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> PaginatedResponse[MigrationSummary]:
    """List migrations with pagination."""
    count_stmt = select(func.count()).select_from(MigrationRecord)
    total = session.exec(count_stmt).one()

    stmt = (
        select(MigrationRecord)
        .order_by(MigrationRecord.created_at.desc())  # type: ignore[union-attr]
        .offset(offset)
        .limit(limit)
    )
    records = list(session.exec(stmt).all())

    page = (offset // limit) + 1 if limit > 0 else 1
    return PaginatedResponse(
        items=[_migration_to_summary(r) for r in records],
        total=total,
        page=page,
        per_page=limit,
    )


@router.get("/api/v1/migrations/{migration_id}", response_model=MigrationDetail)
async def get_migration(
    migration_id: int,
    session: Session = Depends(get_session),
) -> MigrationDetail:
    """Get migration detail by ID."""
    record = _get_migration_or_404(migration_id, session)
    return _migration_to_detail(record, session)


@router.post(
    "/api/v1/migrations",
    response_model=MigrationSummary,
    status_code=201,
    dependencies=[Depends(require_role("editor", "admin"))],
)
async def create_migration(
    request: Request,
    session: Session = Depends(get_session),
) -> MigrationSummary:
    """Create a new migration record (status: pending)."""
    body = await request.json()
    source_model = body.get("source_model")
    target_model = body.get("target_model")
    data_path = body.get("data_path")
    cluster_prompts = body.get("cluster_prompts", False)
    improvement_objectives = body.get("improvement_objectives")

    if not source_model or not target_model:
        raise HTTPException(status_code=422, detail="source_model and target_model are required")

    config = {
        "source_model": source_model,
        "target_model": target_model,
    }
    if data_path:
        config["data_path"] = data_path
    if cluster_prompts:
        config["cluster_prompts"] = cluster_prompts
    if improvement_objectives is not None:
        config["improvement_objectives"] = improvement_objectives

    record = MigrationRecord(
        source_model=source_model,
        target_model=target_model,
        status="pending",
        created_at=datetime.now(UTC),
        config_json=json.dumps(config),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return _migration_to_summary(record)


@router.get(
    "/api/v1/migrations/{migration_id}/test-cases",
    response_model=PaginatedResponse[TestCaseSummary],
)
async def list_test_cases(
    migration_id: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    phase: str | None = Query(None),
    output_type: str | None = Query(None),
    session: Session = Depends(get_session),
) -> PaginatedResponse[TestCaseSummary]:
    """List test cases for a migration with optional filters."""
    _get_migration_or_404(migration_id, session)

    # Build base query
    conditions = [TestCaseRecord.migration_id == migration_id]
    if phase:
        conditions.append(TestCaseRecord.phase == phase)
    if output_type:
        conditions.append(TestCaseRecord.output_type == output_type)

    count_stmt = select(func.count()).select_from(TestCaseRecord).where(*conditions)
    total = session.exec(count_stmt).one()

    stmt = select(TestCaseRecord).where(*conditions).offset(offset).limit(limit)
    records = list(session.exec(stmt).all())

    page = (offset // limit) + 1 if limit > 0 else 1
    return PaginatedResponse(
        items=[_test_case_to_summary(tc) for tc in records],
        total=total,
        page=page,
        per_page=limit,
    )


@router.get(
    "/api/v1/migrations/{migration_id}/test-cases/{tc_id}",
    response_model=TestCaseDetail,
)
async def get_test_case(
    migration_id: int,
    tc_id: int,
    session: Session = Depends(get_session),
) -> TestCaseDetail:
    """Get a single test case detail."""
    _get_migration_or_404(migration_id, session)

    tc = session.get(TestCaseRecord, tc_id)
    if tc is None or tc.migration_id != migration_id:
        raise HTTPException(status_code=404, detail="Test case not found")
    return _test_case_to_detail(tc)


# ---------------------------------------------------------------------------
# HTMX / UI endpoints
# ---------------------------------------------------------------------------


@router.get("/ui/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    empty: str | None = Query(None),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """Models dashboard page."""
    from rosettastone.server.api.models import _model_to_template_dict
    from rosettastone.server.models import RegisteredModel

    # Query registered models ordered by most recent first
    stmt = select(RegisteredModel).order_by(RegisteredModel.added_at.desc())  # type: ignore[union-attr]
    records = list(session.exec(stmt).all())

    if not records and empty != "false":
        # No models registered — show empty state
        return request.app.state.templates.TemplateResponse(
            request, "models_empty.html", {"active_nav": "models"},
        )

    models = [_model_to_template_dict(r) for r in records] if records else DUMMY_MODELS
    return request.app.state.templates.TemplateResponse(
        request, "models.html",
        {"models": models, "alerts": DUMMY_ALERTS, "active_nav": "models"},
    )


@router.get("/ui/migrations", response_class=HTMLResponse)
async def migrations_page(
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """Migrations list page."""
    stmt = select(MigrationRecord).order_by(MigrationRecord.created_at.desc()).limit(50)  # type: ignore[union-attr]
    records = list(session.exec(stmt).all())

    if records:
        migrations = [_migration_to_template_dict(r, session) for r in records]
    else:
        migrations = DUMMY_MIGRATIONS  # fallback when DB is empty

    return request.app.state.templates.TemplateResponse(
        request, "migrations.html",
        {"migrations": migrations, "active_nav": "migrations"},
    )


MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB


@router.get("/ui/migrations/new", response_class=HTMLResponse)
async def new_migration_form(
    request: Request,
    source: str | None = Query(None),
) -> HTMLResponse:
    """Render the new migration form."""
    return request.app.state.templates.TemplateResponse(
        request, "migration_new.html",
        {"active_nav": "migrations", "source_model": source or "", "error": None},
    )


@router.post("/ui/migrations/new")
async def create_migration_from_form(
    request: Request,
    source_model: str = Form(...),
    target_model: str = Form(...),
    data_file: UploadFile = None,  # type: ignore[assignment]
    gepa_auto: str = Form("light"),
    dry_run: str | None = Form(None),
    store_prompt_content: str | None = Form(None),
    reflection_model: str = Form("openai/gpt-4o"),
    judge_model: str = Form("openai/gpt-4o"),
    session: Session = Depends(get_session),
):
    """Handle form submission: validate, save file, create record, submit to executor."""
    templates = request.app.state.templates

    # Validate file upload
    if data_file is None or data_file.filename == "":
        return templates.TemplateResponse(
            request, "migration_new.html",
            {
                "active_nav": "migrations",
                "source_model": source_model,
                "target_model": target_model,
                "error": "Please upload a JSONL data file.",
            },
            status_code=422,
        )

    # Check file size
    content = await data_file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        return templates.TemplateResponse(
            request, "migration_new.html",
            {
                "active_nav": "migrations",
                "source_model": source_model,
                "target_model": target_model,
                "error": "File size exceeds 50MB limit.",
            },
            status_code=422,
        )

    # UTF-8 validation
    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        return templates.TemplateResponse(
            request, "migration_new.html",
            {
                "active_nav": "migrations",
                "source_model": source_model,
                "target_model": target_model,
                "error": "File is not valid UTF-8 text.",
            },
            status_code=422,
        )

    # JSONL parse validation — check first non-empty line
    first_line = ""
    for line in text_content.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break

    if not first_line:
        return templates.TemplateResponse(
            request, "migration_new.html",
            {
                "active_nav": "migrations",
                "source_model": source_model,
                "target_model": target_model,
                "error": "File is empty or contains no data.",
            },
            status_code=422,
        )

    try:
        first_obj = json.loads(first_line)
    except json.JSONDecodeError as e:
        return templates.TemplateResponse(
            request, "migration_new.html",
            {
                "active_nav": "migrations",
                "source_model": source_model,
                "target_model": target_model,
                "error": f"First line is not valid JSON: {e}",
            },
            status_code=422,
        )

    # Schema validation — must have prompt and response keys
    if not isinstance(first_obj, dict) or "prompt" not in first_obj or "response" not in first_obj:
        return templates.TemplateResponse(
            request, "migration_new.html",
            {
                "active_nav": "migrations",
                "source_model": source_model,
                "target_model": target_model,
                "error": "JSONL entries must contain 'prompt' and 'response' fields.",
            },
            status_code=422,
        )

    # Create the migration record first to get the ID
    record = MigrationRecord(
        source_model=source_model,
        target_model=target_model,
        status="pending",
        created_at=datetime.now(UTC),
        config_json=json.dumps({
            "source_model": source_model,
            "target_model": target_model,
        }),
    )
    session.add(record)
    session.commit()
    session.refresh(record)

    migration_id = record.id

    # Save uploaded file to per-migration directory
    from pathlib import Path

    output_dir = Path.home() / ".rosettastone" / "migrations" / str(migration_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.jsonl"
    data_path.write_bytes(content)

    # Build config dict for the background task
    config_dict = {
        "source_model": source_model,
        "target_model": target_model,
        "data_path": str(data_path),
        "gepa_auto": gepa_auto,
        "dry_run": dry_run == "true",
        "store_prompt_content": store_prompt_content == "true",
        "reflection_model": reflection_model,
        "judge_model": judge_model,
    }

    # Submit to background executor
    from rosettastone.server.api.tasks import run_migration_background

    executor = request.app.state.executor
    executor.submit(run_migration_background, migration_id, config_dict)

    # Redirect to the migration detail page (303 See Other)
    return RedirectResponse(
        url=f"/ui/migrations/{migration_id}",
        status_code=303,
    )


@router.get("/ui/migrations/{migration_id}", response_class=HTMLResponse)
async def migration_detail_page(
    migration_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """Migration detail page."""
    record = session.get(MigrationRecord, migration_id)

    if record:
        migration = _migration_to_template_dict(record, session)
    else:
        # Fall back to dummy data
        migration = next((m for m in DUMMY_MIGRATIONS if m["id"] == migration_id), None)
        if migration is None:
            raise HTTPException(status_code=404, detail="Migration not found")

    return request.app.state.templates.TemplateResponse(
        request, "migration_detail.html",
        {"migration": migration, "active_nav": "migrations"},
    )


@router.get("/ui/migrations/{migration_id}/executive", response_class=HTMLResponse)
async def executive_report_page(
    migration_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """Executive report page (standalone, print-ready)."""
    record = session.get(MigrationRecord, migration_id)

    if record:
        migration = _migration_to_template_dict(record, session)
    else:
        migration = next((m for m in DUMMY_MIGRATIONS if m["id"] == migration_id), None)
        if migration is None:
            raise HTTPException(status_code=404, detail="Migration not found")

    report_date = datetime.now(UTC).strftime("%B %-d, %Y")
    return request.app.state.templates.TemplateResponse(
        request, "executive_report.html",
        {"migration": migration, "report_date": report_date, "active_nav": "migrations"},
    )


@router.get("/ui/costs", response_class=HTMLResponse)
async def costs_page(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    """Costs overview page."""
    from rosettastone.server.api.costs import _compute_costs

    costs = _compute_costs(session)
    if costs is None:
        costs = DUMMY_COSTS  # fallback when no data

    return request.app.state.templates.TemplateResponse(
        request, "costs.html", {"costs": costs, "active_nav": "costs"},
    )


@router.get("/ui/alerts", response_class=HTMLResponse)
async def alerts_page(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    """Alerts hub page."""
    from rosettastone.server.api.alerts import _alert_to_template_dict, _generate_alerts
    from rosettastone.server.models import Alert

    # Auto-generate alerts from current data (idempotent)
    _generate_alerts(session)

    # Fetch all alerts newest first
    stmt = select(Alert).order_by(Alert.created_at.desc()).limit(50)  # type: ignore[union-attr]
    records = list(session.exec(stmt).all())

    if records:
        alerts = [_alert_to_template_dict(a) for a in records]
    else:
        alerts = DUMMY_ALERTS  # fallback when DB is empty

    return request.app.state.templates.TemplateResponse(
        request, "alerts.html", {"alerts": alerts, "active_nav": "alerts"},
    )


@router.get("/ui/fragments/migration-list", response_class=HTMLResponse)
async def migration_list_fragment(request: Request) -> HTMLResponse:
    """HTMX partial for migration list."""
    return HTMLResponse("<div>Template pending integration</div>")


@router.get("/ui/fragments/eval-grid/{migration_id}", response_class=HTMLResponse)
async def eval_grid_fragment(migration_id: int, request: Request) -> HTMLResponse:
    """HTMX partial for evaluation grid."""
    return HTMLResponse("<div>Template pending integration</div>")


@router.get("/ui/fragments/test-case/{migration_id}/{tc_id}", response_class=HTMLResponse)
async def test_case_fragment(
    migration_id: int,
    tc_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """HTMX partial for test case detail."""
    migration = session.get(MigrationRecord, migration_id)
    db_tc = session.get(TestCaseRecord, tc_id) if migration else None

    if db_tc and db_tc.migration_id == migration_id:
        scores = json.loads(db_tc.scores_json)
        tc = {
            "tc_id": db_tc.id,
            "is_win": db_tc.is_win,
            "composite_score": round(db_tc.composite_score, 2),
            "output_type": db_tc.output_type.replace("_", " ").title(),
            "phase": db_tc.phase,
            "scores": {
                "bertscore": round(
                    scores.get(
                        "bertscore_f1", scores.get("bertscore", scores.get("bert_score", 0))
                    ),
                    2,
                ),
                "embedding": round(
                    scores.get(
                        "embedding_sim",
                        scores.get("embedding_similarity", scores.get("embedding", 0)),
                    ),
                    2,
                ),
                "composite": round(db_tc.composite_score, 2),
            },
            "prompt": db_tc.prompt_text,
            "source_response": db_tc.response_text,
            "target_response": db_tc.new_response_text,
            "evaluators_used": db_tc.evaluators_used,
            "fallback_triggered": db_tc.fallback_triggered,
        }
    else:
        tc = DUMMY_TEST_CASES.get(
            tc_id,
            DUMMY_TEST_CASES[42],  # default fallback
        )

    return request.app.state.templates.TemplateResponse(
        request, "fragments/test_case_detail.html", {"tc": tc},
    )
