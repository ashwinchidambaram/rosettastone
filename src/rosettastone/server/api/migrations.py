"""Migration CRUD and trigger endpoints."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import Session, func, select

from rosettastone.server.api.utils import _get_migration_or_404
from rosettastone.server.database import get_session
from rosettastone.server.models import (
    GEPAIterationRecord,
    MigrationRecord,
    TestCaseRecord,
    WarningRecord,
)
from rosettastone.server.rate_limit import check_rate_limit
from rosettastone.server.rbac import (
    check_resource_owner,
    get_current_user_id,
    is_admin_user,
    require_role,
)
from rosettastone.server.schemas import (
    GEPAIterationOut,
    MigrationDetail,
    MigrationDiagnostics,
    MigrationSummary,
    PaginatedResponse,
    TestCaseDetail,
    TestCaseSummary,
    TypeScoreStats,
    WarningSchema,
)

router = APIRouter()

_SENSITIVE_CONFIG_KEYS = frozenset({"lm_extra_kwargs"})
# Generic fallback threshold used when computing per-metric above-threshold counts in diagnostics.
# Individual metric scales vary (0-1 for BERTScore/embedding, binary for exact_match), so this
# represents a reasonable midpoint rather than being sourced from DEFAULT_THRESHOLDS.
_DIAGNOSTIC_METRIC_THRESHOLD = 0.7


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
        "wins": 146,
        "losses": 10,
        "score_histogram": [0, 0, 1, 2, 3, 4, 12, 30, 64, 40],
        # Phase A observability fields
        "non_deterministic_count": 0,
        "eval_runs": 1,
        "stage_timing": {},
        # Feature 6 failure reason fields
        "skipped_count": 0,
        "skipped_pairs_by_reason": {},
        # Latency/checkpoint fields
        "checkpoint_stage": None,
        "source_latency_p50": None,
        "source_latency_p95": None,
        "target_latency_p50": None,
        "target_latency_p95": None,
        "projected_source_cost_per_call": None,
        "projected_target_cost_per_call": None,
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
        "wins": 34,
        "losses": 9,
        "score_histogram": [0, 1, 2, 3, 4, 5, 8, 10, 7, 4],
        "non_deterministic_count": 0,
        "eval_runs": 1,
        "stage_timing": {},
        "skipped_count": 0,
        "skipped_pairs_by_reason": {},
        "checkpoint_stage": None,
        "source_latency_p50": None,
        "source_latency_p95": None,
        "target_latency_p50": None,
        "target_latency_p95": None,
        "projected_source_cost_per_call": None,
        "projected_target_cost_per_call": None,
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
        "wins": 60,
        "losses": 29,
        "score_histogram": [2, 5, 8, 10, 4, 6, 12, 18, 14, 10],
        "non_deterministic_count": 0,
        "eval_runs": 1,
        "stage_timing": {},
        "skipped_count": 0,
        "skipped_pairs_by_reason": {},
        "checkpoint_stage": None,
        "source_latency_p50": None,
        "source_latency_p95": None,
        "target_latency_p50": None,
        "target_latency_p95": None,
        "projected_source_cost_per_call": None,
        "projected_target_cost_per_call": None,
    },
]

DUMMY_ALERTS: list[dict[str, Any]] = [
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

DUMMY_TEST_CASES: dict[int, dict[str, Any]] = {
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


def _migration_to_template_dict(record: MigrationRecord, session: Session) -> dict[str, Any]:
    """Convert MigrationRecord to the dict shape templates expect."""
    result: dict[str, Any] = {
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

    # Token tracking
    result["total_tokens"] = record.total_tokens
    result["token_breakdown"] = json.loads(record.token_breakdown_json or "{}")

    # F2: GEPA iteration telemetry presence flag
    result["has_optimization_trace"] = record.optimization_score_history_json not in (
        "[]",
        "",
        None,
    )

    # Checkpoint info
    result["checkpoint_stage"] = record.checkpoint_stage

    # Test case count
    count_stmt = (
        select(func.count())
        .select_from(TestCaseRecord)
        .where(TestCaseRecord.migration_id == record.id)
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

        ci = stats.get("confidence_interval", [0.0, 0.0])
        if not isinstance(ci, (list, tuple)) or len(ci) < 2:
            ci = [0.0, 0.0]
        p10 = stats.get("p10", 0.0)
        p90 = stats.get("p90", 0.0)
        per_type.append(
            {
                "type": type_name.replace("_", " ").title(),
                "wins": wins,
                "total": sample_count,
                "badge": badge,
                "desc": desc,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "p10": p10,
                "p90": p90,
            }
        )
    result["per_type"] = per_type

    # Regressions: worst non-winning validation test cases (up to 5), with metric deltas
    # Build a positional map: validation TC id → corresponding baseline scores
    _baseline_stmt = (
        select(TestCaseRecord)
        .where(
            TestCaseRecord.migration_id == record.id,
            TestCaseRecord.phase == "baseline",
        )
        .order_by(TestCaseRecord.id)  # type: ignore[arg-type]
    )
    _val_order_stmt = (
        select(TestCaseRecord)
        .where(
            TestCaseRecord.migration_id == record.id,
            TestCaseRecord.phase == "validation",
        )
        .order_by(TestCaseRecord.id)  # type: ignore[arg-type]
    )
    _baseline_tcs = list(session.exec(_baseline_stmt).all())
    _val_tcs_ordered = list(session.exec(_val_order_stmt).all())
    # Map: validation tc.id → baseline scores dict (positional match by insertion order).
    # Guard: if counts differ (e.g. due to checkpoint resume), skip positional matching entirely
    # so that delta badges are absent rather than silently wrong.
    _baseline_scores_by_val_id: dict[int, dict[str, float]] = {}
    if len(_baseline_tcs) == len(_val_tcs_ordered):
        for _idx, _val_tc in enumerate(_val_tcs_ordered):
            _b_raw = _baseline_tcs[_idx].scores_json
            _b_scores = json.loads(_b_raw) if _b_raw else {}
            if _val_tc.id is not None:
                _baseline_scores_by_val_id[_val_tc.id] = {
                    k: v for k, v in _b_scores.items() if isinstance(v, (int, float))
                }

    tc_stmt = (
        select(TestCaseRecord)
        .where(
            TestCaseRecord.migration_id == record.id,
            TestCaseRecord.phase == "validation",
            TestCaseRecord.is_win == False,  # noqa: E712
        )
        .order_by(TestCaseRecord.composite_score.asc())  # type: ignore[attr-defined]
        .limit(5)
    )
    regressions = []
    for tc in session.exec(tc_stmt).all():
        reg: dict[str, Any] = {
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
        val_scores = json.loads(tc.scores_json) if tc.scores_json else {}
        baseline_scores = _baseline_scores_by_val_id.get(tc.id or 0, {})
        # Compute per-metric deltas; only include those with abs(delta) > 0.05
        metric_deltas: dict[str, float] = {}
        for m, v_score in val_scores.items():
            if not isinstance(v_score, (int, float)):
                continue
            if m in baseline_scores:
                delta = float(v_score) - float(baseline_scores[m])
                if abs(delta) > 0.05:
                    metric_deltas[m] = round(delta, 3)
        reg["metric_deltas"] = metric_deltas
        regressions.append(reg)
    result["regressions"] = regressions

    # Chart data: win/loss counts and score histogram (10 bins)
    all_tc_stmt = select(TestCaseRecord).where(TestCaseRecord.migration_id == record.id)
    all_tcs = list(session.exec(all_tc_stmt).all())
    chart_wins = sum(1 for tc in all_tcs if tc.is_win)
    chart_losses = len(all_tcs) - chart_wins
    score_histogram = [0] * 10
    for tc in all_tcs:
        bin_idx = min(int(tc.composite_score * 10), 9)
        score_histogram[bin_idx] += 1
    result["wins"] = chart_wins
    result["losses"] = chart_losses
    result["score_histogram"] = score_histogram

    # Phase A observability: stage timing and eval reliability
    _raw_config = json.loads(record.config_json) if record.config_json else {}
    _stage_timing_raw = _raw_config.pop("_stage_timing", {})
    result["stage_timing"] = (
        {k: float(v) for k, v in _stage_timing_raw.items()}
        if isinstance(_stage_timing_raw, dict)
        else {}
    )
    result["non_deterministic_count"] = int(_raw_config.pop("_non_deterministic_count", 0) or 0)
    result["eval_runs"] = int(_raw_config.pop("_eval_runs", 1) or 1)

    # F6: Skipped pairs grouped by failure reason (for "Needs Attention" panel)
    skipped_stmt = select(TestCaseRecord).where(
        TestCaseRecord.migration_id == record.id,
        TestCaseRecord.failure_reason != None,  # noqa: E711
    )
    skipped_tcs = list(session.exec(skipped_stmt).all())
    skipped_by_reason: dict[str, int] = {}
    for _stc in skipped_tcs:
        reason = _stc.failure_reason or "unknown"
        skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
    result["skipped_pairs_by_reason"] = skipped_by_reason
    result["skipped_count"] = len(skipped_tcs)

    return result


def _test_case_to_diff_dict(tc: TestCaseRecord, migration: MigrationRecord) -> dict[str, Any]:
    """Convert TestCaseRecord to the diff dict shape the template expects."""
    _raw_scores = json.loads(tc.scores_json) if tc.scores_json else {}
    scores = {k: round(v, 2) for k, v in _raw_scores.items() if isinstance(v, (int, float))}
    return {
        "tc_id": tc.id,
        "is_win": tc.is_win,
        "composite_score": round(tc.composite_score, 2),
        "output_type": tc.output_type.replace("_", " ").title(),
        "scores": scores,
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
        "expected": (tc.response_text or "Content not stored (run with --store-prompt-content)"),
        "actual": (tc.new_response_text or "Content not stored (run with --store-prompt-content)"),
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
        failure_reason=tc.failure_reason,
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
        failure_reason=tc.failure_reason,
        diff=diff,
    )


def _migration_to_detail(record: MigrationRecord, session: Session) -> MigrationDetail:
    """Convert a MigrationRecord to a MigrationDetail response with nested data."""
    # Parse JSON columns
    config = json.loads(record.config_json)
    config = {k: v for k, v in config.items() if k not in _SENSITIVE_CONFIG_KEYS}
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

    # Extract Phase A observability fields stored in config_json
    stage_timing: dict[str, float] = {}
    _stage_timing_raw = config.pop("_stage_timing", None)
    if isinstance(_stage_timing_raw, dict):
        stage_timing = {k: float(v) for k, v in _stage_timing_raw.items()}
    non_deterministic_count: int = int(config.pop("_non_deterministic_count", 0) or 0)
    eval_runs: int = int(config.pop("_eval_runs", 1) or 1)

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
        stage_timing=stage_timing,
        non_deterministic_count=non_deterministic_count,
        eval_runs=eval_runs,
        test_cases=test_case_summaries,
    )


# ---------------------------------------------------------------------------
# JSON API endpoints
# ---------------------------------------------------------------------------


@router.get("/api/v1/migrations", response_model=PaginatedResponse[MigrationSummary])
async def list_migrations(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> PaginatedResponse[MigrationSummary]:
    """List migrations with pagination."""
    from rosettastone.server.rbac import _is_multi_user

    owner_filter = None
    if _is_multi_user() and not is_admin_user(request):
        owner_filter = get_current_user_id(request)

    base = select(MigrationRecord)
    if owner_filter is not None:
        base = base.where(MigrationRecord.owner_id == owner_filter)

    count_stmt = select(func.count()).select_from(base.subquery())
    total = session.exec(count_stmt).one()

    stmt = (
        base.order_by(MigrationRecord.created_at.desc())  # type: ignore[attr-defined]
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
    request: Request,
    session: Session = Depends(get_session),
) -> MigrationDetail:
    """Get migration detail by ID."""
    record = _get_migration_or_404(migration_id, session)
    check_resource_owner(record.owner_id, request)
    return _migration_to_detail(record, session)


@router.get(
    "/api/v1/migrations/{migration_id}/optimizer-history",
    response_model=list[GEPAIterationOut],
)
async def get_optimizer_history(
    migration_id: int,
    session: Session = Depends(get_session),
) -> list[GEPAIterationOut]:
    """Get GEPA optimizer iteration history for a migration, sorted by iteration asc."""
    _get_migration_or_404(migration_id, session)
    records = session.exec(
        select(GEPAIterationRecord)
        .where(GEPAIterationRecord.migration_id == migration_id)
        .order_by(GEPAIterationRecord.iteration)  # type: ignore[arg-type]
    ).all()
    return [GEPAIterationOut(**r.model_dump()) for r in records]


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
    is_limited, retry_after = check_rate_limit(request, "migration_submit")
    if is_limited:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Too many migration submissions.",
            headers={"Retry-After": str(retry_after)},
        )
    body = await request.json()
    source_model = body.get("source_model")
    target_model = body.get("target_model")
    data_path = body.get("data_path")
    cluster_prompts = body.get("cluster_prompts", False)
    improvement_objectives = body.get("improvement_objectives")
    max_cost_usd = body.get("max_cost_usd")

    if not source_model or not target_model:
        raise HTTPException(status_code=422, detail="source_model and target_model are required")

    # Validate max_cost_usd
    if max_cost_usd is not None and max_cost_usd < 0:
        raise HTTPException(status_code=422, detail="max_cost_usd must be non-negative")

    # Validate data_path is within a safe directory (prevent path traversal)
    if data_path:
        safe_base = Path(os.path.expanduser("~/.rosettastone"))
        try:
            resolved = Path(str(data_path)).resolve()
            resolved.relative_to(safe_base)  # raises ValueError if not under safe_base
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"data_path must be within the rosettastone data directory ({safe_base})",
            )

    # Budget check (multi-user mode only)
    from rosettastone.server.api.costs import check_budget
    from rosettastone.server.rbac import _is_multi_user

    if _is_multi_user():
        user_id = get_current_user_id(request)
        if user_id is not None:
            # Use max_cost_usd from config as estimated cost, or 0.0 if not set
            estimated = max_cost_usd or 0.0
            check_budget(user_id, estimated, session)

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
    if max_cost_usd is not None:
        config["max_cost_usd"] = max_cost_usd

    record = MigrationRecord(
        source_model=source_model,
        target_model=target_model,
        status="pending",
        created_at=datetime.now(UTC),
        config_json=json.dumps(config),
        max_cost_usd=max_cost_usd,
        owner_id=get_current_user_id(request),
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
    failure_reason: str | None = Query(None),
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
    if failure_reason:
        conditions.append(TestCaseRecord.failure_reason == failure_reason)

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


@router.get("/api/v1/migrations/{migration_id}/regressions")
async def get_migration_regressions(
    migration_id: int,
    session: Session = Depends(get_session),
    _user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    """Return per-prompt regression analysis for a migration.

    Computes regressions by zipping baseline and validation TestCaseRecords
    (ordered by insertion ID, which matches the original result list indices).
    """
    from rosettastone.decision.recommendation import DEFAULT_THRESHOLDS

    record = _get_migration_or_404(migration_id, session)

    stored_win_thresholds: dict[str, float] = {}
    if record.config_json:
        try:
            stored_config = json.loads(record.config_json)
            stored_win_thresholds = stored_config.get("win_thresholds", {})
        except (json.JSONDecodeError, TypeError):
            pass
    effective_thresholds = {**DEFAULT_THRESHOLDS, **stored_win_thresholds}

    # Query baseline and validation rows in insertion order
    baseline_stmt = (
        select(TestCaseRecord)
        .where(TestCaseRecord.migration_id == migration_id)
        .where(TestCaseRecord.phase == "baseline")
        .order_by(TestCaseRecord.id)  # type: ignore[arg-type]
    )
    validation_stmt = (
        select(TestCaseRecord)
        .where(TestCaseRecord.migration_id == migration_id)
        .where(TestCaseRecord.phase == "validation")
        .order_by(TestCaseRecord.id)  # type: ignore[arg-type]
    )
    baseline_tcs = list(session.exec(baseline_stmt).all())
    validation_tcs = list(session.exec(validation_stmt).all())

    prompt_regressions: list[dict[str, Any]] = []
    for idx, (base_tc, val_tc) in enumerate(zip(baseline_tcs, validation_tcs)):
        b_score = base_tc.composite_score
        v_score = val_tc.composite_score
        delta = v_score - b_score
        out_type = val_tc.output_type or base_tc.output_type or "unknown"
        threshold = effective_thresholds.get(out_type, 0.80)

        if delta >= 0.05:
            status = "improved"
        elif delta >= -0.05:
            status = "stable"
        elif v_score >= threshold:
            status = "regressed"
        else:
            status = "at_risk"

        base_scores = json.loads(base_tc.scores_json) if base_tc.scores_json else {}
        val_scores = json.loads(val_tc.scores_json) if val_tc.scores_json else {}
        metric_deltas: dict[str, float] = {
            m: val_scores[m] - base_scores[m] for m in base_scores if m in val_scores
        }

        prompt_regressions.append(
            {
                "prompt_index": idx,
                "output_type": out_type,
                "baseline_score": b_score,
                "optimized_score": v_score,
                "delta": delta,
                "baseline_is_win": base_tc.is_win,
                "optimized_is_win": val_tc.is_win,
                "status": status,
                "metric_deltas": metric_deltas,
            }
        )

    # Sort: at_risk first (delta ascending), then regressed, then stable, then improved
    _status_order = {"at_risk": 0, "regressed": 1, "stable": 2, "improved": 3}
    prompt_regressions.sort(key=lambda r: (_status_order.get(r["status"], 99), r["delta"]))

    regression_count = sum(1 for r in prompt_regressions if r["status"] == "regressed")
    at_risk_count = sum(1 for r in prompt_regressions if r["status"] == "at_risk")

    return {
        "migration_id": migration_id,
        "prompt_regressions": prompt_regressions,
        "regression_count": regression_count,
        "at_risk_count": at_risk_count,
        "total_analyzed": len(prompt_regressions),
    }


@router.get("/api/v1/migrations/{migration_id}/optimization-trace")
async def get_optimization_trace(
    migration_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Return the GEPA score trajectory for a migration."""
    record = _get_migration_or_404(migration_id, session)
    check_resource_owner(record.owner_id, request)
    history = json.loads(record.optimization_score_history_json or "[]")
    return {
        "migration_id": migration_id,
        "iterations": history,
        "total_iterations": len(history),
        "final_prompt_length": len(record.optimized_prompt or ""),
    }


@router.get("/ui/migrations/{migration_id}/optimization-trace-fragment")
async def optimization_trace_fragment(
    migration_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """HTMX fragment: GEPA score trajectory chart for the migration detail page."""
    record = _get_migration_or_404(migration_id, session)
    history = json.loads(record.optimization_score_history_json or "[]")
    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "fragments/optimization_trace.html",
        {
            "request": request,
            "iteration_nums": [h["iteration_num"] for h in history],
            "scores": [h["mean_score"] for h in history],
            "total_iterations": len(history),
            "final_prompt_length": len(record.optimized_prompt or ""),
        },
    )


def _build_diagnostics(record: MigrationRecord, session: Session) -> MigrationDiagnostics:
    """Build comprehensive migration diagnostics. NEVER accesses prompt/response text columns."""
    from rosettastone.decision.recommendation import DEFAULT_THRESHOLDS
    from rosettastone.server.schemas import (
        BorderCase,
        MetricWinRate,
        RegressionSummary,
        SafetyDiagnostic,
        TypeDiagnostic,
    )

    stored_config = {}
    if record.config_json:
        try:
            stored_config = json.loads(record.config_json)
        except (json.JSONDecodeError, TypeError):
            pass
    stored_win_thresholds: dict[str, float] = stored_config.get("win_thresholds", {})
    effective_thresholds = {**DEFAULT_THRESHOLDS, **stored_win_thresholds}

    # 1. Per-type breakdown from stored per_type_scores_json
    per_type_raw: dict[str, Any] = {}
    if record.per_type_scores_json:
        try:
            per_type_raw = json.loads(record.per_type_scores_json)
        except (json.JSONDecodeError, TypeError):
            pass
    per_type_diagnostics: list[TypeDiagnostic] = []
    for type_name, stats in per_type_raw.items():
        ci = stats.get("confidence_interval", [0.0, 1.0])
        ci_lower = ci[0] if len(ci) > 0 else 0.0
        ci_upper = ci[1] if len(ci) > 1 else 1.0
        threshold = effective_thresholds.get(type_name, 0.80)
        win_rate = stats.get("win_rate", 0.0)
        per_type_diagnostics.append(
            TypeDiagnostic(
                output_type=type_name,
                win_rate=win_rate,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                p10=stats.get("p10", 0.0),
                p50=stats.get("p50", 0.0),
                p90=stats.get("p90", 0.0),
                sample_count=stats.get("sample_count", 0),
                threshold=threshold,
                # CI lower bound mirrors the recommendation engine's pass criterion
                # (uses confidence_interval[0] < threshold for CONDITIONAL).
                passes=ci_lower >= threshold,
            )
        )

    # 2. Metric win rates from validation TestCaseRecords
    val_tcs = list(
        session.exec(
            select(TestCaseRecord)
            .where(TestCaseRecord.migration_id == record.id)
            .where(TestCaseRecord.phase == "validation")
            .order_by(TestCaseRecord.id)  # type: ignore[arg-type]
        ).all()
    )
    metric_accumulator: dict[str, list[float]] = {}
    for tc in val_tcs:
        try:
            scores = json.loads(tc.scores_json) if tc.scores_json else {}
        except (json.JSONDecodeError, TypeError):
            scores = {}
        for metric_name, val in scores.items():
            if isinstance(val, (int, float)):
                metric_accumulator.setdefault(metric_name, []).append(float(val))
    metric_win_rates: list[MetricWinRate] = []
    for metric_name, values in sorted(metric_accumulator.items()):
        mean_val = sum(values) / len(values) if values else 0.0
        above = sum(1 for v in values if v >= _DIAGNOSTIC_METRIC_THRESHOLD)
        metric_win_rates.append(
            MetricWinRate(
                metric_name=metric_name,
                mean_value=round(mean_val, 4),
                above_threshold_count=above,
                total_count=len(values),
            )
        )

    # 3. Border cases: validation TCs within +/-5% of their type threshold
    border_cases: list[BorderCase] = []
    for tc in val_tcs:
        threshold = effective_thresholds.get(tc.output_type or "unknown", 0.80)
        delta = tc.composite_score - threshold
        if abs(delta) <= 0.05:
            border_cases.append(
                BorderCase(
                    test_case_id=tc.id or 0,
                    output_type=tc.output_type or "unknown",
                    composite_score=tc.composite_score,
                    threshold=threshold,
                    delta_to_threshold=delta,
                )
            )
    border_cases.sort(key=lambda b: b.delta_to_threshold)
    border_cases = border_cases[:20]

    # 4. Regression summary: pair baseline and validation TCs by insertion order
    baseline_tcs = list(
        session.exec(
            select(TestCaseRecord)
            .where(TestCaseRecord.migration_id == record.id)
            .where(TestCaseRecord.phase == "baseline")
            .order_by(TestCaseRecord.id)  # type: ignore[arg-type]
        ).all()
    )
    improved_count = stable_count = regressed_count = at_risk_count = 0
    worst_regressed: list[dict[str, Any]] = []
    for base_tc, val_tc in zip(baseline_tcs, val_tcs):
        b_score = base_tc.composite_score
        v_score = val_tc.composite_score
        delta = v_score - b_score
        out_type = val_tc.output_type or base_tc.output_type or "unknown"
        threshold = effective_thresholds.get(out_type, 0.80)
        if delta >= 0.05:
            improved_count += 1
            status = "improved"
        elif delta >= -0.05:
            stable_count += 1
            status = "stable"
        elif v_score >= threshold:
            regressed_count += 1
            status = "regressed"
        else:
            at_risk_count += 1
            status = "at_risk"
        if status in ("regressed", "at_risk"):
            try:
                base_scores = json.loads(base_tc.scores_json) if base_tc.scores_json else {}
            except (json.JSONDecodeError, TypeError):
                base_scores = {}
            try:
                val_scores = json.loads(val_tc.scores_json) if val_tc.scores_json else {}
            except (json.JSONDecodeError, TypeError):
                val_scores = {}
            metric_deltas: dict[str, float] = {
                m: round(val_scores[m] - base_scores[m], 4)
                for m in base_scores
                if m in val_scores
                and isinstance(base_scores[m], (int, float))
                and isinstance(val_scores[m], (int, float))
            }
            worst_regressed.append(
                {
                    "test_case_id": val_tc.id,
                    "output_type": out_type,
                    "delta": round(delta, 4),
                    "status": status,
                    "metric_deltas": metric_deltas,
                }
            )
    worst_regressed.sort(key=lambda r: r["delta"])
    worst_regressed = worst_regressed[:10]
    regression_summary = RegressionSummary(
        improved_count=improved_count,
        stable_count=stable_count,
        regressed_count=regressed_count,
        at_risk_count=at_risk_count,
        worst_regressed=worst_regressed,
    )

    # 5. Safety: query WarningRecords with warning_type="safety"
    safety_warnings = list(
        session.exec(
            select(WarningRecord)
            .where(WarningRecord.migration_id == record.id)
            .where(WarningRecord.warning_type == "safety")
        ).all()
    )
    high_count = sum(1 for w in safety_warnings if w.severity == "HIGH")
    medium_count = sum(1 for w in safety_warnings if w.severity == "MEDIUM")
    low_count = sum(1 for w in safety_warnings if w.severity == "LOW")
    high_severity_indices = [
        w.id for w in safety_warnings if w.severity == "HIGH" and w.id is not None
    ]
    safety = SafetyDiagnostic(
        high_count=high_count,
        medium_count=medium_count,
        low_count=low_count,
        high_severity_indices=high_severity_indices,
    )

    return MigrationDiagnostics(
        migration_id=record.id,  # type: ignore[arg-type]
        recommendation=record.recommendation,
        per_type=per_type_diagnostics,
        metric_win_rates=metric_win_rates,
        border_cases=border_cases,
        regression_summary=regression_summary,
        safety=safety,
    )


@router.get("/api/v1/migrations/{migration_id}/diagnostics", response_model=MigrationDiagnostics)
async def get_migration_diagnostics(
    migration_id: int,
    request: Request,
    session: Session = Depends(get_session),
    _user_id: str = Depends(get_current_user_id),
) -> MigrationDiagnostics:
    """Return comprehensive migration diagnostics — scores, counts, regressions, safety."""
    record = _get_migration_or_404(migration_id, session)
    check_resource_owner(record.owner_id, request)
    if record.status not in ("complete", "dry_run_complete"):
        raise HTTPException(
            status_code=422,
            detail="Diagnostics only available for completed migrations",
        )
    return _build_diagnostics(record, session)


@router.get("/ui/migrations/{migration_id}/diagnostics-fragment")
async def diagnostics_fragment(
    migration_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """HTMX fragment: diagnostics panel for the migration detail page."""
    record = _get_migration_or_404(migration_id, session)
    if record.status not in ("complete", "dry_run_complete"):
        return HTMLResponse("<p class='text-sm text-on-surface-variant p-4'>Not available yet.</p>")
    diag = _build_diagnostics(record, session)
    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "fragments/diagnostics_panel.html",
        {"request": request, "diag": diag},
    )


@router.get(
    "/api/v1/migrations/{migration_id}/stream",
    response_class=None,  # type: ignore[arg-type]  # streaming response
)
async def stream_migration_progress(
    migration_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> Any:
    """SSE endpoint for real-time migration progress.

    Connects the client to the progress hub. On connect, sends current
    DB state as a catch-up event. Sends keepalive comments every 30s.
    Closes when migration reaches a terminal state (complete/failed/cancelled).
    """
    from rosettastone.server.progress import register_client, unregister_client

    record = _get_migration_or_404(migration_id, session)

    async def event_generator() -> Any:
        # Send catch-up: current state from DB
        catchup: dict[str, Any] = {
            "type": "progress",
            "migration_id": migration_id,
            "status": record.status,
            "current_stage": record.current_stage or "",
            "stage_progress": record.stage_progress or 0.0,
            "overall_progress": record.overall_progress or 0.0,
        }
        # Include cost and warning count from the persisted record when available
        if record.cost_usd is not None and record.cost_usd > 0.0:
            catchup["total_cost_usd"] = round(record.cost_usd, 4)
        try:
            persisted_warnings = json.loads(record.warnings_json or "[]")
            if persisted_warnings:
                catchup["warning_count"] = len(persisted_warnings)
        except (json.JSONDecodeError, TypeError):
            pass
        yield f"data: {json.dumps(catchup)}\n\n"

        # If already terminal, close immediately
        terminal = {"complete", "failed", "cancelled", "blocked", "dry_run_complete"}
        if record.status in terminal:
            return

        q = register_client(migration_id)
        keepalive_elapsed = 0.0
        poll_interval = 1.0  # Check disconnect / accumulate keepalive every 1s
        keepalive_every = 30.0
        try:
            while True:
                if await request.is_disconnected():
                    return
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=poll_interval)
                    keepalive_elapsed = 0.0
                    if payload is None:  # Sentinel: close stream
                        return
                    yield f"data: {payload}\n\n"

                    # Check if terminal state reached
                    try:
                        event_data = json.loads(payload)
                        if event_data.get("status") in terminal:
                            return
                    except (json.JSONDecodeError, AttributeError):
                        pass
                except TimeoutError:
                    keepalive_elapsed += poll_interval
                    if keepalive_elapsed >= keepalive_every:
                        yield ": keepalive\n\n"  # SSE comment for keepalive
                        keepalive_elapsed = 0.0
        finally:
            unregister_client(migration_id, q)

    from starlette.responses import StreamingResponse

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Resume endpoint
# ---------------------------------------------------------------------------


@router.post("/api/v1/migrations/{migration_id}/resume", response_model=MigrationSummary)
def resume_migration(
    migration_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> MigrationSummary:
    """Re-enqueue a failed migration from its last checkpoint."""
    record = _get_migration_or_404(migration_id, session)
    check_resource_owner(record.owner_id, request)

    if record.status != "failed":
        raise HTTPException(status_code=409, detail="Migration is not in failed state")
    if not record.checkpoint_stage:
        raise HTTPException(status_code=409, detail="No checkpoint available for this migration")

    record.status = "pending"
    session.add(record)
    session.commit()
    session.refresh(record)

    # Re-enqueue with checkpoint info injected into the payload
    config = json.loads(record.config_json or "{}")
    request.app.state.task_worker.enqueue(
        "migration",
        migration_id,
        {
            **config,
            "_resume_from": record.checkpoint_stage,
            "_checkpoint_data": record.checkpoint_data_json,
        },
    )
    return _migration_to_summary(record)


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
    stmt = select(RegisteredModel).order_by(RegisteredModel.added_at.desc())  # type: ignore[attr-defined]
    records = list(session.exec(stmt).all())

    if not records and empty != "false":
        # No models registered — show empty state
        return request.app.state.templates.TemplateResponse(  # type: ignore[no-any-return]
            request,
            "models_empty.html",
            {"active_nav": "models"},
        )

    models = [_model_to_template_dict(r) for r in records] if records else DUMMY_MODELS
    return request.app.state.templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "models.html",
        {"models": models, "alerts": DUMMY_ALERTS, "active_nav": "models"},
    )


@router.get("/ui/migrations", response_class=HTMLResponse)
async def migrations_page(
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """Migrations list page."""
    stmt = select(MigrationRecord).order_by(MigrationRecord.created_at.desc()).limit(50)  # type: ignore[attr-defined]
    records = list(session.exec(stmt).all())

    if records:
        migrations = [_migration_to_template_dict(r, session) for r in records]
    else:
        migrations = DUMMY_MIGRATIONS  # fallback when DB is empty

    return request.app.state.templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "migrations.html",
        {"migrations": migrations, "active_nav": "migrations"},
    )


MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB


@router.get("/ui/migrations/new", response_class=HTMLResponse)
async def new_migration_form(
    request: Request,
    source: str | None = Query(None),
) -> HTMLResponse:
    """Render the new migration form."""
    return request.app.state.templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "migration_new.html",
        {"active_nav": "migrations", "source_model": source or "", "error": None},
    )


@router.post("/ui/migrations/new", response_model=None)
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
) -> HTMLResponse | RedirectResponse:
    """Handle form submission: validate, save file, create record, submit to executor."""
    templates = request.app.state.templates

    # Validate file upload
    if data_file is None or data_file.filename == "":
        return templates.TemplateResponse(  # type: ignore[no-any-return]
            request,
            "migration_new.html",
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
        return templates.TemplateResponse(  # type: ignore[no-any-return]
            request,
            "migration_new.html",
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
        return templates.TemplateResponse(  # type: ignore[no-any-return]
            request,
            "migration_new.html",
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
        return templates.TemplateResponse(  # type: ignore[no-any-return]
            request,
            "migration_new.html",
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
        return templates.TemplateResponse(  # type: ignore[no-any-return]
            request,
            "migration_new.html",
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
        return templates.TemplateResponse(  # type: ignore[no-any-return]
            request,
            "migration_new.html",
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
        config_json=json.dumps(
            {
                "source_model": source_model,
                "target_model": target_model,
            }
        ),
        owner_id=get_current_user_id(request),
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

    # Enqueue in DB-backed task queue
    request.app.state.task_worker.enqueue("migration", migration_id, config_dict)

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
        _result: dict[str, Any] | None = next(
            (m for m in DUMMY_MIGRATIONS if m["id"] == migration_id), None
        )
        if _result is None:
            raise HTTPException(status_code=404, detail="Migration not found")
        migration = _result

    return request.app.state.templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "migration_detail.html",
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
        _result: dict[str, Any] | None = next(
            (m for m in DUMMY_MIGRATIONS if m["id"] == migration_id), None
        )
        if _result is None:
            raise HTTPException(status_code=404, detail="Migration not found")
        migration = _result

    report_date = datetime.now(UTC).strftime("%B %-d, %Y")
    return request.app.state.templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "executive_report.html",
        {"migration": migration, "report_date": report_date, "active_nav": "migrations"},
    )


@router.get("/ui/costs", response_class=HTMLResponse)
async def costs_page(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    """Costs overview page."""
    from rosettastone.server.api.costs import _compute_costs
    from rosettastone.server.models import DatasetGenerationRun

    costs = _compute_costs(session)
    if costs is None:
        costs = DUMMY_COSTS  # fallback when no data

    stmt = select(DatasetGenerationRun).order_by(
        DatasetGenerationRun.created_at.desc()  # type: ignore[attr-defined]
    )
    dataset_runs = list(session.exec(stmt).all())

    return request.app.state.templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "costs.html",
        {"costs": costs, "active_nav": "costs", "dataset_runs": dataset_runs},
    )


@router.get("/ui/alerts", response_class=HTMLResponse)
async def alerts_page(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    """Alerts hub page."""
    from rosettastone.server.api.alerts import _alert_to_template_dict, _generate_alerts
    from rosettastone.server.models import Alert

    # Auto-generate alerts from current data (idempotent)
    _generate_alerts(session)

    # Fetch all alerts newest first
    stmt = select(Alert).order_by(Alert.created_at.desc()).limit(50)  # type: ignore[attr-defined]
    records = list(session.exec(stmt).all())

    if records:
        alerts = [_alert_to_template_dict(a) for a in records]
    else:
        alerts = []  # fallback when DB is empty

    return request.app.state.templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "alerts.html",
        {"alerts": alerts, "active_nav": "alerts"},
    )


@router.get("/ui/fragments/migration-list", response_class=HTMLResponse)
async def migration_list_fragment(
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """HTMX partial for migration list — returns card rows for the migrations table."""
    from rosettastone.server.rbac import _is_multi_user

    owner_filter = None
    if _is_multi_user() and not is_admin_user(request):
        owner_filter = get_current_user_id(request)

    base = select(MigrationRecord)
    if owner_filter is not None:
        base = base.where(MigrationRecord.owner_id == owner_filter)

    stmt = base.order_by(MigrationRecord.created_at.desc()).limit(50)  # type: ignore[attr-defined]
    records = list(session.exec(stmt).all())

    if records:
        migrations = [_migration_to_template_dict(r, session) for r in records]
    else:
        migrations = DUMMY_MIGRATIONS

    rows: list[str] = []
    for m in migrations:
        rec_val = m.get("recommendation", "")
        if rec_val == "Safe to ship":
            border_color = "#8B9D83"
            badge_class = "bg-[#8B9D83] text-[#131313]"
            icon_color = "text-[#8B9D83]"
            icon_name = "verified"
            card_hover = "card-glow-sage"
        elif rec_val == "Do not ship":
            border_color = "#D85650"
            badge_class = "bg-[#D85650] text-white"
            icon_color = "text-[#D85650]"
            icon_name = "cancel"
            card_hover = "hover:shadow-xl hover:shadow-[#D85650]/5"
        else:
            border_color = "#D4A574"
            badge_class = "bg-[#D4A574] text-[#131313]"
            icon_color = "text-[#D4A574]"
            icon_name = "warning"
            card_hover = "hover:shadow-xl hover:shadow-[#D4A574]/5"

        mid = m.get("id", "")
        source = m.get("source", "")
        target = m.get("target", "")
        confidence = m.get("confidence", 0)
        test_cases = m.get("test_cases", 0)
        cost = m.get("cost", "$0.00")
        time_ago = m.get("time_ago", "")

        rows.append(
            f'<a href="/ui/migrations/{mid}" class="block">'
            '<div class="group relative bg-[#3D3D3D] rounded-xl overflow-hidden cursor-pointer'
            f' transition-all duration-300 {card_hover} active:scale-[0.99]">'
            f'<div class="absolute left-0 top-0 bottom-0 w-1" style="background:{border_color}">'
            "</div>"
            '<div class="p-6 pl-8">'
            '<div class="flex justify-between items-start mb-6">'
            '<div class="flex items-center gap-3">'
            '<h3 class="font-headline font-medium text-[18px] text-on-surface">'
            f"{source}"
            ' <span class="text-on-surface-variant/40 mx-2">&rarr;</span>'
            f" {target}</h3>"
            "</div>"
            f'<span class="px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider'
            f' {badge_class}">{rec_val}</span>'
            "</div>"
            '<div class="flex items-center gap-4 text-[#BDBDBD] font-label text-[13px]">'
            '<div class="flex items-center gap-1.5">'
            f'<span class="material-symbols-outlined text-[16px] {icon_color}">{icon_name}</span>'
            f" {confidence}% confidence"
            "</div>"
            '<span class="text-white/10">&bull;</span>'
            f"<span>{test_cases} test cases</span>"
            '<span class="text-white/10">&bull;</span>'
            '<div class="flex items-center gap-1">'
            '<span class="material-symbols-outlined text-[16px]">payments</span>'
            f" {cost}"
            "</div>"
            '<span class="text-white/10">&bull;</span>'
            '<div class="flex items-center gap-1">'
            '<span class="material-symbols-outlined text-[16px]">schedule</span>'
            f" {time_ago}"
            "</div>"
            "</div>"
            "</div>"
            "</div>"
            "</a>"
        )

    if not rows:
        html = (
            '<div class="flex flex-col items-center justify-center py-24 text-center'
            ' text-on-surface-variant/50">'
            '<span class="material-symbols-outlined text-5xl mb-4 opacity-30">swap_horiz</span>'
            '<p class="text-lg font-medium">No migrations yet</p>'
            '<p class="text-sm mt-2 opacity-70">Run your first migration to see results here.</p>'
            "</div>"
        )
    else:
        html = "\n".join(rows)

    return HTMLResponse(html)


@router.get("/ui/fragments/eval-grid/{migration_id}", response_class=HTMLResponse)
async def eval_grid_fragment(
    migration_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """HTMX partial for evaluation grid — returns test case rows for a migration."""
    stmt = (
        select(TestCaseRecord)
        .where(TestCaseRecord.migration_id == migration_id)
        .order_by(TestCaseRecord.id.asc())  # type: ignore[union-attr]
        .limit(100)
    )
    test_cases = list(session.exec(stmt).all())

    if not test_cases:
        html = (
            '<tr><td colspan="4" class="px-4 py-8 text-center text-on-surface-variant/50'
            ' text-sm">No test cases found for this migration.</td></tr>'
        )
        return HTMLResponse(html)

    rows: list[str] = []
    for tc in test_cases:
        score_pct = round(tc.composite_score * 100)
        if tc.is_win:
            badge = (
                '<span class="px-2 py-0.5 rounded text-[10px] font-bold'
                ' bg-success/20 text-success">WIN</span>'
            )
        else:
            badge = (
                '<span class="px-2 py-0.5 rounded text-[10px] font-bold'
                ' bg-error/20 text-error">LOSS</span>'
            )
        output_type_display = tc.output_type.replace("_", " ").title()
        phase_display = tc.phase.replace("_", " ").title()
        rows.append(
            "<tr>"
            f'<td class="px-4 py-3 text-sm font-mono text-on-surface">{score_pct}%</td>'
            f'<td class="px-4 py-3 text-sm text-on-surface-variant">{phase_display}</td>'
            f'<td class="px-4 py-3 text-sm text-on-surface-variant">{output_type_display}</td>'
            f'<td class="px-4 py-3">{badge}</td>'
            "</tr>"
        )

    html = "\n".join(rows)
    return HTMLResponse(html)


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
        raw_scores: dict[str, float] = json.loads(db_tc.scores_json) if db_tc.scores_json else {}
        # Build dynamic scores dict: all stored per-metric scores rounded to 2 dp,
        # with composite appended last if not already present.
        dynamic_scores: dict[str, float] = {
            k: round(v, 2) for k, v in raw_scores.items() if isinstance(v, (int, float))
        }
        if "composite" not in dynamic_scores and "composite_score" not in dynamic_scores:
            dynamic_scores["composite"] = round(db_tc.composite_score, 2)
        tc = {
            "tc_id": db_tc.id,
            "is_win": db_tc.is_win,
            "composite_score": round(db_tc.composite_score, 2),
            "output_type": db_tc.output_type.replace("_", " ").title(),
            "phase": db_tc.phase,
            "scores": dynamic_scores,
            "prompt": db_tc.prompt_text,
            "source_response": db_tc.response_text,
            "target_response": db_tc.new_response_text,
            "evaluators_used": db_tc.evaluators_used,
            "fallback_triggered": db_tc.fallback_triggered,
            "failure_reason": db_tc.failure_reason,
        }
    else:
        tc = DUMMY_TEST_CASES.get(
            tc_id,
            DUMMY_TEST_CASES[42],  # default fallback
        )

    return request.app.state.templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "fragments/test_case_detail.html",
        {"tc": tc},
    )


@router.get("/ui/migrations/{migration_id}/test-cases-table", response_class=HTMLResponse)
async def test_cases_table_fragment(
    migration_id: int,
    request: Request,
    outcome: str = Query("all"),  # "win" | "loss" | "skipped" | "all"
    search: str = Query(""),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    failure_reason: str = Query(""),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """HTMX partial for filterable test case grid."""
    _get_migration_or_404(migration_id, session)

    conditions = [TestCaseRecord.migration_id == migration_id]
    if outcome == "win":
        conditions.append(TestCaseRecord.is_win == True)  # noqa: E712
    elif outcome == "loss":
        conditions.append(TestCaseRecord.is_win == False)  # noqa: E712
    elif outcome == "skipped":
        conditions.append(TestCaseRecord.failure_reason != None)  # noqa: E711
    if failure_reason:
        conditions.append(TestCaseRecord.failure_reason == failure_reason)
    if search:
        conditions.append(TestCaseRecord.prompt_text.contains(search))  # type: ignore[union-attr]

    count_stmt = select(func.count()).select_from(TestCaseRecord).where(*conditions)
    total = session.exec(count_stmt).one()

    offset = (page - 1) * page_size
    stmt = (
        select(TestCaseRecord)
        .where(*conditions)
        .order_by(TestCaseRecord.composite_score.asc())  # type: ignore[attr-defined]
        .offset(offset)
        .limit(page_size)
    )
    test_cases = list(session.exec(stmt).all())
    total_pages = max(1, (total + page_size - 1) // page_size)

    tc_list = []
    for tc in test_cases:
        tc_list.append(
            {
                "id": tc.id,
                "prompt_preview": (tc.prompt_text or "")[:80]
                + ("…" if tc.prompt_text and len(tc.prompt_text) > 80 else ""),
                "output_type": tc.output_type.replace("_", " ").title(),
                "composite_score": round(tc.composite_score, 2),
                "is_win": tc.is_win,
                "failure_reason": tc.failure_reason,
            }
        )

    return request.app.state.templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "fragments/test_cases_table.html",
        {
            "migration_id": migration_id,
            "test_cases": tc_list,
            "outcome": outcome,
            "search": search,
            "failure_reason": failure_reason,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
        },
    )


# ---------------------------------------------------------------------------
# P2.2: Inline executive summary fragment
# ---------------------------------------------------------------------------


@router.get("/ui/migrations/{migration_id}/executive-summary", response_class=HTMLResponse)
async def executive_summary_fragment(
    migration_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """HTMX fragment: compact executive summary card."""
    record = session.get(MigrationRecord, migration_id)
    if record:
        migration = _migration_to_template_dict(record, session)
    else:
        _result: dict[str, Any] | None = next(
            (m for m in DUMMY_MIGRATIONS if m["id"] == migration_id), None
        )
        if _result is None:
            raise HTTPException(status_code=404, detail="Migration not found")
        migration = _result

    # Truncate reasoning to first paragraph, max 500 chars
    reasoning = migration.get("reasoning", "") or ""
    narrative = reasoning.split("\n\n")[0]  # first paragraph
    if len(narrative) > 500:
        narrative = narrative[:497] + "\u2026"

    return request.app.state.templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "fragments/executive_summary.html",
        {"migration": migration, "narrative": narrative},
    )


# ---------------------------------------------------------------------------
# UI resume route
# ---------------------------------------------------------------------------


@router.post("/ui/migrations/{migration_id}/resume", response_class=None)  # type: ignore[arg-type]
async def ui_resume_migration(
    migration_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> RedirectResponse:
    """Handle Resume button POST from migration detail page."""
    record = _get_migration_or_404(migration_id, session)
    check_resource_owner(record.owner_id, request)

    if record.status != "failed":
        raise HTTPException(status_code=409, detail="Migration is not in failed state")
    if not record.checkpoint_stage:
        raise HTTPException(status_code=409, detail="No checkpoint available for this migration")

    record.status = "pending"
    session.add(record)
    session.commit()
    session.refresh(record)

    config = json.loads(record.config_json or "{}")
    request.app.state.task_worker.enqueue(
        "migration",
        migration_id,
        {
            **config,
            "_resume_from": record.checkpoint_stage,
            "_checkpoint_data": record.checkpoint_data_json,
        },
    )
    return RedirectResponse(url=f"/ui/migrations/{migration_id}", status_code=303)


# ---------------------------------------------------------------------------
# GEPA optimizer history UI fragment
# ---------------------------------------------------------------------------


@router.get("/ui/migrations/{migration_id}/optimizer-history", response_class=HTMLResponse)
async def optimizer_history_fragment(
    migration_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """HTMX fragment: optimizer iteration history for a completed migration."""
    _get_migration_or_404(migration_id, session)
    records = session.exec(
        select(GEPAIterationRecord)
        .where(GEPAIterationRecord.migration_id == migration_id)
        .order_by(GEPAIterationRecord.iteration)  # type: ignore[arg-type]
    ).all()
    return request.app.state.templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "fragments/optimizer_history.html",
        {"records": records, "migration_id": migration_id},
    )
