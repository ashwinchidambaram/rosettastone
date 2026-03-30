"""A/B testing API router."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from sqlmodel import Session, func, select

from rosettastone.decision.ab_stats import compute_ab_significance
from rosettastone.server.api.audit import log_audit
from rosettastone.server.database import get_session
from rosettastone.server.models import (
    ABTest,
    ABTestResult,
    MigrationRecord,
    MigrationVersion,
)
from rosettastone.server.rbac import require_role
from rosettastone.server.schemas import (
    ABTestCreate,
    ABTestDetail,
    ABTestMetrics,
    ABTestSummary,
    PaginatedResponse,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_metrics(ab_test_id: int, session: Session) -> ABTestMetrics:
    """Compute aggregated metrics from ABTestResult rows."""
    results = list(
        session.exec(select(ABTestResult).where(ABTestResult.ab_test_id == ab_test_id)).all()  # type: ignore[arg-type]
    )

    if not results:
        return ABTestMetrics(
            total_results=0,
            wins_a=0,
            wins_b=0,
            ties=0,
            win_rate_a=0.0,
            win_rate_b=0.0,
            mean_score_a=0.0,
            mean_score_b=0.0,
        )

    wins_a = sum(1 for r in results if r.winner == "a")
    wins_b = sum(1 for r in results if r.winner == "b")
    ties = sum(1 for r in results if r.winner == "tie")
    total_a = sum(1 for r in results if r.assigned_version == "a")
    total_b = sum(1 for r in results if r.assigned_version == "b")

    scores_a = [r.score_a for r in results if r.score_a is not None]
    scores_b = [r.score_b for r in results if r.score_b is not None]

    return ABTestMetrics(
        total_results=len(results),
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        win_rate_a=wins_a / max(total_a, 1),
        win_rate_b=wins_b / max(total_b, 1),
        mean_score_a=sum(scores_a) / max(len(scores_a), 1),
        mean_score_b=sum(scores_b) / max(len(scores_b), 1),
    )


def _ab_test_to_summary(ab_test: ABTest) -> ABTestSummary:
    """Convert an ABTest model to ABTestSummary schema."""
    return ABTestSummary(
        id=ab_test.id,  # type: ignore[arg-type]
        migration_id=ab_test.migration_id,
        name=ab_test.name,
        status=ab_test.status,
        traffic_split=ab_test.traffic_split,
        winner=ab_test.winner,
        created_at=ab_test.created_at,
    )


def _ab_test_to_detail(ab_test: ABTest, metrics: ABTestMetrics | None = None) -> ABTestDetail:
    """Convert an ABTest model to ABTestDetail schema."""
    return ABTestDetail(
        id=ab_test.id,  # type: ignore[arg-type]
        migration_id=ab_test.migration_id,
        version_a_id=ab_test.version_a_id,
        version_b_id=ab_test.version_b_id,
        name=ab_test.name,
        status=ab_test.status,
        traffic_split=ab_test.traffic_split,
        start_time=ab_test.start_time,
        end_time=ab_test.end_time,
        winner=ab_test.winner,
        created_at=ab_test.created_at,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/ab-tests",
    response_model=ABTestSummary,
    status_code=201,
    dependencies=[Depends(require_role("editor", "admin"))],
)
def create_ab_test(
    body: ABTestCreate,
    session: Session = Depends(get_session),
) -> ABTestSummary:
    """Create a new A/B test.

    Validates that the migration exists and both versions exist and belong to
    the migration. Returns the created ABTest as ABTestSummary with status 201.
    """
    migration = session.get(MigrationRecord, body.migration_id)
    if not migration:
        raise HTTPException(status_code=404, detail=f"Migration {body.migration_id} not found")

    version_a = session.get(MigrationVersion, body.version_a_id)
    if not version_a or version_a.migration_id != body.migration_id:
        raise HTTPException(
            status_code=404,
            detail=f"Version {body.version_a_id} not found for migration {body.migration_id}",
        )

    version_b = session.get(MigrationVersion, body.version_b_id)
    if not version_b or version_b.migration_id != body.migration_id:
        raise HTTPException(
            status_code=404,
            detail=f"Version {body.version_b_id} not found for migration {body.migration_id}",
        )

    ab_test = ABTest(
        migration_id=body.migration_id,
        version_a_id=body.version_a_id,
        version_b_id=body.version_b_id,
        name=body.name,
        traffic_split=body.traffic_split,
        status="draft",
    )
    session.add(ab_test)

    log_audit(
        session,
        "ab_test",
        None,
        "create",
        details={
            "migration_id": body.migration_id,
            "version_a_id": body.version_a_id,
            "version_b_id": body.version_b_id,
            "name": body.name,
        },
    )

    session.commit()
    session.refresh(ab_test)

    return _ab_test_to_summary(ab_test)


@router.get("/api/v1/ab-tests", response_model=PaginatedResponse[ABTestSummary])
def list_ab_tests(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> PaginatedResponse[ABTestSummary]:
    """List all A/B tests, paginated and ordered by created_at DESC."""
    count_stmt = select(func.count()).select_from(ABTest)
    total = session.exec(count_stmt).one()

    offset = (page - 1) * per_page
    stmt = (
        select(ABTest)
        .order_by(ABTest.created_at.desc())  # type: ignore[union-attr]
        .offset(offset)
        .limit(per_page)
    )
    records = list(session.exec(stmt).all())

    items = [_ab_test_to_summary(r) for r in records]
    return PaginatedResponse(items=items, total=total, page=page, per_page=per_page)


@router.get("/api/v1/ab-tests/{ab_test_id}", response_model=ABTestDetail)
def get_ab_test(
    ab_test_id: int,
    session: Session = Depends(get_session),
) -> ABTestDetail:
    """Return A/B test detail. Metrics are not included here -- use the /metrics endpoint."""
    ab_test = session.get(ABTest, ab_test_id)
    if not ab_test:
        raise HTTPException(status_code=404, detail=f"A/B test {ab_test_id} not found")

    return _ab_test_to_detail(ab_test)


@router.post(
    "/api/v1/ab-tests/{ab_test_id}/start",
    response_model=ABTestDetail,
    dependencies=[Depends(require_role("editor", "admin"))],
)
def start_ab_test(
    ab_test_id: int,
    session: Session = Depends(get_session),
) -> ABTestDetail:
    """Start an A/B test.

    Transitions status from "draft" to "running" and sets start_time.
    Returns 400 if the test is not in "draft" status.
    """
    ab_test = session.get(ABTest, ab_test_id)
    if not ab_test:
        raise HTTPException(status_code=404, detail=f"A/B test {ab_test_id} not found")

    if ab_test.status != "draft":
        raise HTTPException(
            status_code=400,
            detail=f"A/B test {ab_test_id} is not in 'draft' status (current: {ab_test.status})",
        )

    ab_test.status = "running"
    ab_test.start_time = datetime.now(UTC)
    session.add(ab_test)

    log_audit(
        session,
        "ab_test",
        ab_test_id,
        "start",
        details={"migration_id": ab_test.migration_id},
    )

    session.commit()
    session.refresh(ab_test)

    return _ab_test_to_detail(ab_test)


@router.get("/api/v1/ab-tests/{ab_test_id}/metrics", response_model=ABTestMetrics)
def get_ab_test_metrics(
    ab_test_id: int,
    session: Session = Depends(get_session),
) -> ABTestMetrics:
    """Get live aggregated metrics for an A/B test.

    Queries ABTestResult rows and computes win counts, win rates, and mean scores.
    If the test is concluded, also includes statistical significance values.
    Returns empty metrics (all zeros) if no results exist yet.
    """
    ab_test = session.get(ABTest, ab_test_id)
    if not ab_test:
        raise HTTPException(status_code=404, detail=f"A/B test {ab_test_id} not found")

    metrics = _build_metrics(ab_test_id, session)

    if ab_test.status == "concluded" and metrics.total_results > 0:
        results = list(
            session.exec(
                select(ABTestResult).where(ABTestResult.ab_test_id == ab_test_id)  # type: ignore[arg-type]
            ).all()
        )
        result_dicts = [
            {
                "assigned_version": r.assigned_version,
                "score_a": r.score_a,
                "score_b": r.score_b,
                "winner": r.winner,
            }
            for r in results
        ]
        sig = compute_ab_significance(result_dicts)
        metrics.chi2 = sig.chi2
        metrics.p_value = sig.p_value
        metrics.significant = sig.significant
        metrics.mean_diff = sig.mean_diff
        metrics.ci_lower = sig.ci_lower
        metrics.ci_upper = sig.ci_upper

    return metrics


@router.post(
    "/api/v1/ab-tests/{ab_test_id}/conclude",
    response_model=ABTestDetail,
    dependencies=[Depends(require_role("editor", "admin"))],
)
def conclude_ab_test(
    ab_test_id: int,
    session: Session = Depends(get_session),
) -> ABTestDetail:
    """Conclude an A/B test.

    Transitions status to "concluded", sets end_time, computes statistical
    significance, and sets winner to "a", "b", or "inconclusive".
    Returns 400 if the test is not in "running" status.
    """
    ab_test = session.get(ABTest, ab_test_id)
    if not ab_test:
        raise HTTPException(status_code=404, detail=f"A/B test {ab_test_id} not found")

    if ab_test.status != "running":
        raise HTTPException(
            status_code=400,
            detail=(
                f"A/B test {ab_test_id} is not in 'running' status (current: {ab_test.status})"
            ),
        )

    results = list(
        session.exec(
            select(ABTestResult).where(ABTestResult.ab_test_id == ab_test_id)  # type: ignore[arg-type]
        ).all()
    )
    result_dicts = [
        {
            "assigned_version": r.assigned_version,
            "score_a": r.score_a,
            "score_b": r.score_b,
            "winner": r.winner,
        }
        for r in results
    ]

    winner: str
    if result_dicts:
        sig = compute_ab_significance(result_dicts)
        if sig.significant:
            wins_a = sum(1 for r in result_dicts if r["winner"] == "a")
            wins_b = sum(1 for r in result_dicts if r["winner"] == "b")
            winner = "a" if wins_a >= wins_b else "b"
        else:
            winner = "inconclusive"
    else:
        winner = "inconclusive"

    ab_test.status = "concluded"
    ab_test.end_time = datetime.now(UTC)
    ab_test.winner = winner
    session.add(ab_test)

    log_audit(
        session,
        "ab_test",
        ab_test_id,
        "conclude",
        details={
            "migration_id": ab_test.migration_id,
            "winner": winner,
            "total_results": len(result_dicts),
        },
    )

    session.commit()
    session.refresh(ab_test)

    return _ab_test_to_detail(ab_test)


# ---------------------------------------------------------------------------
# UI routes
# ---------------------------------------------------------------------------


@router.get("/ui/ab-tests", response_class=HTMLResponse)
async def ab_tests_page(
    request: Request,
    session: Session = Depends(get_session),
):
    """List all A/B tests."""
    stmt = select(ABTest).order_by(ABTest.created_at.desc()).limit(100)  # type: ignore[arg-type]
    ab_tests = list(session.exec(stmt).all())
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "ab_tests.html",
        {"active_nav": "ab_tests", "ab_tests": ab_tests},
    )


@router.get("/ui/ab-tests/{ab_test_id}", response_class=HTMLResponse)
async def ab_test_detail_page(
    ab_test_id: int,
    request: Request,
    session: Session = Depends(get_session),
):
    """A/B test detail page."""
    ab_test = session.get(ABTest, ab_test_id)
    if not ab_test:
        raise HTTPException(status_code=404, detail="A/B test not found")
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "ab_test_detail.html",
        {"active_nav": "ab_tests", "ab_test": ab_test},
    )


@router.get("/ui/ab-tests/{ab_test_id}/metrics-fragment", response_class=HTMLResponse)
async def ab_test_metrics_fragment(
    ab_test_id: int,
    request: Request,
    session: Session = Depends(get_session),
):
    """HTMX fragment: live A/B test metrics."""
    metrics = _build_metrics(ab_test_id, session)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "fragments/ab_metrics.html",
        {"metrics": metrics},
    )
