"""Score distribution and diff comparison endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select

from rosettastone.server.api.migrations import _test_case_to_diff_dict
from rosettastone.server.database import get_session
from rosettastone.server.models import MigrationRecord, TestCaseRecord
from rosettastone.server.schemas import DiffData, ScoreDistribution, TypeScoreStats

router = APIRouter()

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Dummy data for UI diff fragment
# ---------------------------------------------------------------------------

DUMMY_DIFF = {
    "tc_id": 42,
    "is_win": False,
    "composite_score": 0.72,
    "output_type": "Classification",
    "scores": {"bertscore": 0.85, "embedding": 0.79, "composite": 0.72},
    "source_model": "gpt-4o",
    "target_model": "claude-sonnet-4",
    "expected": ('{\n  "priority": "urgent",\n  "category": "billing",\n  "confidence": 0.94\n}'),
    "actual": (
        '{\n  "priority": "high_priority",\n  "category": "billing",\n  "confidence": 0.91\n}'
    ),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_migration_or_404(migration_id: int, session: Session) -> MigrationRecord:
    """Fetch a migration by ID or raise 404."""
    record = session.get(MigrationRecord, migration_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Migration not found")
    return record


def _compute_distributions(migration_id: int, session: Session) -> list[ScoreDistribution]:
    """Compute score distributions per output type from TestCaseRecords."""
    stmt = select(TestCaseRecord).where(TestCaseRecord.migration_id == migration_id)
    test_cases = list(session.exec(stmt).all())

    # Group scores by output type
    scores_by_type: dict[str, list[float]] = {}
    for tc in test_cases:
        output_type = tc.output_type
        if output_type not in scores_by_type:
            scores_by_type[output_type] = []
        scores_by_type[output_type].append(tc.composite_score)

    distributions: list[ScoreDistribution] = []
    for output_type, scores in sorted(scores_by_type.items()):
        scores.sort()
        n = len(scores)
        if n == 0:
            continue

        mean_score = sum(scores) / n
        median_score = scores[n // 2] if n % 2 == 1 else (scores[n // 2 - 1] + scores[n // 2]) / 2

        def _percentile(sorted_vals: list[float], p: float) -> float:
            """Compute percentile from sorted values."""
            if len(sorted_vals) == 1:
                return sorted_vals[0]
            idx = p * (len(sorted_vals) - 1)
            lower = int(idx)
            upper = lower + 1
            if upper >= len(sorted_vals):
                return sorted_vals[-1]
            frac = idx - lower
            return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])

        win_count = sum(1 for s in scores if s >= 0.5)

        # Build histogram: 10 buckets from 0.0 to 1.0
        histogram = [0] * 10
        for s in scores:
            bucket = min(int(s * 10), 9)
            histogram[bucket] += 1

        stats = TypeScoreStats(
            win_rate=win_count / n,
            mean=mean_score,
            median=median_score,
            p10=_percentile(scores, 0.10),
            p50=_percentile(scores, 0.50),
            p90=_percentile(scores, 0.90),
            min_score=scores[0],
            max_score=scores[-1],
            sample_count=n,
            confidence_interval=(
                _percentile(scores, 0.025),
                _percentile(scores, 0.975),
            ),
        )

        distributions.append(
            ScoreDistribution(
                output_type=output_type,
                stats=stats,
                histogram=histogram,
            )
        )

    return distributions


def _build_diff_data(tc: TestCaseRecord) -> DiffData:
    """Build diff data from a test case record."""
    available = bool(tc.prompt_text or tc.response_text or tc.new_response_text)
    return DiffData(
        prompt=tc.prompt_text,
        source_response=tc.response_text,
        target_response=tc.new_response_text,
        available=available,
    )


# ---------------------------------------------------------------------------
# JSON API endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/migrations/{migration_id}/distributions",
    response_model=list[ScoreDistribution],
)
async def get_distributions(
    migration_id: int,
    session: Session = Depends(get_session),
) -> list[ScoreDistribution]:
    """Get score distributions per output type."""
    _get_migration_or_404(migration_id, session)
    return _compute_distributions(migration_id, session)


@router.get(
    "/api/v1/migrations/{migration_id}/test-cases/{tc_id}/diff",
    response_model=DiffData,
)
async def get_diff(
    migration_id: int,
    tc_id: int,
    session: Session = Depends(get_session),
) -> DiffData:
    """Get diff data for a test case."""
    _get_migration_or_404(migration_id, session)

    tc = session.get(TestCaseRecord, tc_id)
    if tc is None or tc.migration_id != migration_id:
        raise HTTPException(status_code=404, detail="Test case not found")

    return _build_diff_data(tc)


# ---------------------------------------------------------------------------
# HTMX / UI endpoints
# ---------------------------------------------------------------------------


@router.get("/ui/fragments/diff/{migration_id}/{tc_id}", response_class=HTMLResponse)
async def diff_fragment(
    migration_id: int,
    tc_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """HTMX partial for diff view."""
    migration = session.get(MigrationRecord, migration_id)
    tc = session.get(TestCaseRecord, tc_id) if migration else None

    if tc and tc.migration_id == migration_id:
        diff = _test_case_to_diff_dict(tc, migration)  # type: ignore[arg-type]
    else:
        diff = DUMMY_DIFF  # fallback

    return templates.TemplateResponse(
        "fragments/diff_slideout.html",
        {"request": request, "diff": diff},
    )


@router.get("/ui/fragments/charts/{migration_id}", response_class=HTMLResponse)
async def charts_fragment(migration_id: int, request: Request) -> HTMLResponse:
    """HTMX partial for charts."""
    return HTMLResponse("<div>Template pending integration</div>")
