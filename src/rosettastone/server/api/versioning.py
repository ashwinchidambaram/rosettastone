"""Versioning API router — snapshot and rollback MigrationRecord state."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from sqlmodel import Session, func, select

from rosettastone.server.api.audit import log_audit
from rosettastone.server.database import get_session
from rosettastone.server.models import MigrationRecord, MigrationVersion
from rosettastone.server.rbac import require_role
from rosettastone.server.schemas import (
    MigrationVersionDetail,
    MigrationVersionSummary,
    PaginatedResponse,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def create_version(migration_id: int, session: Session) -> MigrationVersion:
    """Snapshot the current MigrationRecord state as a new version."""
    migration = session.get(MigrationRecord, migration_id)
    if not migration:
        raise ValueError(f"Migration {migration_id} not found")

    # Get next version number
    max_ver_stmt = select(func.max(MigrationVersion.version_number)).where(
        MigrationVersion.migration_id == migration_id  # type: ignore[arg-type]
    )
    max_ver = session.exec(max_ver_stmt).one()
    next_ver = (max_ver or 0) + 1

    # Build snapshot from MigrationRecord fields
    snapshot = {
        "status": migration.status,
        "source_model": migration.source_model,
        "target_model": migration.target_model,
        "optimized_prompt": migration.optimized_prompt,
        "confidence_score": migration.confidence_score,
        "baseline_score": migration.baseline_score,
        "improvement": migration.improvement,
        "cost_usd": migration.cost_usd,
        "duration_seconds": migration.duration_seconds,
        "recommendation": migration.recommendation,
        "recommendation_reasoning": migration.recommendation_reasoning,
        "per_type_scores_json": migration.per_type_scores_json,
        "warnings_json": migration.warnings_json,
        "safety_warnings_json": migration.safety_warnings_json,
        "config_json": migration.config_json,
    }

    version = MigrationVersion(
        migration_id=migration_id,
        version_number=next_ver,
        snapshot_json=json.dumps(snapshot),
        optimized_prompt=migration.optimized_prompt,
        confidence_score=migration.confidence_score,
        created_by="system",
    )
    session.add(version)
    return version


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@router.get("/ui/migrations/{migration_id}/versions", response_class=HTMLResponse)
async def migration_versions_fragment(
    migration_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """HTMX fragment: list versions for a migration."""
    stmt = (
        select(MigrationVersion)
        .where(MigrationVersion.migration_id == migration_id)
        .order_by(MigrationVersion.version_number.desc())  # type: ignore[union-attr]
    )
    versions = list(session.exec(stmt).all())
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "fragments/version_timeline.html",
        {"versions": versions, "migration_id": migration_id},
    )


@router.get(
    "/api/v1/migrations/{migration_id}/versions",
    response_model=PaginatedResponse[MigrationVersionSummary],
)
def list_versions(
    migration_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    session: Session = Depends(get_session),
) -> PaginatedResponse[MigrationVersionSummary]:
    """Paginated list of versions for a migration, ordered by version_number DESC."""
    migration = session.get(MigrationRecord, migration_id)
    if not migration:
        raise HTTPException(status_code=404, detail=f"Migration {migration_id} not found")

    count_stmt = (
        select(func.count())
        .select_from(MigrationVersion)
        .where(
            MigrationVersion.migration_id == migration_id  # type: ignore[arg-type]
        )
    )
    total = session.exec(count_stmt).one()

    offset = (page - 1) * per_page
    stmt = (
        select(MigrationVersion)
        .where(MigrationVersion.migration_id == migration_id)  # type: ignore[arg-type]
        .order_by(MigrationVersion.version_number.desc())  # type: ignore[union-attr]
        .offset(offset)
        .limit(per_page)
    )
    records = list(session.exec(stmt).all())

    items = [
        MigrationVersionSummary(
            id=r.id,  # type: ignore[arg-type]
            migration_id=r.migration_id,
            version_number=r.version_number,
            confidence_score=r.confidence_score,
            created_at=r.created_at,
            created_by=r.created_by,
        )
        for r in records
    ]

    return PaginatedResponse(items=items, total=total, page=page, per_page=per_page)


@router.get(
    "/api/v1/migrations/{migration_id}/versions/{version_id}",
    response_model=MigrationVersionDetail,
)
def get_version(
    migration_id: int,
    version_id: int,
    session: Session = Depends(get_session),
) -> MigrationVersionDetail:
    """Return a single version with its snapshot parsed from JSON to dict."""
    version = session.get(MigrationVersion, version_id)
    if not version or version.migration_id != migration_id:
        raise HTTPException(
            status_code=404,
            detail=f"Version {version_id} not found for migration {migration_id}",
        )

    return MigrationVersionDetail(
        id=version.id,  # type: ignore[arg-type]
        migration_id=version.migration_id,
        version_number=version.version_number,
        snapshot=json.loads(version.snapshot_json),
        optimized_prompt=version.optimized_prompt,
        confidence_score=version.confidence_score,
        created_at=version.created_at,
        created_by=version.created_by,
    )


@router.post(
    "/api/v1/migrations/{migration_id}/versions/{version_id}/rollback",
    response_model=MigrationVersionDetail,
    dependencies=[Depends(require_role("editor", "admin"))],
)
def rollback_version(
    migration_id: int,
    version_id: int,
    session: Session = Depends(get_session),
) -> MigrationVersionDetail:
    """Restore a MigrationRecord to a previous version's snapshot state.

    Loads the target version's snapshot_json, restores MigrationRecord-level fields
    (not test cases), creates a new version capturing the restored state, records an
    audit entry, and returns the new version detail — all within a single transaction.
    """
    migration = session.get(MigrationRecord, migration_id)
    if not migration:
        raise HTTPException(status_code=404, detail=f"Migration {migration_id} not found")

    version = session.get(MigrationVersion, version_id)
    if not version or version.migration_id != migration_id:
        raise HTTPException(
            status_code=404,
            detail=f"Version {version_id} not found for migration {migration_id}",
        )

    snapshot = json.loads(version.snapshot_json)

    # Restore MigrationRecord fields from snapshot (NOT test cases)
    migration.optimized_prompt = snapshot.get("optimized_prompt")
    migration.confidence_score = snapshot.get("confidence_score")
    migration.baseline_score = snapshot.get("baseline_score")
    migration.improvement = snapshot.get("improvement")
    migration.cost_usd = snapshot.get("cost_usd", 0.0)
    migration.duration_seconds = snapshot.get("duration_seconds", 0.0)
    migration.recommendation = snapshot.get("recommendation")
    migration.recommendation_reasoning = snapshot.get("recommendation_reasoning")
    migration.per_type_scores_json = snapshot.get("per_type_scores_json", "{}")
    migration.warnings_json = snapshot.get("warnings_json", "[]")
    migration.safety_warnings_json = snapshot.get("safety_warnings_json", "[]")
    migration.config_json = snapshot.get("config_json", "{}")
    session.add(migration)

    # Create a new version capturing the restored state
    new_version = create_version(migration_id, session)

    # Record audit entry
    log_audit(
        session,
        "migration",
        migration_id,
        "rollback",
        details={"rolled_back_to_version": version_id},
    )

    session.commit()
    session.refresh(new_version)

    return MigrationVersionDetail(
        id=new_version.id,  # type: ignore[arg-type]
        migration_id=new_version.migration_id,
        version_number=new_version.version_number,
        snapshot=json.loads(new_version.snapshot_json),
        optimized_prompt=new_version.optimized_prompt,
        confidence_score=new_version.confidence_score,
        created_at=new_version.created_at,
        created_by=new_version.created_by,
    )
