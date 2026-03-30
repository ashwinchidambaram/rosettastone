"""Annotation queue API router."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlmodel import Session, select

from rosettastone.server.database import get_session
from rosettastone.server.models import Annotation, MigrationRecord
from rosettastone.server.rbac import require_role
from rosettastone.server.schemas import AnnotationCreate, AnnotationSummary

router = APIRouter()

VALID_ANNOTATION_TYPES = {"regression", "improvement", "edge_case"}


def _multi_user_enabled() -> bool:
    return os.environ.get("ROSETTASTONE_MULTI_USER", "").lower() in ("1", "true", "yes")


def _annotation_to_summary(annotation: Annotation) -> AnnotationSummary:
    return AnnotationSummary(
        id=annotation.id,  # type: ignore[arg-type]
        migration_id=annotation.migration_id,
        test_case_id=annotation.test_case_id,
        annotator_id=annotation.annotator_id,
        annotation_type=annotation.annotation_type,
        text=annotation.text,
        created_at=annotation.created_at,
    )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@router.get("/ui/annotations", response_class=HTMLResponse)
async def annotations_page(
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """Render the annotation queue UI page."""
    multi_user = _multi_user_enabled()
    if multi_user:
        stmt = (
            select(Annotation)
            .order_by(Annotation.created_at.desc())  # type: ignore[union-attr]
            .limit(200)
        )
        annotations = list(session.exec(stmt).all())
    else:
        annotations = []
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "annotations.html",
        {"active_nav": "annotations", "annotations": annotations, "multi_user": multi_user},
    )


@router.get(
    "/api/v1/migrations/{migration_id}/annotations",
    response_model=list[AnnotationSummary],
)
def list_annotations(
    migration_id: int,
    session: Session = Depends(get_session),
) -> list[AnnotationSummary]:
    """List annotations for a migration, ordered by created_at DESC.

    Returns 404 when multi-user mode is not enabled or migration is not found.
    """
    if not _multi_user_enabled():
        raise HTTPException(status_code=404, detail="Not found")

    migration = session.get(MigrationRecord, migration_id)
    if not migration:
        raise HTTPException(status_code=404, detail=f"Migration {migration_id} not found")

    stmt = (
        select(Annotation)
        .where(Annotation.migration_id == migration_id)  # type: ignore[arg-type]
        .order_by(Annotation.created_at.desc())  # type: ignore[union-attr]
    )
    records = list(session.exec(stmt).all())
    return [_annotation_to_summary(r) for r in records]


@router.post(
    "/api/v1/migrations/{migration_id}/annotations",
    response_model=AnnotationSummary,
    status_code=201,
    dependencies=[Depends(require_role("editor", "approver", "admin"))],
)
def create_annotation(
    migration_id: int,
    body: AnnotationCreate,
    request: Request,
    session: Session = Depends(get_session),
) -> AnnotationSummary:
    """Create an annotation for a migration.

    Requires editor, approver, or admin role. Sets annotator_id from the
    authenticated user when available. Returns 422 for invalid annotation_type,
    404 when multi-user mode is disabled or migration is not found.
    """
    if not _multi_user_enabled():
        raise HTTPException(status_code=404, detail="Not found")

    migration = session.get(MigrationRecord, migration_id)
    if not migration:
        raise HTTPException(status_code=404, detail=f"Migration {migration_id} not found")

    if body.annotation_type not in VALID_ANNOTATION_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid annotation_type '{body.annotation_type}'. "
                f"Must be one of: {sorted(VALID_ANNOTATION_TYPES)}"
            ),
        )

    user = getattr(request.state, "user", None)
    annotator_id: int | None = None
    if user is not None:
        if isinstance(user, dict):
            annotator_id = user.get("user_id")
        else:
            annotator_id = getattr(user, "user_id", None)

    annotation = Annotation(
        migration_id=migration_id,
        test_case_id=body.test_case_id,
        annotator_id=annotator_id,
        annotation_type=body.annotation_type,
        text=body.text,
    )
    session.add(annotation)
    session.commit()
    session.refresh(annotation)

    return _annotation_to_summary(annotation)


@router.get(
    "/api/v1/annotations/queue",
    response_model=list[AnnotationSummary],
)
def get_annotation_queue(
    request: Request,
    session: Session = Depends(get_session),
) -> list[AnnotationSummary]:
    """Get the annotation queue for the current user.

    Admins receive all annotations (last 100). Other authenticated users receive
    only their own annotations. Returns 404 when multi-user mode is not enabled.
    """
    if not _multi_user_enabled():
        raise HTTPException(status_code=404, detail="Not found")

    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    if isinstance(user, dict):
        user_role = user.get("role", "viewer")
        user_id = user.get("user_id")
    else:
        user_role = getattr(user, "role", "viewer")
        user_id = getattr(user, "user_id", None)

    if user_role == "admin":
        stmt = (
            select(Annotation)
            .order_by(Annotation.created_at.desc())  # type: ignore[union-attr]
            .limit(100)
        )
    else:
        stmt = (
            select(Annotation)
            .where(Annotation.annotator_id == user_id)  # type: ignore[arg-type]
            .order_by(Annotation.created_at.desc())  # type: ignore[union-attr]
        )

    records = list(session.exec(stmt).all())
    return [_annotation_to_summary(r) for r in records]
