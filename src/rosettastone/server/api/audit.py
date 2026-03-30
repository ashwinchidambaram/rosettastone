"""Audit log API router and log_audit utility."""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from sqlmodel import Session, func, select

from rosettastone.server.database import get_session
from rosettastone.server.models import AuditLog
from rosettastone.server.schemas import AuditLogEntry, PaginatedResponse

router = APIRouter()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def log_audit(
    session: Session,
    resource_type: str,
    resource_id: int | None,
    action: str,
    user_id: int | None = None,
    details: dict | None = None,
) -> None:
    """Record an audit log entry. Call within an existing session transaction."""
    entry = AuditLog(
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        user_id=user_id,
        details_json=json.dumps(details) if details else "{}",
    )
    session.add(entry)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@router.get("/api/v1/audit-log", response_model=PaginatedResponse[AuditLogEntry])
def list_audit_log(
    resource_type: str | None = Query(None),
    action: str | None = Query(None),
    user_id: int | None = Query(None),
    start_date: datetime | None = Query(None),
    end_date: datetime | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    session: Session = Depends(get_session),
) -> PaginatedResponse[AuditLogEntry]:
    """List audit log entries with optional filters, ordered by created_at DESC."""
    conditions = []

    if resource_type is not None:
        conditions.append(AuditLog.resource_type == resource_type)
    if action is not None:
        conditions.append(AuditLog.action == action)
    if user_id is not None:
        conditions.append(AuditLog.user_id == user_id)
    if start_date is not None:
        conditions.append(AuditLog.created_at >= start_date)  # type: ignore[operator]
    if end_date is not None:
        conditions.append(AuditLog.created_at <= end_date)  # type: ignore[operator]

    count_stmt = select(func.count()).select_from(AuditLog)
    if conditions:
        count_stmt = count_stmt.where(*conditions)
    total = session.exec(count_stmt).one()

    offset = (page - 1) * per_page
    stmt = (
        select(AuditLog)
        .where(*conditions)
        .order_by(AuditLog.created_at.desc())  # type: ignore[union-attr]
        .offset(offset)
        .limit(per_page)
    )
    records = list(session.exec(stmt).all())

    items = [
        AuditLogEntry(
            id=r.id,  # type: ignore[arg-type]
            resource_type=r.resource_type,
            resource_id=r.resource_id,
            action=r.action,
            user_id=r.user_id,
            details=json.loads(r.details_json),
            created_at=r.created_at,
        )
        for r in records
    ]

    return PaginatedResponse(items=items, total=total, page=page, per_page=per_page)


# ---------------------------------------------------------------------------
# UI routes
# ---------------------------------------------------------------------------


@router.get("/ui/audit-log", response_class=HTMLResponse)
async def audit_log_page(
    request: Request,
    resource_type: str | None = Query(None),
    action: str | None = Query(None),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """Render the audit log UI page with optional filters."""
    conditions = []
    if resource_type:
        conditions.append(AuditLog.resource_type == resource_type)
    if action:
        conditions.append(AuditLog.action == action)

    stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(200)  # type: ignore[union-attr]
    if conditions:
        stmt = stmt.where(*conditions)
    entries = list(session.exec(stmt).all())

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "audit_log.html",
        {"active_nav": "audit", "entries": entries},
    )
