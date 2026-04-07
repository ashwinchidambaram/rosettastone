"""Shared utilities for API route handlers."""

from __future__ import annotations

from fastapi import HTTPException, Request
from sqlmodel import Session

from rosettastone.server.models import MigrationRecord
from rosettastone.server.rbac import check_resource_owner


def _get_migration_or_404(migration_id: int, session: Session) -> MigrationRecord:
    """Fetch a migration by ID or raise 404."""
    record = session.get(MigrationRecord, migration_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Migration not found")
    return record


def _get_migration_with_owner_check(
    migration_id: int, session: Session, request: Request
) -> MigrationRecord:
    """Fetch a migration by ID, verify ownership in multi-user mode, or raise 403/404."""
    record = _get_migration_or_404(migration_id, session)
    check_resource_owner(record.owner_id, request)
    return record
