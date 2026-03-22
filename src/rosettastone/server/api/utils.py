"""Shared utilities for API route handlers."""

from __future__ import annotations

from fastapi import HTTPException
from sqlmodel import Session

from rosettastone.server.models import MigrationRecord


def _get_migration_or_404(migration_id: int, session: Session) -> MigrationRecord:
    """Fetch a migration by ID or raise 404."""
    record = session.get(MigrationRecord, migration_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Migration not found")
    return record
