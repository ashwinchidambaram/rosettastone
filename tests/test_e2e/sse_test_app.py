"""Test app factory wrapping the real FastAPI app with SSE injection endpoints.

These test-only endpoints let Playwright tests:
  1. Create migration records in "running" state (without triggering LLM work)
  2. Inject SSE events directly into the progress hub
  3. Query how many SSE clients are connected (for test synchronization)
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Request
from sqlmodel import Session

from rosettastone.server import progress as _progress_module
from rosettastone.server.app import create_app
from rosettastone.server.database import get_session
from rosettastone.server.models import MigrationRecord
from rosettastone.server.progress import emit_progress


class _CreateRunningMigrationBody:
    """Pydantic-compatible body model parsed via FastAPI."""

    def __init__(
        self,
        source_model: str = "test/mock-source",
        target_model: str = "test/mock-target",
        current_stage: str = "preflight",
    ) -> None:
        self.source_model = source_model
        self.target_model = target_model
        self.current_stage = current_stage


# Use a proper Pydantic model so FastAPI can parse the JSON body automatically.
try:
    from pydantic import BaseModel as _BaseModel

    class CreateRunningMigrationRequest(_BaseModel):
        source_model: str = "test/mock-source"
        target_model: str = "test/mock-target"
        current_stage: str = "preflight"

except ImportError:
    # Should never happen — pydantic is a core dependency — but be defensive.
    CreateRunningMigrationRequest = _CreateRunningMigrationBody  # type: ignore[misc,assignment]


def create_test_app() -> FastAPI:
    """Return the real app augmented with test-only /test/* endpoints."""
    app = create_app()

    # ------------------------------------------------------------------
    # Endpoint 1: POST /test/create-running-migration
    # Creates a MigrationRecord with status="running" (no task queue).
    # ------------------------------------------------------------------

    @app.post("/test/create-running-migration")
    async def create_running_migration(
        body: CreateRunningMigrationRequest,
        session: Session = Depends(get_session),
    ) -> dict[str, Any]:
        record = MigrationRecord(
            source_model=body.source_model,
            target_model=body.target_model,
            status="running",
            current_stage=body.current_stage,
            stage_progress=0.0,
            overall_progress=0.0,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return {"id": record.id}

    # ------------------------------------------------------------------
    # Endpoint 2: POST /test/emit-event/{migration_id}
    # Injects an SSE event payload into the progress hub.
    # ------------------------------------------------------------------

    @app.post("/test/emit-event/{migration_id}")
    async def emit_event(migration_id: int, request: Request) -> dict[str, Any]:
        payload: dict[str, Any] = await request.json()
        emit_progress(migration_id, payload)
        return {"ok": True}

    # ------------------------------------------------------------------
    # Endpoint 3: GET /test/sse-clients/{migration_id}
    # Returns the number of SSE clients currently watching a migration.
    # ------------------------------------------------------------------

    @app.get("/test/sse-clients/{migration_id}")
    async def sse_client_count(migration_id: int) -> dict[str, Any]:
        queue_set = _progress_module._queues.get(migration_id, set())
        return {"count": len(queue_set)}

    return app
