"""Tests for SSE streaming progress endpoint."""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
sqlmodel = pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient  # noqa: E402
from sqlmodel import Session  # noqa: E402

from rosettastone.server.models import MigrationRecord  # noqa: E402


class TestStreamEndpoint:
    def test_stream_endpoint_returns_200_with_sse_content_type(
        self, client: TestClient, engine, sample_migration: MigrationRecord
    ):
        """GET /api/v1/migrations/{id}/stream returns 200 with text/event-stream content type."""
        # Use stream=True to avoid consuming the full streaming body
        with client.stream("GET", f"/api/v1/migrations/{sample_migration.id}/stream") as response:
            assert response.status_code == 200
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type

    def test_stream_endpoint_returns_404_for_unknown_migration(self, client: TestClient):
        """GET /api/v1/migrations/99999/stream returns 404."""
        response = client.get("/api/v1/migrations/99999/stream")
        assert response.status_code == 404

    def test_stream_sends_catchup_on_connect(
        self, client: TestClient, engine, sample_migration: MigrationRecord
    ):
        """Connecting to the stream for a terminal migration yields a catchup event."""
        # sample_migration has status="complete" (terminal), so the stream
        # will send the catch-up event and then close immediately.
        with client.stream("GET", f"/api/v1/migrations/{sample_migration.id}/stream") as response:
            assert response.status_code == 200
            # Read the first chunk of data
            raw = b""
            for chunk in response.iter_bytes(chunk_size=4096):
                raw += chunk
                break  # We only need the first event

        text = raw.decode("utf-8")
        # SSE format: "data: {...}\n\n"
        assert text.startswith("data: ")
        # Extract the JSON payload from the first line
        first_line = text.split("\n")[0]
        assert first_line.startswith("data: ")
        payload = json.loads(first_line[len("data: ") :])

        assert payload["type"] == "progress"
        assert payload["migration_id"] == sample_migration.id
        assert payload["status"] == "complete"

    def test_stream_includes_nginx_no_buffer_header(
        self, client: TestClient, engine, sample_migration: MigrationRecord
    ):
        """Stream response includes X-Accel-Buffering: no header."""
        with client.stream("GET", f"/api/v1/migrations/{sample_migration.id}/stream") as response:
            assert response.headers.get("x-accel-buffering") == "no"

    def test_stream_includes_no_cache_header(
        self, client: TestClient, engine, sample_migration: MigrationRecord
    ):
        """Stream response includes Cache-Control: no-cache header."""
        with client.stream("GET", f"/api/v1/migrations/{sample_migration.id}/stream") as response:
            assert response.headers.get("cache-control") == "no-cache"

    def test_stream_catchup_includes_progress_fields(self, client: TestClient, engine):
        """Catch-up event includes stage progress fields when present in DB."""
        # Use a terminal status so the stream closes immediately after the catch-up event.
        with Session(engine) as session:
            migration = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                status="failed",
                current_stage="baseline_eval",
                stage_progress=0.5,
                overall_progress=0.25,
            )
            session.add(migration)
            session.commit()
            session.refresh(migration)
            migration_id = migration.id

        with client.stream("GET", f"/api/v1/migrations/{migration_id}/stream") as response:
            assert response.status_code == 200
            raw = b""
            for chunk in response.iter_bytes(chunk_size=4096):
                raw += chunk

        text = raw.decode("utf-8")
        first_line = text.split("\n")[0]
        assert first_line.startswith("data: ")
        payload = json.loads(first_line[len("data: ") :])

        assert payload["type"] == "progress"
        assert payload["migration_id"] == migration_id
        assert payload["status"] == "failed"
        assert payload["current_stage"] == "baseline_eval"
        assert payload["stage_progress"] == 0.5
        assert payload["overall_progress"] == 0.25
