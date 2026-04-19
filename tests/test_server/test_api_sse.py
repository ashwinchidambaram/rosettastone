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


class TestEventCallbackShapes:
    """Verify the event payloads emitted by the three SSE callback factories.

    These callbacks are the source of all SSE events. The frontend JS routes
    events by the ``type`` field, so its presence and value are critical.
    """

    def test_progress_writer_emits_correct_shape(self, engine):
        """_make_progress_writer emits events with type='progress' and required fields."""
        import asyncio

        import rosettastone.server.progress as progress_mod
        from rosettastone.server.api.tasks import _make_progress_writer

        progress_mod._queues.clear()

        with Session(engine) as session:
            migration = MigrationRecord(
                source_model="a", target_model="b", status="running",
            )
            session.add(migration)
            session.commit()
            session.refresh(migration)
            mid = migration.id

        async def _run():
            q = progress_mod.register_client(mid)

            writer = _make_progress_writer(mid, engine)
            # Emit from within the running loop so call_soon_threadsafe works
            writer("optimize", 0.5, 0.75)
            await asyncio.sleep(0.05)

            assert not q.empty()
            payload = json.loads(q.get_nowait())
            assert payload["type"] == "progress"
            assert payload["current_stage"] == "optimize"
            assert payload["stage_progress"] == 0.5
            assert payload["overall_progress"] == 0.75
            assert payload["migration_id"] == mid

            progress_mod.unregister_client(mid, q)

        asyncio.run(_run())
        progress_mod._queues.clear()

    def test_progress_writer_includes_eta_when_past_threshold(self, engine):
        """_make_progress_writer includes eta_seconds when overall_pct > 0.05."""
        import asyncio
        import time

        import rosettastone.server.progress as progress_mod
        from rosettastone.server.api.tasks import _make_progress_writer

        progress_mod._queues.clear()

        with Session(engine) as session:
            migration = MigrationRecord(
                source_model="a", target_model="b", status="running",
            )
            session.add(migration)
            session.commit()
            session.refresh(migration)
            mid = migration.id

        async def _run():
            q = progress_mod.register_client(mid)

            writer = _make_progress_writer(mid, engine)
            # Sleep briefly so real elapsed time is non-zero
            time.sleep(0.1)
            writer("optimize", 0.5, 0.50)  # 50% done → should compute ETA

            await asyncio.sleep(0.05)
            payload = json.loads(q.get_nowait())
            # ETA field must be present when overall_pct > 0.05
            assert "eta_seconds" in payload
            assert isinstance(payload["eta_seconds"], (int, float))

            progress_mod.unregister_client(mid, q)

        asyncio.run(_run())
        progress_mod._queues.clear()

    def test_gepa_callback_emits_correct_shape(self, engine):
        """_make_gepa_callback emits events with type='gepa_iteration'."""
        import asyncio

        import rosettastone.server.progress as progress_mod
        from rosettastone.server.api.tasks import _make_gepa_callback

        progress_mod._queues.clear()
        mid = 99999  # Doesn't need a real DB record for the SSE emit path

        async def _run():
            q = progress_mod.register_client(mid)

            cb = _make_gepa_callback(mid, engine)
            cb(3, 25, 0.87)
            await asyncio.sleep(0.05)

            assert not q.empty()
            payload = json.loads(q.get_nowait())
            assert payload["type"] == "gepa_iteration"
            assert payload["migration_id"] == mid
            assert payload["iteration"] == 3
            assert payload["total_iterations"] == 25
            assert payload["running_mean_score"] == 0.87

            progress_mod.unregister_client(mid, q)

        asyncio.run(_run())
        progress_mod._queues.clear()

    def test_eval_pair_callback_emits_correct_shape(self, engine):
        """_make_eval_pair_callback emits events with type='eval_pair'."""
        import asyncio

        import rosettastone.server.progress as progress_mod
        from rosettastone.server.api.tasks import _make_eval_pair_callback

        progress_mod._queues.clear()
        mid = 99998

        async def _run():
            q = progress_mod.register_client(mid)

            cb = _make_eval_pair_callback(mid, engine)
            cb(7, 80, 0.92, "json")
            await asyncio.sleep(0.05)

            assert not q.empty()
            payload = json.loads(q.get_nowait())
            assert payload["type"] == "eval_pair"
            assert payload["migration_id"] == mid
            assert payload["pair_index"] == 7
            assert payload["total_pairs"] == 80
            assert payload["composite_score"] == 0.92
            assert payload["output_type"] == "json"

            progress_mod.unregister_client(mid, q)

        asyncio.run(_run())
        progress_mod._queues.clear()

    def test_terminal_emits_use_type_progress_not_status(self):
        """run_migration_background terminal emits use type='progress' so JS reload handler fires.

        The JS migration_detail.html only checks for terminal status inside the
        ``if (eventType === 'progress')`` branch, so terminal events must use
        ``type: 'progress'``.  This test verifies that neither the completion
        nor the error emit path uses ``"type": "status"`` in any emit_progress
        call.
        """
        import inspect

        from rosettastone.server.api import tasks

        source = inspect.getsource(tasks.run_migration_background)

        # Confirm the old wire format is gone
        assert '"type": "status"' not in source, (
            'Found "type": "status" in run_migration_background — '
            "terminal events must use type='progress' so the JS reload handler fires"
        )

        # Confirm the correct wire format is present for both terminal paths
        assert source.count('"type": "progress"') >= 2, (
            "Expected at least 2 occurrences of '\"type\": \"progress\"' in "
            "run_migration_background (completion + error paths)"
        )
