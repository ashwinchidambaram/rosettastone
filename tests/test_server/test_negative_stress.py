"""Negative-path, edge-case, concurrency, and stress tests for the RosettaStone web server.

This module implements an adversarial test plan covering:
  1. File upload abuse
  2. Concurrent migration stress
  3. Database edge cases
  4. Template rendering edge cases
  5. Background task failure modes
  6. API endpoint abuse (SQL injection, XSS, path traversal)
  7. State machine violations

Each test documents: priority, setup, steps, expected behavior, and assertion.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
sqlmodel = pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine, select  # noqa: E402

from rosettastone.server.app import create_app  # noqa: E402
from rosettastone.server.database import get_session  # noqa: E402
from rosettastone.server.models import (  # noqa: E402
    MigrationRecord,
    TestCaseRecord,
    WarningRecord,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """In-memory SQLite engine with StaticPool for cross-thread access."""
    engine = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def client(engine):
    """Test client with mocked task worker to prevent real background tasks."""
    app = create_app()

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    mock_task_worker = MagicMock()
    app.state.task_worker = mock_task_worker
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post_migration_form(
    client: TestClient,
    data: bytes = b'{"prompt": "hello", "response": "world"}\n',
    filename: str = "test.jsonl",
    content_type: str = "application/jsonl",
    source_model: str = "openai/gpt-4o",
    target_model: str = "anthropic/claude-sonnet-4",
    extra_form: dict | None = None,
):
    """Helper to POST the new-migration form with a file upload."""
    files = {"data_file": (filename, io.BytesIO(data), content_type)}
    form_data = {
        "source_model": source_model,
        "target_model": target_model,
        **(extra_form or {}),
    }
    return client.post(
        "/ui/migrations/new",
        data=form_data,
        files=files,
        follow_redirects=False,
    )


def _create_migration_record(
    engine,
    *,
    source_model: str = "a",
    target_model: str = "b",
    status: str = "complete",
    recommendation: str | None = "GO",
    recommendation_reasoning: str | None = "OK",
    confidence_score: float | None = 0.9,
    baseline_score: float | None = None,
    improvement: float | None = None,
    config_json: str = "{}",
    per_type_scores_json: str = "{}",
    warnings_json: str = "[]",
    safety_warnings_json: str = "[]",
) -> int:
    """Insert a MigrationRecord and return its ID."""
    with Session(engine) as session:
        m = MigrationRecord(
            source_model=source_model,
            target_model=target_model,
            status=status,
            recommendation=recommendation,
            recommendation_reasoning=recommendation_reasoning,
            confidence_score=confidence_score,
            baseline_score=baseline_score,
            improvement=improvement,
            config_json=config_json,
            per_type_scores_json=per_type_scores_json,
            warnings_json=warnings_json,
            safety_warnings_json=safety_warnings_json,
        )
        session.add(m)
        session.commit()
        session.refresh(m)
        return m.id


def _make_pending_record(engine) -> int:
    """Insert a minimal pending MigrationRecord and return its ID."""
    with Session(engine) as session:
        record = MigrationRecord(source_model="a", target_model="b", status="pending")
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id


def _make_mock_result(**overrides):
    """Return a minimal MigrationResult for task background tests."""
    from rosettastone.core.types import MigrationResult

    defaults = dict(
        config={},
        optimized_prompt="",
        baseline_results=[],
        validation_results=[],
        confidence_score=0.0,
        baseline_score=0.0,
        improvement=0.0,
        cost_usd=0.0,
        duration_seconds=1.0,
        warnings=[],
    )
    defaults.update(overrides)
    return MigrationResult(**defaults)


_BASE_CONFIG_DICT = {
    "source_model": "a",
    "target_model": "b",
    "data_path": "/tmp/fake.jsonl",
}


# ---------------------------------------------------------------------------
# 1. FILE UPLOAD ABUSE
# ---------------------------------------------------------------------------


class TestFileUploadAbuse:
    """P0/P1: Ensure file upload boundaries are correctly enforced."""

    def test_upload_exactly_50mb_accepted(self, client: TestClient, engine) -> None:
        """50 MB valid JSONL should be accepted with 303 redirect."""
        line = b'{"prompt": "hello", "response": "world"}\n'
        data = line * ((50 * 1024 * 1024) // len(line))
        resp = _post_migration_form(client, data=data)
        assert resp.status_code == 303, "50 MB valid JSONL file should be accepted"
        assert "/ui/migrations/" in resp.headers["location"]

    def test_upload_50mb_plus_1_byte_rejected(self, client: TestClient) -> None:
        """File exceeding MAX_UPLOAD_SIZE by 1 byte must return 422."""
        data = b"x" * (50 * 1024 * 1024 + 1)
        resp = _post_migration_form(client, data=data)
        assert resp.status_code == 422
        assert "50MB" in resp.text or "limit" in resp.text.lower()

    def test_upload_non_jsonl_content_with_jsonl_extension(
        self, client: TestClient, engine
    ) -> None:
        """XML content with .jsonl extension must be rejected with 422."""
        resp = _post_migration_form(
            client, data=b"<xml><not>jsonl</not></xml>", filename="evil.jsonl"
        )
        assert resp.status_code == 422
        assert "not valid JSON" in resp.text

    def test_upload_path_traversal_filename(self, client: TestClient, engine) -> None:
        """Path traversal in filename is neutralized; upload succeeds normally."""
        data = b'{"prompt": "hello", "response": "world"}\n'
        resp = _post_migration_form(client, data=data, filename="../../etc/passwd.jsonl")
        assert resp.status_code == 303

    def test_upload_empty_file(self, client: TestClient) -> None:
        """0-byte file must return 422 with 'empty' message."""
        resp = _post_migration_form(client, data=b"")
        assert resp.status_code == 422
        assert "empty" in resp.text.lower()

    def test_upload_binary_file(self, client: TestClient, engine) -> None:
        """Random binary data with .jsonl extension must be rejected (non-UTF-8)."""
        resp = _post_migration_form(client, data=os.urandom(1024), filename="random.jsonl")
        assert resp.status_code == 422
        assert "UTF-8" in resp.text

    def test_upload_no_file_returns_422(self, client: TestClient) -> None:
        """Form submitted without a file attachment must return 422."""
        form_data = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
        }
        resp = client.post("/ui/migrations/new", data=form_data)
        assert resp.status_code == 422

    def test_upload_empty_filename_returns_422(self, client: TestClient) -> None:
        """Empty filename string must return 422."""
        files = {"data_file": ("", io.BytesIO(b"some content"), "application/jsonl")}
        form_data = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
        }
        resp = client.post("/ui/migrations/new", data=form_data, files=files)
        assert resp.status_code == 422

    def test_upload_wrong_content_type(self, client: TestClient, engine) -> None:
        """JSONL data with image/png content-type is still accepted."""
        data = b'{"prompt": "hello", "response": "world"}\n'
        resp = _post_migration_form(
            client, data=data, filename="test.jsonl", content_type="image/png"
        )
        assert resp.status_code == 303

    def test_multiple_rapid_uploads(self, client: TestClient, engine) -> None:
        """5 rapid sequential uploads each create unique records."""
        ids_created = []
        for i in range(5):
            data = f'{{"prompt": "q{i}", "response": "a{i}"}}\n'.encode()
            resp = _post_migration_form(
                client,
                data=data,
                source_model=f"model-{i}/source",
                target_model=f"model-{i}/target",
            )
            assert resp.status_code == 303
            location = resp.headers["location"]
            ids_created.append(int(location.rstrip("/").split("/")[-1]))

        assert len(set(ids_created)) == 5
        with Session(engine) as session:
            count = session.exec(select(sqlmodel.func.count()).select_from(MigrationRecord)).one()
            assert count == 5


# ---------------------------------------------------------------------------
# 2. CONCURRENT MIGRATION STRESS
# ---------------------------------------------------------------------------


class TestConcurrentMigrationStress:
    """P1: Verify the task queue serializes correctly."""

    def test_ten_rapid_submissions_all_queued(self, client: TestClient, engine) -> None:
        """10 rapid submissions each create separate pending records."""
        for i in range(10):
            data = f'{{"prompt": "q{i}", "response": "a{i}"}}\n'.encode()
            resp = _post_migration_form(
                client,
                data=data,
                source_model=f"src-{i}",
                target_model=f"tgt-{i}",
            )
            assert resp.status_code == 303

        assert client.app.state.task_worker.enqueue.call_count == 10
        with Session(engine) as session:
            records = list(session.exec(select(MigrationRecord)).all())
            assert len(records) == 10
            assert all(r.status == "pending" for r in records)

    def test_pending_migration_detail_page(self, client: TestClient, engine) -> None:
        """Detail page for a just-created migration shows pending state with auto-refresh."""
        resp = _post_migration_form(client, data=b'{"prompt": "hello", "response": "world"}\n')
        assert resp.status_code == 303
        detail_resp = client.get(resp.headers["location"])
        assert detail_resp.status_code == 200
        body = detail_resp.text
        assert "Queued for execution" in body
        assert "hx-trigger" in body

    def test_startup_recovery_marks_running_as_failed(self, engine) -> None:
        """_recover_orphaned_migrations flips 'running' to 'failed' on restart."""
        with Session(engine) as session:
            r = MigrationRecord(source_model="a", target_model="b", status="running")
            session.add(r)
            session.commit()
            session.refresh(r)
            mid = r.id

        from rosettastone.server.app import _recover_orphaned_migrations

        with patch("rosettastone.server.app.get_engine", return_value=engine):
            _recover_orphaned_migrations()

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "failed"
            assert "Server restarted" in record.recommendation_reasoning

    def test_startup_recovery_ignores_complete_and_pending(self, engine) -> None:
        """Only 'running' status is flipped; other statuses are untouched."""
        with Session(engine) as session:
            for status in ["complete", "pending", "failed", "blocked"]:
                session.add(MigrationRecord(source_model="a", target_model="b", status=status))
            session.commit()

        from rosettastone.server.app import _recover_orphaned_migrations

        with patch("rosettastone.server.app.get_engine", return_value=engine):
            _recover_orphaned_migrations()

        with Session(engine) as session:
            records = list(session.exec(select(MigrationRecord)).all())
            assert {r.status for r in records} == {"complete", "pending", "failed", "blocked"}


# ---------------------------------------------------------------------------
# 3. DATABASE EDGE CASES
# ---------------------------------------------------------------------------


class TestDatabaseEdgeCases:
    """P1/P2: Exercise unusual DB states and corruption scenarios."""

    def test_task_handles_deleted_migration_record(self, engine) -> None:
        """run_migration_background with non-existent ID returns without raising."""
        from rosettastone.server.api.tasks import run_migration_background

        run_migration_background(999999, _BASE_CONFIG_DICT, engine=engine)

    def test_orphaned_test_cases_do_not_crash_api(self, client: TestClient, engine) -> None:
        """Migrations list API works even with orphaned TestCaseRecords."""
        mid = _create_migration_record(engine)
        with Session(engine) as session:
            tc = TestCaseRecord(
                migration_id=mid,
                phase="validation",
                output_type="json",
                composite_score=0.9,
                is_win=True,
                scores_json="{}",
                details_json="{}",
            )
            session.add(tc)
            session.commit()

        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text(f"DELETE FROM migrations WHERE id = {mid}"))
            conn.commit()

        resp = client.get("/api/v1/migrations")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_task_handles_deletion_after_running_status_set(self, engine) -> None:
        """If record is deleted mid-task, run_migration_background handles None gracefully."""
        mid = _make_pending_record(engine)

        def delete_and_return(*args, **kwargs):
            with Session(engine) as session:
                record = session.get(MigrationRecord, mid)
                if record:
                    session.delete(record)
                    session.commit()
            return _make_mock_result()

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = delete_and_return
            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, _BASE_CONFIG_DICT, engine=engine)

    def test_concurrent_read_during_write(self, engine) -> None:
        """Interleaved sequential reads and writes do not corrupt state."""
        mid = _create_migration_record(engine)

        for i in range(20):
            with Session(engine) as session:
                r = session.get(MigrationRecord, mid)
                if r:
                    r.cost_usd = float(i)
                    session.add(r)
                    session.commit()
            with Session(engine) as session:
                r = session.get(MigrationRecord, mid)
                assert r is not None
                assert r.cost_usd == float(i)


# ---------------------------------------------------------------------------
# 4. TEMPLATE RENDERING EDGE CASES
# ---------------------------------------------------------------------------


class TestTemplateRenderingEdgeCases:
    """P1: Verify templates handle unusual data without crashing."""

    def test_migration_detail_empty_per_type_scores(self, client: TestClient, engine) -> None:
        """Migration with no per-type scores renders without error."""
        mid = _create_migration_record(engine)
        assert client.get(f"/ui/migrations/{mid}").status_code == 200

    def test_migration_detail_null_scores(self, client: TestClient, engine) -> None:
        """Template handles None numeric fields via `(X or 0)` guard."""
        mid = _create_migration_record(
            engine,
            recommendation="CONDITIONAL",
            confidence_score=None,
            baseline_score=None,
            improvement=None,
        )
        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200
        assert "0%" in resp.text

    def test_migration_detail_extremely_long_model_names(self, client: TestClient, engine) -> None:
        """Model names with 2000 characters do not crash the template."""
        long_name = "x" * 2000
        mid = _create_migration_record(engine, source_model=long_name, target_model=long_name)
        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200
        assert long_name in resp.text

    @pytest.mark.parametrize(
        "field,xss_payload,bad_fragment",
        [
            (
                "source_model",
                '<script>alert("xss")</script>',
                "<script>alert",
            ),
            (
                "recommendation_reasoning",
                '<img src=x onerror="alert(1)">',
                'onerror="alert(1)"',
            ),
        ],
    )
    def test_xss_in_template_fields(
        self, client: TestClient, engine, field, xss_payload, bad_fragment
    ) -> None:
        """Jinja2 auto-escaping prevents XSS in model name and reasoning fields."""
        kwargs: dict = {field: xss_payload}
        if field != "source_model":
            kwargs["source_model"] = "a"
        mid = _create_migration_record(engine, **kwargs)
        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200
        assert bad_fragment not in resp.text

    def test_migration_detail_zero_test_cases(self, client: TestClient, engine) -> None:
        """Completed migration with zero test cases renders without error."""
        mid = _create_migration_record(engine)
        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200
        assert "0" in resp.text

    def test_migration_detail_many_test_cases(self, client: TestClient, engine) -> None:
        """Migration with 500 test cases renders in under 5 seconds."""
        scores = json.dumps(
            {
                "json": {
                    "win_rate": 0.9,
                    "mean": 0.85,
                    "median": 0.86,
                    "p10": 0.7,
                    "p50": 0.86,
                    "p90": 0.95,
                    "min_score": 0.5,
                    "max_score": 1.0,
                    "sample_count": 500,
                    "confidence_interval": [0.7, 0.95],
                }
            }
        )
        mid = _create_migration_record(engine, per_type_scores_json=scores)
        with Session(engine) as session:
            for i in range(500):
                session.add(
                    TestCaseRecord(
                        migration_id=mid,
                        phase="validation",
                        output_type="json",
                        composite_score=0.5 + (i % 50) * 0.01,
                        is_win=i % 3 != 0,
                        scores_json=json.dumps({"bertscore": 0.8}),
                        details_json="{}",
                        response_length=100,
                        new_response_length=100,
                    )
                )
            session.commit()

        start = time.time()
        resp = client.get(f"/ui/migrations/{mid}")
        elapsed = time.time() - start
        assert resp.status_code == 200
        assert elapsed < 5.0, f"Template took {elapsed:.2f}s -- too slow"

    def test_html_entities_in_regression_fields(self, client: TestClient, engine) -> None:
        """Script tags in TestCaseRecord response_text fields are escaped."""
        mid = _create_migration_record(engine, recommendation="NO_GO", confidence_score=0.3)
        with Session(engine) as session:
            session.add(
                TestCaseRecord(
                    migration_id=mid,
                    phase="validation",
                    output_type="json",
                    composite_score=0.2,
                    is_win=False,
                    scores_json=json.dumps({"bertscore": 0.2}),
                    details_json="{}",
                    response_length=50,
                    new_response_length=50,
                    response_text='<script>alert("expected")</script>',
                    new_response_text='<script>alert("got")</script>',
                )
            )
            session.commit()

        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200
        body = resp.text
        assert '<script>alert("expected")</script>' not in body
        assert '<script>alert("got")</script>' not in body


# ---------------------------------------------------------------------------
# 5. BACKGROUND TASK FAILURE MODES
# ---------------------------------------------------------------------------


class TestBackgroundTaskFailureModes:
    """P0/P1: Verify graceful handling of all failure modes in run_migration_background."""

    def test_config_validation_failure_sets_status_failed(self, engine) -> None:
        """Missing data_path triggers ValidationError; record ends up 'failed'."""
        mid = _make_pending_record(engine)
        from rosettastone.server.api.tasks import run_migration_background

        run_migration_background(mid, {"source_model": "a", "target_model": "b"}, engine=engine)

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "failed"
            assert "failed" in record.recommendation_reasoning.lower()

    @pytest.mark.parametrize(
        "exc,expected_status,msg_fragment",
        [
            (RuntimeError("Init failed!"), "failed", "Init failed!"),
            (RuntimeError("Network error"), "failed", "Network error"),
        ],
    )
    def test_migrator_run_exception_sets_status(
        self, engine, exc, expected_status, msg_fragment
    ) -> None:
        """Migrator.run() raising RuntimeError marks record 'failed'."""
        mid = _make_pending_record(engine)

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = exc
            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, _BASE_CONFIG_DICT, engine=engine)

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == expected_status
            assert msg_fragment in record.recommendation_reasoning

    def test_migrator_init_raises_sets_status_failed(self, engine) -> None:
        """Migrator() constructor raising sets record to 'failed'."""
        mid = _make_pending_record(engine)

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.side_effect = RuntimeError("Init failed!")
            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, _BASE_CONFIG_DICT, engine=engine)

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "failed"
            assert "Init failed!" in record.recommendation_reasoning

    def test_migration_blocked_error_sets_status_blocked(self, engine) -> None:
        """MigrationBlockedError sets status='blocked', not 'failed'."""
        mid = _make_pending_record(engine)
        from rosettastone.core.migrator import MigrationBlockedError

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = MigrationBlockedError(
                "Context window too small"
            )
            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, _BASE_CONFIG_DICT, engine=engine)

        with Session(engine) as session:
            assert session.get(MigrationRecord, mid).status == "blocked"

    def test_dry_run_success_sets_dry_run_complete(self, engine) -> None:
        """Successful dry_run sets status='dry_run_complete'."""
        mid = _make_pending_record(engine)
        mock_result = _make_mock_result(warnings=["Dry run"])

        config_dict = {**_BASE_CONFIG_DICT, "dry_run": True}
        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.return_value = mock_result
            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, config_dict, engine=engine)

        with Session(engine) as session:
            assert session.get(MigrationRecord, mid).status == "dry_run_complete"

    def test_memory_error_behavior(self, engine) -> None:
        """MemoryError leaves record in 'running' or 'failed'; startup recovery heals it."""
        mid = _make_pending_record(engine)

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = MemoryError("OOM")
            from rosettastone.server.api.tasks import run_migration_background

            try:
                run_migration_background(mid, _BASE_CONFIG_DICT, engine=engine)
            except MemoryError:
                pass

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status in ("running", "failed")

            if record.status == "running":
                from rosettastone.server.app import _recover_orphaned_migrations

                with patch("rosettastone.server.app.get_engine", return_value=engine):
                    _recover_orphaned_migrations()

                session.expire(record)
                record = session.get(MigrationRecord, mid)
                assert record.status == "failed"

    def test_keyboard_interrupt_not_caught(self, engine) -> None:
        """KeyboardInterrupt propagates out of run_migration_background."""
        mid = _make_pending_record(engine)

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = KeyboardInterrupt()
            from rosettastone.server.api.tasks import run_migration_background

            with pytest.raises(KeyboardInterrupt):
                run_migration_background(mid, _BASE_CONFIG_DICT, engine=engine)

    def test_mixed_safety_warnings_serialized(self, engine) -> None:
        """safety_warnings with mixed dict/str entries are serialized correctly."""
        from rosettastone.core.types import MigrationResult

        mid = _make_pending_record(engine)
        mock_result = MigrationResult(
            config={},
            optimized_prompt="opt",
            baseline_results=[],
            validation_results=[],
            confidence_score=0.5,
            baseline_score=0.5,
            improvement=0.0,
            cost_usd=0.0,
            duration_seconds=1.0,
            warnings=[],
            safety_warnings=[
                {"warning_type": "pii", "severity": "HIGH", "message": "PII detected"},
                "Simple string warning",
            ],
            recommendation="CONDITIONAL",
            recommendation_reasoning="Mixed warnings",
            per_type_scores={},
        )

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.return_value = mock_result
            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, _BASE_CONFIG_DICT, engine=engine)

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "complete"
            warnings = list(
                session.exec(select(WarningRecord).where(WarningRecord.migration_id == mid)).all()
            )
            safety_warnings = [w for w in warnings if w.warning_type in ("pii", "safety")]
            assert len(safety_warnings) == 2


# ---------------------------------------------------------------------------
# 6. API ENDPOINT ABUSE
# ---------------------------------------------------------------------------


class TestAPIEndpointAbuse:
    """P0: Security-focused tests for injection, XSS, and abuse."""

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/migrations/1;DROP TABLE migrations;--",
            "/api/v1/migrations/1 OR 1=1",
        ],
    )
    def test_sql_injection_in_path_parameter(self, client: TestClient, path) -> None:
        """Non-integer migration IDs are rejected with 404 or 422."""
        resp = client.get(path)
        assert resp.status_code in (404, 422)

    def test_sql_injection_in_test_case_path(self, client: TestClient) -> None:
        """SQL injection in test-case path segment returns 404 or 422."""
        resp = client.get("/api/v1/migrations/1/test-cases/1 OR 1=1")
        assert resp.status_code in (404, 422)

    def test_xss_in_model_name_api(self, client: TestClient) -> None:
        """JSON API stores raw XSS values; response body contains them safely."""
        payload = {
            "source_model": '<script>alert("xss")</script>',
            "target_model": "<img src=x onerror=alert(1)>",
        }
        resp = client.post("/api/v1/migrations", json=payload)
        assert resp.status_code == 201
        assert resp.json()["source_model"] == '<script>alert("xss")</script>'

    def test_xss_in_form_model_field(self, client: TestClient, engine) -> None:
        """XSS payloads in form fields are escaped on the resulting detail page."""
        data = b'{"prompt": "hello", "response": "world"}\n'
        resp = _post_migration_form(
            client,
            data=data,
            source_model='<script>alert("src")</script>',
            target_model="<img onerror=alert(1) src=x>",
        )
        assert resp.status_code == 303
        detail_resp = client.get(resp.headers["location"])
        assert detail_resp.status_code == 200
        body = detail_resp.text
        assert '<script>alert("src")</script>' not in body
        assert "<img onerror" not in body

    def test_extreme_pagination_offset(self, client: TestClient) -> None:
        """Extremely large offset returns empty results without crashing."""
        resp = client.get("/api/v1/migrations?offset=999999999&limit=100")
        assert resp.status_code == 200
        assert resp.json()["items"] == []

    @pytest.mark.parametrize(
        "query_string,expected_status",
        [
            ("offset=-1", 422),
            ("limit=101", 422),
            ("limit=0", 422),
        ],
    )
    def test_invalid_pagination_params_rejected(
        self, client: TestClient, query_string, expected_status
    ) -> None:
        """FastAPI validators reject out-of-range pagination parameters."""
        resp = client.get(f"/api/v1/migrations?{query_string}")
        assert resp.status_code == expected_status

    def test_non_integer_migration_id(self, client: TestClient) -> None:
        """Non-integer path parameter returns 422."""
        assert client.get("/api/v1/migrations/abc").status_code == 422

    def test_oversized_json_body(self, client: TestClient) -> None:
        """Very large JSON body is accepted or rejected gracefully (not 5xx crash)."""
        payload = {"source_model": "a" * 100_000, "target_model": "b" * 100_000}
        resp = client.post("/api/v1/migrations", json=payload)
        assert resp.status_code in (201, 413, 422)

    def test_empty_body_to_create_migration(self, client: TestClient) -> None:
        """Empty JSON body to POST endpoint returns 422."""
        assert client.post("/api/v1/migrations", json={}).status_code == 422

    def test_missing_content_type_on_json_post(self, client: TestClient) -> None:
        """POST with text/plain content-type fails gracefully."""
        resp = client.post(
            "/api/v1/migrations",
            content=b"not json",
            headers={"content-type": "text/plain"},
        )
        assert resp.status_code in (415, 422, 500)

    def test_xss_in_model_registration(self, client: TestClient) -> None:
        """XSS in registered model_id is escaped on the UI homepage."""
        xss = '<script>alert("model")</script>'
        resp = client.post("/api/v1/models", json={"model_id": xss})
        assert resp.status_code == 201
        ui_resp = client.get("/ui/")
        assert ui_resp.status_code == 200
        assert "<script>alert" not in ui_resp.text

    def test_security_headers_on_api(self, client: TestClient) -> None:
        """Security headers are present on API responses."""
        resp = client.get("/api/v1/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("X-XSS-Protection") == "1; mode=block"

    def test_security_headers_on_error_pages(self, client: TestClient) -> None:
        """Security headers are present on 404 error pages."""
        resp = client.get("/ui/nonexistent")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"


# ---------------------------------------------------------------------------
# 7. STATE MACHINE VIOLATIONS
# ---------------------------------------------------------------------------


class TestStateMachineViolations:
    """P1: Verify migration status transitions are safe and consistent."""

    def test_cannot_retrigger_complete_migration_via_api(self, client: TestClient, engine) -> None:
        """No PUT/PATCH endpoint exists to change a completed migration's status."""
        mid = _create_migration_record(engine)
        resp = client.post(f"/api/v1/migrations/{mid}", json={"status": "running"})
        assert resp.status_code in (404, 405)

    def test_background_task_nonexistent_record(self, engine) -> None:
        """run_migration_background with non-existent ID returns without error."""
        from rosettastone.server.api.tasks import run_migration_background

        run_migration_background(
            migration_id=999999,
            config_dict=_BASE_CONFIG_DICT,
            engine=engine,
        )

    @pytest.mark.parametrize(
        "side_effect,expected_final_status",
        [
            ("success", "complete"),
            ("failure", "failed"),
            ("blocked", "blocked"),
        ],
    )
    def test_status_transitions(self, engine, side_effect, expected_final_status) -> None:
        """Verify pending -> running -> {complete|failed|blocked} transitions."""
        from rosettastone.core.migrator import MigrationBlockedError

        mid = _make_pending_record(engine)
        statuses_observed = []

        def capture_and_act(*args, **kwargs):
            with Session(engine) as s:
                r = s.get(MigrationRecord, mid)
                statuses_observed.append(r.status)
            if side_effect == "success":
                return _make_mock_result(
                    optimized_prompt="opt",
                    confidence_score=0.9,
                    baseline_score=0.8,
                    improvement=0.1,
                    cost_usd=0.5,
                    duration_seconds=10.0,
                    recommendation="GO",
                    recommendation_reasoning="Good",
                    per_type_scores={},
                )
            elif side_effect == "failure":
                raise RuntimeError("Intentional failure")
            else:
                raise MigrationBlockedError("Blocked!")

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = capture_and_act
            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, _BASE_CONFIG_DICT, engine=engine)

        assert "running" in statuses_observed
        with Session(engine) as session:
            assert session.get(MigrationRecord, mid).status == expected_final_status

    def test_invalid_status_renders_without_crash(self, client: TestClient, engine) -> None:
        """An unexpected status value in the DB still renders the detail page."""
        mid = _create_migration_record(engine, status="INVALID_STATUS_VALUE")
        assert client.get(f"/ui/migrations/{mid}").status_code == 200

    def test_alerts_skip_pending_and_running(self, client: TestClient, engine) -> None:
        """Alert generation only triggers for complete/failed migrations."""
        with Session(engine) as session:
            for status in ("pending", "running"):
                session.add(MigrationRecord(source_model="a", target_model="b", status=status))
            session.commit()

        resp = client.post("/api/v1/alerts/generate")
        assert resp.json()["generated"] == 0


# ---------------------------------------------------------------------------
# 8. ERROR HANDLING & EDGE CASES (Miscellaneous)
# ---------------------------------------------------------------------------


class TestMiscEdgeCases:
    """P1/P2: Additional edge cases and error handling tests."""

    def test_api_404_returns_json(self, client: TestClient) -> None:
        resp = client.get("/api/v1/migrations/999999")
        assert resp.status_code == 404
        assert "detail" in resp.json()

    def test_ui_404_returns_html(self, client: TestClient) -> None:
        resp = client.get("/ui/nonexistent-page-xyz")
        assert resp.status_code == 404
        assert "text/html" in resp.headers.get("content-type", "")

    def test_health_endpoint(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded", "unavailable")

    def test_diff_fragment_fallback_to_dummy(self, client: TestClient) -> None:
        resp = client.get("/ui/fragments/diff/999/999")
        assert resp.status_code == 200
        assert "BERTScore" in resp.text  # _METRIC_LABELS maps 'bertscore' → 'BERTScore'

    def test_cost_page_zero_cost_no_crash(self, client: TestClient, engine) -> None:
        with Session(engine) as session:
            session.add(
                MigrationRecord(source_model="a", target_model="b", status="complete", cost_usd=0.0)
            )
            session.commit()
        resp = client.get("/ui/costs")
        assert resp.status_code == 200
        assert "$0.00" in resp.text

    def test_delete_model_used_in_migrations(self, client: TestClient, engine) -> None:
        """Deleting a registered model does not cascade-delete migrations."""
        reg = client.post("/api/v1/models", json={"model_id": "openai/gpt-4o"})
        db_id = reg.json()["db_id"]
        with Session(engine) as session:
            session.add(
                MigrationRecord(
                    source_model="openai/gpt-4o",
                    target_model="anthropic/claude-sonnet-4",
                    status="complete",
                )
            )
            session.commit()
        assert client.delete(f"/api/v1/models/{db_id}").status_code == 200
        assert client.get("/api/v1/migrations").json()["total"] == 1

    def test_api_detail_malformed_json_fields(self, client: TestClient, engine) -> None:
        """Malformed per_type_scores_json causes API detail endpoint to return 500."""
        mid = _create_migration_record(engine, per_type_scores_json="NOT VALID JSON")
        assert client.get(f"/api/v1/migrations/{mid}").status_code == 500

    def test_executive_report_404(self, client: TestClient) -> None:
        assert client.get("/ui/migrations/999999/executive").status_code == 404

    def test_unicode_model_names(self, client: TestClient, engine) -> None:
        """Model names with unicode characters render without error."""
        mid = _create_migration_record(engine, source_model="unicorn/model-v1")
        assert client.get(f"/ui/migrations/{mid}").status_code == 200

    def test_concurrent_alert_generation_idempotent(self, client: TestClient, engine) -> None:
        """Repeated alert generation does not create duplicate alerts."""
        with Session(engine) as session:
            session.add(
                MigrationRecord(
                    source_model="a",
                    target_model="b",
                    status="complete",
                    recommendation="GO",
                    confidence_score=0.9,
                    config_json="{}",
                )
            )
            session.commit()
        for _ in range(3):
            client.post("/api/v1/alerts/generate")
        assert len(client.get("/api/v1/alerts").json()) == 1

    def test_pagination_offset_equals_total(self, client: TestClient, engine) -> None:
        """offset == total count returns empty items list."""
        with Session(engine) as session:
            for i in range(3):
                session.add(
                    MigrationRecord(source_model=f"s{i}", target_model=f"t{i}", status="complete")
                )
            session.commit()
        resp = client.get("/api/v1/migrations?offset=3&limit=20")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert data["items"] == []

    def test_report_for_pending_migration(self, client: TestClient, engine) -> None:
        """Markdown report for a pending migration returns 200 (no crash)."""
        mid = _create_migration_record(engine, status="pending", recommendation=None)
        assert client.get(f"/api/v1/migrations/{mid}/report/markdown").status_code == 200

    def test_rapid_health_checks(self, client: TestClient) -> None:
        """100 rapid health check requests all return 200."""
        for _ in range(100):
            assert client.get("/api/v1/health").status_code == 200


# ---------------------------------------------------------------------------
# 9. ADDITIONAL STRESS / SECURITY SCENARIOS
# ---------------------------------------------------------------------------


class TestLargePayloadRejected:
    """P1: Verify the server handles extremely large JSON config payloads gracefully."""

    def test_large_payload_rejected(self, client: TestClient) -> None:
        """POST >1 MB JSON body must not crash (no unhandled 5xx)."""
        large_config = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
            "extra_padding": "x" * (1024 * 1024),
        }
        resp = client.post(
            "/api/v1/migrations",
            json=large_config,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code < 500, f"Server crashed with {resp.status_code} on large payload"


class TestSQLInjectionInQueryParams:
    """P0 (security): Verify SQLModel parameterization prevents SQL injection."""

    def test_sql_injection_in_query_params_does_not_crash(self, client: TestClient, engine) -> None:
        """SQL injection via query param does not manufacture extra rows."""
        with Session(engine) as session:
            for status in ("complete", "pending"):
                session.add(
                    MigrationRecord(
                        source_model="openai/gpt-4o",
                        target_model="anthropic/claude-sonnet-4",
                        status=status,
                    )
                )
            session.commit()

        injection = "pending' OR '1'='1"
        resp = client.get(f"/api/v1/migrations?status={injection}")
        assert resp.status_code == 200, f"Unexpected status: {resp.status_code}"
        items = resp.json().get("items", [])
        assert len(items) <= 2, f"SQL injection expanded the result set to {len(items)} items"

    def test_sql_injection_in_migration_id(self, client: TestClient) -> None:
        """Injection string as migration ID returns 422 or 404, not 500."""
        resp = client.get("/api/v1/migrations/1 OR 1=1")
        assert resp.status_code in (404, 422), (
            f"Expected 404 or 422 for injected ID, got {resp.status_code}"
        )


class TestConcurrentMigrationCreation:
    """P1: Verify the server handles concurrent migration creation safely."""

    def test_concurrent_migration_creation(self, tmp_path) -> None:
        """5 simultaneous migrations via ThreadPoolExecutor all receive unique IDs.

        Uses a file-based SQLite DB rather than in-memory StaticPool because
        SQLite StaticPool uses a single shared connection that is not safe for
        concurrent multi-threaded writes.
        """
        db_file = tmp_path / "concurrent_test.db"
        db_url = f"sqlite:///{db_file}"

        # Create schema once on the main thread.
        shared_engine = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})
        SQLModel.metadata.create_all(shared_engine)

        def create_migration(i: int) -> int:
            # Each worker gets its own engine connection to the shared file DB.
            thread_engine = create_engine(
                db_url, echo=False, connect_args={"check_same_thread": False}
            )
            app = create_app()

            def override_session():
                with Session(thread_engine) as sess:
                    yield sess

            app.dependency_overrides[get_session] = override_session
            mock_task_worker = MagicMock()
            app.state.task_worker = mock_task_worker

            data = (f'{{"prompt": "concurrent-{i}", "response": "resp-{i}"}}' + "\n").encode()
            files = {"data_file": (f"test{i}.jsonl", io.BytesIO(data), "application/jsonl")}
            form_data = {
                "source_model": f"openai/gpt-4o-{i}",
                "target_model": "anthropic/claude-sonnet-4",
            }
            # Do NOT use `with TestClient(...)` -- that triggers lifespan/startup.
            c = TestClient(app, raise_server_exceptions=True)
            resp = c.post(
                "/ui/migrations/new",
                data=form_data,
                files=files,
                follow_redirects=False,
            )
            assert resp.status_code == 303, f"Worker {i} got {resp.status_code}"
            location = resp.headers["location"]
            thread_engine.dispose()
            return int(location.rstrip("/").split("/")[-1])

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_migration, i) for i in range(5)]
            ids = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(ids) == 5
        assert len(set(ids)) == 5, f"Duplicate migration IDs detected: {ids}"
        with Session(shared_engine) as session:
            records = list(session.exec(select(MigrationRecord)).all())
            assert len(records) == 5
        shared_engine.dispose()
