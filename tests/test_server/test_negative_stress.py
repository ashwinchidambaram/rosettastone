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
def session(engine):
    with Session(engine) as s:
        yield s


@pytest.fixture
def client(engine):
    """Test client with mocked executor to prevent real background tasks."""
    app = create_app()

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    mock_executor = MagicMock()
    app.state.executor = mock_executor
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def _sample_migration(session: Session) -> MigrationRecord:
    """Insert a minimal complete migration record."""
    record = MigrationRecord(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        status="complete",
        confidence_score=0.92,
        baseline_score=0.85,
        improvement=0.07,
        cost_usd=1.23,
        duration_seconds=45.6,
        recommendation="GO",
        recommendation_reasoning="All types pass thresholds.",
        config_json=json.dumps(
            {
                "source_model": "openai/gpt-4o",
                "target_model": "anthropic/claude-sonnet-4",
            }
        ),
        per_type_scores_json=json.dumps(
            {
                "json": {
                    "win_rate": 0.95,
                    "mean": 0.93,
                    "median": 0.94,
                    "p10": 0.88,
                    "p50": 0.94,
                    "p90": 0.98,
                    "min_score": 0.85,
                    "max_score": 1.0,
                    "sample_count": 20,
                    "confidence_interval": [0.88, 0.99],
                },
            }
        ),
        warnings_json=json.dumps(["Low sample count for classification"]),
        safety_warnings_json=json.dumps([]),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


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


# ---------------------------------------------------------------------------
# 1. FILE UPLOAD ABUSE
# ---------------------------------------------------------------------------


class TestFileUploadAbuse:
    """P0/P1: Ensure file upload boundaries are correctly enforced."""

    # --- 1a. Exactly 50 MB file (boundary) --- P0
    def test_upload_exactly_50mb_accepted(self, client: TestClient, engine) -> None:
        """Upload a file that is exactly MAX_UPLOAD_SIZE (50 MB) with valid JSONL.
        Expected: accepted, 303 redirect created.
        """
        line = b'{"prompt": "hello", "response": "world"}\n'
        repeat = (50 * 1024 * 1024) // len(line)
        data = line * repeat  # valid JSONL, close to 50 MB
        resp = _post_migration_form(client, data=data)
        assert resp.status_code == 303, "50 MB valid JSONL file should be accepted"
        assert "/ui/migrations/" in resp.headers["location"]

    # --- 1b. 50 MB + 1 byte (should reject) --- P0
    def test_upload_50mb_plus_1_byte_rejected(self, client: TestClient) -> None:
        """Upload a file that exceeds MAX_UPLOAD_SIZE by 1 byte.
        Expected: 422 with error message about size limit.
        """
        data = b"x" * (50 * 1024 * 1024 + 1)
        resp = _post_migration_form(client, data=data)
        assert resp.status_code == 422
        assert "50MB" in resp.text or "limit" in resp.text.lower()

    # --- 1c. Non-JSONL content with .jsonl extension --- P1
    def test_upload_non_jsonl_content_with_jsonl_extension(
        self, client: TestClient, engine
    ) -> None:
        """Upload a file containing invalid JSONL (e.g. XML) but with .jsonl extension.
        Expected: 422 — upload validation now catches non-JSON content.
        """
        xml_content = b"<xml><not>jsonl</not></xml>"
        resp = _post_migration_form(client, data=xml_content, filename="evil.jsonl")
        assert resp.status_code == 422
        assert "not valid JSON" in resp.text

    # --- 1d. Path traversal in filename --- P0 (security)
    def test_upload_path_traversal_filename(self, client: TestClient, engine) -> None:
        """Upload a file with path traversal characters in the filename.
        Expected: the file should be saved safely; no directory escape.
        The endpoint saves to a fixed name (data.jsonl), so traversal is neutralized.
        """
        data = b'{"prompt": "hello", "response": "world"}\n'
        resp = _post_migration_form(client, data=data, filename="../../etc/passwd.jsonl")
        # Should still process normally because the server ignores the filename
        # and saves as output_dir / "data.jsonl"
        assert resp.status_code == 303

    # --- 1e. Empty file (0 bytes) --- P1
    def test_upload_empty_file(self, client: TestClient) -> None:
        """Upload a 0-byte file.
        Expected: 422 — upload validation now catches empty files.
        """
        resp = _post_migration_form(client, data=b"")
        assert resp.status_code == 422
        assert "empty" in resp.text.lower()

    # --- 1f. Binary file (random bytes) --- P1
    def test_upload_binary_file(self, client: TestClient, engine) -> None:
        """Upload a file full of random binary data with .jsonl extension.
        Expected: 422 — upload validation now catches non-UTF-8 content.
        """
        binary_data = os.urandom(1024)
        resp = _post_migration_form(client, data=binary_data, filename="random.jsonl")
        assert resp.status_code == 422
        assert "UTF-8" in resp.text

    # --- 1g. Upload with no file --- P0
    def test_upload_no_file_returns_422(self, client: TestClient) -> None:
        """Submit the form with no file attached.
        Expected: 422 with error about missing file.
        """
        form_data = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
        }
        resp = client.post("/ui/migrations/new", data=form_data)
        assert resp.status_code == 422

    # --- 1h. Upload with empty filename --- P1
    def test_upload_empty_filename_returns_422(self, client: TestClient) -> None:
        """Submit with a file object that has an empty filename string.
        Expected: 422 with error about missing file.
        """
        files = {"data_file": ("", io.BytesIO(b"some content"), "application/jsonl")}
        form_data = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
        }
        resp = client.post("/ui/migrations/new", data=form_data, files=files)
        assert resp.status_code == 422

    # --- 1i. Upload with wrong Content-Type --- P2
    def test_upload_wrong_content_type(self, client: TestClient, engine) -> None:
        """Upload JSONL data with image/png content type.
        Expected: accepted (server does not validate Content-Type of uploads).
        """
        data = b'{"prompt": "hello", "response": "world"}\n'
        resp = _post_migration_form(
            client, data=data, filename="test.jsonl", content_type="image/png"
        )
        assert resp.status_code == 303

    # --- 1j. Multiple rapid uploads (race condition) --- P1
    def test_multiple_rapid_uploads(self, client: TestClient, engine) -> None:
        """Submit 5 migrations rapidly in sequence.
        Expected: each creates a separate record; no collisions.
        """
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
            migration_id = int(location.rstrip("/").split("/")[-1])
            ids_created.append(migration_id)

        # Verify all IDs are unique
        assert len(set(ids_created)) == 5
        # Verify all records exist
        with Session(engine) as session:
            count = session.exec(select(sqlmodel.func.count()).select_from(MigrationRecord)).one()
            assert count == 5


# ---------------------------------------------------------------------------
# 2. CONCURRENT MIGRATION STRESS
# ---------------------------------------------------------------------------


class TestConcurrentMigrationStress:
    """P1: Verify the single-worker ThreadPoolExecutor serializes correctly."""

    # --- 2a. Submit 10 migrations rapidly --- P1
    def test_ten_rapid_submissions_all_queued(self, client: TestClient, engine) -> None:
        """Submit 10 migrations rapidly; verify all create separate records
        and the executor.submit is called 10 times.
        """
        for i in range(10):
            data = f'{{"prompt": "q{i}", "response": "a{i}"}}\n'.encode()
            resp = _post_migration_form(
                client,
                data=data,
                source_model=f"src-{i}",
                target_model=f"tgt-{i}",
            )
            assert resp.status_code == 303

        assert client.app.state.executor.submit.call_count == 10

        with Session(engine) as session:
            records = list(session.exec(select(MigrationRecord)).all())
            assert len(records) == 10
            assert all(r.status == "pending" for r in records)

    # --- 2b. Submit migration, immediately request detail page --- P0
    def test_pending_migration_detail_page(self, client: TestClient, engine) -> None:
        """After creating a migration, immediately GETting its detail page
        should show the 'pending' state with a spinner/auto-refresh.
        """
        data = b'{"prompt": "hello", "response": "world"}\n'
        resp = _post_migration_form(client, data=data)
        assert resp.status_code == 303

        location = resp.headers["location"]
        detail_resp = client.get(location)
        assert detail_resp.status_code == 200
        body = detail_resp.text
        assert "Queued for execution" in body
        assert "hx-trigger" in body

    # --- 2c. Startup recovery marks running -> failed --- P0
    def test_startup_recovery_marks_running_as_failed(self, engine) -> None:
        """If the server restarts while a migration is running,
        _recover_orphaned_migrations should mark it as 'failed'.
        """
        with Session(engine) as session:
            r = MigrationRecord(
                source_model="a",
                target_model="b",
                status="running",
            )
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

    # --- 2d. Startup recovery does not touch completed/pending --- P1
    def test_startup_recovery_ignores_complete_and_pending(self, engine) -> None:
        """Only 'running' status should be flipped to 'failed'."""
        with Session(engine) as session:
            for status in ["complete", "pending", "failed", "blocked"]:
                session.add(
                    MigrationRecord(
                        source_model="a",
                        target_model="b",
                        status=status,
                    )
                )
            session.commit()

        from rosettastone.server.app import _recover_orphaned_migrations

        with patch("rosettastone.server.app.get_engine", return_value=engine):
            _recover_orphaned_migrations()

        with Session(engine) as session:
            records = list(session.exec(select(MigrationRecord)).all())
            statuses = {r.status for r in records}
            assert statuses == {"complete", "pending", "failed", "blocked"}


# ---------------------------------------------------------------------------
# 3. DATABASE EDGE CASES
# ---------------------------------------------------------------------------


class TestDatabaseEdgeCases:
    """P1/P2: Exercise unusual DB states and corruption scenarios."""

    # --- 3a. Migration record deleted while background task references it --- P1
    def test_task_handles_deleted_migration_record(self, engine) -> None:
        """If the record is deleted after the task starts, the task
        should gracefully return (it checks record is None).
        """
        from rosettastone.server.api.tasks import run_migration_background

        config_dict = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
            "data_path": "/tmp/fake.jsonl",
        }
        # Use a non-existent migration_id
        run_migration_background(999999, config_dict, engine=engine)
        # Should not raise; gracefully logs and returns

    # --- 3b. Orphaned test cases (TestCaseRecord with no parent) --- P2
    def test_orphaned_test_cases_do_not_crash_api(self, client: TestClient, engine) -> None:
        """TestCaseRecords whose migration_id references a deleted MigrationRecord.
        SQLAlchemy's relationship cascade prevents direct deletion of the parent
        while children exist, so we use raw SQL to simulate an orphan.
        The API should not crash when querying migrations list.
        """
        with Session(engine) as session:
            m = MigrationRecord(
                source_model="a",
                target_model="b",
                status="complete",
            )
            session.add(m)
            session.commit()
            session.refresh(m)
            mid = m.id

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

        # Use raw SQL to bypass cascade protection and orphan the test case
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text(f"DELETE FROM migrations WHERE id = {mid}"))
            conn.commit()

        # The migrations list API should still work (empty list)
        resp = client.get("/api/v1/migrations")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    # --- 3c. Background task: record deleted between 'running' and 'complete' --- P1
    def test_task_handles_deletion_after_running_status_set(self, engine) -> None:
        """If record is deleted after status is set to 'running' but before
        the task completes, the task should handle None gracefully.
        """
        from rosettastone.core.types import MigrationResult

        with Session(engine) as session:
            record = MigrationRecord(
                source_model="a",
                target_model="b",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        # We'll delete the record in the middle of the task
        def delete_and_return(*args, **kwargs):
            with Session(engine) as session:
                record = session.get(MigrationRecord, mid)
                if record:
                    session.delete(record)
                    session.commit()
            return MigrationResult(
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

        config_dict = {
            "source_model": "a",
            "target_model": "b",
            "data_path": "/tmp/fake.jsonl",
        }

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = delete_and_return

            from rosettastone.server.api.tasks import run_migration_background

            # Should not raise
            run_migration_background(mid, config_dict, engine=engine)

    # --- 3d. Concurrent reads while background task is writing --- P1
    def test_concurrent_read_during_write(self, engine) -> None:
        """Read the migration while the background task writes to it.
        SQLite with StaticPool (in-memory, shared connection) can cause
        issues with truly concurrent access. This test verifies that
        sequential read/write interleaving works, which models the actual
        single-worker executor pattern used in production.
        """
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="a",
                target_model="b",
                status="complete",
                config_json="{}",
                per_type_scores_json="{}",
                warnings_json="[]",
                safety_warnings_json="[]",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        # Interleave reads and writes sequentially (models the single-worker
        # executor reading DB state while the API serves reads)
        for i in range(20):
            # Write
            with Session(engine) as session:
                r = session.get(MigrationRecord, mid)
                if r:
                    r.cost_usd = float(i)
                    session.add(r)
                    session.commit()
            # Read
            with Session(engine) as session:
                r = session.get(MigrationRecord, mid)
                assert r is not None
                assert r.cost_usd == float(i)


# ---------------------------------------------------------------------------
# 4. TEMPLATE RENDERING EDGE CASES
# ---------------------------------------------------------------------------


class TestTemplateRenderingEdgeCases:
    """P1: Verify templates handle unusual data without crashing."""

    # --- 4a. Migration with empty per_type_scores_json ("{}")  --- P0
    def test_migration_detail_empty_per_type_scores(self, client: TestClient, engine) -> None:
        """A migration with no per-type scores should render without error."""
        with Session(engine) as session:
            m = MigrationRecord(
                source_model="a",
                target_model="b",
                status="complete",
                recommendation="GO",
                recommendation_reasoning="All good.",
                config_json="{}",
                per_type_scores_json="{}",
                warnings_json="[]",
                safety_warnings_json="[]",
                confidence_score=0.9,
                baseline_score=0.8,
                improvement=0.1,
            )
            session.add(m)
            session.commit()
            session.refresh(m)
            mid = m.id

        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200

    # --- 4b. Migration with null/None numeric fields --- P1
    def test_migration_detail_null_scores(self, client: TestClient, engine) -> None:
        """A migration where confidence_score, baseline_score, etc. are all None.
        The template uses these via `round((X or 0) * 100)` so None should be safe.
        """
        with Session(engine) as session:
            m = MigrationRecord(
                source_model="a",
                target_model="b",
                status="complete",
                recommendation="CONDITIONAL",
                confidence_score=None,
                baseline_score=None,
                improvement=None,
                config_json="{}",
                per_type_scores_json="{}",
                warnings_json="[]",
                safety_warnings_json="[]",
            )
            session.add(m)
            session.commit()
            session.refresh(m)
            mid = m.id

        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200
        assert "0%" in resp.text  # round((None or 0) * 100) == 0

    # --- 4c. Extremely long model names --- P2
    def test_migration_detail_extremely_long_model_names(self, client: TestClient, engine) -> None:
        """Model names with 1000+ characters should not crash the template."""
        long_name = "x" * 2000
        with Session(engine) as session:
            m = MigrationRecord(
                source_model=long_name,
                target_model=long_name,
                status="complete",
                recommendation="GO",
                recommendation_reasoning="OK",
                config_json="{}",
                per_type_scores_json="{}",
                warnings_json="[]",
                safety_warnings_json="[]",
                confidence_score=0.5,
            )
            session.add(m)
            session.commit()
            session.refresh(m)
            mid = m.id

        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200
        assert long_name in resp.text

    # --- 4d. Special characters in model names (XSS vectors) --- P0 (security)
    def test_migration_detail_xss_in_model_names(self, client: TestClient, engine) -> None:
        """Model names containing HTML/script tags should be auto-escaped by Jinja2.
        IMPORTANT: Jinja2 auto-escapes by default; verify the raw script tag
        does NOT appear unescaped.
        """
        xss_payload = '<script>alert("xss")</script>'
        with Session(engine) as session:
            m = MigrationRecord(
                source_model=xss_payload,
                target_model=xss_payload,
                status="complete",
                recommendation="GO",
                recommendation_reasoning="OK",
                config_json="{}",
                per_type_scores_json="{}",
                warnings_json="[]",
                safety_warnings_json="[]",
                confidence_score=0.5,
            )
            session.add(m)
            session.commit()
            session.refresh(m)
            mid = m.id

        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200
        body = resp.text
        # The raw <script> tag should NOT appear unescaped; Jinja2 auto-escapes it.
        # We check that the literal "<script>alert" does not appear as an HTML tag.
        # Note: the escaped form &lt;script&gt;alert("xss")&lt;/script&gt; IS present,
        # which is the correct safe behavior.
        assert "<script>alert" not in body
        # Verify the escaped form is present (proves the value is rendered but safe)
        assert "&lt;script&gt;" in body or "xss" not in body.split("<script")[0]

    # --- 4e. XSS in recommendation_reasoning (rendered in template) --- P0
    def test_xss_in_recommendation_reasoning(self, client: TestClient, engine) -> None:
        """recommendation_reasoning is rendered in templates; verify XSS is escaped."""
        xss = '<img src=x onerror="alert(1)">'
        with Session(engine) as session:
            m = MigrationRecord(
                source_model="a",
                target_model="b",
                status="complete",
                recommendation="GO",
                recommendation_reasoning=xss,
                config_json="{}",
                per_type_scores_json="{}",
                warnings_json="[]",
                safety_warnings_json="[]",
                confidence_score=0.5,
            )
            session.add(m)
            session.commit()
            session.refresh(m)
            mid = m.id

        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200
        body = resp.text
        assert 'onerror="alert(1)"' not in body
        assert "&lt;img" in body or "onerror" not in body

    # --- 4f. Migration with 0 test cases --- P1
    def test_migration_detail_zero_test_cases(self, client: TestClient, engine) -> None:
        """A completed migration with zero test cases should still render."""
        with Session(engine) as session:
            m = MigrationRecord(
                source_model="a",
                target_model="b",
                status="complete",
                recommendation="GO",
                recommendation_reasoning="OK",
                config_json="{}",
                per_type_scores_json="{}",
                warnings_json="[]",
                safety_warnings_json="[]",
                confidence_score=0.9,
            )
            session.add(m)
            session.commit()
            session.refresh(m)
            mid = m.id

        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200
        # Template should show 0 test cases without error
        assert "0" in resp.text

    # --- 4g. Migration with many test cases (performance) --- P2
    def test_migration_detail_many_test_cases(self, client: TestClient, engine) -> None:
        """A migration with 500 test cases should render without timeout.
        (We use 500 instead of 10k to keep test fast.)
        """
        with Session(engine) as session:
            m = MigrationRecord(
                source_model="a",
                target_model="b",
                status="complete",
                recommendation="GO",
                recommendation_reasoning="OK",
                config_json="{}",
                per_type_scores_json=json.dumps(
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
                        },
                    }
                ),
                warnings_json="[]",
                safety_warnings_json="[]",
                confidence_score=0.9,
            )
            session.add(m)
            session.commit()
            session.refresh(m)
            mid = m.id

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
        # Should render in under 5 seconds
        assert elapsed < 5.0, f"Template took {elapsed:.2f}s -- too slow"

    # --- 4h. HTML entities in regression expected/got fields --- P1
    def test_html_entities_in_regression_fields(self, client: TestClient, engine) -> None:
        """Regression expected/got fields with HTML entities should be escaped."""
        with Session(engine) as session:
            m = MigrationRecord(
                source_model="a",
                target_model="b",
                status="complete",
                recommendation="NO_GO",
                recommendation_reasoning="Bad",
                config_json="{}",
                per_type_scores_json="{}",
                warnings_json="[]",
                safety_warnings_json="[]",
                confidence_score=0.3,
            )
            session.add(m)
            session.commit()
            session.refresh(m)

            tc = TestCaseRecord(
                migration_id=m.id,
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
            session.add(tc)
            session.commit()
            mid = m.id

        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200
        body = resp.text
        # Script tags in response content must be escaped
        assert '<script>alert("expected")</script>' not in body
        assert '<script>alert("got")</script>' not in body


# ---------------------------------------------------------------------------
# 5. BACKGROUND TASK FAILURE MODES
# ---------------------------------------------------------------------------


class TestBackgroundTaskFailureModes:
    """P0/P1: Verify graceful handling of all failure modes in run_migration_background."""

    # --- 5a. MigrationConfig validation fails --- P0
    def test_config_validation_failure_sets_status_failed(self, engine) -> None:
        """If MigrationConfig(**config_dict) raises ValidationError,
        the record should be set to 'failed'.
        """
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="a",
                target_model="b",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        # data_path is required by MigrationConfig and must be a valid Path;
        # omitting it should cause a ValidationError
        config_dict = {
            "source_model": "a",
            "target_model": "b",
            # No data_path -> Pydantic validation error
        }

        from rosettastone.server.api.tasks import run_migration_background

        run_migration_background(mid, config_dict, engine=engine)

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "failed"
            assert "failed" in record.recommendation_reasoning.lower()

    # --- 5b. Migrator.__init__ raises exception --- P1
    def test_migrator_init_raises_sets_status_failed(self, engine) -> None:
        """If Migrator() constructor raises, record should be 'failed'."""
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="a",
                target_model="b",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        config_dict = {
            "source_model": "a",
            "target_model": "b",
            "data_path": "/tmp/fake.jsonl",
        }

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.side_effect = RuntimeError("Init failed!")

            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, config_dict, engine=engine)

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "failed"
            assert "Init failed!" in record.recommendation_reasoning

    # --- 5c. Migrator.run() raises generic exception --- P0
    def test_migrator_run_exception_sets_status_failed(self, engine) -> None:
        """If Migrator.run() raises RuntimeError, record becomes 'failed'."""
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="a",
                target_model="b",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        config_dict = {
            "source_model": "a",
            "target_model": "b",
            "data_path": "/tmp/fake.jsonl",
        }

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = RuntimeError("Network error")

            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, config_dict, engine=engine)

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "failed"
            assert "Network error" in record.recommendation_reasoning

    # --- 5d. MigrationBlockedError sets status to 'blocked' --- P0
    def test_migration_blocked_error_sets_status_blocked(self, engine) -> None:
        """MigrationBlockedError should set status='blocked', not 'failed'."""
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="a",
                target_model="b",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        config_dict = {
            "source_model": "a",
            "target_model": "b",
            "data_path": "/tmp/fake.jsonl",
        }

        from rosettastone.core.migrator import MigrationBlockedError

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = MigrationBlockedError(
                "Context window too small"
            )

            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, config_dict, engine=engine)

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "blocked"

    # --- 5e. Dry run produces dry_run_complete status --- P1
    def test_dry_run_success_sets_dry_run_complete(self, engine) -> None:
        """A successful dry_run migration should set status='dry_run_complete'."""
        from rosettastone.core.types import MigrationResult

        with Session(engine) as session:
            record = MigrationRecord(
                source_model="a",
                target_model="b",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        mock_result = MigrationResult(
            config={},
            optimized_prompt="",
            baseline_results=[],
            validation_results=[],
            confidence_score=0.0,
            baseline_score=0.0,
            improvement=0.0,
            cost_usd=0.0,
            duration_seconds=1.0,
            warnings=["Dry run"],
        )

        config_dict = {
            "source_model": "a",
            "target_model": "b",
            "data_path": "/tmp/fake.jsonl",
            "dry_run": True,
        }

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.return_value = mock_result

            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, config_dict, engine=engine)

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "dry_run_complete"

    # --- 5f. MemoryError during migration run --- P2
    def test_memory_error_behavior(self, engine) -> None:
        """MemoryError inherits from BaseException, not Exception, so the
        `except Exception` clause does NOT catch it. However, the `finally`
        block still runs. Depending on the Python/mock interaction, the
        MemoryError may or may not propagate. This test verifies that:
        - The record transitions to 'running' (set before Migrator.run())
        - The error does not crash silently leaving stale state unrecoverable
        - Startup recovery (_recover_orphaned_migrations) can fix it
        """
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="a",
                target_model="b",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        config_dict = {
            "source_model": "a",
            "target_model": "b",
            "data_path": "/tmp/fake.jsonl",
        }

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = MemoryError("OOM")

            from rosettastone.server.api.tasks import run_migration_background

            # MemoryError may propagate or be swallowed depending on mock behavior.
            # Either way, the record should be in a recoverable state.
            try:
                run_migration_background(mid, config_dict, engine=engine)
            except MemoryError:
                pass  # Expected if it propagates

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            # Status should be 'running' (the except Exception didn't catch MemoryError,
            # so the status was never updated to 'failed') or 'failed' if somehow caught.
            assert record.status in ("running", "failed")

            if record.status == "running":
                # Verify startup recovery fixes it
                from rosettastone.server.app import _recover_orphaned_migrations

                with patch("rosettastone.server.app.get_engine", return_value=engine):
                    _recover_orphaned_migrations()

                session.expire(record)
                record = session.get(MigrationRecord, mid)
                assert record.status == "failed"

    # --- 5g. KeyboardInterrupt during migration run --- P2
    def test_keyboard_interrupt_not_caught(self, engine) -> None:
        """KeyboardInterrupt (BaseException) should propagate, leaving
        the record in 'running' state. Startup recovery handles this.
        """
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="a",
                target_model="b",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        config_dict = {
            "source_model": "a",
            "target_model": "b",
            "data_path": "/tmp/fake.jsonl",
        }

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = KeyboardInterrupt()

            from rosettastone.server.api.tasks import run_migration_background

            with pytest.raises(KeyboardInterrupt):
                run_migration_background(mid, config_dict, engine=engine)

    # --- 5h. Safety warnings as dict and str mixed --- P1
    def test_mixed_safety_warnings_serialized(self, engine) -> None:
        """safety_warnings can contain both dicts and strings;
        the task should handle both formats without error.
        """
        from rosettastone.core.types import MigrationResult

        with Session(engine) as session:
            record = MigrationRecord(
                source_model="a",
                target_model="b",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

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

        config_dict = {
            "source_model": "a",
            "target_model": "b",
            "data_path": "/tmp/fake.jsonl",
        }

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.return_value = mock_result

            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, config_dict, engine=engine)

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "complete"
            # Verify WarningRecords were created for safety warnings
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

    # --- 6a. SQL injection in migration_id path parameter --- P0
    def test_sql_injection_in_migration_id(self, client: TestClient) -> None:
        """migration_id is typed as int in FastAPI; non-numeric values
        should return 422 (validation error).
        """
        resp = client.get("/api/v1/migrations/1;DROP TABLE migrations;--")
        assert resp.status_code in (404, 422)

    # --- 6b. SQL injection via string path segments --- P0
    def test_sql_injection_in_test_case_path(self, client: TestClient) -> None:
        """Test case endpoints also use int IDs; injection should fail."""
        resp = client.get("/api/v1/migrations/1/test-cases/1 OR 1=1")
        assert resp.status_code in (404, 422)

    # --- 6c. XSS in model name form fields (via API) --- P0
    def test_xss_in_model_name_api(self, client: TestClient) -> None:
        """Creating a migration with XSS in model names via the JSON API.
        The API stores it; the UI must escape it (tested in template tests above).
        """
        payload = {
            "source_model": '<script>alert("xss")</script>',
            "target_model": "<img src=x onerror=alert(1)>",
        }
        resp = client.post("/api/v1/migrations", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        # The API stores raw values (JSON responses are not HTML-rendered)
        assert data["source_model"] == '<script>alert("xss")</script>'

    # --- 6d. XSS in form field submission --- P0
    def test_xss_in_form_model_field(self, client: TestClient, engine) -> None:
        """Submit the migration form with XSS payloads in model name fields.
        The redirect page should escape the HTML tags so they are not executable.
        Jinja2 auto-escaping converts '<' to '&lt;' and '>' to '&gt;', which
        means the literal text 'onerror=alert(1)' may appear as visible text
        but is NOT inside an HTML tag and thus NOT exploitable.
        """
        data = b'{"prompt": "hello", "response": "world"}\n'
        resp = _post_migration_form(
            client,
            data=data,
            source_model='<script>alert("src")</script>',
            target_model="<img onerror=alert(1) src=x>",
        )
        assert resp.status_code == 303

        # Follow the redirect and verify the detail page escapes the values
        location = resp.headers["location"]
        detail_resp = client.get(location)
        assert detail_resp.status_code == 200
        body = detail_resp.text
        # The raw <script> tag must NOT appear (must be escaped to &lt;script&gt;)
        assert '<script>alert("src")</script>' not in body
        # The raw <img tag must NOT appear (must be escaped to &lt;img ...)
        assert "<img onerror" not in body

    # --- 6e. Very large offset/limit pagination parameters --- P1
    def test_extreme_pagination_offset(self, client: TestClient) -> None:
        """Extremely large offset should return empty results, not crash."""
        resp = client.get("/api/v1/migrations?offset=999999999&limit=100")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []

    # --- 6f. Negative offset (should be caught by ge=0 validator) --- P0
    def test_negative_offset_rejected(self, client: TestClient) -> None:
        """FastAPI Query(ge=0) should reject negative offsets."""
        resp = client.get("/api/v1/migrations?offset=-1")
        assert resp.status_code == 422

    # --- 6g. Limit > 100 (should be caught by le=100 validator) --- P0
    def test_limit_exceeds_max_rejected(self, client: TestClient) -> None:
        """FastAPI Query(le=100) should reject limits > 100."""
        resp = client.get("/api/v1/migrations?limit=101")
        assert resp.status_code == 422

    # --- 6h. Limit = 0 (should be caught by ge=1 validator) --- P0
    def test_limit_zero_rejected(self, client: TestClient) -> None:
        """FastAPI Query(ge=1) should reject limit=0."""
        resp = client.get("/api/v1/migrations?limit=0")
        assert resp.status_code == 422

    # --- 6i. Non-integer migration_id in API --- P0
    def test_non_integer_migration_id(self, client: TestClient) -> None:
        """Non-integer path parameters should return 422."""
        resp = client.get("/api/v1/migrations/abc")
        assert resp.status_code == 422

    # --- 6j. Oversized JSON body --- P1
    def test_oversized_json_body(self, client: TestClient) -> None:
        """Submit an extremely large JSON body to the create migration API."""
        payload = {
            "source_model": "a" * 100_000,
            "target_model": "b" * 100_000,
        }
        resp = client.post("/api/v1/migrations", json=payload)
        # Should either accept (server stores it) or reject (413)
        assert resp.status_code in (201, 413, 422)

    # --- 6k. Empty body to POST endpoint --- P0
    def test_empty_body_to_create_migration(self, client: TestClient) -> None:
        """POST with empty JSON body should return 422."""
        resp = client.post("/api/v1/migrations", json={})
        assert resp.status_code == 422

    # --- 6l. Missing content-type header on JSON API --- P1
    def test_missing_content_type_on_json_post(self, client: TestClient) -> None:
        """POST without proper content-type should fail gracefully."""
        resp = client.post(
            "/api/v1/migrations",
            content=b"not json",
            headers={"content-type": "text/plain"},
        )
        assert resp.status_code in (415, 422, 500)

    # --- 6m. XSS in registered model_id --- P0
    def test_xss_in_model_registration(self, client: TestClient) -> None:
        """Register a model with XSS payload in model_id.
        The API should store it, but UI must escape it.
        """
        xss = '<script>alert("model")</script>'
        resp = client.post("/api/v1/models", json={"model_id": xss})
        assert resp.status_code == 201

        # Verify the UI escapes it
        ui_resp = client.get("/ui/")
        assert ui_resp.status_code == 200
        assert "<script>alert" not in ui_resp.text

    # --- 6n. Security headers present on all responses --- P0
    def test_security_headers_on_api(self, client: TestClient) -> None:
        """Verify security headers are present on API responses."""
        resp = client.get("/api/v1/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("X-XSS-Protection") == "1; mode=block"

    # --- 6o. Security headers on error pages --- P1
    def test_security_headers_on_error_pages(self, client: TestClient) -> None:
        """Verify security headers on 404 pages."""
        resp = client.get("/ui/nonexistent")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"


# ---------------------------------------------------------------------------
# 7. STATE MACHINE VIOLATIONS
# ---------------------------------------------------------------------------


class TestStateMachineViolations:
    """P1: Verify migration status transitions are safe and consistent."""

    # --- 7a. Can a 'complete' migration be re-triggered? --- P1
    def test_cannot_retrigger_complete_migration_via_api(self, client: TestClient, engine) -> None:
        """There is no 'retrigger' API endpoint. A completed migration stays complete.
        Creating a new migration is the only way to re-run.
        """
        with Session(engine) as session:
            m = MigrationRecord(
                source_model="a",
                target_model="b",
                status="complete",
                recommendation="GO",
                config_json="{}",
                per_type_scores_json="{}",
                warnings_json="[]",
                safety_warnings_json="[]",
            )
            session.add(m)
            session.commit()
            session.refresh(m)
            mid = m.id

        # There's no PUT/PATCH endpoint to change status
        # Try to POST a new body to the migration detail (should 405 or no such route)
        resp = client.post(f"/api/v1/migrations/{mid}", json={"status": "running"})
        assert resp.status_code in (404, 405)

    # --- 7b. Background task on non-existent record --- P0
    def test_background_task_nonexistent_record(self, engine) -> None:
        """run_migration_background with a non-existent migration_id should
        return early without error.
        """
        from rosettastone.server.api.tasks import run_migration_background

        # Should not raise
        run_migration_background(
            migration_id=999999,
            config_dict={
                "source_model": "a",
                "target_model": "b",
                "data_path": "/tmp/fake.jsonl",
            },
            engine=engine,
        )

    # --- 7c. Verify status transitions: pending -> running -> complete --- P0
    def test_status_transition_pending_running_complete(self, engine) -> None:
        """Verify the full lifecycle: pending -> running -> complete."""
        from rosettastone.core.types import MigrationResult

        with Session(engine) as session:
            record = MigrationRecord(
                source_model="a",
                target_model="b",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        statuses_observed = []

        def capture_and_return(*args, **kwargs):
            with Session(engine) as s:
                r = s.get(MigrationRecord, mid)
                statuses_observed.append(r.status)
            return MigrationResult(
                config={},
                optimized_prompt="opt",
                baseline_results=[],
                validation_results=[],
                confidence_score=0.9,
                baseline_score=0.8,
                improvement=0.1,
                cost_usd=0.5,
                duration_seconds=10.0,
                warnings=[],
                recommendation="GO",
                recommendation_reasoning="Good",
                per_type_scores={},
            )

        config_dict = {
            "source_model": "a",
            "target_model": "b",
            "data_path": "/tmp/fake.jsonl",
        }

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = capture_and_return

            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, config_dict, engine=engine)

        assert "running" in statuses_observed

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "complete"

    # --- 7d. Verify status transitions: pending -> running -> failed --- P0
    def test_status_transition_pending_running_failed(self, engine) -> None:
        """Verify: pending -> running -> failed."""
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="a",
                target_model="b",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        statuses_observed = []

        def capture_and_raise(*args, **kwargs):
            with Session(engine) as s:
                r = s.get(MigrationRecord, mid)
                statuses_observed.append(r.status)
            raise RuntimeError("Intentional failure")

        config_dict = {
            "source_model": "a",
            "target_model": "b",
            "data_path": "/tmp/fake.jsonl",
        }

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = capture_and_raise

            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, config_dict, engine=engine)

        assert "running" in statuses_observed

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "failed"

    # --- 7e. Verify status transitions: pending -> running -> blocked --- P0
    def test_status_transition_pending_running_blocked(self, engine) -> None:
        """Verify: pending -> running -> blocked."""
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="a",
                target_model="b",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        from rosettastone.core.migrator import MigrationBlockedError

        config_dict = {
            "source_model": "a",
            "target_model": "b",
            "data_path": "/tmp/fake.jsonl",
        }

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = MigrationBlockedError("Blocked!")

            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, config_dict, engine=engine)

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "blocked"

    # --- 7f. Invalid status value directly written to DB --- P2
    def test_invalid_status_renders_without_crash(self, client: TestClient, engine) -> None:
        """If a record has an unexpected status value, the detail page
        should still render (potentially with default/fallback styling).
        """
        with Session(engine) as session:
            m = MigrationRecord(
                source_model="a",
                target_model="b",
                status="INVALID_STATUS_VALUE",
                config_json="{}",
                per_type_scores_json="{}",
                warnings_json="[]",
                safety_warnings_json="[]",
            )
            session.add(m)
            session.commit()
            session.refresh(m)
            mid = m.id

        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200  # should not crash

    # --- 7g. Verify alerts don't fire for pending/running migrations --- P1
    def test_alerts_skip_pending_and_running(self, client: TestClient, engine) -> None:
        """Alert generation should only trigger for complete/failed migrations."""
        with Session(engine) as session:
            session.add(
                MigrationRecord(
                    source_model="a",
                    target_model="b",
                    status="pending",
                )
            )
            session.add(
                MigrationRecord(
                    source_model="a",
                    target_model="b",
                    status="running",
                )
            )
            session.commit()

        resp = client.post("/api/v1/alerts/generate")
        assert resp.json()["generated"] == 0


# ---------------------------------------------------------------------------
# 8. ERROR HANDLING & EDGE CASES (Miscellaneous)
# ---------------------------------------------------------------------------


class TestMiscEdgeCases:
    """P1/P2: Additional edge cases and error handling tests."""

    # --- 8a. 404 on API returns JSON --- P0
    def test_api_404_returns_json(self, client: TestClient) -> None:
        resp = client.get("/api/v1/migrations/999999")
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data

    # --- 8b. 404 on UI returns HTML --- P0
    def test_ui_404_returns_html(self, client: TestClient) -> None:
        resp = client.get("/ui/nonexistent-page-xyz")
        assert resp.status_code == 404
        assert "text/html" in resp.headers.get("content-type", "")

    # --- 8c. Health endpoint always available --- P0
    def test_health_endpoint(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    # --- 8d. Diff fragment for non-existent migration falls back to dummy --- P2
    def test_diff_fragment_fallback_to_dummy(self, client: TestClient) -> None:
        resp = client.get("/ui/fragments/diff/999/999")
        assert resp.status_code == 200
        assert "BERTScore" in resp.text

    # --- 8e. Cost page with zero-cost migrations (no division by zero) --- P1
    def test_cost_page_zero_cost_no_crash(self, client: TestClient, engine) -> None:
        with Session(engine) as session:
            session.add(
                MigrationRecord(
                    source_model="a",
                    target_model="b",
                    status="complete",
                    cost_usd=0.0,
                )
            )
            session.commit()

        resp = client.get("/ui/costs")
        assert resp.status_code == 200
        assert "$0.00" in resp.text

    # --- 8f. Delete a model that was used in migrations --- P2
    def test_delete_model_used_in_migrations(self, client: TestClient, engine) -> None:
        """Deleting a registered model should not cascade-delete migrations."""
        reg = client.post("/api/v1/models", json={"model_id": "openai/gpt-4o"})
        db_id = reg.json()["db_id"]

        # Create a migration using this model
        with Session(engine) as session:
            session.add(
                MigrationRecord(
                    source_model="openai/gpt-4o",
                    target_model="anthropic/claude-sonnet-4",
                    status="complete",
                )
            )
            session.commit()

        # Delete the registered model
        resp = client.delete(f"/api/v1/models/{db_id}")
        assert resp.status_code == 200

        # The migration should still exist
        migrations = client.get("/api/v1/migrations").json()
        assert migrations["total"] == 1

    # --- 8g. JSON API detail for migration with malformed per_type_scores_json --- P1
    def test_api_detail_malformed_json_fields(self, client: TestClient, engine) -> None:
        """If per_type_scores_json contains invalid JSON, the API should
        handle it gracefully. Note: json.loads will raise ValueError.
        """
        with Session(engine) as session:
            m = MigrationRecord(
                source_model="a",
                target_model="b",
                status="complete",
                config_json="{}",
                per_type_scores_json="NOT VALID JSON",
                warnings_json="[]",
                safety_warnings_json="[]",
            )
            session.add(m)
            session.commit()
            session.refresh(m)
            mid = m.id

        resp = client.get(f"/api/v1/migrations/{mid}")
        # json.loads on malformed data will raise, producing a 500
        assert resp.status_code == 500

    # --- 8h. Executive report for non-existent migration --- P0
    def test_executive_report_404(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations/999999/executive")
        assert resp.status_code == 404

    # --- 8i. Unicode model names --- P2
    def test_unicode_model_names(self, client: TestClient, engine) -> None:
        """Model names with unicode characters should work."""
        with Session(engine) as session:
            m = MigrationRecord(
                source_model="unicorn/model-v1",
                target_model="model-v2",
                status="complete",
                recommendation="GO",
                recommendation_reasoning="OK",
                config_json="{}",
                per_type_scores_json="{}",
                warnings_json="[]",
                safety_warnings_json="[]",
                confidence_score=0.9,
            )
            session.add(m)
            session.commit()
            session.refresh(m)
            mid = m.id

        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200

    # --- 8j. Concurrent alert generation is idempotent --- P1
    def test_concurrent_alert_generation_idempotent(self, client: TestClient, engine) -> None:
        """Calling alert generation multiple times rapidly should not create dupes."""
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

        # Generate alerts 3 times rapidly
        for _ in range(3):
            client.post("/api/v1/alerts/generate")

        alerts = client.get("/api/v1/alerts").json()
        assert len(alerts) == 1  # idempotent

    # --- 8k. Pagination edge: offset == total --- P2
    def test_pagination_offset_equals_total(self, client: TestClient, engine) -> None:
        """When offset equals total count, should return empty items."""
        with Session(engine) as session:
            for i in range(3):
                session.add(
                    MigrationRecord(
                        source_model=f"s{i}",
                        target_model=f"t{i}",
                        status="complete",
                    )
                )
            session.commit()

        resp = client.get("/api/v1/migrations?offset=3&limit=20")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert data["items"] == []

    # --- 8l. Report endpoint for non-complete migration --- P1
    def test_report_for_pending_migration(self, client: TestClient, engine) -> None:
        """Requesting a report for a 'pending' migration should still work
        (it returns a report with placeholder/zero values).
        """
        with Session(engine) as session:
            m = MigrationRecord(
                source_model="a",
                target_model="b",
                status="pending",
                config_json="{}",
                per_type_scores_json="{}",
                warnings_json="[]",
                safety_warnings_json="[]",
            )
            session.add(m)
            session.commit()
            session.refresh(m)
            mid = m.id

        resp = client.get(f"/api/v1/migrations/{mid}/report/markdown")
        # Should return 200 with a minimal report (not crash)
        assert resp.status_code == 200

    # --- 8m. Rapid health check requests (basic load) --- P2
    def test_rapid_health_checks(self, client: TestClient) -> None:
        """Send 100 rapid health check requests; all should succeed."""
        for _ in range(100):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200
