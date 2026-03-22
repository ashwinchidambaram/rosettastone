"""Tests for P4 migration trigger: form endpoints, background task runner."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
sqlmodel = pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine, select  # noqa: E402

from rosettastone.server.app import create_app  # noqa: E402
from rosettastone.server.database import get_session  # noqa: E402
from rosettastone.server.models import MigrationRecord, TestCaseRecord  # noqa: E402


@pytest.fixture
def engine():
    """In-memory SQLite engine for tests."""
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
    """Test client with in-memory DB and mocked executor."""
    app = create_app()

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session

    # Mock the executor to prevent actual background task running
    mock_executor = MagicMock()
    app.state.executor = mock_executor

    return TestClient(app)


# ---------------------------------------------------------------------------
# Layer 1: Endpoint tests
# ---------------------------------------------------------------------------


class TestNewMigrationForm:
    def test_form_renders(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations/new")
        assert resp.status_code == 200
        body = resp.text
        assert "New Migration" in body
        assert 'name="source_model"' in body
        assert 'name="target_model"' in body
        assert 'name="data_file"' in body
        assert 'enctype="multipart/form-data"' in body

    def test_form_prepopulates_source(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations/new?source=openai/gpt-4o")
        assert resp.status_code == 200
        assert "openai/gpt-4o" in resp.text

    def test_form_has_gepa_select(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations/new")
        assert 'name="gepa_auto"' in resp.text
        assert "light" in resp.text
        assert "medium" in resp.text
        assert "heavy" in resp.text

    def test_form_has_dry_run_toggle(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations/new")
        assert 'name="dry_run"' in resp.text

    def test_form_has_store_content_toggle(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations/new")
        assert 'name="store_prompt_content"' in resp.text

    def test_form_has_advanced_section(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations/new")
        body = resp.text
        assert 'name="reflection_model"' in body
        assert 'name="judge_model"' in body
        assert "Advanced settings" in body


class TestFormSubmission:
    def test_submit_creates_record_and_redirects(self, client: TestClient, engine) -> None:
        data = b'{"prompt": "hello", "response": "world"}\n'
        files = {"data_file": ("test.jsonl", io.BytesIO(data), "application/jsonl")}
        form_data = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
            "gepa_auto": "light",
        }

        resp = client.post(
            "/ui/migrations/new",
            data=form_data,
            files=files,
            follow_redirects=False,
        )

        # Should redirect (303 See Other)
        assert resp.status_code == 303
        assert "/ui/migrations/" in resp.headers["location"]

        # Verify record was created in DB
        with Session(engine) as session:
            records = list(session.exec(select(MigrationRecord)).all())
            assert len(records) == 1
            assert records[0].source_model == "openai/gpt-4o"
            assert records[0].target_model == "anthropic/claude-sonnet-4"
            assert records[0].status == "pending"

        # Verify the executor was called
        assert client.app.state.executor.submit.called

    def test_submit_without_file_returns_422(self, client: TestClient) -> None:
        form_data = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
        }
        resp = client.post("/ui/migrations/new", data=form_data)
        assert resp.status_code == 422

    def test_submit_with_empty_filename_returns_422(self, client: TestClient) -> None:
        files = {"data_file": ("", io.BytesIO(b""), "application/jsonl")}
        form_data = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
        }
        resp = client.post("/ui/migrations/new", data=form_data, files=files)
        assert resp.status_code == 422
        assert "JSONL" in resp.text or "file" in resp.text.lower()

    def test_submit_with_dry_run(self, client: TestClient, engine) -> None:
        data = b'{"prompt": "hello", "response": "world"}\n'
        files = {"data_file": ("test.jsonl", io.BytesIO(data), "application/jsonl")}
        form_data = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
            "dry_run": "true",
        }

        resp = client.post(
            "/ui/migrations/new", data=form_data, files=files, follow_redirects=False,
        )
        assert resp.status_code == 303

        # Verify the executor was called with dry_run in config
        call_args = client.app.state.executor.submit.call_args
        config_dict = call_args[0][2]  # Third positional arg
        assert config_dict["dry_run"] is True


# ---------------------------------------------------------------------------
# Layer 2: Background task function tests
# ---------------------------------------------------------------------------


class TestRunMigrationBackground:
    def test_success_updates_status_to_complete(self, engine) -> None:
        """Mocked Migrator.run() succeeds -> record becomes 'complete'."""
        from rosettastone.core.types import EvalResult, MigrationResult, PromptPair

        # Insert a pending record
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        # Build a mock MigrationResult
        mock_result = MigrationResult(
            config={"source_model": "openai/gpt-4o", "target_model": "anthropic/claude-sonnet-4"},
            optimized_prompt="Optimized: {{prompt}}",
            baseline_results=[],
            validation_results=[
                EvalResult(
                    prompt_pair=PromptPair(
                        prompt="hello",
                        response="world",
                        source_model="openai/gpt-4o",
                        metadata={"output_type": "short_text"},
                    ),
                    new_response="world!",
                    scores={"bertscore": 0.95, "embedding": 0.92},
                    composite_score=0.93,
                    is_win=True,
                    details={"note": "good"},
                ),
            ],
            confidence_score=0.92,
            baseline_score=0.85,
            improvement=0.07,
            cost_usd=1.23,
            duration_seconds=42.0,
            warnings=["Low sample count"],
            safety_warnings=[],
            recommendation="GO",
            recommendation_reasoning="All types pass.",
            per_type_scores={"short_text": {"win_rate": 1.0, "sample_count": 1}},
        )

        config_dict = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
            "data_path": "/tmp/fake.jsonl",
        }

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.return_value = mock_result

            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, config_dict, engine=engine)

        # Verify final state
        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "complete"
            assert record.confidence_score == 0.92
            assert record.recommendation == "GO"
            assert record.optimized_prompt == "Optimized: {{prompt}}"

            # Verify TestCaseRecord was created
            tcs = list(
                session.exec(
                    select(TestCaseRecord).where(TestCaseRecord.migration_id == mid)
                ).all()
            )
            assert len(tcs) == 1
            assert tcs[0].phase == "validation"
            assert tcs[0].is_win is True
            assert tcs[0].composite_score == 0.93

    def test_failure_updates_status_to_failed(self, engine) -> None:
        """Migrator.run() raises exception -> record becomes 'failed'."""
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        config_dict = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
            "data_path": "/tmp/fake.jsonl",
        }

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = RuntimeError("boom")

            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, config_dict, engine=engine)

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "failed"
            assert "boom" in record.recommendation_reasoning

    def test_blocked_updates_status_to_blocked(self, engine) -> None:
        """MigrationBlockedError -> record becomes 'blocked'."""
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        config_dict = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
            "data_path": "/tmp/fake.jsonl",
        }

        from rosettastone.core.migrator import MigrationBlockedError

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = MigrationBlockedError("context too small")

            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, config_dict, engine=engine)

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "blocked"

    def test_dry_run_updates_status(self, engine) -> None:
        """Dry run -> record becomes 'dry_run_complete'."""
        from rosettastone.core.types import MigrationResult

        with Session(engine) as session:
            record = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
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
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
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

    def test_running_status_set_before_migrator(self, engine) -> None:
        """Verify the status transitions through 'running' before completion."""
        from rosettastone.core.types import MigrationResult

        with Session(engine) as session:
            record = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        statuses_seen = []

        def capture_status(*args, **kwargs):
            with Session(engine) as s:
                r = s.get(MigrationRecord, mid)
                statuses_seen.append(r.status)
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
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
            "data_path": "/tmp/fake.jsonl",
        }

        with patch("rosettastone.core.migrator.Migrator") as mock_migrator:
            mock_migrator.return_value.run.side_effect = capture_status

            from rosettastone.server.api.tasks import run_migration_background

            run_migration_background(mid, config_dict, engine=engine)

        assert "running" in statuses_seen


# ---------------------------------------------------------------------------
# Startup recovery
# ---------------------------------------------------------------------------


class TestStartupRecovery:
    def test_orphaned_running_migrations_are_failed(self, engine) -> None:
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                status="running",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        from rosettastone.server.app import _recover_orphaned_migrations

        with patch("rosettastone.server.app.get_engine", return_value=engine):
            _recover_orphaned_migrations()

        with Session(engine) as session:
            record = session.get(MigrationRecord, mid)
            assert record.status == "failed"
            assert "Server restarted" in record.recommendation_reasoning


# ---------------------------------------------------------------------------
# Migration detail page states
# ---------------------------------------------------------------------------


class TestMigrationDetailStates:
    def test_pending_state_shows_spinner(self, client: TestClient, engine) -> None:
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                status="pending",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200
        body = resp.text
        assert "Queued for execution" in body
        assert "hx-trigger" in body

    def test_running_state_shows_progress(self, client: TestClient, engine) -> None:
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                status="running",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200
        body = resp.text
        assert "in progress" in body
        assert "every 5s" in body

    def test_failed_state_shows_error(self, client: TestClient, engine) -> None:
        with Session(engine) as session:
            record = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                status="failed",
                recommendation_reasoning="Connection timeout",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            mid = record.id

        resp = client.get(f"/ui/migrations/{mid}")
        assert resp.status_code == 200
        body = resp.text
        assert "Migration Failed" in body
        assert "Connection timeout" in body
