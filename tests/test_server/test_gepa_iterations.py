"""Tests for GEPA iteration history persistence and API."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
sqlmodel = pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine, select  # noqa: E402

from rosettastone.server.app import create_app  # noqa: E402
from rosettastone.server.database import get_session  # noqa: E402
from rosettastone.server.models import GEPAIterationRecord, MigrationRecord  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    eng = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(eng)
    yield eng
    SQLModel.metadata.drop_all(eng)


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


@pytest.fixture
def sample_migration(session: Session) -> MigrationRecord:
    m = MigrationRecord(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        status="complete",
        config_json="{}",
        per_type_scores_json="{}",
        warnings_json="[]",
        safety_warnings_json="[]",
    )
    session.add(m)
    session.commit()
    session.refresh(m)
    return m


@pytest.fixture
def client(engine) -> TestClient:
    app = create_app()

    def override_session():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = override_session
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test 1: GEPAIterationRecord can be written and queried
# ---------------------------------------------------------------------------


class TestGEPAIterationRecordPersistence:
    def test_write_and_read(self, session: Session, sample_migration: MigrationRecord):
        """A GEPAIterationRecord written to DB can be queried back."""
        rec = GEPAIterationRecord(
            migration_id=sample_migration.id,
            iteration=1,
            total_iterations=5,
            mean_score=0.72,
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)

        assert rec.id is not None
        assert rec.migration_id == sample_migration.id
        assert rec.iteration == 1
        assert rec.total_iterations == 5
        assert rec.mean_score == pytest.approx(0.72)
        assert rec.recorded_at is not None

    def test_multiple_iterations_ordered(self, session: Session, sample_migration: MigrationRecord):
        """Multiple iterations are stored independently and queryable by order."""
        for i in range(1, 4):
            session.add(
                GEPAIterationRecord(
                    migration_id=sample_migration.id,
                    iteration=i,
                    total_iterations=3,
                    mean_score=0.6 + i * 0.05,
                )
            )
        session.commit()

        results = session.exec(
            select(GEPAIterationRecord)
            .where(GEPAIterationRecord.migration_id == sample_migration.id)
            .order_by(GEPAIterationRecord.iteration)
        ).all()

        assert len(results) == 3
        assert [r.iteration for r in results] == [1, 2, 3]
        assert results[0].mean_score == pytest.approx(0.65)
        assert results[2].mean_score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Test 2: _make_gepa_callback writes a record to the DB
# ---------------------------------------------------------------------------


class TestGEPACallbackPersistence:
    def test_callback_writes_record(self, engine, sample_migration: MigrationRecord):
        """Invoking the gepa callback should persist a GEPAIterationRecord."""
        from unittest.mock import patch

        from rosettastone.server.api.tasks import _make_gepa_callback

        migration_id = sample_migration.id

        with patch("rosettastone.server.progress.emit_progress", return_value=None):
            cb = _make_gepa_callback(migration_id, engine=engine)
            cb(iteration=2, total_iterations=10, mean_score=0.81)

        with Session(engine) as s:
            records = s.exec(
                select(GEPAIterationRecord).where(GEPAIterationRecord.migration_id == migration_id)
            ).all()

        assert len(records) == 1
        assert records[0].iteration == 2
        assert records[0].total_iterations == 10
        assert records[0].mean_score == pytest.approx(0.81)

    def test_callback_swallows_db_error(self, sample_migration: MigrationRecord):
        """DB error in callback should not propagate (SSE-safe)."""
        from unittest.mock import MagicMock, patch

        from rosettastone.server.api.tasks import _make_gepa_callback

        # A bad engine that raises on use
        bad_session = MagicMock()
        bad_session.__enter__ = MagicMock(side_effect=RuntimeError("DB down"))
        bad_session.__exit__ = MagicMock(return_value=False)

        bad_engine = MagicMock()

        with patch("rosettastone.server.api.tasks.Session", return_value=bad_session):
            with patch("rosettastone.server.progress.emit_progress", return_value=None):
                cb = _make_gepa_callback(sample_migration.id, engine=bad_engine)
                # Should not raise even with a broken DB
                cb(iteration=1, total_iterations=5, mean_score=0.5)


# ---------------------------------------------------------------------------
# Test 3: GET /api/v1/migrations/{id}/optimizer-history — sorted list
# ---------------------------------------------------------------------------


class TestOptimizerHistoryEndpoint:
    def test_returns_sorted_iterations(
        self, client: TestClient, engine, sample_migration: MigrationRecord
    ):
        """Endpoint returns iterations sorted by iteration asc."""
        with Session(engine) as s:
            # Insert out-of-order to verify sorting
            for i in [3, 1, 2]:
                s.add(
                    GEPAIterationRecord(
                        migration_id=sample_migration.id,
                        iteration=i,
                        total_iterations=3,
                        mean_score=0.5 + i * 0.1,
                    )
                )
            s.commit()

        resp = client.get(f"/api/v1/migrations/{sample_migration.id}/optimizer-history")
        assert resp.status_code == 200

        data = resp.json()
        assert len(data) == 3
        assert [d["iteration"] for d in data] == [1, 2, 3]
        assert data[0]["mean_score"] == pytest.approx(0.6)
        assert data[2]["mean_score"] == pytest.approx(0.8)

    def test_returns_empty_list_when_no_records(
        self, client: TestClient, sample_migration: MigrationRecord
    ):
        """Endpoint returns [] when no iteration records exist for a migration."""
        resp = client.get(f"/api/v1/migrations/{sample_migration.id}/optimizer-history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_empty_list_for_nonexistent_migration(self, client: TestClient):
        """Endpoint returns [] for a migration ID that has no records."""
        resp = client.get("/api/v1/migrations/99999/optimizer-history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_response_schema_fields(
        self, client: TestClient, engine, sample_migration: MigrationRecord
    ):
        """Each item in the response has the expected fields."""
        with Session(engine) as s:
            s.add(
                GEPAIterationRecord(
                    migration_id=sample_migration.id,
                    iteration=1,
                    total_iterations=5,
                    mean_score=0.77,
                )
            )
            s.commit()

        resp = client.get(f"/api/v1/migrations/{sample_migration.id}/optimizer-history")
        assert resp.status_code == 200
        item = resp.json()[0]
        assert set(item.keys()) == {"iteration", "total_iterations", "mean_score", "recorded_at"}
        assert item["iteration"] == 1
        assert item["total_iterations"] == 5
        assert item["mean_score"] == pytest.approx(0.77)
