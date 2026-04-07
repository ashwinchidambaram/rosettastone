"""Tests for Task 1.6: multi-user data isolation.

Verifies that resources are scoped to their owner in multi-user mode,
admins can see all, and single-user mode has no filtering.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from rosettastone.server.app import create_app
from rosettastone.server.auth_utils import create_jwt
from rosettastone.server.database import get_session
from rosettastone.server.models import MigrationRecord, PipelineRecord

_TEST_JWT_SECRET = "test-secret-long-enough-for-hmac-sha256-validation"


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
    return eng


@pytest.fixture
def multi_user_client(engine, monkeypatch):
    """TestClient with multi-user mode enabled, using real JWT auth."""
    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", _TEST_JWT_SECRET)
    app = create_app()

    def override_session():
        with Session(engine) as sess:
            yield sess

    app.dependency_overrides[get_session] = override_session
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def single_user_client(engine):
    """Standard single-user TestClient (no multi-user filtering)."""
    app = create_app()

    def override_session():
        with Session(engine) as sess:
            yield sess

    app.dependency_overrides[get_session] = override_session
    return TestClient(app)


def _token(user_id: int, role: str = "editor") -> dict[str, str]:
    """Return Authorization headers for the given user."""
    tok = create_jwt(user_id, role, _TEST_JWT_SECRET)
    return {"Authorization": f"Bearer {tok}"}


def _insert_migration(engine, owner_id: int | None) -> int:
    with Session(engine) as sess:
        m = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="pending",
            owner_id=owner_id,
        )
        sess.add(m)
        sess.commit()
        sess.refresh(m)
        return m.id  # type: ignore[return-value]


def _insert_pipeline(engine, owner_id: int | None) -> int:
    with Session(engine) as sess:
        p = PipelineRecord(
            name="test-pipeline",
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="pending",
            config_yaml="pipeline:\n  name: test-pipeline\n",
            owner_id=owner_id,
        )
        sess.add(p)
        sess.commit()
        sess.refresh(p)
        return p.id  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Migration isolation
# ---------------------------------------------------------------------------


class TestMigrationIsolation:
    def test_user_cannot_see_other_users_migration_in_list(self, multi_user_client, engine) -> None:
        """User 2 must not see migrations owned by user 1."""
        _insert_migration(engine, owner_id=1)

        resp = multi_user_client.get("/api/v1/migrations", headers=_token(2))
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_user_can_see_own_migration_in_list(self, multi_user_client, engine) -> None:
        """User 1 must see their own migration."""
        _insert_migration(engine, owner_id=1)

        resp = multi_user_client.get("/api/v1/migrations", headers=_token(1))
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_cross_user_get_returns_403(self, multi_user_client, engine) -> None:
        """User 2 accessing user 1's migration by ID must get 403."""
        mid = _insert_migration(engine, owner_id=1)

        resp = multi_user_client.get(f"/api/v1/migrations/{mid}", headers=_token(2))
        assert resp.status_code == 403

    def test_owner_get_returns_200(self, multi_user_client, engine) -> None:
        """Owner can access their own migration by ID."""
        mid = _insert_migration(engine, owner_id=1)

        resp = multi_user_client.get(f"/api/v1/migrations/{mid}", headers=_token(1))
        assert resp.status_code == 200

    def test_admin_sees_all_migrations(self, multi_user_client, engine) -> None:
        """Admin can see all migrations regardless of owner."""
        _insert_migration(engine, owner_id=1)
        _insert_migration(engine, owner_id=2)

        resp = multi_user_client.get("/api/v1/migrations", headers=_token(99, "admin"))
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_admin_can_get_any_migration(self, multi_user_client, engine) -> None:
        """Admin can retrieve any user's migration by ID."""
        mid = _insert_migration(engine, owner_id=1)

        resp = multi_user_client.get(f"/api/v1/migrations/{mid}", headers=_token(99, "admin"))
        assert resp.status_code == 200

    def test_single_user_mode_returns_all_regardless(self, single_user_client, engine) -> None:
        """In single-user mode, all migrations are visible with no filtering."""
        _insert_migration(engine, owner_id=1)
        _insert_migration(engine, owner_id=2)

        resp = single_user_client.get("/api/v1/migrations")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2


# ---------------------------------------------------------------------------
# Pipeline isolation
# ---------------------------------------------------------------------------


class TestPipelineIsolation:
    def test_user_cannot_see_other_users_pipeline(self, multi_user_client, engine) -> None:
        """User 2 must not see pipelines owned by user 1."""
        _insert_pipeline(engine, owner_id=1)

        resp = multi_user_client.get("/api/v1/pipelines", headers=_token(2))
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_cross_user_pipeline_status_returns_403(self, multi_user_client, engine) -> None:
        """User 2 accessing user 1's pipeline status must get 403."""
        pid = _insert_pipeline(engine, owner_id=1)

        resp = multi_user_client.get(f"/api/v1/pipelines/{pid}/status", headers=_token(2))
        assert resp.status_code == 403

    def test_admin_sees_all_pipelines(self, multi_user_client, engine) -> None:
        """Admin can see all pipelines."""
        _insert_pipeline(engine, owner_id=1)
        _insert_pipeline(engine, owner_id=2)

        resp = multi_user_client.get("/api/v1/pipelines", headers=_token(99, "admin"))
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_single_user_mode_returns_all_pipelines(self, single_user_client, engine) -> None:
        """In single-user mode, all pipelines visible with no filtering."""
        _insert_pipeline(engine, owner_id=1)
        _insert_pipeline(engine, owner_id=2)

        resp = single_user_client.get("/api/v1/pipelines")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2


# ---------------------------------------------------------------------------
# Comparison / distribution isolation
# ---------------------------------------------------------------------------


class TestComparisonIsolation:
    def test_user_cannot_access_other_users_distributions(self, multi_user_client, engine) -> None:
        """IDOR: user 2 cannot access user 1's migration distributions."""
        mid = _insert_migration(engine, owner_id=1)
        resp = multi_user_client.get(f"/api/v1/migrations/{mid}/distributions", headers=_token(2))
        assert resp.status_code == 403

    def test_owner_can_access_own_distributions(self, multi_user_client, engine) -> None:
        mid = _insert_migration(engine, owner_id=1)
        resp = multi_user_client.get(f"/api/v1/migrations/{mid}/distributions", headers=_token(1))
        # 200 or 404 (no test cases) is fine, NOT 403
        assert resp.status_code != 403

    def test_admin_can_access_any_distributions(self, multi_user_client, engine) -> None:
        mid = _insert_migration(engine, owner_id=1)
        resp = multi_user_client.get(
            f"/api/v1/migrations/{mid}/distributions", headers=_token(99, role="admin")
        )
        assert resp.status_code != 403


# ---------------------------------------------------------------------------
# Report isolation
# ---------------------------------------------------------------------------


class TestReportIsolation:
    def test_user_cannot_access_other_users_report(self, multi_user_client, engine) -> None:
        """IDOR: user 2 cannot access user 1's migration report."""
        mid = _insert_migration(engine, owner_id=1)
        resp = multi_user_client.get(f"/api/v1/migrations/{mid}/report/markdown", headers=_token(2))
        assert resp.status_code == 403

    def test_owner_can_access_own_report(self, multi_user_client, engine) -> None:
        mid = _insert_migration(engine, owner_id=1)
        resp = multi_user_client.get(f"/api/v1/migrations/{mid}/report/markdown", headers=_token(1))
        assert resp.status_code != 403

    def test_admin_can_access_any_report(self, multi_user_client, engine) -> None:
        mid = _insert_migration(engine, owner_id=1)
        resp = multi_user_client.get(
            f"/api/v1/migrations/{mid}/report/markdown", headers=_token(99, role="admin")
        )
        assert resp.status_code != 403
