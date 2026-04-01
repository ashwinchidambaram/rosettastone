"""Tests for the approval workflow API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from rosettastone.server.app import create_app
from rosettastone.server.database import get_session
from rosettastone.server.models import MigrationRecord

_JWT_SECRET = "test-approvals-secret"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_token(user_id: int = 1, role: str = "admin") -> str:
    from rosettastone.server.auth_utils import create_jwt

    return create_jwt(user_id=user_id, role=role, secret=_JWT_SECRET)


def _make_app_and_engine():
    engine = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    app = create_app()

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    return app, engine


def _insert_migration(engine) -> int:
    with Session(engine) as session:
        migration = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
        )
        session.add(migration)
        session.commit()
        session.refresh(migration)
        return migration.id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    """TestClient with ROSETTASTONE_MULTI_USER=true and admin JWT."""
    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", _JWT_SECRET)
    monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)

    app, engine = _make_app_and_engine()
    token = _make_token(role="admin")
    with TestClient(app, headers={"Authorization": f"Bearer {token}"}) as c:
        yield c, engine


@pytest.fixture
def migration_id(client):
    c, engine = client
    return _insert_migration(engine)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------


def _create_workflow(client: TestClient, migration_id: int, required: int = 1) -> dict:
    return client.post(
        f"/api/v1/migrations/{migration_id}/approval-workflow",
        json={"required_approvals": required},
    )


def _approve(client: TestClient, migration_id: int, comment: str = "LGTM") -> dict:
    return client.post(
        f"/api/v1/migrations/{migration_id}/approve",
        json={"decision": "approve", "comment": comment},
    )


def _reject(client: TestClient, migration_id: int, comment: str = "Needs work") -> dict:
    return client.post(
        f"/api/v1/migrations/{migration_id}/reject",
        json={"decision": "reject", "comment": comment},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_create_workflow_201(client, migration_id):
    """POST /api/v1/migrations/{id}/approval-workflow creates workflow and returns 201."""
    c, _ = client
    resp = _create_workflow(c, migration_id)
    assert resp.status_code == 201
    data = resp.json()
    assert data["migration_id"] == migration_id
    assert data["status"] == "pending"
    assert data["required_approvals"] == 1
    assert data["current_approvals"] == 0


def test_create_workflow_migration_not_found_404(client):
    """Creating a workflow for a non-existent migration returns 404."""
    c, _ = client
    resp = _create_workflow(c, migration_id=9999)
    assert resp.status_code == 404


def test_create_workflow_duplicate_409(client, migration_id):
    """Creating a second workflow for the same migration returns 409."""
    c, _ = client
    _create_workflow(c, migration_id)
    resp = _create_workflow(c, migration_id)
    assert resp.status_code == 409


def test_get_approval_status(client, migration_id):
    """GET /api/v1/migrations/{id}/approval-status returns workflow status."""
    c, _ = client
    _create_workflow(c, migration_id)
    resp = c.get(f"/api/v1/migrations/{migration_id}/approval-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending"
    assert data["current_approvals"] == 0


def test_get_approval_status_no_workflow_404(client, migration_id):
    """GET /approval-status returns 404 when no workflow exists for migration."""
    c, _ = client
    resp = c.get(f"/api/v1/migrations/{migration_id}/approval-status")
    assert resp.status_code == 404


def test_approve_workflow(client, migration_id):
    """POST /approve increments current_approvals."""
    c, _ = client
    _create_workflow(c, migration_id, required=2)
    resp = _approve(c, migration_id)
    assert resp.status_code == 200
    data = resp.json()
    assert data["current_approvals"] == 1
    assert data["status"] == "pending"  # threshold not yet met


def test_auto_approve_when_threshold_met(client, migration_id):
    """Workflow transitions to 'approved' when required_approvals is reached."""
    c, _ = client
    _create_workflow(c, migration_id, required=1)
    resp = _approve(c, migration_id)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "approved"
    assert data["current_approvals"] == 1


def test_reject_resets_workflow(client, migration_id):
    """POST /reject resets the workflow back to pending with 0 approvals."""
    c, _ = client
    _create_workflow(c, migration_id, required=2)
    _approve(c, migration_id, comment="First approval")

    resp = _reject(c, migration_id, comment="Found a bug")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending"
    assert data["current_approvals"] == 0


def test_approve_non_pending_workflow_400(client, migration_id):
    """Approving an already-approved workflow returns 400."""
    c, _ = client
    _create_workflow(c, migration_id, required=1)
    _approve(c, migration_id)  # now approved

    resp = _approve(c, migration_id)
    assert resp.status_code == 400


def test_workflow_not_found_404(client, migration_id):
    """Approving a migration with no workflow returns 404."""
    c, _ = client
    resp = _approve(c, migration_id)
    assert resp.status_code == 404


def test_duplicate_approval_rejected(client, migration_id):
    """Submitting a second approval from the same user for the same workflow returns 409."""
    c, _ = client
    _create_workflow(c, migration_id, required=2)
    resp1 = _approve(c, migration_id, comment="First attempt")
    assert resp1.status_code == 200

    resp2 = _approve(c, migration_id, comment="Duplicate attempt")
    assert resp2.status_code == 409


def test_different_users_can_both_approve(monkeypatch):
    """Two different users can each submit an approval for the same workflow."""
    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", _JWT_SECRET)
    monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)

    app, engine = _make_app_and_engine()
    migration_id = _insert_migration(engine)

    token_user1 = _make_token(user_id=1, role="admin")
    token_user2 = _make_token(user_id=2, role="admin")

    with TestClient(app, headers={"Authorization": f"Bearer {token_user1}"}) as client1:
        # Create the workflow requiring 2 approvals
        resp = _create_workflow(client1, migration_id, required=2)
        assert resp.status_code == 201

        # User 1 approves
        resp1 = _approve(client1, migration_id, comment="User 1 LGTM")
        assert resp1.status_code == 200
        assert resp1.json()["current_approvals"] == 1

    with TestClient(app, headers={"Authorization": f"Bearer {token_user2}"}) as client2:
        # User 2 also approves — should succeed and satisfy the threshold
        resp2 = _approve(client2, migration_id, comment="User 2 LGTM")
        assert resp2.status_code == 200
        assert resp2.json()["current_approvals"] == 2
        assert resp2.json()["status"] == "approved"


def test_create_workflow_without_multi_user_404(monkeypatch):
    """Creating a workflow returns 404 when ROSETTASTONE_MULTI_USER is not set."""
    monkeypatch.delenv("ROSETTASTONE_MULTI_USER", raising=False)
    monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)

    engine = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    app = create_app()

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    mid = _insert_migration(engine)
    with TestClient(app) as c:
        resp = c.post(
            f"/api/v1/migrations/{mid}/approval-workflow",
            json={"required_approvals": 1},
        )
        assert resp.status_code == 404
