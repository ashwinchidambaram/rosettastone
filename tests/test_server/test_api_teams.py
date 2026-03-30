"""Tests for the team management API (/api/v1/teams)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from rosettastone.server.app import create_app
from rosettastone.server.database import get_session

_JWT_SECRET = "test-teams-secret"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_admin_token(user_id: int = 1) -> str:
    from rosettastone.server.auth_utils import create_jwt

    return create_jwt(user_id=user_id, role="admin", secret=_JWT_SECRET)


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


def _create_user_in_db(engine, username: str, role: str = "viewer") -> int:
    """Insert a user directly into the DB and return its id."""
    from rosettastone.server.auth_utils import hash_password
    from rosettastone.server.models import User

    with Session(engine) as session:
        user = User(
            username=username,
            hashed_password=hash_password("password"),
            role=role,
            is_active=True,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user.id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client_and_engine(monkeypatch):
    """Yield (TestClient, engine) with multi-user enabled and admin JWT."""
    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", _JWT_SECRET)
    monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)

    app, engine = _make_app_and_engine()
    token = _make_admin_token()
    with TestClient(app, headers={"Authorization": f"Bearer {token}"}) as c:
        yield c, engine


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------


def _create_team(client: TestClient, name: str) -> dict:
    return client.post("/api/v1/teams", json={"name": name})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_create_team_201(client_and_engine):
    c, _ = client_and_engine
    resp = _create_team(c, "Engineering")
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Engineering"
    assert "id" in data


def test_create_duplicate_team_409(client_and_engine):
    c, _ = client_and_engine
    _create_team(c, "Engineering")
    resp = _create_team(c, "Engineering")
    assert resp.status_code == 409


def test_list_teams(client_and_engine):
    c, _ = client_and_engine
    assert c.get("/api/v1/teams").json() == []

    _create_team(c, "Alpha")
    _create_team(c, "Beta")

    resp = c.get("/api/v1/teams")
    assert resp.status_code == 200
    names = [t["name"] for t in resp.json()]
    assert "Alpha" in names
    assert "Beta" in names


def test_get_team_by_id(client_and_engine):
    c, _ = client_and_engine
    team_id = _create_team(c, "Gamma").json()["id"]
    resp = c.get(f"/api/v1/teams/{team_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Gamma"


def test_get_team_not_found_404(client_and_engine):
    c, _ = client_and_engine
    resp = c.get("/api/v1/teams/9999")
    assert resp.status_code == 404


def test_delete_team_204(client_and_engine):
    c, _ = client_and_engine
    team_id = _create_team(c, "ToDelete").json()["id"]
    resp = c.delete(f"/api/v1/teams/{team_id}")
    assert resp.status_code == 204

    resp = c.get(f"/api/v1/teams/{team_id}")
    assert resp.status_code == 404


def test_add_member_to_team(client_and_engine):
    c, engine = client_and_engine
    team_id = _create_team(c, "Squad").json()["id"]
    user_id = _create_user_in_db(engine, "alice")

    resp = c.post(f"/api/v1/teams/{team_id}/members", json={"user_id": user_id})
    assert resp.status_code == 201
    data = resp.json()
    assert data["user_id"] == user_id
    assert data["team_id"] == team_id


def test_add_duplicate_member_409(client_and_engine):
    c, engine = client_and_engine
    team_id = _create_team(c, "Squad2").json()["id"]
    user_id = _create_user_in_db(engine, "bob")

    c.post(f"/api/v1/teams/{team_id}/members", json={"user_id": user_id})
    resp = c.post(f"/api/v1/teams/{team_id}/members", json={"user_id": user_id})
    assert resp.status_code == 409


def test_remove_member_204(client_and_engine):
    c, engine = client_and_engine
    team_id = _create_team(c, "Squad3").json()["id"]
    user_id = _create_user_in_db(engine, "charlie")

    c.post(f"/api/v1/teams/{team_id}/members", json={"user_id": user_id})
    resp = c.delete(f"/api/v1/teams/{team_id}/members/{user_id}")
    assert resp.status_code == 204


def test_list_team_members(client_and_engine):
    c, engine = client_and_engine
    team_id = _create_team(c, "Squad4").json()["id"]
    user_id = _create_user_in_db(engine, "diana")

    c.post(f"/api/v1/teams/{team_id}/members", json={"user_id": user_id})
    resp = c.get(f"/api/v1/teams/{team_id}/members")
    assert resp.status_code == 200
    members = resp.json()
    assert len(members) == 1
    assert members[0]["user_id"] == user_id


def test_endpoints_return_404_without_multi_user(monkeypatch):
    """Team endpoints return 404 when ROSETTASTONE_MULTI_USER is not set."""
    monkeypatch.delenv("ROSETTASTONE_MULTI_USER", raising=False)
    monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)

    app, _ = _make_app_and_engine()
    with TestClient(app) as c:
        resp = c.get("/api/v1/teams")
        assert resp.status_code == 404
