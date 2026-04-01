"""Tests for the user management API (/api/v1/users)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from rosettastone.server.app import create_app
from rosettastone.server.database import get_session

_JWT_SECRET = "test-users-secret"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_admin_token(user_id: int = 1) -> str:
    """Create a valid JWT for an admin user."""
    from rosettastone.server.auth_utils import create_jwt

    return create_jwt(user_id=user_id, role="admin", secret=_JWT_SECRET)


def _make_app_and_engine(multi_user: bool):
    """Build an app + engine pair with optional multi-user mode."""
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    """TestClient with ROSETTASTONE_MULTI_USER=true and an admin JWT header."""
    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", _JWT_SECRET)
    monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)

    app, _ = _make_app_and_engine(multi_user=True)
    token = _make_admin_token()
    with TestClient(app, headers={"Authorization": f"Bearer {token}"}) as c:
        yield c


# ---------------------------------------------------------------------------
# CRUD helper
# ---------------------------------------------------------------------------


def _create_user(client: TestClient, username: str, role: str = "viewer", **kwargs) -> dict:
    resp = client.post(
        "/api/v1/users",
        json={"username": username, "password": "secret", "role": role, **kwargs},
    )
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_users_returns_empty_initially(client):
    """GET /api/v1/users returns an empty list when no users exist."""
    resp = client.get("/api/v1/users")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_user_returns_201(client):
    """POST /api/v1/users creates a user and returns 201."""
    resp = _create_user(client, "alice")
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "alice"
    assert "id" in data
    assert data["is_active"] is True


def test_create_first_user_is_admin(client):
    """The very first user created via POST /api/v1/users receives the admin role."""
    resp = _create_user(client, "first", role="viewer")
    assert resp.status_code == 201
    assert resp.json()["role"] == "admin"


def test_create_second_user_keeps_requested_role(client):
    """Subsequent users keep the role specified in the request body."""
    _create_user(client, "first")
    resp = _create_user(client, "second", role="editor")
    assert resp.status_code == 201
    assert resp.json()["role"] == "editor"


def test_create_duplicate_username_409(client):
    """Creating a user with an existing username returns 409."""
    _create_user(client, "alice")
    resp = _create_user(client, "alice")
    assert resp.status_code == 409


def test_get_user_by_id(client):
    """GET /api/v1/users/{id} returns the user detail."""
    user_id = _create_user(client, "alice").json()["id"]
    resp = client.get(f"/api/v1/users/{user_id}")
    assert resp.status_code == 200
    assert resp.json()["username"] == "alice"


def test_get_user_not_found_404(client):
    """GET /api/v1/users/{id} returns 404 when user does not exist."""
    resp = client.get("/api/v1/users/9999")
    assert resp.status_code == 404


def test_update_user_role(client):
    """PUT /api/v1/users/{id} can update a user's role."""
    _create_user(client, "alice")
    bob_id = _create_user(client, "bob", role="viewer").json()["id"]

    resp = client.put(f"/api/v1/users/{bob_id}", json={"role": "editor"})
    assert resp.status_code == 200
    assert resp.json()["role"] == "editor"


def test_update_user_not_found_404(client):
    """PUT /api/v1/users/{id} returns 404 when user does not exist."""
    resp = client.put("/api/v1/users/9999", json={"email": "x@x.com"})
    assert resp.status_code == 404


def test_delete_user_204(client):
    """DELETE /api/v1/users/{id} removes the user and returns 204."""
    _create_user(client, "alice")
    bob_id = _create_user(client, "bob", role="viewer").json()["id"]

    resp = client.delete(f"/api/v1/users/{bob_id}")
    assert resp.status_code == 204

    resp = client.get(f"/api/v1/users/{bob_id}")
    assert resp.status_code == 404


def test_list_users_returns_404_without_multi_user(monkeypatch):
    """GET /api/v1/users returns 404 when ROSETTASTONE_MULTI_USER is not set."""
    monkeypatch.delenv("ROSETTASTONE_MULTI_USER", raising=False)
    monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)

    app, _ = _make_app_and_engine(multi_user=False)
    with TestClient(app) as c:
        resp = c.get("/api/v1/users")
        assert resp.status_code == 404


def test_non_admin_cannot_update_other_user(monkeypatch):
    """PUT /api/v1/users/{id} returns 403 when a non-admin tries to update a different user."""
    from rosettastone.server.auth_utils import create_jwt

    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", _JWT_SECRET)
    monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)

    app, _ = _make_app_and_engine(multi_user=True)

    # Create alice (admin, first user) and bob (viewer) via admin client
    admin_token = _make_admin_token(user_id=999)
    with TestClient(app, headers={"Authorization": f"Bearer {admin_token}"}) as admin_client:
        alice_resp = _create_user(admin_client, "alice")
        assert alice_resp.status_code == 201
        alice_id = alice_resp.json()["id"]

        bob_resp = _create_user(admin_client, "bob", role="viewer")
        assert bob_resp.status_code == 201
        bob_id = bob_resp.json()["id"]

    # Bob (non-admin, user_id=bob_id) tries to update alice's profile
    bob_token = create_jwt(user_id=bob_id, role="viewer", secret=_JWT_SECRET)
    with TestClient(app, headers={"Authorization": f"Bearer {bob_token}"}) as bob_client:
        resp = bob_client.put(f"/api/v1/users/{alice_id}", json={"email": "hacked@example.com"})
        assert resp.status_code == 403
        assert "own profile" in resp.json()["detail"]


def test_non_admin_can_update_own_profile(monkeypatch):
    """PUT /api/v1/users/{id} succeeds when a non-admin updates their own profile."""
    from rosettastone.server.auth_utils import create_jwt

    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", _JWT_SECRET)
    monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)

    app, _ = _make_app_and_engine(multi_user=True)

    # Create alice (admin, first user) and bob (viewer) via admin client
    admin_token = _make_admin_token(user_id=999)
    with TestClient(app, headers={"Authorization": f"Bearer {admin_token}"}) as admin_client:
        _create_user(admin_client, "alice")

        bob_resp = _create_user(admin_client, "bob", role="viewer")
        assert bob_resp.status_code == 201
        bob_id = bob_resp.json()["id"]

    # Bob updates his own email — should succeed
    bob_token = create_jwt(user_id=bob_id, role="viewer", secret=_JWT_SECRET)
    with TestClient(app, headers={"Authorization": f"Bearer {bob_token}"}) as bob_client:
        resp = bob_client.put(f"/api/v1/users/{bob_id}", json={"email": "bob@example.com"})
        assert resp.status_code == 200
        assert resp.json()["email"] == "bob@example.com"
