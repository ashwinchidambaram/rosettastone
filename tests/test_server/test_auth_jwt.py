"""Tests for JWT-based multi-user auth endpoints: login, register, refresh, me."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rosettastone.server.app import create_app
from rosettastone.server.database import init_db, reset_engine


@pytest.fixture(autouse=True)
def clear_rate_limit():
    """Clear the in-module rate-limit dict before each test to avoid cross-test pollution."""
    from rosettastone.server.api import auth as auth_module

    auth_module._failed_attempts.clear()
    yield
    auth_module._failed_attempts.clear()


def _make_temp_db_client(monkeypatch, multi_user: bool = True) -> TestClient:
    """Build a TestClient backed by an isolated temp-file SQLite DB.

    The auth endpoints call get_session() directly (not via DI), so we
    point ROSETTASTONE_DB_PATH at a fresh temp file and reset the engine
    singleton so the module picks it up.
    """
    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "test.db")
    monkeypatch.setenv("ROSETTASTONE_DB_PATH", db_path)
    monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)

    if multi_user:
        monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
        monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", "test-secret-key-1234567890-long-enough-hmac")
    else:
        monkeypatch.delenv("ROSETTASTONE_MULTI_USER", raising=False)

    # Reset and re-initialise the engine singleton so get_session() uses the temp DB
    reset_engine()
    init_db()

    app = create_app()
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def multi_user_client(monkeypatch):
    """TestClient with temp DB and ROSETTASTONE_MULTI_USER=true."""
    with _make_temp_db_client(monkeypatch, multi_user=True) as c:
        yield c


@pytest.fixture
def no_multi_user_client(monkeypatch):
    """TestClient with temp DB, multi-user disabled."""
    with _make_temp_db_client(monkeypatch, multi_user=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _register(client: TestClient, username: str, password: str = "password123", **kwargs):
    return client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": password, **kwargs},
    )


def _login(client: TestClient, username: str, password: str = "password123"):
    return client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_register_first_user_becomes_admin(multi_user_client):
    """First registered user automatically receives the admin role."""
    resp = _register(multi_user_client, "alice")
    assert resp.status_code == 201
    data = resp.json()
    assert data["role"] == "admin"
    assert data["username"] == "alice"


def test_register_second_user_gets_viewer(multi_user_client):
    """Second registered user receives the default viewer role."""
    _register(multi_user_client, "alice")
    resp = _register(multi_user_client, "bob")
    assert resp.status_code == 201
    assert resp.json()["role"] == "viewer"


def test_register_duplicate_username_409(multi_user_client):
    """Registering with an already-taken username returns 409."""
    _register(multi_user_client, "alice")
    resp = _register(multi_user_client, "alice")
    assert resp.status_code == 409


def test_login_returns_jwt(multi_user_client):
    """Successful login returns 200 with access_token, user_id, and role."""
    _register(multi_user_client, "alice")
    resp = _login(multi_user_client, "alice")
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["role"] == "admin"
    assert isinstance(data["user_id"], int)


def test_login_wrong_password_401(multi_user_client):
    """Login with wrong password returns 401."""
    _register(multi_user_client, "alice")
    resp = _login(multi_user_client, "alice", password="wrong")
    assert resp.status_code == 401


def test_me_returns_user_info(multi_user_client):
    """GET /api/v1/auth/me with a valid Bearer token returns user info."""
    _register(multi_user_client, "alice")
    token = _login(multi_user_client, "alice").json()["access_token"]

    resp = multi_user_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["username"] == "alice"
    assert data["role"] == "admin"
    assert data["is_active"] is True


def test_refresh_returns_new_token(multi_user_client):
    """POST /api/v1/auth/refresh with a valid token returns a new access_token."""
    _register(multi_user_client, "alice")
    token = _login(multi_user_client, "alice").json()["access_token"]

    resp = multi_user_client.post(
        "/api/v1/auth/refresh",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert len(data["access_token"]) > 10


def test_endpoints_return_404_without_multi_user(no_multi_user_client):
    """Auth endpoints return 404 when ROSETTASTONE_MULTI_USER is not set."""
    resp = no_multi_user_client.post(
        "/api/v1/auth/register",
        json={"username": "alice", "password": "password123"},
    )
    assert resp.status_code == 404

    resp = no_multi_user_client.post(
        "/api/v1/auth/login",
        json={"username": "alice", "password": "password123"},
    )
    assert resp.status_code == 404
