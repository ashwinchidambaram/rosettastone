"""Tests for auth middleware, login/logout flows, and CSRF protection."""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from rosettastone.server.api.auth import _create_session_token, _verify_key
from rosettastone.server.app import create_app
from rosettastone.server.database import get_session

_TEST_KEY = "test-key-123"
_WRONG_KEY = "wrong-key-999"


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_client():
    """Return a factory that builds a TestClient backed by a fresh in-memory DB.

    Using a factory so individual tests can control whether the env var is set
    *before* the app is created (middleware reads env at request time, not init
    time, so the standard client fixture also works — but a dedicated fixture
    keeps things tidy).
    """

    def _make():
        engine = create_engine(
            "sqlite://",
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(engine)

        app = create_app()

        def override_session():
            with Session(engine) as s:
                yield s

        app.dependency_overrides[get_session] = override_session
        # follow_redirects=False so we can inspect 302 responses directly
        return TestClient(app, follow_redirects=False)

    return _make


@pytest.fixture()
def authed_client(fresh_client, monkeypatch):
    """TestClient with auth enabled (ROSETTASTONE_API_KEY set)."""
    monkeypatch.setenv("ROSETTASTONE_API_KEY", _TEST_KEY)
    return fresh_client()


def _session_cookie(api_key: str) -> str:
    """Compute the expected session cookie value for *api_key*."""
    return _create_session_token(api_key)


def _login(client: TestClient, key: str = _TEST_KEY) -> str:
    """POST to /ui/login and return the session token value."""
    resp = client.post("/ui/login", data={"api_key": key})
    assert resp.status_code == 302, f"login failed: {resp.status_code}"
    cookie = resp.cookies.get("rosettastone_session")
    assert cookie, "session cookie not set after login"
    return cookie


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


class TestAuthDisabled:
    """When ROSETTASTONE_API_KEY is not set, all endpoints are open."""

    def test_auth_disabled_all_endpoints_accessible(self, fresh_client, monkeypatch):
        monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)
        client = fresh_client()

        r_health = client.get("/api/v1/health")
        assert r_health.status_code == 200

        r_ui = client.get("/ui/", follow_redirects=True)
        assert r_ui.status_code == 200

    def test_login_redirects_when_auth_disabled(self, fresh_client, monkeypatch):
        monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)
        client = fresh_client()

        resp = client.get("/ui/login")
        # login page itself is in _PUBLIC_PREFIXES so auth middleware passes it
        # but the handler redirects to /ui/ when auth is disabled
        assert resp.status_code == 302
        assert resp.headers["location"].endswith("/ui/")

    def test_csrf_disabled_when_no_auth(self, fresh_client, monkeypatch):
        """Without auth, CSRF middleware is a no-op — form POSTs pass through."""
        monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)
        client = fresh_client()

        # POST without any CSRF token; should not get a 403 from CSRF middleware
        resp = client.post(
            "/ui/migrations/new",
            data={
                "source_model": "openai/gpt-4o",
                "target_model": "anthropic/claude-sonnet-4",
                "data_path": "/tmp/fake.jsonl",
            },
        )
        assert resp.status_code != 403


class TestAuthEnabled:
    """When ROSETTASTONE_API_KEY is set, API and UI routes are protected."""

    def test_auth_enabled_api_401_without_header(self, authed_client):
        resp = authed_client.get("/api/v1/migrations")
        assert resp.status_code == 401

    def test_auth_enabled_api_200_with_bearer(self, authed_client):
        resp = authed_client.get(
            "/api/v1/migrations",
            headers={"Authorization": f"Bearer {_TEST_KEY}"},
        )
        # Should succeed (200) — not a 401
        assert resp.status_code != 401

    def test_auth_enabled_api_401_wrong_key(self, authed_client):
        resp = authed_client.get(
            "/api/v1/migrations",
            headers={"Authorization": f"Bearer {_WRONG_KEY}"},
        )
        assert resp.status_code == 401

    def test_auth_enabled_ui_redirects_to_login(self, authed_client):
        resp = authed_client.get("/ui/migrations")
        assert resp.status_code == 302
        assert "/ui/login" in resp.headers["location"]

    def test_health_always_public(self, authed_client):
        resp = authed_client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_static_always_public(self, authed_client):
        resp = authed_client.get("/static/js/app.js")
        # Not 401 — static files are exempt from auth
        assert resp.status_code != 401

    def test_login_page_renders(self, authed_client):
        resp = authed_client.get("/ui/login")
        assert resp.status_code == 200

    def test_login_correct_key_sets_cookie(self, authed_client):
        resp = authed_client.post("/ui/login", data={"api_key": _TEST_KEY})
        assert resp.status_code == 302
        assert "rosettastone_session" in resp.cookies

    def test_login_wrong_key_shows_error(self, authed_client):
        resp = authed_client.post(
            "/ui/login",
            data={"api_key": _WRONG_KEY},
            follow_redirects=True,
        )
        assert resp.status_code == 401
        assert "Invalid" in resp.text

    def test_logout_clears_cookie(self, authed_client):
        # First log in so we have a valid session
        _login(authed_client)

        resp = authed_client.post("/ui/logout")
        assert resp.status_code == 302
        # The session cookie should be deleted (empty value or absent)
        cookie_val = resp.cookies.get("rosettastone_session", "")
        assert cookie_val == ""

    def test_session_cookie_grants_ui_access(self, authed_client):
        # Log in to obtain a session cookie
        session_token = _login(authed_client)

        # Subsequent request carrying the session cookie should succeed
        resp = authed_client.get(
            "/ui/",
            cookies={"rosettastone_session": session_token},
            follow_redirects=True,
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# CSRF tests
# ---------------------------------------------------------------------------


class TestCSRF:
    """Double-submit cookie CSRF protection."""

    def test_csrf_get_sets_cookie(self, authed_client):
        """A GET to a protected UI route should set the CSRF cookie."""
        session_token = _login(authed_client)

        resp = authed_client.get(
            "/ui/migrations/new",
            cookies={"rosettastone_session": session_token},
        )
        assert "rosettastone_csrf" in resp.cookies

    def test_csrf_post_without_token_403(self, authed_client):
        """POST without a CSRF token must be rejected with 403."""
        session_token = _login(authed_client)

        resp = authed_client.post(
            "/ui/migrations/new",
            data={
                "source_model": "openai/gpt-4o",
                "target_model": "anthropic/claude-sonnet-4",
                "data_path": "/tmp/fake.jsonl",
            },
            cookies={"rosettastone_session": session_token},
        )
        assert resp.status_code == 403

    def test_csrf_post_with_valid_token_passes(self, authed_client):
        """POST with matching CSRF cookie + form field must not be rejected."""
        session_token = _login(authed_client)

        # Obtain the CSRF token from a GET
        get_resp = authed_client.get(
            "/ui/migrations/new",
            cookies={"rosettastone_session": session_token},
        )
        csrf_token = get_resp.cookies.get("rosettastone_csrf")
        assert csrf_token, "CSRF cookie not set on GET"

        resp = authed_client.post(
            "/ui/migrations/new",
            data={
                "source_model": "openai/gpt-4o",
                "target_model": "anthropic/claude-sonnet-4",
                "data_path": "/tmp/fake.jsonl",
                "_csrf_token": csrf_token,
            },
            cookies={
                "rosettastone_session": session_token,
                "rosettastone_csrf": csrf_token,
            },
        )
        # CSRF validation passed — not 403
        assert resp.status_code != 403

    def test_csrf_header_works(self, authed_client):
        """X-CSRF-Token header is an acceptable alternative to the form field."""
        session_token = _login(authed_client)

        get_resp = authed_client.get(
            "/ui/migrations/new",
            cookies={"rosettastone_session": session_token},
        )
        csrf_token = get_resp.cookies.get("rosettastone_csrf")
        assert csrf_token

        resp = authed_client.post(
            "/ui/migrations/new",
            data={
                "source_model": "openai/gpt-4o",
                "target_model": "anthropic/claude-sonnet-4",
                "data_path": "/tmp/fake.jsonl",
            },
            headers={"x-csrf-token": csrf_token},
            cookies={
                "rosettastone_session": session_token,
                "rosettastone_csrf": csrf_token,
            },
        )
        assert resp.status_code != 403

    def test_csrf_mismatched_token_403(self, authed_client):
        """POST with wrong CSRF form field must be rejected with 403."""
        session_token = _login(authed_client)

        get_resp = authed_client.get(
            "/ui/migrations/new",
            cookies={"rosettastone_session": session_token},
        )
        csrf_token = get_resp.cookies.get("rosettastone_csrf")
        assert csrf_token

        resp = authed_client.post(
            "/ui/migrations/new",
            data={
                "source_model": "openai/gpt-4o",
                "target_model": "anthropic/claude-sonnet-4",
                "data_path": "/tmp/fake.jsonl",
                "_csrf_token": "totally-wrong-token",
            },
            cookies={
                "rosettastone_session": session_token,
                "rosettastone_csrf": csrf_token,
            },
        )
        assert resp.status_code == 403

    def test_csrf_skipped_for_api_routes(self, authed_client):
        """API routes using Bearer auth are exempt from CSRF checks."""
        # No CSRF cookie — should still succeed (or fail for non-CSRF reasons)
        resp = authed_client.get(
            "/api/v1/migrations",
            headers={"Authorization": f"Bearer {_TEST_KEY}"},
        )
        assert resp.status_code != 403


# ---------------------------------------------------------------------------
# Unit tests for _csrf_enabled()
# ---------------------------------------------------------------------------


class TestCsrfEnabled:
    """Unit tests for the _csrf_enabled() helper in csrf.py."""

    def test_csrf_enabled_with_multi_user_flag(self, monkeypatch):
        """ROSETTASTONE_MULTI_USER=true enables CSRF even without an API key."""
        from rosettastone.server.csrf import _csrf_enabled

        monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)
        monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
        assert _csrf_enabled() is True

    def test_csrf_disabled_without_either_flag(self, monkeypatch):
        """Neither flag set — CSRF must be disabled."""
        from rosettastone.server.csrf import _csrf_enabled

        monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)
        monkeypatch.delenv("ROSETTASTONE_MULTI_USER", raising=False)
        assert _csrf_enabled() is False

    def test_csrf_enabled_with_api_key_only(self, monkeypatch):
        """Backward compat: API key alone still enables CSRF."""
        from rosettastone.server.csrf import _csrf_enabled

        monkeypatch.setenv("ROSETTASTONE_API_KEY", _TEST_KEY)
        monkeypatch.delenv("ROSETTASTONE_MULTI_USER", raising=False)
        assert _csrf_enabled() is True

    def test_csrf_enabled_with_both_flags(self, monkeypatch):
        """Both flags set — CSRF must be enabled."""
        from rosettastone.server.csrf import _csrf_enabled

        monkeypatch.setenv("ROSETTASTONE_API_KEY", _TEST_KEY)
        monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
        assert _csrf_enabled() is True


# ---------------------------------------------------------------------------
# Unit tests for auth helper functions
# ---------------------------------------------------------------------------


class TestAuthHelpers:
    def test_verify_key_correct(self):
        assert _verify_key("my-secret", "my-secret") is True

    def test_verify_key_wrong(self):
        assert _verify_key("not-the-key", "my-secret") is False

    def test_verify_key_empty_provided(self):
        assert _verify_key("", "my-secret") is False

    def test_create_session_token_deterministic(self):
        """Same key must always produce the same token."""
        token_a = _create_session_token("some-key")
        token_b = _create_session_token("some-key")
        assert token_a == token_b

    def test_create_session_token_is_sha256_hex(self):
        """Token should be a 64-character hex string (SHA-256)."""
        token = _create_session_token("any-key")
        assert len(token) == 64
        int(token, 16)  # raises ValueError if not valid hex

    def test_create_session_token_matches_manual_hash(self):
        """Token must match manually computed SHA-256 of 'rosettastone:<key>'."""
        key = "verify-me"
        expected = hashlib.sha256(("rosettastone:" + key).encode()).hexdigest()
        assert _create_session_token(key) == expected

    def test_create_session_token_differs(self):
        """Different keys must produce different tokens."""
        token_a = _create_session_token("key-alpha")
        token_b = _create_session_token("key-beta")
        assert token_a != token_b
