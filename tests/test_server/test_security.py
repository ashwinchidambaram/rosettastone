"""Tests for Task 1.3: JWT secret enforcement, security headers, CORS, cookie flags."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rosettastone.server.app import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(monkeypatch, env: dict[str, str]) -> TestClient:
    """Create a TestClient with the given environment variables set."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# JWT secret enforcement
# ---------------------------------------------------------------------------


class TestJWTSecretEnforcement:
    def test_startup_raises_with_default_jwt_secret_in_multi_user_mode(self, monkeypatch) -> None:
        """Server must refuse to start (RuntimeError) when multi-user mode is active
        and JWT_SECRET is the insecure default value."""
        monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
        monkeypatch.delenv("ROSETTASTONE_JWT_SECRET", raising=False)

        from rosettastone.server.app import _check_jwt_secret

        with pytest.raises(RuntimeError, match="insecure default"):
            _check_jwt_secret()

    def test_startup_ok_with_strong_jwt_secret_in_multi_user_mode(self, monkeypatch) -> None:
        """No error when a strong (>= 32 byte) secret is set."""
        monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
        monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", "a" * 64)

        from rosettastone.server.app import _check_jwt_secret

        _check_jwt_secret()  # should not raise

    def test_startup_ok_single_user_mode_default_secret(self, monkeypatch) -> None:
        """Default secret is allowed in single-user mode (no ROSETTASTONE_MULTI_USER)."""
        monkeypatch.delenv("ROSETTASTONE_MULTI_USER", raising=False)
        monkeypatch.delenv("ROSETTASTONE_JWT_SECRET", raising=False)

        from rosettastone.server.app import _check_jwt_secret

        _check_jwt_secret()  # should not raise


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


class TestSecurityHeaders:
    def test_csp_contains_frame_ancestors_none(self, monkeypatch) -> None:
        """CSP header must include frame-ancestors 'none'."""
        client = _make_client(monkeypatch, {})
        resp = client.get("/api/v1/health")
        csp = resp.headers.get("content-security-policy", "")
        assert "frame-ancestors 'none'" in csp

    def test_csp_contains_object_src_none(self, monkeypatch) -> None:
        """CSP must block object sources."""
        client = _make_client(monkeypatch, {})
        resp = client.get("/api/v1/health")
        csp = resp.headers.get("content-security-policy", "")
        assert "object-src 'none'" in csp

    def test_csp_contains_form_action_self(self, monkeypatch) -> None:
        """CSP must restrict form submissions to same origin."""
        client = _make_client(monkeypatch, {})
        resp = client.get("/api/v1/health")
        csp = resp.headers.get("content-security-policy", "")
        assert "form-action 'self'" in csp

    def test_permissions_policy_header_present(self, monkeypatch) -> None:
        """Permissions-Policy header must be present on all responses."""
        client = _make_client(monkeypatch, {})
        resp = client.get("/api/v1/health")
        pp = resp.headers.get("permissions-policy", "")
        assert pp != "", "Permissions-Policy header must be set"
        assert "camera=()" in pp
        assert "microphone=()" in pp
        assert "geolocation=()" in pp


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCORSBehavior:
    def test_cors_header_absent_when_not_configured(self, monkeypatch) -> None:
        """ACAO header must NOT appear when ROSETTASTONE_CORS_ORIGINS is not set."""
        monkeypatch.delenv("ROSETTASTONE_CORS_ORIGINS", raising=False)
        client = _make_client(monkeypatch, {})
        resp = client.get(
            "/api/v1/health",
            headers={"Origin": "https://evil.com"},
        )
        assert "access-control-allow-origin" not in resp.headers

    def test_cors_header_present_for_allowed_origin(self, monkeypatch) -> None:
        """ACAO header must echo the allowed origin when configured."""
        client = _make_client(monkeypatch, {"ROSETTASTONE_CORS_ORIGINS": "https://app.example.com"})
        resp = client.get(
            "/api/v1/health",
            headers={"Origin": "https://app.example.com"},
        )
        assert resp.headers.get("access-control-allow-origin") == "https://app.example.com"

    def test_cors_header_absent_for_unlisted_origin(self, monkeypatch) -> None:
        """ACAO header must NOT appear for an origin not in the allowlist."""
        client = _make_client(monkeypatch, {"ROSETTASTONE_CORS_ORIGINS": "https://app.example.com"})
        resp = client.get(
            "/api/v1/health",
            headers={"Origin": "https://evil.com"},
        )
        assert resp.headers.get("access-control-allow-origin") != "https://evil.com"


# ---------------------------------------------------------------------------
# Secure cookie flag
# ---------------------------------------------------------------------------


class TestSecureCookieFlag:
    def test_cookie_not_secure_without_https_flag(self, monkeypatch) -> None:
        """Session cookie must NOT have the Secure flag unless ROSETTASTONE_BEHIND_HTTPS is set."""
        monkeypatch.setenv("ROSETTASTONE_API_KEY", "testkey")
        monkeypatch.delenv("ROSETTASTONE_BEHIND_HTTPS", raising=False)
        client = _make_client(monkeypatch, {})
        resp = client.post(
            "/ui/login",
            data={"api_key": "testkey"},
            follow_redirects=False,
        )
        # 302 redirect on success
        assert resp.status_code == 302
        set_cookie = resp.headers.get("set-cookie", "")
        assert "secure" not in set_cookie.lower()

    def test_cookie_secure_with_https_flag(self, monkeypatch) -> None:
        """Session cookie must have Secure flag when ROSETTASTONE_BEHIND_HTTPS=true."""
        monkeypatch.setenv("ROSETTASTONE_API_KEY", "testkey")
        monkeypatch.setenv("ROSETTASTONE_BEHIND_HTTPS", "true")
        client = _make_client(monkeypatch, {})
        resp = client.post(
            "/ui/login",
            data={"api_key": "testkey"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        set_cookie = resp.headers.get("set-cookie", "")
        assert "secure" in set_cookie.lower()
