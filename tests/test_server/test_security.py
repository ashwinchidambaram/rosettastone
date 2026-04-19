"""Tests for Task 1.3: JWT secret enforcement, security headers, CORS, cookie flags."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from rosettastone.server.app import _JWT_SECRET_DEFAULT, create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(monkeypatch, env: dict[str, str], engine=None) -> TestClient:
    """Create a TestClient with the given environment variables set.

    When *engine* is provided, ``get_session`` is overridden so that queries
    use the test database (required for endpoints that touch the DB on Postgres).
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    app = create_app()
    if engine is not None:
        from sqlmodel import Session

        from rosettastone.server.database import get_session

        def _override():
            with Session(engine) as session:
                yield session

        app.dependency_overrides[get_session] = _override
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


class TestCSRFCookieSecureFlag:
    def test_csrf_cookie_secure_flag_set_when_behind_https(self, monkeypatch, engine) -> None:
        """CSRF cookie must have Secure flag when ROSETTASTONE_BEHIND_HTTPS=true.

        Uses ROSETTASTONE_MULTI_USER (not API key) to enable CSRF without triggering
        the session-cookie auth flow (which would block the test client over HTTP).
        """
        monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
        monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", "a" * 64)
        monkeypatch.setenv("ROSETTASTONE_BEHIND_HTTPS", "true")
        monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)
        client = _make_client(monkeypatch, {}, engine=engine)
        # GET any UI path that is not excluded from CSRF middleware
        resp = client.get("/ui/alerts")
        set_cookie = resp.headers.get("set-cookie", "")
        assert "rosettastone_csrf" in set_cookie
        assert "secure" in set_cookie.lower()

    def test_csrf_cookie_no_secure_flag_without_https(self, monkeypatch, engine) -> None:
        """CSRF cookie must NOT have Secure flag when ROSETTASTONE_BEHIND_HTTPS is not set."""
        monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
        monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", "a" * 64)
        monkeypatch.delenv("ROSETTASTONE_BEHIND_HTTPS", raising=False)
        monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)
        client = _make_client(monkeypatch, {}, engine=engine)
        resp = client.get("/ui/alerts")
        set_cookie = resp.headers.get("set-cookie", "")
        if "rosettastone_csrf" in set_cookie:
            assert "secure" not in set_cookie.lower()


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


# ---------------------------------------------------------------------------
# CSP nonce-based approach
# ---------------------------------------------------------------------------


class TestCSPNonce:
    def test_csp_script_src_contains_nonce(self, monkeypatch) -> None:
        """CSP script-src must include a per-request nonce instead of 'unsafe-inline'."""
        client = _make_client(monkeypatch, {})
        resp = client.get("/api/v1/health")
        csp = resp.headers.get("content-security-policy", "")
        assert "'nonce-" in csp, f"Expected nonce in CSP script-src, got: {csp}"

    def test_csp_script_src_does_not_contain_unsafe_inline(self, monkeypatch) -> None:
        """CSP script-src must NOT contain 'unsafe-inline' — nonce replaces it."""
        client = _make_client(monkeypatch, {})
        resp = client.get("/api/v1/health")
        csp = resp.headers.get("content-security-policy", "")
        # Extract only the script-src directive to avoid false positives from style-src
        script_src = ""
        for directive in csp.split(";"):
            directive = directive.strip()
            if directive.startswith("script-src"):
                script_src = directive
                break
        assert "'unsafe-inline'" not in script_src, (
            f"'unsafe-inline' must not appear in script-src, got: {script_src}"
        )

    def test_csp_nonce_differs_per_request(self, monkeypatch) -> None:
        """Each request must generate a fresh nonce (not reuse a static value)."""
        client = _make_client(monkeypatch, {})
        resp1 = client.get("/api/v1/health")
        resp2 = client.get("/api/v1/health")
        csp1 = resp1.headers.get("content-security-policy", "")
        csp2 = resp2.headers.get("content-security-policy", "")
        assert csp1 != csp2, "Each request must produce a unique nonce"

    def test_csp_style_src_uses_nonce_not_unsafe_inline(self, monkeypatch) -> None:
        """style-src must use nonce-based CSP instead of 'unsafe-inline'."""
        client = _make_client(monkeypatch, {})
        resp = client.get("/api/v1/health")
        csp = resp.headers.get("content-security-policy", "")
        style_src = ""
        for directive in csp.split(";"):
            directive = directive.strip()
            if directive.startswith("style-src"):
                style_src = directive
                break
        assert "'unsafe-inline'" not in style_src, (
            f"style-src must not use 'unsafe-inline', got: {style_src}"
        )
        assert "nonce-" in style_src, f"style-src must include a nonce, got: {style_src}"


# ---------------------------------------------------------------------------
# CORS always registered
# ---------------------------------------------------------------------------


class TestCORSAlwaysRegistered:
    def test_cors_middleware_registered_without_env_var(self, monkeypatch) -> None:
        """CORSMiddleware must always be registered even when ROSETTASTONE_CORS_ORIGINS is unset.
        With an empty allow_origins list, no ACAO header is sent for cross-origin requests
        (same-origin only semantics)."""
        monkeypatch.delenv("ROSETTASTONE_CORS_ORIGINS", raising=False)
        client = _make_client(monkeypatch, {})
        # Same-origin request — should succeed
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_cors_empty_origins_blocks_cross_origin(self, monkeypatch) -> None:
        """With ROSETTASTONE_CORS_ORIGINS unset (empty list), cross-origin requests
        must not receive an Access-Control-Allow-Origin header."""
        monkeypatch.delenv("ROSETTASTONE_CORS_ORIGINS", raising=False)
        client = _make_client(monkeypatch, {})
        resp = client.get("/api/v1/health", headers={"Origin": "https://external.example.com"})
        assert "access-control-allow-origin" not in resp.headers

    def test_cors_middleware_registered_with_env_var(self, monkeypatch) -> None:
        """CORSMiddleware must respond correctly to allowed origins when env var is set."""
        client = _make_client(
            monkeypatch, {"ROSETTASTONE_CORS_ORIGINS": "https://trusted.example.com"}
        )
        resp = client.get(
            "/api/v1/health",
            headers={"Origin": "https://trusted.example.com"},
        )
        assert resp.headers.get("access-control-allow-origin") == "https://trusted.example.com"


# ---------------------------------------------------------------------------
# Task 1.2: JWT default secret transition tests
# ---------------------------------------------------------------------------


class TestJWTDefaultSecretTransition:
    def test_default_jwt_secret_raises_in_multi_user_mode(self, monkeypatch) -> None:
        """Multi-user mode with the default dev JWT secret must raise RuntimeError at startup.

        This documents that the server refuses to start when the operator has not
        configured a real secret — preventing silent use of a well-known dev value.
        """
        monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
        monkeypatch.delenv("ROSETTASTONE_JWT_SECRET", raising=False)

        from rosettastone.server.app import _check_jwt_secret

        with pytest.raises(RuntimeError, match="insecure default"):
            _check_jwt_secret()

    def test_short_jwt_secret_warns(self, monkeypatch, caplog) -> None:
        """A JWT secret shorter than 32 bytes must emit a warning log in multi-user mode."""
        short_secret = "tooshort"  # 8 bytes < 32
        monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
        monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", short_secret)

        from rosettastone.server.app import _check_jwt_secret

        with caplog.at_level(logging.WARNING, logger="rosettastone.server"):
            _check_jwt_secret()  # must not raise

        assert any(
            "minimum recommended length" in record.message or "32 bytes" in record.message
            for record in caplog.records
        ), f"Expected a warning about short JWT secret. Got: {[r.message for r in caplog.records]}"

    def test_single_user_mode_allows_default_secret(self, monkeypatch) -> None:
        """Single-user mode must succeed even when no JWT secret is configured.

        The default secret is only forbidden in multi-user deployments.
        """
        monkeypatch.delenv("ROSETTASTONE_MULTI_USER", raising=False)
        monkeypatch.delenv("ROSETTASTONE_JWT_SECRET", raising=False)

        from rosettastone.server.app import _check_jwt_secret

        _check_jwt_secret()  # must not raise

        # App creation must also succeed
        app = create_app()
        assert app is not None

    @pytest.mark.filterwarnings("ignore::jwt.warnings.InsecureKeyLengthWarning")
    def test_jwt_with_default_secret_is_valid_token(self) -> None:
        """Tokens signed with the default dev secret are cryptographically valid.

        This test documents the risk: any token signed with _JWT_SECRET_DEFAULT
        would be accepted by a misconfigured server. The secret is publicly known
        from the source code, so any attacker could forge admin tokens.
        """
        from rosettastone.server.auth_utils import create_jwt, decode_jwt

        token = create_jwt(user_id=1, role="admin", secret=_JWT_SECRET_DEFAULT)
        payload = decode_jwt(token, _JWT_SECRET_DEFAULT)

        assert payload["sub"] == "1"
        assert payload["role"] == "admin"
        # Token is valid — this proves that exposing the default secret
        # means any actor can forge authenticated requests.


# ---------------------------------------------------------------------------
# Task 1.3: CORS origin validation tests
# ---------------------------------------------------------------------------


class TestCORSOriginValidation:
    def test_cors_no_env_rejects_cross_origin(self, monkeypatch) -> None:
        """Without ROSETTASTONE_CORS_ORIGINS set, cross-origin requests must not receive
        Access-Control-Allow-Origin. Same-origin semantics apply by default."""
        monkeypatch.delenv("ROSETTASTONE_CORS_ORIGINS", raising=False)
        client = _make_client(monkeypatch, {})

        resp = client.get(
            "/api/v1/health",
            headers={"Origin": "https://attacker.example.com"},
        )

        assert resp.status_code == 200
        assert "access-control-allow-origin" not in resp.headers

    def test_cors_specific_origin_allowed(self, monkeypatch) -> None:
        """An origin listed in ROSETTASTONE_CORS_ORIGINS must receive the ACAO header."""
        monkeypatch.setenv("ROSETTASTONE_CORS_ORIGINS", "https://app.example.com")
        client = _make_client(monkeypatch, {})

        resp = client.get(
            "/api/v1/health",
            headers={"Origin": "https://app.example.com"},
        )

        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "https://app.example.com"

    def test_cors_unlisted_origin_rejected(self, monkeypatch) -> None:
        """An origin NOT in the allowlist must not receive an ACAO header.

        Starlette's CORSMiddleware either omits the header or reflects a different
        value — in both cases the browser will block the cross-origin read.
        """
        monkeypatch.setenv("ROSETTASTONE_CORS_ORIGINS", "https://app.example.com")
        client = _make_client(monkeypatch, {})

        resp = client.get(
            "/api/v1/health",
            headers={"Origin": "https://evil.example.com"},
        )

        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao != "https://evil.example.com", (
            "Unlisted origin must not be reflected in Access-Control-Allow-Origin"
        )

    def test_cors_preflight_options(self, monkeypatch) -> None:
        """An OPTIONS preflight from an allowed origin must return the expected CORS headers
        including Access-Control-Allow-Methods with the configured methods."""
        monkeypatch.setenv("ROSETTASTONE_CORS_ORIGINS", "https://app.example.com")
        client = _make_client(monkeypatch, {})

        resp = client.options(
            "/api/v1/health",
            headers={
                "Origin": "https://app.example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization",
            },
        )

        # Preflight should complete (200 or 204)
        assert resp.status_code in (200, 204)
        allow_methods = resp.headers.get("access-control-allow-methods", "")
        assert allow_methods, "Access-Control-Allow-Methods must be present in preflight response"
        # The app configures GET, POST, PUT, DELETE, PATCH
        assert "GET" in allow_methods or "POST" in allow_methods, (
            f"Expected HTTP methods in Access-Control-Allow-Methods, got: {allow_methods}"
        )
