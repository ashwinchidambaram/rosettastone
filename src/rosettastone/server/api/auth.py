"""Optional API key authentication for the RosettaStone web server."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections import defaultdict

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

router = APIRouter()

_PUBLIC_PREFIXES = ("/static/", "/favicon", "/api/v1/health", "/ui/login", "/api/v1/auth/")

# Rate limiting: max 5 failed attempts per IP per minute
_failed_attempts: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT_MAX = 5
_RATE_LIMIT_WINDOW = 60.0  # seconds


def get_api_key() -> str | None:
    """Return the configured API key, or None if auth is disabled."""
    return os.environ.get("ROSETTASTONE_API_KEY") or None


def _verify_key(provided: str, expected: str) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    return hmac.compare_digest(provided.encode(), expected.encode())


def _create_session_token(api_key: str) -> str:
    """Derive a session token from the API key."""
    return hashlib.sha256(("rosettastone:" + api_key).encode()).hexdigest()


def _try_decode_jwt(token: str, secret: str, request: Request) -> object | None:
    """Try to decode a JWT token. Returns a minimal user dict or None on failure."""
    try:
        from rosettastone.server.auth_utils import decode_jwt

        payload = decode_jwt(token, secret)
        return {"user_id": int(payload["sub"]), "role": payload.get("role", "viewer")}
    except Exception:
        return None


def _check_rate_limit(ip: str) -> bool:
    """Return True if IP is rate-limited (too many failed attempts)."""
    now = time.time()
    attempts = _failed_attempts[ip]
    # Prune old attempts outside the window
    _failed_attempts[ip] = [t for t in attempts if now - t < _RATE_LIMIT_WINDOW]
    return len(_failed_attempts[ip]) >= _RATE_LIMIT_MAX


def _record_failed_attempt(ip: str) -> None:
    """Record a failed auth attempt for rate limiting."""
    _failed_attempts[ip].append(time.time())


class AuthMiddleware(BaseHTTPMiddleware):
    """Enforce API key auth for /api/ and /ui/ routes when ROSETTASTONE_API_KEY is set."""

    async def dispatch(self, request: Request, call_next):
        api_key = get_api_key()
        multi_user = os.environ.get("ROSETTASTONE_MULTI_USER", "").lower() in ("1", "true", "yes")

        path = request.url.path

        # Auth completely disabled — no API key set
        if api_key is None and not multi_user:
            return await call_next(request)

        # Exempt public paths always
        for prefix in _PUBLIC_PREFIXES:
            if path.startswith(prefix) or path == prefix.rstrip("/"):
                return await call_next(request)

        # Multi-user mode: JWT Bearer token cascade
        if multi_user and path.startswith("/api/"):
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[len("Bearer ") :]
                # Try JWT first
                jwt_secret = os.environ.get(
                    "ROSETTASTONE_JWT_SECRET", "dev-secret-change-in-production"
                )
                user = _try_decode_jwt(token, jwt_secret, request)
                if user is not None:
                    request.state.user = user
                    return await call_next(request)
                # Fall through to legacy API key check
                if api_key and _verify_key(token, api_key):
                    request.state.user = None  # legacy mode, no user object
                    return await call_next(request)
            return JSONResponse(status_code=401, content={"detail": "Authentication required"})

        # Legacy single-key mode (unchanged behavior)
        if api_key is not None:
            if path.startswith("/api/"):
                auth_header = request.headers.get("Authorization", "")
                if not auth_header.startswith("Bearer "):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing API key"},
                    )
                provided = auth_header[len("Bearer ") :]
                if not _verify_key(provided, api_key):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing API key"},
                    )
                return await call_next(request)

            if path.startswith("/ui/"):
                session_cookie = request.cookies.get("rosettastone_session", "")
                expected_token = _create_session_token(api_key)
                if not session_cookie or not hmac.compare_digest(
                    session_cookie.encode(), expected_token.encode()
                ):
                    return RedirectResponse(url="/ui/login", status_code=302)
                return await call_next(request)

        return await call_next(request)


@router.get("/ui/login")
async def login_page(request: Request):
    """Render the login page, or redirect to /ui/ if auth is disabled."""
    if get_api_key() is None:
        return RedirectResponse(url="/ui/", status_code=302)
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "login.html", {})


@router.post("/ui/login")
async def login_submit(request: Request, api_key: str = Form(...)):
    """Verify the submitted API key and set the session cookie on success."""
    templates = request.app.state.templates
    expected = get_api_key()

    if expected is None:
        return RedirectResponse(url="/ui/", status_code=302)

    if not _verify_key(api_key, expected):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Invalid API key. Please try again."},
            status_code=401,
        )

    token = _create_session_token(expected)
    response = RedirectResponse(url="/ui/", status_code=302)
    response.set_cookie(
        key="rosettastone_session",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,  # set to True behind HTTPS in production
    )
    return response


@router.post("/ui/logout")
async def logout(request: Request):
    """Clear the session cookie and redirect to the login page."""
    response = RedirectResponse(url="/ui/login", status_code=302)
    response.delete_cookie(key="rosettastone_session")
    return response
