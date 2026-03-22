"""Optional API key authentication for the RosettaStone web server."""

from __future__ import annotations

import hashlib
import hmac
import os

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

router = APIRouter()

_PUBLIC_PREFIXES = ("/static/", "/favicon", "/api/v1/health", "/ui/login")


def get_api_key() -> str | None:
    """Return the configured API key, or None if auth is disabled."""
    return os.environ.get("ROSETTASTONE_API_KEY") or None


def _verify_key(provided: str, expected: str) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    return hmac.compare_digest(provided.encode(), expected.encode())


def _create_session_token(api_key: str) -> str:
    """Derive a session token from the API key."""
    return hashlib.sha256(("rosettastone:" + api_key).encode()).hexdigest()


class AuthMiddleware(BaseHTTPMiddleware):
    """Enforce API key auth for /api/ and /ui/ routes when ROSETTASTONE_API_KEY is set."""

    async def dispatch(self, request: Request, call_next):
        api_key = get_api_key()

        # Auth disabled — local dev mode
        if api_key is None:
            return await call_next(request)

        path = request.url.path

        # Exempt public paths
        for prefix in _PUBLIC_PREFIXES:
            if path.startswith(prefix) or path == prefix.rstrip("/"):
                return await call_next(request)

        # Protect /api/ routes with Bearer token
        if path.startswith("/api/"):
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
            provided = auth_header[len("Bearer "):]
            if not _verify_key(provided, api_key):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
            return await call_next(request)

        # Protect /ui/ routes with session cookie
        if path.startswith("/ui/"):
            session_cookie = request.cookies.get("rosettastone_session", "")
            expected_token = _create_session_token(api_key)
            if not session_cookie or not hmac.compare_digest(
                session_cookie.encode(), expected_token.encode()
            ):
                return RedirectResponse(url="/ui/login", status_code=302)
            return await call_next(request)

        # All other paths pass through
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
