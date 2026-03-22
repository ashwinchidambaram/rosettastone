"""Double-submit cookie CSRF protection middleware."""

from __future__ import annotations

import os
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_CSRF_COOKIE = "rosettastone_csrf"
_CSRF_HEADER = "x-csrf-token"
_CSRF_FORM_FIELD = "_csrf_token"
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def _csrf_enabled() -> bool:
    """CSRF is only useful when auth is enabled."""
    return bool(os.environ.get("ROSETTASTONE_API_KEY"))


class CSRFMiddleware(BaseHTTPMiddleware):
    """Double-submit cookie CSRF protection.

    Skipped when auth is disabled (no CSRF risk without auth).
    Skipped for /api/ routes (API uses Bearer tokens, not cookies).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if not _csrf_enabled():
            request.state.csrf_token = ""
            return await call_next(request)

        path = request.url.path

        # Skip CSRF for API routes (they use Bearer auth, not cookies)
        if path.startswith("/api/"):
            request.state.csrf_token = ""
            return await call_next(request)

        # Skip CSRF for static, login, and logout paths
        if (
            path.startswith("/static/")
            or path.startswith("/favicon")
            or path == "/ui/login"
            or path == "/ui/logout"
        ):
            request.state.csrf_token = ""
            return await call_next(request)

        # Read or generate the CSRF token
        token = request.cookies.get(_CSRF_COOKIE) or secrets.token_hex(32)
        request.state.csrf_token = token

        if request.method not in _SAFE_METHODS:
            # Validate: submitted token must match cookie token
            submitted = request.headers.get(_CSRF_HEADER, "")

            if not submitted:
                # Check form body
                content_type = request.headers.get("content-type", "")
                if "form" in content_type:
                    form = await request.form()
                    submitted = form.get(_CSRF_FORM_FIELD, "")

            cookie_token = request.cookies.get(_CSRF_COOKIE, "")
            if not cookie_token or not submitted or submitted != cookie_token:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF token missing or invalid"},
                )

        response = await call_next(request)

        # Set/refresh the CSRF cookie on every response (non-HttpOnly so JS can read it)
        response.set_cookie(
            key=_CSRF_COOKIE,
            value=token,
            httponly=False,
            samesite="lax",
            secure=False,
        )

        return response
