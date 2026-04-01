"""FastAPI application factory for RosettaStone web UI."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from starlette.exceptions import HTTPException as StarletteHTTPException
    from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
    from starlette.responses import Response
except ImportError:
    raise ImportError("Web dependencies required. Install with: uv pip install 'rosettastone[web]'")

from rosettastone.server.database import get_engine, init_db

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' cdn.tailwindcss.com; "
            "style-src 'self' 'unsafe-inline' fonts.googleapis.com; "
            "font-src 'self' fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
        return response


def _recover_orphaned_migrations() -> None:
    """Mark any 'running' migrations as 'failed' on startup (server restarted mid-run)."""
    from sqlmodel import Session, select

    from rosettastone.server.models import MigrationRecord, PipelineRecord

    engine = get_engine()
    with Session(engine) as session:
        stmt = select(MigrationRecord).where(MigrationRecord.status == "running")
        orphaned = list(session.exec(stmt).all())
        for record in orphaned:
            record.status = "failed"
            record.recommendation_reasoning = "Server restarted during migration"
            session.add(record)
        if orphaned:
            session.commit()
            logger.info("Recovered %d orphaned migration(s)", len(orphaned))

        # Also recover orphaned PipelineRecord rows stuck in "running"
        stale_pipelines = session.exec(
            select(PipelineRecord).where(PipelineRecord.status == "running")
        ).all()

        for pipeline in stale_pipelines:
            pipeline.status = "failed"
            session.add(pipeline)

        if stale_pipelines:
            logger.warning(
                "Recovered %d orphaned pipeline(s) stuck in 'running' state",
                len(stale_pipelines),
            )
            session.commit()


_JWT_SECRET_ENV = "ROSETTASTONE_JWT_SECRET"
_JWT_SECRET_DEFAULT = "dev-secret-change-in-production"
_JWT_SECRET_MIN_BYTES = 32
_SERVER_LOGGER = logging.getLogger("rosettastone.server")


def _check_jwt_secret() -> None:
    """Warn when multi-user mode is active with an insecure JWT secret."""
    multi_user = os.environ.get("ROSETTASTONE_MULTI_USER", "").lower() in ("1", "true", "yes")
    if not multi_user:
        return

    secret = os.environ.get(_JWT_SECRET_ENV, _JWT_SECRET_DEFAULT)
    if secret == _JWT_SECRET_DEFAULT:
        _SERVER_LOGGER.warning(
            "ROSETTASTONE_JWT_SECRET is set to the default dev value"
            " — set a strong secret before deploying to production"
        )
    elif len(secret) < _JWT_SECRET_MIN_BYTES:
        _SERVER_LOGGER.warning(
            "ROSETTASTONE_JWT_SECRET is %d bytes"
            " — minimum recommended length is 32 bytes for HS256",
            len(secret),
        )


def _check_model_deprecations() -> None:
    """Check registered models for upcoming deprecations and create alerts."""
    try:
        from sqlmodel import Session

        from rosettastone.server.api.deprecation import check_deprecations

        engine = get_engine()
        with Session(engine) as session:
            count = check_deprecations(session)
            if count:
                logger.info("Created %d deprecation alert(s)", count)
    except Exception as exc:
        logger.debug("Deprecation check skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize database and executor on startup."""
    init_db()
    _recover_orphaned_migrations()
    _check_model_deprecations()
    _check_jwt_secret()

    # Single-worker executor — serializes all migrations (DSPy thread-safety)
    executor = ThreadPoolExecutor(max_workers=1)
    app.state.executor = executor

    yield

    # Graceful shutdown: wait up to 30s for in-progress tasks to complete
    executor.shutdown(wait=True, cancel_futures=False)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RosettaStone",
        description="LLM Migration Dashboard",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware order: outermost first in add_middleware calls.
    # Starlette wraps them so the LAST added runs first (innermost).
    # Execution order: SecurityHeaders → Auth → CSRF → route handler
    from rosettastone.server.api.auth import AuthMiddleware
    from rosettastone.server.csrf import CSRFMiddleware

    app.add_middleware(CSRFMiddleware)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

    # Configure Jinja2 templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.templates = templates

    # Mount static files if directory exists
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # --- Error handlers ---
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
        # JSON for /api/ routes
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
            )
        # HTML for /ui/ routes
        if exc.status_code == 404:
            return templates.TemplateResponse(
                request,
                "404.html",
                {"active_nav": ""},
                status_code=404,
            )
        if exc.status_code >= 500:
            return templates.TemplateResponse(
                request,
                "500.html",
                {"active_nav": ""},
                status_code=exc.status_code,
            )
        # Fallback for other status codes
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )
        return templates.TemplateResponse(
            request,
            "500.html",
            {"active_nav": ""},
            status_code=500,
        )

    # Register API routes
    from rosettastone.server.api.ab_testing import router as ab_testing_router
    from rosettastone.server.api.alerts import router as alerts_router
    from rosettastone.server.api.annotations import router as annotations_router
    from rosettastone.server.api.approvals import router as approvals_router
    from rosettastone.server.api.audit import router as audit_router
    from rosettastone.server.api.auth import router as auth_router
    from rosettastone.server.api.comparisons import router as comparisons_router
    from rosettastone.server.api.costs import router as costs_router
    from rosettastone.server.api.migrations import router as migrations_router
    from rosettastone.server.api.models import router as models_router
    from rosettastone.server.api.pipelines import router as pipelines_router
    from rosettastone.server.api.reports import router as reports_router
    from rosettastone.server.api.teams import router as teams_router
    from rosettastone.server.api.users import router as users_router
    from rosettastone.server.api.versioning import router as versioning_router

    app.include_router(migrations_router)
    app.include_router(comparisons_router)
    app.include_router(reports_router)
    app.include_router(models_router)
    app.include_router(costs_router)
    app.include_router(alerts_router)
    app.include_router(auth_router)
    app.include_router(versioning_router)
    app.include_router(audit_router)
    app.include_router(ab_testing_router)
    app.include_router(pipelines_router)
    app.include_router(users_router)
    app.include_router(teams_router)
    app.include_router(annotations_router)
    app.include_router(approvals_router)

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
