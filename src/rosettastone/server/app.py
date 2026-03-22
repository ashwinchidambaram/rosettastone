"""FastAPI application factory for RosettaStone web UI."""

from __future__ import annotations

import logging
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
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError:
    raise ImportError("Web dependencies required. Install with: uv pip install 'rosettastone[web]'")

from rosettastone.server.database import get_engine, init_db

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


def _recover_orphaned_migrations() -> None:
    """Mark any 'running' migrations as 'failed' on startup (server restarted mid-run)."""
    from sqlmodel import Session, select

    from rosettastone.server.models import MigrationRecord

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

    # Single-worker executor — serializes all migrations (DSPy thread-safety)
    executor = ThreadPoolExecutor(max_workers=1)
    app.state.executor = executor

    yield

    executor.shutdown(wait=False)


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
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        # JSON for /api/ routes
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
            )
        # HTML for /ui/ routes
        if exc.status_code == 404:
            return templates.TemplateResponse(
                request, "404.html", {"active_nav": ""}, status_code=404,
            )
        if exc.status_code >= 500:
            return templates.TemplateResponse(
                request, "500.html", {"active_nav": ""}, status_code=exc.status_code,
            )
        # Fallback for other status codes
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )
        return templates.TemplateResponse(
            request, "500.html", {"active_nav": ""}, status_code=500,
        )

    # Register API routes
    from rosettastone.server.api.alerts import router as alerts_router
    from rosettastone.server.api.auth import router as auth_router
    from rosettastone.server.api.comparisons import router as comparisons_router
    from rosettastone.server.api.costs import router as costs_router
    from rosettastone.server.api.migrations import router as migrations_router
    from rosettastone.server.api.models import router as models_router
    from rosettastone.server.api.reports import router as reports_router

    app.include_router(migrations_router)
    app.include_router(comparisons_router)
    app.include_router(reports_router)
    app.include_router(models_router)
    app.include_router(costs_router)
    app.include_router(alerts_router)
    app.include_router(auth_router)

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
