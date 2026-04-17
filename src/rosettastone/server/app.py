"""FastAPI application factory for RosettaStone web UI."""

from __future__ import annotations

import logging
import os
import secrets
import time as _time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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
from rosettastone.server.logging_config import configure_logging, set_request_id

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Module-level readiness cache — avoids hitting DB/Redis on every Kubernetes health probe.
_readiness_cache: dict[str, Any] | None = None
_readiness_cache_time: float = 0.0
_READINESS_TTL: float = 5.0


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        nonce = secrets.token_urlsafe(16)
        request.state.csp_nonce = nonce
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            f"script-src 'self' 'nonce-{nonce}' cdn.tailwindcss.com; "
            f"style-src 'self' 'nonce-{nonce}' fonts.googleapis.com; "
            "font-src 'self' fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "object-src 'none'"
        )
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a UUID4 request ID to each request and surface it as a response header."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        set_request_id(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
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
    """Enforce JWT secret strength in multi-user mode; raise on default dev secret."""
    multi_user = os.environ.get("ROSETTASTONE_MULTI_USER", "").lower() in ("1", "true", "yes")
    if not multi_user:
        return

    secret = os.environ.get(_JWT_SECRET_ENV, _JWT_SECRET_DEFAULT)
    if secret == _JWT_SECRET_DEFAULT:
        raise RuntimeError(
            "ROSETTASTONE_JWT_SECRET is set to the insecure default dev value. "
            "Set a strong secret (>= 32 bytes) before running in multi-user mode."
        )
    if len(secret) < _JWT_SECRET_MIN_BYTES:
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


def _init_sentry() -> None:
    """Initialize Sentry SDK if SENTRY_DSN is configured."""
    sentry_dsn = os.environ.get("SENTRY_DSN")
    if not sentry_dsn:
        return
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=sentry_dsn,
            traces_sample_rate=0.1,
            profiles_sample_rate=0.1,
        )
    except ImportError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize database and task worker on startup."""
    from rosettastone.server.task_dispatch import TaskDispatcher
    from rosettastone.server.task_worker import TaskWorker

    init_db()
    _recover_orphaned_migrations()
    _check_model_deprecations()
    _check_jwt_secret()

    # DB-backed task worker — durable, survives server restarts
    task_worker = TaskWorker(get_engine())
    task_worker.recover_stale_tasks()

    # Dispatcher: uses RQ when REDIS_URL is set and Redis responds, falls back to DB queue
    dispatcher = TaskDispatcher()
    dispatcher.setup(task_worker)
    dispatcher.start()
    app.state.task_worker = dispatcher

    yield

    # Graceful shutdown: stop polling (current in-flight task completes naturally)
    dispatcher.stop()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    configure_logging()
    _init_sentry()

    multi_user = os.environ.get("ROSETTASTONE_MULTI_USER", "").lower() in ("1", "true", "yes")

    app = FastAPI(
        title="RosettaStone",
        description="LLM Migration Dashboard",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware order: outermost first in add_middleware calls.
    # Starlette wraps them so the LAST added runs first (innermost).
    # Execution order: RequestID → SecurityHeaders → CORS → Auth → CSRF → route handler
    from fastapi.middleware.cors import CORSMiddleware

    from rosettastone.server.api.auth import AuthMiddleware
    from rosettastone.server.csrf import CSRFMiddleware

    cors_origins_env = os.environ.get("ROSETTASTONE_CORS_ORIGINS", "")
    cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,  # empty list = same-origin only
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "X-CSRF-Token"],
    )

    # CSRF middleware: enable when auth is active (API key or multi-user mode)
    csrf_active = multi_user or bool(os.environ.get("ROSETTASTONE_API_KEY"))
    if csrf_active:
        app.add_middleware(CSRFMiddleware)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestIDMiddleware)

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
    from rosettastone.server.api.alerts import router as alerts_router
    from rosettastone.server.api.auth import router as auth_router
    from rosettastone.server.api.comparisons import router as comparisons_router
    from rosettastone.server.api.costs import router as costs_router
    from rosettastone.server.api.dataset_runs import router as dataset_runs_router
    from rosettastone.server.api.migrations import router as migrations_router
    from rosettastone.server.api.models import router as models_router
    from rosettastone.server.api.pipelines import router as pipelines_router
    from rosettastone.server.api.reports import router as reports_router
    from rosettastone.server.api.versioning import router as versioning_router

    # Core routers — always registered
    app.include_router(migrations_router)
    app.include_router(comparisons_router)
    app.include_router(reports_router)
    app.include_router(models_router)
    app.include_router(costs_router)
    app.include_router(dataset_runs_router)
    app.include_router(alerts_router)
    app.include_router(auth_router)
    app.include_router(versioning_router)
    app.include_router(pipelines_router)

    # Enterprise routers — only registered when ROSETTASTONE_MULTI_USER is enabled
    if multi_user:
        from rosettastone.server.api.ab_testing import router as ab_testing_router
        from rosettastone.server.api.annotations import router as annotations_router
        from rosettastone.server.api.approvals import router as approvals_router
        from rosettastone.server.api.audit import router as audit_router
        from rosettastone.server.api.teams import router as teams_router
        from rosettastone.server.api.users import router as users_router

        app.include_router(audit_router)
        app.include_router(ab_testing_router)
        app.include_router(users_router)
        app.include_router(teams_router)
        app.include_router(annotations_router)
        app.include_router(approvals_router)

    async def _check_readiness(app: FastAPI) -> dict[str, Any]:
        """Check all system components and return a readiness dict."""
        global _readiness_cache, _readiness_cache_time

        now = _time.monotonic()
        if _readiness_cache is not None and (now - _readiness_cache_time) < _READINESS_TTL:
            return _readiness_cache

        import os as _os

        components: dict[str, Any] = {}

        # Database check
        try:
            from sqlalchemy import text as _text
            from sqlmodel import Session as _Session

            from rosettastone.server.database import get_engine as _get_engine

            _engine = _get_engine()
            with _Session(_engine) as _sess:
                _sess.exec(_text("SELECT 1"))  # type: ignore[call-overload]
            components["database"] = {"status": "ok"}
        except Exception as _exc:
            components["database"] = {"status": "unavailable", "error": str(_exc)[:100]}

        # Task worker check (non-fatal if degraded)
        try:
            dispatcher = getattr(app.state, "task_worker", None)
            if dispatcher is not None:
                _db_worker = getattr(dispatcher, "_db_worker", None)
                _alive = _db_worker is not None and _db_worker._thread.is_alive()
                components["task_worker"] = {"status": "ok" if _alive else "degraded"}
            else:
                components["task_worker"] = {"status": "degraded"}
        except Exception:
            components["task_worker"] = {"status": "degraded"}

        # Redis check (optional, only if REDIS_URL is configured)
        _redis_url = _os.environ.get("REDIS_URL", "")
        if _redis_url:
            try:
                import redis as _redis_lib

                _r = _redis_lib.from_url(_redis_url)
                _r.ping()
                components["redis"] = {"status": "ok"}
            except ImportError:
                components["redis"] = {"status": "skipped"}
            except Exception:
                components["redis"] = {"status": "unavailable"}

        # Overall status: unavailable if DB down, degraded if worker/redis issue
        _db_status = components.get("database", {}).get("status", "unavailable")
        if _db_status == "unavailable":
            overall = "unavailable"
        elif any(v.get("status") not in ("ok", "skipped") for v in components.values()):
            overall = "degraded"
        else:
            overall = "ok"

        result: dict[str, Any] = {"status": overall, "components": components}
        _readiness_cache = result
        _readiness_cache_time = now
        return result

    @app.get("/api/v1/health")
    async def health(request: Request) -> dict[str, Any]:
        """Basic health check — enriched with component statuses."""
        return await _check_readiness(request.app)

    @app.get("/api/v1/health/ready")
    async def health_ready(request: Request) -> Response:
        """Readiness probe — returns 503 when database is unavailable."""
        from fastapi.responses import JSONResponse as _JSONResponse

        result = await _check_readiness(request.app)
        status_code = 503 if result["status"] == "unavailable" else 200
        return _JSONResponse(status_code=status_code, content=result)

    @app.get("/api/v1/health/live")
    async def health_live() -> dict[str, str]:
        """Liveness probe — always returns 200 while the process is running."""
        return {"status": "ok"}

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        """Prometheus metrics endpoint. Returns 404 if prometheus_client not installed."""
        from rosettastone.server.metrics import is_available, metrics_response

        if not is_available():
            return JSONResponse(
                status_code=404,
                content={
                    "detail": (
                        "prometheus_client not installed. "
                        "Install with: uv add 'rosettastone[metrics]'"
                    )
                },
            )
        data, content_type = metrics_response()
        return Response(content=data, media_type=content_type)

    return app
