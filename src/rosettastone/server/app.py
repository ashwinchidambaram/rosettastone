"""FastAPI application factory for RosettaStone web UI."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
except ImportError:
    raise ImportError("Web dependencies required. Install with: uv pip install 'rosettastone[web]'")

from rosettastone.server.database import init_db

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize database on startup."""
    init_db()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RosettaStone",
        description="LLM Migration Dashboard",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure Jinja2 templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.templates = templates

    # Mount static files if directory exists
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Register API routes
    from rosettastone.server.api.alerts import router as alerts_router
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

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
