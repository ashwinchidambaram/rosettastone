"""
RosettaStone Stitch UI — Raw Stitch variant.

Standalone FastAPI app with Jinja2 templates ported directly from
Stitch-generated HTML with minimal changes. All data is hardcoded
dummy data; no database or external imports required.

Run:
    uvicorn rosettastone.server_stitch.app:app --reload --port 8001
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── Paths ────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"

# ── Dummy Data ───────────────────────────────────────────────────

DUMMY_MODELS: list[dict[str, Any]] = [
    {
        "id": "openai/gpt-4o",
        "provider": "OpenAI",
        "status": "active",
        "context": "128K",
        "cost_per_1m": "$2.50",
    },
    {
        "id": "anthropic/claude-sonnet-4",
        "provider": "Anthropic",
        "status": "active",
        "context": "200K",
        "cost_per_1m": "$3.00",
    },
    {
        "id": "openai/gpt-4o-mini",
        "provider": "OpenAI",
        "status": "active",
        "context": "128K",
        "cost_per_1m": "$0.15",
    },
    {
        "id": "openai/gpt-4o-0613",
        "provider": "OpenAI",
        "status": "deprecated",
        "context": "8K",
        "cost_per_1m": "$5.00",
        "retirement_date": "Apr 15, 2026",
        "replacement": "openai/gpt-4o",
    },
]

DUMMY_MIGRATIONS: list[dict[str, Any]] = [
    {
        "id": 1,
        "source": "gpt-4o",
        "target": "claude-sonnet-4",
        "recommendation": "Safe to ship",
        "confidence": 92,
        "test_cases": 156,
        "cost": "$2.34",
        "time_ago": "2 minutes ago",
        "status": "complete",
        "baseline": 85,
        "improvement": 7,
        "reasoning": (
            "All output types meet or exceed quality thresholds. JSON validation "
            "passes 100%. Minimal regressions detected in edge cases only."
        ),
        "per_type": [
            {
                "type": "JSON",
                "wins": 48,
                "total": 48,
                "badge": "PASS",
                "desc": "All fields match",
            },
            {
                "type": "Text",
                "wins": 89,
                "total": 96,
                "badge": "PASS",
                "desc": "Strong semantic match",
            },
            {
                "type": "Code",
                "wins": 5,
                "total": 6,
                "badge": "WARN",
                "desc": "Low sample size (6)",
            },
            {
                "type": "Classification",
                "wins": 4,
                "total": 6,
                "badge": "WARN",
                "desc": "Low sample size (6)",
            },
        ],
        "regressions": [
            {
                "tc_id": 42,
                "score": 0.31,
                "title": "Priority classification mismatch",
                "expected": "urgent",
                "got": "high_priority",
            },
            {
                "tc_id": 87,
                "score": 0.45,
                "title": "Truncated JSON response",
                "expected": '{"status": "complete", ...}',
                "got": '{"status": "done"}',
            },
            {
                "tc_id": 103,
                "score": 0.52,
                "title": "Different code formatting",
                "expected": "def foo():",
                "got": "def foo() ->None:",
            },
        ],
    },
    {
        "id": 2,
        "source": "gpt-4o-0613",
        "target": "gpt-4o",
        "recommendation": "Needs review",
        "confidence": 78,
        "test_cases": 43,
        "cost": "$1.12",
        "time_ago": "1 day ago",
        "status": "complete",
        "baseline": 72,
        "improvement": 6,
        "reasoning": (
            "Most output types pass, but classification accuracy dropped 8%. "
            "Recommend human review of edge cases before deployment."
        ),
    },
    {
        "id": 3,
        "source": "gpt-3.5-turbo",
        "target": "gpt-4o-mini",
        "recommendation": "Do not ship",
        "confidence": 61,
        "test_cases": 89,
        "cost": "$0.87",
        "time_ago": "3 days ago",
        "status": "failed",
        "baseline": 58,
        "improvement": 3,
        "reasoning": (
            "Critical schema violations in 16% of JSON outputs. Code generation "
            "quality dropped significantly. Multiple regressions across all output types."
        ),
        "per_type": [
            {
                "type": "JSON",
                "wins": 30,
                "total": 42,
                "badge": "FAIL",
                "desc": "Schema violations in 12 cases",
            },
            {
                "type": "Text",
                "wins": 22,
                "total": 28,
                "badge": "WARN",
                "desc": "Semantic drift detected",
            },
            {
                "type": "Code",
                "wins": 3,
                "total": 11,
                "badge": "FAIL",
                "desc": "Syntax errors in 8 outputs",
            },
            {
                "type": "Classification",
                "wins": 5,
                "total": 8,
                "badge": "FAIL",
                "desc": "37% accuracy drop",
            },
        ],
        "regressions": [
            {
                "tc_id": 7,
                "score": 0.12,
                "title": "Schema Violation: Missing required key 'metadata'",
                "expected": '{"data": ..., "metadata": ...}',
                "got": '{"data": ...}',
            },
            {
                "tc_id": 19,
                "score": 0.22,
                "title": "Syntax error in generated Python",
                "expected": "valid Python function",
                "got": "SyntaxError on line 3",
            },
            {
                "tc_id": 31,
                "score": 0.28,
                "title": "Wrong classification category",
                "expected": "billing_dispute",
                "got": "general_inquiry",
            },
        ],
    },
]

DUMMY_DIFF: dict[str, Any] = {
    "tc_id": 42,
    "is_win": False,
    "composite_score": 0.72,
    "output_type": "Classification",
    "scores": {"bertscore": 0.85, "embedding": 0.79, "composite": 0.72},
    "source_model": "gpt-4o",
    "target_model": "claude-sonnet-4",
    "expected": ('{\n  "priority": "urgent",\n  "category": "billing",\n  "confidence": 0.94\n}'),
    "actual": (
        '{\n  "priority": "high_priority",\n  "category": "billing",\n  "confidence": 0.91\n}'
    ),
}

DUMMY_ALERTS: list[dict[str, Any]] = [
    {
        "type": "deprecation",
        "model": "gpt-4o-0613",
        "date": "Apr 15, 2026",
        "days_left": 26,
        "message": "Model retiring in 26 days",
        "action": "Start migration to gpt-4o",
    },
    {
        "type": "price_change",
        "model": "claude-sonnet-4",
        "old_price": "$3.00",
        "new_price": "$2.50",
        "message": "Price decreased 17%",
        "action": "No action needed",
    },
    {
        "type": "new_model",
        "model": "claude-opus-4.6",
        "date": "Feb 5, 2026",
        "message": "New model available",
        "action": "+12% reasoning improvement over claude-opus-4",
    },
]

DUMMY_COSTS: dict[str, Any] = {
    "total_month": "$1,247",
    "potential_savings": "$312",
    "after_optimization": "$935",
    "by_model": [
        {"model": "gpt-4o", "cost": "$823", "pct": 66},
        {"model": "claude-sonnet-4", "cost": "$312", "pct": 25},
        {"model": "gpt-4o-mini", "cost": "$112", "pct": 9},
    ],
    "opportunities": [
        {
            "title": "Switch gpt-4o classification to gpt-4o-mini",
            "savings": "$187/mo",
            "confidence": "94% parity",
        },
        {
            "title": "Batch non-urgent gpt-4o requests",
            "savings": "$125/mo",
            "confidence": "No quality impact",
        },
    ],
}


# ── Helpers ──────────────────────────────────────────────────────


def _get_migration(migration_id: int) -> dict[str, Any] | None:
    for m in DUMMY_MIGRATIONS:
        if m["id"] == migration_id:
            return m
    return None


# ── App Factory ──────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the Stitch variant FastAPI app."""
    app = FastAPI(
        title="RosettaStone Stitch UI",
        description="Raw Stitch variant — UI shell with dummy data",
        version="0.1.0",
    )

    # Mount static files
    app.mount(
        "/ui/static",
        StaticFiles(directory=str(_STATIC_DIR)),
        name="stitch_static",
    )

    # Configure Jinja2
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # ── Routes ───────────────────────────────────────────────────

    @app.get("/ui/", response_class=HTMLResponse)
    async def models_page(request: Request, empty: bool = False) -> HTMLResponse:
        """Models dashboard. Pass ?empty=true for the empty/onboarding state."""
        if empty:
            return templates.TemplateResponse(
                "models_empty.html",
                {"request": request, "active_nav": "models"},
            )
        return templates.TemplateResponse(
            "models.html",
            {
                "request": request,
                "active_nav": "models",
                "models": DUMMY_MODELS,
                "alerts": DUMMY_ALERTS[:2],
            },
        )

    @app.get("/ui/migrations", response_class=HTMLResponse)
    async def migrations_page(request: Request) -> HTMLResponse:
        """Migrations list page."""
        return templates.TemplateResponse(
            "migrations.html",
            {
                "request": request,
                "active_nav": "migrations",
                "migrations": DUMMY_MIGRATIONS,
            },
        )

    @app.get("/ui/migrations/{migration_id}", response_class=HTMLResponse)
    async def migration_detail_page(request: Request, migration_id: int) -> HTMLResponse:
        """Migration detail page — layout varies by recommendation."""
        migration = _get_migration(migration_id)
        if migration is None:
            migration = DUMMY_MIGRATIONS[0]
        return templates.TemplateResponse(
            "migration_detail.html",
            {
                "request": request,
                "active_nav": "migrations",
                "migration": migration,
            },
        )

    @app.get("/ui/costs", response_class=HTMLResponse)
    async def costs_page(request: Request) -> HTMLResponse:
        """Cost overview and optimization page."""
        return templates.TemplateResponse(
            "costs.html",
            {
                "request": request,
                "active_nav": "costs",
                "costs": DUMMY_COSTS,
            },
        )

    @app.get("/ui/alerts", response_class=HTMLResponse)
    async def alerts_page(request: Request) -> HTMLResponse:
        """Alerts hub page."""
        return templates.TemplateResponse(
            "alerts.html",
            {
                "request": request,
                "active_nav": "alerts",
                "alerts": DUMMY_ALERTS,
            },
        )

    @app.get("/ui/migrations/{migration_id}/executive", response_class=HTMLResponse)
    async def executive_report_page(request: Request, migration_id: int) -> HTMLResponse:
        """Standalone executive report (light mode, print-ready)."""
        migration = _get_migration(migration_id)
        if migration is None:
            migration = DUMMY_MIGRATIONS[0]
        return templates.TemplateResponse(
            "executive_report.html",
            {
                "request": request,
                "migration": migration,
                "report_date": datetime.date.today().strftime("%B %d, %Y"),
            },
        )

    @app.get(
        "/ui/fragments/diff/{migration_id}/{tc_id}",
        response_class=HTMLResponse,
    )
    async def diff_fragment(request: Request, migration_id: int, tc_id: int) -> HTMLResponse:
        """HTMX fragment — diff slide-over panel content."""
        # Use dummy diff data regardless of IDs (for demo purposes)
        diff = dict(DUMMY_DIFF)
        diff["tc_id"] = tc_id
        return templates.TemplateResponse(
            "fragments/diff_slideout.html",
            {"request": request, "diff": diff},
        )

    return app


# ── Module-level app instance for uvicorn ────────────────────────
app = create_app()
