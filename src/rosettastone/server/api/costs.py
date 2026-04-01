"""Cost aggregation and optimization endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from rosettastone.server.database import get_session
from rosettastone.server.models import MigrationRecord

router = APIRouter()


# ---------------------------------------------------------------------------
# Cost computation helpers
# ---------------------------------------------------------------------------


def _generate_opportunities(model_costs: dict[str, float], session: Session) -> list[dict[str, Any]]:
    """Generate cost optimization suggestions based on migration data."""
    opportunities = []

    # Look for completed migrations with high confidence that moved to cheaper models
    stmt = select(MigrationRecord).where(
        MigrationRecord.status == "complete",
        MigrationRecord.recommendation == "GO",
    )
    successful = list(session.exec(stmt).all())

    for migration in successful:
        source_short = (
            migration.source_model.split("/")[-1]
            if "/" in migration.source_model
            else migration.source_model
        )
        target_short = (
            migration.target_model.split("/")[-1]
            if "/" in migration.target_model
            else migration.target_model
        )
        confidence = round((migration.confidence_score or 0) * 100)

        if source_short in model_costs:
            estimated_savings = model_costs[source_short] * 0.3  # rough estimate
            opportunities.append(
                {
                    "title": f"Switch {source_short} workloads to {target_short}",
                    "savings": f"${estimated_savings:,.0f}/mo",
                    "confidence": f"{confidence}% parity",
                }
            )

    # If no real opportunities, return a generic one based on top-spend model
    if not opportunities and model_costs:
        top_model = max(model_costs, key=lambda m: model_costs[m])
        opportunities.append(
            {
                "title": f"Review {top_model} usage for optimization potential",
                "savings": f"${model_costs[top_model] * 0.2:,.0f}/mo",
                "confidence": "Requires evaluation",
            }
        )

    return opportunities[:3]  # Cap at 3


def _compute_costs(session: Session) -> dict[str, Any] | None:
    """Compute cost aggregation from MigrationRecord rows.

    Returns None when no migration data exists (signals caller to use dummy data).
    """
    stmt = select(
        MigrationRecord.source_model,
        MigrationRecord.target_model,
        MigrationRecord.cost_usd,
    )
    records = session.exec(stmt).all()

    if not records:
        return None  # Signal to use dummy data

    # Aggregate cost by target model
    model_costs: dict[str, float] = {}
    total = 0.0
    for row in records:
        target = row.target_model  # type: ignore[attr-defined]
        cost = row.cost_usd or 0.0  # type: ignore[attr-defined]
        short_name = target.split("/")[-1] if "/" in target else target
        model_costs[short_name] = model_costs.get(short_name, 0.0) + cost
        total += cost

    # Build by_model list sorted by cost descending
    by_model = []
    for model, cost in sorted(model_costs.items(), key=lambda x: x[1], reverse=True):
        pct = round(cost / total * 100) if total > 0 else 0
        by_model.append(
            {
                "model": model,
                "cost": f"${cost:,.2f}",
                "pct": pct,
            }
        )

    # Estimate potential savings (simple heuristic: 25% of total)
    potential_savings = total * 0.25

    # Generate optimization opportunities
    opportunities = _generate_opportunities(model_costs, session)

    return {
        "total_month": f"${total:,.2f}",
        "potential_savings": f"${potential_savings:,.2f}",
        "after_optimization": f"${total - potential_savings:,.2f}",
        "by_model": by_model,
        "opportunities": opportunities,
    }


# ---------------------------------------------------------------------------
# JSON API endpoints
# ---------------------------------------------------------------------------


@router.get("/api/v1/costs")
async def get_costs(session: Session = Depends(get_session)) -> dict[str, Any]:
    """Return aggregated cost data across all migrations."""
    costs = _compute_costs(session)
    if costs is None:
        return {
            "total_month": "$0.00",
            "potential_savings": "$0.00",
            "after_optimization": "$0.00",
            "by_model": [],
            "opportunities": [],
        }
    return costs


@router.get("/api/v1/costs/by-model")
async def get_costs_by_model(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    """Return cost breakdown by target model."""
    costs = _compute_costs(session)
    if costs is None:
        return []
    return costs["by_model"]  # type: ignore[no-any-return]
