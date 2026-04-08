"""Cost aggregation and optimization endpoints + per-user budget management."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from rosettastone.server.database import get_session
from rosettastone.server.models import MigrationRecord, UserBudget
from rosettastone.server.rbac import get_current_user_id, require_role

router = APIRouter()


class BudgetUpdate(BaseModel):
    monthly_limit_usd: float = Field(..., gt=0)


# ---------------------------------------------------------------------------
# Cost computation helpers
# ---------------------------------------------------------------------------


def _generate_opportunities(
    model_costs: dict[str, float], session: Session
) -> list[dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Per-User Budget Tracking (Task 3.5)
# ---------------------------------------------------------------------------


def _current_month() -> str:
    """Return current month in YYYY-MM format."""
    return datetime.now(tz=UTC).strftime("%Y-%m")


def get_or_create_budget(user_id: int, session: Session) -> UserBudget:
    """Get or create UserBudget for user_id. Resets spend if month changed.

    Note: 0.0 monthly_limit_usd means unlimited. None should not be used.
    """
    budget = session.exec(select(UserBudget).where(UserBudget.user_id == user_id)).first()
    month = _current_month()
    if budget is None:
        budget = UserBudget(
            user_id=user_id,
            monthly_limit_usd=0.0,  # 0.0 means unlimited
            current_month_spend_usd=0.0,
            budget_month=month,
        )
        session.add(budget)
        session.commit()
        session.refresh(budget)
    elif budget.budget_month != month:
        budget.current_month_spend_usd = 0.0
        budget.budget_month = month
        session.add(budget)
        session.commit()
        session.refresh(budget)
    return budget


def check_budget(user_id: int, estimated_cost: float, session: Session) -> None:
    """Raise HTTP 402 if adding estimated_cost would exceed the user's monthly limit.

    No-op if user has no limit set (monthly_limit_usd is 0.0 means unlimited).
    """
    budget = get_or_create_budget(user_id, session)
    if budget.monthly_limit_usd <= 0:
        # 0 or negative means unlimited
        return
    if budget.current_month_spend_usd + estimated_cost > budget.monthly_limit_usd:
        remaining = budget.monthly_limit_usd - budget.current_month_spend_usd
        raise HTTPException(
            status_code=402,
            detail=(
                f"Monthly budget exceeded. Limit: ${budget.monthly_limit_usd:.2f}, "
                f"Spent: ${budget.current_month_spend_usd:.2f}, "
                f"Remaining: ${remaining:.2f}, "
                f"Estimated: ${estimated_cost:.2f}"
            ),
        )


def record_spend(user_id: int, actual_cost: float, session: Session) -> None:
    """Add actual_cost to the user's current month spend."""
    if actual_cost <= 0:
        return
    budget = get_or_create_budget(user_id, session)
    budget.current_month_spend_usd = (budget.current_month_spend_usd or 0.0) + actual_cost
    session.add(budget)
    session.commit()


@router.get(
    "/api/v1/budgets/me",
    dependencies=[Depends(require_role("viewer", "editor", "admin"))],
)
async def get_my_budget(
    request: Request, session: Session = Depends(get_session)
) -> dict[str, Any]:
    """Get current user's budget status.

    Returns:
        - monthly_limit_usd: 0.0 means unlimited, positive value is the limit
        - current_month_spend_usd: current month's spend
        - budget_month: YYYY-MM format
    """
    user_id = get_current_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=400, detail="User ID not available")
    budget = get_or_create_budget(user_id, session)
    return {
        "user_id": user_id,
        "monthly_limit_usd": budget.monthly_limit_usd,
        "current_month_spend_usd": budget.current_month_spend_usd,
        "budget_month": budget.budget_month,
    }


@router.put(
    "/api/v1/budgets/{user_id}",
    dependencies=[Depends(require_role("admin"))],
)
async def set_user_budget(
    user_id: int,
    body: BudgetUpdate,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Admin: set monthly budget limit for a user."""
    budget = get_or_create_budget(user_id, session)
    budget.monthly_limit_usd = body.monthly_limit_usd
    session.add(budget)
    session.commit()
    session.refresh(budget)
    return {
        "user_id": user_id,
        "monthly_limit_usd": budget.monthly_limit_usd,
        "current_month_spend_usd": budget.current_month_spend_usd,
        "budget_month": budget.budget_month,
    }
