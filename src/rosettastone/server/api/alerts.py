"""Alert management endpoints — generation, listing, and dismissal."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from rosettastone.server.database import get_session
from rosettastone.server.models import Alert, MigrationRecord

router = APIRouter()


# ---------------------------------------------------------------------------
# Alert generation helpers
# ---------------------------------------------------------------------------


def _generate_alerts(session: Session) -> int:
    """Scan for alert-worthy events and create Alert records. Returns count of new alerts."""
    count = 0

    # Migration completion / failure alerts
    completed = session.exec(
        select(MigrationRecord).where(
            MigrationRecord.status.in_(["complete", "failed"])  # type: ignore[union-attr]
        )
    ).all()

    for migration in completed:
        # Idempotency: skip if an alert already exists for this migration
        existing = session.exec(
            select(Alert).where(
                Alert.migration_id == migration.id,
                Alert.alert_type.in_(["migration_complete", "migration_failed"]),  # type: ignore[union-attr]
            )
        ).first()

        if existing:
            continue

        if migration.status == "complete" and migration.recommendation == "GO":
            alert = Alert(
                alert_type="migration_complete",
                model_id=migration.target_model,
                migration_id=migration.id,
                title=f"Migration ready: {migration.source_model} to {migration.target_model}",
                message=(
                    f"Migration completed with "
                    f"{round((migration.confidence_score or 0) * 100)}% confidence"
                ),
                action="Review and deploy",
                severity="info",
            )
        elif migration.status == "failed" or migration.recommendation == "NO_GO":
            alert = Alert(
                alert_type="migration_failed",
                model_id=migration.target_model,
                migration_id=migration.id,
                title=(f"Migration blocked: {migration.source_model} to {migration.target_model}"),
                message=(
                    f"Migration failed or blocked — "
                    f"{migration.recommendation_reasoning or 'see details'}"
                ),
                action="Review results",
                severity="critical",
            )
        else:
            # CONDITIONAL or other ambiguous status
            alert = Alert(
                alert_type="migration_complete",
                model_id=migration.target_model,
                migration_id=migration.id,
                title=(
                    f"Migration needs review: {migration.source_model} to {migration.target_model}"
                ),
                message=(
                    f"Completed with {round((migration.confidence_score or 0) * 100)}% confidence,"
                    f" needs human review"
                ),
                action="Review edge cases",
                severity="warning",
            )

        session.add(alert)
        count += 1

    if count:
        session.commit()

    return count


def _alert_to_template_dict(alert: Alert) -> dict:
    """Convert an Alert record to the dict shape the template expects."""
    metadata = json.loads(alert.metadata_json)

    # Map internal alert_type to the template's expected "type" values
    type_mapping = {
        "migration_complete": "new_model",
        "migration_failed": "deprecation",
        "deprecation": "deprecation",
        "price_change": "price_change",
        "new_model": "new_model",
    }

    result: dict = {
        "id": alert.id,
        "type": type_mapping.get(alert.alert_type, alert.alert_type),
        "model": alert.model_id or "",
        "message": alert.message,
        "action": alert.action or "",
        "date": alert.created_at.strftime("%b %d, %Y"),
        "severity": alert.severity,
        "is_read": alert.is_read,
    }

    # Type-specific fields from metadata
    if "days_left" in metadata:
        result["days_left"] = metadata["days_left"]
    if "old_price" in metadata:
        result["old_price"] = metadata["old_price"]
    if "new_price" in metadata:
        result["new_price"] = metadata["new_price"]

    return result


# ---------------------------------------------------------------------------
# JSON API endpoints
# ---------------------------------------------------------------------------


@router.get("/api/v1/alerts")
async def list_alerts(
    unread_only: bool = False,
    session: Session = Depends(get_session),
) -> list[dict]:
    """List all alerts, newest first. Pass ?unread_only=true to filter unread."""
    stmt = select(Alert).order_by(Alert.created_at.desc()).limit(100)  # type: ignore[union-attr]
    records = list(session.exec(stmt).all())

    if unread_only:
        records = [a for a in records if not a.is_read]

    return [_alert_to_template_dict(a) for a in records]


@router.post("/api/v1/alerts/generate")
async def generate_alerts(session: Session = Depends(get_session)) -> dict:
    """Trigger alert generation — scans for new events and creates Alert records."""
    count = _generate_alerts(session)
    return {"generated": count}


@router.post("/api/v1/alerts/{alert_id}/read")
async def mark_alert_read(
    alert_id: int,
    session: Session = Depends(get_session),
) -> dict:
    """Mark an alert as read."""
    alert = session.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.is_read = True
    session.add(alert)
    session.commit()
    session.refresh(alert)
    return _alert_to_template_dict(alert)


@router.delete("/api/v1/alerts/{alert_id}")
async def delete_alert(
    alert_id: int,
    session: Session = Depends(get_session),
) -> dict:
    """Dismiss/delete an alert."""
    alert = session.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    session.delete(alert)
    session.commit()
    return {"deleted": True, "id": alert_id}
