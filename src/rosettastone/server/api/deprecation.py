"""Deprecation monitoring — scan registered models for upcoming deprecations."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

from sqlmodel import Session, select

from rosettastone.server.models import Alert, RegisteredModel

logger = logging.getLogger(__name__)

KNOWN_DEPRECATIONS: dict[str, dict] = {
    "openai/gpt-4o-0613": {
        "date": "2026-04-15",
        "replacement": "openai/gpt-4o",
    },
    "openai/gpt-4-0613": {
        "date": "2026-06-01",
        "replacement": "openai/gpt-4o",
    },
    "openai/gpt-3.5-turbo-0613": {
        "date": "2026-03-01",
        "replacement": "openai/gpt-4o-mini",
    },
}


def _load_custom_deprecations() -> dict[str, dict]:
    """Return KNOWN_DEPRECATIONS merged with any overrides from ROSETTASTONE_DEPRECATIONS_JSON."""
    merged = dict(KNOWN_DEPRECATIONS)

    custom_path = os.environ.get("ROSETTASTONE_DEPRECATIONS_JSON")
    if custom_path:
        try:
            with open(custom_path) as fh:
                overrides: dict[str, dict] = json.load(fh)
            merged.update(overrides)
            logger.debug(
                "Loaded %d custom deprecation entries from %s",
                len(overrides),
                custom_path,
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load custom deprecations from %s: %s", custom_path, exc)

    return merged


def check_deprecations(session: Session) -> int:
    """Check registered models against known deprecations and create alerts.

    Returns the number of new Alert records created.
    """
    deprecations = _load_custom_deprecations()
    models = list(session.exec(select(RegisteredModel)).all())

    now = datetime.now(UTC)
    new_alerts = 0

    for model in models:
        dep_info = deprecations.get(model.model_id)
        if dep_info is None:
            continue

        dep_date = datetime.strptime(dep_info["date"], "%Y-%m-%d").replace(tzinfo=UTC)
        days_until = (dep_date - now).days

        # Only alert within 90-day window (includes already-deprecated models where days_until < 0)
        if days_until > 90:
            continue

        # Idempotency: skip if an alert for this model+type already exists
        existing = session.exec(
            select(Alert).where(
                Alert.alert_type == "deprecation",
                Alert.model_id == model.model_id,
            )
        ).first()
        if existing is not None:
            continue

        severity = "critical" if days_until <= 30 else "warning"

        alert = Alert(
            alert_type="deprecation",
            model_id=model.model_id,
            title=f"Model deprecation: {model.model_id}",
            message=(
                f"Retiring in {days_until} days (on {dep_info['date']}). "
                f"Recommended replacement: {dep_info['replacement']}"
            ),
            action=f"Migrate to {dep_info['replacement']}",
            severity=severity,
            metadata_json=json.dumps(
                {
                    "days_left": days_until,
                    "deprecation_date": dep_info["date"],
                    "replacement": dep_info["replacement"],
                }
            ),
        )
        session.add(alert)
        new_alerts += 1
        logger.debug(
            "Queued deprecation alert for %s (days_until=%d, severity=%s)",
            model.model_id,
            days_until,
            severity,
        )

    if new_alerts:
        session.commit()

    return new_alerts
