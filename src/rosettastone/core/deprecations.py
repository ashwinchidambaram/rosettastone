"""Model deprecation registry — importable without the web stack."""
from __future__ import annotations

from datetime import UTC, date, datetime
from typing import NamedTuple


class DeprecationEntry(NamedTuple):
    model_id: str  # e.g. "openai/gpt-3.5-turbo"
    retirement_date: date  # date after which model is unavailable
    replacement: str  # suggested replacement model ID
    notes: str = ""


# Known model deprecations — expand as vendors announce sunset dates
KNOWN_DEPRECATIONS: list[DeprecationEntry] = [
    # OpenAI
    DeprecationEntry(
        model_id="openai/gpt-3.5-turbo",
        retirement_date=date(2025, 9, 30),
        replacement="openai/gpt-4o-mini",
        notes="GPT-3.5-turbo retiring in favor of GPT-4o-mini",
    ),
    DeprecationEntry(
        model_id="openai/gpt-3.5-turbo-0613",
        retirement_date=date(2026, 3, 1),
        replacement="openai/gpt-4o-mini",
        notes="GPT-3.5-turbo-0613 retiring",
    ),
    DeprecationEntry(
        model_id="openai/gpt-4-0613",
        retirement_date=date(2026, 6, 1),
        replacement="openai/gpt-4o",
        notes="GPT-4-0613 retiring in favor of GPT-4o",
    ),
    DeprecationEntry(
        model_id="openai/gpt-4o-0613",
        retirement_date=date(2026, 4, 15),
        replacement="openai/gpt-4o",
        notes="GPT-4o-0613 retiring",
    ),
    # Anthropic
    DeprecationEntry(
        model_id="anthropic/claude-2",
        retirement_date=date(2025, 3, 31),
        replacement="anthropic/claude-3-haiku-20240307",
        notes="Claude 2 retired",
    ),
    DeprecationEntry(
        model_id="anthropic/claude-2.1",
        retirement_date=date(2025, 3, 31),
        replacement="anthropic/claude-3-haiku-20240307",
        notes="Claude 2.1 retired",
    ),
    # Google
    DeprecationEntry(
        model_id="google/palm-2",
        retirement_date=date(2024, 10, 1),
        replacement="google/gemini-pro",
        notes="PaLM 2 retired",
    ),
]


def _deprecation_by_model_id(model_id: str) -> DeprecationEntry | None:
    """Return deprecation entry for model_id, or None if not in registry."""
    for entry in KNOWN_DEPRECATIONS:
        if entry.model_id == model_id:
            return entry
    return None


def days_until_retirement(entry: DeprecationEntry) -> int:
    """Return days until retirement (negative means already retired)."""
    today = datetime.now(tz=UTC).date()
    return (entry.retirement_date - today).days


def check_model_deprecation(model_id: str) -> dict | None:
    """Check if a model is deprecated. Returns dict with info or None.

    Returns:
        None if model is not in deprecation registry.
        dict with keys: model_id, retirement_date, replacement, days_until, severity
        severity is "warning" if >30 days, "critical" if <=30 days or already retired.
    """
    entry = _deprecation_by_model_id(model_id)
    if entry is None:
        return None
    days = days_until_retirement(entry)
    severity = "critical" if days <= 30 else "warning"
    return {
        "model_id": model_id,
        "retirement_date": entry.retirement_date.isoformat(),
        "replacement": entry.replacement,
        "notes": entry.notes,
        "days_until_retirement": days,
        "severity": severity,
        "already_retired": days < 0,
    }
