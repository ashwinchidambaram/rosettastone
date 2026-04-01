"""Tests for core deprecations module — no web stack dependency."""
from __future__ import annotations

from datetime import date, datetime, timedelta

from rosettastone.core.deprecations import (
    KNOWN_DEPRECATIONS,
    DeprecationEntry,
    check_model_deprecation,
    days_until_retirement,
)


def test_importable_without_web_stack() -> None:
    """core.deprecations is importable with no FastAPI/SQLModel."""
    # Just importing above is the test
    assert len(KNOWN_DEPRECATIONS) > 0


def test_check_model_not_in_registry_returns_none() -> None:
    """Unknown model returns None."""
    result = check_model_deprecation("openai/gpt-4o")
    assert result is None


def test_check_model_in_registry_returns_dict() -> None:
    """Known deprecated model returns dict with expected keys."""
    result = check_model_deprecation("openai/gpt-3.5-turbo")
    assert result is not None
    assert "model_id" in result
    assert "retirement_date" in result
    assert "replacement" in result
    assert "severity" in result
    assert "days_until_retirement" in result
    assert "already_retired" in result
    assert result["model_id"] == "openai/gpt-3.5-turbo"


def test_days_until_retirement_past_date_is_negative() -> None:
    """days_until_retirement returns negative for already-retired models."""
    past_entry = DeprecationEntry(
        model_id="test/old-model",
        retirement_date=date(2020, 1, 1),
        replacement="test/new-model",
    )
    days = days_until_retirement(past_entry)
    assert days < 0


def test_days_until_retirement_future_date_is_positive() -> None:
    """days_until_retirement returns positive for future retirement."""
    future_entry = DeprecationEntry(
        model_id="test/future-model",
        retirement_date=date(2099, 1, 1),
        replacement="test/replacement",
    )
    days = days_until_retirement(future_entry)
    assert days > 0


def test_severity_critical_within_30_days() -> None:
    """Severity is critical when retirement is within 30 days."""
    near_date = datetime.now().date() + timedelta(days=10)
    near_entry = DeprecationEntry(
        model_id="test/near-model",
        retirement_date=near_date,
        replacement="test/replacement",
    )
    from rosettastone.core import deprecations

    deprecations.KNOWN_DEPRECATIONS.append(near_entry)
    try:
        result = check_model_deprecation("test/near-model")
        assert result is not None
        assert result["severity"] == "critical"
        assert result["days_until_retirement"] <= 30
    finally:
        deprecations.KNOWN_DEPRECATIONS.remove(near_entry)


def test_severity_critical_already_retired() -> None:
    """Severity is critical when already retired."""
    result = check_model_deprecation("google/palm-2")
    assert result is not None
    assert result["severity"] == "critical"
    assert result["already_retired"] is True


def test_severity_warning_beyond_30_days() -> None:
    """Severity is warning when retirement is beyond 30 days."""
    far_date = datetime.now().date() + timedelta(days=60)
    far_entry = DeprecationEntry(
        model_id="test/far-model",
        retirement_date=far_date,
        replacement="test/replacement",
    )
    from rosettastone.core import deprecations

    deprecations.KNOWN_DEPRECATIONS.append(far_entry)
    try:
        result = check_model_deprecation("test/far-model")
        assert result is not None
        assert result["severity"] == "warning"
        assert result["days_until_retirement"] > 30
    finally:
        deprecations.KNOWN_DEPRECATIONS.remove(far_entry)


def test_deprecation_entry_namedtuple() -> None:
    """DeprecationEntry is a proper NamedTuple."""
    entry = DeprecationEntry(
        model_id="test/model",
        retirement_date=date(2025, 6, 1),
        replacement="test/new",
        notes="Test notes",
    )
    assert entry.model_id == "test/model"
    assert entry.retirement_date == date(2025, 6, 1)
    assert entry.replacement == "test/new"
    assert entry.notes == "Test notes"
