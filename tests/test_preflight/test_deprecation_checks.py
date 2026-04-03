"""Tests for deprecation checks in the preflight module."""
from __future__ import annotations

from pathlib import Path

from rosettastone.config import MigrationConfig
from rosettastone.core.deprecations import KNOWN_DEPRECATIONS
from rosettastone.preflight.checks import run_all_checks


def test_deprecation_check_non_deprecated_models() -> None:
    """Non-deprecated models produce no deprecation warnings."""
    config = MigrationConfig(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        data_path=Path("dummy.jsonl"),
    )
    report = run_all_checks(config)

    # Should have no deprecation-related warnings
    deprecation_warnings = [w for w in report.warnings if "deprecat" in w.lower()]
    assert len(deprecation_warnings) == 0
    assert len(report.blockers) == 0


def test_deprecation_check_already_retired_source_warning() -> None:
    """Source model already retired produces a warning."""
    config = MigrationConfig(
        source_model="google/palm-2",  # Already retired
        target_model="google/gemini-pro",
        data_path=Path("dummy.jsonl"),
    )
    report = run_all_checks(config)

    # Should have a warning about source being retired
    deprecation_warnings = [w for w in report.warnings if "palm-2" in w and "retir" in w]
    assert len(deprecation_warnings) >= 1


def test_deprecation_check_already_retired_target_blocker() -> None:
    """Target model already retired produces a blocker."""
    config = MigrationConfig(
        source_model="openai/gpt-4o",
        target_model="google/palm-2",  # Already retired
        data_path=Path("dummy.jsonl"),
    )
    report = run_all_checks(config)

    # Should have a blocker about target being retired
    blockers = [b for b in report.blockers if "palm-2" in b and "retir" in b]
    assert len(blockers) >= 1


def test_deprecation_check_soon_to_retire() -> None:
    """Models retiring within 30 days produce warnings."""
    config = MigrationConfig(
        source_model="openai/gpt-3.5-turbo-0613",  # Retiring 2026-03-01, before 2026-04-01
        target_model="openai/gpt-4o-mini",
        data_path=Path("dummy.jsonl"),
    )
    report = run_all_checks(config)

    # Should have a warning about imminent retirement
    deprecation_warnings = [
        w for w in report.warnings if "3.5-turbo-0613" in w and "retir" in w
    ]
    assert len(deprecation_warnings) >= 1


def test_deprecation_check_replacement_suggested() -> None:
    """Deprecation warnings suggest a replacement model."""
    config = MigrationConfig(
        source_model="google/palm-2",
        target_model="google/gemini-pro",
        data_path=Path("dummy.jsonl"),
    )
    report = run_all_checks(config)

    # Warning should mention the replacement
    deprecation_warnings = [w for w in report.warnings if "palm-2" in w]
    assert len(deprecation_warnings) >= 1
    # Check that replacement is mentioned
    assert any("gemini-pro" in w for w in deprecation_warnings)


def test_deprecation_registry_loaded() -> None:
    """KNOWN_DEPRECATIONS is populated."""
    assert len(KNOWN_DEPRECATIONS) > 0
    # Verify expected entries exist
    model_ids = {entry.model_id for entry in KNOWN_DEPRECATIONS}
    assert "google/palm-2" in model_ids
    assert "openai/gpt-3.5-turbo" in model_ids
