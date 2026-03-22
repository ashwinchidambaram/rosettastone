"""Tests for the batch migration runner (rosettastone/batch.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError

from rosettastone.batch import (
    BatchEntry,
    BatchManifest,
    BatchResult,
    format_batch_summary,
    load_manifest,
    run_batch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_migration_result(**overrides) -> MagicMock:
    """Return a MagicMock that looks like a MigrationResult."""
    defaults = {
        "recommendation": "GO",
        "confidence_score": 0.92,
        "baseline_score": 0.85,
        "improvement": 0.07,
        "cost_usd": 1.23,
        "duration_seconds": 45.0,
        "optimized_prompt": "test",
        "warnings": [],
        "safety_warnings": [],
        "per_type_scores": {},
        "baseline_results": [],
        "validation_results": [],
    }
    defaults.update(overrides)
    mock = MagicMock()
    for attr, value in defaults.items():
        setattr(mock, attr, value)
    return mock


def _minimal_entry_dict(**overrides) -> dict:
    base = {
        "name": "my-migration",
        "source_model": "openai/gpt-4o",
        "target_model": "anthropic/claude-sonnet-4",
        "data_path": "examples/sample_data.jsonl",
    }
    base.update(overrides)
    return base


def _write_manifest(tmp_path: Path, content: dict) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.dump(content))
    return p


# ---------------------------------------------------------------------------
# load_manifest — valid YAML
# ---------------------------------------------------------------------------

def test_load_manifest_valid(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        {
            "version": 2,
            "defaults": {"gepa_auto": "medium"},
            "migrations": [_minimal_entry_dict()],
        },
    )
    manifest = load_manifest(manifest_path)

    assert isinstance(manifest, BatchManifest)
    assert manifest.version == 2
    assert manifest.defaults == {"gepa_auto": "medium"}
    assert len(manifest.migrations) == 1
    assert manifest.migrations[0].name == "my-migration"
    assert manifest.migrations[0].source_model == "openai/gpt-4o"
    assert manifest.migrations[0].target_model == "anthropic/claude-sonnet-4"


# ---------------------------------------------------------------------------
# load_manifest — invalid YAML
# ---------------------------------------------------------------------------

def test_load_manifest_invalid_yaml(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "manifest.yaml"
    bad_yaml.write_text("migrations: [this: is: not: valid yaml }: }")

    with pytest.raises(Exception):
        load_manifest(bad_yaml)


# ---------------------------------------------------------------------------
# load_manifest — missing required key
# ---------------------------------------------------------------------------

def test_load_manifest_missing_required(tmp_path: Path) -> None:
    # "migrations" key is absent — Pydantic should raise ValidationError
    manifest_path = _write_manifest(tmp_path, {"version": 1, "defaults": {}})

    with pytest.raises(ValidationError):
        load_manifest(manifest_path)


# ---------------------------------------------------------------------------
# Defaults merging
# ---------------------------------------------------------------------------

def test_defaults_merging(tmp_path: Path) -> None:
    # Entry does not specify gepa_auto, so it should inherit from defaults
    manifest_path = _write_manifest(
        tmp_path,
        {
            "defaults": {"gepa_auto": "heavy"},
            "migrations": [_minimal_entry_dict()],
        },
    )
    manifest = load_manifest(manifest_path)

    assert manifest.migrations[0].gepa_auto == "heavy"


def test_defaults_override(tmp_path: Path) -> None:
    # Entry explicitly sets gepa_auto="medium" (non-default value); defaults say "heavy".
    # Because "medium" != the Pydantic field default ("light"), the entry value is preserved.
    entry = _minimal_entry_dict(gepa_auto="medium")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "defaults": {"gepa_auto": "heavy"},
            "migrations": [entry],
        },
    )
    manifest = load_manifest(manifest_path)

    assert manifest.migrations[0].gepa_auto == "medium"


# ---------------------------------------------------------------------------
# BatchEntry defaults
# ---------------------------------------------------------------------------

def test_batch_entry_defaults() -> None:
    entry = BatchEntry(
        name="test",
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        data_path="data.jsonl",
    )
    assert entry.gepa_auto == "light"
    assert entry.dry_run is False
    assert entry.output_dir is None


# ---------------------------------------------------------------------------
# run_batch — success
# ---------------------------------------------------------------------------

def test_run_batch_success(tmp_path: Path) -> None:
    manifest = BatchManifest(
        migrations=[
            BatchEntry(**_minimal_entry_dict()),
        ]
    )
    mock_result = _make_migration_result()

    with (
        patch("rosettastone.config.MigrationConfig") as mock_config,
        patch("rosettastone.core.migrator.Migrator") as mock_migrator,
    ):
        mock_config.return_value = MagicMock()
        mock_migrator.return_value.run.return_value = mock_result

        results = run_batch(manifest, tmp_path)

    assert len(results) == 1
    r = results[0]
    assert r.status == "complete"
    assert r.recommendation == "GO"
    assert r.confidence == 0.92
    assert r.error is None


# ---------------------------------------------------------------------------
# run_batch — blocked
# ---------------------------------------------------------------------------

def test_run_batch_blocked(tmp_path: Path) -> None:
    from rosettastone.core.migrator import MigrationBlockedError

    manifest = BatchManifest(
        migrations=[BatchEntry(**_minimal_entry_dict())]
    )

    with (
        patch("rosettastone.config.MigrationConfig"),
        patch("rosettastone.core.migrator.Migrator") as mock_migrator,
    ):
        mock_migrator.return_value.run.side_effect = MigrationBlockedError("context window too small")

        results = run_batch(manifest, tmp_path)

    assert len(results) == 1
    r = results[0]
    assert r.status == "blocked"
    assert "context window too small" in (r.error or "")


# ---------------------------------------------------------------------------
# run_batch — failed (generic exception)
# ---------------------------------------------------------------------------

def test_run_batch_failed(tmp_path: Path) -> None:
    manifest = BatchManifest(
        migrations=[BatchEntry(**_minimal_entry_dict())]
    )

    with (
        patch("rosettastone.config.MigrationConfig"),
        patch("rosettastone.core.migrator.Migrator") as mock_migrator,
    ):
        mock_migrator.return_value.run.side_effect = RuntimeError("network timeout")

        results = run_batch(manifest, tmp_path)

    assert len(results) == 1
    r = results[0]
    assert r.status == "failed"
    assert "network timeout" in (r.error or "")


# ---------------------------------------------------------------------------
# run_batch — mixed results
# ---------------------------------------------------------------------------

def test_run_batch_mixed_results(tmp_path: Path) -> None:
    entry_a = _minimal_entry_dict(name="alpha")
    entry_b = _minimal_entry_dict(name="beta")
    manifest = BatchManifest(
        migrations=[BatchEntry(**entry_a), BatchEntry(**entry_b)]
    )
    mock_result = _make_migration_result()

    call_count = 0

    def side_effect_run():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_result
        raise RuntimeError("beta exploded")

    with (
        patch("rosettastone.config.MigrationConfig"),
        patch("rosettastone.core.migrator.Migrator") as mock_migrator,
    ):
        mock_migrator.return_value.run.side_effect = side_effect_run

        results = run_batch(manifest, tmp_path)

    assert len(results) == 2
    assert results[0].status == "complete"
    assert results[1].status == "failed"


# ---------------------------------------------------------------------------
# run_batch — continues on failure
# ---------------------------------------------------------------------------

def test_run_batch_continues_on_failure(tmp_path: Path) -> None:
    entry_a = _minimal_entry_dict(name="first")
    entry_b = _minimal_entry_dict(name="second")
    manifest = BatchManifest(
        migrations=[BatchEntry(**entry_a), BatchEntry(**entry_b)]
    )
    mock_result = _make_migration_result()

    call_count = 0

    def side_effect_run():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("first entry fails")
        return mock_result

    with (
        patch("rosettastone.config.MigrationConfig"),
        patch("rosettastone.core.migrator.Migrator") as mock_migrator,
    ):
        mock_migrator.return_value.run.side_effect = side_effect_run

        results = run_batch(manifest, tmp_path)

    # Both entries must be present in results
    assert len(results) == 2
    names = [r.name for r in results]
    assert "first" in names
    assert "second" in names

    second = next(r for r in results if r.name == "second")
    assert second.status == "complete"


# ---------------------------------------------------------------------------
# format_batch_summary — basic table structure
# ---------------------------------------------------------------------------

def test_format_batch_summary_basic() -> None:
    results = [
        BatchResult(
            name="proj-a",
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
            recommendation="GO",
            confidence=0.92,
        )
    ]
    summary = format_batch_summary(results)

    assert "Name" in summary
    assert "Source → Target" in summary
    assert "Status" in summary
    assert "Recommendation" in summary
    assert "Confidence" in summary
    assert "proj-a" in summary
    assert "GO" in summary


# ---------------------------------------------------------------------------
# format_batch_summary — aggregate footer counts
# ---------------------------------------------------------------------------

def test_format_batch_summary_aggregates() -> None:
    results = [
        BatchResult(
            name="a",
            source_model="m1",
            target_model="m2",
            status="complete",
            recommendation="GO",
            confidence=0.9,
        ),
        BatchResult(
            name="b",
            source_model="m1",
            target_model="m2",
            status="complete",
            recommendation="GO",
            confidence=0.85,
        ),
        BatchResult(
            name="c",
            source_model="m1",
            target_model="m2",
            status="failed",
            error="timeout",
        ),
    ]
    summary = format_batch_summary(results)

    assert "2 GO" in summary
    assert "1 failed" in summary


# ---------------------------------------------------------------------------
# format_batch_summary — empty results
# ---------------------------------------------------------------------------

def test_format_batch_summary_empty() -> None:
    summary = format_batch_summary([])
    assert "No results." in summary


# ---------------------------------------------------------------------------
# BatchManifest — version defaults to 1
# ---------------------------------------------------------------------------

def test_batch_manifest_version() -> None:
    manifest = BatchManifest(
        migrations=[BatchEntry(**_minimal_entry_dict())]
    )
    assert manifest.version == 1
