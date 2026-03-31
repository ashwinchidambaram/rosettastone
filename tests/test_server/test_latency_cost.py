"""Tests for latency/cost storage, retrieval, and deprecation monitoring."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session

from rosettastone.server.models import Alert, MigrationRecord, RegisteredModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_migration(session: Session, **kwargs) -> MigrationRecord:
    """Insert a minimal MigrationRecord, optionally overriding fields."""
    defaults = dict(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        status="complete",
        confidence_score=0.90,
        baseline_score=0.85,
        improvement=0.05,
        cost_usd=0.50,
        duration_seconds=30.0,
        recommendation="GO",
        recommendation_reasoning="Thresholds met.",
        config_json=json.dumps(
            {"source_model": "openai/gpt-4o", "target_model": "anthropic/claude-sonnet-4"}
        ),
        per_type_scores_json=json.dumps(
            {
                "json": {
                    "win_rate": 0.92,
                    "mean": 0.91,
                    "median": 0.92,
                    "p10": 0.85,
                    "p50": 0.92,
                    "p90": 0.97,
                    "min_score": 0.80,
                    "max_score": 1.0,
                    "sample_count": 15,
                    "confidence_interval": [0.85, 0.98],
                }
            }
        ),
        warnings_json=json.dumps([]),
        safety_warnings_json=json.dumps([]),
    )
    defaults.update(kwargs)
    record = MigrationRecord(**defaults)
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def _register_model(session: Session, model_id: str) -> RegisteredModel:
    """Insert a RegisteredModel with the given model_id."""
    model = RegisteredModel(model_id=model_id)
    session.add(model)
    session.commit()
    session.refresh(model)
    return model


# ---------------------------------------------------------------------------
# Test 1: MigrationRecord persists latency columns
# ---------------------------------------------------------------------------


def test_migration_record_latency_columns(session: Session) -> None:
    """Create MigrationRecord with latency fields set, verify they persist."""
    record = _make_migration(
        session,
        source_latency_p50=0.42,
        source_latency_p95=0.91,
        target_latency_p50=0.31,
        target_latency_p95=0.77,
    )

    fetched = session.get(MigrationRecord, record.id)
    assert fetched is not None
    assert fetched.source_latency_p50 == pytest.approx(0.42)
    assert fetched.source_latency_p95 == pytest.approx(0.91)
    assert fetched.target_latency_p50 == pytest.approx(0.31)
    assert fetched.target_latency_p95 == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# Test 2: Latency columns are nullable when not provided
# ---------------------------------------------------------------------------


def test_migration_record_latency_columns_nullable(session: Session) -> None:
    """Create MigrationRecord without latency fields — they should be None."""
    record = _make_migration(session)

    fetched = session.get(MigrationRecord, record.id)
    assert fetched is not None
    assert fetched.source_latency_p50 is None
    assert fetched.source_latency_p95 is None
    assert fetched.target_latency_p50 is None
    assert fetched.target_latency_p95 is None


# ---------------------------------------------------------------------------
# Test 3: MigrationRecord persists cost projection columns
# ---------------------------------------------------------------------------


def test_migration_record_cost_columns(session: Session) -> None:
    """Create MigrationRecord with cost projection fields set, verify persistence."""
    record = _make_migration(
        session,
        projected_source_cost_per_call=0.002500,
        projected_target_cost_per_call=0.001875,
    )

    fetched = session.get(MigrationRecord, record.id)
    assert fetched is not None
    assert fetched.projected_source_cost_per_call == pytest.approx(0.002500)
    assert fetched.projected_target_cost_per_call == pytest.approx(0.001875)


# ---------------------------------------------------------------------------
# Test 4: API detail endpoint includes latency fields when set
# ---------------------------------------------------------------------------


def test_migration_detail_includes_latency(client, engine) -> None:
    """Create a MigrationRecord with latency data, call the API endpoint, verify response."""
    with Session(engine) as session:
        record = _make_migration(
            session,
            source_latency_p50=0.55,
            source_latency_p95=1.10,
            target_latency_p50=0.40,
            target_latency_p95=0.85,
            projected_source_cost_per_call=0.003,
            projected_target_cost_per_call=0.002,
        )
        migration_id = record.id

    response = client.get(f"/api/v1/migrations/{migration_id}")
    assert response.status_code == 200
    data = response.json()

    assert data["source_latency_p50"] == pytest.approx(0.55)
    assert data["source_latency_p95"] == pytest.approx(1.10)
    assert data["target_latency_p50"] == pytest.approx(0.40)
    assert data["target_latency_p95"] == pytest.approx(0.85)
    assert data["projected_source_cost_per_call"] == pytest.approx(0.003)
    assert data["projected_target_cost_per_call"] == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# Test 5: API detail endpoint returns null latency when not set
# ---------------------------------------------------------------------------


def test_migration_detail_latency_null_when_not_set(client, engine) -> None:
    """Migration without latency data — API response should have null latency fields."""
    with Session(engine) as session:
        record = _make_migration(session)
        migration_id = record.id

    response = client.get(f"/api/v1/migrations/{migration_id}")
    assert response.status_code == 200
    data = response.json()

    assert data["source_latency_p50"] is None
    assert data["source_latency_p95"] is None
    assert data["target_latency_p50"] is None
    assert data["target_latency_p95"] is None
    assert data["projected_source_cost_per_call"] is None
    assert data["projected_target_cost_per_call"] is None


# ---------------------------------------------------------------------------
# Test 6: _migration_to_template_dict includes latency/cost keys
# ---------------------------------------------------------------------------


def test_template_dict_includes_latency(session: Session) -> None:
    """Verify _migration_to_template_dict includes latency and cost keys in output."""
    from rosettastone.server.api.migrations import _migration_to_template_dict

    record = _make_migration(
        session,
        source_latency_p50=0.33,
        source_latency_p95=0.70,
        target_latency_p50=0.25,
        target_latency_p95=0.60,
        projected_source_cost_per_call=0.0025,
        projected_target_cost_per_call=0.0015,
    )

    result = _migration_to_template_dict(record, session)

    assert "source_latency_p50" in result
    assert "source_latency_p95" in result
    assert "target_latency_p50" in result
    assert "target_latency_p95" in result
    assert "projected_source_cost_per_call" in result
    assert "projected_target_cost_per_call" in result

    assert result["source_latency_p50"] == pytest.approx(0.33)
    assert result["source_latency_p95"] == pytest.approx(0.70)
    assert result["target_latency_p50"] == pytest.approx(0.25)
    assert result["target_latency_p95"] == pytest.approx(0.60)
    assert result["projected_source_cost_per_call"] == pytest.approx(0.0025)
    assert result["projected_target_cost_per_call"] == pytest.approx(0.0015)


# ---------------------------------------------------------------------------
# Test 7: _sample_latency returns None when litellm not available
# ---------------------------------------------------------------------------


def test_sample_latency_no_litellm() -> None:
    """_sample_latency should return None when litellm is not importable."""
    from rosettastone.server.api.tasks import _sample_latency

    prompt_pair = MagicMock()
    prompt_pair.prompt = "Hello, world"

    eval_result = MagicMock()
    eval_result.prompt_pair = prompt_pair

    result = MagicMock()
    result.validation_results = [eval_result]

    config = MagicMock()
    config.source_model = "openai/gpt-4o"
    config.target_model = "anthropic/claude-sonnet-4"

    with patch.dict("sys.modules", {"litellm": None}):
        output = _sample_latency(result, config)

    assert output is None


# ---------------------------------------------------------------------------
# Test 8: _estimate_per_call_cost returns None when litellm not available
# ---------------------------------------------------------------------------


def test_estimate_per_call_cost_no_litellm() -> None:
    """_estimate_per_call_cost should return None when litellm is not importable."""
    from rosettastone.server.api.tasks import _estimate_per_call_cost

    config = MagicMock()
    config.source_model = "openai/gpt-4o"
    config.target_model = "anthropic/claude-sonnet-4"

    with patch.dict("sys.modules", {"litellm": None}):
        output = _estimate_per_call_cost(config)

    assert output is None


# ---------------------------------------------------------------------------
# Test 9: check_deprecations creates critical alert within 30 days
# ---------------------------------------------------------------------------


def test_check_deprecations_within_30_days(session: Session) -> None:
    """Model deprecated within 30 days should produce a critical alert."""
    from sqlmodel import select

    from rosettastone.server.api.deprecation import check_deprecations

    model_id = "openai/gpt-4o-0613"
    _register_model(session, model_id)

    dep_date = (datetime.now(UTC) + timedelta(days=15)).strftime("%Y-%m-%d")
    patched = {model_id: {"date": dep_date, "replacement": "openai/gpt-4o"}}

    with patch("rosettastone.server.api.deprecation.KNOWN_DEPRECATIONS", patched):
        count = check_deprecations(session)

    assert count == 1
    alert = session.exec(
        select(Alert).where(Alert.alert_type == "deprecation", Alert.model_id == model_id)
    ).first()
    assert alert is not None
    assert alert.severity == "critical"


# ---------------------------------------------------------------------------
# Test 10: check_deprecations creates warning alert within 90 days
# ---------------------------------------------------------------------------


def test_check_deprecations_within_90_days(session: Session) -> None:
    """Model deprecated within 60 days (but >30) should produce a warning alert."""
    from sqlmodel import select

    from rosettastone.server.api.deprecation import check_deprecations

    model_id = "openai/gpt-4-0613"
    _register_model(session, model_id)

    dep_date = (datetime.now(UTC) + timedelta(days=60)).strftime("%Y-%m-%d")
    patched = {model_id: {"date": dep_date, "replacement": "openai/gpt-4o"}}

    with patch("rosettastone.server.api.deprecation.KNOWN_DEPRECATIONS", patched):
        count = check_deprecations(session)

    assert count == 1
    alert = session.exec(
        select(Alert).where(Alert.alert_type == "deprecation", Alert.model_id == model_id)
    ).first()
    assert alert is not None
    assert alert.severity == "warning"


# ---------------------------------------------------------------------------
# Test 11: check_deprecations creates no alert beyond 90 days
# ---------------------------------------------------------------------------


def test_check_deprecations_beyond_90_days(session: Session) -> None:
    """Model deprecated 120 days from now should produce no alert."""
    from sqlmodel import select

    from rosettastone.server.api.deprecation import check_deprecations

    model_id = "openai/gpt-4o-0613"
    _register_model(session, model_id)

    dep_date = (datetime.now(UTC) + timedelta(days=120)).strftime("%Y-%m-%d")
    patched = {model_id: {"date": dep_date, "replacement": "openai/gpt-4o"}}

    with patch("rosettastone.server.api.deprecation.KNOWN_DEPRECATIONS", patched):
        count = check_deprecations(session)

    assert count == 0
    alert = session.exec(
        select(Alert).where(Alert.alert_type == "deprecation", Alert.model_id == model_id)
    ).first()
    assert alert is None


# ---------------------------------------------------------------------------
# Test 12: check_deprecations is idempotent (no duplicate alerts)
# ---------------------------------------------------------------------------


def test_check_deprecations_idempotent(session: Session) -> None:
    """Calling check_deprecations twice should only create one alert, not two."""
    from sqlmodel import func, select

    from rosettastone.server.api.deprecation import check_deprecations

    model_id = "openai/gpt-4o-0613"
    _register_model(session, model_id)

    dep_date = (datetime.now(UTC) + timedelta(days=10)).strftime("%Y-%m-%d")
    patched = {model_id: {"date": dep_date, "replacement": "openai/gpt-4o"}}

    with patch("rosettastone.server.api.deprecation.KNOWN_DEPRECATIONS", patched):
        first = check_deprecations(session)
        second = check_deprecations(session)

    assert first == 1
    assert second == 0

    count = session.exec(
        select(func.count())
        .select_from(Alert)
        .where(Alert.alert_type == "deprecation", Alert.model_id == model_id)
    ).one()
    assert count == 1


# ---------------------------------------------------------------------------
# Test 13: check_deprecations picks up entries from custom JSON file
# ---------------------------------------------------------------------------


def test_check_deprecations_custom_json(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """ROSETTASTONE_DEPRECATIONS_JSON env var with custom data creates alerts."""
    from sqlmodel import select

    from rosettastone.server.api.deprecation import check_deprecations

    custom_model_id = "custom/my-model-v1"
    _register_model(session, custom_model_id)

    dep_date = (datetime.now(UTC) + timedelta(days=20)).strftime("%Y-%m-%d")
    custom_data = {custom_model_id: {"date": dep_date, "replacement": "custom/my-model-v2"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump(custom_data, fh)
        tmp_path = fh.name

    try:
        monkeypatch.setenv("ROSETTASTONE_DEPRECATIONS_JSON", tmp_path)
        count = check_deprecations(session)
    finally:
        os.unlink(tmp_path)

    assert count == 1
    alert = session.exec(
        select(Alert).where(Alert.alert_type == "deprecation", Alert.model_id == custom_model_id)
    ).first()
    assert alert is not None
    assert alert.severity == "critical"
    assert "my-model-v2" in alert.action
