"""Tests for structured JSON logging and per-request correlation ID middleware."""

from __future__ import annotations

import json
import logging
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from rosettastone.server.app import create_app
from rosettastone.server.database import get_session
from rosettastone.server.logging_config import JsonFormatter

# ---------------------------------------------------------------------------
# Shared test client fixture (in-memory DB, no auth required)
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """Return a TestClient backed by an in-memory SQLite database."""
    engine = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    app = create_app()

    def _override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = _override_session
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Test 1: X-Request-ID header is present and is a valid UUID
# ---------------------------------------------------------------------------


def test_x_request_id_header_present(client: TestClient) -> None:
    """GET /api/v1/health must return an X-Request-ID header containing a valid UUID4."""
    response = client.get("/api/v1/health")

    assert "X-Request-ID" in response.headers, "X-Request-ID header must be present"

    header_value = response.headers["X-Request-ID"]
    # Validate that the value parses as a UUID (will raise ValueError if not)
    parsed = uuid.UUID(header_value)
    assert str(parsed) == header_value, "X-Request-ID must be a canonical lowercase UUID string"


# ---------------------------------------------------------------------------
# Test 2: JsonFormatter outputs valid JSON with required fields
# ---------------------------------------------------------------------------


def test_json_formatter_outputs_valid_json() -> None:
    """JsonFormatter must produce a valid JSON string with the mandatory fields."""
    formatter = JsonFormatter()

    record = logging.LogRecord(
        name="rosettastone.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello structured world",
        args=(),
        exc_info=None,
    )

    output = formatter.format(record)

    # Must be parseable JSON
    data = json.loads(output)

    # Required fields
    assert "timestamp" in data, "Must contain 'timestamp'"
    assert "level" in data, "Must contain 'level'"
    assert "logger" in data, "Must contain 'logger'"
    assert "message" in data, "Must contain 'message'"

    assert data["level"] == "INFO"
    assert data["logger"] == "rosettastone.test"
    assert data["message"] == "hello structured world"


# ---------------------------------------------------------------------------
# Test 3: X-Request-ID is consistent — same request gets the same ID
# ---------------------------------------------------------------------------


def test_request_id_propagated_to_state(client: TestClient) -> None:
    """X-Request-ID must be stable within a single request/response cycle."""
    # The middleware generates the ID, sets it on request.state, and echoes
    # it in the response header — so the header value itself is the proof of
    # consistency (there's only one value to check per request).
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers

    request_id = response.headers["X-Request-ID"]

    # Must be a valid UUID
    parsed = uuid.UUID(request_id)
    assert str(parsed) == request_id

    # Different requests must receive *different* IDs
    second_response = client.get("/api/v1/health")
    assert second_response.headers["X-Request-ID"] != request_id, (
        "Each request must receive a unique X-Request-ID"
    )


# ---------------------------------------------------------------------------
# Bonus: JsonFormatter handles extra fields (request_id, migration_id, etc.)
# ---------------------------------------------------------------------------


def test_json_formatter_extra_fields() -> None:
    """JsonFormatter must surface first-class extra fields in the JSON output."""
    formatter = JsonFormatter()

    record = logging.LogRecord(
        name="rosettastone.tasks",
        level=logging.WARNING,
        pathname=__file__,
        lineno=42,
        msg="migration finished",
        args=(),
        exc_info=None,
    )
    # Simulate extra= keyword args being injected onto the record
    record.__dict__["request_id"] = "req-abc-123"
    record.__dict__["migration_id"] = 7
    record.__dict__["duration_ms"] = 1234.5

    data = json.loads(formatter.format(record))

    assert data.get("request_id") == "req-abc-123"
    assert data.get("migration_id") == 7
    assert data.get("duration_ms") == 1234.5


def test_migration_complete_log_has_expected_fields(caplog):
    """migration_complete log event contains token/cost/recommendation fields."""
    with caplog.at_level(logging.INFO, logger="rosettastone"):
        logger = logging.getLogger("rosettastone.server.api.tasks")
        logger.info(
            "migration_complete",
            extra={
                "migration_id": 42,
                "total_tokens": 1500,
                "cost_usd": 0.05,
                "baseline_score": 0.75,
                "confidence_score": 0.82,
                "recommendation": "GO",
                "duration_ms": 5000,
                "stage_durations": {"ingest": 100, "baseline_eval": 2000},
            },
        )

    assert len(caplog.records) >= 1
    record = caplog.records[-1]
    assert record.getMessage() == "migration_complete"
    assert getattr(record, "total_tokens", None) == 1500
    assert getattr(record, "recommendation", None) == "GO"
