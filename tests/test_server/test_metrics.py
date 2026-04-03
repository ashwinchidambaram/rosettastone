"""Tests for Prometheus metrics endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rosettastone.server.app import create_app

# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


def test_metrics_unauthenticated_when_no_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /metrics is accessible without credentials when ROSETTASTONE_API_KEY is unset."""
    monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)
    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/metrics")
    # Auth is disabled — endpoint responds (200 or 404 for missing prometheus), not 401
    assert response.status_code != 401


def test_metrics_requires_auth_when_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /metrics returns 401 when ROSETTASTONE_API_KEY is set and no key is provided."""
    monkeypatch.setenv("ROSETTASTONE_API_KEY", "secret")
    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/metrics")
    assert response.status_code == 401


def test_metrics_accessible_with_valid_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /metrics succeeds with a valid Bearer token when ROSETTASTONE_API_KEY is set."""
    monkeypatch.setenv("ROSETTASTONE_API_KEY", "secret")
    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/metrics", headers={"Authorization": "Bearer secret"})
    # Auth passes; endpoint responds (200 or 404 for missing prometheus), not 401
    assert response.status_code != 401


def test_metrics_rejected_with_wrong_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /metrics returns 401 with an invalid Bearer token."""
    monkeypatch.setenv("ROSETTASTONE_API_KEY", "secret")
    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/metrics", headers={"Authorization": "Bearer wrongkey"})
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


def test_metrics_endpoint_returns_404_without_prometheus(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /metrics returns 404 when prometheus_client is not available."""
    # Mock the availability check to return False
    import rosettastone.server.metrics

    monkeypatch.setattr(rosettastone.server.metrics, "_PROMETHEUS_AVAILABLE", False)

    response = client.get("/metrics")
    assert response.status_code == 404
    assert "prometheus_client not installed" in response.json()["detail"]


def test_metrics_endpoint_returns_200_with_prometheus(client: TestClient) -> None:
    """GET /metrics returns 200 and proper content-type when prometheus_client is available."""
    pytest.importorskip("prometheus_client")

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")


def test_migrations_total_counter_exists() -> None:
    """MIGRATIONS_TOTAL counter exists when prometheus_client is available."""
    pytest.importorskip("prometheus_client")

    from rosettastone.server.metrics import MIGRATIONS_TOTAL

    assert MIGRATIONS_TOTAL is not None
    # Verify it's a Counter with the expected label names
    assert hasattr(MIGRATIONS_TOTAL, "_labelnames")
    assert set(MIGRATIONS_TOTAL._labelnames) == {"status", "source_model", "target_model"}
