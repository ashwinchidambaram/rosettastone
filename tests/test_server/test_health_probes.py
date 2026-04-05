"""Tests for health probe endpoints."""
import pytest


@pytest.fixture
def client():
    """FastAPI test client with lifespan disabled for health tests."""
    from fastapi.testclient import TestClient
    from rosettastone.server.app import create_app

    app = create_app()
    # Use TestClient without lifespan to avoid full startup (no task worker needed for probes)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_live_always_200(client):
    """GET /api/v1/health/live always returns 200."""
    resp = client.get("/api/v1/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_backward_compat(client):
    """GET /api/v1/health still returns 200 (backward compatibility)."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    # Should have a status key
    data = resp.json()
    assert "status" in data


def test_health_ready_returns_200_or_503(client):
    """GET /api/v1/health/ready returns 200 or 503 depending on DB state."""
    resp = client.get("/api/v1/health/ready")
    # In test environment DB is SQLite — should succeed
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert "status" in data
    assert "components" in data
    assert "database" in data["components"]


def test_health_ready_response_shape(client):
    """Response has expected shape with status and components."""
    resp = client.get("/api/v1/health/ready")
    data = resp.json()
    assert data["status"] in ("ok", "degraded", "unavailable")
    comps = data["components"]
    assert "database" in comps
    assert comps["database"]["status"] in ("ok", "unavailable", "degraded")
