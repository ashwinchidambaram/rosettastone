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


# ===========================================================================
# Task 6.3 — Health probe extended tests
# ===========================================================================


def test_liveness_always_200(client):
    """GET /api/v1/health/live always returns 200 while the process is alive.

    The liveness probe must never return a non-200 status — it is used by
    Kubernetes to decide whether to restart the container. Failures of
    optional components (DB, Redis, task worker) must not affect liveness.
    """
    resp = client.get("/api/v1/health/live")
    assert resp.status_code == 200, f"Liveness probe must always return 200, got {resp.status_code}"
    body = resp.json()
    assert body.get("status") == "ok", f"Liveness body must have status='ok', got: {body!r}"


def test_readiness_returns_component_status(client):
    """GET /api/v1/health/ready returns JSON with per-component health details.

    The readiness response must include a 'components' dict with at least
    a 'database' key, and each component must have a 'status' sub-key.
    This allows orchestrators to diagnose which dependency is unhealthy.
    """
    resp = client.get("/api/v1/health/ready")
    # Status code is 200 (healthy/degraded) or 503 (DB unavailable) — either is acceptable
    assert resp.status_code in (200, 503), (
        f"Readiness probe must return 200 or 503, got {resp.status_code}"
    )
    data = resp.json()
    assert "components" in data, f"Readiness response must include 'components', got: {data!r}"
    components = data["components"]
    assert isinstance(components, dict), f"'components' must be a dict, got: {type(components)}"
    # At minimum database must be reported
    assert "database" in components, (
        f"'database' must be present in components, got keys: {list(components.keys())}"
    )
    db_component = components["database"]
    assert "status" in db_component, (
        f"database component must have a 'status' key, got: {db_component!r}"
    )
    assert db_component["status"] in ("ok", "unavailable", "degraded"), (
        f"database status must be one of ok/unavailable/degraded, got: {db_component['status']!r}"
    )


def test_health_endpoint_returns_200(client):
    """GET /api/v1/health returns HTTP 200 with a well-formed JSON body.

    The basic health endpoint must always return 200 (it is used for backward
    compatibility and does not gate traffic like the readiness probe).
    """
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200, f"Basic health endpoint must return 200, got {resp.status_code}"
    data = resp.json()
    assert isinstance(data, dict), f"Health response must be a JSON object, got: {type(data)}"
    # Must be parseable — no exception means we're good
    assert len(data) > 0, "Health response body must not be empty"


def test_health_includes_version_or_status(client):
    """GET /api/v1/health response body contains at least a 'status' field.

    Callers rely on the 'status' field to determine overall health at a glance.
    Future enhancements may add 'version' — this test documents the minimum
    contract so regressions are caught immediately.
    """
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200, (
        f"Health endpoint must return 200 before checking body, got {resp.status_code}"
    )
    data = resp.json()
    has_status = "status" in data
    has_version = "version" in data
    assert has_status or has_version, (
        f"Health response must contain 'status' or 'version' field, got keys: {list(data.keys())}"
    )
    if has_status:
        assert isinstance(data["status"], str), (
            f"'status' field must be a string, got: {type(data['status'])}"
        )
    if has_version:
        assert isinstance(data["version"], str), (
            f"'version' field must be a string, got: {type(data['version'])}"
        )
