"""Tests for the annotation queue API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from rosettastone.server.app import create_app
from rosettastone.server.database import get_session
from rosettastone.server.models import MigrationRecord

_JWT_SECRET = "test-annotations-secret-long-enough-hmac"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_token(user_id: int = 1, role: str = "admin") -> str:
    from rosettastone.server.auth_utils import create_jwt

    return create_jwt(user_id=user_id, role=role, secret=_JWT_SECRET)


def _make_app_and_engine():
    engine = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    app = create_app()

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    return app, engine


def _insert_migration(engine) -> int:
    with Session(engine) as session:
        migration = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
        )
        session.add(migration)
        session.commit()
        session.refresh(migration)
        return migration.id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _engine():
    engine = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def client(monkeypatch):
    """TestClient with ROSETTASTONE_MULTI_USER=true and admin JWT."""
    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", _JWT_SECRET)
    monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)

    app, engine = _make_app_and_engine()
    token = _make_token(role="admin")
    with TestClient(app, headers={"Authorization": f"Bearer {token}"}) as c:
        yield c, engine


@pytest.fixture
def migration_id(client):
    c, engine = client
    return _insert_migration(engine)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------


def _create_annotation(
    client: TestClient,
    migration_id: int,
    annotation_type: str = "regression",
    text: str = "This is a regression.",
    test_case_id: int | None = None,
) -> dict:
    payload = {
        "migration_id": migration_id,
        "annotation_type": annotation_type,
        "text": text,
    }
    if test_case_id is not None:
        payload["test_case_id"] = test_case_id
    return client.post(f"/api/v1/migrations/{migration_id}/annotations", json=payload)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_create_annotation_201(client, migration_id):
    """POST /api/v1/migrations/{id}/annotations creates annotation and returns 201."""
    c, _ = client
    resp = _create_annotation(c, migration_id)
    assert resp.status_code == 201
    data = resp.json()
    assert data["annotation_type"] == "regression"
    assert data["migration_id"] == migration_id
    assert "id" in data


def test_create_annotation_improvement_type(client, migration_id):
    """Annotations with type 'improvement' are accepted."""
    c, _ = client
    resp = _create_annotation(c, migration_id, annotation_type="improvement", text="Better!")
    assert resp.status_code == 201
    assert resp.json()["annotation_type"] == "improvement"


def test_create_annotation_invalid_type_422(client, migration_id):
    """An invalid annotation_type returns 422."""
    c, _ = client
    resp = _create_annotation(c, migration_id, annotation_type="bad_type")
    assert resp.status_code == 422


def test_create_annotation_migration_not_found_404(client):
    """Creating annotation for a non-existent migration returns 404."""
    c, _ = client
    resp = _create_annotation(c, migration_id=9999)
    assert resp.status_code == 404


def test_list_annotations_for_migration(client, migration_id):
    """GET /api/v1/migrations/{id}/annotations lists annotations for a migration."""
    c, _ = client
    _create_annotation(c, migration_id, annotation_type="regression")
    _create_annotation(c, migration_id, annotation_type="edge_case")

    resp = c.get(f"/api/v1/migrations/{migration_id}/annotations")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    types = {a["annotation_type"] for a in data}
    assert "regression" in types
    assert "edge_case" in types


def test_list_annotations_migration_not_found_404(client):
    """Listing annotations for a non-existent migration returns 404."""
    c, _ = client
    resp = c.get("/api/v1/migrations/9999/annotations")
    assert resp.status_code == 404


def test_annotation_queue_returns_annotations(client, migration_id):
    """GET /api/v1/annotations/queue returns annotations for an authenticated admin."""
    c, _ = client
    _create_annotation(c, migration_id)

    resp = c.get("/api/v1/annotations/queue")
    assert resp.status_code == 200
    data = resp.json()
    # Admin receives all annotations
    assert isinstance(data, list)
    assert len(data) >= 1


def test_endpoints_return_404_without_multi_user(monkeypatch):
    """Annotation endpoints return 404 when ROSETTASTONE_MULTI_USER is not set."""
    monkeypatch.delenv("ROSETTASTONE_MULTI_USER", raising=False)
    monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)

    engine = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    app = create_app()

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    mid = _insert_migration(engine)
    with TestClient(app) as c:
        resp = c.get(f"/api/v1/migrations/{mid}/annotations")
        assert resp.status_code == 404
