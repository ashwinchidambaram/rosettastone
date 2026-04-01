"""Tests for registered model API endpoints and UI integration."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
sqlmodel = pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient  # noqa: E402
from sqlmodel import Session  # noqa: E402

from rosettastone.server.models import MigrationRecord  # noqa: E402

# ---------------------------------------------------------------------------
# POST /api/v1/models  — register a model
# ---------------------------------------------------------------------------


class TestRegisterModel:
    def test_register_model_returns_201(self, client: TestClient) -> None:
        resp = client.post("/api/v1/models", json={"model_id": "openai/gpt-4o"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "openai/gpt-4o"
        assert data["db_id"] is not None
        assert data["provider"] == "Openai"
        assert data["status"] == "active"
        assert "context" in data
        assert "cost_per_1m" in data

    def test_register_model_missing_model_id(self, client: TestClient) -> None:
        resp = client.post("/api/v1/models", json={})
        assert resp.status_code == 422

    def test_register_model_empty_model_id(self, client: TestClient) -> None:
        resp = client.post("/api/v1/models", json={"model_id": "  "})
        assert resp.status_code == 422

    def test_duplicate_registration_returns_409(self, client: TestClient) -> None:
        client.post("/api/v1/models", json={"model_id": "openai/gpt-4o-mini"})
        resp = client.post("/api/v1/models", json={"model_id": "openai/gpt-4o-mini"})
        assert resp.status_code == 409
        assert "already registered" in resp.json()["detail"]

    def test_register_model_no_slash_provider(self, client: TestClient) -> None:
        """Model IDs without a slash should have provider 'Unknown'."""
        resp = client.post("/api/v1/models", json={"model_id": "gpt-4"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["provider"] == "Unknown"


# ---------------------------------------------------------------------------
# GET /api/v1/models  — list models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_list_empty(self, client: TestClient) -> None:
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_after_registration(self, client: TestClient) -> None:
        client.post("/api/v1/models", json={"model_id": "openai/gpt-4o"})
        client.post("/api/v1/models", json={"model_id": "anthropic/claude-sonnet-4"})
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        ids = [m["id"] for m in data]
        assert "openai/gpt-4o" in ids
        assert "anthropic/claude-sonnet-4" in ids

    def test_list_response_shape(self, client: TestClient) -> None:
        client.post("/api/v1/models", json={"model_id": "openai/gpt-4o"})
        resp = client.get("/api/v1/models")
        item = resp.json()[0]
        assert "id" in item
        assert "db_id" in item
        assert "provider" in item
        assert "status" in item
        assert "context" in item
        assert "cost_per_1m" in item


# ---------------------------------------------------------------------------
# DELETE /api/v1/models/{model_db_id}  — remove a model
# ---------------------------------------------------------------------------


class TestDeleteModel:
    def test_delete_model_returns_200(self, client: TestClient) -> None:
        reg = client.post("/api/v1/models", json={"model_id": "openai/gpt-4o"})
        db_id = reg.json()["db_id"]
        resp = client.delete(f"/api/v1/models/{db_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert resp.json()["model_id"] == "openai/gpt-4o"

    def test_delete_removes_from_list(self, client: TestClient) -> None:
        reg = client.post("/api/v1/models", json={"model_id": "openai/gpt-4o"})
        db_id = reg.json()["db_id"]
        client.delete(f"/api/v1/models/{db_id}")
        resp = client.get("/api/v1/models")
        assert resp.json() == []

    def test_delete_nonexistent_returns_404(self, client: TestClient) -> None:
        resp = client.delete("/api/v1/models/99999")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/v1/models/import-from-migrations
# ---------------------------------------------------------------------------


class TestImportFromMigrations:
    def test_import_from_migrations_with_existing_records(self, client: TestClient, engine) -> None:
        # Seed migration records directly in DB
        with Session(engine) as session:
            for source, target in [
                ("openai/gpt-4o", "anthropic/claude-sonnet-4"),
                ("openai/gpt-4o", "openai/gpt-4o-mini"),  # duplicate source
            ]:
                session.add(
                    MigrationRecord(
                        source_model=source,
                        target_model=target,
                        status="complete",
                    )
                )
            session.commit()

        resp = client.post("/api/v1/models/import-from-migrations")
        assert resp.status_code == 200
        data = resp.json()
        # 3 unique models: openai/gpt-4o, anthropic/claude-sonnet-4, openai/gpt-4o-mini
        assert data["imported"] == 3
        assert data["total_registered"] == 3

    def test_import_skips_already_registered(self, client: TestClient, engine) -> None:
        # Pre-register one model
        client.post("/api/v1/models", json={"model_id": "openai/gpt-4o"})

        with Session(engine) as session:
            session.add(
                MigrationRecord(
                    source_model="openai/gpt-4o",
                    target_model="anthropic/claude-sonnet-4",
                    status="complete",
                )
            )
            session.commit()

        resp = client.post("/api/v1/models/import-from-migrations")
        assert resp.status_code == 200
        data = resp.json()
        # Only claude-sonnet-4 should be new
        assert data["imported"] == 1
        assert data["total_registered"] == 2

    def test_import_no_migrations_returns_zero(self, client: TestClient) -> None:
        resp = client.post("/api/v1/models/import-from-migrations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["imported"] == 0
        assert data["total_registered"] == 0


# ---------------------------------------------------------------------------
# GET /api/v1/models/{model_db_id}/info
# ---------------------------------------------------------------------------


class TestGetModelInfo:
    def test_info_returns_dict(self, client: TestClient) -> None:
        reg = client.post("/api/v1/models", json={"model_id": "openai/gpt-4o"})
        db_id = reg.json()["db_id"]
        resp = client.get(f"/api/v1/models/{db_id}/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "context" in data
        assert "cost_per_1m" in data
        assert "provider" in data

    def test_info_nonexistent_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/v1/models/99999/info")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# UI /ui/  — integration with real DB
# ---------------------------------------------------------------------------


class TestUIModelsIntegration:
    def test_ui_shows_empty_state_when_no_models_registered(self, client: TestClient) -> None:
        """With no registered models, /ui/ should show the empty state page."""
        resp = client.get("/ui/")
        assert resp.status_code == 200
        body = resp.text
        assert "Welcome to RosettaStone" in body

    def test_ui_shows_model_cards_after_registration(self, client: TestClient) -> None:
        """After registering models, /ui/ should show model cards."""
        client.post("/api/v1/models", json={"model_id": "openai/gpt-4o"})
        client.post("/api/v1/models", json={"model_id": "anthropic/claude-sonnet-4"})
        resp = client.get("/ui/")
        assert resp.status_code == 200
        body = resp.text
        assert "Your models" in body
        assert "openai/gpt-4o" in body
        assert "anthropic/claude-sonnet-4" in body

    def test_ui_false_empty_param_forces_model_view_with_dummy_fallback(
        self, client: TestClient
    ) -> None:
        """?empty=false should skip the empty-state check and show model cards (DUMMY fallback)."""
        resp = client.get("/ui/?empty=false")
        assert resp.status_code == 200
        body = resp.text
        assert "Your models" in body

    def test_ui_single_model_shows_active_status(self, client: TestClient) -> None:
        client.post("/api/v1/models", json={"model_id": "openai/gpt-4o-mini"})
        resp = client.get("/ui/")
        assert resp.status_code == 200
        body = resp.text
        assert "Active" in body
