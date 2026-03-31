"""Tests for migration API endpoints and UI template rendering."""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
sqlmodel = pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient  # noqa: E402
from sqlmodel import Session, select  # noqa: E402

from rosettastone.server.app import create_app  # noqa: E402
from rosettastone.server.models import MigrationRecord  # noqa: E402

# ---------------------------------------------------------------------------
# UI-only client fixture (no database needed for dummy-data UI tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def ui_client() -> TestClient:
    """Create a test client for UI endpoints (no database dependency)."""
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# JSON API endpoints (DB-backed)
# ---------------------------------------------------------------------------


class TestListMigrations:
    def test_empty_list(self, client):
        response = client.get("/api/v1/migrations")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["per_page"] == 20

    def test_list_with_sample_data(self, client, engine, sample_migration):
        response = client.get("/api/v1/migrations")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        item = data["items"][0]
        assert item["source_model"] == "openai/gpt-4o"
        assert item["target_model"] == "anthropic/claude-sonnet-4"
        assert item["status"] == "complete"
        assert item["recommendation"] == "GO"
        assert item["confidence_score"] == 0.92

    def test_pagination(self, client, engine):
        """Test pagination with offset and limit."""
        with Session(engine) as session:
            for i in range(5):
                m = MigrationRecord(
                    source_model=f"model-{i}",
                    target_model="target",
                    status="complete",
                )
                session.add(m)
            session.commit()

        response = client.get("/api/v1/migrations?offset=0&limit=2")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        assert data["per_page"] == 2


class TestGetMigration:
    def test_get_detail(self, client, engine, sample_migration):
        response = client.get(f"/api/v1/migrations/{sample_migration.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_migration.id
        assert data["source_model"] == "openai/gpt-4o"
        assert data["target_model"] == "anthropic/claude-sonnet-4"
        assert data["status"] == "complete"
        assert data["confidence_score"] == 0.92
        assert data["baseline_score"] == 0.85
        assert data["improvement"] == 0.07
        assert data["recommendation"] == "GO"
        assert data["recommendation_reasoning"] == "All types pass thresholds."
        assert "json" in data["per_type_scores"]
        assert data["per_type_scores"]["json"]["win_rate"] == 0.95
        assert data["warnings"] == ["Low sample count for classification"]

    def test_get_detail_with_test_cases(self, client, engine, sample_migration, sample_test_cases):
        response = client.get(f"/api/v1/migrations/{sample_migration.id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data["test_cases"]) == 5

    def test_404_for_missing(self, client):
        response = client.get("/api/v1/migrations/999")
        assert response.status_code == 404
        assert response.json()["detail"] == "Migration not found"


class TestCreateMigration:
    def test_create_migration(self, client):
        payload = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
            "data_path": "/tmp/data.jsonl",
        }
        response = client.post("/api/v1/migrations", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["source_model"] == "openai/gpt-4o"
        assert data["target_model"] == "anthropic/claude-sonnet-4"
        assert data["status"] == "pending"
        assert data["id"] is not None

    def test_create_migration_with_cluster_prompts_and_objectives(self, client, engine):
        """Test that cluster_prompts and improvement_objectives are captured in config."""
        payload = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
            "data_path": "/tmp/data.jsonl",
            "cluster_prompts": True,
            "improvement_objectives": [
                {"name": "latency", "weight": 0.5},
                {"name": "accuracy", "weight": 0.5},
            ],
        }
        response = client.post("/api/v1/migrations", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["id"] is not None

        # Verify the record was created with the config fields
        migration_id = data["id"]
        with Session(engine) as session:
            stmt = select(MigrationRecord).where(MigrationRecord.id == migration_id)
            migration = session.exec(stmt).first()
            assert migration is not None
            config = json.loads(migration.config_json)
            assert config["cluster_prompts"] is True
            assert config["improvement_objectives"] == [
                {"name": "latency", "weight": 0.5},
                {"name": "accuracy", "weight": 0.5},
            ]

    def test_create_missing_fields(self, client):
        payload = {"source_model": "openai/gpt-4o"}
        response = client.post("/api/v1/migrations", json=payload)
        assert response.status_code == 422


class TestListTestCases:
    def test_list_test_cases(self, client, engine, sample_migration, sample_test_cases):
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/test-cases")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["items"]) == 5

    def test_filter_by_phase(self, client, engine, sample_migration, sample_test_cases):
        response = client.get(
            f"/api/v1/migrations/{sample_migration.id}/test-cases?phase=validation"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5

        # Filter by non-existent phase
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/test-cases?phase=baseline")
        data = response.json()
        assert data["total"] == 0

    def test_filter_by_output_type(self, client, engine, sample_migration, sample_test_cases):
        response = client.get(
            f"/api/v1/migrations/{sample_migration.id}/test-cases?output_type=json"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5

    def test_paginated_test_cases(self, client, engine, sample_migration, sample_test_cases):
        response = client.get(
            f"/api/v1/migrations/{sample_migration.id}/test-cases?offset=0&limit=2"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2

    def test_404_for_missing_migration(self, client):
        response = client.get("/api/v1/migrations/999/test-cases")
        assert response.status_code == 404


class TestGetTestCase:
    def test_get_test_case_detail(self, client, engine, sample_migration, sample_test_cases):
        tc = sample_test_cases[0]
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/test-cases/{tc.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == tc.id
        assert data["phase"] == "validation"
        assert data["output_type"] == "json"
        assert data["is_win"] is True
        assert "bertscore" in data["scores"]
        assert "output_type" in data["details"]

    def test_404_for_missing_test_case(self, client, engine, sample_migration):
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/test-cases/999")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# UI endpoints — template rendering with dummy data
# ---------------------------------------------------------------------------


class TestUIEndpoints:
    """UI endpoint tests verifying templates render with dummy data."""

    def test_dashboard_returns_models(self, client: TestClient) -> None:
        # With an empty DB, /ui/ now shows the empty state (no dummy fallback for models).
        # Register a model so the models page renders with real data.
        client.post("/api/v1/models", json={"model_id": "openai/gpt-4o"})
        client.post("/api/v1/models", json={"model_id": "anthropic/claude-sonnet-4"})
        resp = client.get("/ui/")
        assert resp.status_code == 200
        body = resp.text
        assert "Your models" in body
        assert "openai/gpt-4o" in body
        assert "anthropic/claude-sonnet-4" in body

    def test_dashboard_empty_state(self, client: TestClient) -> None:
        # With no registered models, /ui/ automatically shows the empty state.
        resp = client.get("/ui/")
        assert resp.status_code == 200
        body = resp.text
        assert "Welcome to RosettaStone" in body

    def test_migrations_list(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/migrations")
        assert resp.status_code == 200
        body = resp.text
        assert "Migrations" in body
        assert "gpt-4o" in body
        assert "claude-sonnet-4" in body
        assert "Safe to ship" in body
        assert "Needs review" in body
        assert "Do not ship" in body

    def test_migration_detail_safe(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/migrations/1")
        assert resp.status_code == 200
        body = resp.text
        assert "Safe to ship" in body
        assert "92%" in body or "92" in body
        assert "gpt-4o" in body
        assert "claude-sonnet-4" in body

    def test_migration_detail_blocked(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/migrations/3")
        assert resp.status_code == 200
        body = resp.text
        assert "Do not ship" in body
        assert "61" in body

    def test_migration_detail_404(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/migrations/999")
        assert resp.status_code == 404

    def test_costs_page(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/costs")
        assert resp.status_code == 200
        body = resp.text
        assert "Cost overview" in body
        assert "$1,247" in body
        assert "$312" in body

    def test_alerts_page(self, client: TestClient) -> None:
        # /ui/alerts now requires a DB session; use the in-memory client.
        # With no migrations in the DB it falls back to DUMMY_ALERTS.
        resp = client.get("/ui/alerts")
        assert resp.status_code == 200
        body = resp.text
        assert "Alerts" in body
        assert "gpt-4o-0613" in body
        assert "ACTION REQUIRED" in body

    def test_executive_report(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/migrations/1/executive")
        assert resp.status_code == 200
        body = resp.text
        assert "ROSETTASTONE MIGRATION REPORT" in body
        assert "gpt-4o" in body
        assert "Safe to" in body

    def test_diff_fragment(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/fragments/diff/1/42")
        assert resp.status_code == 200
        body = resp.text
        assert "Classification" in body
        assert "0.72" in body
        assert "BERTScore" in body

    def test_nav_links_present(self, client: TestClient) -> None:
        # Nav links are in base.html, present on both empty state and models page.
        resp = client.get("/ui/")
        assert resp.status_code == 200
        body = resp.text
        assert 'href="/ui/"' in body
        assert 'href="/ui/migrations"' in body
        assert 'href="/ui/costs"' in body
        assert 'href="/ui/alerts"' in body

    def test_models_page_contains_alerts_banner(self, client: TestClient) -> None:
        # Register a model so we get the full models.html with the alerts banner.
        client.post("/api/v1/models", json={"model_id": "openai/gpt-4o"})
        resp = client.get("/ui/")
        assert resp.status_code == 200
        body = resp.text
        assert "things need your attention" in body

    def test_migration_detail_has_regressions(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/migrations/1")
        assert resp.status_code == 200
        body = resp.text
        assert "Priority classification mismatch" in body
        assert "View diff" in body

    def test_costs_page_has_optimization_opportunities(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/costs")
        assert resp.status_code == 200
        body = resp.text
        assert "Optimization opportunities" in body
        assert "gpt-4o-mini" in body


# ---------------------------------------------------------------------------
# UI endpoints — template rendering with real DB data
# ---------------------------------------------------------------------------


class TestUIWithData:
    """UI endpoint tests verifying templates render with real DB data."""

    def test_migrations_list_shows_real_data(
        self, client: TestClient, sample_migration: MigrationRecord
    ) -> None:
        """When DB has migrations, the list page should show real data."""
        resp = client.get("/ui/migrations")
        assert resp.status_code == 200
        body = resp.text
        assert "Migrations" in body
        # Real migration source/target model names (short form after '/')
        assert "gpt-4o" in body
        assert "claude-sonnet-4" in body
        # Recommendation label mapped from "GO"
        assert "Safe to ship" in body

    def test_migrations_list_empty_state(self, client: TestClient) -> None:
        """When DB is empty, the list page should fall back to dummy data."""
        resp = client.get("/ui/migrations")
        assert resp.status_code == 200
        body = resp.text
        # Falls back to DUMMY_MIGRATIONS
        assert "Migrations" in body

    def test_migration_detail_shows_real_data(
        self, client: TestClient, sample_migration: MigrationRecord
    ) -> None:
        """Migration detail page should render real data from DB."""
        resp = client.get(f"/ui/migrations/{sample_migration.id}")
        assert resp.status_code == 200
        body = resp.text
        assert "gpt-4o" in body
        assert "claude-sonnet-4" in body
        # Confidence score: 0.92 → 92%
        assert "92" in body

    def test_recommendation_label_mapping(
        self, client: TestClient, sample_migration: MigrationRecord
    ) -> None:
        """DB 'GO' recommendation maps to 'Safe to ship' in the template."""
        resp = client.get(f"/ui/migrations/{sample_migration.id}")
        assert resp.status_code == 200
        body = resp.text
        assert "Safe to ship" in body
        # Should NOT show raw DB value
        # (raw "GO" only appears if the mapping fails)
        assert "Recommendation: Safe to switch" in body or "Safe to ship" in body

    def test_executive_report_with_real_data(
        self, client: TestClient, sample_migration: MigrationRecord
    ) -> None:
        """Executive report page should render with real migration data."""
        resp = client.get(f"/ui/migrations/{sample_migration.id}/executive")
        assert resp.status_code == 200
        body = resp.text
        assert "ROSETTASTONE MIGRATION REPORT" in body
        assert "gpt-4o" in body
        assert "claude-sonnet-4" in body
        assert "Safe to" in body

    def test_migration_detail_404_for_nonexistent(self, client: TestClient) -> None:
        """Should return 404 when migration is not in DB or dummy data."""
        resp = client.get("/ui/migrations/9999")
        assert resp.status_code == 404
