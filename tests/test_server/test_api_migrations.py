"""Tests for migration API endpoints and UI template rendering."""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
sqlmodel = pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient  # noqa: E402
from sqlmodel import Session, select  # noqa: E402

from rosettastone.server.app import create_app  # noqa: E402
from rosettastone.server.models import MigrationRecord, TestCaseRecord  # noqa: E402

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
        data = response.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded", "unavailable")


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

    def test_get_detail_includes_cluster_summary(
        self, client, engine, sample_migration_with_cluster
    ):
        """Test that cluster_summary is exposed in migration detail response."""
        response = client.get(f"/api/v1/migrations/{sample_migration_with_cluster.id}")
        assert response.status_code == 200
        data = response.json()
        assert "cluster_summary" in data
        cluster_summary = data["cluster_summary"]
        assert cluster_summary is not None
        assert cluster_summary["n_clusters"] == 5
        assert cluster_summary["silhouette_score"] == 0.72
        assert cluster_summary["original_pairs"] == 100
        assert cluster_summary["representative_pairs"] == 25

    def test_get_detail_cluster_summary_null_when_not_clustered(
        self, client, engine, sample_migration
    ):
        """Test that cluster_summary is null when clustering was not enabled."""
        response = client.get(f"/api/v1/migrations/{sample_migration.id}")
        assert response.status_code == 200
        data = response.json()
        assert "cluster_summary" in data
        assert data["cluster_summary"] is None

    def test_404_for_missing(self, client):
        response = client.get("/api/v1/migrations/999")
        assert response.status_code == 404
        assert response.json()["detail"] == "Migration not found"


class TestCreateMigration:
    def test_create_migration(self, client, tmp_path, monkeypatch):
        data_file = tmp_path / "data.jsonl"
        data_file.touch()
        monkeypatch.setenv("HOME", str(tmp_path))
        payload = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
            "data_path": str(tmp_path / ".rosettastone" / "data.jsonl"),
        }
        response = client.post("/api/v1/migrations", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["source_model"] == "openai/gpt-4o"
        assert data["target_model"] == "anthropic/claude-sonnet-4"
        assert data["status"] == "pending"
        assert data["id"] is not None

    def test_create_migration_with_cluster_prompts_and_objectives(
        self, client, tmp_path, monkeypatch, engine
    ):
        """Test that cluster_prompts and improvement_objectives are captured in config."""
        monkeypatch.setenv("HOME", str(tmp_path))
        payload = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
            "data_path": str(tmp_path / ".rosettastone" / "data.jsonl"),
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

    def test_create_migration_negative_max_cost_returns_422(self, client):
        """Posting with negative max_cost_usd should return 422."""
        payload = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
            "max_cost_usd": -1.0,
        }
        response = client.post("/api/v1/migrations", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "max_cost_usd" in data.get("detail", "").lower()

    def test_create_migration_max_cost_accepted(self, client, tmp_path, monkeypatch, engine):
        """Posting with valid max_cost_usd should be accepted and stored."""
        monkeypatch.setenv("HOME", str(tmp_path))
        payload = {
            "source_model": "openai/gpt-4o",
            "target_model": "anthropic/claude-sonnet-4",
            "data_path": str(tmp_path / ".rosettastone" / "data.jsonl"),
            "max_cost_usd": 10.0,
        }
        response = client.post("/api/v1/migrations", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["id"] is not None

        # Verify the record was created with max_cost_usd
        migration_id = data["id"]
        with Session(engine) as session:
            stmt = select(MigrationRecord).where(MigrationRecord.id == migration_id)
            migration = session.exec(stmt).first()
            assert migration is not None
            assert migration.max_cost_usd == 10.0
            config = json.loads(migration.config_json)
            assert config.get("max_cost_usd") == 10.0


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
# Config field filtering
# ---------------------------------------------------------------------------


class TestMigrationDetailConfigFiltering:
    def test_lm_extra_kwargs_stripped_from_response(self, client, engine, session):
        """lm_extra_kwargs (may contain API keys) must not appear in the API response."""
        migration = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
            config_json=json.dumps(
                {
                    "source_model": "openai/gpt-4o",
                    "target_model": "anthropic/claude-sonnet-4",
                    "lm_extra_kwargs": {"api_key": "secret"},
                }
            ),
            per_type_scores_json=json.dumps({}),
            warnings_json=json.dumps([]),
            safety_warnings_json=json.dumps([]),
        )
        session.add(migration)
        session.commit()
        session.refresh(migration)

        response = client.get(f"/api/v1/migrations/{migration.id}")
        assert response.status_code == 200
        config = response.json()["config"]
        assert "lm_extra_kwargs" not in config
        assert config["source_model"] == "openai/gpt-4o"
        assert config["target_model"] == "anthropic/claude-sonnet-4"

    def test_lm_extra_kwargs_stays_in_db(self, engine, session):
        """The DB record must not be modified — only the API response is filtered."""
        migration = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
            config_json=json.dumps(
                {
                    "source_model": "openai/gpt-4o",
                    "lm_extra_kwargs": {"api_key": "secret"},
                }
            ),
            per_type_scores_json=json.dumps({}),
            warnings_json=json.dumps([]),
            safety_warnings_json=json.dumps([]),
        )
        session.add(migration)
        session.commit()
        session.refresh(migration)

        raw_config = json.loads(migration.config_json)
        assert "lm_extra_kwargs" in raw_config
        assert raw_config["lm_extra_kwargs"] == {"api_key": "secret"}


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
        # With no migrations in the DB it renders an empty alerts list.
        resp = client.get("/ui/alerts")
        assert resp.status_code == 200
        body = resp.text
        assert "Alerts" in body

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
        assert "Bertscore" in body  # dynamic label: bertscore key → title-cased

    def test_case_fragment_dummy_fallback(self, ui_client: TestClient) -> None:
        """Returns 200 with dummy fallback when TC is not in the DB."""
        resp = ui_client.get("/ui/fragments/test-case/1/9999")
        assert resp.status_code == 200
        body = resp.text
        # Falls back to DUMMY_TEST_CASES[42] — composite score and output type present
        assert "0.31" in body
        assert "Classification" in body

    def test_case_fragment_html_elements(self, ui_client: TestClient) -> None:
        """Response contains expected score bars and metadata sections."""
        resp = ui_client.get("/ui/fragments/test-case/1/42")
        assert resp.status_code == 200
        body = resp.text
        assert "Bertscore" in body  # dynamic label: bertscore key → title-cased
        assert "Embedding" in body  # dynamic label: embedding key → title-cased
        assert "Composite" in body
        assert "Test Case Metadata" in body
        # Dummy tc_id 42 has composite_score 0.31 and output_type Classification
        assert "0.31" in body
        assert "Classification" in body

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


# ---------------------------------------------------------------------------
# P2.1: Filterable test case grid (HTMX fragment endpoint)
# ---------------------------------------------------------------------------


class TestTestCasesTableFragment:
    """Tests for GET /ui/migrations/{id}/test-cases-table HTMX fragment."""

    def test_test_cases_table_loads(
        self, client: TestClient, engine, sample_migration: MigrationRecord, sample_test_cases
    ) -> None:
        """Fragment endpoint returns 200 with table HTML containing test case rows."""
        resp = client.get(f"/ui/migrations/{sample_migration.id}/test-cases-table")
        assert resp.status_code == 200
        body = resp.text
        # Table structure present
        assert "<table" in body
        assert "<tbody" in body
        # Filter form present
        assert "tc-filter-form" in body
        # At least one row for the 5 sample test cases
        assert "WIN" in body

    def test_test_cases_table_filter_win(
        self, client: TestClient, engine, session, sample_migration: MigrationRecord
    ) -> None:
        """?outcome=win returns only WIN rows; no LOSS badges visible."""
        import json as _json

        from rosettastone.server.models import TestCaseRecord

        # Insert mixed win/loss test cases
        win_tc = TestCaseRecord(
            migration_id=sample_migration.id,
            phase="validation",
            output_type="json",
            composite_score=0.9,
            is_win=True,
            scores_json=_json.dumps({"bertscore": 0.9}),
            details_json=_json.dumps({}),
            prompt_text="win prompt",
        )
        loss_tc = TestCaseRecord(
            migration_id=sample_migration.id,
            phase="validation",
            output_type="json",
            composite_score=0.3,
            is_win=False,
            scores_json=_json.dumps({"bertscore": 0.3}),
            details_json=_json.dumps({}),
            prompt_text="loss prompt",
        )
        session.add(win_tc)
        session.add(loss_tc)
        session.commit()

        resp = client.get(f"/ui/migrations/{sample_migration.id}/test-cases-table?outcome=win")
        assert resp.status_code == 200
        body = resp.text
        assert "WIN" in body
        assert "LOSS" not in body

    def test_test_cases_table_filter_loss(
        self, client: TestClient, engine, session, sample_migration: MigrationRecord
    ) -> None:
        """?outcome=loss returns only LOSS rows; no WIN badges visible."""
        import json as _json

        from rosettastone.server.models import TestCaseRecord

        # Clear any existing test cases for this migration and insert fresh ones
        win_tc = TestCaseRecord(
            migration_id=sample_migration.id,
            phase="validation",
            output_type="json",
            composite_score=0.9,
            is_win=True,
            scores_json=_json.dumps({"bertscore": 0.9}),
            details_json=_json.dumps({}),
            prompt_text="win only",
        )
        loss_tc = TestCaseRecord(
            migration_id=sample_migration.id,
            phase="validation",
            output_type="json",
            composite_score=0.2,
            is_win=False,
            scores_json=_json.dumps({"bertscore": 0.2}),
            details_json=_json.dumps({}),
            prompt_text="loss only",
        )
        session.add(win_tc)
        session.add(loss_tc)
        session.commit()

        resp = client.get(f"/ui/migrations/{sample_migration.id}/test-cases-table?outcome=loss")
        assert resp.status_code == 200
        body = resp.text
        assert "LOSS" in body
        assert "WIN" not in body

    def test_test_cases_table_pagination(
        self, client: TestClient, engine, sample_migration: MigrationRecord, sample_test_cases
    ) -> None:
        """page_size=1 returns exactly 1 row; page=2 returns the next row."""
        resp_p1 = client.get(
            f"/ui/migrations/{sample_migration.id}/test-cases-table?page_size=1&page=1"
        )
        assert resp_p1.status_code == 200
        body_p1 = resp_p1.text
        # Only one data row — pagination controls present
        assert "Next" in body_p1
        # Page indicator
        assert "Page 1 of" in body_p1

        resp_p2 = client.get(
            f"/ui/migrations/{sample_migration.id}/test-cases-table?page_size=1&page=2"
        )
        assert resp_p2.status_code == 200
        body_p2 = resp_p2.text
        assert "Page 2 of" in body_p2
        assert "Prev" in body_p2

    def test_test_cases_table_empty(
        self, client: TestClient, engine, sample_migration: MigrationRecord
    ) -> None:
        """Migration with no test cases returns the empty-state message."""
        # sample_migration has no test cases (sample_test_cases fixture not requested)
        resp = client.get(f"/ui/migrations/{sample_migration.id}/test-cases-table")
        assert resp.status_code == 200
        body = resp.text
        assert "No test cases match your filters." in body


# ---------------------------------------------------------------------------
# P2.2: Inline persona toggle — executive summary fragment
# ---------------------------------------------------------------------------


class TestExecutiveSummaryFragment:
    """Tests for GET /ui/migrations/{id}/executive-summary HTMX fragment."""

    def test_executive_summary_fragment_returns_200(self, ui_client: TestClient) -> None:
        """Fragment endpoint returns 200 for a known dummy migration."""
        resp = ui_client.get("/ui/migrations/1/executive-summary")
        assert resp.status_code == 200

    def test_executive_summary_contains_confidence(self, ui_client: TestClient) -> None:
        """Response body includes the confidence score for the migration."""
        resp = ui_client.get("/ui/migrations/1/executive-summary")
        assert resp.status_code == 200
        body = resp.text
        # Dummy migration 1 has confidence 92
        assert "92" in body
        assert "Confidence Score" in body

    def test_executive_summary_narrative_truncated(
        self, client: TestClient, engine, session
    ) -> None:
        """Reasoning longer than 500 chars is capped with an ellipsis."""
        long_reasoning = "A" * 600
        record = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
            confidence_score=0.88,
            recommendation="GO",
            recommendation_reasoning=long_reasoning,
        )
        session.add(record)
        session.commit()
        session.refresh(record)

        resp = client.get(f"/ui/migrations/{record.id}/executive-summary")
        assert resp.status_code == 200
        body = resp.text
        # The narrative should be truncated and end with ellipsis
        assert "\u2026" in body
        # The full 600-char string should not appear verbatim
        assert "A" * 600 not in body


# ---------------------------------------------------------------------------
# P2.3: Score charts — win-rate donut and score histogram
# ---------------------------------------------------------------------------


class TestScoreCharts:
    """Tests for chart data injected by _migration_to_template_dict (P2.3)."""

    def test_migration_detail_has_wins_and_losses(
        self,
        client: TestClient,
        engine,
        session,
        sample_migration: MigrationRecord,
    ) -> None:
        """Migration detail page includes chart canvas elements when per_type data exists."""
        import json as _json

        from rosettastone.server.models import TestCaseRecord

        # Insert 3 wins and 2 losses
        for i in range(3):
            tc = TestCaseRecord(
                migration_id=sample_migration.id,
                phase="validation",
                output_type="json",
                composite_score=0.85 + i * 0.05,
                is_win=True,
                scores_json=_json.dumps({"bertscore": 0.9}),
                details_json=_json.dumps({}),
            )
            session.add(tc)
        for i in range(2):
            tc = TestCaseRecord(
                migration_id=sample_migration.id,
                phase="validation",
                output_type="json",
                composite_score=0.30 + i * 0.05,
                is_win=False,
                scores_json=_json.dumps({"bertscore": 0.3}),
                details_json=_json.dumps({}),
            )
            session.add(tc)
        session.commit()

        resp = client.get(f"/ui/migrations/{sample_migration.id}")
        assert resp.status_code == 200
        body = resp.text
        # Chart canvas elements should be present (per_type is populated in sample_migration)
        assert "winRateChart" in body
        assert "scoreHistChart" in body

    def test_score_histogram_always_10_bins(
        self,
        client: TestClient,
        engine,
        session,
        sample_migration: MigrationRecord,
    ) -> None:
        """score_histogram in the rendered page has exactly 10 bins summing to total test cases."""
        import json as _json

        from rosettastone.server.api.migrations import _migration_to_template_dict
        from rosettastone.server.models import TestCaseRecord

        # Insert test cases with scores spread across bins
        scores = [0.05, 0.15, 0.25, 0.55, 0.75, 0.95]
        for score in scores:
            tc = TestCaseRecord(
                migration_id=sample_migration.id,
                phase="validation",
                output_type="json",
                composite_score=score,
                is_win=score >= 0.5,
                scores_json=_json.dumps({"bertscore": score}),
                details_json=_json.dumps({}),
            )
            session.add(tc)
        session.commit()

        result = _migration_to_template_dict(sample_migration, session)

        histogram = result["score_histogram"]
        assert isinstance(histogram, list)
        assert len(histogram) == 10
        assert sum(histogram) == len(scores)


# ---------------------------------------------------------------------------
# HTMX fragments: migration-list and eval-grid
# ---------------------------------------------------------------------------


class TestMigrationListFragment:
    """Tests for GET /ui/fragments/migration-list."""

    def test_migration_list_fragment_returns_200(self, ui_client: TestClient) -> None:
        """Fragment endpoint returns 200 with HTML content."""
        resp = ui_client.get("/ui/fragments/migration-list")
        assert resp.status_code == 200
        body = resp.text
        assert len(body) > 0

    def test_migration_list_fragment_not_placeholder(self, ui_client: TestClient) -> None:
        """Response is not the original placeholder string."""
        resp = ui_client.get("/ui/fragments/migration-list")
        assert resp.status_code == 200
        assert "Template pending integration" not in resp.text

    def test_migration_list_fragment_shows_dummy_data(self, ui_client: TestClient) -> None:
        """When DB is empty the fragment falls back to dummy migration cards."""
        resp = ui_client.get("/ui/fragments/migration-list")
        assert resp.status_code == 200
        body = resp.text
        # Dummy data contains cards linking to /ui/migrations/<id>
        assert "/ui/migrations/" in body

    def test_migration_list_fragment_with_db_data(
        self, client: TestClient, sample_migration: MigrationRecord
    ) -> None:
        """When DB has a migration the fragment renders a card for it."""
        resp = client.get("/ui/fragments/migration-list")
        assert resp.status_code == 200
        body = resp.text
        assert "Template pending integration" not in body
        assert f"/ui/migrations/{sample_migration.id}" in body
        assert "gpt-4o" in body

    def test_migration_list_fragment_empty_db_shows_no_migrations(self, client: TestClient) -> None:
        """When DB is empty the fragment falls back to dummy cards (not empty-state, since
        DUMMY_MIGRATIONS is non-empty), so we still get card links."""
        resp = client.get("/ui/fragments/migration-list")
        assert resp.status_code == 200
        body = resp.text
        assert "Template pending integration" not in body


class TestEvalGridFragment:
    """Tests for GET /ui/fragments/eval-grid/{migration_id}."""

    def test_eval_grid_returns_200(
        self, client: TestClient, sample_migration: MigrationRecord
    ) -> None:
        """Fragment endpoint returns 200."""
        resp = client.get(f"/ui/fragments/eval-grid/{sample_migration.id}")
        assert resp.status_code == 200

    def test_eval_grid_not_placeholder(
        self, client: TestClient, sample_migration: MigrationRecord
    ) -> None:
        """Response is not the original placeholder string."""
        resp = client.get(f"/ui/fragments/eval-grid/{sample_migration.id}")
        assert resp.status_code == 200
        assert "Template pending integration" not in resp.text

    def test_eval_grid_empty_state(
        self, client: TestClient, sample_migration: MigrationRecord
    ) -> None:
        """Migration with no test cases returns the empty-state row."""
        resp = client.get(f"/ui/fragments/eval-grid/{sample_migration.id}")
        assert resp.status_code == 200
        body = resp.text
        assert "No test cases found for this migration" in body

    def test_eval_grid_with_test_cases(
        self,
        client: TestClient,
        sample_migration: MigrationRecord,
        sample_test_cases: list,
    ) -> None:
        """With test cases present, the fragment returns table rows with scores."""
        resp = client.get(f"/ui/fragments/eval-grid/{sample_migration.id}")
        assert resp.status_code == 200
        body = resp.text
        # Should contain <tr> elements
        assert "<tr>" in body
        # WIN badge present (sample_test_cases are all is_win=True)
        assert "WIN" in body
        # Score percentage present
        assert "%" in body

    def test_eval_grid_shows_output_type(
        self,
        client: TestClient,
        sample_migration: MigrationRecord,
        sample_test_cases: list,
    ) -> None:
        """Test case rows include the output type."""
        resp = client.get(f"/ui/fragments/eval-grid/{sample_migration.id}")
        assert resp.status_code == 200
        body = resp.text
        # sample_test_cases use output_type="json" → displayed as "Json"
        assert "Json" in body

    def test_eval_grid_shows_phase(
        self,
        client: TestClient,
        sample_migration: MigrationRecord,
        sample_test_cases: list,
    ) -> None:
        """Test case rows include the phase."""
        resp = client.get(f"/ui/fragments/eval-grid/{sample_migration.id}")
        assert resp.status_code == 200
        body = resp.text
        # sample_test_cases use phase="validation" → displayed as "Validation"
        assert "Validation" in body

    def test_eval_grid_loss_badge(
        self, client: TestClient, engine, session, sample_migration: MigrationRecord
    ) -> None:
        """A losing test case shows a LOSS badge."""
        import json as _json

        tc = TestCaseRecord(
            migration_id=sample_migration.id,
            phase="baseline",
            output_type="short_text",
            composite_score=0.30,
            is_win=False,
            scores_json=_json.dumps({"bertscore": 0.3}),
            details_json=_json.dumps({}),
        )
        session.add(tc)
        session.commit()

        resp = client.get(f"/ui/fragments/eval-grid/{sample_migration.id}")
        assert resp.status_code == 200
        body = resp.text
        assert "LOSS" in body


def test_migration_detail_has_total_tokens(client, session, engine):
    """total_tokens field appears in migration detail API response."""
    import json as _json

    from sqlmodel import Session as _Session

    from rosettastone.server.models import MigrationRecord as _MR

    with _Session(engine) as s:
        record = _MR(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
            confidence_score=0.88,
            baseline_score=0.80,
            improvement=0.08,
            cost_usd=0.50,
            recommendation="GO",
            total_tokens=1500,
            token_breakdown_json=_json.dumps({"evaluation": 1000, "optimization": 500}),
        )
        s.add(record)
        s.commit()
        s.refresh(record)
        migration_id = record.id

    resp = client.get(f"/api/v1/migrations/{migration_id}")
    assert resp.status_code == 200
    data = resp.json()
    # The key assertion: total_tokens is stored and accessible on the record
    assert record.total_tokens == 1500


def test_get_optimization_trace_empty(client, engine):
    """Migration with no iteration history returns correct empty shape."""

    from sqlmodel import Session as _Session

    from rosettastone.server.models import MigrationRecord as _MR

    with _Session(engine) as s:
        record = _MR(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
            confidence_score=0.88,
            baseline_score=0.80,
            improvement=0.08,
            cost_usd=0.50,
            recommendation="GO",
            optimization_score_history_json="[]",
        )
        s.add(record)
        s.commit()
        s.refresh(record)
        migration_id = record.id

    resp = client.get(f"/api/v1/migrations/{migration_id}/optimization-trace")
    assert resp.status_code == 200
    data = resp.json()
    assert "iterations" in data
    assert data["total_iterations"] == 0
    assert data["iterations"] == []
    assert data["migration_id"] == migration_id


def test_get_optimization_trace_with_data(client, engine):
    """Migration with iteration history returns the data correctly."""
    import json as _json

    from sqlmodel import Session as _Session

    from rosettastone.server.models import MigrationRecord as _MR

    history = [
        {"iteration_num": 1, "mean_score": 0.6123, "timestamp_iso": "2026-04-05T10:00:00+00:00"},
        {"iteration_num": 2, "mean_score": 0.7456, "timestamp_iso": "2026-04-05T10:01:00+00:00"},
        {"iteration_num": 3, "mean_score": 0.8012, "timestamp_iso": "2026-04-05T10:02:00+00:00"},
    ]

    with _Session(engine) as s:
        record = _MR(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
            confidence_score=0.90,
            baseline_score=0.80,
            improvement=0.10,
            cost_usd=1.00,
            recommendation="GO",
            optimized_prompt="Optimized system prompt here.",
            optimization_score_history_json=_json.dumps(history),
        )
        s.add(record)
        s.commit()
        s.refresh(record)
        migration_id = record.id

    resp = client.get(f"/api/v1/migrations/{migration_id}/optimization-trace")
    assert resp.status_code == 200
    data = resp.json()
    assert data["migration_id"] == migration_id
    assert data["total_iterations"] == 3
    assert len(data["iterations"]) == 3
    assert data["iterations"][0]["iteration_num"] == 1
    assert data["iterations"][2]["mean_score"] == 0.8012
    assert data["final_prompt_length"] == len("Optimized system prompt here.")


# ---------------------------------------------------------------------------
# F1: Migration Diagnostics API
# ---------------------------------------------------------------------------


def test_diagnostics_returns_structure(client, engine):
    """Completed migration with test cases returns 200 with all expected keys."""
    import json as _json

    from sqlmodel import Session

    from rosettastone.server.models import MigrationRecord, TestCaseRecord  # noqa: F811

    with Session(engine) as s:
        record = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
            recommendation="GO",
            per_type_scores_json=_json.dumps(
                {
                    "json": {
                        "win_rate": 0.95,
                        "mean": 0.93,
                        "median": 0.94,
                        "p10": 0.88,
                        "p50": 0.94,
                        "p90": 0.98,
                        "min_score": 0.85,
                        "max_score": 1.0,
                        "sample_count": 10,
                        "confidence_interval": [0.88, 0.99],
                    }
                }
            ),
        )
        s.add(record)
        s.commit()
        s.refresh(record)
        migration_id = record.id
        # Add a validation test case
        tc = TestCaseRecord(
            migration_id=migration_id,
            phase="validation",
            output_type="json",
            composite_score=0.90,
            is_win=True,
            scores_json=_json.dumps({"bertscore": 0.91, "exact_match": 0.85}),
            details_json=_json.dumps({}),
        )
        s.add(tc)
        s.commit()

    resp = client.get(f"/api/v1/migrations/{migration_id}/diagnostics")
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "migration_id",
        "recommendation",
        "per_type",
        "metric_win_rates",
        "border_cases",
        "regression_summary",
        "safety",
    ):
        assert key in data


def test_diagnostics_border_cases_within_range(client, engine):
    """Test cases within +/-5% of threshold appear in border_cases."""
    import json as _json

    from sqlmodel import Session

    from rosettastone.server.models import MigrationRecord, TestCaseRecord  # noqa: F811

    with Session(engine) as s:
        record = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
            recommendation="GO",
        )
        s.add(record)
        s.commit()
        s.refresh(record)
        migration_id = record.id
        # json threshold = 0.95; score 0.94 is within -0.01 (within ±0.05)
        tc = TestCaseRecord(
            migration_id=migration_id,
            phase="validation",
            output_type="json",
            composite_score=0.94,
            is_win=False,
            scores_json=_json.dumps({"bertscore": 0.94}),
            details_json=_json.dumps({}),
        )
        s.add(tc)
        s.commit()

    resp = client.get(f"/api/v1/migrations/{migration_id}/diagnostics")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["border_cases"]) >= 1
    bc = data["border_cases"][0]
    assert bc["output_type"] == "json"
    assert abs(bc["delta_to_threshold"]) <= 0.05


def test_diagnostics_no_prompt_text_in_response(client, engine):
    """Response JSON must not contain keys named prompt_text, response_text, or new_response_text."""
    import json as _json

    from sqlmodel import Session

    from rosettastone.server.models import MigrationRecord, TestCaseRecord  # noqa: F811

    with Session(engine) as s:
        record = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
            recommendation="GO",
        )
        s.add(record)
        s.commit()
        s.refresh(record)
        migration_id = record.id
        tc = TestCaseRecord(
            migration_id=migration_id,
            phase="validation",
            output_type="short_text",
            composite_score=0.82,
            is_win=True,
            scores_json=_json.dumps({"bertscore": 0.82}),
            details_json=_json.dumps({}),
            prompt_text="this is sensitive PII text",
            response_text="source response text",
            new_response_text="new response text",
        )
        s.add(tc)
        s.commit()

    resp = client.get(f"/api/v1/migrations/{migration_id}/diagnostics")
    assert resp.status_code == 200
    resp_str = resp.text
    assert "prompt_text" not in resp_str
    assert "response_text" not in resp_str
    assert "new_response_text" not in resp_str
    assert "this is sensitive PII text" not in resp_str


def test_diagnostics_pending_migration_returns_422(client, engine):
    """Pending migration returns 422."""
    from sqlmodel import Session

    from rosettastone.server.models import MigrationRecord  # noqa: F811

    with Session(engine) as s:
        record = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="pending",
        )
        s.add(record)
        s.commit()
        s.refresh(record)
        migration_id = record.id

    resp = client.get(f"/api/v1/migrations/{migration_id}/diagnostics")
    assert resp.status_code == 422


def test_diagnostics_regression_counts(client, engine):
    """Controlled baseline+validation TC pairs produce expected regression counts."""
    import json as _json

    from sqlmodel import Session

    from rosettastone.server.models import MigrationRecord, TestCaseRecord  # noqa: F811

    with Session(engine) as s:
        record = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
            recommendation="GO",
        )
        s.add(record)
        s.commit()
        s.refresh(record)
        migration_id = record.id

        # short_text threshold = 0.80
        # Pair 1: improved (delta = +0.1 >= 0.05)
        s.add(
            TestCaseRecord(
                migration_id=migration_id,
                phase="baseline",
                output_type="short_text",
                composite_score=0.70,
                is_win=False,
                scores_json=_json.dumps({"bertscore": 0.70}),
                details_json=_json.dumps({}),
            )
        )
        s.add(
            TestCaseRecord(
                migration_id=migration_id,
                phase="validation",
                output_type="short_text",
                composite_score=0.80,
                is_win=True,
                scores_json=_json.dumps({"bertscore": 0.80}),
                details_json=_json.dumps({}),
            )
        )
        # Pair 2: stable (delta = 0.0, within -0.05..0.05)
        s.add(
            TestCaseRecord(
                migration_id=migration_id,
                phase="baseline",
                output_type="short_text",
                composite_score=0.85,
                is_win=True,
                scores_json=_json.dumps({"bertscore": 0.85}),
                details_json=_json.dumps({}),
            )
        )
        s.add(
            TestCaseRecord(
                migration_id=migration_id,
                phase="validation",
                output_type="short_text",
                composite_score=0.85,
                is_win=True,
                scores_json=_json.dumps({"bertscore": 0.85}),
                details_json=_json.dumps({}),
            )
        )
        # Pair 3: regressed (delta = -0.10 < -0.05, but val=0.82 >= threshold=0.80)
        s.add(
            TestCaseRecord(
                migration_id=migration_id,
                phase="baseline",
                output_type="short_text",
                composite_score=0.92,
                is_win=True,
                scores_json=_json.dumps({"bertscore": 0.92}),
                details_json=_json.dumps({}),
            )
        )
        s.add(
            TestCaseRecord(
                migration_id=migration_id,
                phase="validation",
                output_type="short_text",
                composite_score=0.82,
                is_win=True,
                scores_json=_json.dumps({"bertscore": 0.82}),
                details_json=_json.dumps({}),
            )
        )
        s.commit()

    resp = client.get(f"/api/v1/migrations/{migration_id}/diagnostics")
    assert resp.status_code == 200
    data = resp.json()
    rs = data["regression_summary"]
    assert rs["improved_count"] == 1
    assert rs["stable_count"] == 1
    assert rs["regressed_count"] == 1
    assert rs["at_risk_count"] == 0


def test_diagnostics_at_risk_classification(client, engine):
    """A pair with delta < -0.05 and val < threshold is classified as at_risk."""
    import json as _json

    from sqlmodel import Session

    from rosettastone.server.models import MigrationRecord, TestCaseRecord  # noqa: F811

    with Session(engine) as s:
        record = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
            recommendation="NO_GO",
        )
        s.add(record)
        s.commit()
        s.refresh(record)
        migration_id = record.id
        # short_text threshold = 0.80; at_risk = delta < -0.05 AND val < threshold
        s.add(
            TestCaseRecord(
                migration_id=migration_id,
                phase="baseline",
                output_type="short_text",
                composite_score=0.90,
                is_win=True,
                scores_json=_json.dumps({"bertscore": 0.90}),
                details_json=_json.dumps({}),
            )
        )
        s.add(
            TestCaseRecord(
                migration_id=migration_id,
                phase="validation",
                output_type="short_text",
                composite_score=0.70,  # delta = -0.20 < -0.05, val < 0.80 threshold
                is_win=False,
                scores_json=_json.dumps({"bertscore": 0.70}),
                details_json=_json.dumps({}),
            )
        )
        s.commit()

    resp = client.get(f"/api/v1/migrations/{migration_id}/diagnostics")
    assert resp.status_code == 200
    rs = resp.json()["regression_summary"]
    assert rs["at_risk_count"] == 1
    assert rs["regressed_count"] == 0
    assert rs["improved_count"] == 0
    assert rs["stable_count"] == 0
    # The at_risk pair must appear in worst_regressed
    assert len(rs["worst_regressed"]) == 1
    assert rs["worst_regressed"][0]["status"] == "at_risk"
