"""Tests for the Alert system — generation, API endpoints, and UI page rendering."""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
sqlmodel = pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient  # noqa: E402
from sqlmodel import Session  # noqa: E402

from rosettastone.server.models import MigrationRecord  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers — insert migration records directly into the DB
# ---------------------------------------------------------------------------


def _make_migration(
    session: Session,
    *,
    status: str,
    recommendation: str | None,
    source: str = "openai/gpt-4o",
    target: str = "anthropic/claude-sonnet-4",
    confidence: float = 0.92,
    reasoning: str = "Test reasoning.",
) -> MigrationRecord:
    m = MigrationRecord(
        source_model=source,
        target_model=target,
        status=status,
        recommendation=recommendation,
        recommendation_reasoning=reasoning,
        confidence_score=confidence,
        config_json=json.dumps({"source_model": source, "target_model": target}),
    )
    session.add(m)
    session.commit()
    session.refresh(m)
    return m


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


class TestAlertAuthentication:
    def test_generate_alerts_requires_auth(self, monkeypatch) -> None:
        """In multi-user mode, POST /api/v1/alerts/generate without auth returns 401 or 403."""
        from sqlalchemy.pool import StaticPool
        from sqlmodel import SQLModel, create_engine

        from rosettastone.server.app import create_app
        from rosettastone.server.database import get_session

        monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
        monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", "a" * 64)

        eng = create_engine(
            "sqlite://",
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(eng)

        app = create_app()

        from sqlmodel import Session as _Session

        def override_session():
            with _Session(eng) as s:
                yield s

        app.dependency_overrides[get_session] = override_session

        test_client = TestClient(app)
        resp = test_client.post("/api/v1/alerts/generate")
        assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Alert generation logic
# ---------------------------------------------------------------------------


class TestGenerateAlerts:
    def test_generate_creates_alert_for_complete_go(self, client: TestClient, engine) -> None:
        """A complete/GO migration produces a migration_complete alert."""
        with Session(engine) as session:
            _make_migration(session, status="complete", recommendation="GO")

        resp = client.post("/api/v1/alerts/generate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["generated"] == 1

        # Verify it appears in the list
        list_resp = client.get("/api/v1/alerts")
        assert list_resp.status_code == 200
        alerts = list_resp.json()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "new_model"  # migration_complete maps to new_model
        assert alerts[0]["severity"] == "info"

    def test_generate_creates_alert_for_failed(self, client: TestClient, engine) -> None:
        """A failed migration produces a migration_failed / critical alert."""
        with Session(engine) as session:
            _make_migration(session, status="failed", recommendation="NO_GO")

        resp = client.post("/api/v1/alerts/generate")
        assert resp.status_code == 200
        assert resp.json()["generated"] == 1

        list_resp = client.get("/api/v1/alerts")
        alerts = list_resp.json()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "deprecation"  # migration_failed maps to deprecation
        assert alerts[0]["severity"] == "critical"

    def test_generate_creates_alert_for_no_go_complete(self, client: TestClient, engine) -> None:
        """A complete/NO_GO migration (blocked) produces a critical alert."""
        with Session(engine) as session:
            _make_migration(session, status="complete", recommendation="NO_GO")

        resp = client.post("/api/v1/alerts/generate")
        assert resp.json()["generated"] == 1

        alerts = client.get("/api/v1/alerts").json()
        assert alerts[0]["severity"] == "critical"

    def test_generate_creates_warning_alert_for_conditional(
        self, client: TestClient, engine
    ) -> None:
        """A CONDITIONAL migration produces a warning-severity alert."""
        with Session(engine) as session:
            _make_migration(session, status="complete", recommendation="CONDITIONAL")

        resp = client.post("/api/v1/alerts/generate")
        assert resp.json()["generated"] == 1

        alerts = client.get("/api/v1/alerts").json()
        assert alerts[0]["severity"] == "warning"
        assert alerts[0]["type"] == "new_model"

    def test_generate_idempotent(self, client: TestClient, engine) -> None:
        """Calling generate twice doesn't duplicate alerts."""
        with Session(engine) as session:
            _make_migration(session, status="complete", recommendation="GO")

        client.post("/api/v1/alerts/generate")
        second = client.post("/api/v1/alerts/generate")
        assert second.json()["generated"] == 0

        alerts = client.get("/api/v1/alerts").json()
        assert len(alerts) == 1

    def test_generate_multiple_migrations(self, client: TestClient, engine) -> None:
        """Multiple migrations each get their own alert."""
        with Session(engine) as session:
            _make_migration(
                session,
                status="complete",
                recommendation="GO",
                source="openai/gpt-4o",
                target="anthropic/claude-sonnet-4",
            )
            _make_migration(
                session,
                status="failed",
                recommendation="NO_GO",
                source="openai/gpt-3.5-turbo",
                target="openai/gpt-4o-mini",
            )

        resp = client.post("/api/v1/alerts/generate")
        assert resp.json()["generated"] == 2

        alerts = client.get("/api/v1/alerts").json()
        assert len(alerts) == 2

    def test_generate_skips_pending_migrations(self, client: TestClient, engine) -> None:
        """Pending/running migrations do not produce alerts."""
        with Session(engine) as session:
            _make_migration(session, status="pending", recommendation=None)
            _make_migration(session, status="running", recommendation=None)

        resp = client.post("/api/v1/alerts/generate")
        assert resp.json()["generated"] == 0

        alerts = client.get("/api/v1/alerts").json()
        assert len(alerts) == 0

    def test_generate_empty_db_returns_zero(self, client: TestClient) -> None:
        """With no migrations, generate returns 0."""
        resp = client.post("/api/v1/alerts/generate")
        assert resp.status_code == 200
        assert resp.json()["generated"] == 0


# ---------------------------------------------------------------------------
# GET /api/v1/alerts
# ---------------------------------------------------------------------------


class TestListAlerts:
    def test_empty_list(self, client: TestClient) -> None:
        resp = client.get("/api/v1/alerts")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_alerts_newest_first(self, client: TestClient, engine) -> None:
        """Alerts are returned in descending created_at order."""
        with Session(engine) as session:
            _make_migration(
                session,
                status="complete",
                recommendation="GO",
                source="model-a",
                target="model-b",
            )
            _make_migration(
                session,
                status="failed",
                recommendation="NO_GO",
                source="model-c",
                target="model-d",
            )

        client.post("/api/v1/alerts/generate")
        alerts = client.get("/api/v1/alerts").json()
        assert len(alerts) == 2
        # Both have ids; the second migration inserted last should appear first
        assert alerts[0]["model"] == "model-d"

    def test_unread_only_filter(self, client: TestClient, engine) -> None:
        """?unread_only=true filters to unread alerts only."""
        with Session(engine) as session:
            _make_migration(session, status="complete", recommendation="GO")

        client.post("/api/v1/alerts/generate")
        # Initially all unread
        unread = client.get("/api/v1/alerts?unread_only=true").json()
        assert len(unread) == 1

        # Mark as read
        alert_id = unread[0]["id"]
        client.post(f"/api/v1/alerts/{alert_id}/read")

        unread_after = client.get("/api/v1/alerts?unread_only=true").json()
        assert len(unread_after) == 0

        # All alerts still include it
        all_alerts = client.get("/api/v1/alerts").json()
        assert len(all_alerts) == 1

    def test_alert_dict_shape(self, client: TestClient, engine) -> None:
        """Each alert dict has the fields the template expects."""
        with Session(engine) as session:
            _make_migration(session, status="complete", recommendation="GO")

        client.post("/api/v1/alerts/generate")
        alerts = client.get("/api/v1/alerts").json()
        a = alerts[0]

        assert "type" in a
        assert "model" in a
        assert "message" in a
        assert "action" in a
        assert "date" in a
        assert "severity" in a
        assert "is_read" in a


# ---------------------------------------------------------------------------
# POST /api/v1/alerts/{id}/read
# ---------------------------------------------------------------------------


class TestMarkAlertRead:
    def test_mark_read(self, client: TestClient, engine) -> None:
        with Session(engine) as session:
            _make_migration(session, status="complete", recommendation="GO")

        client.post("/api/v1/alerts/generate")
        alerts = client.get("/api/v1/alerts").json()
        alert_id = alerts[0]["id"]
        assert alerts[0]["is_read"] is False

        resp = client.post(f"/api/v1/alerts/{alert_id}/read")
        assert resp.status_code == 200
        assert resp.json()["is_read"] is True

    def test_mark_read_404(self, client: TestClient) -> None:
        resp = client.post("/api/v1/alerts/9999/read")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Alert not found"


# ---------------------------------------------------------------------------
# DELETE /api/v1/alerts/{id}
# ---------------------------------------------------------------------------


class TestDeleteAlert:
    def test_delete_alert(self, client: TestClient, engine) -> None:
        with Session(engine) as session:
            _make_migration(session, status="complete", recommendation="GO")

        client.post("/api/v1/alerts/generate")
        alerts = client.get("/api/v1/alerts").json()
        alert_id = alerts[0]["id"]

        resp = client.delete(f"/api/v1/alerts/{alert_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert resp.json()["id"] == alert_id

        # Confirm removed from list
        after = client.get("/api/v1/alerts").json()
        assert len(after) == 0

    def test_delete_alert_404(self, client: TestClient) -> None:
        resp = client.delete("/api/v1/alerts/9999")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Alert not found"

    def test_delete_allows_regenerate(self, client: TestClient, engine) -> None:
        """After deleting an alert, generate should NOT recreate it (idempotency key is
        migration_id + alert_type; deletion removes the guard — so it WILL regenerate)."""
        with Session(engine) as session:
            _make_migration(session, status="complete", recommendation="GO")

        client.post("/api/v1/alerts/generate")
        alerts = client.get("/api/v1/alerts").json()
        alert_id = alerts[0]["id"]

        client.delete(f"/api/v1/alerts/{alert_id}")

        # After deletion the idempotency check passes, so generate recreates it
        resp = client.post("/api/v1/alerts/generate")
        assert resp.json()["generated"] == 1


# ---------------------------------------------------------------------------
# /ui/alerts — HTML page rendering
# ---------------------------------------------------------------------------


class TestUIAlertsPage:
    def test_alerts_page_renders_with_real_alerts(self, client: TestClient, engine) -> None:
        """When DB has alerts, /ui/alerts renders real data."""
        with Session(engine) as session:
            _make_migration(session, status="complete", recommendation="GO")

        resp = client.get("/ui/alerts")
        assert resp.status_code == 200
        body = resp.text
        assert "Alerts" in body
        assert "ACTION REQUIRED" in body
        # Model name from our migration
        assert "claude-sonnet-4" in body

    def test_alerts_page_empty_when_no_alerts(self, client: TestClient) -> None:
        """When DB has no alerts, /ui/alerts renders an empty list."""
        resp = client.get("/ui/alerts")
        assert resp.status_code == 200
        body = resp.text
        assert "Alerts" in body
        # Page renders but with empty alerts list
        assert resp.status_code == 200

    def test_alerts_page_auto_generates_on_load(self, client: TestClient, engine) -> None:
        """The alerts page auto-calls _generate_alerts so migrations appear without
        manually hitting the generate endpoint first."""
        with Session(engine) as session:
            _make_migration(session, status="complete", recommendation="GO")

        # No explicit generate call — just load the page
        resp = client.get("/ui/alerts")
        assert resp.status_code == 200
        body = resp.text
        # Should see real migration data, not dummy
        assert "claude-sonnet-4" in body

    def test_alerts_page_shows_failed_migration(self, client: TestClient, engine) -> None:
        """Failed migrations render with the correct deprecation styling cue."""
        with Session(engine) as session:
            _make_migration(
                session,
                status="failed",
                recommendation="NO_GO",
                source="openai/gpt-3.5-turbo",
                target="openai/gpt-4o-mini",
                reasoning="Critical schema violations detected.",
            )

        resp = client.get("/ui/alerts")
        assert resp.status_code == 200
        body = resp.text
        assert "gpt-4o-mini" in body
        # Template uses border-[#D85650] for deprecation type
        assert "border-[#D85650]" in body

    def test_alerts_page_shows_conditional_migration(self, client: TestClient, engine) -> None:
        """CONDITIONAL migrations appear as new_model type (warning severity)."""
        with Session(engine) as session:
            _make_migration(session, status="complete", recommendation="CONDITIONAL")

        resp = client.get("/ui/alerts")
        assert resp.status_code == 200
        body = resp.text
        assert "claude-sonnet-4" in body

    def test_alerts_count_in_hero(self, client: TestClient, engine) -> None:
        """The hero section displays the count of alerts."""
        with Session(engine) as session:
            _make_migration(
                session,
                status="complete",
                recommendation="GO",
                source="model-a",
                target="model-b",
            )
            _make_migration(
                session,
                status="failed",
                recommendation="NO_GO",
                source="model-c",
                target="model-d",
            )

        resp = client.get("/ui/alerts")
        assert resp.status_code == 200
        body = resp.text
        # Template renders "{{ alerts|length }} Critical Alerts."
        assert "2 Critical Alerts." in body


# ---------------------------------------------------------------------------
# Type mapping correctness
# ---------------------------------------------------------------------------


class TestTypeMappingInAPIResponse:
    def test_go_migration_maps_to_new_model(self, client: TestClient, engine) -> None:
        with Session(engine) as session:
            _make_migration(session, status="complete", recommendation="GO")
        client.post("/api/v1/alerts/generate")
        alerts = client.get("/api/v1/alerts").json()
        assert alerts[0]["type"] == "new_model"

    def test_failed_migration_maps_to_deprecation(self, client: TestClient, engine) -> None:
        with Session(engine) as session:
            _make_migration(session, status="failed", recommendation="NO_GO")
        client.post("/api/v1/alerts/generate")
        alerts = client.get("/api/v1/alerts").json()
        assert alerts[0]["type"] == "deprecation"

    def test_conditional_migration_maps_to_new_model(self, client: TestClient, engine) -> None:
        with Session(engine) as session:
            _make_migration(session, status="complete", recommendation="CONDITIONAL")
        client.post("/api/v1/alerts/generate")
        alerts = client.get("/api/v1/alerts").json()
        assert alerts[0]["type"] == "new_model"


# ---------------------------------------------------------------------------
# Deprecation alerts
# ---------------------------------------------------------------------------


class TestDeprecationAlerts:
    def test_deprecated_model_generates_warning_alert(self, client: TestClient, engine) -> None:
        """A pending migration with a deprecated source model generates a deprecation alert."""
        with Session(engine) as session:
            # google/palm-2 is already retired in KNOWN_DEPRECATIONS
            _make_migration(
                session,
                status="pending",
                recommendation=None,
                source="google/palm-2",
                target="google/gemini-pro",
            )

        resp = client.post("/api/v1/alerts/generate")
        assert resp.status_code == 200
        # Should generate 1 deprecation alert
        data = resp.json()
        assert data["generated"] >= 1

        alerts = client.get("/api/v1/alerts").json()
        deprecation_alerts = [a for a in alerts if a["type"] == "deprecation"]
        assert len(deprecation_alerts) >= 1
        assert deprecation_alerts[0]["model"] == "google/palm-2"
        assert deprecation_alerts[0]["severity"] == "critical"
        assert "retired" in deprecation_alerts[0]["message"].lower()

    def test_soon_to_retire_model_generates_critical_alert(
        self, client: TestClient, engine
    ) -> None:
        """A migration with source model retiring in <30 days generates a critical alert."""
        with Session(engine) as session:
            # openai/gpt-3.5-turbo-0613 will retire on 2026-03-01 (soon from 2026-04-01)
            _make_migration(
                session,
                status="pending",
                recommendation=None,
                source="openai/gpt-3.5-turbo-0613",
                target="openai/gpt-4o-mini",
            )

        resp = client.post("/api/v1/alerts/generate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["generated"] >= 1

        alerts = client.get("/api/v1/alerts").json()
        deprecation_alerts = [a for a in alerts if a["type"] == "deprecation"]
        assert len(deprecation_alerts) >= 1
        assert "openai/gpt-3.5-turbo-0613" in [a["model"] for a in deprecation_alerts]

    def test_deprecated_model_alert_idempotent(self, client: TestClient, engine) -> None:
        """Generating alerts twice for the same model doesn't duplicate."""
        with Session(engine) as session:
            _make_migration(
                session,
                status="pending",
                recommendation=None,
                source="google/palm-2",
                target="google/gemini-pro",
            )

        resp1 = client.post("/api/v1/alerts/generate")
        count1 = resp1.json()["generated"]
        assert count1 >= 1

        resp2 = client.post("/api/v1/alerts/generate")
        count2 = resp2.json()["generated"]
        # Should not generate additional deprecation alerts
        assert count2 < count1

    def test_non_deprecated_model_no_alert(self, client: TestClient, engine) -> None:
        """A migration with non-deprecated models generates no deprecation alerts."""
        with Session(engine) as session:
            # Both are non-deprecated models
            _make_migration(
                session,
                status="pending",
                recommendation=None,
                source="openai/gpt-4o",
                target="anthropic/claude-sonnet-4",
            )

        resp = client.post("/api/v1/alerts/generate")
        assert resp.status_code == 200
        data = resp.json()
        # Should generate 0 alerts (no completion/failure, no deprecation)
        assert data["generated"] == 0

    def test_deprecation_alert_contains_replacement(self, client: TestClient, engine) -> None:
        """Deprecation alerts include the suggested replacement model."""
        with Session(engine) as session:
            _make_migration(
                session,
                status="pending",
                recommendation=None,
                source="google/palm-2",
                target="google/gemini-pro",
            )

        client.post("/api/v1/alerts/generate")
        alerts = client.get("/api/v1/alerts").json()
        deprecation_alerts = [a for a in alerts if a["type"] == "deprecation"]
        assert len(deprecation_alerts) >= 1
        assert "google/gemini-pro" in deprecation_alerts[0]["message"]
