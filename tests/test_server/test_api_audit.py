"""Tests for the audit log API router and log_audit utility."""

from __future__ import annotations

import json

from sqlmodel import Session, select

from rosettastone.server.api.audit import log_audit
from rosettastone.server.models import AuditLog


class TestLogAudit:
    def test_creates_audit_entry(self, session):
        """log_audit adds an AuditLog row to the session."""
        log_audit(session, "migration", 1, "create")
        session.commit()

        entries = list(session.exec(select(AuditLog)).all())
        assert len(entries) == 1
        assert entries[0].resource_type == "migration"
        assert entries[0].resource_id == 1
        assert entries[0].action == "create"
        assert entries[0].user_id is None

    def test_with_details(self, session):
        """log_audit serializes details to JSON."""
        log_audit(session, "migration", 1, "rollback", details={"version": 3})
        session.commit()

        entry = session.exec(select(AuditLog)).one()
        assert json.loads(entry.details_json) == {"version": 3}

    def test_with_user_id(self, session):
        """log_audit records user_id when provided."""
        log_audit(session, "model", 5, "create", user_id=42)
        session.commit()

        entry = session.exec(select(AuditLog)).one()
        assert entry.user_id == 42


class TestListAuditLog:
    def _seed_entries(self, session: Session) -> None:
        """Seed multiple audit entries for testing."""
        entries = [
            ("migration", 1, "create"),
            ("migration", 1, "complete"),
            ("migration", 2, "create"),
            ("model", None, "create"),
            ("migration", 1, "rollback"),
        ]
        for rt, rid, action in entries:
            log_audit(session, rt, rid, action)
        session.commit()

    def test_list_all(self, client, engine):
        """GET /api/v1/audit-log returns all entries."""
        with Session(engine) as s:
            self._seed_entries(s)

        resp = client.get("/api/v1/audit-log")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["items"]) == 5

    def test_filter_by_resource_type(self, client, engine):
        """GET with resource_type filter returns matching entries."""
        with Session(engine) as s:
            self._seed_entries(s)

        resp = client.get("/api/v1/audit-log?resource_type=model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["resource_type"] == "model"

    def test_filter_by_action(self, client, engine):
        """GET with action filter returns matching entries."""
        with Session(engine) as s:
            self._seed_entries(s)

        resp = client.get("/api/v1/audit-log?action=create")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3

    def test_pagination(self, client, engine):
        """GET with page/per_page paginates correctly."""
        with Session(engine) as s:
            self._seed_entries(s)

        resp = client.get("/api/v1/audit-log?per_page=2&page=1")
        data = resp.json()
        assert len(data["items"]) == 2
        assert data["total"] == 5
        assert data["page"] == 1
        assert data["per_page"] == 2

    def test_empty_log(self, client):
        """GET returns empty list when no entries exist."""
        resp = client.get("/api/v1/audit-log")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []
