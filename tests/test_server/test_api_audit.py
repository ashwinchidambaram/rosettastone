"""Tests for the audit log API router and log_audit utility."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from rosettastone.server.api.audit import log_audit
from rosettastone.server.app import create_app
from rosettastone.server.auth_utils import create_jwt
from rosettastone.server.database import get_session
from rosettastone.server.models import AuditLog

_TEST_JWT_SECRET = "test-secret-long-enough-for-hmac-sha256-validation"


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

    def test_list_all(self, multi_user_client, engine):
        """GET /api/v1/audit-log returns all entries."""
        with Session(engine) as s:
            self._seed_entries(s)

        resp = multi_user_client.get("/api/v1/audit-log")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["items"]) == 5

    def test_filter_by_resource_type(self, multi_user_client, engine):
        """GET with resource_type filter returns matching entries."""
        with Session(engine) as s:
            self._seed_entries(s)

        resp = multi_user_client.get("/api/v1/audit-log?resource_type=model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["resource_type"] == "model"

    def test_filter_by_action(self, multi_user_client, engine):
        """GET with action filter returns matching entries."""
        with Session(engine) as s:
            self._seed_entries(s)

        resp = multi_user_client.get("/api/v1/audit-log?action=create")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3

    def test_pagination(self, multi_user_client, engine):
        """GET with page/per_page paginates correctly."""
        with Session(engine) as s:
            self._seed_entries(s)

        resp = multi_user_client.get("/api/v1/audit-log?per_page=2&page=1")
        data = resp.json()
        assert len(data["items"]) == 2
        assert data["total"] == 5
        assert data["page"] == 1
        assert data["per_page"] == 2

    def test_empty_log(self, multi_user_client):
        """GET returns empty list when no entries exist."""
        resp = multi_user_client.get("/api/v1/audit-log")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []


# ---------------------------------------------------------------------------
# Task 1.4: Audit log access control tests
# ---------------------------------------------------------------------------


def _make_isolated_engine():
    """Create a fresh in-memory SQLite engine for isolation between test cases."""
    eng = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(eng)
    return eng


def _multi_user_client(engine, monkeypatch) -> TestClient:
    """Build a TestClient with multi-user mode enabled and the test DB wired in."""
    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", _TEST_JWT_SECRET)
    app = create_app()

    def override_session():
        with Session(engine) as sess:
            yield sess

    app.dependency_overrides[get_session] = override_session
    return TestClient(app, raise_server_exceptions=True)


def _single_user_client(engine) -> TestClient:
    """Build a TestClient in single-user mode with the test DB wired in."""
    app = create_app()

    def override_session():
        with Session(engine) as sess:
            yield sess

    app.dependency_overrides[get_session] = override_session
    return TestClient(app, raise_server_exceptions=True)


def _auth_header(user_id: int, role: str = "editor") -> dict[str, str]:
    tok = create_jwt(user_id, role, _TEST_JWT_SECRET)
    return {"Authorization": f"Bearer {tok}"}


class TestAuditLogAccessControl:
    def test_audit_log_not_accessible_in_single_user_mode(self, engine) -> None:
        """In single-user mode, the audit log endpoint is not registered and returns 404."""
        client = _single_user_client(engine)

        resp = client.get("/api/v1/audit-log")

        assert resp.status_code == 404

    def test_audit_log_any_user_can_read_all_entries(self, engine, monkeypatch) -> None:
        """In multi-user mode, user 2 can read audit entries created by user 1's actions.

        This documents the IDOR / over-broad access vulnerability: the audit log
        endpoint applies no per-user scoping, so any authenticated user can read the
        full audit history of every other user.
        """
        # Seed an audit entry attributed to user 1
        with Session(engine) as sess:
            log_audit(sess, "migration", 1, "create", user_id=1)
            sess.commit()

        client = _multi_user_client(engine, monkeypatch)

        # User 2 reads the audit log — they should see user 1's entry
        resp = client.get("/api/v1/audit-log", headers=_auth_header(2))

        assert resp.status_code == 200
        data = resp.json()
        # The entry belongs to user_id=1, yet user 2 can see it — IDOR confirmed
        assert data["total"] >= 1
        user_ids_in_response = {item["user_id"] for item in data["items"]}
        assert 1 in user_ids_in_response, (
            "User 2 must be able to read user 1's audit entries "
            "(documents the IDOR vulnerability in audit log access)"
        )

    def test_audit_log_user_id_filter_param(self, engine, monkeypatch) -> None:
        """The user_id query parameter filters audit log entries to a specific user."""
        # Seed entries for two different users
        with Session(engine) as sess:
            log_audit(sess, "migration", 1, "create", user_id=10)
            log_audit(sess, "migration", 2, "create", user_id=20)
            log_audit(sess, "migration", 3, "complete", user_id=10)
            sess.commit()

        client = _multi_user_client(engine, monkeypatch)

        # Filter to user_id=10
        resp = client.get("/api/v1/audit-log?user_id=10", headers=_auth_header(10))

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2, f"Expected 2 entries for user_id=10, got {data['total']}"
        for item in data["items"]:
            assert item["user_id"] == 10, (
                f"Filter returned entry with user_id={item['user_id']}, expected 10"
            )

        # Filter to user_id=20
        resp2 = client.get("/api/v1/audit-log?user_id=20", headers=_auth_header(20))
        data2 = resp2.json()
        assert data2["total"] == 1
        assert data2["items"][0]["user_id"] == 20
