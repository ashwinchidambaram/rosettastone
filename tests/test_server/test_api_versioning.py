"""Tests for the versioning API router and create_version helper."""

from __future__ import annotations

import json

import pytest
from sqlmodel import Session

from rosettastone.server.api.versioning import create_version
from rosettastone.server.models import MigrationRecord


class TestCreateVersion:
    def test_creates_version_from_migration(self, session, sample_migration):
        """create_version snapshots the current migration state."""
        version = create_version(sample_migration.id, session)
        session.commit()
        session.refresh(version)

        assert version.migration_id == sample_migration.id
        assert version.version_number == 1
        assert version.confidence_score == sample_migration.confidence_score
        assert version.optimized_prompt == sample_migration.optimized_prompt
        assert version.created_by == "system"

        snapshot = json.loads(version.snapshot_json)
        assert snapshot["source_model"] == "openai/gpt-4o"
        assert snapshot["recommendation"] == "GO"

    def test_increments_version_number(self, session, sample_migration):
        """Each call creates a new version with incremented number."""
        v1 = create_version(sample_migration.id, session)
        session.commit()
        v2 = create_version(sample_migration.id, session)
        session.commit()

        session.refresh(v1)
        session.refresh(v2)
        assert v1.version_number == 1
        assert v2.version_number == 2

    def test_raises_on_missing_migration(self, session):
        """create_version raises ValueError for non-existent migration."""
        with pytest.raises(ValueError, match="not found"):
            create_version(999, session)


class TestListVersions:
    def test_empty_list(self, client, sample_migration):
        """GET versions returns empty when no versions exist."""
        resp = client.get(f"/api/v1/migrations/{sample_migration.id}/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_after_create(self, client, engine, sample_migration):
        """GET versions returns created versions."""
        with Session(engine) as s:
            create_version(sample_migration.id, s)
            create_version(sample_migration.id, s)
            s.commit()

        resp = client.get(f"/api/v1/migrations/{sample_migration.id}/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        # Ordered by version_number DESC
        assert data["items"][0]["version_number"] == 2
        assert data["items"][1]["version_number"] == 1

    def test_404_for_missing_migration(self, client):
        """GET versions returns 404 for non-existent migration."""
        resp = client.get("/api/v1/migrations/999/versions")
        assert resp.status_code == 404


class TestGetVersion:
    def test_get_version_detail(self, client, engine, sample_migration):
        """GET version detail returns snapshot as parsed dict."""
        with Session(engine) as s:
            v = create_version(sample_migration.id, s)
            s.commit()
            s.refresh(v)
            vid = v.id

        resp = client.get(f"/api/v1/migrations/{sample_migration.id}/versions/{vid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version_number"] == 1
        assert isinstance(data["snapshot"], dict)
        assert data["snapshot"]["source_model"] == "openai/gpt-4o"

    def test_404_wrong_migration(self, client, engine, sample_migration):
        """GET version returns 404 when version doesn't belong to the migration."""
        with Session(engine) as s:
            v = create_version(sample_migration.id, s)
            s.commit()
            s.refresh(v)
            vid = v.id

        resp = client.get(f"/api/v1/migrations/999/versions/{vid}")
        assert resp.status_code == 404


class TestRollback:
    def test_rollback_restores_state(self, client, engine, sample_migration):
        """POST rollback restores migration to version state."""
        # Create a version
        with Session(engine) as s:
            v = create_version(sample_migration.id, s)
            s.commit()
            s.refresh(v)
            vid = v.id

        # Modify the migration
        with Session(engine) as s:
            m = s.get(MigrationRecord, sample_migration.id)
            m.confidence_score = 0.5
            m.recommendation = "NO_GO"
            s.add(m)
            s.commit()

        # Rollback to original version
        resp = client.post(
            f"/api/v1/migrations/{sample_migration.id}/versions/{vid}/rollback"
        )
        assert resp.status_code == 200
        data = resp.json()
        # New version created with incremented number
        assert data["version_number"] == 2

        # Verify migration state restored
        with Session(engine) as s:
            m = s.get(MigrationRecord, sample_migration.id)
            assert m.confidence_score == 0.92  # original value
            assert m.recommendation == "GO"  # original value

    def test_rollback_creates_audit_entry(self, client, engine, sample_migration):
        """POST rollback creates an audit log entry."""
        from rosettastone.server.models import AuditLog

        with Session(engine) as s:
            v = create_version(sample_migration.id, s)
            s.commit()
            s.refresh(v)
            vid = v.id

        client.post(f"/api/v1/migrations/{sample_migration.id}/versions/{vid}/rollback")

        with Session(engine) as s:
            from sqlmodel import select
            audits = list(s.exec(select(AuditLog)).all())
            assert len(audits) == 1
            assert audits[0].action == "rollback"
            assert audits[0].resource_type == "migration"
            details = json.loads(audits[0].details_json)
            assert details["rolled_back_to_version"] == vid

    def test_rollback_404_missing_version(self, client, sample_migration):
        """POST rollback returns 404 for non-existent version."""
        resp = client.post(
            f"/api/v1/migrations/{sample_migration.id}/versions/999/rollback"
        )
        assert resp.status_code == 404
