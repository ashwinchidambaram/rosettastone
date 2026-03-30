"""Tests for the A/B testing API router."""

from __future__ import annotations

from sqlmodel import Session

from rosettastone.server.api.versioning import create_version


class TestCreateABTest:
    def test_create(self, client, engine, sample_migration):
        """POST /api/v1/ab-tests creates a test."""
        # Create two versions first
        with Session(engine) as s:
            v1 = create_version(sample_migration.id, s)
            v2 = create_version(sample_migration.id, s)
            s.commit()
            s.refresh(v1)
            s.refresh(v2)

        resp = client.post("/api/v1/ab-tests", json={
            "migration_id": sample_migration.id,
            "version_a_id": v1.id,
            "version_b_id": v2.id,
            "name": "Test AB",
            "traffic_split": 0.6,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test AB"
        assert data["status"] == "draft"

    def test_create_missing_migration(self, client):
        """POST with non-existent migration returns 404."""
        resp = client.post("/api/v1/ab-tests", json={
            "migration_id": 999,
            "version_a_id": 1,
            "version_b_id": 2,
        })
        assert resp.status_code == 404

    def test_create_missing_version(self, client, engine, sample_migration):
        """POST with non-existent version returns 404."""
        with Session(engine) as s:
            v1 = create_version(sample_migration.id, s)
            s.commit()
            s.refresh(v1)

        resp = client.post("/api/v1/ab-tests", json={
            "migration_id": sample_migration.id,
            "version_a_id": v1.id,
            "version_b_id": 999,
        })
        assert resp.status_code == 404


class TestListABTests:
    def test_empty_list(self, client):
        """GET /api/v1/ab-tests returns empty list."""
        resp = client.get("/api/v1/ab-tests")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_with_data(self, client, engine, sample_migration):
        """GET returns created tests."""
        with Session(engine) as s:
            v1 = create_version(sample_migration.id, s)
            v2 = create_version(sample_migration.id, s)
            s.commit()
            s.refresh(v1)
            s.refresh(v2)

        client.post("/api/v1/ab-tests", json={
            "migration_id": sample_migration.id,
            "version_a_id": v1.id,
            "version_b_id": v2.id,
        })

        resp = client.get("/api/v1/ab-tests")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1


class TestABTestLifecycle:
    def _create_test(self, client, engine, sample_migration):
        """Helper: create versions and an A/B test."""
        with Session(engine) as s:
            v1 = create_version(sample_migration.id, s)
            v2 = create_version(sample_migration.id, s)
            s.commit()
            s.refresh(v1)
            s.refresh(v2)

        resp = client.post("/api/v1/ab-tests", json={
            "migration_id": sample_migration.id,
            "version_a_id": v1.id,
            "version_b_id": v2.id,
            "name": "Lifecycle test",
        })
        return resp.json()["id"]

    def test_start(self, client, engine, sample_migration):
        """POST /start transitions from draft to running."""
        test_id = self._create_test(client, engine, sample_migration)

        resp = client.post(f"/api/v1/ab-tests/{test_id}/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_start_wrong_status(self, client, engine, sample_migration):
        """POST /start on a non-draft test returns 400."""
        test_id = self._create_test(client, engine, sample_migration)
        client.post(f"/api/v1/ab-tests/{test_id}/start")

        resp = client.post(f"/api/v1/ab-tests/{test_id}/start")
        assert resp.status_code == 400

    def test_conclude(self, client, engine, sample_migration):
        """POST /conclude transitions running test to concluded."""
        test_id = self._create_test(client, engine, sample_migration)
        client.post(f"/api/v1/ab-tests/{test_id}/start")

        resp = client.post(f"/api/v1/ab-tests/{test_id}/conclude")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "concluded"
        assert data["winner"] is not None

    def test_conclude_wrong_status(self, client, engine, sample_migration):
        """POST /conclude on a draft test returns 400."""
        test_id = self._create_test(client, engine, sample_migration)

        resp = client.post(f"/api/v1/ab-tests/{test_id}/conclude")
        assert resp.status_code == 400

    def test_metrics_empty(self, client, engine, sample_migration):
        """GET /metrics returns zeros when no results exist."""
        test_id = self._create_test(client, engine, sample_migration)

        resp = client.get(f"/api/v1/ab-tests/{test_id}/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_results"] == 0
        assert data["wins_a"] == 0

    def test_get_detail(self, client, engine, sample_migration):
        """GET /ab-tests/{id} returns detail."""
        test_id = self._create_test(client, engine, sample_migration)

        resp = client.get(f"/api/v1/ab-tests/{test_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == test_id
        assert data["name"] == "Lifecycle test"

    def test_get_404(self, client):
        """GET non-existent test returns 404."""
        resp = client.get("/api/v1/ab-tests/999")
        assert resp.status_code == 404
