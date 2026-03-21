"""Tests for comparison API endpoints."""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
sqlmodel = pytest.importorskip("sqlmodel")

from sqlmodel import Session  # noqa: E402

from rosettastone.server.models import TestCaseRecord  # noqa: E402


class TestDistributions:
    def test_distributions_with_data(self, client, engine, sample_migration, sample_test_cases):
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/distributions")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1  # only "json" output type
        dist = data[0]
        assert dist["output_type"] == "json"
        assert dist["stats"]["sample_count"] == 5
        assert dist["stats"]["min_score"] == 0.85
        assert dist["stats"]["max_score"] == 0.97
        assert len(dist["histogram"]) == 10
        # All scores are 0.85-0.97, so bucket 8 (0.80-0.90) and 9 (0.90-1.00) should have counts
        assert sum(dist["histogram"]) == 5  # total matches test case count

    def test_distributions_empty(self, client, engine, sample_migration):
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/distributions")
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_distributions_404(self, client):
        response = client.get("/api/v1/migrations/999/distributions")
        assert response.status_code == 404

    def test_distributions_multiple_types(self, client, engine, sample_migration):
        """Test distributions with multiple output types."""
        with Session(engine) as session:
            for i in range(3):
                session.add(
                    TestCaseRecord(
                        migration_id=sample_migration.id,
                        phase="validation",
                        output_type="json",
                        composite_score=0.9 + i * 0.02,
                        is_win=True,
                        scores_json=json.dumps({"bertscore": 0.9}),
                        details_json="{}",
                        response_length=100,
                        new_response_length=100,
                    )
                )
            for i in range(2):
                session.add(
                    TestCaseRecord(
                        migration_id=sample_migration.id,
                        phase="validation",
                        output_type="classification",
                        composite_score=0.8 + i * 0.05,
                        is_win=True,
                        scores_json=json.dumps({"exact_match": 1.0}),
                        details_json="{}",
                        response_length=50,
                        new_response_length=50,
                    )
                )
            session.commit()

        response = client.get(f"/api/v1/migrations/{sample_migration.id}/distributions")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        types = {d["output_type"] for d in data}
        assert types == {"json", "classification"}


class TestDiff:
    def test_diff_no_content(self, client, engine, sample_migration, sample_test_cases):
        """Test diff returns available=False when no content stored."""
        tc = sample_test_cases[0]
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/test-cases/{tc.id}/diff")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is False
        assert data["prompt"] is None
        assert data["source_response"] is None
        assert data["target_response"] is None

    def test_diff_with_content(self, client, engine, sample_migration):
        """Test diff returns content when stored."""
        with Session(engine) as session:
            tc = TestCaseRecord(
                migration_id=sample_migration.id,
                phase="validation",
                output_type="json",
                composite_score=0.9,
                is_win=True,
                scores_json="{}",
                details_json="{}",
                response_length=100,
                new_response_length=95,
                prompt_text="Test prompt",
                response_text="Original response",
                new_response_text="New response",
            )
            session.add(tc)
            session.commit()
            session.refresh(tc)
            tc_id = tc.id

        response = client.get(f"/api/v1/migrations/{sample_migration.id}/test-cases/{tc_id}/diff")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert data["prompt"] == "Test prompt"
        assert data["source_response"] == "Original response"
        assert data["target_response"] == "New response"

    def test_diff_404_missing_test_case(self, client, engine, sample_migration):
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/test-cases/999/diff")
        assert response.status_code == 404

    def test_diff_404_missing_migration(self, client):
        response = client.get("/api/v1/migrations/999/test-cases/1/diff")
        assert response.status_code == 404


class TestUIFragments:
    def test_diff_fragment_fallback(self, client):
        """When migration/tc don't exist, falls back to dummy diff data."""
        response = client.get("/ui/fragments/diff/999/999")
        assert response.status_code == 200
        body = response.text
        # Dummy data has these values; template always renders outer structure
        assert "BERTScore" in body

    def test_diff_fragment_with_real_data(self, client, engine, sample_migration, sample_test_cases):
        """When real TC exists, the fragment renders with DB data."""
        tc = sample_test_cases[0]
        response = client.get(f"/ui/fragments/diff/{sample_migration.id}/{tc.id}")
        assert response.status_code == 200
        body = response.text
        assert "BERTScore" in body
        # Real data: scores rendered as numbers
        assert "0." in body  # composite_score is a float like 0.85

    def test_charts_fragment(self, client):
        response = client.get("/ui/fragments/charts/1")
        assert response.status_code == 200
        assert "Template pending" in response.text
