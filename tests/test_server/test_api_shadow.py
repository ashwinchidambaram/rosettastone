"""Tests for shadow deployment API endpoints."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi")
sqlmodel = pytest.importorskip("sqlmodel")
yaml = pytest.importorskip("yaml")


class TestShadowConfigDownload:
    def test_shadow_config_returns_yaml(self, client, engine, sample_migration):
        response = client.get(
            f"/api/v1/migrations/{sample_migration.id}/shadow/config.yaml"
        )
        assert response.status_code == 200
        assert "text/yaml" in response.headers["content-type"]

    def test_shadow_config_is_valid_yaml(self, client, engine, sample_migration):
        response = client.get(
            f"/api/v1/migrations/{sample_migration.id}/shadow/config.yaml"
        )
        assert response.status_code == 200
        parsed = yaml.safe_load(response.text)
        assert isinstance(parsed, dict)

    def test_shadow_config_contains_models(self, client, engine, sample_migration):
        response = client.get(
            f"/api/v1/migrations/{sample_migration.id}/shadow/config.yaml"
        )
        assert response.status_code == 200
        parsed = yaml.safe_load(response.text)
        assert parsed["source_model"] == sample_migration.source_model
        assert parsed["target_model"] == sample_migration.target_model

    def test_shadow_config_structure(self, client, engine, sample_migration):
        response = client.get(
            f"/api/v1/migrations/{sample_migration.id}/shadow/config.yaml"
        )
        assert response.status_code == 200
        parsed = yaml.safe_load(response.text)
        # Required structural keys
        assert "primary" in parsed
        assert parsed["primary"] == "source"
        assert "sample_rate" in parsed
        assert "duration_hours" in parsed
        assert "log_path" in parsed
        assert "rollback" in parsed
        assert "endpoints" in parsed
        assert "source" in parsed["endpoints"]
        assert "target" in parsed["endpoints"]

    def test_shadow_config_content_disposition(self, client, engine, sample_migration):
        response = client.get(
            f"/api/v1/migrations/{sample_migration.id}/shadow/config.yaml"
        )
        assert response.status_code == 200
        assert "shadow_config.yaml" in response.headers.get("content-disposition", "")

    def test_shadow_config_optimized_prompt(self, client, session, engine):
        from rosettastone.server.models import MigrationRecord

        migration = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
            optimized_prompt="Use concise answers.",
            cost_usd=0.5,
            duration_seconds=10.0,
            config_json=json.dumps(
                {"source_model": "openai/gpt-4o", "target_model": "anthropic/claude-sonnet-4"}
            ),
            per_type_scores_json="{}",
            warnings_json="[]",
            safety_warnings_json="[]",
        )
        session.add(migration)
        session.commit()
        session.refresh(migration)

        response = client.get(f"/api/v1/migrations/{migration.id}/shadow/config.yaml")
        assert response.status_code == 200
        parsed = yaml.safe_load(response.text)
        assert parsed["optimized_prompt"] == "Use concise answers."

    def test_shadow_config_404(self, client):
        response = client.get("/api/v1/migrations/999/shadow/config.yaml")
        assert response.status_code == 404


class TestShadowEvaluate:
    def _make_jsonl_file(self, entries: list[dict]) -> tuple[str, bytes]:
        """Build an in-memory JSONL file for upload."""
        content = "\n".join(json.dumps(e) for e in entries).encode()
        return ("shadow_logs.jsonl", content)

    def test_evaluate_returns_scores(self, client, engine, sample_migration):
        """Evaluate endpoint returns win_rate and per_type_scores when evaluator works."""
        fake_results = {
            "win_rate": 0.8,
            "total_pairs": 5,
            "wins": 4,
            "non_deterministic_count": 0,
            "cost_usd": 0.01,
            "per_type_scores": {"short_text": {"win_rate": 0.8, "sample_count": 5, "mean": 0.82}},
            "warnings": [],
        }

        jsonl_content = b'{"prompt": "hello", "source_response": "hi", "target_response": "hello there", "source_model": "openai/gpt-4o", "target_model": "anthropic/claude-sonnet-4", "request_id": "r1"}\n'

        with patch(
            "rosettastone.shadow.evaluator.score_shadow_logs", return_value=fake_results
        ):
            response = client.post(
                f"/api/v1/migrations/{sample_migration.id}/shadow/evaluate",
                files={"file": ("shadow.jsonl", jsonl_content, "application/octet-stream")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "win_rate" in data
        assert data["win_rate"] == 0.8
        assert "total_pairs" in data
        assert "per_type_scores" in data

    def test_evaluate_empty_file_returns_422(self, client, engine, sample_migration):
        """Evaluate endpoint returns 422 when uploaded file is empty."""
        response = client.post(
            f"/api/v1/migrations/{sample_migration.id}/shadow/evaluate",
            files={"file": ("shadow.jsonl", b"", "application/octet-stream")},
        )
        assert response.status_code == 422

    def test_evaluate_404_for_unknown_migration(self, client):
        jsonl_content = b'{"prompt": "test"}\n'
        response = client.post(
            "/api/v1/migrations/999/shadow/evaluate",
            files={"file": ("shadow.jsonl", jsonl_content, "application/octet-stream")},
        )
        assert response.status_code == 404

    def test_evaluate_handles_evaluator_exception(self, client, engine, sample_migration):
        """If score_shadow_logs raises, endpoint returns 422."""
        jsonl_content = b'{"prompt": "hello"}\n'

        with patch(
            "rosettastone.shadow.evaluator.score_shadow_logs",
            side_effect=ValueError("bad log format"),
        ):
            response = client.post(
                f"/api/v1/migrations/{sample_migration.id}/shadow/evaluate",
                files={"file": ("shadow.jsonl", jsonl_content, "application/octet-stream")},
            )

        assert response.status_code == 422
        assert "bad log format" in response.json()["detail"]

    def test_evaluate_whitespace_only_file_returns_422(self, client, engine, sample_migration):
        """Whitespace-only file treated as empty."""
        response = client.post(
            f"/api/v1/migrations/{sample_migration.id}/shadow/evaluate",
            files={"file": ("shadow.jsonl", b"   \n  \n", "application/octet-stream")},
        )
        assert response.status_code == 422
