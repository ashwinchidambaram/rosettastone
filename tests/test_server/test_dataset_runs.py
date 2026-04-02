"""Tests for dataset generation run tracking API endpoints."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
sqlmodel = pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient  # noqa: E402
from sqlmodel import Session  # noqa: E402

from rosettastone.server.models import DatasetGenerationRun  # noqa: E402 isort:skip


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _create_run(
    session: Session,
    dataset_name: str = "test_dataset",
    source_model: str = "openai/gpt-4o",
    status: str = "running",
) -> DatasetGenerationRun:
    run = DatasetGenerationRun(
        dataset_name=dataset_name,
        source_model=source_model,
        status=status,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateRun:
    def test_create_run(self, client: TestClient, engine) -> None:
        """POST creates a run, returns id, run is in DB with status='running'."""
        resp = client.post(
            "/api/v1/dataset-runs",
            json={"dataset_name": "fintech_extraction", "source_model": "openai/gpt-4o"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        run_id = data["id"]
        assert isinstance(run_id, int)

        # Verify it's in the DB with the right defaults
        with Session(engine) as session:
            run = session.get(DatasetGenerationRun, run_id)
            assert run is not None
            assert run.dataset_name == "fintech_extraction"
            assert run.source_model == "openai/gpt-4o"
            assert run.status == "running"
            assert run.tuning_cost_usd == 0.0
            assert run.production_cost_usd == 0.0
            assert run.total_cost_usd == 0.0
            assert run.pairs_generated == 0


class TestPatchRun:
    def test_patch_updates_costs(self, client: TestClient, engine) -> None:
        """PATCH with tuning_cost and production_cost -> total_cost_usd = sum."""
        with Session(engine) as session:
            run = _create_run(session)
            run_id = run.id

        resp = client.patch(
            f"/api/v1/dataset-runs/{run_id}",
            json={"tuning_cost_usd": 1.25, "production_cost_usd": 3.50},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert abs(data["tuning_cost_usd"] - 1.25) < 1e-6
        assert abs(data["production_cost_usd"] - 3.50) < 1e-6
        assert abs(data["total_cost_usd"] - 4.75) < 1e-6

    def test_patch_updates_status(self, client: TestClient, engine) -> None:
        """PATCH status field to 'complete'."""
        with Session(engine) as session:
            run = _create_run(session)
            run_id = run.id

        resp = client.patch(
            f"/api/v1/dataset-runs/{run_id}",
            json={"status": "complete", "pairs_generated": 500},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "complete"
        assert data["pairs_generated"] == 500

    def test_patch_404_on_unknown_id(self, client: TestClient) -> None:
        """PATCH with non-existent id returns 404."""
        resp = client.patch(
            "/api/v1/dataset-runs/999999",
            json={"status": "complete"},
        )
        assert resp.status_code == 404

    def test_patch_partial_cost_update_recalculates_total(
        self, client: TestClient, engine
    ) -> None:
        """PATCH with only tuning_cost recalculates total from both fields."""
        with Session(engine) as session:
            run = _create_run(session)
            run.production_cost_usd = 2.00
            run.total_cost_usd = 2.00
            session.add(run)
            session.commit()
            run_id = run.id

        resp = client.patch(
            f"/api/v1/dataset-runs/{run_id}",
            json={"tuning_cost_usd": 1.00},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert abs(data["total_cost_usd"] - 3.00) < 1e-6


class TestListRuns:
    def test_list_returns_newest_first(self, client: TestClient, engine) -> None:
        """Create 2 runs, GET returns them newest-first."""
        with Session(engine) as session:
            _create_run(session, dataset_name="first")
            _create_run(session, dataset_name="second")

        resp = client.get("/api/v1/dataset-runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) >= 2

        # Newest (second insert) should come first
        names = [r["dataset_name"] for r in runs]
        idx_a = names.index("first")
        idx_b = names.index("second")
        assert idx_b < idx_a, "Expected 'second' (newer) to appear before 'first'"

    def test_list_empty_when_no_runs(self, client: TestClient) -> None:
        resp = client.get("/api/v1/dataset-runs")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# Client helper: silent on server unavailable
# ---------------------------------------------------------------------------


class TestClientSilentOnServerUnavailable:
    def test_report_run_start_silent(self, monkeypatch) -> None:
        """report_run_start returns None without raising when server is unreachable."""
        import sys

        monkeypatch.setenv("ROSETTASTONE_SERVER_URL", "http://127.0.0.1:19999")

        # Force reload of the module so it picks up the new env var
        scripts_path = str(
            __import__("pathlib").Path(__file__).parent.parent.parent / "scripts"
        )
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)

        import importlib

        import dataset_cost_client

        importlib.reload(dataset_cost_client)

        result = dataset_cost_client.report_run_start("test_ds", "openai/gpt-4o")
        assert result is None

    def test_report_run_update_silent(self, monkeypatch) -> None:
        """report_run_update does not raise when server is unreachable."""
        import sys

        monkeypatch.setenv("ROSETTASTONE_SERVER_URL", "http://127.0.0.1:19999")

        scripts_path = str(
            __import__("pathlib").Path(__file__).parent.parent.parent / "scripts"
        )
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)

        import importlib

        import dataset_cost_client

        importlib.reload(dataset_cost_client)

        # Should not raise; returns None implicitly
        dataset_cost_client.report_run_update(
            999,
            tuning_cost=1.0,
            production_cost=2.0,
            pairs=100,
            status="complete",
            output_path="/tmp/out.jsonl",
        )
