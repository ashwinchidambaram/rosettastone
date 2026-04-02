"""Lightweight client for reporting dataset generation costs to RosettaStone server."""

from __future__ import annotations

import os

# Server URL from env var, default to localhost
ROSETTASTONE_SERVER_URL = os.environ.get("ROSETTASTONE_SERVER_URL", "http://localhost:8000")


def report_run_start(dataset_name: str, source_model: str) -> int | None:
    """POST to /api/v1/dataset-runs. Returns run_id or None if server unavailable."""
    try:
        import httpx

        resp = httpx.post(
            f"{ROSETTASTONE_SERVER_URL}/api/v1/dataset-runs",
            json={"dataset_name": dataset_name, "source_model": source_model},
            timeout=5.0,
        )
        resp.raise_for_status()
        return int(resp.json()["id"])
    except Exception:
        return None


def report_run_update(
    run_id: int,
    *,
    tuning_cost: float = 0.0,
    production_cost: float = 0.0,
    pairs: int = 0,
    status: str = "running",
    output_path: str = "",
) -> None:
    """PATCH /api/v1/dataset-runs/{run_id}. Silent no-op if server unavailable."""
    try:
        import httpx

        payload: dict = {}
        if tuning_cost != 0.0:
            payload["tuning_cost_usd"] = tuning_cost
        if production_cost != 0.0:
            payload["production_cost_usd"] = production_cost
        if pairs != 0:
            payload["pairs_generated"] = pairs
        if status != "running":
            payload["status"] = status
        if output_path != "":
            payload["output_path"] = output_path

        if not payload:
            return

        resp = httpx.patch(
            f"{ROSETTASTONE_SERVER_URL}/api/v1/dataset-runs/{run_id}",
            json=payload,
            timeout=5.0,
        )
        resp.raise_for_status()
    except Exception:
        return
