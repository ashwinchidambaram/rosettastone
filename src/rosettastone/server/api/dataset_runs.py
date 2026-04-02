"""Dataset generation run tracking endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from rosettastone.server.database import get_session
from rosettastone.server.models import DatasetGenerationRun

router = APIRouter()


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class DatasetRunCreate(BaseModel):
    dataset_name: str
    source_model: str


class DatasetRunUpdate(BaseModel):
    tuning_cost_usd: float | None = None
    production_cost_usd: float | None = None
    pairs_generated: int | None = None
    status: str | None = None
    output_path: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/api/v1/dataset-runs")
async def create_dataset_run(
    body: DatasetRunCreate,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Create a new dataset generation run with status='running'."""
    run = DatasetGenerationRun(
        dataset_name=body.dataset_name,
        source_model=body.source_model,
        status="running",
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return {"id": run.id}


@router.get("/api/v1/dataset-runs")
async def list_dataset_runs(
    session: Session = Depends(get_session),
) -> list[DatasetGenerationRun]:
    """Return all dataset generation runs, newest first."""
    stmt = select(DatasetGenerationRun).order_by(
        DatasetGenerationRun.created_at.desc()  # type: ignore[attr-defined]
    )
    return list(session.exec(stmt).all())


@router.patch("/api/v1/dataset-runs/{run_id}")
async def update_dataset_run(
    run_id: int,
    body: DatasetRunUpdate,
    session: Session = Depends(get_session),
) -> DatasetGenerationRun:
    """Update fields on an existing dataset generation run."""
    run = session.get(DatasetGenerationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Dataset run {run_id} not found")

    if body.tuning_cost_usd is not None:
        run.tuning_cost_usd = body.tuning_cost_usd
    if body.production_cost_usd is not None:
        run.production_cost_usd = body.production_cost_usd
    if body.pairs_generated is not None:
        run.pairs_generated = body.pairs_generated
    if body.status is not None:
        run.status = body.status
    if body.output_path is not None:
        run.output_path = body.output_path

    # Recalculate total whenever either cost field was touched
    if body.tuning_cost_usd is not None or body.production_cost_usd is not None:
        run.total_cost_usd = run.tuning_cost_usd + run.production_cost_usd

    session.add(run)
    session.commit()
    session.refresh(run)
    return run
