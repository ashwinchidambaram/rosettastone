"""Registered model CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from rosettastone.server.database import get_session
from rosettastone.server.models import MigrationRecord, RegisteredModel

router = APIRouter()


# ---------------------------------------------------------------------------
# LiteLLM model info helper
# ---------------------------------------------------------------------------


def _get_model_info(model_id: str) -> dict:
    """Fetch model metadata from LiteLLM. Returns safe defaults on failure."""
    try:
        import litellm  # type: ignore[import-untyped]

        info = litellm.get_model_info(model_id)
        context_tokens = info.get("max_input_tokens", 0) or 0
        context = f"{context_tokens // 1000}K" if context_tokens else "Unknown"
        input_cost = info.get("input_cost_per_token", 0) or 0
        cost_per_1m = f"${input_cost * 1_000_000:.2f}" if input_cost else "Unknown"
        provider = model_id.split("/")[0].title() if "/" in model_id else "Unknown"
        return {"context": context, "cost_per_1m": cost_per_1m, "provider": provider}
    except Exception:
        provider = model_id.split("/")[0].title() if "/" in model_id else "Unknown"
        return {"context": "Unknown", "cost_per_1m": "Unknown", "provider": provider}


def _model_to_template_dict(record: RegisteredModel) -> dict:
    """Convert a RegisteredModel record to the dict shape templates expect."""
    info = _get_model_info(record.model_id)
    return {
        "id": record.model_id,
        "db_id": record.id,
        "provider": info["provider"],
        "status": "active" if record.is_active else "deprecated",
        "context": info["context"],
        "cost_per_1m": info["cost_per_1m"],
    }


# ---------------------------------------------------------------------------
# JSON API endpoints
# ---------------------------------------------------------------------------


@router.get("/api/v1/models")
async def list_models(session: Session = Depends(get_session)) -> list[dict]:
    """List all registered models with metadata."""
    stmt = select(RegisteredModel).order_by(RegisteredModel.added_at.desc())  # type: ignore[union-attr]
    records = list(session.exec(stmt).all())
    return [_model_to_template_dict(r) for r in records]


@router.post("/api/v1/models", status_code=201)
async def register_model(
    request_body: dict,
    session: Session = Depends(get_session),
) -> dict:
    """Register a new model."""
    model_id = request_body.get("model_id", "").strip()
    if not model_id:
        raise HTTPException(status_code=422, detail="model_id is required")

    # Check for duplicate
    existing = session.exec(
        select(RegisteredModel).where(RegisteredModel.model_id == model_id)
    ).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Model '{model_id}' is already registered")

    record = RegisteredModel(model_id=model_id)
    session.add(record)
    session.commit()
    session.refresh(record)
    return _model_to_template_dict(record)


@router.delete("/api/v1/models/{model_db_id}", status_code=200)
async def delete_model(
    model_db_id: int,
    session: Session = Depends(get_session),
) -> dict:
    """Remove a registered model by its DB id."""
    record = session.get(RegisteredModel, model_db_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Model not found")
    session.delete(record)
    session.commit()
    return {"deleted": True, "model_id": record.model_id}


@router.get("/api/v1/models/{model_db_id}/info")
async def get_model_info(
    model_db_id: int,
    session: Session = Depends(get_session),
) -> dict:
    """Get model metadata (proxied from LiteLLM) by DB id."""
    record = session.get(RegisteredModel, model_db_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return _get_model_info(record.model_id)


@router.post("/api/v1/models/import-from-migrations")
async def import_from_migrations(session: Session = Depends(get_session)) -> dict:
    """Auto-register models found in existing migration records."""
    migrations = list(session.exec(select(MigrationRecord)).all())
    candidate_ids: set[str] = set()
    for m in migrations:
        if m.source_model:
            candidate_ids.add(m.source_model)
        if m.target_model:
            candidate_ids.add(m.target_model)

    # Find which are already registered
    existing_rows = list(session.exec(select(RegisteredModel)).all())
    existing = {r.model_id for r in existing_rows}

    imported = 0
    for model_id in sorted(candidate_ids):
        if model_id not in existing:
            session.add(RegisteredModel(model_id=model_id))
            imported += 1

    if imported:
        session.commit()

    return {"imported": imported, "total_registered": len(existing) + imported}
