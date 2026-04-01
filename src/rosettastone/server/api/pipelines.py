"""Pipeline migration API router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from sqlmodel import Session, func, select

from rosettastone.server.database import get_session
from rosettastone.server.models import PipelineRecord, PipelineStageRecord
from rosettastone.server.rbac import require_role
from rosettastone.server.schemas import (
    PaginatedResponse,
    PipelineCreate,
    PipelineDetail,
    PipelineStageSummary,
    PipelineSummary,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stage_to_summary(stage: PipelineStageRecord) -> PipelineStageSummary:
    """Convert a PipelineStageRecord to PipelineStageSummary schema."""
    return PipelineStageSummary(
        module_name=stage.module_name,
        status=stage.status,
        optimized_prompt=stage.optimized_prompt,
        score=stage.score,
        duration_seconds=stage.duration_seconds,
    )


def _pipeline_to_summary(pipeline: PipelineRecord) -> PipelineSummary:
    """Convert a PipelineRecord to PipelineSummary schema."""
    return PipelineSummary(
        id=pipeline.id,  # type: ignore[arg-type]
        name=pipeline.name,
        source_model=pipeline.source_model,
        target_model=pipeline.target_model,
        status=pipeline.status,
        created_at=pipeline.created_at,
    )


def _pipeline_to_detail(
    pipeline: PipelineRecord, stages: list[PipelineStageRecord]
) -> PipelineDetail:
    """Convert a PipelineRecord + stages to PipelineDetail schema."""
    return PipelineDetail(
        id=pipeline.id,  # type: ignore[arg-type]
        name=pipeline.name,
        source_model=pipeline.source_model,
        target_model=pipeline.target_model,
        status=pipeline.status,
        created_at=pipeline.created_at,
        stages=[_stage_to_summary(s) for s in stages],
        config_yaml=pipeline.config_yaml,
    )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/pipelines/migrate",
    response_model=PipelineSummary,
    status_code=201,
    dependencies=[Depends(require_role("editor", "admin"))],
)
def create_pipeline(
    body: PipelineCreate,
    request: Request,
    session: Session = Depends(get_session),
) -> PipelineSummary:
    """Submit a YAML pipeline config to create a new PipelineRecord.

    Parses the YAML body to extract pipeline metadata (name, source_model,
    target_model). Returns the created PipelineSummary with status 201.
    Returns 422 if the YAML is invalid or missing required fields.
    """
    import yaml
    from pydantic import ValidationError

    from rosettastone.optimize.pipeline_config import PipelineConfig

    try:
        raw = yaml.safe_load(body.config_yaml)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="YAML must be a mapping at the top level")

    pipeline_data = raw.get("pipeline", raw)

    try:
        config = PipelineConfig(**pipeline_data)
    except (ValidationError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid pipeline config: {exc}") from exc

    pipeline = PipelineRecord(
        name=config.name,
        config_yaml=body.config_yaml,
        source_model=config.source_model,
        target_model=config.target_model,
        status="pending",
    )
    session.add(pipeline)
    session.commit()
    session.refresh(pipeline)

    # Submit to background executor
    from rosettastone.server.pipeline_runner import run_pipeline_background

    executor = request.app.state.executor
    executor.submit(run_pipeline_background, pipeline.id)

    return _pipeline_to_summary(pipeline)


@router.get("/api/v1/pipelines", response_model=PaginatedResponse[PipelineSummary])
def list_pipelines(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> PaginatedResponse[PipelineSummary]:
    """List all pipelines, paginated and ordered by created_at DESC."""
    count_stmt = select(func.count()).select_from(PipelineRecord)
    total = session.exec(count_stmt).one()

    offset = (page - 1) * per_page
    stmt = (
        select(PipelineRecord)
        .order_by(PipelineRecord.created_at.desc())  # type: ignore[attr-defined]
        .offset(offset)
        .limit(per_page)
    )
    records = list(session.exec(stmt).all())

    items = [_pipeline_to_summary(r) for r in records]
    return PaginatedResponse(items=items, total=total, page=page, per_page=per_page)


@router.get("/api/v1/pipelines/{pipeline_id}/status", response_model=PipelineDetail)
def get_pipeline_status(
    pipeline_id: int,
    session: Session = Depends(get_session),
) -> PipelineDetail:
    """Return overall pipeline status plus per-stage progress.

    Returns 404 if the pipeline does not exist.
    """
    pipeline = session.get(PipelineRecord, pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

    stages = list(
        session.exec(
            select(PipelineStageRecord).where(PipelineStageRecord.pipeline_id == pipeline_id)
        ).all()
    )

    return _pipeline_to_detail(pipeline, stages)


@router.get("/api/v1/pipelines/{pipeline_id}/modules", response_model=list[PipelineStageSummary])
def get_pipeline_modules(
    pipeline_id: int,
    session: Session = Depends(get_session),
) -> list[PipelineStageSummary]:
    """Return per-module detail for a pipeline.

    Returns 404 if the pipeline does not exist.
    """
    pipeline = session.get(PipelineRecord, pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

    stages = list(
        session.exec(
            select(PipelineStageRecord).where(PipelineStageRecord.pipeline_id == pipeline_id)
        ).all()
    )

    return [_stage_to_summary(s) for s in stages]


# ---------------------------------------------------------------------------
# UI routes
# ---------------------------------------------------------------------------


@router.get("/ui/pipelines", response_class=HTMLResponse)
async def pipelines_page(
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """List all pipelines."""
    stmt = (
        select(PipelineRecord)
        .order_by(PipelineRecord.created_at.desc())  # type: ignore[attr-defined]
        .limit(100)
    )
    pipelines = list(session.exec(stmt).all())
    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "pipelines.html",
        {"active_nav": "pipelines", "pipelines": pipelines},
    )


@router.get("/ui/pipelines/{pipeline_id}", response_class=HTMLResponse)
async def pipeline_detail_page(
    pipeline_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """Pipeline detail page."""
    pipeline = session.get(PipelineRecord, pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    stages = list(
        session.exec(
            select(PipelineStageRecord).where(PipelineStageRecord.pipeline_id == pipeline_id)
        ).all()
    )

    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "pipeline_detail.html",
        {"active_nav": "pipelines", "pipeline": pipeline, "stages": stages},
    )


@router.get("/ui/pipelines/{pipeline_id}/stages-fragment", response_class=HTMLResponse)
async def pipeline_stages_fragment(
    pipeline_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """HTMX fragment: returns the pipeline stages table for polling updates."""
    pipeline = session.get(PipelineRecord, pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    stages = list(
        session.exec(
            select(PipelineStageRecord).where(PipelineStageRecord.pipeline_id == pipeline_id)
        ).all()
    )

    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "fragments/pipeline_stages.html",
        {"stages": stages},
    )
