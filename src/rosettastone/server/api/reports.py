"""Report download endpoints."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlmodel import Session

from rosettastone.server.database import get_session
from rosettastone.server.models import MigrationRecord

router = APIRouter()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_migration_or_404(migration_id: int, session: Session) -> MigrationRecord:
    """Fetch a migration by ID or raise 404."""
    record = session.get(MigrationRecord, migration_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Migration not found")
    return record


def _reconstruct_migration_result(record: MigrationRecord) -> dict:
    """Reconstruct a minimal MigrationResult dict from DB record."""
    config = json.loads(record.config_json)
    per_type_scores = json.loads(record.per_type_scores_json)
    warnings = json.loads(record.warnings_json)
    safety_warnings = json.loads(record.safety_warnings_json)

    return {
        "config": config,
        "optimized_prompt": record.optimized_prompt or "",
        "baseline_results": [],
        "validation_results": [],
        "confidence_score": record.confidence_score or 0.0,
        "baseline_score": record.baseline_score or 0.0,
        "improvement": record.improvement or 0.0,
        "cost_usd": record.cost_usd,
        "duration_seconds": record.duration_seconds,
        "warnings": warnings,
        "safety_warnings": safety_warnings,
        "recommendation": record.recommendation,
        "recommendation_reasoning": record.recommendation_reasoning,
        "per_type_scores": per_type_scores,
    }


# ---------------------------------------------------------------------------
# JSON API endpoints
# ---------------------------------------------------------------------------


@router.get("/api/v1/migrations/{migration_id}/report/markdown")
async def get_markdown_report(
    migration_id: int,
    session: Session = Depends(get_session),
) -> Response:
    """Generate and return a markdown migration report."""
    record = _get_migration_or_404(migration_id, session)

    result_dict = _reconstruct_migration_result(record)

    # Try to use the existing report generator
    try:
        from rosettastone.core.types import MigrationResult
        from rosettastone.report.markdown import generate_markdown_report

        migration_result = MigrationResult.model_validate(result_dict)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = generate_markdown_report(migration_result, Path(tmp_dir))
            content = output_path.read_text()

        return Response(
            content=content,
            media_type="text/markdown",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="migration_{migration_id}_report.md"'
                )
            },
        )
    except ImportError:
        # Fall back to a simple markdown report
        md_lines = [
            f"# Migration Report #{migration_id}",
            "",
            f"**Source:** {record.source_model}",
            f"**Target:** {record.target_model}",
            f"**Status:** {record.status}",
            f"**Recommendation:** {record.recommendation or 'N/A'}",
            "",
            f"**Confidence Score:** {record.confidence_score or 'N/A'}",
            f"**Baseline Score:** {record.baseline_score or 'N/A'}",
            f"**Improvement:** {record.improvement or 'N/A'}",
            f"**Cost (USD):** ${record.cost_usd:.2f}",
            f"**Duration:** {record.duration_seconds:.1f}s",
            "",
        ]

        warnings = json.loads(record.warnings_json)
        if warnings:
            md_lines.append("## Warnings")
            for w in warnings:
                md_lines.append(f"- {w}")
            md_lines.append("")

        content = "\n".join(md_lines)
        return Response(
            content=content,
            media_type="text/markdown",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="migration_{migration_id}_report.md"'
                )
            },
        )


@router.get("/api/v1/migrations/{migration_id}/report/pdf")
async def get_pdf_report(
    migration_id: int,
    session: Session = Depends(get_session),
) -> Response:
    """Generate and return a PDF migration report."""
    _get_migration_or_404(migration_id, session)

    try:
        import weasyprint  # noqa: F401
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="PDF generation requires weasyprint. Install with: pip install weasyprint",
        )

    # If weasyprint is available, still return 501 until full implementation
    raise HTTPException(
        status_code=501,
        detail="PDF report generation not yet implemented",
    )


@router.get("/api/v1/migrations/{migration_id}/report/html")
async def get_html_report(
    migration_id: int,
    session: Session = Depends(get_session),
) -> Response:
    """Return an HTML migration report."""
    _get_migration_or_404(migration_id, session)
    raise HTTPException(
        status_code=501,
        detail="HTML report generation not yet implemented",
    )


@router.get("/api/v1/migrations/{migration_id}/report/executive")
async def get_executive_report(
    migration_id: int,
    session: Session = Depends(get_session),
) -> Response:
    """Return an executive narrative report."""
    _get_migration_or_404(migration_id, session)
    raise HTTPException(
        status_code=501,
        detail="Executive report generation not yet implemented",
    )
