"""Report download endpoints."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlmodel import Session, select

from rosettastone.server.api.utils import _get_migration_or_404
from rosettastone.server.database import get_session
from rosettastone.server.models import MigrationRecord, TestCaseRecord

router = APIRouter()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _reconstruct_migration_result(record: MigrationRecord, session: Session):
    """Reconstruct a MigrationResult from DB record, querying TestCaseRecord rows.

    Returns a validated MigrationResult object. Missing fields (prompt/response
    content not stored unless --store-prompt-content was used) are filled with
    safe placeholder values.
    """
    from rosettastone.core.types import EvalResult, MigrationResult, PromptPair

    config = json.loads(record.config_json)
    per_type_scores = json.loads(record.per_type_scores_json)
    warnings = json.loads(record.warnings_json)
    safety_warnings = json.loads(record.safety_warnings_json)

    source_model = record.source_model

    # Query all test case records for this migration
    test_cases = session.exec(
        select(TestCaseRecord).where(TestCaseRecord.migration_id == record.id)
    ).all()

    def _build_eval_result(tc: TestCaseRecord) -> EvalResult:
        scores = json.loads(tc.scores_json) if tc.scores_json else {}
        details = json.loads(tc.details_json) if tc.details_json else {}

        # Use stored content if available, otherwise use safe placeholders
        prompt_text = tc.prompt_text or ""
        response_text = tc.response_text or ""
        new_response_text = tc.new_response_text or ""

        prompt_pair = PromptPair(
            prompt=prompt_text,
            response=response_text,
            source_model=source_model,
            metadata={"output_type": tc.output_type} if tc.output_type else {},
        )
        return EvalResult(
            prompt_pair=prompt_pair,
            new_response=new_response_text,
            scores=scores,
            composite_score=tc.composite_score,
            is_win=tc.is_win,
            details=details,
        )

    baseline_results = [_build_eval_result(tc) for tc in test_cases if tc.phase == "baseline"]
    validation_results = [_build_eval_result(tc) for tc in test_cases if tc.phase == "validation"]

    result_dict = {
        "config": config,
        "optimized_prompt": record.optimized_prompt or "",
        "baseline_results": [r.model_dump() for r in baseline_results],
        "validation_results": [r.model_dump() for r in validation_results],
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

    return MigrationResult.model_validate(result_dict)


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
    result = _reconstruct_migration_result(record, session)

    # Try to use the existing report generator
    try:
        from rosettastone.report.markdown import generate_markdown_report

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = generate_markdown_report(result, Path(tmp_dir))
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

        warnings_list = json.loads(record.warnings_json)
        if warnings_list:
            md_lines.append("## Warnings")
            for w in warnings_list:
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


@router.get("/api/v1/migrations/{migration_id}/report/html")
async def get_html_report(
    migration_id: int,
    session: Session = Depends(get_session),
) -> Response:
    """Return an HTML migration report."""
    record = _get_migration_or_404(migration_id, session)
    result = _reconstruct_migration_result(record, session)

    try:
        from rosettastone.report.html_generator import generate_html_report

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = generate_html_report(result, Path(tmp_dir))
            content = output_path.read_text()

        return Response(
            content=content,
            media_type="text/html",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="migration_{migration_id}_report.html"'
                )
            },
        )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="HTML report generation requires jinja2",
        )


@router.get("/api/v1/migrations/{migration_id}/report/pdf")
async def get_pdf_report(
    migration_id: int,
    session: Session = Depends(get_session),
) -> Response:
    """Generate and return a PDF migration report."""
    record = _get_migration_or_404(migration_id, session)

    try:
        from rosettastone.report.pdf_generator import generate_pdf_report
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="PDF generation requires weasyprint. Install with: pip install weasyprint",
        )

    result = _reconstruct_migration_result(record, session)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = generate_pdf_report(result, Path(tmp_dir))
            content = output_path.read_bytes()
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="PDF generation requires weasyprint. Install with: pip install weasyprint",
        )

    return Response(
        content=content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (f'attachment; filename="migration_{migration_id}_report.pdf"')
        },
    )


@router.get("/api/v1/migrations/{migration_id}/report/executive")
async def get_executive_report(
    migration_id: int,
    session: Session = Depends(get_session),
) -> Response:
    """Return an executive narrative report."""
    record = _get_migration_or_404(migration_id, session)
    result = _reconstruct_migration_result(record, session)

    try:
        from rosettastone.report.narrative import generate_executive_narrative

        # Use local_only=True by default — no LLM call from the web server
        narrative = generate_executive_narrative(result, local_only=True)
    except ImportError:
        narrative = (
            f"# Executive Summary\n\n"
            f"Migration: {record.source_model} \u2192 {record.target_model}\n"
            f"Recommendation: {record.recommendation or 'N/A'}"
        )

    return Response(
        content=narrative,
        media_type="text/markdown",
        headers={
            "Content-Disposition": (f'attachment; filename="migration_{migration_id}_executive.md"')
        },
    )
