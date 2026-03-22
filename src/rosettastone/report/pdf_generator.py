"""PDF report generator — converts HTML report to PDF via weasyprint."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from rosettastone.core.types import MigrationResult

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _stats_to_dict(stats: Any) -> dict[str, Any]:
    """Convert a TypeStats dataclass (or dict) to a plain dict."""
    if isinstance(stats, dict):
        return stats
    if hasattr(stats, "__dataclass_fields__"):
        import dataclasses

        return dataclasses.asdict(stats)
    return {}


def generate_pdf_report(result: MigrationResult, output_dir: Path) -> Path:
    """Generate a PDF migration report.

    Uses weasyprint to convert the HTML report template to PDF.
    Raises ImportError if weasyprint is not installed.

    Args:
        result: The migration result to report on.
        output_dir: Directory to write the PDF file to.

    Returns:
        Path to the generated PDF file.
    """
    try:
        import weasyprint
    except ImportError:
        raise ImportError(
            "weasyprint is required for PDF generation. "
            "Install with: uv pip install 'rosettastone[web]'"
        )

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("report.html.jinja")

    raw_per_type = getattr(result, "per_type_scores", {}) or {}
    per_type_scores = {k: _stats_to_dict(v) for k, v in raw_per_type.items()}

    html_content = template.render(
        config=result.config,
        optimized_prompt=result.optimized_prompt,
        confidence_score=result.confidence_score,
        baseline_score=result.baseline_score,
        improvement=result.improvement,
        cost_usd=result.cost_usd,
        duration_seconds=result.duration_seconds,
        baseline_results=result.baseline_results,
        validation_results=result.validation_results,
        warnings=result.warnings,
        total_test_cases=len(result.validation_results),
        wins=sum(1 for r in result.validation_results if r.is_win),
        recommendation=getattr(result, "recommendation", None),
        recommendation_reasoning=getattr(result, "recommendation_reasoning", None),
        per_type_scores=per_type_scores,
        safety_warnings=getattr(result, "safety_warnings", []),
        embed_mode=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "migration_report.pdf"

    doc = weasyprint.HTML(string=html_content)
    doc.write_pdf(str(output_path))

    return output_path
