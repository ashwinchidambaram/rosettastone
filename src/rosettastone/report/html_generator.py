"""HTML report generator — self-contained interactive report with Chart.js.

Chart.js is bundled from the local static asset at build time so the generated
HTML file works fully offline without any CDN dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

if TYPE_CHECKING:
    from rosettastone.core.types import MigrationResult

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Local Chart.js bundle — served from the static directory bundled with the package.
# Falls back to an empty string (charts will not render) if the file is missing.
_CHARTJS_PATH = (
    Path(__file__).parents[2] / "rosettastone" / "server" / "static" / "js" / "chart.min.js"
)


def _stats_to_dict(stats: Any) -> dict[str, Any]:
    """Convert a TypeStats dataclass (or dict) to a plain dict."""
    if isinstance(stats, dict):
        return stats
    if hasattr(stats, "__dataclass_fields__"):
        import dataclasses

        return dataclasses.asdict(stats)
    return {}


def generate_html_report(result: MigrationResult, output_dir: Path) -> Path:
    """Generate a self-contained HTML migration report.

    Renders the report.html.jinja template with Chart.js for interactive
    score distribution charts. Chart.js is inlined from the local static
    bundle so the output file works offline without any network access.

    Args:
        result: The migration result to report on.
        output_dir: Directory to write the HTML file to.

    Returns:
        Path to the generated HTML file.
    """
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("report.html.jinja")

    raw_per_type = getattr(result, "per_type_scores", {}) or {}
    per_type_scores = {k: _stats_to_dict(v) for k, v in raw_per_type.items()}

    chart_js_source = (
        Markup(_CHARTJS_PATH.read_text(encoding="utf-8")) if _CHARTJS_PATH.is_file() else Markup("")
    )

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
        chart_js_source=chart_js_source,
        cost_breakdown=getattr(result, "cost_breakdown", {}),
        # T3: regression fields
        prompt_regressions=getattr(result, "prompt_regressions", []),
        regression_count=getattr(result, "regression_count", 0),
        at_risk_count=getattr(result, "at_risk_count", 0),
        # T4: multi-run fields
        eval_runs=getattr(result, "eval_runs", 1),
        non_deterministic_count=getattr(result, "non_deterministic_count", 0),
        variance_flag_threshold=result.config.get("variance_flag_threshold", 0.1),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "migration_report.html"
    output_path.write_text(html_content)

    return output_path
