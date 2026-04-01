"""Generate markdown migration report."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from rosettastone.core.types import MigrationResult

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _stats_to_dict(stats: Any) -> dict[str, Any]:
    """Convert a TypeStats dataclass (or already-a-dict) to a plain dict for Jinja2."""
    if isinstance(stats, dict):
        return stats
    if hasattr(stats, "__dataclass_fields__"):
        import dataclasses

        d = dataclasses.asdict(stats)
        return d
    return {}


def generate_markdown_report(result: MigrationResult, output_dir: Path) -> Path:
    """Generate a markdown migration report and write it to output_dir."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template("report.md.jinja")

    # Convert per_type_scores dataclasses to plain dicts for Jinja2.
    raw_per_type: dict[str, Any] = getattr(result, "per_type_scores", {}) or {}
    per_type_scores = {k: _stats_to_dict(v) for k, v in raw_per_type.items()}

    report_content = template.render(
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
        # New decision-layer fields (fall back gracefully if absent).
        recommendation=getattr(result, "recommendation", None),
        recommendation_reasoning=getattr(result, "recommendation_reasoning", None),
        per_type_scores=per_type_scores,
        safety_warnings=getattr(result, "safety_warnings", []),
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
    output_path = output_dir / "migration_report.md"
    output_path.write_text(report_content, encoding="utf-8")

    # Also write the optimized prompt
    prompt_path = output_dir / "optimized_prompt.txt"
    prompt_path.write_text(result.optimized_prompt, encoding="utf-8")

    return output_path
