"""Generate markdown migration report."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from rosettastone.core.types import MigrationResult

TEMPLATES_DIR = Path(__file__).parent / "templates"


def generate_markdown_report(result: MigrationResult, output_dir: Path) -> Path:
    """Generate a markdown migration report and write it to output_dir."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template("report.md.jinja")

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
    )

    output_path = output_dir / "migration_report.md"
    output_path.write_text(report_content)

    # Also write the optimized prompt
    prompt_path = output_dir / "optimized_prompt.txt"
    prompt_path.write_text(result.optimized_prompt)

    return output_path
