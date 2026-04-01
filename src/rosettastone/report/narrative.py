"""Executive narrative generator — LLM-powered migration summary."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader

from rosettastone.report.executive_prompt import format_executive_prompt

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.types import MigrationResult

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    """Get a value from config whether it's a dict or a Pydantic model."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _stats_to_dict(stats: Any) -> dict[str, Any]:
    """Convert a TypeStats dataclass (or dict) to a plain dict."""
    if isinstance(stats, dict):
        return stats
    if hasattr(stats, "__dataclass_fields__"):
        import dataclasses

        return dataclasses.asdict(stats)
    return {}


# Prompt for generating executive narrative — NO prompt/response content, scores only
EXECUTIVE_PROMPT = (
    "You are a technical writer creating an executive summary "
    "of an LLM model migration evaluation.\n"
    "\n"
    "## Migration Context\n"
    "- Source model: {source_model}\n"
    "- Target model: {target_model}\n"
    "- Recommendation: {recommendation}\n"
    "- Overall confidence: {confidence_score:.1%}\n"
    "- Baseline score: {baseline_score:.1%}\n"
    "- Improvement: +{improvement:.1%}\n"
    "- Total cost: ${cost_usd:.4f}\n"
    "- Duration: {duration_seconds:.1f}s\n"
    "- Total test cases: {total_test_cases}\n"
    "- Win count: {wins}\n"
    "\n"
    "## Per-Type Performance\n"
    "{per_type_summary}\n"
    "\n"
    "## Safety Findings\n"
    "{safety_summary}\n"
    "\n"
    "## Warnings\n"
    "{warnings_summary}\n"
    "\n"
    "## Instructions\n"
    "Write a 3-5 paragraph executive summary that:\n"
    "1. States the recommendation clearly with confidence level\n"
    "2. Highlights which output types performed best and worst\n"
    "3. Notes any safety concerns or caveats\n"
    "4. Provides actionable next steps\n"
    "5. Is honest about limitations — don't cherry-pick positive results\n"
    "\n"
    "Keep the tone professional and confident but not promotional. "
    "If the recommendation is NO_GO or CONDITIONAL, lead with the concerns.\n"
    "\n"
    "Do NOT include any raw prompt or response content — "
    "only discuss scores, metrics, and structural findings."
)


def _format_per_type(per_type_scores: dict[str, Any]) -> str:
    """Format per-type scores into a readable summary."""
    if not per_type_scores:
        return "No per-type breakdown available."
    lines = []
    for type_name, stats in per_type_scores.items():
        if isinstance(stats, dict):
            s = stats
        elif hasattr(stats, "__dataclass_fields__"):
            import dataclasses

            s = dataclasses.asdict(stats)
        else:
            continue
        lines.append(
            f"- {type_name}: win_rate={s.get('win_rate', 0):.1%}, "
            f"samples={s.get('sample_count', 0)}, "
            f"mean={s.get('mean', 0):.3f}, p10={s.get('p10', 0):.3f}, "
            f"p50={s.get('p50', 0):.3f}, p90={s.get('p90', 0):.3f}"
        )
    return "\n".join(lines) or "No per-type data."


def _format_safety(safety_warnings: list[Any]) -> str:
    """Format safety warnings into a readable summary."""
    if not safety_warnings:
        return "No safety issues found."
    lines = []
    for w in safety_warnings:
        if isinstance(w, dict):
            lines.append(f"- [{w.get('severity', 'INFO')}] {w.get('message', str(w))}")
        elif hasattr(w, "severity"):
            lines.append(f"- [{w.severity}] {w.message}")
        else:
            lines.append(f"- {w}")
    return "\n".join(lines)


def generate_executive_narrative(
    result: MigrationResult,
    config: MigrationConfig | None = None,
    local_only: bool = False,
) -> str:
    """Generate an executive narrative summary of migration results.

    Uses LiteLLM to call an LLM with structured scores/metadata only (NO prompt content).
    Falls back to a Jinja2 template if local_only=True or if the LLM call fails.

    Args:
        result: Migration results with scores and metadata.
        config: Optional migration config for additional context.
        local_only: If True, skip LLM call and use template fallback.

    Returns:
        Executive narrative as a string.
    """
    per_type = getattr(result, "per_type_scores", {}) or {}
    safety = getattr(result, "safety_warnings", []) or []
    warnings = result.warnings or []

    if local_only:
        logger.info("Executive narrative: using %s", "template fallback")
        return _template_fallback(result, per_type, safety)

    llm_succeeded = False
    try:
        import litellm

        messages = format_executive_prompt(
            source_model=_config_get(result.config, "source_model", "unknown"),
            target_model=_config_get(result.config, "target_model", "unknown"),
            recommendation=getattr(result, "recommendation", "N/A") or "N/A",
            confidence_score=result.confidence_score,
            baseline_score=result.baseline_score,
            improvement=result.improvement,
            cost_usd=result.cost_usd,
            duration_seconds=result.duration_seconds,
            total_test_cases=len(result.validation_results),
            wins=sum(1 for r in result.validation_results if r.is_win),
            per_type_scores=per_type,
            safety_warnings=safety,
            warnings=warnings,
        )

        judge_model = getattr(config, "judge_model", None) if config else None
        response = litellm.completion(
            model=judge_model or "openai/gpt-4o",
            messages=messages,
            max_tokens=1000,
            temperature=0.3,
        )

        narrative = response.choices[0].message.content
        if narrative:
            llm_succeeded = True
            logger.info("Executive narrative: using %s", "LLM")
            return narrative

        logger.info("Executive narrative: using %s", "template fallback")
        return _template_fallback(result, per_type, safety)

    except Exception:
        logger.warning(
            "Executive narrative LLM call failed, using template fallback", exc_info=True
        )
        fallback_or_llm = "LLM" if llm_succeeded else "template fallback"
        logger.info("Executive narrative: using %s", fallback_or_llm)
        return _template_fallback(result, per_type, safety)


def _template_fallback(
    result: MigrationResult,
    per_type: dict[str, Any],
    safety: list[Any],
) -> str:
    """Render the executive narrative using a Jinja2 template (no LLM needed)."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    try:
        template = env.get_template("executive.md.jinja")
    except Exception:
        # If template doesn't exist, return a basic summary
        return _basic_summary(result, per_type)

    converted = {k: _stats_to_dict(v) for k, v in per_type.items()}

    return template.render(
        source_model=_config_get(result.config, "source_model", "unknown"),
        target_model=_config_get(result.config, "target_model", "unknown"),
        recommendation=getattr(result, "recommendation", "N/A"),
        recommendation_reasoning=getattr(result, "recommendation_reasoning", ""),
        confidence_score=result.confidence_score,
        baseline_score=result.baseline_score,
        improvement=result.improvement,
        cost_usd=result.cost_usd,
        duration_seconds=result.duration_seconds,
        total_test_cases=len(result.validation_results),
        wins=sum(1 for r in result.validation_results if r.is_win),
        per_type_scores=converted,
        safety_warnings=safety,
        warnings=result.warnings,
    )


def _basic_summary(result: MigrationResult, per_type: dict[str, Any] | None = None) -> str:
    """Minimal plain-text summary when no template is available."""
    rec = getattr(result, "recommendation", "N/A") or "N/A"
    source = _config_get(result.config, "source_model", "unknown")
    target = _config_get(result.config, "target_model", "unknown")
    wins = sum(1 for r in result.validation_results if r.is_win)
    total = len(result.validation_results)

    return (
        f"Migration from {source} to {target}: {rec}\n\n"
        f"Confidence: {result.confidence_score:.1%}, "
        f"Baseline: {result.baseline_score:.1%}, "
        f"Improvement: +{result.improvement:.1%}\n"
        f"Win rate: {wins}/{total} ({wins / max(total, 1):.0%})\n"
        f"Cost: ${result.cost_usd:.4f}, Duration: {result.duration_seconds:.1f}s"
    )
