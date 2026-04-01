"""CI/CD output formatters for RosettaStone migration results."""

from __future__ import annotations

import json

from rosettastone.core.types import MigrationResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RECOMMENDATION_EMOJI: dict[str | None, str] = {
    "GO": "✅",
    "CONDITIONAL": "⚠️",
    "NO_GO": "❌",
    None: "❓",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def format_ci_json(result: MigrationResult) -> str:
    """Format MigrationResult as CI-friendly JSON string.

    Includes: recommendation, confidence_score, baseline_score, improvement,
    cost_usd, duration_seconds, warnings, safety_warnings, per_type_scores.
    Floats rounded to 4 decimal places.
    """
    data: dict = {
        "recommendation": result.recommendation,
        "confidence_score": round(result.confidence_score, 4),
        "baseline_score": round(result.baseline_score, 4),
        "improvement": round(result.improvement, 4),
        "cost_usd": round(result.cost_usd, 4),
        "duration_seconds": round(result.duration_seconds, 4),
        "warnings": list(result.warnings),
        "safety_warnings": list(result.safety_warnings),
        "per_type_scores": dict(result.per_type_scores),
    }
    return json.dumps(data, indent=2)


def format_pr_comment(result: MigrationResult, source: str, target: str) -> str:
    """Format MigrationResult as a GitHub PR comment in markdown.

    Includes: header with source→target, recommendation with emoji,
    scores table, cost, safety warnings section, warnings section.
    """
    rec = result.recommendation
    emoji = _RECOMMENDATION_EMOJI.get(rec, "❓")
    rec_display = rec if rec is not None else "UNKNOWN"

    lines: list[str] = [
        "## 🔄 RosettaStone Migration Check",
        "",
        f"**Source:** `{source}` → **Target:** `{target}`",
        "",
        f"### Recommendation: {emoji} {rec_display}",
        "",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Confidence | {result.confidence_score:.4g} |",
        f"| Baseline | {result.baseline_score:.4g} |",
        f"| Improvement | {result.improvement:+.4g} |",
        "",
        f"**Cost:** ${result.cost_usd:.2f}",
    ]

    # Warnings section
    if result.warnings:
        lines.append("")
        lines.append("### ⚠️ Warnings")
        for w in result.warnings:
            lines.append(f"- {w}")

    # Safety warnings section
    if result.safety_warnings:
        lines.append("")
        lines.append("### 🔒 Safety")
        for w in result.safety_warnings:
            lines.append(f"- {w}")

    return "\n".join(lines)


def format_quality_diff(
    current: MigrationResult,
    baseline: MigrationResult | None = None,
) -> str:
    """Format quality comparison between current and baseline results.

    If baseline is None, just show current scores.
    If baseline provided, show delta (improvement/regression) per metric.
    """
    lines: list[str] = ["## Quality Report"]
    lines.append("")

    if baseline is None:
        # Show current scores only
        lines.append("### Current Scores")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        lines.append(f"| Confidence | {current.confidence_score:.4f} |")
        lines.append(f"| Baseline Score | {current.baseline_score:.4f} |")
        lines.append(f"| Improvement | {current.improvement:+.4f} |")

        if current.per_type_scores:
            lines.append("")
            lines.append("### Per-Type Breakdown")
            lines.append("")
            lines.append("| Type | Score |")
            lines.append("|------|-------|")
            for output_type, stats in current.per_type_scores.items():
                if isinstance(stats, dict):
                    score = stats.get("avg_score", stats.get("win_rate", "—"))
                else:
                    score = stats
                lines.append(f"| {output_type} | {score} |")
    else:
        # Compare current vs baseline
        conf_delta = current.confidence_score - baseline.confidence_score
        base_delta = current.baseline_score - baseline.baseline_score
        imp_delta = current.improvement - baseline.improvement

        lines.append("### Score Comparison (current vs baseline)")
        lines.append("")
        lines.append("| Metric | Current | Baseline | Delta |")
        lines.append("|--------|---------|----------|-------|")
        lines.append(
            f"| Confidence | {current.confidence_score:.4f} "
            f"| {baseline.confidence_score:.4f} "
            f"| {conf_delta:+.4f} |"
        )
        lines.append(
            f"| Baseline Score | {current.baseline_score:.4f} "
            f"| {baseline.baseline_score:.4f} "
            f"| {base_delta:+.4f} |"
        )
        lines.append(
            f"| Improvement | {current.improvement:+.4f} "
            f"| {baseline.improvement:+.4f} "
            f"| {imp_delta:+.4f} |"
        )

        if current.per_type_scores:
            lines.append("")
            lines.append("### Per-Type Breakdown")
            lines.append("")
            lines.append("| Type | Score |")
            lines.append("|------|-------|")
            for output_type, stats in current.per_type_scores.items():
                if isinstance(stats, dict):
                    score = stats.get("avg_score", stats.get("win_rate", "—"))
                else:
                    score = stats
                lines.append(f"| {output_type} | {score} |")

    return "\n".join(lines)
