"""Executive narrative prompt template and formatting utilities."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are a senior technical writer at a technology company. \
You write clear, honest executive summaries of LLM model migration evaluations.

Your audience is engineering leadership — they understand metrics but want concise, \
actionable summaries rather than raw data tables.

## Rules
1. NEVER include raw prompt text, response text, or any production data content
2. Only reference scores, metrics, sample sizes, and structural metadata
3. Be honest — don't cherry-pick positive results or hide regressions
4. If the recommendation is NO_GO or CONDITIONAL, lead with the concerns
5. Always mention confidence intervals and sample sizes to contextualize results
6. Use specific numbers — don't say "high confidence", say "92.3% confidence"
7. Keep it to 3-5 paragraphs
8. End with clear, actionable next steps"""


USER_PROMPT_TEMPLATE = """Generate an executive summary for this LLM model migration evaluation.

## Migration Overview
- **Source Model:** {source_model}
- **Target Model:** {target_model}
- **Recommendation:** {recommendation}
- **Confidence Score:** {confidence_pct}
- **Baseline Score:** {baseline_pct}
- **Improvement:** {improvement_pct}
- **Total Cost:** ${cost_usd:.4f}
- **Duration:** {duration_seconds:.1f}s

## Test Results
- **Total Test Cases:** {total_test_cases}
- **Wins:** {wins} ({win_rate_pct})

## Per-Output-Type Performance
{per_type_block}

## Safety Findings
{safety_block}

## Pipeline Warnings
{warnings_block}

---

Write the executive summary now. Remember: no raw content, only metrics and structural findings."""


# Few-shot examples for consistent output quality
FEW_SHOT_EXAMPLES = [
    {
        "scenario": "GO recommendation with strong results",
        "input_summary": (
            "GPT-4o to Claude Sonnet, GO, 94.2% confidence, "
            "json win_rate=97%, classification win_rate=93%"
        ),
        "output": (
            "The migration evaluation from GPT-4o to Claude Sonnet yields a **GO** "
            "recommendation with 94.2% overall confidence. Across 156 test cases, the "
            "target model matched or exceeded baseline performance in 91% of cases, with "
            "a net improvement of +7.3%.\n\n"
            "JSON-structured outputs performed exceptionally well at 97% win rate (n=82, "
            "95% CI [0.91, 0.99]), comfortably exceeding the 95% threshold. Classification "
            "tasks achieved 93% win rate (n=48, 95% CI [0.82, 0.98]), clearing the 90% bar. "
            "Median scores were 0.96 and 0.91 respectively, indicating consistently strong "
            "performance rather than high variance.\n\n"
            "No safety findings were flagged during evaluation. The migration cost $2.34 and "
            "completed in 127 seconds.\n\n"
            "**Next steps:** Proceed with staged rollout. Monitor the classification category "
            "in production as its confidence interval lower bound (0.82) is closer to threshold "
            "than JSON outputs. Consider expanding the test set for long_text outputs if those "
            "become relevant to the workload."
        ),
    },
    {
        "scenario": "CONDITIONAL recommendation with mixed results",
        "input_summary": (
            "GPT-4o to Claude Haiku, CONDITIONAL, 78% confidence, "
            "json win_rate=96%, long_text win_rate=62%"
        ),
        "output": (
            "The migration evaluation from GPT-4o to Claude Haiku returns a **CONDITIONAL** "
            "recommendation at 78.1% overall confidence. While structured outputs perform "
            "well, long-form text generation shows significant regression that requires "
            "attention before proceeding.\n\n"
            "JSON outputs achieved a strong 96% win rate (n=64, 95% CI [0.88, 0.99]), well "
            "above the 95% threshold. However, long_text outputs fell to 62% win rate (n=31, "
            "95% CI [0.44, 0.78]) — substantially below the 75% threshold. The p10 score of "
            "0.41 for long_text indicates a tail of poor-quality responses that could impact "
            "user experience.\n\n"
            "No HIGH-severity safety issues were found, though one MEDIUM warning noted "
            "inconsistent formatting in 8% of responses. The evaluation cost $1.87 over "
            "94 seconds.\n\n"
            "**Next steps:** Do not proceed with full migration. Consider (1) using Claude "
            "Haiku only for JSON/structured workloads where it excels, (2) investigating the "
            "long_text regressions to determine if prompt optimization can close the gap, or "
            "(3) evaluating a mid-tier model for the long_text workload."
        ),
    },
    {
        "scenario": "NO_GO recommendation due to safety findings",
        "input_summary": ("GPT-4o to local model, NO_GO, HIGH severity safety warning"),
        "output": (
            "The migration evaluation from GPT-4o to llama-3-70b is **blocked (NO_GO)** due "
            "to a HIGH-severity safety finding. This recommendation stands regardless of "
            "performance metrics until the safety concern is resolved.\n\n"
            "The safety evaluation flagged that the target model produces outputs containing "
            "instruction-following artifacts (system prompt leakage) in 3.2% of test cases. "
            "This represents an unacceptable risk for production deployment where prompts may "
            "contain proprietary business logic.\n\n"
            "Setting aside the safety blocker, performance metrics were mixed: JSON outputs "
            "achieved 89% win rate (n=45) — below the 95% threshold — while classification "
            "reached 91% (n=38). Overall confidence was 71.4% with a total evaluation cost of "
            "$0.12 over 203 seconds.\n\n"
            "**Next steps:** (1) Address the system prompt leakage issue — this may require "
            "model configuration changes or guardrail implementation. (2) Once resolved, re-run "
            "the evaluation with the safety fix in place. (3) Consider whether the JSON output "
            "regression is acceptable or if further prompt optimization is needed."
        ),
    },
]


def format_executive_prompt(
    *,
    source_model: str,
    target_model: str,
    recommendation: str,
    confidence_score: float,
    baseline_score: float,
    improvement: float,
    cost_usd: float,
    duration_seconds: float,
    total_test_cases: int,
    wins: int,
    per_type_scores: dict[str, Any],
    safety_warnings: list[Any],
    warnings: list[str],
) -> list[dict[str, str]]:
    """Format the executive prompt into a messages list for LiteLLM.

    Returns a list of message dicts (system + few-shot examples + user prompt)
    suitable for passing to litellm.completion().
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add few-shot examples
    for example in FEW_SHOT_EXAMPLES:
        messages.append(
            {
                "role": "user",
                "content": f"Summarize this migration: {example['input_summary']}",
            }
        )
        messages.append({"role": "assistant", "content": example["output"]})

    # Format the actual prompt
    per_type_block = _format_per_type_block(per_type_scores)
    safety_block = _format_safety_block(safety_warnings)
    warnings_block = _format_warnings_block(warnings)

    win_rate = wins / max(total_test_cases, 1)

    user_content = USER_PROMPT_TEMPLATE.format(
        source_model=source_model,
        target_model=target_model,
        recommendation=recommendation,
        confidence_pct=f"{confidence_score:.1%}",
        baseline_pct=f"{baseline_score:.1%}",
        improvement_pct=f"+{improvement:.1%}",
        cost_usd=cost_usd,
        duration_seconds=duration_seconds,
        total_test_cases=total_test_cases,
        wins=wins,
        win_rate_pct=f"{win_rate:.0%}",
        per_type_block=per_type_block,
        safety_block=safety_block,
        warnings_block=warnings_block,
    )

    messages.append({"role": "user", "content": user_content})
    return messages


def _format_per_type_block(per_type_scores: dict[str, Any]) -> str:
    """Format per-type scores into readable block for prompt."""
    if not per_type_scores:
        return "No per-type breakdown available."

    lines = []
    for type_name, stats in sorted(per_type_scores.items()):
        if isinstance(stats, dict):
            s = stats
        elif hasattr(stats, "__dataclass_fields__"):
            import dataclasses

            s = dataclasses.asdict(stats)
        else:
            continue

        ci = s.get("confidence_interval", (0, 0))
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        else:
            ci_str = "N/A"

        lines.append(
            f"- **{type_name}**: win_rate={s.get('win_rate', 0):.1%}, "
            f"samples={s.get('sample_count', 0)}, "
            f"mean={s.get('mean', 0):.3f}, median={s.get('median', 0):.3f}, "
            f"p10={s.get('p10', 0):.3f}, p50={s.get('p50', 0):.3f}, "
            f"p90={s.get('p90', 0):.3f}, "
            f"CI(95%)={ci_str}"
        )
    return "\n".join(lines)


def _format_safety_block(safety_warnings: list[Any]) -> str:
    """Format safety warnings into readable block."""
    if not safety_warnings:
        return "No safety issues found."

    lines = []
    for w in safety_warnings:
        if isinstance(w, dict):
            severity = w.get("severity", "INFO")
            message = w.get("message", str(w))
        elif hasattr(w, "severity"):
            severity = str(w.severity)
            message = str(w.message)
        else:
            severity = "INFO"
            message = str(w)
        lines.append(f"- [{severity}] {message}")
    return "\n".join(lines)


def _format_warnings_block(warnings: list[str]) -> str:
    """Format pipeline warnings into readable block."""
    if not warnings:
        return "No pipeline warnings."
    return "\n".join(f"- {w}" for w in warnings)
