"""Shadow log evaluator — scores target vs source using CompositeEvaluator."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig


def score_shadow_logs(
    log_dir: Path,
    config: MigrationConfig,
) -> dict[str, Any]:
    """Score shadow deployment logs using CompositeEvaluator.

    Reads JSONL logs from log_dir, treats source_response as baseline and
    target_response as the optimized response, runs evaluation, and returns
    a summary dict with win_rate, total_pairs, and per-type breakdown.

    Args:
        log_dir: Directory containing shadow_*.jsonl log files.
        config: MigrationConfig with source/target model settings.

    Returns:
        Summary dict with keys: win_rate, total_pairs, wins,
        non_deterministic_count, cost_usd, per_type_scores, warnings.
    """
    from rosettastone.core.context import PipelineContext
    from rosettastone.core.types import PromptPair
    from rosettastone.evaluate.composite import CompositeEvaluator
    from rosettastone.shadow.log_format import read_log_entries

    entries = read_log_entries(log_dir)
    if not entries:
        return {
            "win_rate": 0.0,
            "total_pairs": 0,
            "wins": 0,
            "non_deterministic_count": 0,
            "cost_usd": 0.0,
            "per_type_scores": {},
            "warnings": ["No shadow log entries found in: " + str(log_dir)],
        }

    # Build PromptPairs: use source_response as the "gold" response,
    # target_response is what the new model said (passed via metadata for eval)
    pairs: list[PromptPair] = []
    for entry in entries:
        pairs.append(
            PromptPair(
                prompt=entry.prompt,
                response=entry.source_response,
                source_model=entry.source_model,
                metadata={
                    "shadow_request_id": entry.request_id,
                    "target_response": entry.target_response,
                },
            )
        )

    ctx = PipelineContext()
    evaluator = CompositeEvaluator(config, ctx=ctx)

    # Evaluate target responses against source responses as baseline
    # Use optimized_prompt="" to signal we're scoring the target model's actual responses
    results = evaluator.evaluate_multi_run(pairs, optimized_prompt="")

    wins = sum(1 for r in results if r.is_win)
    total = len(results)
    win_rate = wins / max(total, 1)

    # Per-type aggregation
    per_type: dict[str, dict[str, Any]] = {}
    for r in results:
        ot = r.details.get("output_type", "unknown")
        if ot not in per_type:
            per_type[ot] = {"wins": 0, "total": 0, "scores": []}
        per_type[ot]["total"] += 1
        per_type[ot]["scores"].append(r.composite_score)
        if r.is_win:
            per_type[ot]["wins"] += 1

    per_type_scores: dict[str, Any] = {}
    for ot, data in per_type.items():
        scores = data["scores"]
        per_type_scores[ot] = {
            "win_rate": data["wins"] / max(data["total"], 1),
            "sample_count": data["total"],
            "mean": sum(scores) / max(len(scores), 1),
        }

    non_det = sum(1 for r in results if r.is_non_deterministic)

    return {
        "win_rate": win_rate,
        "total_pairs": total,
        "wins": wins,
        "non_deterministic_count": non_det,
        "cost_usd": sum(ctx.costs.values()),
        "per_type_scores": per_type_scores,
        "warnings": ctx.warnings,
    }
