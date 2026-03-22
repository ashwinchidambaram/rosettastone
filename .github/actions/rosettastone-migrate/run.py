#!/usr/bin/env python3
"""GitHub Action runner for RosettaStone migrations.

Reads configuration from environment variables, runs a migration,
writes results to GITHUB_STEP_SUMMARY, and exits 1 on NO_GO if configured.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    source_model = os.environ["RS_SOURCE_MODEL"]
    target_model = os.environ["RS_TARGET_MODEL"]
    data_path = os.environ["RS_DATA_PATH"]
    gepa_auto = os.environ.get("RS_GEPA_AUTO", "light")
    fail_on_nogo = os.environ.get("RS_FAIL_ON_NOGO", "true").lower() == "true"

    from rosettastone.config import MigrationConfig
    from rosettastone.core.migrator import MigrationBlockedError, Migrator

    config = MigrationConfig(
        source_model=source_model,
        target_model=target_model,
        data_path=Path(data_path),
        gepa_auto=gepa_auto,
    )

    try:
        result = Migrator(config).run()
    except MigrationBlockedError as exc:
        _write_summary(
            source_model, target_model,
            status="BLOCKED",
            recommendation="BLOCKED",
            confidence=None,
            baseline=None,
            improvement=None,
            cost=None,
            error=str(exc),
        )
        return 1
    except Exception as exc:
        _write_summary(
            source_model, target_model,
            status="ERROR",
            recommendation="ERROR",
            confidence=None,
            baseline=None,
            improvement=None,
            cost=None,
            error=str(exc),
        )
        return 1

    _write_summary(
        source_model, target_model,
        status="Complete",
        recommendation=result.recommendation or "Unknown",
        confidence=result.confidence_score,
        baseline=result.baseline_score,
        improvement=result.improvement,
        cost=result.cost_usd,
    )

    if fail_on_nogo and result.recommendation == "NO_GO":
        print("::error::Migration recommendation is NO_GO")
        return 1

    return 0


def _write_summary(
    source: str,
    target: str,
    *,
    status: str,
    recommendation: str,
    confidence: float | None,
    baseline: float | None,
    improvement: float | None,
    cost: float | None,
    error: str | None = None,
) -> None:
    """Write markdown summary to GITHUB_STEP_SUMMARY."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        # Not running in GitHub Actions — print to stdout
        print(
            f"Migration: {source} → {target} | "
            f"Status: {status} | Recommendation: {recommendation}"
        )
        return

    lines = [
        "## RosettaStone Migration Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Source Model | `{source}` |",
        f"| Target Model | `{target}` |",
        f"| Status | {status} |",
        f"| Recommendation | **{recommendation}** |",
        f"| Confidence | {f'{confidence:.0%}' if confidence is not None else 'N/A'} |",
        f"| Baseline | {f'{baseline:.0%}' if baseline is not None else 'N/A'} |",
        f"| Improvement | {f'+{improvement:.0%}' if improvement is not None else 'N/A'} |",
        f"| Cost | {f'${cost:.2f}' if cost is not None else 'N/A'} |",
    ]

    if error:
        lines.extend(["", f"> **Error:** {error}"])

    lines.append("")

    with open(summary_path, "a") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
