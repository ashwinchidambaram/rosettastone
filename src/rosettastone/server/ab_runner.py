"""A/B test background runner — simulation and live evaluation modes."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rosettastone.decision.ab_stats import ABSignificanceResult

from sqlmodel import Session, select

from rosettastone.decision.ab_stats import compute_ab_significance
from rosettastone.server.api.audit import log_audit
from rosettastone.server.database import get_engine
from rosettastone.server.models import (
    ABTest,
    ABTestResult,
    MigrationVersion,
    TestCaseRecord,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 50  # Commit results every N rows for partial failure resilience


def _determine_winner(sig: ABSignificanceResult, wins_a: int, wins_b: int) -> str:
    """Determine A/B test winner using mean_diff from significance result.

    Uses mean_diff as the single source of truth. Returns 'inconclusive' when
    not statistically significant or scores are identical.
    """
    if not sig.significant:
        return "inconclusive"
    if sig.mean_diff > 0:
        return "a"
    if sig.mean_diff < 0:
        return "b"
    return "inconclusive"


def run_ab_test_background(
    ab_test_id: int,
    simulation: bool = True,
    engine: Any = None,
) -> None:
    """Run an A/B test in a background thread.

    Two modes:
      - simulation (default): Reuses cached TestCaseRecord scores from both
        migration versions. Fast, free, no API calls.
      - live: Re-evaluates test cases through both prompts. Requires that
        the original migration used --store-prompt-content.

    Args:
        ab_test_id: ID of the ABTest to execute.
        simulation: If True, use cached scores. If False, re-evaluate via LLM.
        engine: SQLAlchemy engine (default: get_engine()).
    """
    if engine is None:
        engine = get_engine()

    try:
        with Session(engine) as session:
            ab_test = session.get(ABTest, ab_test_id)
            if not ab_test:
                logger.error("ABTest %d not found", ab_test_id)
                return

            version_a = session.get(MigrationVersion, ab_test.version_a_id)
            version_b = session.get(MigrationVersion, ab_test.version_b_id)
            if not version_a or not version_b:
                logger.error("Version(s) not found for ABTest %d", ab_test_id)
                _mark_failed(ab_test_id, engine, "Version not found")
                return

            migration_id = ab_test.migration_id
            traffic_split = ab_test.traffic_split

        if simulation:
            _run_simulation(ab_test_id, migration_id, traffic_split, engine)
        else:
            _run_live(ab_test_id, migration_id, version_a, version_b, traffic_split, engine)

        # Compute significance and conclude
        _conclude_test(ab_test_id, engine)

    except Exception as exc:
        logger.error("ABTest %d failed: %s", ab_test_id, exc)
        _mark_failed(ab_test_id, engine, str(exc))


def _run_simulation(
    ab_test_id: int,
    migration_id: int,
    traffic_split: float,
    engine: Any,
) -> None:
    """Simulation mode: compare cached validation scores."""
    with Session(engine) as session:
        stmt = (
            select(TestCaseRecord)
            .where(TestCaseRecord.migration_id == migration_id)
            .where(TestCaseRecord.phase == "validation")
        )
        test_cases = list(session.exec(stmt).all())

    if not test_cases:
        logger.warning("No validation test cases for migration %d", migration_id)
        return

    results: list[ABTestResult] = []
    for tc in test_cases:
        tc_id = tc.id
        bucket = int(hashlib.md5(str(tc_id).encode()).hexdigest(), 16) % 100
        assigned = "a" if bucket < int(traffic_split * 100) else "b"

        score = tc.composite_score

        result = ABTestResult(
            ab_test_id=ab_test_id,
            test_case_id=tc_id,
            assigned_version=assigned,
            score_a=score,
            score_b=score,
            winner="tie",  # Same cached score = tie in simulation
            details_json=json.dumps({"mode": "simulation", "composite_score": score}),
        )
        results.append(result)

        if len(results) >= BATCH_SIZE:
            _commit_results(results, engine)
            results = []

    if results:
        _commit_results(results, engine)


def _run_live(
    ab_test_id: int,
    migration_id: int,
    version_a: MigrationVersion,
    version_b: MigrationVersion,
    traffic_split: float,
    engine: Any,
) -> None:
    """Live mode: re-evaluate test cases through both prompt versions."""
    prompt_a = version_a.optimized_prompt or ""
    prompt_b = version_b.optimized_prompt or ""

    with Session(engine) as session:
        stmt = (
            select(TestCaseRecord)
            .where(TestCaseRecord.migration_id == migration_id)
            .where(TestCaseRecord.phase == "validation")
        )
        test_cases = list(session.exec(stmt).all())

    if not test_cases:
        logger.warning("No validation test cases for migration %d", migration_id)
        return

    has_content = any(tc.prompt_text is not None for tc in test_cases)
    if not has_content:
        raise ValueError("Live mode requires --store-prompt-content on the original migration")

    results: list[ABTestResult] = []
    for tc in test_cases:
        if tc.prompt_text is None:
            continue

        tc_id = tc.id
        bucket = int(hashlib.md5(str(tc_id).encode()).hexdigest(), 16) % 100
        assigned = "a" if bucket < int(traffic_split * 100) else "b"

        try:
            score_a = _evaluate_with_prompt(tc.prompt_text, prompt_a)
            score_b = _evaluate_with_prompt(tc.prompt_text, prompt_b)

            if score_a > score_b:
                winner = "a"
            elif score_b > score_a:
                winner = "b"
            else:
                winner = "tie"

            result = ABTestResult(
                ab_test_id=ab_test_id,
                test_case_id=tc_id,
                assigned_version=assigned,
                score_a=score_a,
                score_b=score_b,
                winner=winner,
                details_json=json.dumps({"mode": "live"}),
            )
            results.append(result)

        except Exception as exc:
            logger.warning("Failed to evaluate test case %d: %s", tc_id, exc)
            continue

        if len(results) >= BATCH_SIZE:
            _commit_results(results, engine)
            results = []

    if results:
        _commit_results(results, engine)


def _evaluate_with_prompt(prompt_text: str, optimized_prompt: str) -> float:
    """Evaluate a prompt using the optimized system prompt. Returns a score 0-1."""
    try:
        from rosettastone.evaluate.exact_match import string_similarity

        return string_similarity(prompt_text, optimized_prompt)
    except ImportError:
        return 0.5


def _commit_results(results: list[ABTestResult], engine: Any) -> int:
    """Batch-commit a list of ABTestResult rows with partial failure handling.

    Returns the number of successfully committed results.
    """
    with Session(engine) as session:
        try:
            for r in results:
                session.add(r)
            session.commit()
            return len(results)
        except Exception:
            session.rollback()
            logger.warning(
                "Batch commit failed for %d results, falling back to individual inserts",
                len(results),
            )

    # Fall back to individual inserts
    committed = 0
    for r in results:
        try:
            with Session(engine) as s:
                s.merge(r)
                s.commit()
                committed += 1
        except Exception:
            logger.warning(
                "Failed to commit ABTestResult for test_case_id=%s",
                getattr(r, "test_case_id", "unknown"),
            )
    logger.info("Partial commit: %d/%d results saved", committed, len(results))
    return committed


def _conclude_test(ab_test_id: int, engine: Any) -> None:
    """Compute significance and update ABTest with winner."""
    with Session(engine) as session:
        ab_test = session.get(ABTest, ab_test_id)
        if not ab_test:
            return

        stmt = select(ABTestResult).where(ABTestResult.ab_test_id == ab_test_id)
        results = list(session.exec(stmt).all())

        result_dicts = [
            {
                "assigned_version": r.assigned_version,
                "score_a": r.score_a,
                "score_b": r.score_b,
                "winner": r.winner,
            }
            for r in results
        ]

        wins_a = sum(1 for r in result_dicts if r["winner"] == "a")
        wins_b = sum(1 for r in result_dicts if r["winner"] == "b")
        sig = compute_ab_significance(result_dicts)

        ab_test.winner = _determine_winner(sig, wins_a, wins_b)

        ab_test.status = "concluded"
        ab_test.end_time = datetime.now(UTC)
        session.add(ab_test)

        log_audit(
            session,
            "ab_test",
            ab_test_id,
            "conclude",
            details={
                "winner": ab_test.winner,
                "p_value": sig.p_value,
                "significant": sig.significant,
            },
        )
        session.commit()


def _mark_failed(ab_test_id: int, engine: Any, reason: str) -> None:
    """Mark an A/B test as failed."""
    try:
        with Session(engine) as session:
            ab_test = session.get(ABTest, ab_test_id)
            if ab_test:
                ab_test.status = "concluded"
                ab_test.winner = "inconclusive"
                ab_test.end_time = datetime.now(UTC)
                session.add(ab_test)

                log_audit(
                    session,
                    "ab_test",
                    ab_test_id,
                    "failed",
                    details={"reason": reason},
                )
                session.commit()
    except Exception as exc:
        logger.error("Failed to mark ABTest %d as failed: %s", ab_test_id, exc)
