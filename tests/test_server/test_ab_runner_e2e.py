"""E2E tests for run_ab_test_background() — the completely untested background A/B execution path.

These tests call the real function with a real in-memory database.
No API endpoints, no TestClient — direct function invocation.
"""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from rosettastone.server.ab_runner import _conclude_test, run_ab_test_background
from rosettastone.server.models import (
    ABTest,
    ABTestResult,
    MigrationRecord,
    MigrationVersion,
    TestCaseRecord,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Fresh in-memory SQLite engine with all tables created."""
    eng = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def ab_test_data(engine):
    """Create a complete A/B test setup: migration + versions + test cases + AB test."""
    with Session(engine) as s:
        # Migration record
        mr = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            status="complete",
        )
        s.add(mr)
        s.flush()

        # Test cases (validation phase with scores)
        for i in range(10):
            tc = TestCaseRecord(
                migration_id=mr.id,
                phase="validation",
                output_type="json",
                composite_score=0.80 + i * 0.02,
                is_win=True,
                scores_json='{"bertscore": 0.9}',
                details_json="{}",
                response_length=100,
                new_response_length=95,
            )
            s.add(tc)

        # Migration versions -- snapshot_json is required (non-nullable)
        v1 = MigrationVersion(
            migration_id=mr.id,
            version_number=1,
            snapshot_json="{}",
            optimized_prompt="Version A prompt",
        )
        v2 = MigrationVersion(
            migration_id=mr.id,
            version_number=2,
            snapshot_json="{}",
            optimized_prompt="Version B prompt",
        )
        s.add(v1)
        s.add(v2)
        s.flush()

        # AB test linking the two versions
        ab = ABTest(
            migration_id=mr.id,
            version_a_id=v1.id,
            version_b_id=v2.id,
            name="test-ab",
            status="running",
            traffic_split=0.5,
        )
        s.add(ab)
        s.commit()
        s.refresh(mr)
        s.refresh(v1)
        s.refresh(v2)
        s.refresh(ab)

        return {
            "migration": mr,
            "ab_test": ab,
            "version_a": v1,
            "version_b": v2,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimulationModeCreatesResults:
    """run_ab_test_background creates one ABTestResult per TestCaseRecord."""

    def test_simulation_mode_creates_results(self, engine, ab_test_data):
        """Simulation mode creates one ABTestResult per validation TestCaseRecord."""
        ab_test_id = ab_test_data["ab_test"].id
        migration_id = ab_test_data["migration"].id

        run_ab_test_background(ab_test_id, simulation=True, engine=engine)

        with Session(engine) as s:
            results = list(
                s.exec(
                    select(ABTestResult).where(ABTestResult.ab_test_id == ab_test_id)
                ).all()
            )
            test_cases = list(
                s.exec(
                    select(TestCaseRecord)
                    .where(TestCaseRecord.migration_id == migration_id)
                    .where(TestCaseRecord.phase == "validation")
                ).all()
            )

        assert len(results) == len(test_cases), (
            f"Expected {len(test_cases)} results, got {len(results)}"
        )
        for r in results:
            assert r.assigned_version in ("a", "b"), (
                f"assigned_version must be 'a' or 'b', got {r.assigned_version!r}"
            )
            assert r.score_a is not None, "score_a must be populated in simulation mode"
            assert r.score_b is not None, "score_b must be populated in simulation mode"


class TestSimulationModeDeterministicAssignment:
    """MD5-based bucket assignment is deterministic across runs."""

    def test_simulation_mode_deterministic_assignment(self, engine, ab_test_data):
        """Same test cases always get assigned to the same version bucket."""
        ab_test_id = ab_test_data["ab_test"].id

        # First run
        run_ab_test_background(ab_test_id, simulation=True, engine=engine)

        with Session(engine) as s:
            first_run = {
                r.test_case_id: r.assigned_version
                for r in s.exec(
                    select(ABTestResult).where(ABTestResult.ab_test_id == ab_test_id)
                ).all()
            }
            # Delete results to allow a second run
            for row in s.exec(
                select(ABTestResult).where(ABTestResult.ab_test_id == ab_test_id)
            ).all():
                s.delete(row)
            # Reset AB test status so it can run again
            ab_test = s.get(ABTest, ab_test_id)
            ab_test.status = "running"
            ab_test.winner = None
            ab_test.end_time = None
            s.commit()

        # Second run
        run_ab_test_background(ab_test_id, simulation=True, engine=engine)

        with Session(engine) as s:
            second_run = {
                r.test_case_id: r.assigned_version
                for r in s.exec(
                    select(ABTestResult).where(ABTestResult.ab_test_id == ab_test_id)
                ).all()
            }

        assert first_run == second_run, (
            "Bucket assignments changed between runs -- MD5 hashing must be deterministic"
        )

    def test_assignment_matches_md5_formula(self, engine, ab_test_data):
        """Each result's assigned_version matches the expected MD5-based bucket."""
        ab_test_id = ab_test_data["ab_test"].id
        traffic_split = ab_test_data["ab_test"].traffic_split

        run_ab_test_background(ab_test_id, simulation=True, engine=engine)

        with Session(engine) as s:
            results = list(
                s.exec(
                    select(ABTestResult).where(ABTestResult.ab_test_id == ab_test_id)
                ).all()
            )

        for r in results:
            bucket = int(hashlib.md5(str(r.test_case_id).encode()).hexdigest(), 16) % 100
            expected = "a" if bucket < int(traffic_split * 100) else "b"
            assert r.assigned_version == expected, (
                f"test_case_id={r.test_case_id}: expected bucket '{expected}', "
                f"got '{r.assigned_version}'"
            )


class TestSimulationModeScoresMatchCached:
    """In simulation mode, score_a == score_b == TestCaseRecord.composite_score."""

    def test_simulation_mode_scores_match_cached(self, engine, ab_test_data):
        """Simulation mode reuses the cached composite_score for both score_a and score_b."""
        ab_test_id = ab_test_data["ab_test"].id
        migration_id = ab_test_data["migration"].id

        run_ab_test_background(ab_test_id, simulation=True, engine=engine)

        with Session(engine) as s:
            results = list(
                s.exec(
                    select(ABTestResult).where(ABTestResult.ab_test_id == ab_test_id)
                ).all()
            )
            # Build a lookup of test_case_id -> composite_score
            test_cases = {
                tc.id: tc.composite_score
                for tc in s.exec(
                    select(TestCaseRecord)
                    .where(TestCaseRecord.migration_id == migration_id)
                    .where(TestCaseRecord.phase == "validation")
                ).all()
            }

        for r in results:
            cached_score = test_cases.get(r.test_case_id)
            assert cached_score is not None, f"No test case found for id={r.test_case_id}"
            assert r.score_a == pytest.approx(cached_score), (
                f"score_a {r.score_a} != cached composite_score {cached_score}"
            )
            assert r.score_b == pytest.approx(cached_score), (
                f"score_b {r.score_b} != cached composite_score {cached_score}"
            )
            assert r.winner == "tie", (
                f"When score_a == score_b the winner should be 'tie', got {r.winner!r}"
            )


class TestConcludeDeterminesWinner:
    """_conclude_test updates status, sets end_time, and picks a winner."""

    def test_conclude_determines_winner(self, engine, ab_test_data):
        """After simulation, run_ab_test_background marks the AB test as concluded."""
        ab_test_id = ab_test_data["ab_test"].id

        run_ab_test_background(ab_test_id, simulation=True, engine=engine)

        with Session(engine) as s:
            ab_test = s.get(ABTest, ab_test_id)
            assert ab_test.status == "concluded", (
                f"Expected status='concluded', got {ab_test.status!r}"
            )
            assert ab_test.winner in ("a", "b", "inconclusive"), (
                f"winner must be 'a', 'b', or 'inconclusive', got {ab_test.winner!r}"
            )
            assert ab_test.end_time is not None, "end_time must be set after conclude"

    def test_conclude_test_direct_call(self, engine, ab_test_data):
        """_conclude_test can be called directly and produces a valid conclusion."""
        ab_test_id = ab_test_data["ab_test"].id

        # Seed a few results manually so _conclude_test has something to process
        with Session(engine) as s:
            for version in ("a", "b", "a"):
                row = ABTestResult(
                    ab_test_id=ab_test_id,
                    assigned_version=version,
                    score_a=0.8,
                    score_b=0.7,
                    winner="tie",
                )
                s.add(row)
            s.commit()

        _conclude_test(ab_test_id, engine)

        with Session(engine) as s:
            ab_test = s.get(ABTest, ab_test_id)
            assert ab_test.status == "concluded"
            assert ab_test.winner in ("a", "b", "inconclusive")
            assert ab_test.end_time is not None


class TestSimulationWithEmptyTestCases:
    """Simulation with no validation test cases completes cleanly."""

    def test_simulation_with_empty_test_cases(self, engine):
        """run_ab_test_background with no test cases produces 0 results and does not raise."""
        with Session(engine) as s:
            mr = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                status="complete",
            )
            s.add(mr)
            s.flush()

            v1 = MigrationVersion(
                migration_id=mr.id, version_number=1, snapshot_json="{}"
            )
            v2 = MigrationVersion(
                migration_id=mr.id, version_number=2, snapshot_json="{}"
            )
            s.add(v1)
            s.add(v2)
            s.flush()

            ab = ABTest(
                migration_id=mr.id,
                version_a_id=v1.id,
                version_b_id=v2.id,
                name="empty-test",
                status="running",
                traffic_split=0.5,
            )
            s.add(ab)
            s.commit()
            s.refresh(ab)
            ab_test_id = ab.id

        # Should not raise even though there are no test cases
        run_ab_test_background(ab_test_id, simulation=True, engine=engine)

        with Session(engine) as s:
            result_count = len(
                list(
                    s.exec(
                        select(ABTestResult).where(ABTestResult.ab_test_id == ab_test_id)
                    ).all()
                )
            )

        assert result_count == 0, f"Expected 0 results for empty test cases, got {result_count}"


class TestSimulationBatchCommits:
    """Simulation with > BATCH_SIZE (50) test cases commits across multiple batches."""

    def test_simulation_batches_commits(self, engine):
        """60 test cases (>BATCH_SIZE=50) are all committed across two batches."""
        num_cases = 60

        with Session(engine) as s:
            mr = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                status="complete",
            )
            s.add(mr)
            s.flush()

            for i in range(num_cases):
                tc = TestCaseRecord(
                    migration_id=mr.id,
                    phase="validation",
                    output_type="text",
                    composite_score=0.5 + (i % 10) * 0.01,
                    is_win=(i % 2 == 0),
                    scores_json="{}",
                    details_json="{}",
                    response_length=50,
                    new_response_length=48,
                )
                s.add(tc)

            v1 = MigrationVersion(
                migration_id=mr.id, version_number=1, snapshot_json="{}"
            )
            v2 = MigrationVersion(
                migration_id=mr.id, version_number=2, snapshot_json="{}"
            )
            s.add(v1)
            s.add(v2)
            s.flush()

            ab = ABTest(
                migration_id=mr.id,
                version_a_id=v1.id,
                version_b_id=v2.id,
                name="batch-test",
                status="running",
                traffic_split=0.5,
            )
            s.add(ab)
            s.commit()
            s.refresh(ab)
            ab_test_id = ab.id

        run_ab_test_background(ab_test_id, simulation=True, engine=engine)

        with Session(engine) as s:
            results = list(
                s.exec(
                    select(ABTestResult).where(ABTestResult.ab_test_id == ab_test_id)
                ).all()
            )

        assert len(results) == num_cases, (
            f"Expected all {num_cases} results committed across batches, got {len(results)}"
        )


class TestFailedAbTestMarkedInconclusive:
    """When version IDs are invalid, run_ab_test_background marks winner='inconclusive'."""

    def test_failed_ab_test_marked_inconclusive(self, engine):
        """AB test with non-existent version IDs is marked inconclusive via _mark_failed."""
        with Session(engine) as s:
            mr = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                status="complete",
            )
            s.add(mr)
            s.flush()

            # Create real versions so the migration exists, but AB test points at
            # non-existent version IDs (9999, 9998) to trigger _mark_failed.
            v1 = MigrationVersion(
                migration_id=mr.id, version_number=1, snapshot_json="{}"
            )
            s.add(v1)
            s.flush()

            ab = ABTest(
                migration_id=mr.id,
                version_a_id=9999,  # non-existent
                version_b_id=9998,  # non-existent
                name="bad-versions-test",
                status="running",
                traffic_split=0.5,
            )
            s.add(ab)
            s.commit()
            s.refresh(ab)
            ab_test_id = ab.id

        # Must not raise -- errors are caught and _mark_failed is called
        run_ab_test_background(ab_test_id, simulation=True, engine=engine)

        with Session(engine) as s:
            ab_test = s.get(ABTest, ab_test_id)

        assert ab_test.winner == "inconclusive", (
            f"Expected winner='inconclusive' from _mark_failed, got {ab_test.winner!r}"
        )
        assert ab_test.status == "concluded", (
            f"Expected status='concluded' from _mark_failed, got {ab_test.status!r}"
        )
        assert ab_test.end_time is not None, "end_time must be set even on failure path"

    def test_nonexistent_ab_test_id_is_no_op(self, engine):
        """Calling run_ab_test_background with a non-existent id returns without error."""
        # Should log an error but not raise
        run_ab_test_background(99999, simulation=True, engine=engine)
        # If we reach here without exception, the test passes
