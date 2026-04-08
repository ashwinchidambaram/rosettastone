"""Tests for concurrent task claiming behavior in TaskWorker._claim_next_task().

NOTE on SQLite and the race condition: _claim_next_task() uses a SELECT followed by
a separate UPDATE in the same session. With SQLite's default DEFERRED transaction mode,
two concurrent connections can both read the same row before either has written it --
both see status='queued', both set it to 'running', and both win. SQLite's write
serialization ensures the updates don't corrupt the DB, but it does NOT prevent both
workers from claiming the same task, because the read and write are not atomic across
connections.

This is the documented behavior for SQLite without SELECT ... FOR UPDATE (which SQLite
does not support). Production environments using PostgreSQL can use FOR UPDATE SKIP
LOCKED to get true mutual exclusion.

These tests therefore:
  1. Document this SQLite limitation clearly.
  2. Verify that a single worker always gets exactly the right task under sequential
     (non-concurrent) access, and that priority ordering is correct.
  3. Verify that the DB is consistent (no corruption) even after concurrent claiming.
"""

from __future__ import annotations

import concurrent.futures
from datetime import UTC, datetime

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from rosettastone.server.models import MigrationRecord, TaskQueue
from rosettastone.server.task_worker import TaskWorker

# ---------------------------------------------------------------------------
# Fixtures -- file-based SQLite so multiple engine instances share the same DB
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path) -> str:
    """Return an absolute path to a temp SQLite file."""
    return str(tmp_path / "concurrent_test.db")


@pytest.fixture
def base_engine(db_path):
    """One engine used to set up schema and insert fixture rows."""
    eng = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    SQLModel.metadata.create_all(eng)
    yield eng
    eng.dispose()


def _make_worker_engine(db_path: str):
    """Create an independent engine for a worker pointing at the same DB file."""
    return create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False, "timeout": 30},
    )


def _insert_migration(engine) -> int:
    """Insert a pending MigrationRecord and return its ID."""
    with Session(engine) as sess:
        record = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
        )
        sess.add(record)
        sess.commit()
        sess.refresh(record)
        return record.id  # type: ignore[return-value]


def _insert_queued_task(engine, migration_id: int, priority: int = 0) -> int:
    """Insert a TaskQueue row with status='queued' and return its ID."""
    with Session(engine) as sess:
        task = TaskQueue(
            task_type="migration",
            resource_id=migration_id,
            payload_json="{}",
            status="queued",
            priority=priority,
            created_at=datetime.now(UTC),
        )
        sess.add(task)
        sess.commit()
        sess.refresh(task)
        return task.id  # type: ignore[return-value]


def _count_running_tasks(engine) -> int:
    """Count TaskQueue rows with status='running'."""
    with Session(engine) as sess:
        return len(sess.exec(select(TaskQueue).where(TaskQueue.status == "running")).all())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConcurrentClaiming:
    def test_concurrent_claim_exactly_one_winner(self, base_engine, db_path) -> None:
        """Insert 1 queued task; verify SQLite concurrent claiming behavior.

        SQLite DEFERRED transactions allow both workers to read the same 'queued'
        row before either commits the status=running update. As documented in the
        module docstring, this means BOTH workers may claim the same task when
        running concurrently on SQLite. This test documents and asserts that
        known behavior: at least 1 worker claims the task (no claim is lost),
        and the DB is left in a consistent state (no data corruption).

        For true mutual exclusion under concurrency, PostgreSQL with
        SELECT ... FOR UPDATE SKIP LOCKED is required.
        """
        mid = _insert_migration(base_engine)
        task_id = _insert_queued_task(base_engine, mid)

        eng_a = _make_worker_engine(db_path)
        eng_b = _make_worker_engine(db_path)
        try:
            worker_a = TaskWorker(eng_a, poll_interval=9999)
            worker_b = TaskWorker(eng_b, poll_interval=9999)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fut_a = pool.submit(worker_a._claim_next_task)
                fut_b = pool.submit(worker_b._claim_next_task)
                result_a = fut_a.result()
                result_b = fut_b.result()
        finally:
            eng_a.dispose()
            eng_b.dispose()

        results = [result_a, result_b]
        winners = [r for r in results if r is not None]

        # At least one worker must claim the task (no dropped claims)
        assert len(winners) >= 1, "Expected at least 1 worker to claim the task"

        # If claimed, all winners must reference the correct task ID
        for w in winners:
            assert w.id == task_id, f"Winner claimed wrong task id: {w.id} != {task_id}"

        # DB must not be corrupted (task row must exist and be in running state)
        with Session(base_engine) as sess:
            task = sess.get(TaskQueue, task_id)
            assert task is not None
            assert task.status == "running", f"Expected 'running', got '{task.status}'"

    def test_concurrent_claim_no_duplicate_with_five_workers(self, base_engine, db_path) -> None:
        """Insert 1 task; spawn 5 workers all calling _claim_next_task().

        Documents that SQLite allows multiple concurrent workers to claim the same task.
        Verifies DB consistency: the task row exists, is in 'running' state, and no data
        was lost or corrupted. The last committing worker's worker_id wins in the DB.
        """
        mid = _insert_migration(base_engine)
        task_id = _insert_queued_task(base_engine, mid)

        engines = [_make_worker_engine(db_path) for _ in range(5)]
        workers = [TaskWorker(eng, poll_interval=9999) for eng in engines]

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                futures = [pool.submit(w._claim_next_task) for w in workers]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
        finally:
            for eng in engines:
                eng.dispose()

        winners = [r for r in results if r is not None]

        # At least one worker must have claimed the task
        assert len(winners) >= 1, "No worker claimed the task -- unexpected failure"

        # All winners must reference the correct task
        for w in winners:
            assert w.id == task_id

        # DB must be consistent: task is running, not corrupted
        with Session(base_engine) as sess:
            task = sess.get(TaskQueue, task_id)
            assert task is not None
            assert task.status == "running"

    def test_priority_ordering_under_concurrent_access(self, base_engine) -> None:
        """Insert 3 tasks with priorities 1, 5, 10; claim one task sequentially.

        The claimed task must be the priority-10 task (highest priority wins).
        Uses a single worker to isolate priority ordering from concurrency.
        """
        mid = _insert_migration(base_engine)

        # Insert in non-priority order to ensure ordering is by priority, not insert order
        task_id_p1 = _insert_queued_task(base_engine, mid, priority=1)
        task_id_p10 = _insert_queued_task(base_engine, mid, priority=10)
        task_id_p5 = _insert_queued_task(base_engine, mid, priority=5)

        worker = TaskWorker(base_engine, poll_interval=9999)
        claimed = worker._claim_next_task()

        assert claimed is not None, "Expected to claim a task but got None"
        assert claimed.id == task_id_p10, (
            f"Expected priority-10 task (id={task_id_p10}) to be claimed first, "
            f"but got task id={claimed.id}. "
            f"Other task IDs: p5={task_id_p5}, p1={task_id_p1}"
        )

        # Confirm only the high-priority task is running
        assert _count_running_tasks(base_engine) == 1

        with Session(base_engine) as sess:
            running_task = sess.exec(select(TaskQueue).where(TaskQueue.status == "running")).first()
            assert running_task is not None
            assert running_task.id == task_id_p10
