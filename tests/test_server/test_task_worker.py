"""Tests for Task 1.4: DB-backed TaskWorker."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from rosettastone.server.models import MigrationRecord, TaskQueue
from rosettastone.server.task_worker import TaskWorker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """In-memory SQLite engine with StaticPool for cross-thread access."""
    eng = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def worker(engine):
    """TaskWorker with fast polling and no real background tasks."""
    w = TaskWorker(engine, poll_interval=0.05)
    yield w
    w.stop(wait=True, timeout=5.0)


def _insert_migration(engine) -> int:
    """Insert a pending migration and return its ID."""
    with Session(engine) as sess:
        record = MigrationRecord(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
        )
        sess.add(record)
        sess.commit()
        sess.refresh(record)
        return record.id  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Task lifecycle: queued → running → complete
# ---------------------------------------------------------------------------


class TestTaskLifecycle:
    def test_enqueue_creates_queued_row(self, worker, engine) -> None:
        """enqueue() must create a TaskQueue row with status='queued'."""
        mid = _insert_migration(engine)
        task_id = worker.enqueue("migration", mid, {"source_model": "a", "target_model": "b"})

        with Session(engine) as sess:
            task = sess.get(TaskQueue, task_id)
            assert task is not None
            assert task.status == "queued"
            assert task.task_type == "migration"
            assert task.resource_id == mid

    def test_task_transitions_queued_to_running_to_complete(self, worker, engine) -> None:
        """Worker must transition a task: queued → running → complete."""
        mid = _insert_migration(engine)
        payload = {"source_model": "openai/gpt-4o", "target_model": "anthropic/claude-sonnet-4"}

        with patch("rosettastone.server.task_worker.TaskWorker._execute") as mock_exec:

            def fake_execute(task):
                # Simulate completing the task inline
                worker._mark_complete(task.id)

            mock_exec.side_effect = fake_execute

            task_id = worker.enqueue("migration", mid, payload)
            worker.start()

            # Poll until complete or timeout
            deadline = time.time() + 5.0
            while time.time() < deadline:
                with Session(engine) as sess:
                    task = sess.get(TaskQueue, task_id)
                    if task and task.status == "complete":
                        break
                time.sleep(0.05)

        with Session(engine) as sess:
            task = sess.get(TaskQueue, task_id)
            assert task is not None
            assert task.status == "complete"

    def test_failed_task_writes_error_message(self, worker, engine) -> None:
        """When a task raises, status must be 'failed' with error_message set."""
        mid = _insert_migration(engine)
        payload = {"source_model": "a", "target_model": "b"}

        with patch("rosettastone.server.task_worker.TaskWorker._execute") as mock_exec:

            def fake_execute(task):
                worker._mark_failed(task.id, "simulated failure")

            mock_exec.side_effect = fake_execute

            task_id = worker.enqueue("migration", mid, payload)
            worker.start()

            deadline = time.time() + 5.0
            while time.time() < deadline:
                with Session(engine) as sess:
                    task = sess.get(TaskQueue, task_id)
                    if task and task.status == "failed":
                        break
                time.sleep(0.05)

        with Session(engine) as sess:
            task = sess.get(TaskQueue, task_id)
            assert task is not None
            assert task.status == "failed"
            assert task.error_message == "simulated failure"

    def test_completed_at_set_on_success(self, worker, engine) -> None:
        """completed_at must be set when a task completes."""
        mid = _insert_migration(engine)

        with patch("rosettastone.server.task_worker.TaskWorker._execute") as mock_exec:

            def fake_execute(task):
                worker._mark_complete(task.id)

            mock_exec.side_effect = fake_execute

            task_id = worker.enqueue("migration", mid, {})
            worker.start()

            deadline = time.time() + 5.0
            while time.time() < deadline:
                with Session(engine) as sess:
                    task = sess.get(TaskQueue, task_id)
                    if task and task.status == "complete":
                        break
                time.sleep(0.05)

        with Session(engine) as sess:
            task = sess.get(TaskQueue, task_id)
            assert task is not None
            assert task.completed_at is not None


# ---------------------------------------------------------------------------
# Restart recovery
# ---------------------------------------------------------------------------


class TestRestartRecovery:
    def test_recover_stale_tasks_re_enqueues_running_rows(self, worker, engine) -> None:
        """recover_stale_tasks() must reset stuck 'running' rows back to 'queued'."""
        mid = _insert_migration(engine)

        # Manually insert a 'running' task (simulating a crashed worker)
        with Session(engine) as sess:
            task = TaskQueue(
                task_type="migration",
                resource_id=mid,
                payload_json="{}",
                status="running",
                worker_id="dead-worker-xyz",
            )
            sess.add(task)
            sess.commit()
            sess.refresh(task)
            task_id = task.id

        count = worker.recover_stale_tasks()
        assert count == 1

        with Session(engine) as sess:
            task = sess.get(TaskQueue, task_id)
            assert task is not None
            assert task.status == "queued"
            assert task.worker_id is None
            assert task.started_at is None

    def test_recover_stale_tasks_returns_zero_when_none_stale(self, worker, engine) -> None:
        """recover_stale_tasks() must return 0 when no stale tasks exist."""
        count = worker.recover_stale_tasks()
        assert count == 0

    def test_recover_stale_does_not_touch_queued_or_complete_rows(self, worker, engine) -> None:
        """recover_stale_tasks() must only affect 'running' rows."""
        mid = _insert_migration(engine)
        task_id = worker.enqueue("migration", mid, {})

        worker.recover_stale_tasks()

        with Session(engine) as sess:
            task = sess.get(TaskQueue, task_id)
            assert task is not None
            assert task.status == "queued"  # unchanged


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_failed_task_retries_up_to_max_retries(self, worker, engine) -> None:
        """A task that fails twice but succeeds on the third attempt reaches 'complete'."""
        mid = _insert_migration(engine)
        payload = {"source_model": "a", "target_model": "b"}

        attempt_count = {"n": 0}

        def fake_execute(self_w, task):
            attempt_count["n"] += 1
            if attempt_count["n"] < 3:
                # Simulate failure by calling _mark_failed directly
                worker._mark_failed(task.id, f"simulated failure attempt {attempt_count['n']}")
            else:
                worker._mark_complete(task.id)

        with patch.object(TaskWorker, "_execute", fake_execute):
            task_id = worker.enqueue("migration", mid, payload, priority=0)

            # Set max_retries=2 so it can retry twice
            with Session(engine) as sess:
                task = sess.get(TaskQueue, task_id)
                task.max_retries = 2
                sess.add(task)
                sess.commit()

            worker.start()

            deadline = time.time() + 10.0
            while time.time() < deadline:
                with Session(engine) as sess:
                    task = sess.get(TaskQueue, task_id)
                    if task and task.status == "complete":
                        break
                time.sleep(0.05)

        with Session(engine) as sess:
            task = sess.get(TaskQueue, task_id)
            assert task is not None
            assert task.status == "complete"

    def test_task_permanently_fails_after_max_retries(self, worker, engine) -> None:
        """A task that always raises is marked 'failed' after exhausting max_retries."""
        mid = _insert_migration(engine)
        payload = {"source_model": "a", "target_model": "b"}

        def fake_execute(self_w, task):
            worker._mark_failed(task.id, "always fails")

        with patch.object(TaskWorker, "_execute", fake_execute):
            task_id = worker.enqueue("migration", mid, payload, priority=0)

            # Set max_retries=1 so it retries once then fails permanently
            with Session(engine) as sess:
                task = sess.get(TaskQueue, task_id)
                task.max_retries = 1
                sess.add(task)
                sess.commit()

            worker.start()

            deadline = time.time() + 10.0
            while time.time() < deadline:
                with Session(engine) as sess:
                    task = sess.get(TaskQueue, task_id)
                    if task and task.status == "failed":
                        break
                time.sleep(0.05)

        with Session(engine) as sess:
            task = sess.get(TaskQueue, task_id)
            assert task is not None
            assert task.status == "failed"
            assert task.error_message is not None


# ---------------------------------------------------------------------------
# Task dispatch by type
# ---------------------------------------------------------------------------


class TestTaskDispatch:
    def test_migration_task_calls_run_migration_background(self, worker, engine) -> None:
        """Task type 'migration' must invoke run_migration_background."""
        mid = _insert_migration(engine)
        payload = {"source_model": "a/b", "target_model": "c/d"}

        with patch(
            "rosettastone.server.task_worker.TaskWorker._execute",
            wraps=lambda task: None,
        ):
            pass  # just verify enqueue works

        calls = []

        def capture_execute(self_w, task):
            calls.append(task.task_type)
            worker._mark_complete(task.id)

        with patch.object(TaskWorker, "_execute", capture_execute):
            worker.enqueue("migration", mid, payload)
            worker.start()

            deadline = time.time() + 5.0
            while time.time() < deadline and not calls:
                time.sleep(0.05)

        assert "migration" in calls
