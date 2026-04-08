"""DB-backed task queue worker for RosettaStone background jobs.

Replaces the single-threaded ThreadPoolExecutor with a durable, restartable
polling worker. Tasks are persisted in the TaskQueue table so they survive
server restarts.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)


class TaskWorker:
    """Single-threaded background worker that polls the TaskQueue table.

    Design notes:
    - One polling thread, one in-flight task at a time (preserves DSPy thread-safety).
    - Atomic claim: UPDATE status queued to running before executing.
    - Restart recovery: stale running rows are re-queued on startup.
    """

    def __init__(self, engine: Any, poll_interval: float = 1.0) -> None:
        self._engine = engine
        self._poll_interval = poll_interval
        self._worker_id = f"worker-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="task-worker")

    def start(self) -> None:
        """Start the background polling thread."""
        logger.info("TaskWorker %s starting", self._worker_id)
        self._thread.start()

    def stop(self, wait: bool = True, timeout: float = 30.0) -> None:
        """Signal the worker to stop and optionally wait for it to finish."""
        self._stop_event.set()
        if wait and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def enqueue(
        self,
        task_type: str,
        resource_id: int,
        payload: dict[str, Any],
        priority: int = 0,
        correlation_id: str | None = None,
    ) -> int:
        """Insert a TaskQueue row with status='queued'. Returns the task ID."""
        from sqlmodel import Session

        from rosettastone.server.models import TaskQueue

        with Session(self._engine) as sess:
            task = TaskQueue(
                task_type=task_type,
                resource_id=resource_id,
                payload_json=json.dumps(payload),
                status="queued",
                priority=priority,
                correlation_id=correlation_id,
            )
            sess.add(task)
            sess.commit()
            sess.refresh(task)
            task_id = task.id
        logger.debug("Enqueued %s task id=%s resource_id=%s", task_type, task_id, resource_id)
        return task_id  # type: ignore[return-value]

    def recover_stale_tasks(self) -> int:
        """Re-enqueue any TaskQueue rows stuck in running status (from a crashed worker).

        Returns the number of tasks re-enqueued.
        """
        from sqlmodel import Session, select

        from rosettastone.server.models import TaskQueue

        with Session(self._engine) as sess:
            stale = list(sess.exec(select(TaskQueue).where(TaskQueue.status == "running")).all())
            for task in stale:
                task.status = "queued"
                task.worker_id = None
                task.started_at = None
                sess.add(task)
            if stale:
                sess.commit()
                logger.warning(
                    "TaskWorker: re-enqueued %d stale running task(s) from previous run",
                    len(stale),
                )
        return len(stale)

    # ------------------------------------------------------------------
    # Internal polling loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main polling loop — runs in the background thread."""
        while not self._stop_event.is_set():
            task = self._claim_next_task()
            if task is None:
                self._stop_event.wait(self._poll_interval)
                continue
            self._execute(task)

    def _claim_next_task(self) -> Any:
        """Claim the next queued task (highest priority, oldest first).

        Returns a SimpleNamespace with task fields, or None if queue is empty.
        The row is updated to status='running' inside this method.
        """
        from sqlmodel import Session, select

        from rosettastone.server.models import TaskQueue

        with Session(self._engine) as sess:
            task = sess.exec(
                select(TaskQueue)
                .where(TaskQueue.status == "queued")
                .order_by(TaskQueue.priority.desc(), TaskQueue.created_at.asc())  # type: ignore[attr-defined]
                .limit(1)
            ).first()
            if task is None:
                return None
            task.status = "running"
            task.started_at = datetime.now(UTC)
            task.worker_id = self._worker_id
            sess.add(task)
            sess.commit()
            # Record queue wait time metric
            try:
                from datetime import UTC as _UTC
                from datetime import datetime as _dt

                _wait = (_dt.now(_UTC) - task.created_at).total_seconds()
                from rosettastone.server.metrics import record_task_queue_wait

                record_task_queue_wait(max(0.0, _wait))
            except Exception:
                pass
            # Capture scalar values before session closes
            return SimpleNamespace(
                id=task.id,
                task_type=task.task_type,
                resource_id=task.resource_id,
                payload_json=task.payload_json,
            )

    def _execute(self, task: Any) -> None:
        """Execute a claimed task and update its final status."""
        logger.info(
            "TaskWorker executing task id=%s type=%s resource_id=%s",
            task.id,
            task.task_type,
            task.resource_id,
        )
        try:
            payload = json.loads(task.payload_json or "{}")
            if task.task_type == "migration":
                from rosettastone.server.api.tasks import run_migration_background

                run_migration_background(task.resource_id, payload, self._engine)
            elif task.task_type == "pipeline":
                from rosettastone.server.pipeline_runner import run_pipeline_background

                run_pipeline_background(task.resource_id, self._engine)
            else:
                raise ValueError(f"Unknown task_type: {task.task_type!r}")

            self._mark_complete(task.id)
        except Exception as exc:
            logger.error(
                "Task id=%s type=%s failed: %s",
                task.id,
                task.task_type,
                exc,
                exc_info=True,
            )
            self._mark_failed(task.id, str(exc))

    def _mark_complete(self, task_id: int) -> None:
        from sqlmodel import Session

        from rosettastone.server.models import TaskQueue

        with Session(self._engine) as sess:
            task = sess.get(TaskQueue, task_id)
            if task:
                task.status = "complete"
                task.completed_at = datetime.now(UTC)
                sess.add(task)
                sess.commit()

    def _mark_failed(self, task_id: int, error_message: str) -> None:
        from sqlmodel import Session

        from rosettastone.server.models import TaskQueue

        with Session(self._engine) as sess:
            task = sess.get(TaskQueue, task_id)
            if task:
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = "queued"
                    task.started_at = None
                    task.worker_id = None
                    task.error_message = (
                        f"Retry {task.retry_count}/{task.max_retries}: {error_message[:200]}"
                    )
                else:
                    task.status = "failed"
                    task.completed_at = datetime.now(UTC)
                    task.error_message = error_message[:500]
                sess.add(task)
                sess.commit()
