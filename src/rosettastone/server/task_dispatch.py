"""Task dispatch abstraction — uses RQ when Redis is available, DB queue otherwise."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _get_redis_url() -> str | None:
    return os.environ.get("REDIS_URL")


class TaskDispatcher:
    """Dispatches background tasks to RQ (if Redis available) or DB queue."""

    def __init__(self) -> None:
        self._rq_available = False
        self._redis_conn: Any = None
        self._db_worker: Any = None
        self._queue: Any = None

    def setup(self, db_worker: Any) -> None:
        """Initialize dispatcher. Tries RQ first, falls back to db_worker."""
        self._db_worker = db_worker
        redis_url = _get_redis_url()
        if redis_url:
            try:
                import redis
                import rq

                self._redis_conn = redis.from_url(redis_url)
                self._redis_conn.ping()
                self._queue = rq.Queue("rosettastone", connection=self._redis_conn)
                self._rq_available = True
                logger.info("Task dispatch: using RQ (Redis at %s)", redis_url)
            except Exception as exc:
                logger.warning("RQ/Redis unavailable (%s), falling back to DB queue", exc)
                self._rq_available = False
        else:
            logger.info("Task dispatch: REDIS_URL not set, using DB queue")

    def enqueue(self, task_type: str, resource_id: int, payload: dict[str, Any]) -> None:
        """Enqueue a background task."""
        if self._rq_available and self._queue is not None:
            # RQ path: dispatch to worker function
            self._queue.enqueue(
                _rq_task_handler,
                task_type,
                resource_id,
                payload,
                job_timeout=3600,
            )
        else:
            # DB queue fallback
            self._db_worker.enqueue(task_type, resource_id, payload)

    def start(self) -> None:
        """Start the DB queue worker (no-op if using RQ)."""
        if not self._rq_available and self._db_worker:
            self._db_worker.start()

    def stop(self) -> None:
        """Stop the DB queue worker (no-op if using RQ)."""
        if not self._rq_available and self._db_worker:
            self._db_worker.stop()

    @property
    def is_rq(self) -> bool:
        return self._rq_available


def _rq_task_handler(task_type: str, resource_id: int, payload: dict[str, Any]) -> None:
    """RQ worker function — runs in the rq-worker process."""
    # Import here to avoid circular imports in worker process
    if task_type == "migration":
        from rosettastone.server.api.tasks import run_migration_background

        run_migration_background(resource_id, payload)
    elif task_type == "pipeline":
        from rosettastone.server.pipeline_runner import run_pipeline_background

        run_pipeline_background(resource_id)
    else:
        logger.warning("Unknown task type: %s", task_type)
