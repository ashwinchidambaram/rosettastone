"""SSE progress hub for migration status streaming."""

from __future__ import annotations

import asyncio
import json
from typing import Any

# Map migration_id -> set of asyncio.Queue objects (one per SSE client)
_queues: dict[int, set[asyncio.Queue[str | None]]] = {}


def _get_or_create_queue_set(migration_id: int) -> set[asyncio.Queue[str | None]]:
    if migration_id not in _queues:
        _queues[migration_id] = set()
    return _queues[migration_id]


def emit_progress(
    migration_id: int,
    data: dict[str, Any],
    loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    """Emit a progress event to all SSE clients watching migration_id.

    Thread-safe: background worker threads call this.
    If loop is None, tries to get the running loop. If no loop is running,
    silently skips (e.g. during unit tests).
    """
    if migration_id not in _queues or not _queues[migration_id]:
        return

    payload = json.dumps(data)

    def _put_all() -> None:
        qs = _queues.get(migration_id, set())
        for q in list(qs):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass  # Drop if slow consumer

    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(_put_all)
    else:
        try:
            running = asyncio.get_running_loop()
            running.call_soon_threadsafe(_put_all)
        except RuntimeError:
            pass  # No running loop — skip (unit test context)


def register_client(migration_id: int) -> asyncio.Queue[str | None]:
    """Register a new SSE client and return its queue."""
    q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=100)
    _get_or_create_queue_set(migration_id).add(q)
    return q


def unregister_client(migration_id: int, q: asyncio.Queue[str | None]) -> None:
    """Remove a client queue when the SSE connection closes."""
    qs = _queues.get(migration_id)
    if qs:
        qs.discard(q)
        if not qs:
            del _queues[migration_id]
