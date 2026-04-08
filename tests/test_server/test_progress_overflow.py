"""Tests for SSE progress queue overflow handling in rosettastone.server.progress.

The progress module uses asyncio.Queue objects (maxsize=100) per SSE client.
These tests verify:
  1. emit_progress() with no registered clients is a no-op (no exceptions).
  2. register_client() / unregister_client() lifecycle works correctly.
  3. Flooding the queue beyond capacity drops events silently (QueueFull caught).

Because emit_progress() schedules work via loop.call_soon_threadsafe() when a
running event loop is present, and silently skips when no loop is running, the
tests that need to inspect queue state must run inside an async context so that
the asyncio machinery actually delivers the queued items.
"""

from __future__ import annotations

import asyncio

import rosettastone.server.progress as progress_module
from rosettastone.server.progress import (
    emit_progress,
    register_client,
    unregister_client,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_queues() -> None:
    """Clear the global _queues dict between tests to prevent state leakage."""
    progress_module._queues.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmitProgressNoClients:
    def test_emit_progress_no_clients_is_noop(self) -> None:
        """emit_progress() with no registered clients must not raise any exception.

        This covers the early-return guard:
            if migration_id not in _queues or not _queues[migration_id]:
                return
        """
        _clear_queues()
        # Should complete without exception even with no clients registered
        emit_progress(99999, {"stage": "baseline", "progress": 0.5})


class TestRegisterUnregisterLifecycle:
    def test_register_unregister_lifecycle(self) -> None:
        """register_client() creates a queue; unregister_client() removes it.

        After unregistering the last client for a migration_id, the entry must
        be removed from _queues entirely (not left as an empty set).
        """
        _clear_queues()
        migration_id = 42

        # Register a client -- queue must be created
        q = register_client(migration_id)
        assert migration_id in progress_module._queues, (
            "register_client() must add migration_id to _queues"
        )
        assert q in progress_module._queues[migration_id], (
            "Registered queue must be present in the client set"
        )
        assert isinstance(q, asyncio.Queue)

        # Unregister -- migration_id entry must be removed entirely
        unregister_client(migration_id, q)
        assert migration_id not in progress_module._queues, (
            "unregister_client() must remove the migration_id key when the set is empty"
        )

    def test_multiple_clients_same_migration(self) -> None:
        """Multiple clients can register for the same migration_id."""
        _clear_queues()
        migration_id = 7

        q1 = register_client(migration_id)
        q2 = register_client(migration_id)
        q3 = register_client(migration_id)

        assert len(progress_module._queues[migration_id]) == 3

        # Unregister one -- others remain
        unregister_client(migration_id, q1)
        assert migration_id in progress_module._queues
        assert len(progress_module._queues[migration_id]) == 2

        # Unregister rest -- key removed
        unregister_client(migration_id, q2)
        unregister_client(migration_id, q3)
        assert migration_id not in progress_module._queues


class TestEmitProgressQueueOverflow:
    def test_emit_progress_drops_on_queue_full(self) -> None:
        """Flooding emit_progress() beyond queue capacity must not raise.

        The asyncio.Queue has maxsize=100. Calling emit_progress() more than
        100 times must silently drop events via the QueueFull exception handler.
        The queue size must not exceed maxsize.

        We run inside asyncio.run() so that the event loop is present and
        loop.call_soon_threadsafe(_put_all) actually executes.
        """
        _clear_queues()

        async def _run() -> None:
            migration_id = 1
            q = register_client(migration_id)

            loop = asyncio.get_running_loop()

            # Emit 200 events -- 100 more than queue capacity
            for i in range(200):
                emit_progress(migration_id, {"i": i}, loop=loop)

            # Give the event loop a moment to process the call_soon_threadsafe callbacks
            await asyncio.sleep(0.05)

            # Queue must not exceed maxsize (100)
            assert q.qsize() <= q.maxsize, (
                f"Queue size {q.qsize()} exceeded maxsize {q.maxsize}. "
                "QueueFull should be caught and events dropped silently."
            )
            assert q.maxsize == 100

            # Queue must have exactly maxsize items (not zero -- some events did land)
            assert q.qsize() == q.maxsize, (
                f"Expected queue to be full ({q.maxsize}), but got {q.qsize()}. "
                "The first 100 events should have filled the queue."
            )

            # Cleanup
            unregister_client(migration_id, q)

        asyncio.run(_run())

    def test_emit_progress_no_exception_on_overflow_sync_context(self) -> None:
        """emit_progress() overflow path in no-loop context must not raise.

        When called from a context without a running event loop (e.g. a
        synchronous background thread), emit_progress() catches RuntimeError
        from asyncio.get_running_loop() and silently skips. This test verifies
        that even if a queue were full, the sync path raises no exceptions.
        """
        _clear_queues()
        migration_id = 2
        q = register_client(migration_id)

        # Pre-fill the queue to simulate it being full
        for i in range(100):
            try:
                q.put_nowait(f"item-{i}")
            except asyncio.QueueFull:
                break

        # Now emit without a running event loop -- must not raise
        # (the RuntimeError path in emit_progress silently skips)
        for _ in range(10):
            emit_progress(migration_id, {"overflow": True})

        # Queue size unchanged (events were dropped or skipped)
        assert q.qsize() <= q.maxsize

        unregister_client(migration_id, q)
        _clear_queues()
