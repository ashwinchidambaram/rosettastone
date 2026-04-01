"""Tests for TaskDispatcher — RQ vs DB queue fallback."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

from rosettastone.server.task_dispatch import TaskDispatcher


def _make_fake_rq_modules():
    """Build fake redis and rq modules for use in tests without real deps."""
    fake_redis_mod = types.ModuleType("redis")
    fake_rq_mod = types.ModuleType("rq")

    mock_conn = MagicMock()
    mock_conn.ping.return_value = True
    mock_queue = MagicMock()

    fake_redis_mod.from_url = MagicMock(return_value=mock_conn)  # type: ignore[attr-defined]
    fake_rq_mod.Queue = MagicMock(return_value=mock_queue)  # type: ignore[attr-defined]

    return fake_redis_mod, fake_rq_mod, mock_conn, mock_queue


class TestTaskDispatcherDBFallback:
    def test_uses_db_queue_when_redis_url_not_set(self, monkeypatch) -> None:
        """TaskDispatcher uses DB queue when REDIS_URL is not set."""
        monkeypatch.delenv("REDIS_URL", raising=False)
        mock_db_worker = MagicMock()

        dispatcher = TaskDispatcher()
        dispatcher.setup(mock_db_worker)

        assert not dispatcher.is_rq

    def test_enqueue_delegates_to_db_worker(self, monkeypatch) -> None:
        """enqueue() calls db_worker.enqueue when RQ unavailable."""
        monkeypatch.delenv("REDIS_URL", raising=False)
        mock_db_worker = MagicMock()

        dispatcher = TaskDispatcher()
        dispatcher.setup(mock_db_worker)
        dispatcher.enqueue("migration", 42, {"key": "val"})

        mock_db_worker.enqueue.assert_called_once_with("migration", 42, {"key": "val"})

    def test_start_delegates_to_db_worker(self, monkeypatch) -> None:
        """start() calls db_worker.start when not using RQ."""
        monkeypatch.delenv("REDIS_URL", raising=False)
        mock_db_worker = MagicMock()

        dispatcher = TaskDispatcher()
        dispatcher.setup(mock_db_worker)
        dispatcher.start()

        mock_db_worker.start.assert_called_once()

    def test_stop_delegates_to_db_worker(self, monkeypatch) -> None:
        """stop() calls db_worker.stop when not using RQ."""
        monkeypatch.delenv("REDIS_URL", raising=False)
        mock_db_worker = MagicMock()

        dispatcher = TaskDispatcher()
        dispatcher.setup(mock_db_worker)
        dispatcher.stop()

        mock_db_worker.stop.assert_called_once()


class TestTaskDispatcherRQPath:
    def test_uses_rq_when_redis_available(self, monkeypatch) -> None:
        """TaskDispatcher uses RQ when REDIS_URL is set and Redis responds."""
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        mock_db_worker = MagicMock()

        fake_redis_mod, fake_rq_mod, _conn, _queue = _make_fake_rq_modules()
        monkeypatch.setitem(sys.modules, "redis", fake_redis_mod)
        monkeypatch.setitem(sys.modules, "rq", fake_rq_mod)

        dispatcher = TaskDispatcher()
        dispatcher.setup(mock_db_worker)

        assert dispatcher.is_rq

    def test_enqueue_uses_rq_when_available(self, monkeypatch) -> None:
        """enqueue() calls rq.Queue.enqueue when RQ is available."""
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        mock_db_worker = MagicMock()

        fake_redis_mod, fake_rq_mod, _conn, mock_queue = _make_fake_rq_modules()
        monkeypatch.setitem(sys.modules, "redis", fake_redis_mod)
        monkeypatch.setitem(sys.modules, "rq", fake_rq_mod)

        dispatcher = TaskDispatcher()
        dispatcher.setup(mock_db_worker)
        dispatcher.enqueue("migration", 42, {"key": "val"})

        mock_queue.enqueue.assert_called_once()
        call_args = mock_queue.enqueue.call_args
        assert call_args.args[1] == "migration"
        assert call_args.args[2] == 42
        mock_db_worker.enqueue.assert_not_called()

    def test_falls_back_to_db_on_redis_error(self, monkeypatch) -> None:
        """Falls back to DB queue when Redis connection fails."""
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        mock_db_worker = MagicMock()

        fake_redis_mod = types.ModuleType("redis")
        fake_redis_mod.from_url = MagicMock(side_effect=Exception("Connection refused"))  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "redis", fake_redis_mod)

        dispatcher = TaskDispatcher()
        dispatcher.setup(mock_db_worker)

        assert not dispatcher.is_rq
        dispatcher.enqueue("migration", 1, {})
        mock_db_worker.enqueue.assert_called_once()

    def test_start_noop_when_rq(self, monkeypatch) -> None:
        """start() is a no-op when using RQ."""
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        mock_db_worker = MagicMock()

        fake_redis_mod, fake_rq_mod, _conn, _queue = _make_fake_rq_modules()
        monkeypatch.setitem(sys.modules, "redis", fake_redis_mod)
        monkeypatch.setitem(sys.modules, "rq", fake_rq_mod)

        dispatcher = TaskDispatcher()
        dispatcher.setup(mock_db_worker)
        dispatcher.start()

        mock_db_worker.start.assert_not_called()

    def test_stop_noop_when_rq(self, monkeypatch) -> None:
        """stop() is a no-op when using RQ."""
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        mock_db_worker = MagicMock()

        fake_redis_mod, fake_rq_mod, _conn, _queue = _make_fake_rq_modules()
        monkeypatch.setitem(sys.modules, "redis", fake_redis_mod)
        monkeypatch.setitem(sys.modules, "rq", fake_rq_mod)

        dispatcher = TaskDispatcher()
        dispatcher.setup(mock_db_worker)
        dispatcher.stop()

        mock_db_worker.stop.assert_not_called()
