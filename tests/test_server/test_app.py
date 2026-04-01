"""Tests for app factory and Sentry integration."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


def test_init_sentry_calls_sdk_when_dsn_set(monkeypatch) -> None:
    """_init_sentry() calls sentry_sdk.init with correct DSN."""
    monkeypatch.setenv("SENTRY_DSN", "https://fake@sentry.io/123")
    mock_sdk = MagicMock()
    with patch.dict("sys.modules", {"sentry_sdk": mock_sdk}):
        from rosettastone.server.app import _init_sentry

        _init_sentry()
    mock_sdk.init.assert_called_once()
    assert mock_sdk.init.call_args.kwargs["dsn"] == "https://fake@sentry.io/123"
    assert mock_sdk.init.call_args.kwargs.get("traces_sample_rate") == 0.1
    assert mock_sdk.init.call_args.kwargs.get("profiles_sample_rate") == 0.1


def test_init_sentry_noop_when_dsn_absent(monkeypatch) -> None:
    """_init_sentry() does nothing when SENTRY_DSN is not set."""
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    mock_sdk = MagicMock()
    with patch.dict("sys.modules", {"sentry_sdk": mock_sdk}):
        from rosettastone.server.app import _init_sentry

        _init_sentry()
    mock_sdk.init.assert_not_called()


def test_init_sentry_graceful_import_error(monkeypatch) -> None:
    """_init_sentry() does not crash when sentry-sdk is not installed."""
    monkeypatch.setenv("SENTRY_DSN", "https://fake@sentry.io/123")
    # Simulate sentry_sdk not installed by setting it to None in sys.modules
    # This will trigger ImportError when trying to import it
    saved = sys.modules.get("sentry_sdk")
    sys.modules["sentry_sdk"] = None  # type: ignore[assignment]
    try:
        from rosettastone.server.app import _init_sentry

        _init_sentry()  # Should not raise
    finally:
        if saved is not None:
            sys.modules["sentry_sdk"] = saved
        elif "sentry_sdk" in sys.modules:
            del sys.modules["sentry_sdk"]


def test_create_app_succeeds(monkeypatch) -> None:
    """create_app() successfully creates a FastAPI app."""
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    from rosettastone.server.app import create_app

    app = create_app()
    assert app is not None
    assert app.title == "RosettaStone"
