"""Tests for JWT secret startup validation in app.py."""

from __future__ import annotations

import logging

import pytest


def _call_check(monkeypatch, *, multi_user: str, secret: str | None) -> None:
    """Helper: set env vars and invoke _check_jwt_secret()."""
    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", multi_user)
    if secret is None:
        monkeypatch.delenv("ROSETTASTONE_JWT_SECRET", raising=False)
    else:
        monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", secret)

    # Re-import to pick up the function after env is patched
    from rosettastone.server.app import _check_jwt_secret

    _check_jwt_secret()


def test_default_secret_warns_when_multi_user_enabled(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="rosettastone.server"):
        _call_check(monkeypatch, multi_user="true", secret=None)

    assert any(
        "default dev value" in record.message
        for record in caplog.records
        if record.name == "rosettastone.server"
    )


def test_short_custom_secret_warns_when_multi_user_enabled(monkeypatch, caplog):
    short_secret = "tooshort"  # 8 bytes < 32
    with caplog.at_level(logging.WARNING, logger="rosettastone.server"):
        _call_check(monkeypatch, multi_user="true", secret=short_secret)

    messages = [r.message for r in caplog.records if r.name == "rosettastone.server"]
    assert any("8 bytes" in msg and "32 bytes" in msg for msg in messages)


def test_secure_secret_no_warning_when_multi_user_enabled(monkeypatch, caplog):
    secure_secret = "a" * 32  # exactly 32 bytes
    with caplog.at_level(logging.WARNING, logger="rosettastone.server"):
        _call_check(monkeypatch, multi_user="true", secret=secure_secret)

    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == "rosettastone.server"
    ]
    assert warnings == []


def test_no_warning_when_multi_user_disabled_default_secret(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="rosettastone.server"):
        _call_check(monkeypatch, multi_user="false", secret=None)

    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == "rosettastone.server"
    ]
    assert warnings == []


def test_no_warning_when_multi_user_disabled_short_secret(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="rosettastone.server"):
        _call_check(monkeypatch, multi_user="0", secret="tiny")

    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == "rosettastone.server"
    ]
    assert warnings == []


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "True", "YES"])
def test_multi_user_truthy_variants_all_warn_with_default(monkeypatch, caplog, truthy):
    with caplog.at_level(logging.WARNING, logger="rosettastone.server"):
        _call_check(monkeypatch, multi_user=truthy, secret=None)

    assert any(
        "default dev value" in record.message
        for record in caplog.records
        if record.name == "rosettastone.server"
    ), f"Expected warning for ROSETTASTONE_MULTI_USER={truthy!r}"
