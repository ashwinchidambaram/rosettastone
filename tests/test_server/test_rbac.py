"""Tests for the RBAC require_role dependency."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Direct unit tests for the require_role dependency function
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_require_role_passes_without_multi_user(monkeypatch):
    """When ROSETTASTONE_MULTI_USER is not set, require_role is a no-op for all requests."""
    monkeypatch.delenv("ROSETTASTONE_MULTI_USER", raising=False)
    from rosettastone.server.rbac import require_role

    dep = require_role("admin")
    request = MagicMock()
    # Should not raise regardless of request state
    await dep(request)


@pytest.mark.asyncio
async def test_require_role_raises_401_no_user(monkeypatch):
    """With multi-user enabled and no user in request.state, require_role raises 401."""
    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    from rosettastone.server.rbac import require_role

    dep = require_role("admin")
    request = MagicMock()
    request.state.user = None

    with pytest.raises(HTTPException) as exc_info:
        await dep(request)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_require_role_raises_403_wrong_role(monkeypatch):
    """With multi-user enabled and a viewer user, require_role('admin') raises 403."""
    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    from rosettastone.server.rbac import require_role

    dep = require_role("admin")
    request = MagicMock()
    request.state.user = {"user_id": 1, "role": "viewer"}

    with pytest.raises(HTTPException) as exc_info:
        await dep(request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_require_role_passes_matching_role(monkeypatch):
    """With multi-user enabled and an admin user, require_role('admin') does not raise."""
    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    from rosettastone.server.rbac import require_role

    dep = require_role("admin")
    request = MagicMock()
    request.state.user = {"user_id": 1, "role": "admin"}

    # Should not raise
    await dep(request)


@pytest.mark.asyncio
async def test_require_role_passes_one_of_multiple_roles(monkeypatch):
    """require_role('editor', 'admin') should pass for an editor user."""
    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    from rosettastone.server.rbac import require_role

    dep = require_role("editor", "admin")
    request = MagicMock()
    request.state.user = {"user_id": 2, "role": "editor"}

    await dep(request)


@pytest.mark.asyncio
async def test_require_role_raises_403_for_viewer_on_editor_endpoint(monkeypatch):
    """A viewer cannot access an editor-only endpoint."""
    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    from rosettastone.server.rbac import require_role

    dep = require_role("editor", "admin")
    request = MagicMock()
    request.state.user = {"user_id": 3, "role": "viewer"}

    with pytest.raises(HTTPException) as exc_info:
        await dep(request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_require_role_handles_object_user(monkeypatch):
    """require_role works when request.state.user is an object (not dict)."""
    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    from rosettastone.server.rbac import require_role

    dep = require_role("admin")
    request = MagicMock()
    user_obj = MagicMock()
    user_obj.role = "admin"
    request.state.user = user_obj

    await dep(request)


@pytest.mark.asyncio
async def test_require_role_no_user_attr_on_state(monkeypatch):
    """When request.state has no 'user' attribute at all, require_role raises 401."""
    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    from rosettastone.server.rbac import require_role

    dep = require_role("admin")
    request = MagicMock(spec=["headers", "url"])
    # state without 'user' — getattr returns None
    request.state = MagicMock(spec=[])

    with pytest.raises(HTTPException) as exc_info:
        await dep(request)
    assert exc_info.value.status_code == 401
