"""Role-based access control dependency for FastAPI endpoints."""

from __future__ import annotations

from fastapi import HTTPException, Request


# Roles are checked by exact membership — not hierarchy.
# To allow both editors and admins, pass require_role("editor", "admin").
def require_role(*roles: str):
    """FastAPI dependency factory that requires one of the specified roles.

    When ROSETTASTONE_MULTI_USER is not set, this is a no-op (all requests pass).
    When set, checks request.state.user against the allowed roles.

    Usage:
        @router.post("/something", dependencies=[Depends(require_role("editor", "admin"))])
    """

    async def _check(request: Request) -> None:
        import os

        multi_user = os.environ.get("ROSETTASTONE_MULTI_USER", "").lower() in ("1", "true", "yes")
        if not multi_user:
            return  # No-op when multi-user disabled

        user = getattr(request.state, "user", None)
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required")

        if isinstance(user, dict):
            user_role = user.get("role", "viewer")
        else:
            user_role = getattr(user, "role", "viewer")
        if user_role not in roles:
            raise HTTPException(status_code=403, detail=f"Requires one of roles: {list(roles)}")

    return _check


async def get_current_user(request: Request) -> dict | None:
    """Return the current user from request.state, or None in legacy mode."""
    return getattr(request.state, "user", None)
