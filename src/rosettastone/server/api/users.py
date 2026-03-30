"""User management API router."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, func, select

from rosettastone.server.auth_utils import hash_password
from rosettastone.server.database import get_session
from rosettastone.server.models import User
from rosettastone.server.rbac import require_role
from rosettastone.server.schemas import UserCreate, UserMe, UserUpdate

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _multi_user_gate() -> None:
    """Raise 404 if multi-user mode is not enabled."""
    if os.environ.get("ROSETTASTONE_MULTI_USER", "").lower() not in ("1", "true", "yes"):
        raise HTTPException(status_code=404, detail="Multi-user mode not enabled")


def _user_to_me(user: User) -> UserMe:
    """Convert a User model to UserMe schema."""
    return UserMe(
        id=user.id,  # type: ignore[arg-type]
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
    )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/users",
    response_model=list[UserMe],
    dependencies=[Depends(require_role("admin"))],
)
def list_users(
    session: Session = Depends(get_session),
) -> list[UserMe]:
    """List all users. Admin only. Requires ROSETTASTONE_MULTI_USER."""
    _multi_user_gate()
    users = list(session.exec(select(User).order_by(User.id)).all())
    return [_user_to_me(u) for u in users]


@router.post(
    "/api/v1/users",
    response_model=UserMe,
    status_code=201,
    dependencies=[Depends(require_role("admin"))],
)
def create_user(
    body: UserCreate,
    session: Session = Depends(get_session),
) -> UserMe:
    """Create a new user. Admin only. Requires ROSETTASTONE_MULTI_USER.

    The first user created automatically receives the admin role.
    Returns 409 if the username is already taken.
    """
    _multi_user_gate()

    existing = session.exec(select(User).where(User.username == body.username)).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Username '{body.username}' already exists")

    count = session.exec(select(func.count()).select_from(User)).one()
    role = "admin" if count == 0 else body.role

    user = User(
        username=body.username,
        hashed_password=hash_password(body.password),
        email=body.email,
        role=role,
        is_active=True,
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    return _user_to_me(user)


@router.get("/api/v1/users/{user_id}", response_model=UserMe)
def get_user(
    user_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> UserMe:
    """Get user detail. Admin or the user themselves. Requires ROSETTASTONE_MULTI_USER."""
    _multi_user_gate()

    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    current = getattr(request.state, "user", None)
    if current is not None:
        if isinstance(current, dict):
            current_role = current.get("role", "viewer")
            current_id = current.get("user_id")
        else:
            current_role = getattr(current, "role", "viewer")
            current_id = getattr(current, "id", None)
        if current_role != "admin" and current_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

    return _user_to_me(user)


@router.put("/api/v1/users/{user_id}", response_model=UserMe)
def update_user(
    user_id: int,
    body: UserUpdate,
    request: Request,
    session: Session = Depends(get_session),
) -> UserMe:
    """Update a user. Admin can update any field. Non-admins can only update own password/email.

    Requires ROSETTASTONE_MULTI_USER.
    """
    _multi_user_gate()

    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    current = getattr(request.state, "user", None)
    is_admin = False
    if current is not None:
        if isinstance(current, dict):
            current_role = current.get("role", "viewer")
        else:
            current_role = getattr(current, "role", "viewer")
        is_admin = current_role == "admin"

    if is_admin:
        if body.role is not None:
            user.role = body.role
        if body.is_active is not None:
            user.is_active = body.is_active
        if body.email is not None:
            user.email = body.email
        if body.password is not None:
            user.hashed_password = hash_password(body.password)
    else:
        if body.email is not None:
            user.email = body.email
        if body.password is not None:
            user.hashed_password = hash_password(body.password)

    session.add(user)
    session.commit()
    session.refresh(user)

    return _user_to_me(user)


@router.delete(
    "/api/v1/users/{user_id}",
    status_code=204,
    dependencies=[Depends(require_role("admin"))],
)
def delete_user(
    user_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> None:
    """Delete a user. Admin only. Cannot delete self. Requires ROSETTASTONE_MULTI_USER."""
    _multi_user_gate()

    current = getattr(request.state, "user", None)
    if current is not None:
        current_id = (
            current.get("user_id") if isinstance(current, dict) else getattr(current, "id", None)
        )
        if current_id == user_id:
            raise HTTPException(status_code=400, detail="Cannot delete your own account")

    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    session.delete(user)
    session.commit()
