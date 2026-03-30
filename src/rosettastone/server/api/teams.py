"""Team management API router."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from sqlmodel import Session, select

from rosettastone.server.database import get_session
from rosettastone.server.models import Team, TeamMembership, User
from rosettastone.server.rbac import require_role
from rosettastone.server.schemas import AddTeamMember, TeamCreate, TeamMemberSummary, TeamSummary

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _multi_user_enabled() -> bool:
    return os.environ.get("ROSETTASTONE_MULTI_USER", "").lower() in ("1", "true", "yes")


def _require_multi_user() -> None:
    if not _multi_user_enabled():
        raise HTTPException(status_code=404, detail="Multi-user mode is not enabled")


def _team_to_summary(team: Team) -> TeamSummary:
    return TeamSummary(
        id=team.id,  # type: ignore[arg-type]
        name=team.name,
        created_at=team.created_at,
    )


def _membership_to_summary(membership: TeamMembership) -> TeamMemberSummary:
    return TeamMemberSummary(
        user_id=membership.user_id,
        team_id=membership.team_id,
        role=membership.role,
    )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@router.get("/ui/teams", response_class=HTMLResponse)
async def teams_page(
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """Render the teams UI page."""
    multi_user = _multi_user_enabled()
    teams = list(session.exec(select(Team).order_by(Team.id)).all()) if multi_user else []
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "teams.html",
        {"active_nav": "teams", "teams": teams, "multi_user": multi_user},
    )


@router.get("/api/v1/teams", response_model=list[TeamSummary])
def list_teams(
    session: Session = Depends(get_session),
) -> list[TeamSummary]:
    """List all teams. Returns 404 when multi-user mode is not enabled."""
    _require_multi_user()
    teams = list(session.exec(select(Team)).all())
    return [_team_to_summary(t) for t in teams]


@router.post(
    "/api/v1/teams",
    response_model=TeamSummary,
    status_code=201,
    dependencies=[Depends(require_role("admin"))],
)
def create_team(
    body: TeamCreate,
    session: Session = Depends(get_session),
) -> TeamSummary:
    """Create a new team. Admin only. Returns 409 if name already exists."""
    _require_multi_user()

    existing = session.exec(select(Team).where(Team.name == body.name)).first()  # type: ignore[arg-type]
    if existing:
        raise HTTPException(status_code=409, detail=f"Team '{body.name}' already exists")

    team = Team(name=body.name)
    session.add(team)
    session.commit()
    session.refresh(team)

    return _team_to_summary(team)


@router.get("/api/v1/teams/{team_id}", response_model=TeamSummary)
def get_team(
    team_id: int,
    session: Session = Depends(get_session),
) -> TeamSummary:
    """Return team detail. Returns 404 if team not found or multi-user disabled."""
    _require_multi_user()

    team = session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    return _team_to_summary(team)


@router.delete(
    "/api/v1/teams/{team_id}",
    status_code=204,
    dependencies=[Depends(require_role("admin"))],
)
def delete_team(
    team_id: int,
    session: Session = Depends(get_session),
) -> Response:
    """Delete a team and all its memberships. Admin only."""
    _require_multi_user()

    team = session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    memberships = list(
        session.exec(select(TeamMembership).where(TeamMembership.team_id == team_id)).all()  # type: ignore[arg-type]
    )
    for membership in memberships:
        session.delete(membership)

    session.delete(team)
    session.commit()

    return Response(status_code=204)


@router.post(
    "/api/v1/teams/{team_id}/members",
    response_model=TeamMemberSummary,
    status_code=201,
    dependencies=[Depends(require_role("admin"))],
)
def add_team_member(
    team_id: int,
    body: AddTeamMember,
    session: Session = Depends(get_session),
) -> TeamMemberSummary:
    """Add a member to a team. Admin only. 404 if team/user not found, 409 if already member."""
    _require_multi_user()

    team = session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    user = session.get(User, body.user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {body.user_id} not found")

    existing = session.exec(
        select(TeamMembership).where(  # type: ignore[arg-type]
            TeamMembership.team_id == team_id,
            TeamMembership.user_id == body.user_id,
        )
    ).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"User {body.user_id} is already a member of team {team_id}",
        )

    membership = TeamMembership(user_id=body.user_id, team_id=team_id, role=body.role)
    session.add(membership)
    session.commit()
    session.refresh(membership)

    return _membership_to_summary(membership)


@router.delete(
    "/api/v1/teams/{team_id}/members/{user_id}",
    status_code=204,
    dependencies=[Depends(require_role("admin"))],
)
def remove_team_member(
    team_id: int,
    user_id: int,
    session: Session = Depends(get_session),
) -> Response:
    """Remove a member from a team. Admin only. Returns 404 if team or membership not found."""
    _require_multi_user()

    team = session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    membership = session.exec(
        select(TeamMembership).where(  # type: ignore[arg-type]
            TeamMembership.team_id == team_id,
            TeamMembership.user_id == user_id,
        )
    ).first()
    if not membership:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} is not a member of team {team_id}",
        )

    session.delete(membership)
    session.commit()

    return Response(status_code=204)


@router.get("/api/v1/teams/{team_id}/members", response_model=list[TeamMemberSummary])
def list_team_members(
    team_id: int,
    session: Session = Depends(get_session),
) -> list[TeamMemberSummary]:
    """List all members of a team. Returns 404 if team not found or multi-user disabled."""
    _require_multi_user()

    team = session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    memberships = list(
        session.exec(select(TeamMembership).where(TeamMembership.team_id == team_id)).all()  # type: ignore[arg-type]
    )
    return [_membership_to_summary(m) for m in memberships]
