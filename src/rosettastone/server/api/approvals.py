"""Approval workflow API router."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, func, select

from rosettastone.server.api.audit import log_audit
from rosettastone.server.database import get_session
from rosettastone.server.models import Approval, ApprovalWorkflow, MigrationRecord
from rosettastone.server.rbac import require_role
from rosettastone.server.schemas import (
    ApprovalCreate,
    ApprovalWorkflowCreate,
    ApprovalWorkflowSummary,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _multi_user_gate() -> None:
    """Raise 404 if multi-user mode is not enabled."""
    if os.environ.get("ROSETTASTONE_MULTI_USER", "").lower() not in ("1", "true", "yes"):
        raise HTTPException(status_code=404, detail="Multi-user mode not enabled")


def _count_approvals(workflow_id: int, session: Session) -> int:
    """Count Approval rows with decision='approve' for a given workflow."""
    stmt = (
        select(func.count())
        .select_from(Approval)
        .where(Approval.workflow_id == workflow_id)  # type: ignore[arg-type]
        .where(Approval.decision == "approve")
    )
    return session.exec(stmt).one()


def _workflow_to_summary(
    workflow: ApprovalWorkflow, current_approvals: int
) -> ApprovalWorkflowSummary:
    """Convert an ApprovalWorkflow model to ApprovalWorkflowSummary schema."""
    return ApprovalWorkflowSummary(
        id=workflow.id,  # type: ignore[arg-type]
        migration_id=workflow.migration_id,
        required_approvals=workflow.required_approvals,
        status=workflow.status,
        current_approvals=current_approvals,
    )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/migrations/{migration_id}/approval-workflow",
    response_model=ApprovalWorkflowSummary,
    status_code=201,
    dependencies=[Depends(require_role("admin"))],
)
def create_approval_workflow(
    migration_id: int,
    body: ApprovalWorkflowCreate,
    session: Session = Depends(get_session),
) -> ApprovalWorkflowSummary:
    """Create an approval workflow for a migration. Admin only.

    Returns 404 if the migration does not exist.
    Returns 409 if a workflow already exists for this migration.
    Requires ROSETTASTONE_MULTI_USER.
    """
    _multi_user_gate()

    migration = session.get(MigrationRecord, migration_id)
    if not migration:
        raise HTTPException(status_code=404, detail=f"Migration {migration_id} not found")

    existing = session.exec(
        select(ApprovalWorkflow).where(ApprovalWorkflow.migration_id == migration_id)  # type: ignore[arg-type]
    ).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Approval workflow already exists for migration {migration_id}",
        )

    workflow = ApprovalWorkflow(
        migration_id=migration_id,
        required_approvals=body.required_approvals,
        status="pending",
    )
    session.add(workflow)
    session.flush()  # populate workflow.id before audit log

    log_audit(
        session,
        "approval_workflow",
        workflow.id,
        "create",
        details={"migration_id": migration_id},
    )

    session.commit()
    session.refresh(workflow)

    return _workflow_to_summary(workflow, 0)


@router.get(
    "/api/v1/migrations/{migration_id}/approval-status",
    response_model=ApprovalWorkflowSummary,
    dependencies=[Depends(require_role("viewer", "editor", "approver", "admin"))],
)
def get_approval_status(
    migration_id: int,
    session: Session = Depends(get_session),
) -> ApprovalWorkflowSummary:
    """Return current approval workflow status for a migration.

    Returns 404 if no workflow exists for this migration.
    Requires ROSETTASTONE_MULTI_USER.
    """
    _multi_user_gate()

    workflow = session.exec(
        select(ApprovalWorkflow).where(ApprovalWorkflow.migration_id == migration_id)  # type: ignore[arg-type]
    ).first()
    if not workflow:
        raise HTTPException(
            status_code=404,
            detail=f"No approval workflow found for migration {migration_id}",
        )

    current_approvals = _count_approvals(workflow.id, session)  # type: ignore[arg-type]
    return _workflow_to_summary(workflow, current_approvals)


@router.post(
    "/api/v1/migrations/{migration_id}/approve",
    response_model=ApprovalWorkflowSummary,
    dependencies=[Depends(require_role("approver", "admin"))],
)
def submit_approval(
    migration_id: int,
    body: ApprovalCreate,
    request: Request,
    session: Session = Depends(get_session),
) -> ApprovalWorkflowSummary:
    """Submit an approval for a migration.

    Creates an Approval row with decision='approve'. Auto-approves the workflow
    when the approval count reaches required_approvals.
    Returns 404 if no workflow exists.
    Returns 400 if the workflow is not in 'pending' status.
    Requires ROSETTASTONE_MULTI_USER.
    """
    _multi_user_gate()

    workflow = session.exec(
        select(ApprovalWorkflow).where(ApprovalWorkflow.migration_id == migration_id)  # type: ignore[arg-type]
    ).first()
    if not workflow:
        raise HTTPException(
            status_code=404,
            detail=f"No approval workflow found for migration {migration_id}",
        )

    if workflow.status != "pending":
        raise HTTPException(status_code=400, detail="Workflow is not pending")

    user_id = request.state.user.get("user_id") if request.state.user else None

    approval = Approval(
        workflow_id=workflow.id,
        user_id=user_id,
        decision="approve",
        comment=body.comment,
    )
    session.add(approval)
    session.flush()  # persist approval so count includes it

    current_approvals = _count_approvals(workflow.id, session)  # type: ignore[arg-type]

    if current_approvals >= workflow.required_approvals:
        workflow.status = "approved"
        session.add(workflow)

    log_audit(
        session,
        "approval_workflow",
        workflow.id,
        "approve",
        details={"migration_id": migration_id},
    )

    session.commit()
    session.refresh(workflow)

    return _workflow_to_summary(workflow, current_approvals)


@router.post(
    "/api/v1/migrations/{migration_id}/reject",
    response_model=ApprovalWorkflowSummary,
    dependencies=[Depends(require_role("approver", "admin"))],
)
def submit_rejection(
    migration_id: int,
    body: ApprovalCreate,
    request: Request,
    session: Session = Depends(get_session),
) -> ApprovalWorkflowSummary:
    """Submit a rejection for a migration.

    Creates an Approval row with decision='reject', then resets the workflow:
    deletes all existing Approval rows and sets status back to 'pending'.
    This allows a fresh round of approvals after a rejection.
    Returns 404 if no workflow exists.
    Returns 400 if the workflow is not in 'pending' status.
    Requires ROSETTASTONE_MULTI_USER.
    """
    _multi_user_gate()

    workflow = session.exec(
        select(ApprovalWorkflow).where(ApprovalWorkflow.migration_id == migration_id)  # type: ignore[arg-type]
    ).first()
    if not workflow:
        raise HTTPException(
            status_code=404,
            detail=f"No approval workflow found for migration {migration_id}",
        )

    if workflow.status != "pending":
        raise HTTPException(status_code=400, detail="Workflow is not pending")

    user_id = request.state.user.get("user_id") if request.state.user else None

    rejection = Approval(
        workflow_id=workflow.id,
        user_id=user_id,
        decision="reject",
        comment=body.comment,
    )
    session.add(rejection)
    session.flush()

    # Reset logic: delete all Approval rows for this workflow and reset status
    existing_approvals = list(
        session.exec(
            select(Approval).where(Approval.workflow_id == workflow.id)  # type: ignore[arg-type]
        ).all()
    )
    for a in existing_approvals:
        session.delete(a)

    workflow.status = "pending"
    session.add(workflow)

    log_audit(
        session,
        "approval_workflow",
        workflow.id,
        "reject",
        details={"migration_id": migration_id},
    )

    session.commit()
    session.refresh(workflow)

    return _workflow_to_summary(workflow, 0)
