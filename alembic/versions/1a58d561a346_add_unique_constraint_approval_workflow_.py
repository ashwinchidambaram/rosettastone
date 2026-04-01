"""add_unique_constraint_approval_workflow_user

Revision ID: 1a58d561a346
Revises: c39645f955dc
Create Date: 2026-04-01 00:17:31.862093

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1a58d561a346'
down_revision: Union[str, Sequence[str], None] = 'c39645f955dc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("approvals", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            "uq_approval_workflow_user", ["workflow_id", "user_id"]
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("approvals", schema=None) as batch_op:
        batch_op.drop_constraint("uq_approval_workflow_user", type_="unique")
