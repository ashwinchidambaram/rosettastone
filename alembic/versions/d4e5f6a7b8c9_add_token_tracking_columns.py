"""add_token_tracking_columns

Revision ID: d4e5f6a7b8c9
Revises: b7c8d9e0f1a2
Create Date: 2026-04-05 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: str | Sequence[str] | None = "b7c8d9e0f1a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add token tracking columns to migrations table."""
    with op.batch_alter_table("migrations") as batch_op:
        batch_op.add_column(sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("token_breakdown_json", sqlmodel.sql.sqltypes.AutoString(), nullable=False, server_default="{}"))


def downgrade() -> None:
    """Remove token tracking columns from migrations table."""
    with op.batch_alter_table("migrations") as batch_op:
        batch_op.drop_column("token_breakdown_json")
        batch_op.drop_column("total_tokens")
