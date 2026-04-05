"""add_optimization_score_history

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-04-05 13:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: str | Sequence[str] | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add optimization_score_history_json column to migrations table."""
    with op.batch_alter_table("migrations") as batch_op:
        batch_op.add_column(
            sa.Column(
                "optimization_score_history_json",
                sqlmodel.sql.sqltypes.AutoString(),
                nullable=False,
                server_default="[]",
            )
        )


def downgrade() -> None:
    """Remove optimization_score_history_json column from migrations table."""
    with op.batch_alter_table("migrations") as batch_op:
        batch_op.drop_column("optimization_score_history_json")
