"""production_readiness_v1

Revision ID: a1b2c3d4e5f6
Revises: 1a58d561a346
Create Date: 2026-04-01 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "1a58d561a346"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add production-readiness columns and tables."""
    # --- New tables ---
    op.create_table(
        "task_queue",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("task_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("resource_id", sa.Integer(), nullable=True),
        sa.Column("payload_json", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("worker_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("max_retries", sa.Integer(), nullable=False),
        sa.Column("error_message", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("correlation_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "user_budgets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("monthly_limit_usd", sa.Float(), nullable=False),
        sa.Column("current_month_spend_usd", sa.Float(), nullable=False),
        sa.Column("budget_month", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    # --- Add columns to existing tables ---
    with op.batch_alter_table("migrations", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("checkpoint_stage", sqlmodel.sql.sqltypes.AutoString(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("checkpoint_data_json", sqlmodel.sql.sqltypes.AutoString(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("current_stage", sqlmodel.sql.sqltypes.AutoString(), nullable=True)
        )
        batch_op.add_column(sa.Column("stage_progress", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("overall_progress", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("max_cost_usd", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("estimated_cost_usd", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("owner_id", sa.Integer(), nullable=True))

    with op.batch_alter_table("pipelines", schema=None) as batch_op:
        batch_op.add_column(sa.Column("owner_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("overall_progress", sa.Float(), nullable=True))
        batch_op.add_column(
            sa.Column("current_module", sqlmodel.sql.sqltypes.AutoString(), nullable=True)
        )

    with op.batch_alter_table("ab_tests", schema=None) as batch_op:
        batch_op.add_column(sa.Column("owner_id", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Remove production-readiness columns and tables."""
    with op.batch_alter_table("ab_tests", schema=None) as batch_op:
        batch_op.drop_column("owner_id")

    with op.batch_alter_table("pipelines", schema=None) as batch_op:
        batch_op.drop_column("current_module")
        batch_op.drop_column("overall_progress")
        batch_op.drop_column("owner_id")

    with op.batch_alter_table("migrations", schema=None) as batch_op:
        batch_op.drop_column("owner_id")
        batch_op.drop_column("estimated_cost_usd")
        batch_op.drop_column("max_cost_usd")
        batch_op.drop_column("overall_progress")
        batch_op.drop_column("stage_progress")
        batch_op.drop_column("current_stage")
        batch_op.drop_column("checkpoint_data_json")
        batch_op.drop_column("checkpoint_stage")

    op.drop_table("user_budgets")
    op.drop_table("task_queue")
