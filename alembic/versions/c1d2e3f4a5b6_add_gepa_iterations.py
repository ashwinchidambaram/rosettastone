"""add_gepa_iterations

Revision ID: c1d2e3f4a5b6
Revises: b7c8d9e0f1a2
Create Date: 2026-04-05 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c1d2e3f4a5b6"
down_revision: str | Sequence[str] | None = "b7c8d9e0f1a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add gepa_iterations table for GEPA optimizer iteration history."""
    op.create_table(
        "gepa_iterations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("migration_id", sa.Integer(), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("total_iterations", sa.Integer(), nullable=False),
        sa.Column("mean_score", sa.Float(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["migration_id"], ["migrations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_gepa_iterations_migration_id"),
        "gepa_iterations",
        ["migration_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop gepa_iterations table."""
    op.drop_index(op.f("ix_gepa_iterations_migration_id"), table_name="gepa_iterations")
    op.drop_table("gepa_iterations")
