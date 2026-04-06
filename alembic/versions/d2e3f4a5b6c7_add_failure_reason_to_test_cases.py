"""add_failure_reason_to_test_cases

Revision ID: d2e3f4a5b6c7
Revises: c1d2e3f4a5b6
Create Date: 2026-04-06 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d2e3f4a5b6c7"
down_revision: str | None = "c1d2e3f4a5b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Feature 6: Failure Reason Taxonomy — categorical error classification
    # Values: api_error / timeout / rate_limit / no_response / json_gate_failed
    op.add_column(
        "test_cases",
        sa.Column("failure_reason", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("test_cases", "failure_reason")
