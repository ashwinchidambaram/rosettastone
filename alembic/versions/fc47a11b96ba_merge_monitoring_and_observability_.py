"""merge_monitoring_and_observability_branches

Revision ID: fc47a11b96ba
Revises: d2e3f4a5b6c7, e5f6a7b8c9d0
Create Date: 2026-04-05 19:59:32.513228

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fc47a11b96ba'
down_revision: Union[str, Sequence[str], None] = ('d2e3f4a5b6c7', 'e5f6a7b8c9d0')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
