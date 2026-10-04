"""drop chunks.kind: section-aware chunking was removed (no measured benefit, see DECISIONS.md)

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("ck_chunks_kind", "chunks", type_="check")
    op.drop_column("chunks", "kind")


def downgrade() -> None:
    op.add_column("chunks", sa.Column("kind", sa.Text(), nullable=False, server_default="window"))
    op.alter_column("chunks", "kind", server_default=None)
    op.create_check_constraint("ck_chunks_kind", "chunks", "kind IN ('section', 'window')")
