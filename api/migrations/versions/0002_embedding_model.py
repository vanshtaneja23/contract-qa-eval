"""record which model produced each chunk embedding

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("embedding_model", sa.Text(), nullable=True))
    op.create_check_constraint(
        "ck_chunks_embedding_model", "chunks", "(embedding IS NULL) = (embedding_model IS NULL)"
    )


def downgrade() -> None:
    op.drop_constraint("ck_chunks_embedding_model", "chunks", type_="check")
    op.drop_column("chunks", "embedding_model")
