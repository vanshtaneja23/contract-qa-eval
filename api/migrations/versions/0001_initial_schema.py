"""initial schema: matters, users, members, documents, chunks, audit log

Revision ID: 0001
Revises:
Create Date: 2026-10-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID_PK = dict(server_default=sa.text("gen_random_uuid()"), primary_key=True)


def _created_at() -> sa.Column:  # type: ignore[type-arg]
    return sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "matters",
        sa.Column("id", sa.Uuid(), **UUID_PK),
        sa.Column("name", sa.Text(), nullable=False),
        _created_at(),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), **UUID_PK),
        sa.Column("email", sa.Text(), nullable=False, unique=True),
        sa.Column("display_name", sa.Text(), nullable=False),
        _created_at(),
    )
    op.create_table(
        "matter_members",
        sa.Column("matter_id", sa.Uuid(), sa.ForeignKey("matters.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("role", sa.Text(), server_default="member", nullable=False),
        sa.CheckConstraint("role IN ('owner', 'member')", name="ck_matter_members_role"),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), **UUID_PK),
        sa.Column("matter_id", sa.Uuid(), sa.ForeignKey("matters.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        _created_at(),
        sa.UniqueConstraint("matter_id", "sha256", name="uq_documents_matter_sha256"),
        sa.UniqueConstraint("id", "matter_id", name="uq_documents_id_matter"),
    )
    op.create_index("ix_documents_matter_id", "documents", ["matter_id"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("matter_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id", "matter_id"],
            ["documents.id", "documents.matter_id"],
            ondelete="CASCADE",
            name="fk_chunks_document_matter",
        ),
        sa.UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
        sa.CheckConstraint("start_char >= 0 AND end_char > start_char", name="ck_chunks_offsets"),
        sa.CheckConstraint("char_length(text) = end_char - start_char", name="ck_chunks_text_length"),
        sa.CheckConstraint("kind IN ('section', 'window')", name="ck_chunks_kind"),
    )
    op.create_index("ix_chunks_matter_id", "chunks", ["matter_id"])
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")

    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        _created_at(),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("matter_id", sa.Uuid(), sa.ForeignKey("matters.id", ondelete="SET NULL"), nullable=True),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("detail", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
    )


def downgrade() -> None:
    for table in ("audit_log", "chunks", "documents", "matter_members", "users", "matters"):
        op.drop_table(table)
