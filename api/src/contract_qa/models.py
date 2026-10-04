"""Database schema. The Alembic migration in api/migrations is the source of truth
for DDL; an integration test checks these models and the migration agree."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# all-MiniLM-L6-v2 and bge-small-en-v1.5 both produce 384-d vectors (chosen in M2).
EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(primary_key=True, server_default=text("gen_random_uuid()"))


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


class Matter(Base):
    __tablename__ = "matters"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(Text, unique=True)
    display_name: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


class MatterMember(Base):
    __tablename__ = "matter_members"
    __table_args__ = (CheckConstraint("role IN ('owner', 'member')", name="ck_matter_members_role"),)

    matter_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("matters.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role: Mapped[str] = mapped_column(Text, server_default="member")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("matter_id", "sha256", name="uq_documents_matter_sha256"),
        # Target for the composite FK from chunks (see Chunk.__table_args__).
        UniqueConstraint("id", "matter_id", name="uq_documents_id_matter"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    matter_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("matters.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    char_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = _created_at()


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        # matter_id is copied onto chunks so retrieval and row-level security can
        # filter on it without a join. The composite FK makes it impossible for
        # the copy to disagree with the parent document's matter.
        ForeignKeyConstraint(
            ["document_id", "matter_id"],
            ["documents.id", "documents.matter_id"],
            ondelete="CASCADE",
            name="fk_chunks_document_matter",
        ),
        UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
        CheckConstraint("start_char >= 0 AND end_char > start_char", name="ck_chunks_offsets"),
        CheckConstraint("char_length(text) = end_char - start_char", name="ck_chunks_text_length"),
        CheckConstraint("kind IN ('section', 'window')", name="ck_chunks_kind"),
        CheckConstraint("(embedding IS NULL) = (embedding_model IS NULL)", name="ck_chunks_embedding_model"),
        Index("ix_chunks_matter_id", "matter_id"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column()
    matter_id: Mapped[uuid.UUID] = mapped_column()
    ordinal: Mapped[int] = mapped_column(Integer)
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    # Which model produced `embedding`; queries filter on it so vectors from
    # different models are never compared with each other.
    embedding_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    tsv: Mapped[Any] = mapped_column(TSVECTOR, Computed("to_tsvector('english', text)", persisted=True))


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    created_at: Mapped[datetime] = _created_at()
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    matter_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("matters.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(Text)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
