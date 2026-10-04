"""Store a document and its chunks. Source-agnostic: knows nothing about CUAD."""

from __future__ import annotations

import hashlib
import uuid
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from contract_qa.chunking import chunk_document
from contract_qa.models import Chunk, Document


def ingest_document(
    session: Session,
    *,
    matter_id: uuid.UUID,
    title: str,
    text: str,
    source: str,
    max_words: int = 200,
    overlap_words: int = 40,
    strategy: Literal["section", "window"] = "section",
) -> tuple[Document, bool]:
    """Insert the document and its chunks. Returns (document, created).

    Idempotent per matter: re-ingesting identical text returns the existing row.
    The caller owns the transaction (commit/rollback).
    """
    if "\x00" in text:
        raise ValueError("Postgres text columns cannot store NUL characters")

    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    existing = session.scalar(
        select(Document).where(Document.matter_id == matter_id, Document.sha256 == sha256)
    )
    if existing is not None:
        return existing, False

    doc = Document(
        matter_id=matter_id, title=title, source=source, text=text, sha256=sha256, char_count=len(text)
    )
    session.add(doc)
    session.flush()  # assigns doc.id

    session.add_all(
        Chunk(
            document_id=doc.id,
            matter_id=matter_id,
            ordinal=i,
            start_char=c.start,
            end_char=c.end,
            kind=c.kind,
            text=c.text,
        )
        for i, c in enumerate(
            chunk_document(
                text,
                max_words=max_words,
                overlap_words=overlap_words,
                use_headings=strategy == "section",
            )
        )
    )
    session.flush()
    return doc, True
