import uuid

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from contract_qa.ingest import ingest_document
from contract_qa.models import Base

pytestmark = pytest.mark.integration

DOC = (
    "1. Parties\nThis Agreement is between Zürich Widgets GmbH and “Beta” LLC — effective today.\n\n"
    "2. Governing Law\nThis Agreement is governed by the laws of the State of New York 🙂.\n\n"
    "3. Term\nThe term is three (3) years from the Effective Date unless terminated earlier.\n"
) * 3


def test_migration_matches_models(engine: Engine) -> None:
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == [], diff


def test_offsets_round_trip_inside_postgres(session: Session, matter_id: uuid.UUID) -> None:
    doc, created = ingest_document(
        session, matter_id=matter_id, title="t", text=DOC, source="test", max_words=20, overlap_words=5
    )
    assert created
    # Postgres substring() is 1-based and counts characters (not bytes), like
    # Python slicing counts code points, so the same offsets work in both.
    rows = (
        session.execute(
            text(
                "SELECT c.text = substring(d.text FROM c.start_char + 1 FOR c.end_char - c.start_char) "
                "FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.id = :id"
            ),
            {"id": doc.id},
        )
        .scalars()
        .all()
    )
    assert rows and all(rows)


def test_full_text_column_is_populated(session: Session, matter_id: uuid.UUID) -> None:
    ingest_document(session, matter_id=matter_id, title="t", text=DOC, source="test")
    hits = session.execute(
        text("SELECT count(*) FROM chunks WHERE tsv @@ plainto_tsquery('english', 'governed laws')")
    ).scalar_one()
    assert hits >= 1


def test_reingest_is_idempotent(session: Session, matter_id: uuid.UUID) -> None:
    first, created1 = ingest_document(session, matter_id=matter_id, title="t", text=DOC, source="s")
    second, created2 = ingest_document(session, matter_id=matter_id, title="t", text=DOC, source="s")
    assert created1 and not created2 and first.id == second.id
    n = session.execute(text("SELECT count(*) FROM documents WHERE matter_id = :m"), {"m": matter_id})
    assert n.scalar_one() == 1


def test_db_rejects_chunk_whose_matter_differs_from_its_document(
    session: Session, matter_id: uuid.UUID
) -> None:
    doc, _ = ingest_document(session, matter_id=matter_id, title="t", text=DOC, source="s")
    other = session.execute(text("INSERT INTO matters (name) VALUES ('other') RETURNING id")).scalar_one()
    with pytest.raises(IntegrityError, match="fk_chunks_document_matter"), session.begin_nested():
        session.execute(
            text(
                "INSERT INTO chunks (document_id, matter_id, ordinal, start_char, end_char, text) "
                "VALUES (:d, :m, 999, 0, 2, '1.')"
            ),
            {"d": doc.id, "m": other},
        )


def test_db_rejects_text_that_does_not_match_offsets(session: Session, matter_id: uuid.UUID) -> None:
    doc, _ = ingest_document(session, matter_id=matter_id, title="t", text=DOC, source="s")
    with pytest.raises(IntegrityError, match="ck_chunks_text_length"), session.begin_nested():
        session.execute(
            text(
                "INSERT INTO chunks (document_id, matter_id, ordinal, start_char, end_char, text) "
                "VALUES (:d, :m, 999, 0, 10, 'too short')"
            ),
            {"d": doc.id, "m": matter_id},
        )


def test_rejects_nul_characters(session: Session, matter_id: uuid.UUID) -> None:
    with pytest.raises(ValueError, match="NUL"):
        ingest_document(session, matter_id=matter_id, title="t", text="a\x00b", source="s")
