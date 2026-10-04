import hashlib
import math
import re
import uuid
from collections.abc import Sequence

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from contract_qa.embedding import embed_chunks
from contract_qa.ingest import ingest_document
from contract_qa.models import EMBEDDING_DIM, Chunk
from contract_qa.retrieval import Retriever, lexical_search, vector_search

pytestmark = pytest.mark.integration
Docs = tuple[uuid.UUID, uuid.UUID]


class FakeEmbedder:
    """Deterministic bag-of-words hashing: texts sharing words get similar vectors."""

    def __init__(self, key: str = "fake") -> None:
        self.key = key

    def _vec(self, s: str) -> list[float]:
        v = [0.0] * EMBEDDING_DIM
        for w in re.findall(r"[a-z]+", s.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % EMBEDDING_DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


CONTRACT_A = (
    "1. Term\nThis agreement lasts five years from the effective date unless terminated.\n\n"
    "2. Governing Law\nThis agreement is governed by the laws of the State of Delaware.\n\n"
    "3. Insurance\nSupplier shall maintain general liability insurance of two million dollars.\n"
)
CONTRACT_B = (
    "1. Payment\nFees are payable within thirty days.\n\n"
    "2. Governing Law\nThis agreement is governed by the laws of England and Wales.\n\n"
    "3. Notices\nNotices must be in writing.\n"
)


@pytest.fixture
def two_docs(session: Session, matter_id: uuid.UUID) -> tuple[uuid.UUID, uuid.UUID]:
    a, _ = ingest_document(session, matter_id=matter_id, title="A", text=CONTRACT_A, source="t")
    b, _ = ingest_document(session, matter_id=matter_id, title="B", text=CONTRACT_B, source="t")
    return a.id, b.id


def _section(session: Session, chunk_id: int) -> str:
    return session.scalar(select(Chunk.text).where(Chunk.id == chunk_id)) or ""


def test_lexical_finds_governing_law_within_scope(session: Session, two_docs: Docs) -> None:
    a, _ = two_docs
    hits = lexical_search(session, "Which law governs this agreement?", [a], k=3)
    assert hits and "Delaware" in hits[0].text
    assert all(h.document_id == a for h in hits)  # never leaks the other contract


def test_lexical_with_operator_characters_does_not_error(session: Session, two_docs: Docs) -> None:
    a, _ = two_docs
    assert isinstance(lexical_search(session, "law' & !(x) | :* <->", [a]), list)


def test_lexical_returns_nothing_for_stopwords_only(session: Session, two_docs: Docs) -> None:
    a, _ = two_docs
    assert lexical_search(session, "is there a", [a]) == []


def test_embed_then_vector_search(session: Session, two_docs: Docs) -> None:
    a, _ = two_docs
    emb = FakeEmbedder()
    n = embed_chunks(session, emb)
    assert n == session.scalar(text("SELECT count(*) FROM chunks"))
    assert embed_chunks(session, emb) == 0  # idempotent

    hits = vector_search(session, emb.embed_query("insurance liability million"), emb.key, [a], k=2)
    assert "insurance" in hits[0].text.lower()
    assert hits[0].score >= hits[-1].score


def test_vector_search_ignores_vectors_from_another_model(session: Session, two_docs: Docs) -> None:
    a, _ = two_docs
    embed_chunks(session, FakeEmbedder("model-x"))
    q = FakeEmbedder().embed_query("insurance")
    assert vector_search(session, q, "model-y", [a]) == []


def test_db_rejects_embedding_without_model_name(session: Session, two_docs: Docs) -> None:
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError, match="ck_chunks_embedding_model"), session.begin_nested():
        session.execute(
            text("UPDATE chunks SET embedding = :v"), {"v": "[" + ",".join(["0"] * EMBEDDING_DIM) + "]"}
        )


def test_hybrid_through_retriever(session: Session, two_docs: Docs) -> None:
    _, b = two_docs
    emb = FakeEmbedder()
    embed_chunks(session, emb)
    r = Retriever(session, emb)
    for method in ("lexical", "vector", "hybrid"):
        hits = r.search("governed by the laws of which state", [b], method, k=3)  # type: ignore[arg-type]
        assert hits, method
        assert "England" in _section(session, hits[0].chunk_id), method
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_hybrid_returns_raw_signals(session: Session, two_docs: Docs) -> None:
    a, _ = two_docs
    emb = FakeEmbedder()
    embed_chunks(session, emb)
    r = Retriever(session, emb)
    hits, signals = r.hybrid("governed by the laws", [a], k=3)
    assert hits and 0 < signals.top_cosine <= 1.0 and signals.top_ts_rank > 0
    _, none = r.hybrid("is there a", [a], k=3)  # stopwords only: no lexical match
    assert none.top_ts_rank == 0.0
