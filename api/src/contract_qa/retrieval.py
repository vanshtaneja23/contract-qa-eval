"""Find the chunks most relevant to a question, within a set of documents.

Three methods, compared in eval/results/retrieval.md:
- lexical: Postgres full-text search. Note: ts_rank is NOT BM25. It has no IDF
  (rare words don't count more) and no term-frequency saturation.
- vector:  cosine similarity between pgvector embeddings (exact scan; a single
  document has ~65 chunks, so an ANN index would buy nothing).
- hybrid:  Reciprocal Rank Fusion of lexical and vector rankings.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from contract_qa.embedding import Embedder
from contract_qa.models import Chunk

Method = Literal["lexical", "vector", "hybrid"]
METHODS: tuple[Method, ...] = ("lexical", "vector", "hybrid")


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: int
    document_id: uuid.UUID
    start_char: int
    end_char: int
    text: str
    score: float
    rank: int  # 1-based


_TERM = re.compile(r"[A-Za-z0-9]+")
_COLUMNS = (Chunk.id, Chunk.document_id, Chunk.start_char, Chunk.end_char, Chunk.text)


def or_tsquery(question: str) -> str | None:
    """'Is there a cap on liability?' -> 'is | there | a | cap | on | liability'.

    OR, not AND: plainto_tsquery ANDs every word, so a natural question matches
    almost nothing. Only [A-Za-z0-9] runs survive, so user input can't inject
    tsquery operators. Postgres then drops stopwords and stems the rest.
    """
    terms = dict.fromkeys(t.lower() for t in _TERM.findall(question))
    return " | ".join(terms) or None


def _to_hits(rows: Sequence[Any]) -> list[Hit]:
    return [
        Hit(r.id, r.document_id, r.start_char, r.end_char, r.text, float(r.score), rank)
        for rank, r in enumerate(rows, start=1)
    ]


def lexical_search(
    session: Session, question: str, document_ids: Sequence[uuid.UUID], k: int = 10
) -> list[Hit]:
    q = or_tsquery(question)
    if q is None or not document_ids:
        return []
    tsquery = func.to_tsquery("english", q)
    score = func.ts_rank(Chunk.tsv, tsquery)
    rows = session.execute(
        select(*_COLUMNS, score.label("score"))
        .where(Chunk.document_id.in_(document_ids), Chunk.tsv.op("@@")(tsquery))
        .order_by(score.desc(), Chunk.id)
        .limit(k)
    ).all()
    return _to_hits(rows)


def vector_search(
    session: Session,
    query_vector: Sequence[float],
    model_key: str,
    document_ids: Sequence[uuid.UUID],
    k: int = 10,
) -> list[Hit]:
    if not document_ids:
        return []
    distance = Chunk.embedding.cosine_distance(query_vector)
    rows = session.execute(
        select(*_COLUMNS, (1 - distance).label("score"))
        .where(Chunk.document_id.in_(document_ids), Chunk.embedding_model == model_key)
        .order_by(distance, Chunk.id)
        .limit(k)
    ).all()
    return _to_hits(rows)


def rrf_fuse(rankings: Sequence[Sequence[Hit]], k: int = 10, c: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: score = sum over rankings of 1 / (c + rank).

    Uses ranks only, so lexical and cosine scores never have to be put on a
    common scale. c=60 is the constant from the original RRF paper.
    """
    scores: dict[int, float] = {}
    first_seen: dict[int, Hit] = {}
    for hits in rankings:
        for h in hits:
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (c + h.rank)
            first_seen.setdefault(h.chunk_id, h)
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]
    return [replace(first_seen[cid], score=scores[cid], rank=i) for i, cid in enumerate(ordered, 1)]


class Retriever:
    def __init__(self, session: Session, embedder: Embedder | None = None, candidates: int = 50) -> None:
        self.session = session
        self.embedder = embedder
        self.candidates = candidates  # per-method depth fed into fusion

    def search(
        self, question: str, document_ids: Sequence[uuid.UUID], method: Method, k: int = 10
    ) -> list[Hit]:
        if method == "lexical":
            return lexical_search(self.session, question, document_ids, k)
        if self.embedder is None:
            raise ValueError(f"method {method!r} needs an embedder")
        qvec = self.embedder.embed_query(question)
        if method == "vector":
            return vector_search(self.session, qvec, self.embedder.key, document_ids, k)
        return rrf_fuse(
            [
                lexical_search(self.session, question, document_ids, self.candidates),
                vector_search(self.session, qvec, self.embedder.key, document_ids, self.candidates),
            ],
            k=k,
        )
