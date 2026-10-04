"""Local sentence-transformers embeddings: no API cost, and contract text never
leaves the machine to be embedded.

sentence-transformers (and torch) live in the `embed` dependency group, so CI and
unit tests don't download them; tests use a fake Embedder instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from contract_qa.models import EMBEDDING_DIM, Chunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@dataclass(frozen=True, slots=True)
class ModelSpec:
    hf_name: str
    query_prefix: str = ""


MODELS: dict[str, ModelSpec] = {
    "minilm": ModelSpec("sentence-transformers/all-MiniLM-L6-v2"),
    # BGE was trained with this instruction prepended to queries (not passages).
    "bge-small": ModelSpec(
        "BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "
    ),
}


class Embedder(Protocol):
    key: str

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, key: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.key = key
        self.spec = MODELS[key]
        self.model: SentenceTransformer = SentenceTransformer(self.spec.hf_name, device=device)
        dim = self.model.get_sentence_embedding_dimension()
        if dim != EMBEDDING_DIM:
            raise ValueError(f"{key} produces {dim}-d vectors; the schema expects {EMBEDDING_DIM}")

    @property
    def max_seq_length(self) -> int:
        limit = self.model.max_seq_length
        if limit is None:
            raise ValueError(f"{self.key} reports no max_seq_length")
        return int(limit)

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Tokens per text before truncation, so we can see how much gets cut off."""
        encoded: Any = self.model.tokenizer(list(texts), add_special_tokens=True)
        return [len(ids) for ids in encoded["input_ids"]]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self.model.encode(list(texts), batch_size=32, normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        (vec,) = self.model.encode([self.spec.query_prefix + text], normalize_embeddings=True)
        return [float(x) for x in vec]


def embed_chunks(session: Session, embedder: Embedder, batch_size: int = 256) -> int:
    """Embed every chunk that has no embedding from this model yet. Returns how many.

    Re-running with a different model overwrites the vectors, and the
    embedding_model column records which model they came from.
    """
    total = 0
    stale = or_(Chunk.embedding_model.is_(None), Chunk.embedding_model != embedder.key)
    while True:
        rows = session.execute(
            select(Chunk.id, Chunk.text).where(stale).order_by(Chunk.id).limit(batch_size)
        ).all()
        if not rows:
            return total
        vectors = embedder.embed_passages([r.text for r in rows])
        session.execute(
            update(Chunk),
            [
                {"id": r.id, "embedding": v, "embedding_model": embedder.key}
                for r, v in zip(rows, vectors, strict=True)
            ],
        )
        total += len(rows)
