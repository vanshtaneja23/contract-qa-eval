"""Split a contract into retrievable chunks that point back into the source.

Invariant (tested): for every chunk, ``source[chunk.start:chunk.end] == chunk.text``.
Citations are later verified against these offsets, so they must be exact.

Chunks are overlapping windows of ``max_words`` words. A section-aware
chunker (cut at "1. Term", "ARTICLE IV", ...) was built and measured in M2: it
did not improve retrieval (window - section, hybrid recall@5: +0.011, 95% CI
[-0.019, +0.040]), so it was removed. See DECISIONS.md.

"Words" are runs of non-whitespace. That is a proxy for model tokens: it keeps
offsets trivial (every word is an exact span of the source) at the cost of
chunk sizes being approximate in tokenizer terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class Chunk:
    start: int
    end: int
    text: str


def chunk_document(text: str, max_words: int = 200, overlap_words: int = 40) -> list[Chunk]:
    if not 0 <= overlap_words < max_words:
        raise ValueError("need 0 <= overlap_words < max_words")
    words = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
    step = max_words - overlap_words
    chunks = []
    for i in range(0, len(words), step):
        window = words[i : i + max_words]
        s, e = window[0][0], window[-1][1]
        chunks.append(Chunk(s, e, text[s:e]))
        if i + max_words >= len(words):
            break
    return chunks
