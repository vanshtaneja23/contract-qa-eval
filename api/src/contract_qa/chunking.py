"""Split a contract into retrievable chunks that point back into the source.

Invariant (tested): for every chunk, ``source[chunk.start:chunk.end] == chunk.text``.
Citations are later verified against these offsets, so they must be exact.

Strategy:
1. Find section headings at the start of a line ("1. Term", "2.1 Fees",
   "ARTICLE IV", "Section 3"). If there are enough, cut the document at them.
2. Merge tiny sections (a bare heading line) into the next one.
3. Any section still longer than ``max_words`` is split into overlapping word
   windows, so no chunk is too long to embed well.
4. If the document has too few headings to trust (common in CUAD, where PDF
   extraction often flattens headings into running text), use word windows
   over the whole document.

"Words" are runs of non-whitespace. That is a proxy for model tokens: it keeps
offsets trivial (every word is an exact span of the source) at the cost of
chunk sizes being approximate in tokenizer terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

HEADING_RE = re.compile(
    r"""(?mx)
    ^[ \t]*
    (?:
        (?:ARTICLE|Article|SECTION|Section)[ \t]+(?:\d+|[IVXLC]+)\b   # ARTICLE IV, Section 3
      | \d{1,2}\.(?:\d{1,2}\.?)*[ \t]+(?=[A-Z(])                     # 1. Term / 2.1 Fees / 2.1.3 (a)
    )
    """
)
WORD_RE = re.compile(r"\S+")

MIN_HEADINGS = 3


@dataclass(frozen=True, slots=True)
class Chunk:
    start: int
    end: int
    text: str
    kind: Literal["section", "window"]


def find_headings(text: str) -> list[int]:
    """Character offsets where a section heading line begins."""
    return [m.start() for m in HEADING_RE.finditer(text)]


def _word_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text, start, end)]


def _windows(text: str, start: int, end: int, max_words: int, overlap: int) -> list[Chunk]:
    words = _word_spans(text, start, end)
    if not words:
        return []
    step = max_words - overlap
    chunks = []
    for i in range(0, len(words), step):
        window = words[i : i + max_words]
        s, e = window[0][0], window[-1][1]
        chunks.append(Chunk(s, e, text[s:e], "window"))
        if i + max_words >= len(words):
            break
    return chunks


def _trimmed(text: str, start: int, end: int) -> tuple[int, int] | None:
    words = _word_spans(text, start, end)
    if not words:
        return None
    return words[0][0], words[-1][1]


def chunk_document(
    text: str,
    max_words: int = 200,
    overlap_words: int = 40,
    min_section_words: int = 25,
) -> list[Chunk]:
    if not 0 <= overlap_words < max_words:
        raise ValueError("need 0 <= overlap_words < max_words")

    headings = find_headings(text)
    if len(headings) < MIN_HEADINGS:
        return _windows(text, 0, len(text), max_words, overlap_words)

    cuts = sorted({0, *headings, len(text)})
    sections = list(pairwise(cuts))

    merged: list[tuple[int, int]] = []
    for s, e in sections:
        if merged and len(_word_spans(text, *merged[-1])) < min_section_words:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    chunks: list[Chunk] = []
    for s, e in merged:
        span = _trimmed(text, s, e)
        if span is None:
            continue
        s, e = span
        if len(_word_spans(text, s, e)) <= max_words:
            chunks.append(Chunk(s, e, text[s:e], "section"))
        else:
            chunks.extend(_windows(text, s, e, max_words, overlap_words))
    return chunks
