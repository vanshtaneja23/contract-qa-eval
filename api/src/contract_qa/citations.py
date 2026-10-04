"""Citations: turn a model's quote into exact offsets, and verify any citation.

Two separate steps, on purpose:

1. resolve_quote(): the model returns only a chunk id and a quote. Code, not the
   model, finds that quote inside the excerpts the model was shown and computes
   start/end. Models are unreliable at character arithmetic; string search isn't.
2. verify_citation(): an independent check of a finished citation against the
   source text. It runs on every citation before an answer is shown, whatever
   produced it, so a bug in step 1 (or a citation from anywhere else) can't
   slip through.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

MatchMode = Literal["exact", "whitespace"]


class Citation(BaseModel):
    document_id: uuid.UUID
    start: int
    end: int
    quoted_text: str


class CitationError(StrEnum):
    EMPTY_QUOTE = "empty_quote"
    DOCUMENT_OUT_OF_SCOPE = "document_out_of_scope"
    BAD_OFFSETS = "bad_offsets"
    TEXT_MISMATCH = "text_mismatch"
    NOT_IN_CONTEXT = "not_in_context"


def merge_regions(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching [start, end) spans. Overlapping windows
    become one continuous region, so a quote crossing a chunk boundary is found."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def resolve_quote(
    quote: str, doc_text: str, regions: Sequence[tuple[int, int]]
) -> tuple[int, int, MatchMode] | None:
    """Find `quote` inside one of `regions` of `doc_text`.

    First an exact match. Failing that, a match where any run of whitespace in
    the quote may stand for any run of whitespace in the source (models often
    collapse CUAD's double spaces and line breaks). In both cases the caller
    stores the *source* slice, so the citation text is always verbatim; a
    changed word still fails.
    """
    if not quote.strip():
        return None
    for start, end in regions:
        i = doc_text.find(quote, start, end)
        if i != -1:
            return i, i + len(quote), "exact"
    words = quote.split()
    pattern = re.compile(r"\s+".join(re.escape(w) for w in words))
    for start, end in regions:
        m = pattern.search(doc_text, start, end)
        if m:
            return m.start(), m.end(), "whitespace"
    return None


def verify_citation(
    citation: Citation,
    documents: Mapping[uuid.UUID, str],
    context_spans: Mapping[uuid.UUID, Sequence[tuple[int, int]]] | None = None,
) -> CitationError | None:
    """None if the citation is valid, else the first reason it is not.

    `documents` holds only the documents the question was asked over, so a
    citation to any other document (another contract, another client's matter)
    is out of scope. `context_spans`, if given, are the regions the model was
    shown; a citation outside them can't be grounded in what it read.
    """
    if not citation.quoted_text.strip():
        return CitationError.EMPTY_QUOTE
    text = documents.get(citation.document_id)
    if text is None:
        return CitationError.DOCUMENT_OUT_OF_SCOPE
    if not 0 <= citation.start < citation.end <= len(text):
        return CitationError.BAD_OFFSETS
    if text[citation.start : citation.end] != citation.quoted_text:
        return CitationError.TEXT_MISMATCH
    if context_spans is not None:
        regions = merge_regions(context_spans.get(citation.document_id, []))
        if not any(s <= citation.start and citation.end <= e for s, e in regions):
            return CitationError.NOT_IN_CONTEXT
    return None
