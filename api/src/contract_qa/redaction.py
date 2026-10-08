"""Replace names, emails, phone numbers and addresses with placeholders before any
text is sent to a model, and map the model's output back to the real text.

Placeholders are consistent per document: every occurrence of "Acme Widgets Inc."
(and its core name "Acme Widgets") becomes PARTY_1 in every excerpt, so the model
can still reason about who does what.

Detection is deterministic regex, chosen over an NER model for speed and
auditability. Known limits: person names are only found in signature/notice
patterns (By:, /s/, Name:, Attention:), and organisations only when they carry a
corporate suffix (Inc., LLC, GmbH, ...) or are the core of one that does.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

KINDS = ("PARTY", "PERSON", "EMAIL", "PHONE", "ADDRESS")

_SUFFIX = (
    r"Inc\.?|Incorporated|Corporation|Corp\.?|LLC|L\.L\.C\.|Ltd\.?|Limited|L\.P\.|LP|LLP|PLC|plc"
    r"|GmbH|S\.A\.|N\.V\.|AG|B\.V\.|Co\."
)
ORG_RE = re.compile(rf"\b((?:[A-Z][A-Za-z0-9&'-]*\.?,?\s+){{1,5}})({_SUFFIX})(?![A-Za-z])")
PERSON_RE = re.compile(
    r"(?:/s/|\bBy:|\bName:|\bAttention:|\bAttn\.?:)[ \t]*(?:/s/[ \t]*)?"
    r"([A-Z][A-Za-z'-]+\.?(?:[ \t]+[A-Z][A-Za-z'-]*\.?){1,3})"
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\b\d{3})[\s.-]\d{3}[\s.-]\d{4}\b")
ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][A-Za-z0-9.'-]*\s+){1,5}"
    r"(?:Street|St\.|Avenue|Ave\.?|Road|Rd\.|Boulevard|Blvd\.?|Drive|Dr\.|Lane|Ln\.|Way|Parkway"
    r"|Pkwy\.?|Place|Pl\.|Court|Ct\.|Plaza|Circle|Highway|Hwy\.?)(?![A-Za-z])"
)
PLACEHOLDER_RE = re.compile(r"\b(?:" + "|".join(KINDS) + r")_\d+\b")

# Capitalised words that start many matches but are not names ("This Agreement ...").
_STOP_STARTS = {"The", "This", "That", "Each", "Any", "All", "Such", "Said", "Between", "And", "Of", "By"}


@dataclass(frozen=True, slots=True)
class Entity:
    start: int
    end: int
    kind: str
    placeholder: str


@dataclass(slots=True)
class RedactedSpan:
    """Redacted text for one range of a document, plus a map back to the original.
    Each segment is (red_start, red_end, orig_start, orig_end, is_placeholder)."""

    text: str
    segments: list[tuple[int, int, int, int, bool]] = field(default_factory=list)

    def to_original(self, start: int, end: int) -> tuple[int, int]:
        """Map a [start, end) range of the redacted text to the original document.
        A range touching a placeholder expands to cover the whole original entity."""
        o_start = o_end = None
        for r0, r1, o0, o1, is_ph in self.segments:
            if o_start is None and r0 <= start < r1:
                o_start = o0 if is_ph else o0 + (start - r0)
            if r0 < end <= r1:
                o_end = o1 if is_ph else o0 + (end - r0)
        if o_start is None or o_end is None:
            raise ValueError("range outside redacted text")
        return o_start, o_end


def _clean(surface: str) -> str:
    """Collapse whitespace and drop edge commas. Periods are kept: they belong to
    suffixes like "Inc." and initials like "Q."."""
    return " ".join(surface.split()).strip(" ,")


def _clean_person(name: str) -> str:
    """A sentence-final period after a surname is punctuation, not part of the name."""
    name = _clean(name)
    last = name.split()[-1]
    return name[:-1] if name.endswith(".") and len(last) > 2 else name


def _normal(surface: str) -> str:
    return _clean(surface).casefold()


class DocumentRedactor:
    def __init__(self, text: str) -> None:
        self.text = text
        surfaces: dict[str, str] = {}  # surface -> kind
        canonical: dict[str, str] = {}  # normalized surface -> normalized canonical entity

        for m in ORG_RE.finditer(text):
            words = m.group(1).split()
            while words and words[0].strip(",.") in _STOP_STARTS:
                words.pop(0)
            if not words:
                continue
            core = _clean(" ".join(words))
            full = _clean(" ".join([*words, m.group(2)]))
            surfaces[full] = "PARTY"
            canonical[_normal(full)] = _normal(full)
            if len(core) >= 4:
                surfaces[core] = "PARTY"
                canonical.setdefault(_normal(core), _normal(full))
        for m in PERSON_RE.finditer(text):
            name = _clean_person(m.group(1))
            if name.split()[0] not in _STOP_STARTS:
                surfaces[name] = "PERSON"
        for rx, kind in ((EMAIL_RE, "EMAIL"), (PHONE_RE, "PHONE"), (ADDRESS_RE, "ADDRESS")):
            for m in rx.finditer(text):
                surfaces[_clean(m.group(0)) if kind == "ADDRESS" else m.group(0)] = kind

        # Every occurrence of every detected surface (longest first wins overlaps).
        found: list[tuple[int, int, str, str]] = []
        for surface, kind in surfaces.items():
            pattern = r"(?<![\w@])" + r"\s+".join(map(re.escape, surface.split())) + r"(?![\w@])"
            for m in re.finditer(pattern, text):
                key = canonical.get(_normal(surface), _normal(surface))
                found.append((m.start(), m.end(), kind, key))
        found.sort(key=lambda f: (f[0], -(f[1] - f[0])))

        self.entities: list[Entity] = []
        self.mapping: dict[str, str] = {}  # placeholder -> original text of first occurrence
        numbers: dict[tuple[str, str], str] = {}
        per_kind: Counter[str] = Counter()
        last_end = -1
        for start, end, kind, key in found:
            if start < last_end:
                continue
            if (kind, key) not in numbers:
                per_kind[kind] += 1
                numbers[(kind, key)] = f"{kind}_{per_kind[kind]}"
            ph = numbers[(kind, key)]
            self.mapping.setdefault(ph, text[start:end])
            self.entities.append(Entity(start, end, kind, ph))
            last_end = end
        self._surface_to_placeholder = {
            text[e.start : e.end]: e.placeholder for e in sorted(self.entities, key=lambda e: e.start)
        }

    def redact(self, start: int, end: int) -> RedactedSpan:
        """Redact text[start:end]. An entity crossing the range edge is still replaced."""
        out: list[str] = []
        segments: list[tuple[int, int, int, int, bool]] = []
        pos, r = start, 0
        for e in self.entities:
            if e.end <= start or e.start >= end:
                continue
            e_start, e_end = max(e.start, start), min(e.end, end)
            if e_start > pos:
                plain = self.text[pos:e_start]
                segments.append((r, r + len(plain), pos, e_start, False))
                out.append(plain)
                r += len(plain)
            segments.append((r, r + len(e.placeholder), e_start, e_end, True))
            out.append(e.placeholder)
            r += len(e.placeholder)
            pos = e_end
        if pos < end:
            plain = self.text[pos:end]
            segments.append((r, r + len(plain), pos, end, False))
            out.append(plain)
        return RedactedSpan("".join(out), segments)

    def redact_question(self, question: str) -> str:
        """Replace any of this document's detected entities that appear in a question."""
        for surface in sorted(self._surface_to_placeholder, key=len, reverse=True):
            question = question.replace(surface, self._surface_to_placeholder[surface])
        return question

    def counts(self, spans: list[tuple[int, int]] | None = None) -> dict[str, int]:
        """Entity counts by type (never values), optionally only within spans."""

        def inside(e: Entity) -> bool:
            return spans is None or any(e.start < b and a < e.end for a, b in spans)

        return dict(Counter(e.kind for e in self.entities if inside(e)))


def unredact(text: str, mapping: Mapping[str, str]) -> str:
    """Put real names back into model output. Unknown placeholders are left as is."""
    return PLACEHOLDER_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)
