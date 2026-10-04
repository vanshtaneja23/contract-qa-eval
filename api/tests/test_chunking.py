from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from contract_qa.chunking import Chunk, chunk_document, find_headings

CONTRACT = """EXHIBIT 10.1

                 SUPPLY AGREEMENT

This Supply Agreement is made between Acme Corp. ("Supplier") and Beta LLC.

1. Definitions

"Products" means the widgets described in Exhibit A, as amended from time to time by the parties.

2. Term

2.1 Initial Term. This Agreement starts on the Effective Date and continues for five (5) years.

2.2 Renewal. It renews automatically for one (1) year periods unless either party gives notice.

ARTICLE IV
GOVERNING LAW

This Agreement is governed by the laws of the State of Delaware, without regard to conflicts rules.
"""


def assert_round_trip(source: str, chunks: list[Chunk]) -> None:
    for c in chunks:
        assert source[c.start : c.end] == c.text
        assert c.text == c.text.strip() and c.text


def assert_covers_all_words(source: str, chunks: list[Chunk]) -> None:
    covered = [False] * len(source)
    for c in chunks:
        for i in range(c.start, c.end):
            covered[i] = True
    missing = [i for i, ch in enumerate(source) if not ch.isspace() and not covered[i]]
    assert not missing, f"uncovered chars at {missing[:5]}"


def test_finds_numbered_article_and_section_headings() -> None:
    starts = find_headings(CONTRACT)
    lines = [CONTRACT[s:].lstrip().split("\n", 1)[0] for s in starts]
    assert lines == [
        "1. Definitions",
        "2. Term",
        "2.1 Initial Term. This Agreement starts on the Effective Date and continues for five (5) years.",
        "2.2 Renewal. It renews automatically for one (1) year periods unless either party gives notice.",
        "ARTICLE IV",
    ]


def test_does_not_treat_numbers_in_prose_as_headings() -> None:
    text = "pursuant to Section\n7 hereof, payable within\n10 Business Days and\n1.5 million units"
    assert find_headings(text) == []


def test_section_chunks_round_trip_and_cover_document() -> None:
    chunks = chunk_document(CONTRACT, min_section_words=5)
    assert {c.kind for c in chunks} == {"section"}
    assert_round_trip(CONTRACT, chunks)
    assert_covers_all_words(CONTRACT, chunks)
    assert any(c.text.startswith("ARTICLE IV") and "Delaware" in c.text for c in chunks)


def test_section_chunks_are_ordered_and_disjoint() -> None:
    chunks = chunk_document(CONTRACT, min_section_words=5)
    for a, b in pairwise(chunks):
        assert a.end <= b.start


def test_tiny_heading_only_sections_merge_forward() -> None:
    chunks = chunk_document(CONTRACT, min_section_words=5)
    assert not any(c.text in {"1. Definitions", "2. Term"} for c in chunks)


def test_falls_back_to_windows_without_headings() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_document(text, max_words=100, overlap_words=20)
    assert {c.kind for c in chunks} == {"window"}
    assert_round_trip(text, chunks)
    assert_covers_all_words(text, chunks)
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]  # overlap
    assert all(len(c.text.split()) <= 100 for c in chunks)


def test_long_section_is_windowed() -> None:
    long_body = " ".join(["indemnify"] * 450)
    text = f"1. Scope\n{'x ' * 30}\n2. Indemnity\n{long_body}\n3. Notices\n{'y ' * 30}"
    chunks = chunk_document(text, max_words=200, overlap_words=40)
    assert {c.kind for c in chunks} == {"section", "window"}
    assert all(len(c.text.split()) <= 200 for c in chunks)
    assert_round_trip(text, chunks)
    assert_covers_all_words(text, chunks)


def test_handles_unicode_and_crlf_offsets() -> None:
    text = "1. Définitions\r\n“Licensor” means Zürich GmbH — 🙂 party.\r\n" * 4
    chunks = chunk_document(text, min_section_words=1)
    assert_round_trip(text, chunks)
    assert_covers_all_words(text, chunks)


@pytest.mark.parametrize("text", ["", "   \n\t  "])
def test_empty_and_whitespace_documents(text: str) -> None:
    assert chunk_document(text) == []


def test_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_document("a b c", max_words=10, overlap_words=10)


# Property test: random documents built from headings, prose and odd whitespace.
_piece = st.one_of(
    st.sampled_from(["\n1. Term\n", "\n2.3 Fees ", "\nARTICLE II\n", "\nSection 9 ", "\n\n", "\r\n", "\t"]),
    st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=40),
)


@settings(max_examples=300, deadline=None)
@given(
    pieces=st.lists(_piece, max_size=60),
    max_words=st.integers(min_value=2, max_value=40),
    data=st.data(),
)
def test_property_offsets_round_trip(pieces: list[str], max_words: int, data: st.DataObject) -> None:
    text = "".join(pieces)
    overlap = data.draw(st.integers(min_value=0, max_value=max_words - 1))
    chunks = chunk_document(text, max_words=max_words, overlap_words=overlap, min_section_words=3)
    assert_round_trip(text, chunks)
    assert_covers_all_words(text, chunks)
    assert all(len(c.text.split()) <= max_words for c in chunks if c.kind == "window")
