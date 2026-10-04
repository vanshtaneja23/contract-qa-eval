import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from contract_qa.chunking import Chunk, chunk_document

CONTRACT = """EXHIBIT 10.1

                 SUPPLY AGREEMENT

This Supply Agreement is made between Acme Corp. ("Supplier") and Beta LLC.

1. Definitions
"Products" means the widgets described in Exhibit A, as amended from time to time by the parties.

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


def test_windows_round_trip_cover_and_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_document(text, max_words=100, overlap_words=20)
    assert_round_trip(text, chunks)
    assert_covers_all_words(text, chunks)
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]
    assert all(len(c.text.split()) <= 100 for c in chunks)
    assert chunks[-1].text.split()[-1] == "word499"


def test_short_document_is_one_chunk() -> None:
    chunks = chunk_document(CONTRACT)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("EXHIBIT") and chunks[0].text.endswith("conflicts rules.")


def test_no_trailing_duplicate_window() -> None:
    # 180 words, window 100, step 80: windows start at 0 and 80; the second reaches the end.
    text = " ".join(["w"] * 180)
    assert len(chunk_document(text, max_words=100, overlap_words=20)) == 2


def test_handles_unicode_and_crlf_offsets() -> None:
    text = "1. Définitions\r\n“Licensor” means Zürich GmbH — 🙂 party.\r\n" * 20
    chunks = chunk_document(text, max_words=15, overlap_words=3)
    assert len(chunks) > 1
    assert_round_trip(text, chunks)
    assert_covers_all_words(text, chunks)


@pytest.mark.parametrize("text", ["", "   \n\t  "])
def test_empty_and_whitespace_documents(text: str) -> None:
    assert chunk_document(text) == []


@pytest.mark.parametrize(("max_words", "overlap"), [(10, 10), (10, 11), (10, -1)])
def test_rejects_bad_overlap(max_words: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_document("a b c", max_words=max_words, overlap_words=overlap)


# Property test: random documents with headings, prose, emoji and odd whitespace.
_piece = st.one_of(
    st.sampled_from(["\n1. Term\n", "\n2.3 Fees ", "\nARTICLE II\n", "\n\n", "\r\n", "\t", "🙂"]),
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
    chunks = chunk_document(text, max_words=max_words, overlap_words=overlap)
    assert_round_trip(text, chunks)
    assert_covers_all_words(text, chunks)
    assert all(len(c.text.split()) <= max_words for c in chunks)
