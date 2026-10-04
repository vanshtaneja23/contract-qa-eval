import uuid

import pytest
from hypothesis import given
from hypothesis import strategies as st

from contract_qa.citations import Citation, CitationError, merge_regions, resolve_quote, verify_citation

DOC_A = uuid.uuid4()
DOC_B = uuid.uuid4()
TEXT_A = (
    "1.  Term.  This Agreement  continues for five (5) years.\n"
    "2.  Governing Law.  This Agreement is governed by the laws of the\n"
    "    State of Delaware.  Fees are $1,000.00 per month (net 30).\n"
    "3.  Notices to Zürich GmbH 🙂 must be in writing.\n"
)
TEXT_B = "This Agreement is governed by the laws of England and Wales."
DOCS = {DOC_A: TEXT_A, DOC_B: TEXT_B}


def cite(doc: uuid.UUID, quote: str, text: str = TEXT_A, occurrence: int = 0) -> Citation:
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(quote, start + 1)
    return Citation(document_id=doc, start=start, end=start + len(quote), quoted_text=quote)


# --- verifier -----------------------------------------------------------------


def test_valid_citation_passes() -> None:
    assert verify_citation(cite(DOC_A, "continues for five (5) years."), DOCS) is None


def test_unicode_offsets_are_code_points() -> None:
    assert verify_citation(cite(DOC_A, "Zürich GmbH 🙂 must"), DOCS) is None


def test_paraphrased_quote_is_rejected() -> None:
    c = cite(DOC_A, "continues for five (5) years.")
    paraphrase = c.model_copy(update={"quoted_text": "lasts for five (5) years....."})
    assert verify_citation(paraphrase, DOCS) is CitationError.TEXT_MISMATCH


def test_right_text_wrong_offsets_is_rejected() -> None:
    c = cite(DOC_A, "State of Delaware")
    shifted = c.model_copy(update={"start": c.start + 1, "end": c.end + 1})
    assert verify_citation(shifted, DOCS) is CitationError.TEXT_MISMATCH


def test_quote_with_normalized_whitespace_is_rejected_by_verifier() -> None:
    # The verifier is strict: "Agreement  continues" in the source has two spaces.
    c = cite(DOC_A, "Agreement  continues")
    collapsed = c.model_copy(update={"quoted_text": "Agreement continues", "end": c.end - 1})
    assert verify_citation(collapsed, DOCS) is CitationError.TEXT_MISMATCH


@pytest.mark.parametrize(("start", "end"), [(-1, 5), (5, 5), (10, 3), (0, len(TEXT_A) + 1)])
def test_out_of_range_offsets_are_rejected(start: int, end: int) -> None:
    c = Citation(document_id=DOC_A, start=start, end=end, quoted_text="x")
    assert verify_citation(c, DOCS) is CitationError.BAD_OFFSETS


def test_citation_from_another_document_is_rejected() -> None:
    # The quote really is verbatim in DOC_B, but only DOC_A is in scope.
    c = cite(DOC_B, "governed by the laws of England", text=TEXT_B)
    assert verify_citation(c, {DOC_A: TEXT_A}) is CitationError.DOCUMENT_OUT_OF_SCOPE


def test_offsets_valid_in_one_document_but_claimed_for_another() -> None:
    # Offsets and text taken from DOC_B but labelled as DOC_A.
    c = cite(DOC_B, "This Agreement is governed", text=TEXT_B)
    forged = c.model_copy(update={"document_id": DOC_A})
    assert verify_citation(forged, DOCS) is CitationError.TEXT_MISMATCH


def test_empty_quote_is_rejected() -> None:
    c = Citation(document_id=DOC_A, start=0, end=3, quoted_text="   ")
    assert verify_citation(c, DOCS) is CitationError.EMPTY_QUOTE


def test_verbatim_text_outside_shown_context_is_rejected() -> None:
    c = cite(DOC_A, "must be in writing.")
    context = {DOC_A: [(0, 60)]}  # the model was only shown the first line
    assert verify_citation(c, DOCS, context) is CitationError.NOT_IN_CONTEXT
    assert verify_citation(cite(DOC_A, "five (5) years."), DOCS, context) is None


# --- resolver -----------------------------------------------------------------

WHOLE_A = [(0, len(TEXT_A))]


def test_resolve_exact() -> None:
    start, end, mode = resolve_quote("State of Delaware", TEXT_A, WHOLE_A) or (0, 0, "")
    assert TEXT_A[start:end] == "State of Delaware" and mode == "exact"


def test_resolve_tolerates_collapsed_whitespace_and_returns_source_slice() -> None:
    model_quote = "governed by the laws of the State of Delaware."  # source has a newline + indent
    start, end, mode = resolve_quote(model_quote, TEXT_A, WHOLE_A) or (0, 0, "")
    assert mode == "whitespace"
    assert TEXT_A[start:end] == "governed by the laws of the\n    State of Delaware."
    # The slice the caller stores passes the strict verifier.
    c = Citation(document_id=DOC_A, start=start, end=end, quoted_text=TEXT_A[start:end])
    assert verify_citation(c, DOCS) is None


@pytest.mark.parametrize(
    "quote",
    [
        "governed by the laws of the State of New York.",  # changed word
        "This Agreement lasts for five (5) years.",  # paraphrase
        "",
        "   ",
    ],
)
def test_resolve_rejects_changed_words_and_empty(quote: str) -> None:
    assert resolve_quote(quote, TEXT_A, WHOLE_A) is None


def test_resolve_escapes_regex_characters() -> None:
    hit = resolve_quote("Fees are $1,000.00 per month (net 30).", TEXT_A, WHOLE_A)
    assert hit is not None and hit[2] == "exact"
    hit2 = resolve_quote("Fees are $1,000.00  per month (net 30).", TEXT_A, WHOLE_A)
    assert hit2 is not None and hit2[2] == "whitespace"


def test_resolve_only_searches_given_regions() -> None:
    assert resolve_quote("must be in writing.", TEXT_A, [(0, 60)]) is None


def test_resolve_quote_crossing_overlapping_chunks() -> None:
    # Two overlapping chunks; the quote starts in the first and ends in the second.
    i = TEXT_A.index("Governing Law")
    regions = merge_regions([(0, i + 5), (i - 5, len(TEXT_A))])
    assert resolve_quote("Governing Law.  This Agreement", TEXT_A, regions) is not None


def test_merge_regions() -> None:
    assert merge_regions([(10, 20), (0, 5), (15, 30), (30, 40), (50, 60)]) == [(0, 5), (10, 40), (50, 60)]


@given(data=st.data())
def test_property_resolved_quotes_always_verify(data: st.DataObject) -> None:
    text = data.draw(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=200))
    start = data.draw(st.integers(0, len(text) - 1))
    end = data.draw(st.integers(start + 1, len(text)))
    quote = text[start:end]
    hit = resolve_quote(quote, text, [(0, len(text))])
    if not quote.strip():
        assert hit is None
        return
    assert hit is not None
    s, e, _ = hit
    doc = uuid.uuid4()
    c = Citation(document_id=doc, start=s, end=e, quoted_text=text[s:e])
    assert verify_citation(c, {doc: text}) is None
