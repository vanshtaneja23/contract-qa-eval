from hypothesis import given
from hypothesis import strategies as st

from contract_qa.redaction import DocumentRedactor, unredact

DOC = (
    'SUPPLY AGREEMENT between Acme Widgets Inc. ("Supplier") and Beta Retail, LLC ("Buyer").\n'
    "Acme Widgets shall deliver the Products to Beta Retail, LLC at 1200 Harbor Boulevard.\n"
    "Notices: Attention: Jane Q. Doe, jane.doe@acme-widgets.com, (415) 555-0134.\n"
    "Acme Widgets Inc. may not assign this Agreement.\n"
    "By: /s/ Jane Q. Doe\n"
)
PII = [
    "Acme Widgets",
    "Beta Retail",
    "Jane Q. Doe",
    "jane.doe@acme-widgets.com",
    "(415) 555-0134",
    "1200 Harbor Boulevard",
]


def test_every_kind_is_redacted_and_no_value_leaks() -> None:
    r = DocumentRedactor(DOC)
    red = r.redact(0, len(DOC)).text
    for value in PII:
        assert value not in red, value
    assert set(r.counts()) == {"PARTY", "PERSON", "EMAIL", "PHONE", "ADDRESS"}


def test_placeholders_are_consistent_within_a_document() -> None:
    r = DocumentRedactor(DOC)
    red = r.redact(0, len(DOC)).text
    # Full name, core name and repeat mentions of Acme all map to the same token.
    assert red.startswith('SUPPLY AGREEMENT between PARTY_1 ("Supplier") and PARTY_2 ("Buyer")')
    assert "PARTY_1 shall deliver the Products to PARTY_2 at ADDRESS_1." in red
    assert "PARTY_1 may not assign" in red
    assert red.count("PERSON_1") == 2  # notice line and signature
    assert r.mapping["PARTY_1"] == "Acme Widgets Inc."


def test_role_words_and_ordinary_text_are_kept() -> None:
    red = DocumentRedactor(DOC).redact(0, len(DOC)).text
    assert '("Supplier")' in red and "may not assign this Agreement" in red


def test_redacting_a_chunk_uses_document_level_numbering() -> None:
    r = DocumentRedactor(DOC)
    start = DOC.index("Acme Widgets Inc. may")
    assert r.redact(start, len(DOC)).text.startswith("PARTY_1 may not assign")


def test_map_back_plain_text_is_exact() -> None:
    span = DocumentRedactor(DOC).redact(0, len(DOC))
    i = span.text.index("shall deliver the Products")
    o0, o1 = span.to_original(i, i + len("shall deliver the Products"))
    assert DOC[o0:o1] == "shall deliver the Products"


def test_map_back_through_placeholder_restores_the_real_name() -> None:
    span = DocumentRedactor(DOC).redact(0, len(DOC))
    quote = "PARTY_1 shall deliver the Products to PARTY_2"
    i = span.text.index(quote)
    o0, o1 = span.to_original(i, i + len(quote))
    assert DOC[o0:o1] == "Acme Widgets shall deliver the Products to Beta Retail, LLC"


def test_entity_crossing_chunk_edge_is_still_redacted() -> None:
    r = DocumentRedactor(DOC)
    start = DOC.index("Widgets Inc. (")  # chunk starts mid-name
    red = r.redact(start, start + 40)
    assert "Widgets" not in red.text and red.text.startswith("PARTY_1")
    assert red.to_original(0, len("PARTY_1"))[0] == start


def test_unredact_answer_and_unknown_placeholders() -> None:
    r = DocumentRedactor(DOC)
    assert unredact("PARTY_1 sells to PARTY_2.", r.mapping) == "Acme Widgets Inc. sells to Beta Retail, LLC."
    assert unredact("PARTY_9 is unknown", r.mapping) == "PARTY_9 is unknown"
    assert unredact("PARTY_10", {"PARTY_1": "x", "PARTY_10": "y"}) == "y"


def test_question_mentioning_a_party_is_redacted() -> None:
    r = DocumentRedactor(DOC)
    assert r.redact_question("Can Acme Widgets Inc. assign?") == "Can PARTY_1 assign?"


def test_counts_within_spans_report_types_only() -> None:
    r = DocumentRedactor(DOC)
    assert r.counts([(0, DOC.index("\n"))]) == {"PARTY": 2}


def test_document_without_pii_is_unchanged() -> None:
    text = "The term of this Agreement is five (5) years."
    r = DocumentRedactor(text)
    assert r.redact(0, len(text)).text == text and r.entities == []


@given(data=st.data())
def test_property_ranges_map_back_inside_original(data: st.DataObject) -> None:
    span = DocumentRedactor(DOC).redact(0, len(DOC))
    a = data.draw(st.integers(0, len(span.text) - 1))
    b = data.draw(st.integers(a + 1, len(span.text)))
    o0, o1 = span.to_original(a, b)
    assert 0 <= o0 < o1 <= len(DOC)
    touches_placeholder = any(is_ph for r0, r1, _, _, is_ph in span.segments if r0 < b and a < r1)
    if not touches_placeholder:
        assert DOC[o0:o1] == span.text[a:b]
