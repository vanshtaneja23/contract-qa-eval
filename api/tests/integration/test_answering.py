import uuid
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from contract_qa.answering import AnswerConfig, GateConfig, answer_question
from contract_qa.embedding import embed_chunks
from contract_qa.ingest import ingest_document
from contract_qa.models import AuditLog
from contract_qa.retrieval import Retriever
from fakes import FakeEmbedder, ScriptedClient

pytestmark = pytest.mark.integration

CONTRACT = (
    "SUPPLY AGREEMENT between Acme Widgets Inc. and Beta Retail LLC.\n\n"
    "1. Term. This Agreement continues for five (5) years from the Effective Date.\n\n"
    "2. Governing Law. This Agreement is governed by the laws of the\n"
    "   State of Delaware, without regard to its conflict of laws rules.\n\n"
    "3. Insurance. Supplier shall maintain general liability insurance of $2,000,000.\n"
)
OTHER = "OTHER AGREEMENT. This Agreement is governed by the laws of England and Wales.\n"


@pytest.fixture
def setup(session: Session, matter_id: uuid.UUID) -> tuple[uuid.UUID, uuid.UUID, Retriever]:
    doc, _ = ingest_document(session, matter_id=matter_id, title="Supply", text=CONTRACT, source="t")
    other, _ = ingest_document(session, matter_id=matter_id, title="Other", text=OTHER, source="t")
    emb = FakeEmbedder()
    embed_chunks(session, emb)
    return doc.id, other.id, Retriever(session, emb)


def ask(session: Session, setup: Any, response: Any, **cfg: Any) -> Any:
    doc, _, retriever = setup
    client = ScriptedClient(response)
    result = answer_question(
        session,
        "Which law governs?",
        [doc],
        client,
        retriever,
        AnswerConfig(**cfg) if cfg else AnswerConfig(),
    )
    return result, client


def answer(quote: str, chunk: str = "C1") -> dict[str, Any]:
    return {
        "status": "answered",
        "answer": "Delaware law.",
        "citations": [{"chunk_id": chunk, "quote": quote}],
    }


def last_audit(session: Session) -> AuditLog:
    row = session.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(1)).one()
    return row


def test_valid_answer_gets_exact_verified_offsets(session: Session, setup: Any) -> None:
    result, client = ask(session, setup, answer("governed by the laws of the\n   State of Delaware"))
    assert result.status == "answered" and result.answer == "Delaware law."
    (c,) = result.citations
    assert CONTRACT[c.start : c.end] == c.quoted_text == "governed by the laws of the\n   State of Delaware"
    assert result.whitespace_matches == 0
    assert '<excerpt id="C1" document="Document 1">' in client.prompts[0]  # title redacted
    assert last_audit(session).detail["status"] == "answered"


def test_collapsed_whitespace_is_resolved_to_the_verbatim_source(session: Session, setup: Any) -> None:
    result, _ = ask(session, setup, answer("governed by the laws of the State of Delaware"))
    assert result.status == "answered" and result.whitespace_matches == 1
    assert "\n" in result.citations[0].quoted_text  # stored text is the source, not the model's


def test_paraphrased_quote_rejects_and_withholds_answer(session: Session, setup: Any) -> None:
    result, _ = ask(session, setup, answer("This contract is subject to Delaware law"))
    assert result.status == "rejected" and result.answer is None and result.citations == []
    assert result.rejection_reasons == ["quote_not_found:C1"]
    audit = last_audit(session)
    assert audit.action == "ask" and audit.detail["status"] == "rejected"
    assert audit.detail["rejection_reasons"] == ["quote_not_found:C1"]


def test_one_bad_citation_rejects_the_whole_answer(session: Session, setup: Any) -> None:
    resp = answer("governed by the laws of the")
    resp["citations"].append({"chunk_id": "C1", "quote": "governed by the laws of New York"})
    result, _ = ask(session, setup, resp)
    assert result.status == "rejected" and result.answer is None


def test_unknown_chunk_label_is_rejected(session: Session, setup: Any) -> None:
    result, _ = ask(session, setup, answer("governed by the laws of England", chunk="C9"))
    assert result.status == "rejected" and result.rejection_reasons == ["unknown_chunk:C9"]


def test_quote_from_a_document_not_in_scope_is_rejected(session: Session, setup: Any) -> None:
    # The other contract's text is verbatim in the DB, but it was never in scope or shown.
    result, _ = ask(session, setup, answer("governed by the laws of England and Wales"))
    assert result.status == "rejected"


def test_answer_without_citations_is_rejected(session: Session, setup: Any) -> None:
    result, _ = ask(session, setup, {"status": "answered", "answer": "Delaware.", "citations": []})
    assert result.status == "rejected" and result.rejection_reasons == ["no_citations"]


def test_model_not_found_passes_through(session: Session, setup: Any) -> None:
    result, _ = ask(session, setup, {"status": "not_found", "answer": "No such clause.", "citations": []})
    assert result.status == "not_found" and result.model_status == "not_found" and not result.gated


@pytest.mark.parametrize("raw", ["not json", '{"status": "maybe", "answer": "x", "citations": []}'])
def test_malformed_output_is_rejected(session: Session, setup: Any, raw: str) -> None:
    result, _ = ask(session, setup, raw)
    assert result.status == "rejected" and result.rejection_reasons == ["invalid_json"]


def test_gate_skips_the_model_when_retrieval_is_weak(session: Session, setup: Any) -> None:
    result, client = ask(session, setup, answer("x"), gate=GateConfig("top_cosine", 1.01))
    assert result.status == "not_found" and result.gated
    assert client.prompts == []  # no model call, no cost
    assert last_audit(session).detail["gated"] is True


def test_first_chunk_is_included_when_configured(session: Session, setup: Any) -> None:
    doc, _, retriever = setup
    client = ScriptedClient({"status": "not_found", "answer": "n/a", "citations": []})
    answer_question(
        session, "insurance amount", [doc], client, retriever, AnswerConfig(k=1, include_first_chunk=True)
    )
    assert "SUPPLY AGREEMENT between PARTY_1" in client.prompts[0]


def test_unknown_document_id_raises(session: Session, setup: Any) -> None:
    _, _, retriever = setup
    with pytest.raises(ValueError):
        answer_question(session, "q", [uuid.uuid4()], ScriptedClient("{}"), retriever)


# --- redaction end to end ---------------------------------------------------------


def test_prompt_contains_no_pii_and_audit_logs_only_counts(session: Session, setup: Any) -> None:
    result, client = ask(session, setup, answer("governed by the laws of the\n   State of Delaware"))
    prompt = client.prompts[0]
    for value in ("Acme Widgets", "Beta Retail", "Supply"):
        assert value not in prompt
    assert result.redactions == {"PARTY": 2}
    audit = last_audit(session)
    assert audit.detail["redactions"] == {"PARTY": 2}
    assert "Acme" not in str(audit.detail["redactions"])


def test_quote_with_placeholders_maps_back_to_real_text(session: Session, setup: Any) -> None:
    resp = {
        "status": "answered",
        "answer": "The parties are PARTY_1 and PARTY_2.",
        "citations": [{"chunk_id": "C1", "quote": "SUPPLY AGREEMENT between PARTY_1 and PARTY_2."}],
    }
    result, _ = ask(session, setup, resp)
    assert result.status == "answered"
    assert result.answer == "The parties are Acme Widgets Inc. and Beta Retail LLC."
    (c,) = result.citations
    assert c.quoted_text == "SUPPLY AGREEMENT between Acme Widgets Inc. and Beta Retail LLC."
    assert CONTRACT[c.start : c.end] == c.quoted_text  # verified against the original


def test_no_redact_sends_original_text(session: Session, setup: Any) -> None:
    result, client = ask(session, setup, answer("State of Delaware"), redact=False)
    assert "Acme Widgets Inc." in client.prompts[0] and 'document="Supply"' in client.prompts[0]
    assert result.status == "answered" and result.redactions == {}
