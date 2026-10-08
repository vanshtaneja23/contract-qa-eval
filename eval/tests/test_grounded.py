from contract_eval.answer_eval import grounded_correct
from contract_eval.cuad import GoldSpan

GOLD = {"x__Governing Law": (GoldSpan(100, "a" * 20),), "x__Cap On Liability": ()}


def row(cuad_id: str, impossible: bool, status: str, cites: list[tuple[int, int]]) -> dict:  # type: ignore[type-arg]
    return {
        "question": {"cuad_id": cuad_id, "is_impossible": impossible},
        "result": {"status": status, "citations": [{"start": s, "end": e} for s, e in cites]},
    }


def test_answered_with_overlapping_citation_is_correct() -> None:
    assert grounded_correct(row("x__Governing Law", False, "answered", [(110, 130)]), GOLD)


def test_answered_citing_elsewhere_or_abstaining_is_wrong() -> None:
    assert not grounded_correct(row("x__Governing Law", False, "answered", [(0, 50)]), GOLD)
    assert not grounded_correct(row("x__Governing Law", False, "not_found", []), GOLD)
    assert not grounded_correct(row("x__Governing Law", False, "rejected", []), GOLD)


def test_absent_clause_is_correct_only_when_abstaining() -> None:
    assert grounded_correct(row("x__Cap On Liability", True, "not_found", []), GOLD)
    assert not grounded_correct(row("x__Cap On Liability", True, "answered", [(0, 5)]), GOLD)
