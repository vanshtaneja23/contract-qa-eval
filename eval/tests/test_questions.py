from collections import Counter

from contract_eval.cuad import CuadContract, CuadQuestion, GoldSpan
from contract_eval.questions import CATEGORY_QUESTIONS, all_pairs, build_question_set


def _contract(i: int) -> CuadContract:
    qs = []
    for j, cat in enumerate(CATEGORY_QUESTIONS):
        impossible = (i + j) % 3 == 0  # a third of pairs have no such clause
        spans = () if impossible else (GoldSpan(0, "x"),)
        qs.append(CuadQuestion(f"C{i:02d}__{cat}", cat, "q", impossible, spans))
    qs.append(CuadQuestion(f"C{i:02d}__Document Name", "Document Name", "q", False, (GoldSpan(0, "x"),)))
    return CuadContract(f"C{i:02d}", "x" * 10, tuple(qs))


SUBSET = [_contract(i) for i in range(40)]


def test_all_pairs_only_uses_chosen_categories() -> None:
    pairs = all_pairs(SUBSET, answerable=True)
    assert {p.category for p in pairs} <= set(CATEGORY_QUESTIONS)
    assert all(not p.is_impossible for p in pairs)


def test_question_set_sizes_and_impossible_share() -> None:
    qs = build_question_set(SUBSET, n_answerable=90, n_impossible=30, seed=42)
    assert len(qs) == 120
    assert sum(q.is_impossible for q in qs) == 30
    assert len({q.cuad_id for q in qs}) == 120  # no duplicates
    assert [q.id for q in qs] == [f"q{i:03d}" for i in range(1, 121)]


def test_question_set_spreads_across_categories() -> None:
    qs = build_question_set(SUBSET, seed=42)
    counts = Counter(q.category for q in qs if not q.is_impossible)
    assert len(counts) == len(CATEGORY_QUESTIONS)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_question_set_is_deterministic() -> None:
    assert build_question_set(SUBSET, seed=1) == build_question_set(SUBSET, seed=1)
    assert build_question_set(SUBSET, seed=1) != build_question_set(SUBSET, seed=2)
