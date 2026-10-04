import pytest

from contract_eval.bm25 import BM25, tokenize
from contract_eval.cuad import GoldSpan
from contract_eval.retrieval_eval import (
    bootstrap_ci,
    first_hit_rank,
    overlaps,
    paired_diff_ci,
    recall_at,
    reciprocal_rank_at,
    summarize,
)

SPAN = GoldSpan(100, "x" * 50)  # covers [100, 150)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (0, 100, False),  # ends exactly where the span starts: touching, not overlapping
        (150, 200, False),  # starts exactly where the span ends
        (0, 101, True),
        (149, 300, True),
        (110, 120, True),  # inside
        (0, 500, True),  # contains
    ],
)
def test_overlap_boundaries(start: int, end: int, expected: bool) -> None:
    assert overlaps(start, end, SPAN) is expected


def test_first_hit_rank() -> None:
    hits = [(0, 10), (20, 30), (140, 160)]
    assert first_hit_rank(hits, [SPAN]) == 3
    assert first_hit_rank(hits[:2], [SPAN]) is None
    assert first_hit_rank(hits, [SPAN, GoldSpan(25, "ab")]) == 2


def test_recall_and_mrr() -> None:
    ranks = [1, 3, None, 7]
    assert recall_at(ranks, 5) == [1, 1, 0, 0]
    assert reciprocal_rank_at(ranks, 10) == [1, 1 / 3, 0, 1 / 7]
    s = summarize(ranks)
    assert s["recall@5"] == 0.5 and s["recall@1"] == 0.25


def test_bootstrap_ci_is_deterministic_and_brackets_mean() -> None:
    vals = [1.0] * 70 + [0.0] * 30
    lo, hi = bootstrap_ci(vals, seed=1)
    assert (lo, hi) == bootstrap_ci(vals, seed=1)
    assert lo < 0.7 < hi and lo > 0.55 and hi < 0.85


def test_paired_diff_ci() -> None:
    a = [1.0] * 50 + [0.0] * 50
    mean, lo, hi = paired_diff_ci(a, a)
    assert (mean, lo, hi) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        paired_diff_ci([1.0], [1.0, 0.0])


def test_tokenize_stems_and_drops_stopwords() -> None:
    assert tokenize("The Licensee shall indemnify; indemnification") == ["license", "indemnifi", "indemnif"]


def test_bm25_idf_prefers_rare_terms() -> None:
    corpus = {
        1: "agreement agreement agreement",
        2: "agreement indemnify",
        3: "agreement",
        4: "agreement term",
    }
    bm = BM25(corpus)
    ranked = bm.rank("agreement indemnify", corpus.keys())
    assert ranked[0][0] == 2  # the rare word outweighs repeating the common one


def test_bm25_saturates_term_frequency() -> None:
    bm = BM25({1: "fee", 2: "fee " * 2, 3: "fee " * 50, 4: "other words here"})
    s1, s2, s50 = (bm.score({"fee"}, i) for i in (1, 2, 3))
    assert s1 < s2 < s50 < s1 * (bm.k1 + 1)


def test_bm25_rank_limited_to_candidates() -> None:
    bm = BM25({1: "insurance", 2: "insurance policy", 3: "notices"})
    assert [cid for cid, _ in bm.rank("insurance", [1, 3])] == [1]
