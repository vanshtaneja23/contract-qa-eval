import uuid

from contract_qa.retrieval import Hit, or_tsquery, rrf_fuse

DOC = uuid.uuid4()


def hit(cid: int, rank: int) -> Hit:
    return Hit(cid, DOC, 0, 1, "x", 0.0, rank)


def test_or_tsquery_joins_unique_lowercase_terms() -> None:
    assert or_tsquery("Is there a CAP on liability? cap!") == "is | there | a | cap | on | liability"


def test_or_tsquery_strips_tsquery_operators() -> None:
    # Characters like ' & | ! ( ) : * are tsquery syntax; none may pass through.
    q = or_tsquery("law' & !(x) | y:* <-> z")
    assert q == "law | x | y | z"


def test_or_tsquery_empty_when_no_terms() -> None:
    assert or_tsquery("?! ...") is None


def test_rrf_scores_and_order() -> None:
    a = [hit(1, 1), hit(2, 2), hit(3, 3)]
    b = [hit(3, 1), hit(1, 2)]
    fused = rrf_fuse([a, b], k=10, c=60)
    assert [h.chunk_id for h in fused] == [1, 3, 2]
    assert fused[0].score == 1 / 61 + 1 / 62
    assert [h.rank for h in fused] == [1, 2, 3]


def test_rrf_respects_k_and_breaks_ties_by_chunk_id() -> None:
    fused = rrf_fuse([[hit(9, 1)], [hit(4, 1)]], k=1)
    assert [h.chunk_id for h in fused] == [4]


def test_rrf_of_empty_rankings() -> None:
    assert rrf_fuse([[], []]) == []
