import pytest

from contract_eval.gate_eval import auc, gate_stats, threshold_for_false_abstain


def test_auc_perfect_random_and_inverted() -> None:
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert auc([0.3, 0.7], [0.5]) == 0.5
    with pytest.raises(ValueError):
        auc([], [0.1])


def test_threshold_blocks_at_most_the_allowed_share() -> None:
    pos = [i / 100 for i in range(100)]  # 0.00 .. 0.99
    thr = threshold_for_false_abstain(pos, 0.02)
    assert thr == 0.02
    stats = gate_stats(pos, [0.0, 0.01, 0.5], thr)
    assert stats["answerable_blocked"] == 0.02
    assert stats["absent_caught"] == pytest.approx(2 / 3)


def test_zero_rate_blocks_nothing_answerable() -> None:
    pos = [0.4, 0.6, 0.8]
    thr = threshold_for_false_abstain(pos, 0.0)
    assert gate_stats(pos, [0.1], thr) == {"answerable_blocked": 0.0, "absent_caught": 1.0}
