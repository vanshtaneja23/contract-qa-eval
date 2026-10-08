import math

import pytest

from contract_eval.judge import agreement, cohen_kappa, judge_prompt


def test_kappa_perfect_chance_and_opposite() -> None:
    assert cohen_kappa(["c", "i", "c", "i"], ["c", "i", "c", "i"]) == 1.0
    assert cohen_kappa(["c", "c", "i", "i"], ["c", "i", "c", "i"]) == 0.0
    assert cohen_kappa(["c", "i"], ["i", "c"]) == -1.0
    assert math.isnan(cohen_kappa(["c", "c"], ["c", "c"]))  # no variation: undefined
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_kappa_known_value() -> None:
    # 10 items: 8 agree; human 6c/4i, judge 6c/4i -> p_o=.8, p_e=.52, kappa=.5833
    human = ["c"] * 5 + ["i"] * 3 + ["c", "i"]
    judge = ["c"] * 5 + ["i"] * 3 + ["i", "c"]
    assert cohen_kappa(human, judge) == pytest.approx((0.8 - 0.52) / 0.48)


def test_agreement_report() -> None:
    r = agreement(["correct", "incorrect", "correct"], ["correct", "correct", "correct"])
    assert r["n"] == 3 and r["raw_agreement"] == pytest.approx(2 / 3)
    assert r["confusion"] == {"human=correct judge=correct": 2, "human=incorrect judge=correct": 1}


def test_judge_prompt_clips_and_lists_quotes() -> None:
    p = judge_prompt("Which law governs?", ["x" * 2000], "Delaware law.", [])
    assert "Question: Which law governs?" in p and "- (none)" in p
    assert len(p) < 1000
