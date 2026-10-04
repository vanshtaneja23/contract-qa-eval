"""Calibrating the confidence gate: does a raw retrieval score separate questions
whose clause exists from questions whose clause doesn't?"""

from __future__ import annotations

import statistics
from collections.abc import Sequence


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """ROC AUC = P(random positive scores higher than random negative); ties count half.
    0.5 means the score carries no information, 1.0 means perfect separation."""
    if not positives or not negatives:
        raise ValueError("need at least one positive and one negative")
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def threshold_for_false_abstain(positives: Sequence[float], max_rate: float) -> float:
    """Highest threshold that blocks (score < threshold) at most `max_rate` of
    answerable questions."""
    ordered = sorted(positives)
    return ordered[int(max_rate * len(ordered))]


def gate_stats(positives: Sequence[float], negatives: Sequence[float], threshold: float) -> dict[str, float]:
    return {
        "answerable_blocked": statistics.fmean(p < threshold for p in positives),
        "absent_caught": statistics.fmean(n < threshold for n in negatives),
    }
