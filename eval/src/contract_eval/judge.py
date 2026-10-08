"""LLM-as-judge for answer correctness, and the statistics used to validate it
against human labels.

The judge only grades answers that were actually shown (status "answered") to
questions whose clause exists. Everything else is scored without a judge:
abstaining on an absent clause is correct, answering one is wrong, and
abstaining on (or failing verification for) an answerable question is a miss.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

JUDGE_SYSTEM = """\
You grade answers to questions about contracts. You get the question, the reference clause \
text that expert lawyers marked as answering it, and a system's answer with the quotes it cited.

Grade "correct" if the answer is consistent with the reference and answers the question: the \
key facts (which party, which law, what duration or amount, whether something is allowed or \
restricted) must match the reference. Extra accurate detail is fine; wording need not match.

Grade "incorrect" if the answer contradicts the reference, answers a different question, \
misses the substance of the reference, or says the clause is absent when the reference shows it.

Give the verdict and a one-sentence reason."""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "incorrect"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def judge_prompt(question: str, references: Sequence[str], answer: str, quotes: Sequence[str]) -> str:
    refs = "\n".join(f"- {_clip(r, 600)}" for r in references[:3])
    cited = "\n".join(f"- {_clip(q, 300)}" for q in quotes) or "- (none)"
    return (
        f"Question: {question}\n\nReference clause text:\n{refs}\n\n"
        f"System answer: {_clip(answer, 800)}\n\nQuotes the system cited:\n{cited}"
    )


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Agreement between two raters corrected for chance: (p_o - p_e) / (1 - p_e).
    1 = perfect, 0 = no better than chance. NaN if chance agreement is 1."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return float("nan") if p_e == 1 else (p_o - p_e) / (1 - p_e)


def agreement(human: Sequence[str], judge: Sequence[str]) -> dict[str, Any]:
    pairs = list(zip(human, judge, strict=True))
    return {
        "n": len(pairs),
        "raw_agreement": sum(h == j for h, j in pairs) / len(pairs),
        "cohen_kappa": cohen_kappa(human, judge),
        "confusion": {f"human={h} judge={j}": c for (h, j), c in sorted(Counter(pairs).items())},
    }
