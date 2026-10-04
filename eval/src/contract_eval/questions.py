"""Natural-language questions built from CUAD categories.

CUAD's own prompts ("Highlight the parts (if any) of this contract related to
...") are annotation instructions, not things a lawyer would type, so each
category gets a plain question instead. The gold answer is still CUAD's.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from contract_eval.cuad import CuadContract

# 20 of CUAD's 41 categories: ones that read naturally as questions and have
# enough positive examples in the 40-contract subset. Parties and Agreement Date
# are kept on purpose: they are the categories PII redaction should hurt.
CATEGORY_QUESTIONS: dict[str, str] = {
    "Parties": "Who are the parties to this agreement?",
    "Agreement Date": "On what date was this agreement made?",
    "Governing Law": "Which jurisdiction's law governs this agreement?",
    "Expiration Date": "When does this agreement expire or end?",
    "Anti-Assignment": "Can a party assign this agreement, or is the other party's consent required?",
    "Termination For Convenience": "Can a party terminate this agreement for convenience, without cause?",
    "Cap On Liability": "Is there a cap or limit on either party's liability?",
    "Audit Rights": "Does either party have the right to audit the other's books or records?",
    "Post-Termination Services": "What obligations continue after this agreement terminates or expires?",
    "Insurance": "Is either party required to maintain insurance?",
    "Exclusivity": "Does this agreement grant any exclusive rights?",
    "Renewal Term": "Does this agreement renew automatically, and for how long?",
    "Minimum Commitment": "Is there a minimum purchase, volume or spending commitment?",
    "Change Of Control": "What happens if a party undergoes a change of control, merger or acquisition?",
    "Revenue/Profit Sharing": "Does this agreement require sharing revenue or profits, such as royalties?",
    "Non-Compete": "Is there a non-compete restriction on either party?",
    "Notice Period To Terminate Renewal": "How much notice is needed to stop this agreement from renewing?",
    "Ip Ownership Assignment": "Is intellectual property created under this agreement assigned to a party?",
    "License Grant": "What license, if any, does this agreement grant?",
    "Liquidated Damages": "Does this agreement provide for liquidated damages?",
}


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    id: str
    contract_title: str
    category: str
    question: str
    is_impossible: bool
    cuad_id: str


def all_pairs(subset: list[CuadContract], answerable: bool) -> list[EvalQuestion]:
    """Every (contract, category) pair in the chosen categories with the given answerability."""
    out = []
    for c in subset:
        for q in c.questions:
            if q.category in CATEGORY_QUESTIONS and q.is_impossible != answerable:
                out.append(
                    EvalQuestion(
                        id=q.id,
                        contract_title=c.title,
                        category=q.category,
                        question=CATEGORY_QUESTIONS[q.category],
                        is_impossible=q.is_impossible,
                        cuad_id=q.id,
                    )
                )
    return out


def _round_robin(pool: list[EvalQuestion], n: int, rng: random.Random) -> list[EvalQuestion]:
    """Pick n items cycling through categories, so no category dominates."""
    by_cat: dict[str, list[EvalQuestion]] = {}
    for q in sorted(pool, key=lambda q: q.cuad_id):
        by_cat.setdefault(q.category, []).append(q)
    for items in by_cat.values():
        rng.shuffle(items)
    cats = [c for c in CATEGORY_QUESTIONS if c in by_cat]
    picked: list[EvalQuestion] = []
    while len(picked) < n and any(by_cat[c] for c in cats):
        for c in cats:
            if by_cat[c] and len(picked) < n:
                picked.append(by_cat[c].pop())
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} questions available, wanted {n}")
    return picked


def build_question_set(
    subset: list[CuadContract], n_answerable: int = 90, n_impossible: int = 30, seed: int = 42
) -> list[EvalQuestion]:
    rng = random.Random(seed)
    chosen = _round_robin(all_pairs(subset, True), n_answerable, rng) + _round_robin(
        all_pairs(subset, False), n_impossible, rng
    )
    chosen.sort(key=lambda q: q.cuad_id)
    return [
        EvalQuestion(f"q{i:03d}", q.contract_title, q.category, q.question, q.is_impossible, q.cuad_id)
        for i, q in enumerate(chosen, start=1)
    ]


def write_questions(path: Path, questions: list[EvalQuestion]) -> None:
    path.write_text("".join(json.dumps(asdict(q)) + "\n" for q in questions))


def read_questions(path: Path) -> list[EvalQuestion]:
    return [EvalQuestion(**json.loads(line)) for line in path.read_text().splitlines() if line]
