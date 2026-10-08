import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from contract_eval import m5
from contract_eval.cuad import CuadContract, CuadQuestion, GoldSpan


def row(
    qid: str, impossible: bool, status: str, model_status: str | None, gated: bool = False
) -> dict[str, Any]:
    return {
        "question": {
            "id": qid,
            "cuad_id": f"C__{qid}",
            "category": "Governing Law",
            "question": "Which law?",
            "is_impossible": impossible,
        },
        "result": {
            "status": status,
            "model_status": model_status,
            "gated": gated,
            "answer": "Delaware",
            "citations": [{"quoted_text": "laws of Delaware", "start": 0, "end": 16}],
            "rejection_reasons": [],
            "context_chunk_ids": [],
            "latency_ms": 1000.0,
            "input_tokens": 2000,
            "model": None if gated else "ollama:x",
        },
    }


def test_scoring_rules() -> None:
    # absent clause: abstaining is correct; answering is confidently wrong
    assert m5.score_row(row("a", True, "not_found", "not_found"), None)["correct"]
    s = m5.score_row(row("b", True, "answered", "answered"), None)
    assert not s["correct"] and s["shown_wrong"] and s["model_wrong"]
    # absent clause, model answered but the verifier rejected it: not shown, but the model was wrong
    s = m5.score_row(row("c", True, "rejected", "answered"), None)
    assert not s["correct"] and not s["shown_wrong"] and s["model_wrong"]
    # answerable: the judge decides; a wrong shown answer is confidently wrong
    assert m5.score_row(row("d", False, "answered", "answered"), "correct")["correct"]
    assert m5.score_row(row("e", False, "answered", "answered"), "incorrect")["shown_wrong"]
    # answerable but abstained: a miss, not confidently wrong
    s = m5.score_row(row("f", False, "not_found", "not_found"), None)
    assert not s["correct"] and s["abstained"] and not s["shown_wrong"] and not s["model_wrong"]
    with pytest.raises(ValueError):
        m5.score_row(row("g", False, "answered", "answered"), None)


def test_model_metrics() -> None:
    rows = [
        row("1", False, "answered", "answered"),
        row("2", False, "rejected", "answered"),
        row("3", False, "not_found", "not_found"),
        row("4", True, "not_found", "not_found", gated=True),
    ]
    x = m5.model_metrics(rows, {"1": "correct"})
    assert x["accuracy"] == 0.5  # q1 and q4
    assert x["citation_validity"] == 0.5  # 1 of 2 asserted answers verified
    assert x["abstention_absent"] == 1.0 and x["over_abstention_answerable"] == pytest.approx(1 / 3)
    assert x["confidently_wrong_shown"] == 0.0 and x["confidently_wrong_model"] == 0.25
    assert x["gated"] == 1 and x["latency_p50_s"] == 1.0 and x["cost_per_100_usd"] == 0.0


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    answers = tmp_path / "answers"
    answers.mkdir()
    for label in ("m1", "m2", "m3"):
        rows = [row(f"q{i:03d}", False, "answered", "answered") for i in range(12)]
        (answers / f"{label}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(m5, "ANSWERS_DIR", answers)
    monkeypatch.setattr(m5, "LABELS_DIR", tmp_path / "labels")
    monkeypatch.setattr(m5, "QUEUE_PATH", tmp_path / "labels" / "queue.jsonl")
    monkeypatch.setattr(m5, "HUMAN_PATH", tmp_path / "labels" / "human.jsonl")
    monkeypatch.setattr(m5, "REPO_ROOT", tmp_path)
    questions = tuple(
        CuadQuestion(f"C__q{i:03d}", "Governing Law", "q", False, (GoldSpan(0, "x"),)) for i in range(12)
    )
    monkeypatch.setattr(m5, "load_subset", lambda: [CuadContract("C", "x", questions)])
    return tmp_path


def test_queue_is_balanced_across_models(results: Path) -> None:
    queue = m5.build_queue(["m1", "m2", "m3"], 30)
    assert len(queue) == 30
    counts = {lb: sum(q["label"] == lb for q in queue) for lb in ("m1", "m2", "m3")}
    assert counts == {"m1": 10, "m2": 10, "m3": 10}
    assert len({q["item_id"] for q in queue}) == 30


def test_label_cli_saves_skips_quits_and_resumes(results: Path) -> None:
    args = argparse.Namespace(labels="m1,m2,m3", n=6)
    keys = iter(["c", "x", "i", "s", "q"])  # "x" is invalid and asked again
    m5.cmd_label(args, ask=lambda _: next(keys))
    saved = m5.read_jsonl(m5.HUMAN_PATH)
    assert [h["human"] for h in saved] == ["correct", "incorrect"]
    # Resuming continues where it stopped; labelled items are not shown again.
    keys2 = iter(["c"] * 10)
    m5.cmd_label(args, ask=lambda _: next(keys2))
    saved = m5.read_jsonl(m5.HUMAN_PATH)
    assert len(saved) == 6 and len({h["item_id"] for h in saved}) == 6
