"""M5: judge answers, hand-label a sample, validate the judge, and write the report.

Commands (registered in cli.py):
  judge            grade answered questions with a local judge model
  label            hand-label 30 answers blind to the judge (interactive)
  judge-agreement  human vs judge agreement and Cohen's kappa
  m5-report        eval/results/report.md: comparison table, per category, failures
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from contract_eval.common import CACHE_DIR, RESULTS_DIR, git_commit, load_subset
from contract_eval.cuad import REPO_ROOT, GoldSpan
from contract_eval.judge import JUDGE_SCHEMA, JUDGE_SYSTEM, agreement, judge_prompt
from contract_eval.retrieval_eval import bootstrap_ci, overlaps, paired_diff_ci
from contract_qa.db import make_engine
from contract_qa.llm import CachedClient, make_client
from contract_qa.models import Chunk

ANSWERS_DIR = RESULTS_DIR / "answers"
JUDGMENTS_DIR = RESULTS_DIR / "judgments"
LABELS_DIR = REPO_ROOT / "eval" / "labels"
QUEUE_PATH = LABELS_DIR / "queue.jsonl"
HUMAN_PATH = LABELS_DIR / "human.jsonl"
AGREEMENT_PATH = RESULTS_DIR / "judge_agreement.json"
HARDWARE = (
    "Apple M4 Pro, 24 GB unified memory, Ollama 0.40.1; 4-bit weights "
    "(llama3.1:8b, qwen2.5:7b, qwen2.5:14b: Q4_K_M; gemma2:9b: Q4_0); one model loaded at a time"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def needs_judge(row: dict[str, Any]) -> bool:
    return (not row["question"]["is_impossible"]) and row["result"]["status"] == "answered"


# --- scoring -----------------------------------------------------------------------


def score_row(row: dict[str, Any], verdict: str | None) -> dict[str, bool]:
    """Per-question outcome. `verdict` is the judge's, required when needs_judge(row)."""
    q, r = row["question"], row["result"]
    asserted = r["model_status"] == "answered"  # the model claimed an answer (before verification)
    if q["is_impossible"]:
        return {
            "correct": r["status"] == "not_found",
            "abstained": r["status"] == "not_found",
            "shown_wrong": r["status"] == "answered",
            "model_wrong": asserted,
        }
    if r["status"] == "answered":
        if verdict is None:
            raise ValueError(f"missing judge verdict for {q['id']}")
        wrong = verdict != "correct"
        return {"correct": not wrong, "abstained": False, "shown_wrong": wrong, "model_wrong": wrong}
    return {
        "correct": False,
        "abstained": r["status"] == "not_found",
        "shown_wrong": False,
        "model_wrong": asserted,  # asserted an answer whose citations failed verification
    }


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0.0


def model_metrics(rows: list[dict[str, Any]], verdicts: dict[str, str]) -> dict[str, Any]:
    scores = [score_row(r, verdicts.get(r["question"]["id"])) for r in rows]
    absent = [s for r, s in zip(rows, scores, strict=True) if r["question"]["is_impossible"]]
    present = [s for r, s in zip(rows, scores, strict=True) if not r["question"]["is_impossible"]]
    asserted = [r for r in rows if r["result"]["model_status"] == "answered"]
    acc = [float(s["correct"]) for s in scores]
    latencies = [r["result"]["latency_ms"] for r in rows if r["result"]["model"]]
    in_tok = [r["result"]["input_tokens"] for r in rows if r["result"]["model"]]
    return {
        "n": len(rows),
        "accuracy": statistics.fmean(acc),
        "accuracy_ci": list(bootstrap_ci(acc)),
        "accuracy_answerable": statistics.fmean(s["correct"] for s in present),
        "abstention_absent": statistics.fmean(s["abstained"] for s in absent),
        "over_abstention_answerable": statistics.fmean(s["abstained"] for s in present),
        "citation_validity": (
            sum(r["result"]["status"] == "answered" for r in asserted) / len(asserted) if asserted else None
        ),
        "confidently_wrong_shown": statistics.fmean(s["shown_wrong"] for s in scores),
        "confidently_wrong_model": statistics.fmean(s["model_wrong"] for s in scores),
        "gated": sum(r["result"]["gated"] for r in rows),
        "latency_p50_s": _pct(latencies, 0.5) / 1000,
        "latency_p95_s": _pct(latencies, 0.95) / 1000,
        "mean_input_tokens": statistics.fmean(in_tok) if in_tok else 0.0,
        "cost_per_100_usd": 0.0,
    }


# --- judge -------------------------------------------------------------------------


def cmd_judge(args: argparse.Namespace) -> None:
    gold = {q.id: q.spans for c in load_subset() for q in c.questions}
    client = CachedClient(make_client(args.judge), CACHE_DIR)
    JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for label in args.labels.split(","):
        rows = read_jsonl(ANSWERS_DIR / f"{label}.jsonl")
        out = []
        for row in rows:
            if not needs_judge(row):
                continue
            q, r = row["question"], row["result"]
            prompt = judge_prompt(
                q["question"],
                [s.text for s in gold[q["cuad_id"]]],
                r["answer"],
                [c["quoted_text"] for c in r["citations"]],
            )
            c = client.complete(JUDGE_SYSTEM, prompt, JUDGE_SCHEMA)
            try:
                d = json.loads(c.text)
                verdict, reason = d["verdict"], d["reason"]
            except (ValueError, KeyError):
                verdict, reason = "incorrect", f"unparseable judge output: {c.text[:80]}"
            out.append(
                {"question_id": q["id"], "cuad_id": q["cuad_id"], "verdict": verdict, "reason": reason}
            )
        path = JUDGMENTS_DIR / f"{label}.jsonl"
        path.write_text("".join(json.dumps(o) + "\n" for o in out))
        n_ok = sum(o["verdict"] == "correct" for o in out)
        print(f"[{label}] judged {len(out)} answered questions with {args.judge}: {n_ok} correct")


def verdicts_for(label: str) -> dict[str, str]:
    return {j["question_id"]: j["verdict"] for j in read_jsonl(JUDGMENTS_DIR / f"{label}.jsonl")}


# --- hand labelling ----------------------------------------------------------------


def build_queue(labels: list[str], n: int, seed: int = 0) -> list[dict[str, str]]:
    """n answered items, spread evenly over the models, in a seeded random order."""
    rng = random.Random(seed)
    per = n // len(labels)
    queue: list[dict[str, str]] = []
    for i, label in enumerate(labels):
        rows = [r for r in read_jsonl(ANSWERS_DIR / f"{label}.jsonl") if needs_judge(r)]
        k = per + (1 if i < n - per * len(labels) else 0)
        for r in rng.sample(rows, min(k, len(rows))):
            qid = r["question"]["id"]
            queue.append({"item_id": f"{label}:{qid}", "label": label, "question_id": qid})
    rng.shuffle(queue)  # interleave models so the labeller can't tell them apart by order
    return queue


def cmd_label(args: argparse.Namespace, ask: Callable[[str], str] = input) -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if not QUEUE_PATH.exists():
        queue = build_queue(args.labels.split(","), args.n)
        QUEUE_PATH.write_text("".join(json.dumps(q) + "\n" for q in queue))
    queue = read_jsonl(QUEUE_PATH)
    done = {h["item_id"] for h in read_jsonl(HUMAN_PATH)}
    gold = {q.id: q.spans for c in load_subset() for q in c.questions}
    rows = {
        f"{item['label']}:{r['question']['id']}": r
        for item in queue
        for r in read_jsonl(ANSWERS_DIR / f"{item['label']}.jsonl")
        if r["question"]["id"] == item["question_id"]
    }
    todo = [q for q in queue if q["item_id"] not in done]
    print(f"{len(done)} labelled, {len(todo)} to go. Keys: c = correct, i = incorrect, s = skip, q = quit.")
    print("Model names are hidden. Judge only whether the answer is right according to the reference.\n")
    for item in todo:
        row = rows[item["item_id"]]
        q, r = row["question"], row["result"]
        print("=" * 80)
        print(f"[{len(done) + 1}/{len(queue)}] {q['category']}\nQ: {q['question']}\n")
        print("REFERENCE (CUAD expert annotation):")
        for s in gold[q["cuad_id"]][:3]:
            print("  - " + " ".join(s.text.split())[:700])
        print(f"\nANSWER: {r['answer']}\nCITED:")
        for c in r["citations"]:
            print("  - " + " ".join(c["quoted_text"].split())[:400])
        while (key := ask("\nlabel [c/i/s/q]: ").strip().lower()) not in {"c", "i", "s", "q"}:
            pass
        if key == "q":
            break
        if key == "s":
            continue
        record = {
            "item_id": item["item_id"],
            "human": "correct" if key == "c" else "incorrect",
            "labeled_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        with HUMAN_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
        done.add(item["item_id"])
    print(f"\n{len(done)} of {len(queue)} labelled. Saved to {HUMAN_PATH.relative_to(REPO_ROOT)}")


def cmd_judge_agreement(args: argparse.Namespace) -> None:
    human = read_jsonl(HUMAN_PATH)
    if not human:
        raise SystemExit("no human labels yet: run `uv run cqa-eval label` first")
    pairs = []
    for h in human:
        label, qid = h["item_id"].split(":", 1)
        verdict = verdicts_for(label).get(qid)
        if verdict is not None:
            pairs.append((h, verdict))
    result = agreement([h["human"] for h, _ in pairs], [v for _, v in pairs])
    result["disagreements"] = [h["item_id"] for h, v in pairs if h["human"] != v]
    result["date"] = datetime.date.today().isoformat()
    result["git_commit"] = git_commit()
    AGREEMENT_PATH.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "git_commit"}, indent=1))


# --- report ------------------------------------------------------------------------


def _failure_examples(
    rows: list[dict[str, Any]],
    verdicts: dict[str, str],
    reasons: dict[str, str],
    gold: dict[str, tuple[GoldSpan, ...]],
    chunk_spans: dict[int, tuple[int, int]],
    k: int = 5,
) -> list[str]:
    """Up to k failures, worst kind first: shown-but-wrong, rejected, then missed."""
    shown_wrong, rejected, missed = [], [], []
    for row in rows:
        q, r = row["question"], row["result"]
        s = score_row(row, verdicts.get(q["id"]))
        head = f"**{q['category']}** ({q['id']}): _{q['question']}_"
        if s["shown_wrong"]:
            if q["is_impossible"]:
                why = "the contract has no such clause (CUAD), but the model answered and its quotes verified"
            else:
                why = f"judge: {reasons.get(q['id'], '')}"
            shown_wrong.append(f"- {head}\n  Shown answer: {r['answer']!r}. Confidently wrong: {why}.")
        elif r["status"] == "rejected":
            bad = r["rejection_reasons"]
            rejected.append(
                f"- {head}\n  Rejected by the verifier ({', '.join(bad)}); the answer was withheld. "
                "The model's quote was not verbatim in the excerpts it was shown."
            )
        elif not q["is_impossible"] and r["status"] == "not_found":
            in_ctx = any(
                overlaps(*chunk_spans[cid], g) for cid in r["context_chunk_ids"] for g in gold[q["cuad_id"]]
            )
            where = (
                "the gold clause WAS in the context, so the model was over-cautious"
                if in_ctx
                else "the gold clause was not retrieved, so this is a retrieval miss"
            )
            kind = "gated by the confidence gate" if r["gated"] else "answered not_found"
            missed.append(f"- {head}\n  Missed: {kind}; {where}.")
    return (shown_wrong + rejected + missed)[:k]


def cmd_m5_report(args: argparse.Namespace) -> None:
    labels = args.labels.split(",")
    gold = {q.id: q.spans for c in load_subset() for q in c.questions}
    with Session(make_engine()) as session:
        chunk_spans = {
            cid: (s, e) for cid, s, e in session.execute(select(Chunk.id, Chunk.start_char, Chunk.end_char))
        }
    metrics, per_cat, failures, meta = {}, {}, {}, {}
    per_question: dict[str, dict[str, list[Any]]] = {}
    for label in labels:
        rows = read_jsonl(ANSWERS_DIR / f"{label}.jsonl")
        summary = json.loads((ANSWERS_DIR / f"{label}.summary.json").read_text())
        verdicts = verdicts_for(label)
        reasons = {j["question_id"]: j["reason"] for j in read_jsonl(JUDGMENTS_DIR / f"{label}.jsonl")}
        missing = [
            r["question"]["id"] for r in rows if needs_judge(r) and r["question"]["id"] not in verdicts
        ]
        if missing:
            raise SystemExit(f"{label}: {len(missing)} answered questions not judged; run `cqa-eval judge`")
        metrics[label] = model_metrics(rows, verdicts)
        scored = [score_row(r, verdicts.get(r["question"]["id"])) for r in rows]
        per_question[label] = {
            "ids": [r["question"]["id"] for r in rows],
            "correct": [float(x["correct"]) for x in scored],
            "shown_wrong": [float(x["shown_wrong"]) for x in scored],
        }
        meta[label] = {
            "model": summary["model"],
            "date": summary["date"],
            "git_commit": summary["git_commit"],
        }
        cats: dict[str, list[float]] = {}
        for r in rows:
            cats.setdefault(r["question"]["category"], []).append(
                float(score_row(r, verdicts.get(r["question"]["id"]))["correct"])
            )
        per_cat[label] = {c: statistics.fmean(v) for c, v in cats.items()}
        failures[label] = _failure_examples(rows, verdicts, reasons, gold, chunk_spans)

    paired = []
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            if per_question[a]["ids"] != per_question[b]["ids"]:
                raise SystemExit("runs cover different questions; cannot pair them")
            paired.append(
                {
                    "a": a,
                    "b": b,
                    "accuracy_diff": list(
                        paired_diff_ci(per_question[a]["correct"], per_question[b]["correct"])
                    ),
                    "shown_wrong_diff": list(
                        paired_diff_ci(per_question[a]["shown_wrong"], per_question[b]["shown_wrong"])
                    ),
                }
            )
    agreement_data = json.loads(AGREEMENT_PATH.read_text()) if AGREEMENT_PATH.exists() else None
    report = {
        "date": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "judge": args.judge,
        "hardware": args.hardware,
        "models": meta,
        "metrics": metrics,
        "per_category": per_cat,
        "judge_agreement": agreement_data,
        "paired": paired,
    }
    (RESULTS_DIR / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    (RESULTS_DIR / "report.md").write_text(render_report(report, failures))
    print((RESULTS_DIR / "report.md").read_text().split("## Per category")[0])


def _name(meta: dict[str, Any], label: str) -> str:
    return f"`{meta[label]['model'].removeprefix('ollama:')}`"


def render_report(report: dict[str, Any], failures: dict[str, list[str]]) -> str:
    m, meta = report["metrics"], report["models"]
    labels = list(m)

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.0f}%"

    out = [
        "# Model comparison (M5)",
        "",
        f"Generated {report['date']} at commit {report['git_commit']} by `uv run cqa-eval m5-report`. "
        "All models are open-weight and ran locally via Ollama; **no paid or frontier API models were run.**",
        "",
        f"- Hardware: {report['hardware']}",
        "- Questions: `eval/questions.jsonl`, 120 (90 where the clause exists, 30 where it does not), "
        "same retrieval, same 6-excerpt context, same prompt, PII redacted; only the model differs.",
        f"- Answer correctness: LLM judge `{report['judge']}` (a different, larger model than those graded) "
        "on shown answers to answerable questions; everything else is scored by rule.",
        "",
        "| model | accuracy (95% CI) | abstains on absent clause | citation validity "
        "| confidently wrong (shown) | confidently wrong (before verifier) | p50 / p95 latency "
        "| cost per 100 Qs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in labels:
        x = m[label]
        ci = x["accuracy_ci"]
        out.append(
            f"| {_name(meta, label)} | {pct(x['accuracy'])} [{pct(ci[0])}, {pct(ci[1])}] "
            f"| {pct(x['abstention_absent'])} | {pct(x['citation_validity'])} "
            f"| {pct(x['confidently_wrong_shown'])} | {pct(x['confidently_wrong_model'])} "
            f"| {x['latency_p50_s']:.1f}s / {x['latency_p95_s']:.1f}s | $0 (local) |"
        )
    out += [
        "",
        "Definitions: **accuracy** = correct answers to answerable questions plus correct abstentions on "
        "absent clauses, over all 120. **Citation validity** = share of the model's answers whose every "
        "quote verified (the rest are rejected and withheld). **Confidently wrong (shown)** = a wrong "
        "answer the user would actually see. **Before verifier** also counts answers the verifier rejected.",
        "",
        "| model | accuracy on answerable | over-abstention on answerable | questions skipped by the gate |",
        "|---|---|---|---|",
    ]
    for label in labels:
        x = m[label]
        out.append(
            f"| {_name(meta, label)} | {pct(x['accuracy_answerable'])} "
            f"| {pct(x['over_abstention_answerable'])} | {x['gated']} |"
        )
    out += [
        "",
        "Paired differences (same 120 questions, bootstrap 95% CI). An interval that includes 0 means "
        "the data cannot tell the two models apart.",
        "",
        "| comparison | Δ accuracy | Δ confidently wrong (shown) |",
        "|---|---|---|",
    ]
    for d in report["paired"]:
        acc, cw = d["accuracy_diff"], d["shown_wrong_diff"]
        out.append(
            f"| {_name(meta, d['a'])} minus {_name(meta, d['b'])} "
            f"| {100 * acc[0]:+.1f} pts [{100 * acc[1]:+.1f}, {100 * acc[2]:+.1f}] "
            f"| {100 * cw[0]:+.1f} pts [{100 * cw[1]:+.1f}, {100 * cw[2]:+.1f}] |"
        )
    agr = report["judge_agreement"]
    out += ["", "## Judge validation", ""]
    if agr:
        out.append(
            f"{agr['n']} answers hand-labelled blind to the judge: "
            f"raw agreement {pct(agr['raw_agreement'])}, "
            f"Cohen's kappa {agr['cohen_kappa']:.2f}. Confusion: {agr['confusion']}. "
            f"Disagreements: {', '.join(agr['disagreements']) or 'none'}."
        )
    else:
        out.append(
            "**Pending:** the 30 hand labels (`uv run cqa-eval label`) have not been collected yet, "
            "so the judge is not yet validated. Treat judged accuracy as provisional."
        )
    out += [
        "",
        "## Per category (accuracy)",
        "",
        "| category | " + " | ".join(_name(meta, label) for label in labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    cats = sorted({c for label in labels for c in report["per_category"][label]})
    for c in cats:
        out.append(f"| {c} | " + " | ".join(pct(report["per_category"][lb].get(c)) for lb in labels) + " |")
    out += ["", "## Failure examples (5 per model)", ""]
    for label in labels:
        out += [f"### {_name(meta, label)}", "", *failures[label], ""]
    return "\n".join(out) + "\n"


def add_commands(sub: Any) -> None:
    default_labels = "m5-llama3.1-8b,m5-qwen2.5-7b,m5-gemma2-9b"
    p = sub.add_parser("judge", help="grade answered questions with a local judge model")
    p.add_argument("--labels", default=default_labels)
    p.add_argument("--judge", default="ollama:qwen2.5:14b")
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("label", help="hand-label answers blind to the judge (interactive)")
    p.add_argument("--labels", default=default_labels)
    p.add_argument("--n", type=int, default=30)
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("judge-agreement", help="human labels vs judge: agreement and Cohen's kappa")
    p.set_defaults(func=cmd_judge_agreement)

    p = sub.add_parser("m5-report", help="write eval/results/report.md")
    p.add_argument("--labels", default=default_labels)
    p.add_argument("--judge", default="ollama:qwen2.5:14b")
    p.add_argument(
        "--hardware",
        default=HARDWARE,
    )
    p.set_defaults(func=cmd_m5_report)
