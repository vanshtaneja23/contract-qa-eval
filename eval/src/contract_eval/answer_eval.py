"""M3 eval commands: context experiment, gate calibration, and the cited-answer run.

Only `answer-eval` calls a paid model, and it always plans first: it counts
cached responses, estimates the cost of the rest, and refuses to run above
--max-cost (default $0: the project runs at zero API spend).
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import shlex
import statistics
import sys
from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from contract_eval.common import CACHE_DIR, QUESTIONS_PATH, RESULTS_DIR, git_commit, load_subset
from contract_eval.gate_eval import auc, gate_stats, threshold_for_false_abstain
from contract_eval.questions import EvalQuestion, all_pairs, read_questions
from contract_eval.retrieval_eval import bootstrap_ci, overlaps, paired_diff_ci
from contract_qa.answering import (
    ANSWER_SCHEMA,
    SYSTEM_PROMPT,
    AnswerConfig,
    GateConfig,
    answer_question,
    build_context,
    build_prompt,
)
from contract_qa.db import make_engine
from contract_qa.llm import CachedClient, Usage, cost_usd, estimate_tokens, make_client
from contract_qa.models import Document
from contract_qa.retrieval import Retriever

GATE_RESULTS = RESULTS_DIR / "gate.json"


def _meta(argv: list[str] | None) -> dict[str, Any]:
    return {
        "date": datetime.date.today().isoformat(),
        "command": "uv run cqa-eval " + shlex.join(sys.argv[1:] if argv is None else argv),
        "git_commit": git_commit(),
    }


def _retriever(session: Session) -> Retriever:
    from contract_qa.embedding import SentenceTransformerEmbedder

    return Retriever(session, SentenceTransformerEmbedder("bge-small"))


def _documents_by_title(session: Session) -> dict[str, Document]:
    return {d.title: d for d in session.scalars(select(Document))}


# --- context experiment ---------------------------------------------------------

CONTEXT_CONFIGS: list[tuple[str, int, bool]] = [
    ("top-5", 5, False),
    ("top-6", 6, False),
    ("top-6, first chunk reserved", 6, True),
]


def cmd_context_eval(args: argparse.Namespace) -> None:
    """Context recall: is any gold span overlapped by a chunk the model would see?"""
    subset = load_subset()
    spans = {q.id: q.spans for c in subset for q in c.questions}
    cases = all_pairs(subset, answerable=True)
    hits: dict[str, list[float]] = {label: [] for label, _, _ in CONTEXT_CONFIGS}
    with Session(make_engine()) as session:
        retriever = _retriever(session)
        docs = _documents_by_title(session)
        for case in cases:
            doc = docs[case.contract_title]
            ranked, _ = retriever.hybrid(case.question, [doc.id], k=20)
            ids = [h.chunk_id for h in ranked]
            for label, k, first in CONTEXT_CONFIGS:
                ctx = build_context(session, ids, {doc.id: doc}, k, first)
                found = any(overlaps(c.start, c.end, s) for c in ctx for s in spans[case.cuad_id])
                hits[label].append(1.0 if found else 0.0)

    base_label, six_label, first_label = (label for label, _, _ in CONTEXT_CONFIGS)
    by_category: dict[str, dict[str, float]] = {}
    for cat in sorted({c.category for c in cases}):
        idx = [i for i, c in enumerate(cases) if c.category == cat]
        by_category[cat] = {"n": len(idx)} | {
            label: statistics.fmean(hits[label][i] for i in idx) for label, _, _ in CONTEXT_CONFIGS
        }
    result: dict[str, Any] = _meta(args.argv) | {
        "n": len(cases),
        "configs": {
            label: {"context_recall": statistics.fmean(v), "ci": list(bootstrap_ci(v))}
            for label, v in hits.items()
        },
        "diffs": {
            f"{six_label} - {base_label}": list(paired_diff_ci(hits[six_label], hits[base_label])),
            f"{first_label} - {six_label}": list(paired_diff_ci(hits[first_label], hits[six_label])),
        },
        "by_category": by_category,
    }
    (RESULTS_DIR / "context.json").write_text(json.dumps(result, indent=1) + "\n")
    for label, v in result["configs"].items():
        print(f"{label:30s} context recall {v['context_recall']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
    for name, (mean, lo, hi) in result["diffs"].items():
        print(f"{name:45s} {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for cat in ("Parties", "Agreement Date"):
        row = by_category[cat]
        print(f"  {cat}: " + "  ".join(f"{label}={row[label]:.2f}" for label, _, _ in CONTEXT_CONFIGS))


# --- gate calibration -------------------------------------------------------------


def cmd_gate_calibrate(args: argparse.Namespace) -> None:
    """Choose the gate signal and threshold on a dev split; report on the held-out eval set."""
    subset = load_subset()
    test = read_questions(QUESTIONS_PATH)
    test_ids = {q.cuad_id for q in test}
    dev = [p for p in all_pairs(subset, True) + all_pairs(subset, False) if p.cuad_id not in test_ids]

    def signals(questions: list[EvalQuestion]) -> list[tuple[bool, dict[str, float]]]:
        out = []
        for q in questions:
            _, s = retriever.hybrid(q.question, [docs[q.contract_title].id], k=5)
            out.append((not q.is_impossible, {"top_cosine": s.top_cosine, "top_ts_rank": s.top_ts_rank}))
        return out

    with Session(make_engine()) as session:
        retriever = _retriever(session)
        docs = _documents_by_title(session)
        dev_sig, test_sig = signals(dev), signals(test)

    report: dict[str, Any] = {}
    for name in ("top_cosine", "top_ts_rank"):
        dpos = [s[name] for ans, s in dev_sig if ans]
        dneg = [s[name] for ans, s in dev_sig if not ans]
        tpos = [s[name] for ans, s in test_sig if ans]
        tneg = [s[name] for ans, s in test_sig if not ans]
        thr = threshold_for_false_abstain(dpos, args.max_false_abstain)
        report[name] = {
            "auc_dev": auc(dpos, dneg),
            "auc_test": auc(tpos, tneg),
            "threshold": thr,
            "dev": gate_stats(dpos, dneg, thr),
            "test": gate_stats(tpos, tneg, thr),
        }
    chosen = max(report, key=lambda n: report[n]["auc_dev"])
    n_dev_pos = sum(ans for ans, _ in dev_sig)
    result: dict[str, Any] = _meta(args.argv) | {
        "max_false_abstain": args.max_false_abstain,
        "dev": {"n_answerable": n_dev_pos, "n_absent": len(dev_sig) - n_dev_pos},
        "test": {
            "n_answerable": sum(ans for ans, _ in test_sig),
            "n_absent": sum(not a for a, _ in test_sig),
        },
        "signals": report,
        "chosen": {"signal": chosen, "threshold": report[chosen]["threshold"]},
    }
    GATE_RESULTS.write_text(json.dumps(result, indent=1) + "\n")
    print(f"dev: {result['dev']}  test: {result['test']}")
    for name, r in report.items():
        print(f"{name:12s} AUC dev={r['auc_dev']:.3f} test={r['auc_test']:.3f} thr={r['threshold']:.4f}")
        for split in ("dev", "test"):
            st = r[split]
            print(
                f"    {split}: blocks {st['answerable_blocked']:.1%} of answerable, "
                f"catches {st['absent_caught']:.1%} of absent"
            )
    print(f"chosen: {result['chosen']}")


def load_gate() -> GateConfig:
    chosen = json.loads(GATE_RESULTS.read_text())["chosen"]
    return GateConfig(chosen["signal"], float(chosen["threshold"]))


# --- cited-answer run -------------------------------------------------------------


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0.0


def cmd_answer_eval(args: argparse.Namespace) -> None:
    questions = read_questions(QUESTIONS_PATH)
    random.Random(0).shuffle(questions)  # so --limit takes a mix of contracts and categories
    questions = questions[: args.limit] if args.limit else questions
    config = AnswerConfig(
        k=args.k, include_first_chunk=not args.no_first_chunk, gate=None if args.no_gate else load_gate()
    )
    client = CachedClient(make_client(args.model), CACHE_DIR, offline=args.offline)
    label = args.label or args.model.replace(":", "_")

    with Session(make_engine()) as session:
        retriever = _retriever(session)
        docs = _documents_by_title(session)

        # Plan: same retrieval, gate and prompt as answer_question, so cache keys match.
        gated = cached = 0
        est = 0.0
        for q in questions:
            doc = docs[q.contract_title]
            ranked, signals = retriever.hybrid(q.question, [doc.id], k=max(config.k, 20))
            if config.gate and config.gate.blocks(signals):
                gated += 1
                continue
            ctx = build_context(
                session, [h.chunk_id for h in ranked], {doc.id: doc}, config.k, config.include_first_chunk
            )
            prompt = build_prompt(q.question, ctx)
            if client.is_cached(SYSTEM_PROMPT, prompt, ANSWER_SCHEMA):
                cached += 1
                continue
            tokens_in = estimate_tokens(SYSTEM_PROMPT + prompt + json.dumps(ANSWER_SCHEMA))
            est += cost_usd(args.model, Usage(tokens_in, args.est_output_tokens))
        to_call = len(questions) - gated - cached
        print(
            f"[{label}] {len(questions)} questions: {gated} gated (no call), {cached} cached, "
            f"{to_call} to call. Estimated cost of new calls: ${est:.3f} "
            f"(assumes {args.est_output_tokens} output tokens each)"
        )
        if args.dry_run:
            return
        if est > args.max_cost:
            raise SystemExit(f"estimate ${est:.2f} exceeds --max-cost ${args.max_cost:.2f}; not running")

        out_dir = RESULTS_DIR / "answers"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        spent = 0.0
        for i, q in enumerate(questions, start=1):
            doc = docs[q.contract_title]
            result = answer_question(session, q.question, [doc.id], client, retriever, config)
            session.commit()
            if result.model and not result.cached:
                spent += cost_usd(args.model, Usage(result.input_tokens, result.output_tokens))
            rows.append(
                {
                    "question": q.__dict__ | {"document_id": str(doc.id)},
                    "result": result.model_dump(mode="json"),
                }
            )
            if i % 10 == 0 or i == len(questions):
                print(f"  {i}/{len(questions)} done, spent ${spent:.4f}")

    (out_dir / f"{label}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = (
        _meta(args.argv)
        | {
            "label": label,
            "model": args.model,
            "config": {
                "k": config.k,
                "include_first_chunk": config.include_first_chunk,
                "gate": None
                if config.gate is None
                else {"signal": config.gate.signal, "threshold": config.gate.threshold},
            },
        }
        | summarize_answers(rows, args.model)
        | {"spent_usd_this_run": spent}
    )
    (out_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("command",)}, indent=1))


def summarize_answers(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Smoke-level metrics (answer correctness vs CUAD comes in M5)."""
    results = [r["result"] for r in rows]
    answerable = [r["result"] for r in rows if not r["question"]["is_impossible"]]
    absent = [r["result"] for r in rows if r["question"]["is_impossible"]]
    attempted = [r for r in results if r["model_status"] == "answered"]
    reasons = Counter(reason.split(":")[0] for r in results for reason in r["rejection_reasons"])
    latencies = [r["latency_ms"] for r in results if r["model"] and not r["cached"]]
    tokens = Usage(sum(r["input_tokens"] for r in results), sum(r["output_tokens"] for r in results))
    return {
        "n": len(results),
        "status": dict(Counter(r["status"] for r in results)),
        "answerable": {"n": len(answerable), **Counter(r["status"] for r in answerable)},
        "absent": {"n": len(absent), **Counter(r["status"] for r in absent)},
        "gated": sum(r["gated"] for r in results),
        "citation_validity": (
            sum(r["status"] == "answered" for r in attempted) / len(attempted) if attempted else None
        ),
        "rejection_reasons": dict(reasons),
        "whitespace_matches": sum(r["whitespace_matches"] for r in results),
        "citations": sum(len(r["citations"]) for r in results),
        "tokens": {"input": tokens.input_tokens, "output": tokens.output_tokens},
        "cost_usd_all_calls": cost_usd(model, tokens),
        "latency_ms": {"p50": _pct(latencies, 0.5), "p95": _pct(latencies, 0.95), "n": len(latencies)},
    }


def add_commands(sub: Any) -> None:
    p = sub.add_parser("context-eval", help="context recall: top-5 vs top-6 vs first chunk reserved")
    p.set_defaults(func=cmd_context_eval)

    p = sub.add_parser("gate-calibrate", help="pick the confidence-gate signal/threshold on a dev split")
    p.add_argument(
        "--max-false-abstain",
        type=float,
        default=0.02,
        help="max share of answerable dev questions the gate may block",
    )
    p.set_defaults(func=cmd_gate_calibrate)

    p = sub.add_parser("answer-eval", help="cited answers on eval/questions.jsonl (paid model calls)")
    p.add_argument("--model", required=True, help="e.g. ollama:qwen2.5:7b (local, free)")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions (seeded shuffle)")
    p.add_argument("--dry-run", action="store_true", help="plan and estimate cost; call nothing")
    p.add_argument("--offline", action="store_true", help="use only cached responses; never call the API")
    p.add_argument("--max-cost", type=float, default=0.0, help="refuse to run above this estimate (USD)")
    p.add_argument(
        "--est-output-tokens", type=int, default=1000, help="per-call output estimate for --dry-run"
    )
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--no-first-chunk", action="store_true")
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--label")
    p.set_defaults(func=cmd_answer_eval)
