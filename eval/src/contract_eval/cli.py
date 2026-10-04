"""cqa-eval: command line for data ingestion (and, later, benchmarks)."""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
import time
from collections import Counter
from typing import cast

from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from contract_eval.cuad import (
    REPO_ROOT,
    SUBSET_PATH,
    CuadContract,
    apply_manifest,
    build_manifest,
    download_cuad,
    load_cuad,
    select_subset,
)
from contract_eval.questions import all_pairs, build_question_set, read_questions, write_questions
from contract_qa.db import make_engine
from contract_qa.ingest import ingest_document
from contract_qa.models import Chunk, Document, Matter

QUESTIONS_PATH = REPO_ROOT / "eval" / "questions.jsonl"
RETRIEVAL_RESULTS = REPO_ROOT / "eval" / "results" / "retrieval.json"


def _get_or_create_matter(session: Session, name: str) -> Matter:
    matter = session.scalar(select(Matter).where(Matter.name == name))
    if matter is None:
        matter = Matter(name=name)
        session.add(matter)
        session.flush()
    return matter


def _load_subset(n: int, seed: int) -> list[CuadContract]:
    contracts = load_cuad(download_cuad())
    if SUBSET_PATH.exists():
        manifest = json.loads(SUBSET_PATH.read_text())
        if manifest["n"] != n or manifest["seed"] != seed:
            raise SystemExit(
                f"{SUBSET_PATH} pins n={manifest['n']} seed={manifest['seed']}; delete it to resample"
            )
        return apply_manifest(contracts, manifest)
    subset = select_subset(contracts, n, seed)
    SUBSET_PATH.write_text(json.dumps(build_manifest(subset, seed), indent=2) + "\n")
    print(f"wrote {SUBSET_PATH.relative_to(SUBSET_PATH.parents[1])}")
    return subset


def _gold_span_stats(subset: list[CuadContract], chunks_by_title: dict[str, list[Chunk]]) -> tuple[int, int]:
    """How many gold spans sit entirely inside at least one chunk."""
    inside = total = 0
    for c in subset:
        chunks = chunks_by_title[c.title]
        for q in c.questions:
            for s in q.spans:
                total += 1
                inside += any(ch.start_char <= s.start and s.end <= ch.end_char for ch in chunks)
    return inside, total


def cmd_ingest(args: argparse.Namespace) -> None:
    subset = _load_subset(args.n, args.seed)
    engine = make_engine()
    created = 0
    chunks_by_title: dict[str, list[Chunk]] = {}
    with Session(engine, expire_on_commit=False) as session, session.begin():
        matters = [_get_or_create_matter(session, f"CUAD matter {i + 1}") for i in range(args.matters)]
        for i, contract in enumerate(subset):
            doc, was_created = ingest_document(
                session,
                matter_id=matters[i % len(matters)].id,
                title=contract.title,
                text=contract.text,
                source=f"cuad-v1:{contract.title}",
                strategy=args.strategy,
            )
            created += was_created
            chunks_by_title[contract.title] = list(
                session.scalars(select(Chunk).where(Chunk.document_id == doc.id).order_by(Chunk.ordinal))
            )

    all_chunks = [ch for chs in chunks_by_title.values() for ch in chs]
    words = sorted(len(ch.text.split()) for ch in all_chunks)
    kinds = Counter(ch.kind for ch in all_chunks)
    window_only = sum(all(ch.kind == "window" for ch in chs) for chs in chunks_by_title.values())
    inside, total = _gold_span_stats(subset, chunks_by_title)
    print(f"documents: {len(subset)} ({created} newly inserted) across {len(matters)} matters")
    print(f"chunks: {len(all_chunks)}  section={kinds['section']}  window={kinds['window']}")
    print(f"window-only documents (few headings or long sections): {window_only}/{len(subset)}")
    print(
        f"words/chunk: median={statistics.median(words):.0f}  p95={words[int(0.95 * (len(words) - 1))]}"
        f"  max={words[-1]}"
    )
    print(f"gold spans fully inside one chunk: {inside}/{total} ({inside / total:.1%})")


def cmd_embed(args: argparse.Namespace) -> None:
    from contract_qa.embedding import SentenceTransformerEmbedder, embed_chunks

    embedder = SentenceTransformerEmbedder(args.model)
    with Session(make_engine()) as session, session.begin():
        texts = list(session.scalars(select(Chunk.text)))
        t0 = time.perf_counter()
        n = embed_chunks(session, embedder)
        elapsed = time.perf_counter() - t0
    tokens = sorted(embedder.count_tokens(texts))
    limit = embedder.max_seq_length
    over = sum(t > limit for t in tokens)
    print(f"embedded {n} chunks with {args.model} in {elapsed:.1f}s")
    print(
        f"tokens/chunk: median={statistics.median(tokens):.0f} p95={tokens[int(0.95 * (len(tokens) - 1))]}"
        f"  model limit={limit}  truncated chunks={over}/{len(tokens)} ({over / len(tokens):.1%})"
    )


def cmd_build_questions(args: argparse.Namespace) -> None:
    if QUESTIONS_PATH.exists() and not args.force:
        raise SystemExit(f"{QUESTIONS_PATH} exists; pass --force to rebuild")
    questions = build_question_set(
        _load_subset(40, 42), n_answerable=args.answerable, n_impossible=args.impossible, seed=args.seed
    )
    write_questions(QUESTIONS_PATH, questions)
    impossible = sum(q.is_impossible for q in questions)
    print(
        f"wrote {len(questions)} questions ({impossible} with no such clause, "
        f"{impossible / len(questions):.0%}) to eval/questions.jsonl"
    )
    for cat, n in Counter(q.category for q in questions).most_common():
        print(f"  {n:3d}  {cat}")


def cmd_retrieval_eval(args: argparse.Namespace) -> None:
    from contract_eval.bm25 import BM25
    from contract_eval.retrieval_eval import first_hit_rank, paired_diff_ci, recall_at, reciprocal_rank_at
    from contract_eval.retrieval_eval import summarize as summarize_ranks
    from contract_qa.retrieval import Method, Retriever

    methods: list[str] = args.methods.split(",")
    unknown = set(methods) - {"lexical", "bm25", "vector", "hybrid"}
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}")
    subset = _load_subset(40, 42)
    spans_by_cuad_id = {q.id: q.spans for c in subset for q in c.questions}
    if args.set == "all":
        cases = all_pairs(subset, answerable=True)
    else:
        cases = [q for q in read_questions(QUESTIONS_PATH) if not q.is_impossible]

    engine = make_engine()
    with Session(engine) as session:
        doc_ids = dict(session.execute(select(Document.title, Document.id)).tuples().all())
        chunk_rows = session.execute(
            select(Chunk.id, Chunk.document_id, Chunk.start_char, Chunk.end_char, Chunk.text, Chunk.kind)
        ).all()
        models_in_db = set(session.scalars(select(Chunk.embedding_model).distinct()))

        embedder = None
        if {"vector", "hybrid"} & set(methods):
            from contract_qa.embedding import SentenceTransformerEmbedder

            if models_in_db != {args.embedding}:
                raise SystemExit(
                    f"chunks are embedded with {models_in_db}; run `cqa-eval embed --model "
                    f"{args.embedding}` first"
                )
            embedder = SentenceTransformerEmbedder(args.embedding)
        retriever = Retriever(session, embedder)

        bm25 = BM25({r.id: r.text for r in chunk_rows}) if "bm25" in methods else None
        offsets = {r.id: (r.start_char, r.end_char) for r in chunk_rows}
        chunks_of: dict[object, list[int]] = {}
        for r in chunk_rows:
            chunks_of.setdefault(r.document_id, []).append(r.id)

        ranks: dict[str, list[int | None]] = {m: [] for m in methods}
        for case in cases:
            doc_id = doc_ids[case.contract_title]
            spans = spans_by_cuad_id[case.cuad_id]
            for m in methods:
                if m == "bm25":
                    assert bm25 is not None
                    hits = [offsets[cid] for cid, _ in bm25.rank(case.question, chunks_of[doc_id], k=10)]
                else:
                    found = retriever.search(case.question, [doc_id], cast(Method, m), k=10)
                    hits = [(h.start_char, h.end_char) for h in found]
                ranks[m].append(first_hit_rank(hits, spans))

    baseline = methods[0]
    run = {
        "date": datetime.date.today().isoformat(),
        "command": "uv run cqa-eval " + " ".join(sys.argv[1:] if args.argv is None else args.argv),
        "database": make_url(str(engine.url)).database,
        "chunks": dict(Counter(r.kind for r in chunk_rows)),
        "embedding_model": args.embedding if embedder else None,
        "question_set": args.set,
        "n": len(cases),
        "methods": {m: summarize_ranks(ranks[m]) for m in methods},
        "vs_baseline": {
            m: {
                "baseline": baseline,
                "recall@5_diff": paired_diff_ci(recall_at(ranks[m], 5), recall_at(ranks[baseline], 5)),
                "mrr@10_diff": paired_diff_ci(
                    reciprocal_rank_at(ranks[m], 10), reciprocal_rank_at(ranks[baseline], 10)
                ),
            }
            for m in methods[1:]
        },
        "categories": [c.category for c in cases],
        "ranks": ranks,
    }
    results = json.loads(RETRIEVAL_RESULTS.read_text()) if RETRIEVAL_RESULTS.exists() else {}
    results[args.label] = run
    RETRIEVAL_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_RESULTS.write_text(json.dumps(results, indent=1) + "\n")
    print(f"[{args.label}] n={len(cases)} chunks={run['chunks']} embedding={run['embedding_model']}")
    for m in methods:
        s = run["methods"][m]
        print(
            f"  {m:8s} recall@5={s['recall@5']:.3f} {_ci(s['recall@5_ci'])}  "
            f"mrr@10={s['mrr@10']:.3f} {_ci(s['mrr@10_ci'])}"
        )


def _ci(ci: list[float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def cmd_retrieval_report(args: argparse.Namespace) -> None:
    from contract_eval.retrieval_eval import render_report

    path = RETRIEVAL_RESULTS.with_suffix(".md")
    path.write_text(render_report(json.loads(RETRIEVAL_RESULTS.read_text())))
    print(f"wrote {path.relative_to(path.parents[2])}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cqa-eval")
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("ingest", help="download CUAD, pin a subset, ingest it into Postgres")
    p.add_argument("--n", type=int, default=40, help="number of contracts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--matters", type=int, default=4, help="spread contracts over this many matters")
    p.add_argument("--strategy", choices=["section", "window"], default="section")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("embed", help="embed all chunks with a local model")
    p.add_argument("--model", choices=["minilm", "bge-small"], default="bge-small")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("build-questions", help="write the eval question set (eval/questions.jsonl)")
    p.add_argument("--answerable", type=int, default=90)
    p.add_argument("--impossible", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_build_questions)

    p = sub.add_parser("retrieval-eval", help="recall@k / MRR of retrieval methods vs CUAD gold spans")
    p.add_argument("--label", required=True)
    p.add_argument("--methods", default="lexical,bm25,vector,hybrid", help="first one is the baseline")
    p.add_argument("--embedding", choices=["minilm", "bge-small"], default="bge-small")
    p.add_argument(
        "--set",
        choices=["all", "questions"],
        default="all",
        help="all: every answerable pair in the 20 categories; questions: answerable part of questions.jsonl",
    )
    p.set_defaults(func=cmd_retrieval_eval)

    p = sub.add_parser("retrieval-report", help="render eval/results/retrieval.md")
    p.set_defaults(func=cmd_retrieval_report)

    args = parser.parse_args(argv)
    args.argv = argv
    args.func(args)


if __name__ == "__main__":
    main()
