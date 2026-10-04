"""cqa-eval: command line for data ingestion (and, later, benchmarks)."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from contract_eval.cuad import (
    SUBSET_PATH,
    CuadContract,
    apply_manifest,
    build_manifest,
    download_cuad,
    load_cuad,
    select_subset,
)
from contract_qa.db import make_engine
from contract_qa.ingest import ingest_document
from contract_qa.models import Chunk, Matter


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cqa-eval")
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("ingest", help="download CUAD, pin a subset, ingest it into Postgres")
    p.add_argument("--n", type=int, default=40, help="number of contracts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--matters", type=int, default=4, help="spread contracts over this many matters")
    p.set_defaults(func=cmd_ingest)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
