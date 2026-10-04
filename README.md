# contract-qa-eval

[![CI](https://github.com/vanshtaneja23/contract-qa-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/vanshtaneja23/contract-qa-eval/actions/workflows/ci.yml)

Ask a question about a contract → get an answer with citations to the exact clause text →
and know which frontier model does this most reliably.

> Work in progress, built milestone by milestone. This README is completed in Milestone 7.
> Every number here comes from a real run, listed with the command and the date.

## Status

- [x] **M1** Repo, schema, ingestion with offset-exact chunking
- [x] **M2** Retrieval (BM25 / vector / hybrid) and retrieval eval
- [ ] M3 Cited answers and citation verifier
- [ ] M4 Row-level security, PII redaction, audit log
- [ ] M5 Multi-model evaluation
- [ ] M6 API and TypeScript frontend
- [ ] M7 CI and docs

## Layout

```
api/     Python 3.12 package `contract_qa`: chunking, schema, ingestion (FastAPI arrives in M6)
eval/    Python package `contract_eval`: CUAD loading, `cqa-eval` CLI, benchmarks
web/     Next.js 16 + TypeScript (strict)
scripts/ pre-commit secret scanner
```

## Quickstart

Requires Docker, [uv](https://docs.astral.sh/uv/) and Node 20+.

```bash
cp .env.example .env
make setup      # deps + installs the pre-commit secret scanner
make ingest     # starts Postgres, runs migrations, downloads CUAD, ingests 40 contracts
uv run cqa-eval embed --model bge-small   # local embeddings, ~11 s
make check      # lint, types, all tests, web typecheck + lint
```

## Ingestion (M1)

`uv run cqa-eval ingest` (40 contracts, seed 42, 4 matters), run 2026-10-03:

| metric | value |
|---|---|
| contracts / matters | 40 / 4 |
| chunks (200-word windows, 40-word overlap) | 2,163 |
| CUAD gold spans that fit entirely in one chunk | 946 / 1,034 (91.5%) |
| chunks whose stored text ≠ `substring(document, start, end)` in SQL | 0 / 2,163 |

## Retrieval (M2)

375 answerable questions (20 CUAD categories × 40 contracts), retrieval scoped to the contract
being asked about; a hit = a retrieved chunk overlapping a CUAD gold span. 95% bootstrap CIs.
`uv run cqa-eval retrieval-eval --label 'windows / bge-small (chosen)' --embedding bge-small`,
run 2026-10-03 at commit 807abd3:

| method | recall@5 (95% CI) | MRR@10 (95% CI) |
|---|---|---|
| Postgres full-text (`ts_rank`) | 0.72 [0.68, 0.77] | 0.58 [0.54, 0.62] |
| BM25 (reference implementation) | 0.72 [0.68, 0.77] | 0.60 [0.56, 0.64] |
| vector (bge-small-en-v1.5, local) | 0.72 [0.67, 0.77] | 0.52 [0.48, 0.56] |
| **hybrid (full-text + vector, RRF)** | **0.75 [0.71, 0.80]** | **0.61 [0.57, 0.65]** |

Hybrid beats full-text alone by +0.029 recall@5 (paired 95% CI [+0.003, +0.056]). A
section-aware chunker was built, measured, and removed because it didn't help. Weakest category:
Parties (0.23), whose answer is in the contract's first chunk 40/40 times; M3 tests a fix.
Full tables: [eval/results/retrieval.md](eval/results/retrieval.md). Reasoning:
[DECISIONS.md](DECISIONS.md).

## Data

Uses [CUAD v1](https://github.com/TheAtticusProject/cuad) by The Atticus Project, licensed
CC BY 4.0. See [DATA.md](DATA.md) for attribution and how the subset is pinned.
Not legal advice.
