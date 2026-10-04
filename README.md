# contract-qa-eval

Ask a question about a contract → get an answer with citations to the exact clause text →
and know which frontier model does this most reliably.

> Work in progress, built milestone by milestone. This README is completed in Milestone 7.
> Every number here comes from a real run, listed with the command and the date.

## Status

- [x] **M1** Repo, schema, ingestion with offset-exact chunking
- [ ] M2 Retrieval (BM25 / vector / hybrid) and retrieval eval
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
make check      # lint, types, all tests, web typecheck + lint
```

## Ingestion results (M1)

`uv run cqa-eval ingest --n 40 --seed 42 --matters 4`, run 2026-10-03:

| metric | value |
|---|---|
| contracts / matters | 40 / 4 |
| chunks | 2,611 (770 section, 1,841 window) |
| contracts with only window chunks (few headings, or every section > 200 words) | 12 / 40 |
| words per chunk (median / max) | 200 / 200 |
| CUAD gold spans that fit entirely in one chunk | 987 / 1,034 (95.5%) |
| chunks whose stored text ≠ `substring(document, start, end)` in SQL | 0 / 2,611 |

## Data

Uses [CUAD v1](https://github.com/TheAtticusProject/cuad) by The Atticus Project, licensed
CC BY 4.0. See [DATA.md](DATA.md) for attribution and how the subset is pinned.
Not legal advice.
