# Handoff: contract-qa-eval (2026-10-08)

- **Repo:** https://github.com/vanshtaneja23/contract-qa-eval
- **CI:** GitHub Actions `CI` with three jobs: `python` (ruff, mypy, unit tests, secret scan),
  `integration` (Postgres 16 + pgvector service container), `web` (tsc, eslint). Green on the
  handoff commit; live status is the badge in the README.
- **Total API spend: $0.** No paid or frontier model was called. All model runs used open-weight
  models locally via Ollama. Real Anthropic/OpenAI clients refuse to start without
  `CQA_ALLOW_PAID_CALLS=1`, and `answer-eval --max-cost` defaults to 0.

## Model comparison (M5)

120 CUAD questions (90 clause present, 30 absent), same retrieval, context, prompt and
redaction; only the model differs. Apple M4 Pro, 24 GB, Ollama 0.40.1, 4-bit weights
(gemma2:9b Q4_0, others Q4_K_M), run 2026-10-08. Judge: `qwen2.5:14b`.

| model | accuracy (95% CI) | abstains on absent clause | citation validity | confidently wrong (shown) | confidently wrong (before verifier) | p50 / p95 latency | cost / 100 Qs |
|---|---|---|---|---|---|---|---|
| `llama3.1:8b` | 46% [38%, 55%] | 80% | 71% | 18% | 37% | 6.6 s / 10.2 s | $0 |
| `qwen2.5:7b` | 54% [45%, 63%] | 87% | 86% | 18% | 27% | 6.7 s / 10.9 s | $0 |
| `gemma2:9b` | 50% [41%, 59%] | 77% | 80% | 23% | 37% | 11.4 s / 17.2 s | $0 |

- **No winner can be declared:** no paired accuracy difference excludes zero (largest:
  qwen − llama +8.3 pts, 95% CI [−0.8, +17.5]). The prompt was tuned on qwen2.5:7b (dev split).
- **Robust finding:** the citation verifier cuts confidently-wrong answers from 27–37% of
  questions (asserted by the model) to 18–23% (shown to a user).
- Full report with per-category accuracy and 5 failure examples per model:
  `eval/results/report.md`.

## Judge agreement: PENDING (needs you)

Not measured yet: it needs 30 hand labels, and only you can provide them.

```bash
uv run cqa-eval label            # 30 answers, 10 per model, shuffled, model names hidden; c / i / s / q
uv run cqa-eval judge-agreement  # raw agreement, Cohen's kappa, confusion, disagreements
make up && uv run cqa-eval m5-report   # re-renders report.md with the agreement filled in
```

`label` needs neither the database nor a model server, and you can quit (`q`) and resume any
time. What to look for: the judge was visibly too strict in some failure examples (e.g. a
correct "ACCURAY INCORPORATED and SIEMENS AKTIENGESELLSCHAFT" marked wrong), so expect
"human = correct, judge = incorrect" disagreements. That would mean judged accuracy understates
all three models.

## Redaction cost

40 dev questions (disjoint from the eval set), qwen2.5:7b, with vs without redaction,
2026-10-08: grounded accuracy **0.525 vs 0.525**, paired 95% CI **[−0.100, +0.100]**. 296
entities redacted across 39 prompts (PARTY 204, PERSON 49, ADDRESS 32, PHONE 7, EMAIL 4). No
cost detectable at this size; Parties is the category where a real cost is expected.

## M3 smoke run (20 eval questions, qwen2.5:7b)

| | prompt v1 (first run) | final pipeline (prompt v2, redaction, output cap) |
|---|---|---|
| answerable answered / 16 | 3 | 9 |
| citation validity | 50% | 85% |
| rejection reasons | 3 × quote_not_found | 2 × quote_not_found |
| absent clause abstained / 4 | 2 (other 2 rejected) | 2 (other 2 answered) |
| latency p50 / p95 | 6.6 s / 8.7 s | 6.7 s / 10.9 s (M5 run, same config) |

In the v1 run, 11 of 12 abstentions had the clause in context, which led to prompt v2.

## Tests

| suite | count | runs in |
|---|---|---|
| Python unit (`pytest -m "not integration"`) | 106 | CI `python` job |
| Python integration, real Postgres+pgvector (`pytest -m integration`) | 33 | CI `integration` job |
| Web | 0 tests; `tsc` and `eslint` only | CI `web` job |

## Implemented

- Monorepo (uv workspace, Next.js scaffold), docker-compose Postgres 16 + pgvector, Alembic.
- Offset-exact chunking (200-word windows), DB constraints that keep offsets honest.
- Hybrid retrieval (Postgres FTS + local bge-small vectors, RRF) with a measured comparison.
- Cited answers: model gives chunk id + quote, code computes offsets, strict verifier, whole
  answer rejected on any invalid citation; confidence gate; 6-chunk context with first chunk.
- PII redaction with per-document placeholders mapped back in answers and citations.
- Audit log of every question (outcome, rejection reasons, redaction counts by type).
- Local model client (Ollama), paid-call guard, Anthropic/OpenAI adapters tested with fixtures.
- Disk cache of every model response (committed); `--offline` reproduces all results (verified:
  120/120 identical for llama3.1:8b with no model server running).
- Eval CLI: retrieval eval, context eval, gate calibration, answer eval (dev/eval splits,
  dry run, cost guard), LLM judge, blind labelling CLI, judge agreement, M5 report.
- CI with three jobs; pre-commit and CI secret scanning.
- `DECISIONS.md` (D1–D22) with alternatives and evidence for each choice.

## Not implemented

- **Judge validation numbers** (blocked on your 30 labels; tooling is done).
- **M4 row-level security** and its Testcontainers isolation tests. Scoping is application-level
  only; the composite FK keeps data consistent but is not access control.
- **Users / authentication / matter-membership enforcement** (tables exist, unused).
- **M6:** FastAPI endpoints, rate limiting, structured request logs, the Next.js UI
  (citations, compare view, dashboard), Vitest and Playwright tests.
- **Multi-document questions with redaction** (raises `NotImplementedError` by design).
- **Frontier/paid model evaluation** (deliberately not run).
- NER-based redaction (regex only; misses names in running prose).

## Known caveats to mention in an interview

- Small eval (40 contracts, 120 questions): differences under ~10 points are not resolvable.
- CUAD distinguishes Agreement Date from Effective Date; models answering "when was this made?"
  from an effective-date clause are scored wrong. One gold answer is an unfilled template.
- Bootstrap CIs resample questions, not contracts (questions cluster within 40 contracts), so the
  intervals are somewhat optimistic.
