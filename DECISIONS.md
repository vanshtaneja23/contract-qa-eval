# Decisions

Each entry: what was chosen, what else was considered, and the evidence. Numbers come from
real runs; the command, date and git commit are given (or are in `eval/results/retrieval.json`).

---

## M1: data and storage

### D1. CUAD from the official zip, pinned by SHA-256
- **Chosen:** download `data.zip` from the CUAD GitHub repo, verify its SHA-256, and pin the
  40-contract subset by per-contract text hash (`eval/subset.json`).
- **Alternative:** the Hugging Face `theatticusproject/cuad-qa` dataset.
- **Evidence:** that HF dataset is a loader script that downloads the same zip, so it adds a
  dependency without adding data. Every gold span is checked against its contract text on load,
  and loading fails if any is misaligned; none are.

### D2. Character offsets as the source of truth for citations
- **Chosen:** each chunk stores `start_char`/`end_char` into the unmodified contract, with a DB
  `CHECK (char_length(text) = end_char - start_char)`.
- **Alternative:** store chunk text only and verify citations by substring search.
- **Why:** substring search can't tell which occurrence was cited ("the State of Delaware" can
  appear in several clauses). Python slicing and Postgres `substring()` both count Unicode code
  points, so offsets mean the same thing in both (tested with emoji and accents). JavaScript
  counts UTF-16 units, so the frontend must convert (M6).
- **Evidence:** 0 of 2,163 stored chunks differ from `substring(document, start, end)` in SQL
  (checked 2026-10-03).

### D3. `matter_id` copied onto chunks, guarded by a composite foreign key
- **Chosen:** `chunks.matter_id` plus `FOREIGN KEY (document_id, matter_id) REFERENCES
  documents(id, matter_id)`.
- **Alternative:** derive the matter by joining chunks → documents at query time.
- **Why:** row-level security (M4) and retrieval can filter chunks directly. The composite FK
  makes it impossible for the copy to disagree with the document's matter (integration test).
  This keeps data consistent; it is not access control (RLS does that).

### D4. One Postgres for relational data, full-text search and vectors
- **Chosen:** Postgres 16 + pgvector.
- **Alternative:** a separate vector database.
- **Why:** one transaction boundary, and the M4 isolation policy covers lexical and vector
  search alike. At 2,163 chunks an exact vector scan is fast, so there is no ANN index.

---

## M2: retrieval

All retrieval numbers: 375 answerable (contract, question) pairs, i.e. every pair in the 20
eval categories over the 40 contracts. Retrieval is scoped to the one contract being asked
about. A hit = a retrieved chunk overlapping any CUAD gold span. 95% CIs are percentile
bootstrap over questions; differences between methods are paired (same questions). Full tables:
[`eval/results/retrieval.md`](eval/results/retrieval.md).

### D5. Removed section-aware chunking; plain 200-word windows with 40-word overlap
- **Tried:** cutting at headings ("1. Term", "ARTICLE IV"), merging heading-only sections,
  windowing long sections, falling back to windows when a contract has under 3 headings.
- **Evidence:** windows matched or beat it for every method (window − section):

  | method | Δ recall@5 (95% CI) | Δ MRR@10 (95% CI) |
  |---|---|---|
  | Postgres FTS | +0.011 [−0.016, +0.037] | +0.017 [−0.006, +0.040] |
  | BM25 | +0.011 [−0.013, +0.035] | +0.004 [−0.017, +0.024] |
  | vector (bge-small) | +0.000 [−0.037, +0.035] | −0.011 [−0.046, +0.020] |
  | hybrid | +0.011 [−0.019, +0.040] | +0.013 [−0.016, +0.042] |

- **Why it didn't help:** 71% of its chunks were windows anyway (CUAD's PDF-extracted text often
  flattens headings into running text, and many sections exceed 200 words). 12/40 contracts got
  no section chunks at all.
- **What was lost:** gold spans fully inside a single chunk: 95.5% (section) vs 91.5% (windows).
  That could matter for answer quality if the model needs a whole clause. Revisit if M5 shows
  failures caused by clauses split across chunks.
- **Reproduce the section runs:** `git checkout ec7e9b8`; runs labelled `section chunks / …`.

### D6. Hybrid retrieval: Postgres FTS + bge-small vectors, fused with Reciprocal Rank Fusion
- **Alternatives:** lexical only, vector only, BM25 only, weighted score fusion.
- **Evidence** (`windows / bge-small (chosen)`, 2026-10-03, commit 807abd3):

  | method | recall@5 (95% CI) | MRR@10 (95% CI) |
  |---|---|---|
  | Postgres FTS | 0.72 [0.68, 0.77] | 0.58 [0.54, 0.62] |
  | BM25 | 0.72 [0.68, 0.77] | 0.60 [0.56, 0.64] |
  | vector | 0.72 [0.67, 0.77] | 0.52 [0.48, 0.56] |
  | **hybrid** | **0.75 [0.71, 0.80]** | **0.61 [0.57, 0.65]** |

  Hybrid − FTS: recall@5 +0.029 [+0.003, +0.056], MRR@10 +0.032 [+0.007, +0.058]. These are the
  only intervals in the table that exclude zero in hybrid's favour. Hybrid − BM25 is
  +0.029 [−0.003, +0.061], suggestive but not significant.
- **Why RRF, not weighted scores:** RRF uses ranks only, so `ts_rank` values and cosine
  similarities never need to be put on one scale. Weighted fusion was not tried.
- **Lexical and vector fail differently:** vector is better on Agreement Date (0.72 vs 0.58) and
  Non-Compete (0.70 vs 0.30); lexical is better on License Grant (0.94 vs 0.56) and Audit Rights
  (0.72 vs 0.56). That complementarity is why fusing them helps.

### D7. Postgres `ts_rank` kept, although it is not BM25
- `ts_rank` has no IDF and no term-frequency saturation. A 30-line BM25 (same Snowball
  stemmer) was built to check whether that matters.
- **Evidence:** same recall@5 (+0.000 [−0.027, +0.027]); BM25 has better MRR@10
  (+0.023 [+0.001, +0.043]).
- **Why keep `ts_rank`:** it runs inside Postgres, where M4's row-level security applies, and it
  needs no corpus statistics to maintain. BM25 would need an index outside the database.
  **Not tested yet:** hybrid with BM25 instead of `ts_rank`.

### D8. OR query semantics for full-text search
- `plainto_tsquery` ANDs every word of the question. Measured over the 20 eval questions: on
  average only 7.8 of 40 contracts had any chunk matching all the words (8 questions matched
  none). With OR: 39.3 of 40. Query terms are restricted to `[A-Za-z0-9]` runs, so user input
  cannot inject tsquery operators (unit tested).

### D9. Embedding model: bge-small-en-v1.5, chosen on truncation, not accuracy
- **Alternative:** all-MiniLM-L6-v2. Both are local and 384-dimensional.
- **Evidence:** accuracy is indistinguishable (bge − MiniLM on windows: vector recall@5
  +0.019 [−0.027, +0.064], hybrid recall@5 −0.005 [−0.035, +0.024]). But MiniLM's 256-token
  limit truncates **61.3%** of chunks (median chunk = 262 tokens), against **2.4%** for bge-small
  (512-token limit). Truncated text is invisible to vector search, and this eval doesn't
  isolate that effect. Embedding all 2,163 chunks took 11.1 s (bge) vs 4.6 s (MiniLM) on an
  Apple-silicon laptop.

### D10. Eval design choices and their limits
- **All 375 answerable pairs, not just the 90 answerable questions** in `eval/questions.jsonl`:
  retrieval is free, and more questions mean tighter intervals. The 120-question set (25% with no
  such clause) is for the paid model evals in M5.
- **Overlap, not containment, counts as a hit:** 8.5% of gold spans (88 of 1,034) are not fully
  inside any single chunk.
- **Limitation:** questions are clustered within 40 contracts, and the bootstrap resamples
  questions, not contracts, so the intervals are somewhat optimistic.

### Finding for M3: Parties fails at retrieval, not generation
Parties has recall@5 0.23 with hybrid, because the question's words ("parties", "agreement")
appear in nearly every chunk. Yet the answer overlaps the contract's **first chunk in 40/40**
contracts (Agreement Date: 29/36). M3 will test always including the first chunk in the
model's context, and keep it only if it measurably helps.

---

## M3: cited answers, verification, confidence gate

### D11. The model never produces offsets; code computes them, and a verifier re-checks
- **Chosen:** the model returns `{chunk_id, quote}`. `resolve_quote` finds the quote inside the
  excerpts the model was shown and computes `start`/`end`. `verify_citation` then independently
  checks every final citation: document in scope, offsets in range, `text[start:end] ==
  quoted_text`, and inside the shown context.
- **Alternatives:** ask the model for character offsets (models are unreliable at character
  arithmetic), or the Claude API's built-in document citations (incompatible with structured
  output, and Anthropic-only, while M5 needs the same pipeline for every provider).
- **Why two layers:** construction makes citations correct by design; verification catches bugs
  in construction and checks citations from any other source. Adversarial unit tests cover
  paraphrases, shifted offsets, forged document ids, out-of-scope documents and text outside
  the shown context. A property test checks that every resolved quote verifies.

### D12. Whitespace-tolerant quote matching, storing the exact source slice
- CUAD text has double spaces and hard line breaks that models tend to collapse. A quote that
  matches only after treating any whitespace run as equal is accepted, but the citation stores
  the **source** slice, so `quoted_text` is still verbatim. A changed word still fails.
- **Status:** each result records `whitespace_matches`; the smoke run will measure how often
  this rescues a citation. If it is rare, it gets removed (simplest design that works).

### D13. One invalid citation rejects the whole answer
- **Alternative:** drop the bad citation and keep the rest. Rejected because a claim whose
  supporting quote is fabricated is exactly the "confidently wrong" failure this project targets.
  A rejected answer is withheld (`answer: null`) and logged to `audit_log` with its reasons.
  "Answered" with no citations is also rejected: an uncited answer can't be verified.

### D14. Context: 6 chunks with each document's first chunk reserved
- **Evidence** (`uv run cqa-eval context-eval`, 2026-10-04, 375 answerable questions; context
  recall = a gold span overlaps some chunk the model sees):

  | context | recall (95% CI) |
  |---|---|
  | top-5 | 0.752 [0.709, 0.797] |
  | top-6 | 0.768 [0.725, 0.813] |
  | **top-6, first chunk reserved** | **0.864 [0.829, 0.901]** |

  Reserved − top-6: +0.096 [+0.064, +0.131], at the same token budget. Parties 0.25 → 1.00,
  Agreement Date 0.69 → 0.97. Cost: one fewer ranked chunk, which lost one question each in
  Non-Compete, Post-Termination Services, Anti-Assignment and Expiration Date.

### D15. Confidence gate on top `ts_rank`, calibrated on a dev split
- **Method:** dev = every pair in the 20 categories except the 120 eval questions (285
  answerable, 395 absent); test = the 120 eval questions. Threshold = the highest value that
  blocks at most 2% of answerable dev questions (`uv run cqa-eval gate-calibrate`, 2026-10-04).

  | signal | AUC dev | AUC test | absent caught (dev / test) | answerable blocked (dev / test) |
  |---|---|---|---|---|
  | top cosine | 0.708 | 0.657 | 8.4% / 6.7% | 1.8% / 0.0% |
  | **top ts_rank** | **0.845** | **0.725** | **22.3% / 13.3%** | **1.8% / 0.0%** |

- **Why ts_rank:** when a clause is absent, its distinctive words usually don't occur in the
  contract, so full-text finds little; embeddings always find something semantically nearby.
- **Limits:** the drop from dev to test AUC shows the dev numbers are optimistic (the test set has
  only 30 absent questions). The gate is a cheap first filter; most abstention has to come from
  the model. M5 measures abstention with and without the gate.
- **Why not RRF scores:** RRF encodes rank only, so its top score is roughly constant whatever the
  match quality. Hence `Retriever.hybrid` returns the raw signals.

### D16. Structured outputs; no refusal fallbacks
- JSON comes from `output_config.format` (JSON schema), not forced tool use, which returns 400 on
  Claude Opus 5.5 and Sonnet 5.5. A malformed response is still possible in principle and
  is rejected as `invalid_json`.
- The Claude API can re-run a refused request on a fallback model. That is off, because in a
  model comparison it would silently score a different model's answer. Refusals are recorded
  as their own rejection reason.

### D17. Cost controls
- Every model response is cached on disk under `sha256(model, settings, system, prompt, schema)`.
  `--offline` refuses uncached calls, so re-runs are reproducible and CI cannot spend money.
- `answer-eval` plans first: it counts gated and cached questions, estimates the rest (1 token
  per 3.5 characters, deliberately high, plus `--est-output-tokens`), and refuses to run above
  `--max-cost` (default $2).

---

## Zero API spend: local open-weight models (2026-10-08)

### D18. All model runs are local (Ollama); paid APIs are blocked, not just unused
- **Decision:** no Anthropic/OpenAI key, no paid calls. Real paid clients raise
  `PaidCallsDisabled` unless `CQA_ALLOW_PAID_CALLS=1`, and `answer-eval --max-cost` defaults to 0.
  The Anthropic and OpenAI adapters stay (same interface) and are tested only against fixtures:
  the Ollama fixture is a recorded local response; the Anthropic/OpenAI fixtures are built in the
  documented response shapes and say so in a `_note` field.
- **Models:** `llama3.1:8b`, `qwen2.5:7b`, `gemma2:9b` (three vendors, similar size; ~5 GB each at
  4-bit, comfortable on an M4 Pro with 24 GB). Judge: `qwen2.5:14b`, a different and larger model
  than the graded ones, to avoid a model grading its own answers.
- **Consequence:** the project compares open-weight models, not frontier models. The README says so.

### D19. Ollama settings that change results
- `temperature 0`, `seed 0` for repeatability.
- `num_ctx 8192` set explicitly: prompts are ~2.3k tokens, and Ollama silently truncates prompts
  longer than its context window. The client also errors if a prompt fills the window.
- `num_predict 1024`: added after a model looped inside its JSON and generated 24k+ tokens until
  the HTTP timeout. A capped response is rejected with reason `length`.

### D20. Prompt v2, tuned on a dev slice, not on the eval set
- **Problem (M3 smoke run, 20 eval questions, qwen2.5:7b, v1 prompt):** 12 of 16 answerable
  questions came back `not_found`, and in 11 of those 12 the gold clause *was* in the context. The
  model was over-cautious, not under-informed.
- **Method:** `--split dev` draws questions disjoint from `eval/questions.jsonl` (same 3:1
  answerable/absent mix). Metric: grounded accuracy (answerable = answered with a citation
  overlapping a CUAD gold span; absent = abstained), no judge needed.
- **Evidence** (40 dev questions, qwen2.5:7b, 2026-10-08):

  | prompt | grounded accuracy | answerable answered | absent abstained | citation validity |
  |---|---|---|---|---|
  | v1 | 0.375 | 5 / 30 | 10 / 10 | 0.71 |
  | v2 | 0.525 | 18 / 30 | 7 / 10 | 0.74 |

  v2 − v1: +0.150, paired 95% CI [−0.050, +0.325]: **not significant at n=40.** Adopted anyway
  because v1's failure (abstaining with the clause in front of it) is the product's core failure
  mode; the cost is visible (3 more absent-clause questions answered) and M5 measures both.
- **Limitation:** tuned on one model (qwen2.5:7b), which may favour it slightly in M5.
- Whitespace-tolerant quote matching (D12) earned its place: in the smoke run it rescued 2 of the 3
  valid citations.

### D21. PII redaction before every model call, and what it costs
- **What:** organisations (corporate suffix, plus their core name), people in signature/notice
  patterns, emails, phones and street addresses become `PARTY_n` / `PERSON_n` / `EMAIL_n` /
  `PHONE_n` / `ADDRESS_n`, numbered per document so the same party is the same token in every
  excerpt. Document titles become "Document 1" (CUAD titles start with the filer's name). The
  model's quotes are located in the redacted text, mapped back through a segment offset map, and
  verified against the original. `audit_log` stores counts by type, never values.
- **Why regex, not NER:** deterministic, fast, and every rule is testable. **Known gaps:** person
  names in running prose, and organisations without a corporate suffix, are not caught.
- **Cost** (`answer-eval --split dev --limit 40`, with and without `--no-redact`, qwen2.5:7b,
  2026-10-08): 296 entities redacted across 39 prompts (PARTY 204, PERSON 49, ADDRESS 32,
  PHONE 7, EMAIL 4).

  | | grounded accuracy | answerable answered | absent abstained | citation validity |
  |---|---|---|---|---|
  | no redaction | 0.525 | 18 / 30 | 7 / 10 | 0.74 |
  | redaction | 0.525 | 18 / 30 | 8 / 10 | 0.77 |

  Redacted − plain: **+0.000, paired 95% CI [−0.100, +0.100]**: no cost detectable at n=40, which
  is not the same as no cost. By category the changes go both ways (Parties −1 of 2, Liquidated
  Damages −1 of 1; Exclusivity +1 of 2, Change of Control +1 of 1). **Parties is where a real cost
  is expected:** the model sees `PARTY_1` instead of a name and has to rely on the mapping back.
- **Single-document only:** placeholder numbers are per document, so a multi-document question
  with redaction raises `NotImplementedError` rather than risk mapping a name to the wrong party.

---

## M5: model comparison (2026-10-08)

### D22. Scoring: judge only where judgment is needed
- **Rules:** absent clause → correct iff the system abstained (`not_found`); answering it is
  *confidently wrong*. Answerable → a shown answer is graded by the judge; abstaining or being
  rejected is a miss (wrong, but not confidently wrong).
- **Two confidently-wrong rates:** *shown* (what a user would see) and *before verifier* (also
  counting answers the verifier rejected). The gap is what the verifier buys: 27–37% of questions
  → 18–23% across the three models.
- **Judge:** `qwen2.5:14b`, larger than and different from the graded models (avoids a model
  grading itself). It sees the question, CUAD's expert-marked clause text and the system's answer
  with its quotes. **Validation is pending:** `cqa-eval label` collects 30 blind human labels
  (10 per model, shuffled, model hidden); `cqa-eval judge-agreement` reports raw agreement and
  Cohen's kappa. Visible strictness in the failure examples (a correct party list marked wrong
  over legal-name formatting) suggests judged accuracy is an underestimate.
- **Statistics:** per-model 95% bootstrap CIs and paired differences on the same questions. No
  pairwise accuracy difference excludes zero at n=120 (largest: qwen2.5:7b − llama3.1:8b
  +8.3 pts, [−0.8, +17.5]), so the report does not declare a winner.
- **Latency:** measured per call on one machine with models run one at a time (no contention).
  Includes prompt processing of ~2.3k tokens; gemma2:9b is ~1.7× slower at p50.
