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
