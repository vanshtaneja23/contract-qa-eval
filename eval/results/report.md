# Model comparison (M5)

Generated 2026-10-08 at commit 303ffba-dirty by `uv run cqa-eval m5-report`. All models are open-weight and ran locally via Ollama; **no paid or frontier API models were run.**

- Hardware: Apple M4 Pro, 24 GB unified memory, Ollama 0.40.1; 4-bit weights (llama3.1:8b, qwen2.5:7b, qwen2.5:14b: Q4_K_M; gemma2:9b: Q4_0); one model loaded at a time
- Questions: `eval/questions.jsonl`, 120 (90 where the clause exists, 30 where it does not), same retrieval, same 6-excerpt context, same prompt, PII redacted; only the model differs.
- Answer correctness: LLM judge `ollama:qwen2.5:14b` (a different, larger model than those graded) on shown answers to answerable questions; everything else is scored by rule.

| model | accuracy (95% CI) | abstains on absent clause | citation validity | confidently wrong (shown) | confidently wrong (before verifier) | p50 / p95 latency | cost per 100 Qs |
|---|---|---|---|---|---|---|---|
| `llama3.1:8b` | 46% [38%, 55%] | 80% | 71% | 18% | 37% | 6.6s / 10.2s | $0 (local) |
| `qwen2.5:7b` | 54% [45%, 63%] | 87% | 86% | 18% | 27% | 6.7s / 10.9s | $0 (local) |
| `gemma2:9b` | 50% [41%, 59%] | 77% | 80% | 23% | 37% | 11.4s / 17.2s | $0 (local) |

Definitions: **accuracy** = correct answers to answerable questions plus correct abstentions on absent clauses, over all 120. **Citation validity** = share of the model's answers whose every quote verified (the rest are rejected and withheld). **Confidently wrong (shown)** = a wrong answer the user would actually see. **Before verifier** also counts answers the verifier rejected.

| model | accuracy on answerable | over-abstention on answerable | questions skipped by the gate |
|---|---|---|---|
| `llama3.1:8b` | 34% | 22% | 4 |
| `qwen2.5:7b` | 43% | 26% | 4 |
| `gemma2:9b` | 41% | 18% | 4 |

Paired differences (same 120 questions, bootstrap 95% CI). An interval that includes 0 means the data cannot tell the two models apart.

| comparison | Δ accuracy | Δ confidently wrong (shown) |
|---|---|---|
| `llama3.1:8b` minus `qwen2.5:7b` | -8.3 pts [-17.5, +0.8] | +0.0 pts [-7.5, +7.5] |
| `llama3.1:8b` minus `gemma2:9b` | -4.2 pts [-12.5, +4.2] | -5.0 pts [-14.2, +3.3] |
| `qwen2.5:7b` minus `gemma2:9b` | +4.2 pts [-4.2, +13.3] | -5.0 pts [-12.5, +2.5] |

## Judge validation

**Pending:** the 30 hand labels (`uv run cqa-eval label`) have not been collected yet, so the judge is not yet validated. Treat judged accuracy as provisional.

## Per category (accuracy)

| category | `llama3.1:8b` | `qwen2.5:7b` | `gemma2:9b` |
|---|---|---|---|
| Agreement Date | 14% | 29% | 29% |
| Anti-Assignment | 43% | 86% | 57% |
| Audit Rights | 71% | 71% | 86% |
| Cap On Liability | 57% | 57% | 71% |
| Change Of Control | 40% | 60% | 20% |
| Exclusivity | 33% | 67% | 50% |
| Expiration Date | 43% | 14% | 43% |
| Governing Law | 43% | 71% | 57% |
| Insurance | 57% | 57% | 43% |
| Ip Ownership Assignment | 80% | 80% | 100% |
| License Grant | 40% | 80% | 40% |
| Liquidated Damages | 40% | 40% | 40% |
| Minimum Commitment | 60% | 40% | 20% |
| Non-Compete | 40% | 40% | 20% |
| Notice Period To Terminate Renewal | 80% | 60% | 80% |
| Parties | 20% | 40% | 20% |
| Post-Termination Services | 14% | 29% | 29% |
| Renewal Term | 67% | 67% | 83% |
| Revenue/Profit Sharing | 20% | 40% | 20% |
| Termination For Convenience | 57% | 57% | 71% |

## Failure examples (5 per model)

### `llama3.1:8b`

- **Post-Termination Services** (q112): _What obligations continue after this agreement terminates or expires?_
  Shown answer: "The parties' obligations to not unreasonably delay or withhold agreement, approval, acceptance, consent, or similar action continue after this agreement terminates or expires (C4).". Confidently wrong: the contract has no such clause (CUAD), but the model answered and its quotes verified.
- **Audit Rights** (q048): _Does either party have the right to audit the other's books or records?_
  Shown answer: "Yes, Reseller agrees to allow Diversinet Corp. or its agents and representatives the right to examine and audit such books, records and accounts during Reseller's normal business hours for no more than once per calendar quarter upon reasonable notice.". Confidently wrong: judge: The system's answer correctly identifies that Reseller allows Diversinet or its agents to audit books, records, and accounts, but it misses the substance of the first reference clause which grants Licensee broader audit rights beyond just financial records..
- **Agreement Date** (q068): _On what date was this agreement made?_
  Shown answer: 'the later of the two signature dates below (the '. Confidently wrong: the contract has no such clause (CUAD), but the model answered and its quotes verified.
- **Parties** (q003): _Who are the parties to this agreement?_
  Shown answer: 'ACCURAY INCORPORATED, a Delaware corporation with its executive offices located at 1310 Chesapeake Terrace, Sunnyvale, California 94089, USA ('. Confidently wrong: judge: The system's answer only partially identifies one party (Accuray) and does not mention the other party (Siemens/Siemens Aktiengesellschaft), and the cited quotes do not relate to the parties involved in the agreement..
- **Post-Termination Services** (q038): _What obligations continue after this agreement terminates or expires?_
  Shown answer: 'Sections 5.02, 5.04, 5.05 and 8.02, and Articles Article IV, Article VII and Article IX shall survive the expiration or termination of this Agreement in accordance with the respective terms thereof.'. Confidently wrong: judge: The system answer does not address the specific obligations that continue after the agreement terminates or expires, but instead refers to the general survival of certain sections and articles without specifying the obligations mentioned in the reference clause text..

### `qwen2.5:7b`

- **Agreement Date** (q004): _On what date was this agreement made?_
  Shown answer: 'The agreement was made on the ____ day of ___________, 2000.'. Confidently wrong: judge: The system answer does not provide the specific date on which the agreement was made, as the blanks are not filled in with actual date information, despite the reference clause indicating a format for the date..
- **Agreement Date** (q068): _On what date was this agreement made?_
  Shown answer: "The agreement was made on the later of the two signature dates, which are July 24 and July 27. Specifically, it states that the agreement is 'entered into and effective as of the later of the two signature dates below (the ". Confidently wrong: the contract has no such clause (CUAD), but the model answered and its quotes verified.
- **Agreement Date** (q076): _On what date was this agreement made?_
  Shown answer: 'The agreement was made on 1.4.2011.'. Confidently wrong: the contract has no such clause (CUAD), but the model answered and its quotes verified.
- **Insurance** (q017): _Is either party required to maintain insurance?_
  Shown answer: 'Yes, both parties are required to maintain insurance under the agreement. The Company is required to provide insurance certificates to the Repairer, and the Repairer is required to maintain specific types of insurance.'. Confidently wrong: judge: The system answer incorrectly states that both parties are required to maintain insurance, but the reference clause text only explicitly mentions the Repairer's requirement to maintain specific types of insurance and the Company's obligation to provide insurance certificates, not to maintain insurance..
- **Parties** (q003): _Who are the parties to this agreement?_
  Shown answer: "The parties to this agreement are ACCURAY INCORPORATED and SIEMENS AKTIENGESELLSCHAFT. This is cited from Excerpt C1: 'ACCURAY INCORPORATED, a Delaware corporation... (". Confidently wrong: judge: The system's answer incorrectly lists 'ACCURAY INCORPORATED' as a party, while the reference mentions 'Accuray' without specifying the full legal name, and also includes 'Siemens' which the system omits. The reference does not provide a definitive full legal name for Accuray, and the system incorrectly assumes 'ACCURAY INCORPORATED' is a party without clear evidence from the reference..

### `gemma2:9b`

- **Agreement Date** (q068): _On what date was this agreement made?_
  Shown answer: 'This Agreement is entered into and effective as of the later of the two signature dates below (the "Effective Date")'. Confidently wrong: the contract has no such clause (CUAD), but the model answered and its quotes verified.
- **Agreement Date** (q076): _On what date was this agreement made?_
  Shown answer: 'This Turn - Key Manufacturing Agreement (the "Agreement") is effective as of the 1.4.2011 (the "Effective Date")'. Confidently wrong: the contract has no such clause (CUAD), but the model answered and its quotes verified.
- **Insurance** (q017): _Is either party required to maintain insurance?_
  Shown answer: 'Yes, both parties are required to maintain insurance. The Company shall procure and maintain insurance as specified in Clause 8, and the Repairer shall procure and maintain Aviation Liability insurance as specified in Clause 8.7.'. Confidently wrong: judge: The system's answer incorrectly states that both parties are required to maintain insurance, but the reference clause only explicitly mentions the Company's insurance requirements. The Repairer's insurance requirement is mentioned but not in the same context as the Company's comprehensive insurance obligations..
- **Parties** (q003): _Who are the parties to this agreement?_
  Shown answer: 'The parties to this agreement are ACCURAY INCORPORATED and SIEMENS AKTIENGESELLSCHAFT.'. Confidently wrong: judge: The system's answer incorrectly states that ACCURAY INCORPORATED is a party to the agreement, while the reference indicates that the party is Accuray (without 'INCORPORATED'). Although SIEMENS AKTIENGESELLSCHAFT is correctly identified, the answer contains an inaccuracy regarding Accuray's full legal name or designation..
- **Minimum Commitment** (q009): _Is there a minimum purchase, volume or spending commitment?_
  Shown answer: 'It is acknowledged that the purchase of Shares in the Offering is subject to the maximum and minimum purchase limitations as described in the Plan and that the Company and the Bank may reject, in whole or in part, any orders received in the Community Offering or Syndicated Community Offering.'. Confidently wrong: judge: The system's answer and cited quote do not address a minimum purchase commitment but rather mention maximum and minimum limitations without specifying the minimum purchase requirement of 708,050 Shares as stated in the reference clause..

