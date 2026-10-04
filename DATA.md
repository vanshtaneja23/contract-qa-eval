# Data

## Source

This project uses **CUAD v1**, the Contract Understanding Atticus Dataset, created by
[The Atticus Project](https://www.atticusprojectai.org/cuad).

- Official repository: https://github.com/TheAtticusProject/cuad
- File used: `data.zip` → `CUADv1.json`, fetched from
  `https://github.com/TheAtticusProject/cuad/raw/main/data.zip`
- Pinned SHA-256 of `data.zip`: `f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a`
  (the download is rejected if it does not match)
- The Hugging Face dataset `theatticusproject/cuad-qa` is a loader script that downloads this
  same zip, so the zip is fetched directly.

## License and attribution

CUAD is licensed under [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

> Dan Hendrycks, Collin Burns, Anya Chen, Spencer Ball. *CUAD: An Expert-Annotated NLP
> Dataset for Legal Contract Review.* NeurIPS 2021 Datasets and Benchmarks Track.
> https://arxiv.org/abs/2103.06268

The project's code is MIT-licensed (see `LICENSE`). That license does not cover CUAD.

## What is in this repository

The contract texts are **not** committed. `cqa-eval ingest` downloads them into `eval/data/`
(gitignored). The repository only contains `eval/subset.json`, which lists the contract titles
in the subset together with a SHA-256 hash of each contract's text, so the exact subset is
reproducible and any change in the upstream text is detected.

No contract text is modified. Chunks are stored as character offsets into the unmodified text.

## Subset

40 of CUAD's 510 contracts, chosen by a seeded random sample (`seed=42`) over contracts sorted
by title (`contract_eval.cuad.select_subset`). They are spread round-robin over 4 "matters" so
that matter isolation can be tested later.

## Ground truth

Each CUAD contract has 41 category questions (e.g. "Governing Law", "Cap On Liability",
"Non-Compete"). An answer is one or more expert-highlighted spans, each given as a character
offset plus its text. `is_impossible: true` means the contract has no clause of that category,
which is the ground truth for "the clause is not present" questions.

On load, every gold span is checked: `context[start:start+len(text)] == text`. Loading fails if
any span is misaligned.
