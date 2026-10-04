"""Paths and helpers shared by the eval commands."""

from __future__ import annotations

import json
import subprocess

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

QUESTIONS_PATH = REPO_ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
CACHE_DIR = REPO_ROOT / "eval" / "cache"


def load_subset(n: int = 40, seed: int = 42) -> list[CuadContract]:
    """The pinned contract subset: from eval/subset.json if present, else sampled and pinned."""
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
    print(f"wrote {SUBSET_PATH.relative_to(REPO_ROOT)}")
    return subset


def git_commit() -> str:
    """Short HEAD hash, with '-dirty' if source files have uncommitted changes."""
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "diff", "--quiet", "--", "api", "eval/src"]).returncode != 0
    return out.stdout.strip() + ("-dirty" if dirty else "")
