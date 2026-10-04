"""Load CUAD v1 (Contract Understanding Atticus Dataset, CC BY 4.0).

Official source: https://github.com/TheAtticusProject/cuad (data.zip). The
Hugging Face dataset `theatticusproject/cuad-qa` is a loader script that
downloads this same zip, so we fetch it directly and pin its SHA-256.

CUAD is in SQuAD format: one record per contract, with 41 "category"
questions; each answer is an expert-highlighted span (character offset + text).
`is_impossible` means the contract has no clause of that category.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "eval" / "data"
SUBSET_PATH = REPO_ROOT / "eval" / "subset.json"

CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
CUAD_ZIP_SHA256 = "f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a"
CUAD_JSON_NAME = "CUADv1.json"


@dataclass(frozen=True, slots=True)
class GoldSpan:
    start: int
    text: str

    @property
    def end(self) -> int:
        return self.start + len(self.text)


@dataclass(frozen=True, slots=True)
class CuadQuestion:
    id: str
    category: str
    question: str
    is_impossible: bool
    spans: tuple[GoldSpan, ...]


@dataclass(frozen=True, slots=True)
class CuadContract:
    title: str
    text: str
    questions: tuple[CuadQuestion, ...]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_cuad(data_dir: Path = DATA_DIR) -> Path:
    """Download and verify data.zip once; return the path to CUADv1.json."""
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / CUAD_JSON_NAME
    if json_path.exists():
        return json_path
    zip_path = data_dir / "cuad_data.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(CUAD_URL, zip_path)
    actual = _sha256_file(zip_path)
    if actual != CUAD_ZIP_SHA256:
        zip_path.unlink()
        raise RuntimeError(f"CUAD zip checksum mismatch: {actual}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(CUAD_JSON_NAME, data_dir)
    return json_path


def parse_cuad(raw: dict[str, Any]) -> list[CuadContract]:
    contracts = []
    for record in raw["data"]:
        (paragraph,) = record["paragraphs"]  # CUAD has one paragraph = whole contract
        text: str = paragraph["context"]
        questions = []
        for qa in paragraph["qas"]:
            spans = tuple(GoldSpan(a["answer_start"], a["text"]) for a in qa["answers"])
            for s in spans:
                if text[s.start : s.end] != s.text:
                    raise ValueError(f"gold span does not match context in {qa['id']}")
            questions.append(
                CuadQuestion(
                    id=qa["id"],
                    category=qa["id"].rsplit("__", 1)[1],
                    question=qa["question"],
                    is_impossible=bool(qa["is_impossible"]),
                    spans=spans,
                )
            )
        contracts.append(CuadContract(record["title"], text, tuple(questions)))
    return contracts


def load_cuad(path: Path) -> list[CuadContract]:
    return parse_cuad(json.loads(path.read_text(encoding="utf-8")))


def select_subset(contracts: list[CuadContract], n: int, seed: int) -> list[CuadContract]:
    """Deterministic random sample (sorted by title first, so input order doesn't matter)."""
    ordered = sorted(contracts, key=lambda c: c.title)
    return sorted(random.Random(seed).sample(ordered, n), key=lambda c: c.title)


def build_manifest(subset: list[CuadContract], seed: int) -> dict[str, Any]:
    return {
        "source_url": CUAD_URL,
        "zip_sha256": CUAD_ZIP_SHA256,
        "license": "CC BY 4.0, The Atticus Project",
        "seed": seed,
        "n": len(subset),
        "contracts": [{"title": c.title, "sha256": c.sha256, "chars": len(c.text)} for c in subset],
    }


def apply_manifest(contracts: list[CuadContract], manifest: dict[str, Any]) -> list[CuadContract]:
    """Return exactly the contracts pinned in the manifest, verifying their text hashes."""
    by_title = {c.title: c for c in contracts}
    out = []
    for entry in manifest["contracts"]:
        c = by_title.get(entry["title"])
        if c is None:
            raise ValueError(f"contract in manifest not found in CUAD: {entry['title']}")
        if c.sha256 != entry["sha256"]:
            raise ValueError(f"contract text changed since manifest was written: {entry['title']}")
        out.append(c)
    return out
