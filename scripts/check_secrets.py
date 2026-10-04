#!/usr/bin/env python3
"""Block commits that contain API keys or other credentials.

Usage:
    check_secrets.py            # scan files staged for commit (pre-commit hook)
    check_secrets.py --all      # scan every tracked file (CI)
    check_secrets.py FILE...    # scan specific files

Exits 1 and prints file:line for every hit. Matched values are masked in the
output so the secret does not end up in terminal scrollback or CI logs.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PATTERNS: dict[str, re.Pattern[str]] = {
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "openai_key": re.compile(r"sk-(?!ant-)(?:proj-|svcacct-)?[A-Za-z0-9_\-]{32,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    # key = "long literal" style assignments. Placeholders like
    # "your-...-here" / "change-me" are short or contain obvious words.
    "generic_secret": re.compile(
        r"""(?i)\b(?:api[_-]?key|secret|token|passwd|password)\b\s*[:=]\s*["']([A-Za-z0-9_\-/+=]{24,})["']"""
    ),
}

# Files that must never be committed, whatever their content.
FORBIDDEN_NAMES = re.compile(r"(^|/)\.env(\.(?!example$)[^/]+)?$")

SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".woff", ".woff2", ".lock"}


def _mask(value: str) -> str:
    return value[:6] + "…" if len(value) > 6 else "…"


def scan_text(text: str) -> list[tuple[int, str, str]]:
    """Return (line_number, pattern_name, masked_match) for every hit."""
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if "secret-scan: allow" in line:
            continue
        for name, pattern in PATTERNS.items():
            for m in pattern.finditer(line):
                hits.append((lineno, name, _mask(m.group(0))))
    return hits


def scan_path(path: str, content: str | None = None) -> list[str]:
    problems: list[str] = []
    if FORBIDDEN_NAMES.search(path):
        problems.append(f"{path}: .env files must not be committed (use .env.example)")
        return problems
    if Path(path).suffix.lower() in SKIP_SUFFIXES:
        return problems
    if content is None:
        try:
            content = Path(path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            return problems
    for lineno, name, masked in scan_text(content):
        problems.append(f"{path}:{lineno}: possible {name} ({masked})")
    return problems


def _git_lines(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def _staged_content(path: str) -> str | None:
    # Scan what is actually staged (the index), not the working-tree copy.
    res = subprocess.run(["git", "show", f":{path}"], capture_output=True)
    try:
        return res.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def main(argv: list[str]) -> int:
    problems: list[str] = []
    if argv[:1] == ["--all"]:
        for path in _git_lines("ls-files"):
            problems += scan_path(path)
    elif argv:
        for path in argv:
            problems += scan_path(path)
    else:
        for path in _git_lines("diff", "--cached", "--name-only", "--diff-filter=ACMR"):
            content = _staged_content(path)
            if content is None and not FORBIDDEN_NAMES.search(path):
                continue
            problems += scan_path(path, content if content is not None else "")

    if problems:
        print("secret scan FAILED:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        print(
            "Move the value to .env (gitignored). For a false positive, add a "
            "'secret-scan: allow' comment on that line.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
