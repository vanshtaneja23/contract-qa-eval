"""Okapi BM25, written out so it can be compared against Postgres ts_rank.

score(q, d) = sum over query terms t of
    idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

What ts_rank lacks: idf (a rare word like "indemnify" counts the same as a
common one like "agreement") and the k1 saturation (the 10th occurrence of a
word adds as much as the 1st).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

import snowballstemmer

# Same Snowball English stemmer Postgres' 'english' config uses.
_STEMMER = snowballstemmer.stemmer("english")
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS_TEXT = """
a an and any are as at be been but by can could do does for from had has have how i if in into is
it its may of on or our shall should such than that the their them then there these they this those
to under upon was we were what when where which who whom will with would you
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def tokenize(text: str) -> list[str]:
    words = [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]
    return list(_STEMMER.stemWords(words))


class BM25:
    """IDF is computed over the whole corpus; ranking can be limited to candidates
    (e.g. one contract's chunks), matching how the product scopes retrieval."""

    def __init__(self, corpus: dict[int, str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = {cid: Counter(tokenize(text)) for cid, text in corpus.items()}
        self.length = {cid: sum(c.values()) for cid, c in self.tf.items()}
        n = len(self.tf)
        self.avgdl = sum(self.length.values()) / n if n else 0.0
        df: Counter[str] = Counter()
        for counts in self.tf.values():
            df.update(counts.keys())
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query_terms: set[str], cid: int) -> float:
        tf, dl = self.tf[cid], self.length[cid]
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        return sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in query_terms if t in tf)

    def rank(self, query: str, candidates: Iterable[int], k: int = 10) -> list[tuple[int, float]]:
        terms = set(tokenize(query))
        scored = [(cid, self.score(terms, cid)) for cid in candidates]
        scored = [(cid, s) for cid, s in scored if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
