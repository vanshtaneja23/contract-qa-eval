"""Test doubles shared by unit and integration tests. No network, no model downloads."""

import hashlib
import json
import math
import re
from collections.abc import Sequence
from typing import Any

from contract_qa.llm import Completion, Usage
from contract_qa.models import EMBEDDING_DIM


class FakeEmbedder:
    """Deterministic bag-of-words hashing: texts sharing words get similar vectors."""

    def __init__(self, key: str = "fake") -> None:
        self.key = key

    def _vec(self, s: str) -> list[float]:
        v = [0.0] * EMBEDDING_DIM
        for w in re.findall(r"[a-z]+", s.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % EMBEDDING_DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class ScriptedClient:
    """Returns a fixed model response and records the prompts it was sent."""

    model = "claude-haiku-4-5"

    def __init__(self, response: dict[str, Any] | str, stop_reason: str = "end_turn") -> None:
        self.text = response if isinstance(response, str) else json.dumps(response)
        self.stop_reason = stop_reason
        self.prompts: list[str] = []

    def cache_params(self) -> dict[str, Any]:
        return {}

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> Completion:
        self.prompts.append(user)
        return Completion(self.text, Usage(1000, 100), self.model, self.stop_reason, 5.0)
