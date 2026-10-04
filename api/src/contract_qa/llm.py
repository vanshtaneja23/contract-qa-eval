"""Model clients: one small interface, one adapter per provider, plus a disk cache.

Every adapter takes the same (system, user, JSON schema) and returns raw JSON
text plus token usage, so an eval can hold the prompt and context fixed and
vary only the model.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import anthropic

# USD per million tokens (input, output). Anthropic first-party API rates,
# checked 2026-10-04. Thinking tokens are billed as output.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    usage: Usage
    model: str
    stop_reason: str
    latency_ms: float
    cached: bool = False


class ModelClient(Protocol):
    model: str

    def cache_params(self) -> dict[str, Any]:
        """Request settings that change the output; part of the cache key."""
        ...

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> Completion: ...


def cost_usd(model: str, usage: Usage) -> float:
    if model not in PRICES:
        raise KeyError(f"no price for {model!r}; add it to PRICES (use (0, 0) for local models)")
    price_in, price_out = PRICES[model]
    return (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Rough token count for dry-run cost estimates: 1 token per 3.5 characters.
    English averages ~4 characters per token, so this deliberately errs high."""
    return math.ceil(len(text) / 3.5)


class AnthropicClient:
    def __init__(
        self, model: str, max_tokens: int = 16000, client: anthropic.Anthropic | None = None
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self._client = client

    def cache_params(self) -> dict[str, Any]:
        return {"provider": "anthropic", "max_tokens": self.max_tokens}

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> Completion:
        # Deliberately no server-side refusal fallbacks: in a model comparison a
        # fallback would silently score another model's answer. Refusals are
        # recorded as their own outcome instead.
        t0 = time.perf_counter()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = "".join(block.text for block in resp.content if block.type == "text")
        return Completion(
            text=text,
            usage=Usage(resp.usage.input_tokens, resp.usage.output_tokens),
            model=resp.model,
            stop_reason=str(resp.stop_reason),
            latency_ms=latency_ms,
        )


class CacheMiss(RuntimeError):
    """Raised in offline mode when a response is not cached (no API call is made)."""


class CachedClient:
    """Wraps any ModelClient with an on-disk cache keyed by
    sha256(model + request settings + system + user + schema).

    offline=True never calls the model: re-runs and CI are reproducible from the
    cache and cannot spend money.
    """

    def __init__(self, inner: ModelClient, cache_dir: Path, offline: bool = False) -> None:
        self.inner = inner
        self.model = inner.model
        self.cache_dir = cache_dir
        self.offline = offline

    def cache_params(self) -> dict[str, Any]:
        return self.inner.cache_params()

    def key(self, system: str, user: str, schema: dict[str, Any]) -> str:
        payload = {
            "model": self.inner.model,
            "params": self.inner.cache_params(),
            "system": system,
            "user": user,
            "schema": schema,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def path(self, key: str) -> Path:
        return self.cache_dir / self.inner.model.replace("/", "_").replace(":", "_") / f"{key}.json"

    def is_cached(self, system: str, user: str, schema: dict[str, Any]) -> bool:
        return self.path(self.key(system, user, schema)).exists()

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> Completion:
        path = self.path(self.key(system, user, schema))
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return Completion(
                text=d["text"],
                usage=Usage(d["input_tokens"], d["output_tokens"]),
                model=d["model"],
                stop_reason=d["stop_reason"],
                latency_ms=d["latency_ms"],
                cached=True,
            )
        if self.offline:
            raise CacheMiss(f"{self.inner.model}: response not cached ({path.name})")
        c = self.inner.complete(system, user, schema)
        record = {
            "text": c.text,
            "input_tokens": c.usage.input_tokens,
            "output_tokens": c.usage.output_tokens,
            "model": c.model,
            "stop_reason": c.stop_reason,
            "latency_ms": c.latency_ms,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)  # atomic: an interrupted run never leaves a half-written entry
        return c
