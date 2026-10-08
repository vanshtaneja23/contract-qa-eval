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


OLLAMA_PREFIX = "ollama:"


def is_local(model: str) -> bool:
    return model.startswith(OLLAMA_PREFIX)


def cost_usd(model: str, usage: Usage) -> float:
    if is_local(model):
        return 0.0
    if model not in PRICES:
        raise KeyError(f"no price for {model!r}; add it to PRICES before running it")
    price_in, price_out = PRICES[model]
    return (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000


class PaidCallsDisabled(RuntimeError):
    """Raised when constructing a real paid-API client without explicit opt-in."""


def _require_paid_opt_in(provider: str) -> None:
    # The project runs at zero API spend: real paid clients refuse to start
    # unless CQA_ALLOW_PAID_CALLS=1. Tests inject stub clients instead.
    if os.environ.get("CQA_ALLOW_PAID_CALLS") != "1":
        raise PaidCallsDisabled(
            f"{provider} calls cost money and are disabled. Set CQA_ALLOW_PAID_CALLS=1 to opt in."
        )


def estimate_tokens(text: str) -> int:
    """Rough token count for dry-run cost estimates: 1 token per 3.5 characters.
    English averages ~4 characters per token, so this deliberately errs high."""
    return math.ceil(len(text) / 3.5)


class AnthropicClient:
    def __init__(
        self, model: str, max_tokens: int = 16000, client: anthropic.Anthropic | None = None
    ) -> None:
        if client is None:
            _require_paid_opt_in("Anthropic")
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


class OpenAIClient:
    """OpenAI Chat Completions with strict JSON-schema output. Not run in this
    project (zero API spend); covered by fixture tests only. Needs the `openai`
    package and CQA_ALLOW_PAID_CALLS=1, and a price added to PRICES."""

    def __init__(self, model: str, client: Any = None) -> None:
        if client is None:
            _require_paid_opt_in("OpenAI")
            import openai  # type: ignore[import-not-found]

            client = openai.OpenAI()
        self.model = model
        self._client = client

    def cache_params(self) -> dict[str, Any]:
        return {"provider": "openai"}

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> Completion:
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            },
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        choice = resp.choices[0]
        return Completion(
            text=choice.message.content or "",
            usage=Usage(resp.usage.prompt_tokens, resp.usage.completion_tokens),
            model=resp.model,
            stop_reason=str(choice.finish_reason),
            latency_ms=latency_ms,
        )


class OllamaClient:
    """A local open-weight model served by Ollama (free; nothing leaves the machine).

    temperature 0 and a fixed seed for repeatable outputs. num_ctx is set
    explicitly: Ollama's default context window is small and it silently drops
    the start of longer prompts, which would cut off excerpts without any error.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        num_ctx: int = 8192,
        timeout_s: float = 600.0,
    ) -> None:
        self.name = model.removeprefix(OLLAMA_PREFIX)
        self.model = OLLAMA_PREFIX + self.name
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s

    def options(self) -> dict[str, Any]:
        return {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx}

    def cache_params(self) -> dict[str, Any]:
        return {"provider": "ollama", "options": self.options()}

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> Completion:
        import urllib.request

        body = {
            "model": self.name,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "options": self.options(),
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            d = json.loads(r.read())
        latency_ms = (time.perf_counter() - t0) * 1000
        prompt_tokens = int(d.get("prompt_eval_count", 0))
        if prompt_tokens >= self.num_ctx:
            raise RuntimeError(f"prompt filled the {self.num_ctx}-token context; it may have been truncated")
        return Completion(
            text=d["message"]["content"],
            usage=Usage(prompt_tokens, int(d.get("eval_count", 0))),
            model=self.model,
            stop_reason=str(d.get("done_reason", "")),
            latency_ms=latency_ms,
        )


def make_client(model: str) -> ModelClient:
    """'ollama:qwen2.5:7b' -> local; 'claude-*' -> Anthropic; 'gpt-*' / 'o*' -> OpenAI."""
    if is_local(model):
        return OllamaClient(model)
    if model.startswith("claude-"):
        return AnthropicClient(model)
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return OpenAIClient(model)
    raise ValueError(f"unknown model {model!r}")


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
