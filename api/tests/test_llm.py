from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from contract_qa.llm import (
    AnthropicClient,
    CachedClient,
    CacheMiss,
    Completion,
    Usage,
    cost_usd,
    estimate_tokens,
)

SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}


class StubMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text='{"x": "hi"}')],
            usage=SimpleNamespace(input_tokens=120, output_tokens=30),
            model=kwargs["model"],
            stop_reason="end_turn",
        )


def test_anthropic_adapter_sends_schema_and_reads_text_and_usage() -> None:
    stub = StubMessages()
    client = AnthropicClient("claude-haiku-4-5", client=SimpleNamespace(messages=stub))  # type: ignore[arg-type]
    c = client.complete("sys", "user text", SCHEMA)
    (call,) = stub.calls
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "user text"}]
    assert call["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert "fallbacks" not in call  # see AnthropicClient.complete
    assert c.text == '{"x": "hi"}' and c.usage == Usage(120, 30) and c.stop_reason == "end_turn"


class CountingClient:
    model = "claude-haiku-4-5"

    def __init__(self) -> None:
        self.n = 0

    def cache_params(self) -> dict[str, Any]:
        return {"max_tokens": 100}

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> Completion:
        self.n += 1
        return Completion(f"answer {self.n}", Usage(10, 5), self.model, "end_turn", 12.5)


def test_cache_hits_disk_on_second_call(tmp_path: Path) -> None:
    inner = CountingClient()
    cached = CachedClient(inner, tmp_path)
    first = cached.complete("s", "u", SCHEMA)
    second = cached.complete("s", "u", SCHEMA)
    assert inner.n == 1
    assert not first.cached and second.cached
    assert second.text == first.text and second.usage == first.usage
    # A fresh wrapper (new process) reads the same file.
    assert CachedClient(CountingClient(), tmp_path).complete("s", "u", SCHEMA).cached


def test_cache_key_depends_on_every_input(tmp_path: Path) -> None:
    c = CachedClient(CountingClient(), tmp_path)
    base = c.key("s", "u", SCHEMA)
    assert c.key("s2", "u", SCHEMA) != base
    assert c.key("s", "u2", SCHEMA) != base
    assert c.key("s", "u", {**SCHEMA, "required": []}) != base
    other_model = CountingClient()
    other_model.model = "claude-opus-5-5"
    assert CachedClient(other_model, tmp_path).key("s", "u", SCHEMA) != base


def test_offline_mode_never_calls_the_model(tmp_path: Path) -> None:
    inner = CountingClient()
    with pytest.raises(CacheMiss):
        CachedClient(inner, tmp_path, offline=True).complete("s", "u", SCHEMA)
    assert inner.n == 0


def test_cost_and_estimates() -> None:
    assert cost_usd("claude-opus-5-5", Usage(1_000_000, 100_000)) == pytest.approx(4.0 + 2.0)
    assert cost_usd("claude-haiku-4-5", Usage(2000, 500)) == pytest.approx(0.0045)
    with pytest.raises(KeyError):
        cost_usd("unknown-model", Usage(1, 1))
    assert estimate_tokens("x" * 35) == 10
