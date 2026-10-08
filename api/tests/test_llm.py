import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from contract_qa.llm import (
    AnthropicClient,
    CachedClient,
    CacheMiss,
    Completion,
    OllamaClient,
    OpenAIClient,
    PaidCallsDisabled,
    Usage,
    cost_usd,
    estimate_tokens,
    make_client,
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


# --- fixture-based adapter tests (no network, no spend) ---------------------------

FIXTURES = Path(__file__).parent / "fixtures"


def _ns(obj: Any) -> Any:
    """JSON -> attribute access, mimicking SDK response objects."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def test_anthropic_adapter_with_fixture_response() -> None:
    resp = _ns(_fixture("anthropic_message.json"))
    stub = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp))
    c = AnthropicClient("claude-haiku-4-5", client=stub).complete("s", "u", SCHEMA)  # type: ignore[arg-type]
    assert json.loads(c.text)["citations"][0]["chunk_id"] == "C1"
    assert c.usage == Usage(61, 44) and c.stop_reason == "end_turn"


def test_openai_adapter_requests_strict_schema_and_reads_fixture() -> None:
    calls: list[dict[str, Any]] = []

    def create(**kw: Any) -> Any:
        calls.append(kw)
        return _ns(_fixture("openai_chat_completion.json"))

    stub = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    c = OpenAIClient("gpt-fixture", client=stub).complete("sys", "user", SCHEMA)
    (call,) = calls
    assert call["messages"][0] == {"role": "system", "content": "sys"}
    assert call["response_format"]["json_schema"] == {"name": "answer", "schema": SCHEMA, "strict": True}
    assert json.loads(c.text)["status"] == "not_found"
    assert c.usage == Usage(58, 21) and c.stop_reason == "stop"


def test_ollama_adapter_with_recorded_response(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def read(self) -> bytes:
            return (FIXTURES / "ollama_chat.json").read_bytes()

    def fake_urlopen(req: Any, timeout: float) -> FakeResponse:
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data)
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OllamaClient("ollama:qwen2.5:7b", base_url="http://localhost:11434")
    c = client.complete("sys", "user", SCHEMA)
    assert sent["url"] == "http://localhost:11434/api/chat"
    assert sent["body"]["model"] == "qwen2.5:7b" and sent["body"]["format"] == SCHEMA
    assert sent["body"]["options"] == {"temperature": 0, "seed": 0, "num_ctx": 8192}
    assert c.model == "ollama:qwen2.5:7b" and c.usage == Usage(50, 43)
    assert json.loads(c.text)["citations"][0]["quote"].startswith("The term of this Agreement")


def test_paid_clients_refuse_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CQA_ALLOW_PAID_CALLS", raising=False)
    for model in ("claude-haiku-4-5", "gpt-fixture"):
        with pytest.raises(PaidCallsDisabled):
            make_client(model)


def test_local_models_are_free_and_routed_to_ollama() -> None:
    assert isinstance(make_client("ollama:llama3.1:8b"), OllamaClient)
    assert cost_usd("ollama:llama3.1:8b", Usage(10**9, 10**9)) == 0.0
    with pytest.raises(ValueError):
        make_client("mystery-model")
