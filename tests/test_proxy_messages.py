import json
import pytest
from fastapi.testclient import TestClient
from agentfactory.config import FactoryConfig, ProxyConfig, ModelConfig, BackendConfig
from agentfactory import proxy as proxy_mod
from agentfactory.proxy import create_app


def _config():
    return FactoryConfig(
        proxy=ProxyConfig(passthrough_models=["claude-*"],
                          upstream_anthropic="https://api.anthropic.com"),
        backends={"b": BackendConfig(engine="vllm", base_url="http://be/v1", api_key="k")},
        models={"local/x": ModelConfig(alias="local/x", backend="b",
                model="Qwen/X", guided_decoding=True)},
    )


def test_unknown_model_returns_anthropic_error():
    client = TestClient(create_app(_config()))
    resp = client.post("/v1/messages", json={"model": "mystery", "messages": []})
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found_error"


def test_local_non_streaming_translates_both_ways(monkeypatch):
    async def fake_call(base_url, api_key, payload):
        assert base_url == "http://be/v1"
        assert payload["model"] == "Qwen/X"
        return 200, {"choices": [{"message": {"content": "done"},
                                  "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 2, "completion_tokens": 1}}
    monkeypatch.setattr(proxy_mod, "call_openai_backend", fake_call)

    client = TestClient(create_app(_config()))
    resp = client.post("/v1/messages", json={
        "model": "local/x", "max_tokens": 50,
        "messages": [{"role": "user", "content": "go"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == [{"type": "text", "text": "done"}]
    assert body["model"] == "local/x"


def test_local_streaming_emits_anthropic_sse(monkeypatch):
    async def fake_stream(base_url, api_key, payload):
        for c in [
            {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]:
            yield c
    monkeypatch.setattr(proxy_mod, "stream_openai_backend", fake_stream)

    client = TestClient(create_app(_config()))
    with client.stream("POST", "/v1/messages", json={
        "model": "local/x", "stream": True,
        "messages": [{"role": "user", "content": "go"}]}) as resp:
        text = "".join(resp.iter_text())
    assert "event: message_start" in text
    assert "event: content_block_delta" in text
    assert "event: message_stop" in text


def test_passthrough_forwards_upstream(monkeypatch):
    captured = {}

    async def fake_upstream(method, url, headers, content):
        captured["url"] = url
        captured["body"] = json.loads(content)

        async def body():
            yield b'{"type":"message","role":"assistant"}'
        return 200, {"content-type": "application/json"}, body()
    monkeypatch.setattr(proxy_mod, "stream_upstream", fake_upstream)

    client = TestClient(create_app(_config()))
    resp = client.post("/v1/messages",
                       json={"model": "claude-opus-4-8", "messages": []},
                       headers={"x-api-key": "secret"})
    assert resp.status_code == 200
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["body"]["model"] == "claude-opus-4-8"
