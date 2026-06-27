# tests/test_integration_phase1.py
from fastapi.testclient import TestClient
from agentfactory.config import FactoryConfig, ProxyConfig, ModelConfig, BackendConfig
from agentfactory import proxy as proxy_mod
from agentfactory.proxy import create_app


def _config():
    return FactoryConfig(
        proxy=ProxyConfig(passthrough_models=["claude-*"]),
        backends={"b": BackendConfig(engine="vllm", base_url="http://be/v1", api_key="k")},
        models={"local/qwen3-coder": ModelConfig(alias="local/qwen3-coder",
                backend="b", model="Qwen/X", guided_decoding=True)},
    )


def test_full_tool_call_roundtrip(monkeypatch):
    # The backend "decides" to call a tool; assert Claude Code sees a tool_use.
    async def fake_call(base_url, api_key, payload):
        # the proxy must have translated tools into OpenAI function form
        assert payload["tools"][0]["function"]["name"] == "write_file"
        assert payload["tool_choice"] == "auto"
        return 200, {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "call_9", "type": "function", "function": {
                "name": "write_file",
                "arguments": '{"path": "a.py", "contents": "x = 1\\n"}'}}]},
            "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 12}}
    monkeypatch.setattr(proxy_mod, "call_openai_backend", fake_call)

    client = TestClient(create_app(_config()))
    resp = client.post("/v1/messages", json={
        "model": "local/qwen3-coder",
        "max_tokens": 512,
        "system": "You write files.",
        "messages": [{"role": "user", "content": "create a.py with x=1"}],
        "tools": [{"name": "write_file", "description": "write a file",
                   "input_schema": {"type": "object", "properties": {
                       "path": {"type": "string"},
                       "contents": {"type": "string"}}}}],
    })

    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "tool_use"
    tool = body["content"][0]
    assert tool["type"] == "tool_use"
    assert tool["name"] == "write_file"
    assert tool["input"] == {"path": "a.py", "contents": "x = 1\n"}
    assert body["usage"]["input_tokens"] == 20
