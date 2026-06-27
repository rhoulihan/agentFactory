# tests/test_smoke.py
import httpx
import respx
from agentfactory.config import FactoryConfig, ProxyConfig, ModelConfig, BackendConfig
from agentfactory.smoke import smoke_test_model


def _config():
    return FactoryConfig(
        proxy=ProxyConfig(host="127.0.0.1", port=8787),
        backends={"b": BackendConfig(engine="vllm", base_url="http://be/v1", api_key="k")},
        models={"local/x": ModelConfig(alias="local/x", backend="b", model="m")},
    )


@respx.mock
async def test_smoke_passes_when_tool_use_returned():
    respx.post("http://127.0.0.1:8787/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "content": [{"type": "tool_use", "id": "t1",
                         "name": "factory_probe", "input": {"ok": True}}],
            "stop_reason": "tool_use"}))
    passed, detail = await smoke_test_model(_config(), "local/x")
    assert passed is True


@respx.mock
async def test_smoke_fails_when_no_tool_use():
    respx.post("http://127.0.0.1:8787/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "content": [{"type": "text", "text": "I won't call it"}],
            "stop_reason": "end_turn"}))
    passed, detail = await smoke_test_model(_config(), "local/x")
    assert passed is False
    assert "no tool_use" in detail
