from fastapi.testclient import TestClient
from agentfactory.config import FactoryConfig, ProxyConfig, ModelConfig, BackendConfig
from agentfactory.proxy import create_app


def _config():
    return FactoryConfig(
        proxy=ProxyConfig(passthrough_models=["claude-*"]),
        backends={"b": BackendConfig(engine="vllm", base_url="http://be/v1", api_key="k")},
        models={"local/qwen3-coder": ModelConfig(alias="local/qwen3-coder",
                backend="b", model="Qwen/X")},
    )


def test_discovery_lists_local_alias():
    client = TestClient(create_app(_config()))
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "local/qwen3-coder" in ids
