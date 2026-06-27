from fastapi.testclient import TestClient
from agentfactory.config import FactoryConfig, ModelConfig, BackendConfig
from agentfactory.proxy import create_app


def test_healthz():
    cfg = FactoryConfig(
        backends={"b": BackendConfig(engine="vllm", base_url="http://be/v1", api_key="k")},
        models={"local/x": ModelConfig(alias="local/x", backend="b", model="m")},
    )
    client = TestClient(create_app(cfg))
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "local/x" in resp.json()["models"]
