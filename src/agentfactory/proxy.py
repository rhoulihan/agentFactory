from __future__ import annotations

from fastapi import FastAPI

from .config import FactoryConfig


def create_app(config: FactoryConfig) -> FastAPI:
    app = FastAPI(title="agentFactory proxy")
    app.state.config = config

    @app.get("/v1/models")
    async def list_models() -> dict:
        return {"data": [{"type": "model", "id": name}
                         for name in config.discovery_models()]}

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "models": list(config.models.keys())}

    return app
