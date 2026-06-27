# src/agentfactory/config.py
from __future__ import annotations

import fnmatch
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class BackendConfig(BaseModel):
    engine: str
    base_url: str
    api_key: str = "x"


class ModelConfig(BaseModel):
    alias: str = ""
    backend: str
    model: str
    tool_parser: str | None = None
    guided_decoding: bool = False
    context: int = 32768


class ProxyConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8787
    upstream_anthropic: str = "https://api.anthropic.com"
    passthrough_models: list[str] = Field(default_factory=lambda: ["claude-*"])


class FactoryConfig(BaseModel):
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    backends: dict[str, BackendConfig] = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "FactoryConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        cfg = cls.model_validate(data)
        for alias, mc in cfg.models.items():
            mc.alias = alias
        return cfg

    def is_passthrough(self, model: str) -> bool:
        return any(fnmatch.fnmatch(model, pat) for pat in self.proxy.passthrough_models)

    def resolve_model(self, model: str) -> ModelConfig | None:
        return self.models.get(model)

    def backend_for(self, mc: ModelConfig) -> BackendConfig:
        return self.backends[mc.backend]

    def discovery_models(self) -> list[str]:
        literal = [m for m in self.proxy.passthrough_models if not any(c in m for c in "*?[")]
        return literal + list(self.models.keys())
