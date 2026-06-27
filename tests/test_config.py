# tests/test_config.py
from pathlib import Path
from agentfactory.config import FactoryConfig

CONFIG = """
proxy:
  host: 127.0.0.1
  port: 8787
  upstream_anthropic: https://api.anthropic.com
  passthrough_models: ["claude-*"]
backends:
  local-nvidia:
    engine: vllm
    base_url: http://127.0.0.1:8000/v1
    api_key: vllm
models:
  local/qwen3-coder:
    backend: local-nvidia
    model: Qwen/Qwen3-Coder-30B-A3B-Instruct
    tool_parser: qwen3_coder
    guided_decoding: true
    context: 65536
"""

def _cfg(tmp_path: Path) -> FactoryConfig:
    p = tmp_path / "factory.yaml"
    p.write_text(CONFIG)
    return FactoryConfig.load(p)

def test_load_parses_sections(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.proxy.port == 8787
    assert cfg.backends["local-nvidia"].base_url == "http://127.0.0.1:8000/v1"
    assert cfg.models["local/qwen3-coder"].tool_parser == "qwen3_coder"

def test_is_passthrough_matches_glob(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.is_passthrough("claude-opus-4-8") is True
    assert cfg.is_passthrough("local/qwen3-coder") is False

def test_resolve_model(tmp_path):
    cfg = _cfg(tmp_path)
    mc = cfg.resolve_model("local/qwen3-coder")
    assert mc is not None and mc.model == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert cfg.resolve_model("local/nope") is None

def test_backend_for(tmp_path):
    cfg = _cfg(tmp_path)
    mc = cfg.resolve_model("local/qwen3-coder")
    assert cfg.backend_for(mc).engine == "vllm"

def test_discovery_lists_passthrough_and_aliases(tmp_path):
    cfg = _cfg(tmp_path)
    models = cfg.discovery_models()
    assert "local/qwen3-coder" in models
    # globbed passthrough entries are not emitted as concrete model ids
    assert "claude-*" not in models
