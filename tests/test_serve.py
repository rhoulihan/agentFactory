# tests/test_serve.py
import pytest
from agentfactory.config import ModelConfig, BackendConfig, FactoryConfig, ProxyConfig
from agentfactory.serve import (
    build_vllm_argv, port_from_base_url, is_local_host, serve_vllm,
)

BACKEND = BackendConfig(engine="vllm", base_url="http://127.0.0.1:8000/v1", api_key="vllm")
MC = ModelConfig(alias="local/qwen3-coder", backend="b",
                 model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                 tool_parser="qwen3_coder", guided_decoding=True, context=65536)


def test_port_from_base_url():
    assert port_from_base_url("http://127.0.0.1:8000/v1") == 8000


def test_is_local_host():
    assert is_local_host("http://127.0.0.1:8000/v1") is True
    assert is_local_host("http://10.0.0.20:8000/v1") is False


def test_build_vllm_argv_full():
    argv = build_vllm_argv(MC, BACKEND)
    assert argv == [
        "vllm", "serve", "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "--host", "127.0.0.1", "--port", "8000",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--guided-decoding-backend", "xgrammar",
        "--max-model-len", "65536",
    ]


def test_build_vllm_argv_no_guided_decoding_omits_flag():
    mc = MC.model_copy(update={"guided_decoding": False})
    argv = build_vllm_argv(mc, BACKEND)
    assert "--guided-decoding-backend" not in argv
    assert "--tool-call-parser" in argv


def test_build_vllm_argv_requires_tool_parser():
    mc = MC.model_copy(update={"tool_parser": None})
    with pytest.raises(ValueError, match="tool_parser"):
        build_vllm_argv(mc, BACKEND)


def _config():
    return FactoryConfig(
        proxy=ProxyConfig(),
        backends={"b": BACKEND, "remote": BackendConfig(
            engine="vllm", base_url="http://10.0.0.20:8000/v1", api_key="x")},
        models={
            "local/qwen3-coder": MC,
            "local/remote-model": MC.model_copy(update={"backend": "remote"}),
        },
    )


def test_serve_dry_run_prints_and_returns_zero(capsys):
    rc = serve_vllm(_config(), "local/qwen3-coder", dry_run=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct" in out
    assert "--tool-call-parser qwen3_coder" in out


def test_serve_remote_backend_prints_not_launches(capsys):
    rc = serve_vllm(_config(), "local/remote-model", dry_run=False)
    assert rc == 0
    assert "vllm serve" in capsys.readouterr().out


def test_serve_unknown_alias_returns_one():
    assert serve_vllm(_config(), "local/nope", dry_run=True) == 1


def test_serve_missing_tool_parser_returns_one(capsys):
    cfg = _config()
    cfg.models["local/noparser"] = MC.model_copy(update={"tool_parser": None})
    rc = serve_vllm(cfg, "local/noparser", dry_run=True)
    assert rc == 1
    assert "tool_parser" in capsys.readouterr().out
