# src/agentfactory/serve.py
from __future__ import annotations

import os
import shutil
from urllib.parse import urlparse

from .config import BackendConfig, FactoryConfig, ModelConfig

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0"}


def port_from_base_url(base_url: str) -> int:
    port = urlparse(base_url).port
    if port is None:
        raise ValueError(f"no port in base_url: {base_url}")
    return port


def is_local_host(base_url: str) -> bool:
    return (urlparse(base_url).hostname or "") in _LOCAL_HOSTS


def build_vllm_argv(mc: ModelConfig, backend: BackendConfig) -> list[str]:
    if not mc.tool_parser:
        raise ValueError(
            f"model '{mc.alias or mc.model}' has no tool_parser; tool calling "
            "would silently break. Set tool_parser in factory.yaml."
        )
    argv = [
        "vllm", "serve", mc.model,
        "--host", "127.0.0.1",
        "--port", str(port_from_base_url(backend.base_url)),
        "--enable-auto-tool-choice",
        "--tool-call-parser", mc.tool_parser,
    ]
    if mc.guided_decoding:
        argv += ["--guided-decoding-backend", "xgrammar"]
    if mc.context:
        argv += ["--max-model-len", str(mc.context)]
    return argv


def serve_vllm(config: FactoryConfig, alias: str, dry_run: bool = False) -> int:
    mc = config.resolve_model(alias)
    if mc is None:
        print(f"unknown model alias: {alias}")
        return 1
    backend = config.backend_for(mc)
    try:
        argv = build_vllm_argv(mc, backend)
    except ValueError as exc:
        print(str(exc))
        return 1
    cmd = " ".join(argv)

    if dry_run or not is_local_host(backend.base_url):
        if not is_local_host(backend.base_url):
            print(f"# backend '{mc.backend}' is remote — run this on that host:")
        print(cmd)
        return 0

    if shutil.which("vllm") is None:
        print("vllm not found on PATH. Install vLLM, then re-run. Command would be:")
        print(cmd)
        return 1

    os.execvp("vllm", argv)  # replaces this process
    return 0  # unreachable, but keeps the type checker happy
