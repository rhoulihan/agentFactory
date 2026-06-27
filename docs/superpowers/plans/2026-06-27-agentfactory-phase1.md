# agentFactory Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the core routing proxy + CLI that lets a Claude Code orchestrator (Anthropic) dispatch a local-model subagent that generates code in an isolated worktree, with all traffic flowing through one Anthropic-Messages-API proxy that routes by model name.

**Architecture:** A FastAPI proxy sits at `ANTHROPIC_BASE_URL`. It reads each request's `model` field: `claude-*` is forwarded verbatim to `api.anthropic.com` (passthrough); a registered `local/*` alias is translated Anthropic↔OpenAI and forwarded to a local vLLM (OpenAI-compatible) backend, then translated back — including SSE streaming. A `factory` CLI runs the proxy daemon and verifies backends with a live tool-call smoke test. A `local-coder` subagent template plus a worktree review convention complete the loop.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, httpx, pydantic v2, PyYAML, Typer. Tooling: `uv`, pytest, pytest-asyncio, respx.

## Global Constraints

- Python **3.11+** (dev/CI on 3.12). Manage env and deps with `uv`.
- Translation is **hand-rolled** — do NOT depend on LiteLLM/claude-code-router for the wire mapping.
- **Core invariant:** `local/*` aliases are generation-only; `claude-*` always takes the passthrough path. Review/verification is never routed to a local backend.
- **`tool_use` ↔ `tool_calls` round-tripping is load-bearing** — it is the agent loop. Test it explicitly, including streamed tool-argument deltas.
- **Guided decoding is mandatory** when tools are present on a local request (the proxy must ensure `tool_choice` is set so vLLM engages its tool parser).
- **Never log credentials** (`x-api-key`, `Authorization`, backend `api_key`).
- Default proxy bind: `127.0.0.1:8787`. Default upstream: `https://api.anthropic.com`. Default backend: vLLM OpenAI-compatible at a configured `base_url` ending in `/v1`.
- All public functions are typed. Tests are written first (TDD) and committed with their implementation.

---

## File Structure

```
agentFactory/
  pyproject.toml                     # uv project + deps + pytest config
  factory.example.yaml               # sample registry
  src/agentfactory/
    __init__.py
    errors.py                        # Anthropic-shaped error dict helper
    config.py                        # pydantic models + loader + model resolution
    translate.py                     # non-streaming Anthropic<->OpenAI mapping
    streaming.py                     # OpenAI chunk stream -> Anthropic SSE events
    backends.py                      # httpx calls: passthrough + openai backend
    proxy.py                         # FastAPI app: /v1/messages, /v1/models, /healthz
    smoke.py                         # live tool-call smoke test
    cli.py                           # `factory` CLI: up/down/status/doctor
  templates/agents/local-coder.md    # installed into a project's .claude/agents/
  tests/
    test_config.py
    test_errors.py
    test_translate_request.py
    test_translate_response.py
    test_streaming.py
    test_backends.py
    test_proxy_discovery.py
    test_proxy_messages.py
    test_proxy_health.py
    test_smoke.py
    test_cli.py
```

Responsibilities: `config` owns the registry + routing decisions (passthrough vs alias). `translate` and `streaming` are pure mapping logic with no I/O. `backends` is the only module that does outbound HTTP. `proxy` wires them into routes. `smoke`/`cli` are operator tooling. Each is independently testable.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/agentfactory/__init__.py`
- Create: `tests/test_smoke_import.py` (temporary sanity test)

**Interfaces:**
- Produces: an installed package `agentfactory` importable in tests; `pytest` runnable via `uv run pytest`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke_import.py
def test_package_imports():
    import agentfactory
    assert agentfactory.__version__ == "0.1.0"
```

- [ ] **Step 2: Create the package init**

```python
# src/agentfactory/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 3: Create pyproject.toml**

```toml
[project]
name = "agentfactory"
version = "0.1.0"
description = "Orchestrate local-model subagents from a Claude Code parent process"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "httpx>=0.27",
    "pydantic>=2.7",
    "pyyaml>=6.0",
    "typer>=0.12",
]

[project.scripts]
factory = "agentfactory.cli:app"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/agentfactory"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 4: Install and run the test**

Run: `uv sync && uv run pytest tests/test_smoke_import.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/agentfactory/__init__.py tests/test_smoke_import.py uv.lock
git commit -m "chore: scaffold agentfactory package with uv + pytest"
```

---

### Task 2: Anthropic-shaped error helper

**Files:**
- Create: `src/agentfactory/errors.py`
- Test: `tests/test_errors.py`

**Interfaces:**
- Produces: `anthropic_error(message: str, err_type: str = "invalid_request_error") -> dict` returning `{"type": "error", "error": {"type": ..., "message": ...}}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_errors.py
from agentfactory.errors import anthropic_error

def test_anthropic_error_shape():
    err = anthropic_error("no such model: local/foo", "not_found_error")
    assert err == {
        "type": "error",
        "error": {"type": "not_found_error", "message": "no such model: local/foo"},
    }

def test_anthropic_error_default_type():
    assert anthropic_error("bad")["error"]["type"] == "invalid_request_error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentfactory.errors'`

- [ ] **Step 3: Implement**

```python
# src/agentfactory/errors.py
def anthropic_error(message: str, err_type: str = "invalid_request_error") -> dict:
    return {"type": "error", "error": {"type": err_type, "message": message}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_errors.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentfactory/errors.py tests/test_errors.py
git commit -m "feat: Anthropic-shaped error helper"
```

---

### Task 3: Config models, loader, and routing decisions

**Files:**
- Create: `src/agentfactory/config.py`
- Create: `factory.example.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `BackendConfig(engine: str, base_url: str, api_key: str)`
  - `ModelConfig(alias: str, backend: str, model: str, tool_parser: str | None, guided_decoding: bool, context: int)`
  - `ProxyConfig(host: str, port: int, upstream_anthropic: str, passthrough_models: list[str])`
  - `FactoryConfig` with fields `proxy`, `backends: dict[str, BackendConfig]`, `models: dict[str, ModelConfig]` and methods:
    - `load(path: str | Path) -> FactoryConfig` (classmethod)
    - `is_passthrough(model: str) -> bool` — true if `model` matches any `passthrough_models` glob
    - `resolve_model(model: str) -> ModelConfig | None` — alias lookup
    - `backend_for(mc: ModelConfig) -> BackendConfig`
    - `discovery_models() -> list[str]` — passthrough names (literal, non-glob) + all alias names

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentfactory.config'`

- [ ] **Step 3: Implement config.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Create the example config**

```yaml
# factory.example.yaml
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
  # remote-ubuntu:
  #   engine: vllm
  #   base_url: http://10.0.0.20:8000/v1
  #   api_key: ${REMOTE_VLLM_KEY}

models:
  local/qwen3-coder:
    backend: local-nvidia
    model: Qwen/Qwen3-Coder-30B-A3B-Instruct
    tool_parser: qwen3_coder
    guided_decoding: true
    context: 65536
```

- [ ] **Step 6: Commit**

```bash
git add src/agentfactory/config.py factory.example.yaml tests/test_config.py
git commit -m "feat: config registry with passthrough/alias routing decisions"
```

---

### Task 4: Translation — Anthropic request → OpenAI request

**Files:**
- Create: `src/agentfactory/translate.py`
- Test: `tests/test_translate_request.py`

**Interfaces:**
- Consumes: `ModelConfig` from `config.py`.
- Produces: `anthropic_to_openai_request(body: dict, mc: ModelConfig) -> dict`. Maps `system` → leading system message; content blocks `text`/`tool_use`/`tool_result` → OpenAI messages; Anthropic `tools` → OpenAI function tools; `stop_sequences` → `stop`; sets `model` to `mc.model`. When `tools` present and `mc.guided_decoding`, ensures `tool_choice` is set (default `"auto"`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_translate_request.py
from agentfactory.config import ModelConfig
from agentfactory.translate import anthropic_to_openai_request

MC = ModelConfig(alias="local/x", backend="b", model="Qwen/X", guided_decoding=True)

def test_system_lifted_to_message():
    body = {"model": "local/x", "max_tokens": 100, "system": "be brief",
            "messages": [{"role": "user", "content": "hi"}]}
    out = anthropic_to_openai_request(body, MC)
    assert out["model"] == "Qwen/X"
    assert out["messages"][0] == {"role": "system", "content": "be brief"}
    assert out["messages"][1] == {"role": "user", "content": "hi"}
    assert out["max_tokens"] == 100

def test_system_block_list_concatenated():
    body = {"messages": [], "system": [{"type": "text", "text": "a"},
                                       {"type": "text", "text": "b"}]}
    out = anthropic_to_openai_request(body, MC)
    assert out["messages"][0] == {"role": "system", "content": "a\nb"}

def test_tool_use_and_result_roundtrip_into_openai():
    body = {"messages": [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "bash",
             "input": {"command": "ls"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.txt"}]},
    ]}
    out = anthropic_to_openai_request(body, MC)
    asst = out["messages"][0]
    assert asst["role"] == "assistant"
    assert asst["tool_calls"][0]["id"] == "toolu_1"
    assert asst["tool_calls"][0]["function"]["name"] == "bash"
    assert asst["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'
    tool_msg = out["messages"][1]
    assert tool_msg == {"role": "tool", "tool_call_id": "toolu_1", "content": "a.txt"}

def test_tools_mapped_and_tool_choice_defaulted():
    body = {"messages": [{"role": "user", "content": "go"}],
            "tools": [{"name": "bash", "description": "run",
                       "input_schema": {"type": "object",
                                        "properties": {"command": {"type": "string"}}}}]}
    out = anthropic_to_openai_request(body, MC)
    fn = out["tools"][0]["function"]
    assert out["tools"][0]["type"] == "function"
    assert fn["name"] == "bash"
    assert fn["parameters"]["properties"]["command"]["type"] == "string"
    assert out["tool_choice"] == "auto"

def test_stop_sequences_mapped():
    body = {"messages": [], "stop_sequences": ["STOP"]}
    out = anthropic_to_openai_request(body, MC)
    assert out["stop"] == ["STOP"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_translate_request.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentfactory.translate'`

- [ ] **Step 3: Implement the request side of translate.py**

```python
# src/agentfactory/translate.py
from __future__ import annotations

import json

from .config import ModelConfig


def _system_to_text(system) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    return "\n".join(b.get("text", "") for b in system if b.get("type") == "text")


def _result_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def _convert_message(msg: dict) -> list[dict]:
    """Convert one Anthropic message into one or more OpenAI messages."""
    role = msg["role"]
    content = msg.get("content")
    if isinstance(content, str):
        return [{"role": role, "content": content}]

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    for block in content or []:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block.get("input", {})),
                },
            })
        elif btype == "tool_result":
            tool_results.append({
                "role": "tool",
                "tool_call_id": block["tool_use_id"],
                "content": _result_content_to_text(block.get("content", "")),
            })

    out: list[dict] = []
    if role == "assistant" and tool_calls:
        out.append({
            "role": "assistant",
            "content": "\n".join(text_parts) or None,
            "tool_calls": tool_calls,
        })
    elif text_parts:
        out.append({"role": role, "content": "\n".join(text_parts)})
    out.extend(tool_results)
    return out


def _convert_tools(tools: list[dict]) -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
        },
    } for t in tools]


def anthropic_to_openai_request(body: dict, mc: ModelConfig) -> dict:
    messages: list[dict] = []
    system = _system_to_text(body.get("system"))
    if system is not None:
        messages.append({"role": "system", "content": system})
    for msg in body.get("messages", []):
        messages.extend(_convert_message(msg))

    payload: dict = {"model": mc.model, "messages": messages}
    if "max_tokens" in body:
        payload["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        payload["temperature"] = body["temperature"]
    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]

    tools = body.get("tools")
    if tools:
        payload["tools"] = _convert_tools(tools)
        if "tool_choice" in body and isinstance(body["tool_choice"], dict):
            tc = body["tool_choice"]
            if tc.get("type") == "tool" and tc.get("name"):
                payload["tool_choice"] = {"type": "function",
                                          "function": {"name": tc["name"]}}
            elif tc.get("type") == "any":
                payload["tool_choice"] = "required"
            else:
                payload["tool_choice"] = "auto"
        elif mc.guided_decoding:
            payload["tool_choice"] = "auto"
    return payload
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_translate_request.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentfactory/translate.py tests/test_translate_request.py
git commit -m "feat: translate Anthropic request -> OpenAI request"
```

---

### Task 5: Translation — OpenAI response → Anthropic response

**Files:**
- Modify: `src/agentfactory/translate.py`
- Test: `tests/test_translate_response.py`

**Interfaces:**
- Produces: `openai_to_anthropic_response(resp: dict, requested_model: str) -> dict`. Maps `choices[0].message.content` → `text` block; `message.tool_calls[]` → `tool_use` blocks (parse `arguments` JSON → dict); `finish_reason` → `stop_reason` via `map_stop_reason`; `usage` → `input_tokens`/`output_tokens`.
- Produces: `map_stop_reason(finish_reason: str | None) -> str` with `tool_calls→tool_use`, `length→max_tokens`, `stop→end_turn`, default `end_turn`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_translate_response.py
from agentfactory.translate import openai_to_anthropic_response, map_stop_reason

def test_map_stop_reason():
    assert map_stop_reason("tool_calls") == "tool_use"
    assert map_stop_reason("length") == "max_tokens"
    assert map_stop_reason("stop") == "end_turn"
    assert map_stop_reason(None) == "end_turn"

def test_text_response():
    resp = {"choices": [{"message": {"role": "assistant", "content": "hello"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    out = openai_to_anthropic_response(resp, "local/x")
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["model"] == "local/x"
    assert out["content"] == [{"type": "text", "text": "hello"}]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert out["id"].startswith("msg_")

def test_tool_call_response_parses_arguments():
    resp = {"choices": [{"message": {"role": "assistant", "content": None,
              "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
              "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9}}
    out = openai_to_anthropic_response(resp, "local/x")
    assert out["stop_reason"] == "tool_use"
    block = out["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "call_1"
    assert block["name"] == "bash"
    assert block["input"] == {"command": "ls"}

def test_text_and_tool_call_both_present():
    resp = {"choices": [{"message": {"role": "assistant", "content": "running it",
              "tool_calls": [{"id": "call_2", "type": "function",
                "function": {"name": "bash", "arguments": "{}"}}]},
              "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    out = openai_to_anthropic_response(resp, "local/x")
    assert out["content"][0] == {"type": "text", "text": "running it"}
    assert out["content"][1]["type"] == "tool_use"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_translate_response.py -v`
Expected: FAIL with `ImportError: cannot import name 'openai_to_anthropic_response'`

- [ ] **Step 3: Append the response side to translate.py**

```python
# --- append to src/agentfactory/translate.py ---
import uuid

_STOP_MAP = {"tool_calls": "tool_use", "length": "max_tokens", "stop": "end_turn"}


def map_stop_reason(finish_reason: str | None) -> str:
    return _STOP_MAP.get(finish_reason or "", "end_turn")


def openai_to_anthropic_response(resp: dict, requested_model: str) -> dict:
    choice = resp["choices"][0]
    message = choice.get("message", {})
    content: list[dict] = []

    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        content.append({
            "type": "tool_use",
            "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
            "name": fn.get("name", ""),
            "input": args,
        })

    usage = resp.get("usage", {})
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": map_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_translate_response.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentfactory/translate.py tests/test_translate_response.py
git commit -m "feat: translate OpenAI response -> Anthropic response"
```

---

### Task 6: Streaming re-framer (OpenAI chunks → Anthropic SSE events)

**Files:**
- Create: `src/agentfactory/streaming.py`
- Test: `tests/test_streaming.py`

**Interfaces:**
- Consumes: `map_stop_reason` from `translate.py`.
- Produces:
  - `async openai_stream_to_anthropic_events(chunks, requested_model) -> AsyncIterator[dict]` where `chunks` is an async iterator of parsed OpenAI chunk dicts. Yields Anthropic event dicts in order: `message_start`, then per-block `content_block_start`/`content_block_delta`/`content_block_stop`, then `message_delta`, `message_stop`. Text deltas use `{"type":"text_delta","text":...}`; tool-arg deltas use `{"type":"input_json_delta","partial_json":...}`.
  - `format_sse(event: dict) -> bytes` → `b"event: <type>\ndata: <json>\n\n"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_streaming.py
import json
from agentfactory.streaming import openai_stream_to_anthropic_events, format_sse


async def _gen(items):
    for it in items:
        yield it


async def _collect(chunks, model="local/x"):
    return [e async for e in openai_stream_to_anthropic_events(_gen(chunks), model)]


def _types(events):
    return [e["type"] for e in events]


async def test_text_stream_event_order():
    chunks = [
        {"choices": [{"delta": {"content": "He"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "llo"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    events = await _collect(chunks)
    assert _types(events) == [
        "message_start", "content_block_start",
        "content_block_delta", "content_block_delta",
        "content_block_stop", "message_delta", "message_stop",
    ]
    deltas = [e["delta"]["text"] for e in events if e["type"] == "content_block_delta"]
    assert deltas == ["He", "llo"]
    assert events[0]["message"]["model"] == "local/x"
    assert events[-2]["delta"]["stop_reason"] == "end_turn"


async def test_tool_call_stream_reassembles_arguments():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "bash", "arguments": ""}}]},
            "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"comm'}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'and": "ls"}'}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    events = await _collect(chunks)
    start = next(e for e in events if e["type"] == "content_block_start")
    assert start["content_block"]["type"] == "tool_use"
    assert start["content_block"]["name"] == "bash"
    assert start["content_block"]["id"] == "call_1"
    partials = "".join(
        e["delta"]["partial_json"] for e in events
        if e["type"] == "content_block_delta"
    )
    assert json.loads(partials) == {"command": "ls"}
    assert events[-2]["delta"]["stop_reason"] == "tool_use"


def test_format_sse():
    raw = format_sse({"type": "message_stop"})
    assert raw == b'event: message_stop\ndata: {"type": "message_stop"}\n\n'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streaming.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentfactory.streaming'`

- [ ] **Step 3: Implement streaming.py**

```python
# src/agentfactory/streaming.py
from __future__ import annotations

import json
import uuid
from typing import AsyncIterator

from .translate import map_stop_reason


def format_sse(event: dict) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode("utf-8")


async def openai_stream_to_anthropic_events(
    chunks: AsyncIterator[dict], requested_model: str
) -> AsyncIterator[dict]:
    yield {
        "type": "message_start",
        "message": {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": requested_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }

    next_index = 0
    text_index: int | None = None
    tool_block_index: dict[int, int] = {}   # openai tool index -> anthropic block index
    open_blocks: set[int] = set()
    finish_reason: str | None = None

    async for chunk in chunks:
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

        if delta.get("content"):
            if text_index is None:
                text_index = next_index
                next_index += 1
                open_blocks.add(text_index)
                yield {"type": "content_block_start", "index": text_index,
                       "content_block": {"type": "text", "text": ""}}
            yield {"type": "content_block_delta", "index": text_index,
                   "delta": {"type": "text_delta", "text": delta["content"]}}

        for tc in delta.get("tool_calls") or []:
            oai_idx = tc.get("index", 0)
            if oai_idx not in tool_block_index:
                idx = next_index
                next_index += 1
                tool_block_index[oai_idx] = idx
                open_blocks.add(idx)
                fn = tc.get("function", {})
                yield {"type": "content_block_start", "index": idx,
                       "content_block": {
                           "type": "tool_use",
                           "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                           "name": fn.get("name", ""),
                           "input": {},
                       }}
            args = (tc.get("function") or {}).get("arguments")
            if args:
                yield {"type": "content_block_delta",
                       "index": tool_block_index[oai_idx],
                       "delta": {"type": "input_json_delta", "partial_json": args}}

    for idx in sorted(open_blocks):
        yield {"type": "content_block_stop", "index": idx}

    yield {"type": "message_delta",
           "delta": {"stop_reason": map_stop_reason(finish_reason),
                     "stop_sequence": None},
           "usage": {"output_tokens": 0}}
    yield {"type": "message_stop"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streaming.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentfactory/streaming.py tests/test_streaming.py
git commit -m "feat: SSE re-framer for OpenAI->Anthropic streaming"
```

---

### Task 7: Backend HTTP clients

**Files:**
- Create: `src/agentfactory/backends.py`
- Test: `tests/test_backends.py`

**Interfaces:**
- Produces:
  - `async stream_upstream(method, url, headers: dict, content: bytes) -> tuple[int, dict, AsyncIterator[bytes]]` — streams an upstream Anthropic response back (status, response headers as dict, byte iterator). Used for passthrough.
  - `async call_openai_backend(base_url, api_key, payload) -> tuple[int, dict]` — POST `{base_url}/chat/completions`, return `(status, json)`.
  - `async stream_openai_backend(base_url, api_key, payload) -> AsyncIterator[dict]` — POST with `stream=True`, yield parsed chunk dicts, stop at `[DONE]`.
  - `filter_request_headers(headers) -> dict` — drop hop-by-hop headers (`host`, `content-length`, `connection`, `accept-encoding`) before forwarding.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backends.py
import httpx
import respx
from agentfactory.backends import (
    call_openai_backend, stream_openai_backend, filter_request_headers,
)


def test_filter_request_headers_drops_hop_by_hop():
    out = filter_request_headers({"Host": "x", "Content-Length": "3",
                                  "x-api-key": "secret", "anthropic-version": "2023-06-01"})
    assert "host" not in {k.lower() for k in out}
    assert "content-length" not in {k.lower() for k in out}
    assert out["x-api-key"] == "secret"
    assert out["anthropic-version"] == "2023-06-01"


@respx.mock
async def test_call_openai_backend():
    route = respx.post("http://be:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
    status, body = await call_openai_backend("http://be:8000/v1", "k", {"model": "m"})
    assert status == 200
    assert body["choices"][0]["message"]["content"] == "ok"
    assert route.calls.last.request.headers["authorization"] == "Bearer k"


@respx.mock
async def test_stream_openai_backend_parses_sse():
    sse = (b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
           b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
           b'data: [DONE]\n\n')
    respx.post("http://be:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=sse,
                                    headers={"content-type": "text/event-stream"}))
    out = [c async for c in stream_openai_backend("http://be:8000/v1", "k", {"stream": True})]
    assert out[0]["choices"][0]["delta"]["content"] == "hi"
    assert out[-1]["choices"][0]["finish_reason"] == "stop"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backends.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentfactory.backends'`

- [ ] **Step 3: Implement backends.py**

```python
# src/agentfactory/backends.py
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

_HOP_BY_HOP = {"host", "content-length", "connection", "accept-encoding"}


def filter_request_headers(headers) -> dict:
    return {k: v for k, v in dict(headers).items() if k.lower() not in _HOP_BY_HOP}


async def stream_upstream(
    method: str, url: str, headers: dict, content: bytes
) -> tuple[int, dict, AsyncIterator[bytes]]:
    client = httpx.AsyncClient(timeout=None)
    req = client.build_request(method, url, headers=headers, content=content)
    resp = await client.send(req, stream=True)

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return resp.status_code, dict(resp.headers), body()


async def call_openai_backend(base_url: str, api_key: str, payload: dict) -> tuple[int, dict]:
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        return resp.status_code, resp.json()


async def stream_openai_backend(
    base_url: str, api_key: str, payload: dict
) -> AsyncIterator[dict]:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if data == "[DONE]":
                    return
                yield json.loads(data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backends.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentfactory/backends.py tests/test_backends.py
git commit -m "feat: backend HTTP clients (passthrough + OpenAI backend)"
```

---

### Task 8: Proxy app — `/v1/models` discovery and `/healthz`

**Files:**
- Create: `src/agentfactory/proxy.py`
- Test: `tests/test_proxy_discovery.py`, `tests/test_proxy_health.py`

**Interfaces:**
- Consumes: `FactoryConfig`.
- Produces: `create_app(config: FactoryConfig) -> FastAPI` with:
  - `GET /v1/models` → `{"data": [{"type": "model", "id": <name>} for name in config.discovery_models()]}`
  - `GET /healthz` → `{"status": "ok", "models": [...aliases...]}`
  - (route `/v1/messages` is added in Task 9)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_proxy_discovery.py
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
```

```python
# tests/test_proxy_health.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_proxy_discovery.py tests/test_proxy_health.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentfactory.proxy'`

- [ ] **Step 3: Implement proxy.py (discovery + health only)**

```python
# src/agentfactory/proxy.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_proxy_discovery.py tests/test_proxy_health.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentfactory/proxy.py tests/test_proxy_discovery.py tests/test_proxy_health.py
git commit -m "feat: proxy /v1/models discovery and /healthz"
```

---

### Task 9: Proxy app — `/v1/messages` routing (non-streaming + passthrough + streaming)

**Files:**
- Modify: `src/agentfactory/proxy.py`
- Test: `tests/test_proxy_messages.py`

**Interfaces:**
- Consumes: `anthropic_to_openai_request`, `openai_to_anthropic_response` (translate); `openai_stream_to_anthropic_events`, `format_sse` (streaming); `call_openai_backend`, `stream_openai_backend`, `stream_upstream`, `filter_request_headers` (backends); `anthropic_error` (errors).
- Produces: `POST /v1/messages` that — passthrough for `claude-*` (verbatim upstream stream), translate→backend for `local/*` (non-stream JSON or SSE), and a `404` Anthropic error for unknown models. Backend functions are referenced as module attributes so tests can monkeypatch them.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_proxy_messages.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_proxy_messages.py -v`
Expected: FAIL — the four tests error (no `/v1/messages` route → 404 for the non-error cases, import errors for patched names).

- [ ] **Step 3: Extend proxy.py with `/v1/messages`**

Add imports at the top of `proxy.py` and the route inside `create_app` (before `return app`):

```python
# --- add to imports in src/agentfactory/proxy.py ---
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from .backends import (
    call_openai_backend, stream_openai_backend, stream_upstream,
    filter_request_headers,
)
from .errors import anthropic_error
from .streaming import format_sse, openai_stream_to_anthropic_events
from .translate import anthropic_to_openai_request, openai_to_anthropic_response
```

```python
# --- add inside create_app, before `return app` ---
    @app.post("/v1/messages")
    async def messages(request: Request):
        raw = await request.body()
        body = await request.json()
        model = body.get("model", "")
        stream = bool(body.get("stream"))

        if config.is_passthrough(model):
            url = config.proxy.upstream_anthropic.rstrip("/") + "/v1/messages"
            headers = filter_request_headers(request.headers)
            status, up_headers, body_iter = await stream_upstream(
                "POST", url, headers, raw)
            media = up_headers.get("content-type", "application/json")
            return StreamingResponse(body_iter, status_code=status, media_type=media)

        mc = config.resolve_model(model)
        if mc is None:
            return JSONResponse(
                status_code=404,
                content=anthropic_error(f"no such model: {model}", "not_found_error"))

        backend = config.backend_for(mc)
        payload = anthropic_to_openai_request(body, mc)

        if stream:
            payload["stream"] = True

            async def event_stream():
                chunks = stream_openai_backend(
                    backend.base_url, backend.api_key, payload)
                async for event in openai_stream_to_anthropic_events(chunks, model):
                    yield format_sse(event)

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        status, oai = await call_openai_backend(
            backend.base_url, backend.api_key, payload)
        if status >= 400:
            return JSONResponse(
                status_code=status,
                content=anthropic_error(f"backend error: {oai}", "api_error"))
        return JSONResponse(openai_to_anthropic_response(oai, model))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_proxy_messages.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS (all green)

- [ ] **Step 6: Commit**

```bash
git add src/agentfactory/proxy.py tests/test_proxy_messages.py
git commit -m "feat: /v1/messages routing (passthrough, local non-stream, local stream)"
```

---

### Task 10: Live tool-call smoke test

**Files:**
- Create: `src/agentfactory/smoke.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: `FactoryConfig`.
- Produces: `async smoke_test_model(config, alias, base_url=None) -> tuple[bool, str]` — POSTs a minimal tool-bearing `/v1/messages` request to the *running proxy* (default `http://{proxy.host}:{proxy.port}`) for `alias`, returns `(passed, detail)` where `passed` is true iff the response contains a `tool_use` block naming the probe tool.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentfactory.smoke'`

- [ ] **Step 3: Implement smoke.py**

```python
# src/agentfactory/smoke.py
from __future__ import annotations

import httpx

from .config import FactoryConfig

_PROBE_TOOL = {
    "name": "factory_probe",
    "description": "Acknowledge readiness. Call this tool with ok=true.",
    "input_schema": {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    },
}


async def smoke_test_model(
    config: FactoryConfig, alias: str, base_url: str | None = None
) -> tuple[bool, str]:
    base = base_url or f"http://{config.proxy.host}:{config.proxy.port}"
    request = {
        "model": alias,
        "max_tokens": 256,
        "tools": [_PROBE_TOOL],
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user",
                      "content": "Call the factory_probe tool with ok=true."}],
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{base}/v1/messages", json=request)
    except httpx.HTTPError as exc:
        return False, f"request failed: {exc}"

    if resp.status_code != 200:
        return False, f"http {resp.status_code}: {resp.text[:200]}"

    blocks = resp.json().get("content", [])
    for block in blocks:
        if block.get("type") == "tool_use" and block.get("name") == "factory_probe":
            return True, "tool_use returned"
    return False, "no tool_use in response"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentfactory/smoke.py tests/test_smoke.py
git commit -m "feat: live tool-call smoke test"
```

---

### Task 11: `factory` CLI — up / down / status / doctor

**Files:**
- Create: `src/agentfactory/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `FactoryConfig.load`, `create_app`, `smoke_test_model`.
- Produces a Typer `app` with commands:
  - `up --config factory.yaml` — start uvicorn serving `create_app(config)` (foreground; `--daemon` writes a PID file `.factory/proxy.pid` and detaches).
  - `down` — read PID file, terminate, remove file.
  - `status --config` — GET `/healthz`, print status + models.
  - `doctor --config` — GET `/healthz`, then run `smoke_test_model` for each alias; non-zero exit if any fails.
  - `_pid_path() -> Path` returning `.factory/proxy.pid` (testable helper).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
from typer.testing import CliRunner
from agentfactory.cli import app, _pid_path

runner = CliRunner()


def test_pid_path_is_under_dot_factory():
    assert _pid_path().as_posix().endswith(".factory/proxy.pid")


def test_doctor_reports_pass(monkeypatch, tmp_path):
    cfg = tmp_path / "factory.yaml"
    cfg.write_text(
        "proxy: {host: 127.0.0.1, port: 8787}\n"
        "backends: {b: {engine: vllm, base_url: 'http://be/v1', api_key: k}}\n"
        "models: {local/x: {backend: b, model: m}}\n"
    )

    import agentfactory.cli as cli_mod

    async def fake_smoke(config, alias, base_url=None):
        return True, "ok"

    def fake_health(url, timeout=5):
        class R:
            status_code = 200
            def json(self_inner):
                return {"status": "ok", "models": ["local/x"]}
        return R()

    monkeypatch.setattr(cli_mod, "smoke_test_model", fake_smoke)
    monkeypatch.setattr(cli_mod, "_get_health", fake_health)

    result = runner.invoke(app, ["doctor", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "local/x" in result.stdout
    assert "PASS" in result.stdout


def test_doctor_fails_when_smoke_fails(monkeypatch, tmp_path):
    cfg = tmp_path / "factory.yaml"
    cfg.write_text(
        "proxy: {host: 127.0.0.1, port: 8787}\n"
        "backends: {b: {engine: vllm, base_url: 'http://be/v1', api_key: k}}\n"
        "models: {local/x: {backend: b, model: m}}\n"
    )

    import agentfactory.cli as cli_mod

    async def fake_smoke(config, alias, base_url=None):
        return False, "no tool_use in response"

    def fake_health(url, timeout=5):
        class R:
            status_code = 200
            def json(self_inner):
                return {"status": "ok", "models": ["local/x"]}
        return R()

    monkeypatch.setattr(cli_mod, "smoke_test_model", fake_smoke)
    monkeypatch.setattr(cli_mod, "_get_health", fake_health)

    result = runner.invoke(app, ["doctor", "--config", str(cfg)])
    assert result.exit_code == 1
    assert "FAIL" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentfactory.cli'`

- [ ] **Step 3: Implement cli.py**

```python
# src/agentfactory/cli.py
from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

import httpx
import typer
import uvicorn

from .config import FactoryConfig
from .proxy import create_app
from .smoke import smoke_test_model

app = typer.Typer(help="agentFactory control CLI")

DEFAULT_CONFIG = "factory.yaml"


def _pid_path() -> Path:
    return Path(".factory") / "proxy.pid"


def _base_url(cfg: FactoryConfig) -> str:
    return f"http://{cfg.proxy.host}:{cfg.proxy.port}"


def _get_health(url: str, timeout: int = 5):
    return httpx.get(url, timeout=timeout)


@app.command()
def up(config: str = DEFAULT_CONFIG, daemon: bool = False) -> None:
    """Start the routing proxy."""
    cfg = FactoryConfig.load(config)
    if daemon:
        pid = os.fork()
        if pid > 0:
            _pid_path().parent.mkdir(parents=True, exist_ok=True)
            _pid_path().write_text(str(pid))
            typer.echo(f"proxy started (pid {pid}) on {_base_url(cfg)}")
            return
    uvicorn.run(create_app(cfg), host=cfg.proxy.host, port=cfg.proxy.port,
                log_level="info")


@app.command()
def down() -> None:
    """Stop a daemonized proxy."""
    p = _pid_path()
    if not p.exists():
        typer.echo("no pid file; proxy not running via daemon")
        raise typer.Exit(0)
    pid = int(p.read_text())
    try:
        os.kill(pid, signal.SIGTERM)
        typer.echo(f"stopped proxy (pid {pid})")
    finally:
        p.unlink(missing_ok=True)


@app.command()
def status(config: str = DEFAULT_CONFIG) -> None:
    """Show proxy health."""
    cfg = FactoryConfig.load(config)
    try:
        resp = _get_health(f"{_base_url(cfg)}/healthz")
    except httpx.HTTPError as exc:
        typer.echo(f"proxy unreachable: {exc}")
        raise typer.Exit(1)
    data = resp.json()
    typer.echo(f"status: {data['status']}")
    typer.echo(f"models: {', '.join(data['models'])}")


@app.command()
def doctor(config: str = DEFAULT_CONFIG) -> None:
    """Check proxy health and run a tool-call smoke test per local model."""
    cfg = FactoryConfig.load(config)
    try:
        health = _get_health(f"{_base_url(cfg)}/healthz")
    except httpx.HTTPError as exc:
        typer.echo(f"proxy unreachable: {exc}")
        raise typer.Exit(1)
    if health.status_code != 200:
        typer.echo(f"proxy unhealthy: http {health.status_code}")
        raise typer.Exit(1)
    typer.echo("proxy: OK")

    all_ok = True
    for alias in cfg.models:
        passed, detail = asyncio.run(smoke_test_model(cfg, alias))
        mark = "PASS" if passed else "FAIL"
        typer.echo(f"  {alias}: {mark} ({detail})")
        all_ok = all_ok and passed

    raise typer.Exit(0 if all_ok else 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentfactory/cli.py tests/test_cli.py
git commit -m "feat: factory CLI (up/down/status/doctor)"
```

---

### Task 12: `local-coder` subagent template + worktree review docs

**Files:**
- Create: `templates/agents/local-coder.md`
- Create: `docs/worktree-review.md`
- Modify: `README.md`

**Interfaces:**
- Produces: a ready-to-install Claude Code subagent definition pointing at `local/qwen3-coder`, and a documented worktree review loop. No code; deliverable is the template + docs. (Automated `factory install` is Phase 2.)

- [ ] **Step 1: Write the subagent template**

```markdown
<!-- templates/agents/local-coder.md -->
---
name: local-coder
description: >-
  Generates code for well-specified, mechanical, high-volume tasks using a
  local model. Use for boilerplate, repetitive edits, scaffolding, and
  straightforward implementations from a precise spec. NOT for architecture,
  ambiguous tasks, or review — those stay on the parent model.
model: local/qwen3-coder
---

You are a code-generation worker backed by a local model. You operate ONLY
inside the git worktree you were given. You must:

1. Make the smallest change that satisfies the task as specified.
2. Run the relevant tests or build for the files you touched.
3. Return a concise unified diff of your changes plus a one-paragraph summary
   (what changed, test/build result). Do NOT narrate your process.

You do not have authority to merge. The parent process reviews your diff and
decides whether to integrate it.
```

- [ ] **Step 2: Write the worktree review doc**

```markdown
<!-- docs/worktree-review.md -->
# Worktree review loop

Local subagents generate code in isolation; the parent (Anthropic) reviews
before anything touches the main tree.

1. **Dispatch in isolation.** Launch `local-coder` with worktree isolation so
   it works on its own branch/copy of the repo.
2. **Generate + self-check.** The local model edits and runs tests inside the
   worktree. It cannot touch the main tree.
3. **Return a diff.** The subagent returns a unified diff + summary.
4. **Parent review (always Anthropic).** The orchestrator reviews the diff for
   correctness, conventions, and scope. Review is NEVER routed to a local model.
5. **Decide.** Merge the branch, send specific feedback for another pass, or
   discard the worktree (nothing leaked).

## Manual setup (Phase 1)

```bash
# 1. start the proxy
factory up --config factory.yaml      # or: --daemon

# 2. point Claude Code at the proxy + enable model discovery
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1

# 3. install the subagent template into your project
mkdir -p .claude/agents && cp templates/agents/local-coder.md .claude/agents/

# 4. verify
factory doctor --config factory.yaml
```

Automated `factory install` (settings.json wiring + template copy) lands in
Phase 2.
```

- [ ] **Step 3: Update README with quickstart pointer**

Replace the body of `README.md` with:

```markdown
# agentFactory

Orchestrate local-model subagents from a Claude Code parent process. Local
models do the grunt-work code generation; the Anthropic parent orchestrates and
reviews every result.

- Design spec: [docs/superpowers/specs/2026-06-27-agentfactory-design.md](docs/superpowers/specs/2026-06-27-agentfactory-design.md)
- Phase 1 plan: [docs/superpowers/plans/2026-06-27-agentfactory-phase1.md](docs/superpowers/plans/2026-06-27-agentfactory-phase1.md)
- Worktree review loop: [docs/worktree-review.md](docs/worktree-review.md)

## Quickstart

```bash
uv sync
cp factory.example.yaml factory.yaml   # edit backends/models for your hardware
uv run factory up --config factory.yaml
```

Then follow [docs/worktree-review.md](docs/worktree-review.md) to wire Claude
Code to the proxy and install the `local-coder` subagent.
```

- [ ] **Step 4: Verify the template is valid front-matter**

Run: `uv run python -c "import yaml,io; t=open('templates/agents/local-coder.md').read(); fm=t.split('---')[1]; d=yaml.safe_load(fm); assert d['name']=='local-coder' and d['model']=='local/qwen3-coder'; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add templates/agents/local-coder.md docs/worktree-review.md README.md
git commit -m "feat: local-coder subagent template + worktree review docs"
```

---

### Task 13: Phase 1 integration verification

**Files:**
- Create: `tests/test_integration_phase1.py`

**Interfaces:**
- Consumes: everything. End-to-end test through the real FastAPI app with only the backend HTTP boundary faked — proves the full path (Anthropic request in → translated → backend → translated back) including a tool-call round trip.

- [ ] **Step 1: Write the end-to-end test**

```python
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
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `uv run pytest tests/test_integration_phase1.py -v`
Expected: PASS (1 passed)

- [ ] **Step 3: Run the entire suite**

Run: `uv run pytest -v`
Expected: PASS (all tests green across all files)

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_phase1.py
git commit -m "test: phase-1 end-to-end tool-call round trip"
```

---

## Self-Review

**Spec coverage (Phase 1 items from the design spec §7):**
- Routing proxy: passthrough + one local backend → Tasks 7, 9 ✓
- `/v1/messages` streaming + non-streaming → Task 9 ✓
- `/v1/models` discovery → Task 8 ✓
- Translation module → Tasks 4, 5 ✓
- Guided decoding (ensure `tool_choice` set) → Task 4 ✓
- SSE re-framing → Task 6 ✓
- `factory.yaml` registry → Task 3 ✓
- `factory up`/`down`/`doctor` incl. tool-call smoke test → Tasks 10, 11 ✓
- One `local-coder` template; manual worktree review → Task 12 ✓
- Core invariant (local-only generation; claude passthrough) → enforced in Task 9 routing + Task 12 template; verified Task 13 ✓
- Never log credentials → no logging of headers/keys anywhere in implementation ✓
- Phase 1 exit criteria (orchestrator → local-coder → tool-call → review) → mechanism proven by Task 13 end-to-end test ✓

**Deferred to later plans (correctly out of Phase 1 scope):** `factory install`, `factory models`, `factory report`, merge helper, remote backend config exercise, tool-call repair, observability/cost split (Phase 2); docker-compose, LM Studio native path, reviewer-subagent, ReAct fallback (Phase 3).

**Placeholder scan:** No TBD/TODO/"add error handling" placeholders — every code step contains complete, runnable code.

**Type consistency:** Names verified consistent across tasks — `anthropic_to_openai_request`, `openai_to_anthropic_response`, `map_stop_reason`, `openai_stream_to_anthropic_events`, `format_sse`, `call_openai_backend`, `stream_openai_backend`, `stream_upstream`, `filter_request_headers`, `create_app`, `smoke_test_model`, `_pid_path`, `_get_health`. Config types (`FactoryConfig`, `ProxyConfig`, `BackendConfig`, `ModelConfig`) used with the same fields throughout. Proxy references backend functions as module attributes (`proxy_mod.call_openai_backend`, etc.) so monkeypatching in Tasks 9/13 works.
