# agentFactory Phase 2a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Phase 1 actually usable on the local NVIDIA box: `factory install` wires a project, `factory serve` launches vLLM correctly, `factory apply` lands a reviewed local diff safely, and `local-coder` returns an apply-able diff — proven by a reproducible first-run walkthrough.

**Architecture:** Three thin, single-purpose modules (`serve`, `apply`, `install`) and three new `factory` subcommands sit on top of the unchanged Phase 1 proxy/translation core. The local model only ever writes inside its isolated worktree; its `git diff HEAD` is returned in-message, reviewed by the parent, and applied to main via `git apply --3way`.

**Tech Stack:** Python 3.12, Typer, stdlib (`json`, `subprocess`, `shutil`, `urllib.parse`, `pathlib`). Tooling: `uv`, pytest.

## Global Constraints

- Python **3.11+** (dev on 3.12); env + deps via `uv`; run tests with `uv run pytest`.
- **Never log or persist credentials** (`x-api-key`, `Authorization`, backend `api_key`).
- `factory install` is **idempotent**, backs up `settings.local.json` before changing it, writes atomically, and **never clobbers unrelated keys**. Target is `.claude/settings.local.json` (gitignored), env keys exactly `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY="1"`.
- `factory serve` **must** emit `--enable-auto-tool-choice --tool-call-parser <tool_parser>`; a model with no `tool_parser` is an error (tool calling would silently break).
- Merge handoff is **diff-in-message + `factory apply`**; `git apply --3way` is all-or-nothing; a conflicting/empty diff is rejected and the tree is left clean.
- `local-coder` works only inside its worktree, returns `git --no-pager diff HEAD` in one ```diff block, and never commits.
- Follow Phase 1 patterns: pure logic separated from I/O; one responsibility per module; typed public functions; TDD; frequent commits.

---

## File Structure

```
src/agentfactory/serve.py      (new) — build_vllm_argv (pure) + serve_vllm launcher
src/agentfactory/apply.py      (new) — apply_diff against the current git repo
src/agentfactory/install.py    (new) — merge_settings_env (pure) + install_project
src/agentfactory/cli.py        (modify) — add install / serve / apply commands
templates/agents/local-coder.md (modify) — diff-in-message contract
docs/first-run.md              (new) — reproducible walkthrough
.gitignore                     (modify) — ignore settings.local.json + .bak
tests/test_serve.py            (new)
tests/test_apply.py            (new)
tests/test_install.py          (new)
tests/test_cli.py              (modify) — wiring for the 3 new commands
```

`serve`/`apply`/`install` are independent; the CLI task consumes all three. Build order: serve → apply → install → CLI wiring → template+docs → integration.

---

### Task 1: `serve.py` — vLLM command builder + launcher

**Files:**
- Create: `src/agentfactory/serve.py`
- Test: `tests/test_serve.py`

**Interfaces:**
- Consumes: `ModelConfig`, `BackendConfig`, `FactoryConfig` from `config.py` (fields: `model`, `tool_parser`, `guided_decoding`, `context`; `BackendConfig.base_url`).
- Produces:
  - `build_vllm_argv(mc: ModelConfig, backend: BackendConfig) -> list[str]` — **pure**. Raises `ValueError` if `mc.tool_parser` is falsy.
  - `port_from_base_url(base_url: str) -> int`
  - `is_local_host(base_url: str) -> bool` — true for `127.0.0.1`/`localhost`/`0.0.0.0`.
  - `serve_vllm(config: FactoryConfig, alias: str, dry_run: bool = False) -> int` — resolves alias; for `dry_run` or a non-local backend, prints the command and returns 0; otherwise `os.execvp`s vLLM (returns 1 if `vllm` not on PATH or alias unknown).

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_serve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentfactory.serve'`

- [ ] **Step 3: Implement serve.py**

```python
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
    argv = build_vllm_argv(mc, backend)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_serve.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentfactory/serve.py tests/test_serve.py
git commit -m "feat: factory serve vLLM command builder + launcher"
```

---

### Task 2: `apply.py` — apply a reviewed diff to the main tree

**Files:**
- Create: `src/agentfactory/apply.py`
- Test: `tests/test_apply.py`

**Interfaces:**
- Produces:
  - `ApplyResult` dataclass: `ok: bool`, `changed_files: list[str]`, `message: str`.
  - `apply_diff(diff_text: str, check_only: bool = False, cwd: str | Path | None = None) -> ApplyResult` — empty diff → `ok=False`; non-git cwd → `ok=False`; `check_only` runs `git apply --check --3way` (no changes); otherwise `git apply --3way`. `changed_files` derived from `git apply --numstat`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_apply.py
import subprocess
from pathlib import Path
from agentfactory.apply import apply_diff, ApplyResult


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, text=True)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "t@t"], repo)
    _git(["config", "user.name", "t"], repo)
    (repo / "a.txt").write_text("hello\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "init"], repo)
    return repo


def _diff_for_edit(repo: Path) -> str:
    (repo / "a.txt").write_text("hello world\n")
    diff = subprocess.run(["git", "--no-pager", "diff"], cwd=repo,
                          capture_output=True, text=True).stdout
    _git(["checkout", "--", "a.txt"], repo)  # restore
    return diff


def test_apply_valid_diff_changes_file(tmp_path):
    repo = _make_repo(tmp_path)
    diff = _diff_for_edit(repo)
    res = apply_diff(diff, cwd=repo)
    assert res.ok is True
    assert "a.txt" in res.changed_files
    assert (repo / "a.txt").read_text() == "hello world\n"


def test_check_only_does_not_modify(tmp_path):
    repo = _make_repo(tmp_path)
    diff = _diff_for_edit(repo)
    res = apply_diff(diff, check_only=True, cwd=repo)
    assert res.ok is True
    assert (repo / "a.txt").read_text() == "hello\n"  # unchanged


def test_conflicting_diff_rejected_tree_clean(tmp_path):
    repo = _make_repo(tmp_path)
    bogus = (
        "diff --git a/a.txt b/a.txt\n"
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -5,1 +5,1 @@\n"
        "-nonexistent line\n"
        "+changed\n"
    )
    res = apply_diff(bogus, cwd=repo)
    assert res.ok is False
    assert (repo / "a.txt").read_text() == "hello\n"  # untouched


def test_empty_diff_rejected(tmp_path):
    repo = _make_repo(tmp_path)
    res = apply_diff("   \n", cwd=repo)
    assert res.ok is False
    assert "empty" in res.message.lower()


def test_non_git_dir_rejected(tmp_path):
    res = apply_diff("diff --git a/x b/x\n", cwd=tmp_path)
    assert res.ok is False
    assert "git" in res.message.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_apply.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentfactory.apply'`

- [ ] **Step 3: Implement apply.py**

```python
# src/agentfactory/apply.py
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ApplyResult:
    ok: bool
    changed_files: list[str] = field(default_factory=list)
    message: str = ""


def _run(args: list[str], cwd, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=cwd, input=stdin, capture_output=True, text=True
    )


def _changed_files(diff_text: str, cwd) -> list[str]:
    proc = _run(["git", "apply", "--numstat"], cwd, stdin=diff_text)
    files: list[str] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            files.append(parts[2])
    return files


def apply_diff(
    diff_text: str, check_only: bool = False, cwd: str | Path | None = None
) -> ApplyResult:
    cwd = str(cwd) if cwd is not None else None
    if not diff_text.strip():
        return ApplyResult(False, [], "empty diff")

    inside = _run(["git", "rev-parse", "--is-inside-work-tree"], cwd)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return ApplyResult(False, [], "not a git repository")

    files = _changed_files(diff_text, cwd)
    args = ["git", "apply", "--3way"]
    if check_only:
        args = ["git", "apply", "--check", "--3way"]
    proc = _run(args, cwd, stdin=diff_text)
    if proc.returncode != 0:
        return ApplyResult(False, files, proc.stderr.strip() or "apply failed")
    return ApplyResult(True, files, "checked" if check_only else "applied")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_apply.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentfactory/apply.py tests/test_apply.py
git commit -m "feat: factory apply a reviewed diff to the main tree"
```

---

### Task 3: `install.py` — wire a project to the proxy

**Files:**
- Create: `src/agentfactory/install.py`
- Modify: `.gitignore`
- Test: `tests/test_install.py`

**Interfaces:**
- Consumes: `FactoryConfig` (`config.proxy.host`, `config.proxy.port`).
- Produces:
  - `merge_settings_env(existing: dict, host: str, port: int) -> dict` — **pure**. Returns a new dict where `env.ANTHROPIC_BASE_URL = f"http://{host}:{port}"` and `env.CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY = "1"`, preserving all other keys and all other `env` entries.
  - `InstallResult` dataclass: `actions: list[str]`.
  - `install_project(target_root: Path, config: FactoryConfig, force: bool = False, package_root: Path | None = None) -> InstallResult` — creates `factory.yaml` from `package_root/factory.example.yaml` if absent; backs up + atomically writes `.claude/settings.local.json` with merged env; copies `package_root/templates/agents/local-coder.md` → `.claude/agents/local-coder.md` (skip if exists unless `force`). `package_root` defaults to the agentFactory repo root resolved from this file.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_install.py
import json
from pathlib import Path
from agentfactory.config import FactoryConfig, ProxyConfig
from agentfactory.install import merge_settings_env, install_project


def test_merge_settings_env_sets_keys():
    out = merge_settings_env({}, "127.0.0.1", 8787)
    assert out["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert out["env"]["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"


def test_merge_settings_env_preserves_other_keys():
    existing = {"permissions": {"allow": ["Bash"]},
                "env": {"FOO": "bar"}}
    out = merge_settings_env(existing, "127.0.0.1", 8787)
    assert out["permissions"] == {"allow": ["Bash"]}
    assert out["env"]["FOO"] == "bar"
    assert out["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def _pkg(tmp_path: Path) -> Path:
    pkg = tmp_path / "pkg"
    (pkg / "templates" / "agents").mkdir(parents=True)
    (pkg / "factory.example.yaml").write_text("proxy:\n  host: 127.0.0.1\n  port: 8787\n")
    (pkg / "templates" / "agents" / "local-coder.md").write_text("---\nname: local-coder\n---\nbody\n")
    return pkg


def _cfg():
    return FactoryConfig(proxy=ProxyConfig(host="127.0.0.1", port=8787))


def test_install_creates_files(tmp_path):
    pkg = _pkg(tmp_path)
    target = tmp_path / "proj"
    target.mkdir()
    res = install_project(target, _cfg(), package_root=pkg)
    assert (target / "factory.yaml").exists()
    slj = json.loads((target / ".claude" / "settings.local.json").read_text())
    assert slj["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert (target / ".claude" / "agents" / "local-coder.md").exists()
    assert any("settings.local.json" in a for a in res.actions)


def test_install_idempotent_and_backs_up(tmp_path):
    pkg = _pkg(tmp_path)
    target = tmp_path / "proj"
    target.mkdir()
    install_project(target, _cfg(), package_root=pkg)
    # second run: backup created, env still correct, template skipped
    res = install_project(target, _cfg(), package_root=pkg)
    assert (target / ".claude" / "settings.local.json.bak").exists()
    slj = json.loads((target / ".claude" / "settings.local.json").read_text())
    assert slj["env"]["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
    assert any("skipped" in a for a in res.actions)


def test_install_preserves_existing_settings(tmp_path):
    pkg = _pkg(tmp_path)
    target = tmp_path / "proj"
    (target / ".claude").mkdir(parents=True)
    (target / ".claude" / "settings.local.json").write_text(
        json.dumps({"permissions": {"allow": ["Bash"]}}))
    install_project(target, _cfg(), package_root=pkg)
    slj = json.loads((target / ".claude" / "settings.local.json").read_text())
    assert slj["permissions"] == {"allow": ["Bash"]}
    assert slj["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def test_install_does_not_overwrite_existing_factory_yaml(tmp_path):
    pkg = _pkg(tmp_path)
    target = tmp_path / "proj"
    target.mkdir()
    (target / "factory.yaml").write_text("proxy:\n  port: 9999\n")
    install_project(target, _cfg(), package_root=pkg)
    assert "9999" in (target / "factory.yaml").read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_install.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentfactory.install'`

- [ ] **Step 3: Implement install.py**

```python
# src/agentfactory/install.py
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .config import FactoryConfig


def merge_settings_env(existing: dict, host: str, port: int) -> dict:
    out = dict(existing)
    env = dict(out.get("env") or {})
    env["ANTHROPIC_BASE_URL"] = f"http://{host}:{port}"
    env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    out["env"] = env
    return out


@dataclass
class InstallResult:
    actions: list[str] = field(default_factory=list)


def _package_root() -> Path:
    # src/agentfactory/install.py -> repo root
    return Path(__file__).resolve().parents[2]


def install_project(
    target_root: Path,
    config: FactoryConfig,
    force: bool = False,
    package_root: Path | None = None,
) -> InstallResult:
    target_root = Path(target_root)
    package_root = Path(package_root) if package_root else _package_root()
    res = InstallResult()

    # 1. factory.yaml
    fy = target_root / "factory.yaml"
    example = package_root / "factory.example.yaml"
    if fy.exists():
        res.actions.append("factory.yaml exists (kept)")
    elif example.exists():
        shutil.copyfile(example, fy)
        res.actions.append("created factory.yaml")

    # 2. .claude/settings.local.json (atomic, backed up, non-clobbering)
    claude = target_root / ".claude"
    claude.mkdir(parents=True, exist_ok=True)
    slj = claude / "settings.local.json"
    existing: dict = {}
    if slj.exists():
        existing = json.loads(slj.read_text() or "{}")
        (claude / "settings.local.json.bak").write_text(json.dumps(existing, indent=2))
        res.actions.append("backed up settings.local.json")
    merged = merge_settings_env(existing, config.proxy.host, config.proxy.port)
    tmp = claude / "settings.local.json.tmp"
    tmp.write_text(json.dumps(merged, indent=2) + "\n")
    tmp.replace(slj)
    res.actions.append("wrote settings.local.json env")

    # 3. local-coder template
    agents = claude / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    src = package_root / "templates" / "agents" / "local-coder.md"
    dst = agents / "local-coder.md"
    if dst.exists() and not force:
        res.actions.append("local-coder.md exists (skipped; use --force)")
    elif src.exists():
        shutil.copyfile(src, dst)
        res.actions.append("installed local-coder.md")

    return res
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_install.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Ignore the generated local settings**

Add these lines to `.gitignore` (under the existing `# agentFactory runtime` section):

```
# Claude Code local settings written by `factory install`
.claude/settings.local.json
.claude/settings.local.json.bak
```

- [ ] **Step 6: Commit**

```bash
git add src/agentfactory/install.py tests/test_install.py .gitignore
git commit -m "feat: factory install wires a project to the proxy (idempotent)"
```

---

### Task 4: CLI wiring — `install` / `serve` / `apply`

**Files:**
- Modify: `src/agentfactory/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `install_project` (install.py), `serve_vllm` (serve.py), `apply_diff` (apply.py), `FactoryConfig`.
- Produces three Typer commands on the existing `app`:
  - `install --config factory.yaml --force` → loads config (defaults if `factory.yaml` absent), calls `install_project(Path.cwd(), cfg, force)`, prints actions.
  - `serve <alias> --config factory.yaml --dry-run` → `raise typer.Exit(serve_vllm(cfg, alias, dry_run))`.
  - `apply [diff] --check` → reads diff from the path arg or stdin, calls `apply_diff`, prints result, exits 0/1 on `ok`.
- Module references `install_project`, `serve_vllm`, `apply_diff` as module-level imports so tests can monkeypatch them.

- [ ] **Step 1: Write the failing tests (append to tests/test_cli.py)**

```python
# --- append to tests/test_cli.py ---
import agentfactory.cli as cli_mod
from agentfactory.install import InstallResult
from agentfactory.apply import ApplyResult


def test_install_command_invokes_install_project(monkeypatch, tmp_path):
    called = {}

    def fake_install(root, cfg, force=False, package_root=None):
        called["root"] = root
        called["force"] = force
        return InstallResult(actions=["created factory.yaml", "wrote settings.local.json env"])

    monkeypatch.setattr(cli_mod, "install_project", fake_install)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["install"])
    assert result.exit_code == 0
    assert "settings.local.json" in result.stdout
    assert called["force"] is False


def test_serve_command_returns_serve_vllm_code(monkeypatch, tmp_path):
    cfg = tmp_path / "factory.yaml"
    cfg.write_text(
        "proxy: {host: 127.0.0.1, port: 8787}\n"
        "backends: {b: {engine: vllm, base_url: 'http://127.0.0.1:8000/v1', api_key: k}}\n"
        "models: {local/x: {backend: b, model: m, tool_parser: hermes}}\n"
    )

    def fake_serve(config, alias, dry_run=False):
        assert alias == "local/x"
        assert dry_run is True
        return 0

    monkeypatch.setattr(cli_mod, "serve_vllm", fake_serve)
    result = runner.invoke(app, ["serve", "local/x", "--config", str(cfg), "--dry-run"])
    assert result.exit_code == 0


def test_apply_command_reads_stdin_and_reports(monkeypatch):
    def fake_apply(text, check_only=False, cwd=None):
        assert "diff --git" in text
        assert check_only is True
        return ApplyResult(True, ["a.txt"], "checked")

    monkeypatch.setattr(cli_mod, "apply_diff", fake_apply)
    result = runner.invoke(app, ["apply", "--check"],
                           input="diff --git a/a.txt b/a.txt\n")
    assert result.exit_code == 0
    assert "a.txt" in result.stdout


def test_apply_command_nonzero_on_failure(monkeypatch):
    monkeypatch.setattr(cli_mod, "apply_diff",
                        lambda text, check_only=False, cwd=None: ApplyResult(False, [], "empty diff"))
    result = runner.invoke(app, ["apply"], input="x")
    assert result.exit_code == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — the new tests error (commands `install`/`serve`/`apply` not registered; names not importable on `cli_mod`).

- [ ] **Step 3: Extend cli.py**

Add imports near the existing imports in `src/agentfactory/cli.py`:

```python
# --- add to imports in src/agentfactory/cli.py ---
import sys

from .apply import apply_diff
from .install import install_project
from .serve import serve_vllm
```

Add these three commands after the existing `doctor` command (before end of file):

```python
@app.command()
def install(config: str = DEFAULT_CONFIG, force: bool = False) -> None:
    """Wire the current project to the proxy (idempotent)."""
    cfg = FactoryConfig.load(config) if Path(config).exists() else FactoryConfig()
    res = install_project(Path.cwd(), cfg, force=force)
    for action in res.actions:
        typer.echo(f"  {action}")


@app.command()
def serve(alias: str, config: str = DEFAULT_CONFIG, dry_run: bool = False) -> None:
    """Launch vLLM for a local model alias with the correct tool-calling flags."""
    cfg = FactoryConfig.load(config)
    raise typer.Exit(serve_vllm(cfg, alias, dry_run=dry_run))


@app.command()
def apply(diff: str = typer.Argument(None), check: bool = False) -> None:
    """Apply a reviewed unified diff to the main tree (or --check to dry-run)."""
    text = Path(diff).read_text() if diff else sys.stdin.read()
    res = apply_diff(text, check_only=check)
    for f in res.changed_files:
        typer.echo(f"  {f}")
    typer.echo(res.message)
    raise typer.Exit(0 if res.ok else 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (all cli tests, including the 4 new ones)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (all green)

- [ ] **Step 6: Commit**

```bash
git add src/agentfactory/cli.py tests/test_cli.py
git commit -m "feat: wire install/serve/apply into the factory CLI"
```

---

### Task 5: `local-coder` diff-in-message contract + first-run walkthrough

**Files:**
- Modify: `templates/agents/local-coder.md`
- Create: `docs/first-run.md`

**Interfaces:**
- Produces the operational contract local subagents follow and the reproducible NVIDIA-box walkthrough. No code; deliverable is the template + doc. Validate the template's YAML front-matter still parses with `name: local-coder` and `model: local/qwen3-coder`.

- [ ] **Step 1: Rewrite templates/agents/local-coder.md**

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
inside the git worktree you were dispatched into. You must NOT modify anything
outside this worktree, and you must NOT commit.

Workflow:

1. Make the smallest change that satisfies the task as specified.
2. Run the relevant tests or build for the files you touched, inside this
   worktree.
3. Produce the patch by running exactly:

   `git --no-pager diff HEAD`

   Return that output verbatim inside a single fenced ```diff block, followed
   by a one-paragraph summary: what changed and the test/build result.

Do not narrate your process. Do not commit, push, or merge. The parent process
reviews your diff and, if accepted, applies it to the main tree with
`factory apply`. If you made no changes, say so explicitly instead of
returning an empty diff.
```

- [ ] **Step 2: Verify the front-matter still parses**

Run: `uv run python -c "import yaml; t=open('templates/agents/local-coder.md').read(); d=yaml.safe_load(t.split('---')[1]); assert d['name']=='local-coder' and d['model']=='local/qwen3-coder'; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Create docs/first-run.md**

```markdown
<!-- docs/first-run.md -->
# First real run (local NVIDIA box)

End-to-end walkthrough: a Claude Code orchestrator (Anthropic) dispatches the
`local-coder` subagent, a local Qwen model generates a change in an isolated
worktree, you review the diff, and `factory apply` lands it on the main tree.

> Tested with vLLM (pin your version here after the first run) and
> `Qwen/Qwen3-Coder-30B-A3B-Instruct` on a single NVIDIA GPU.

## 1. Configure

```bash
cp factory.example.yaml factory.yaml      # edit model/alias for your GPU if needed
```

## 2. Launch the local model

```bash
factory serve local/qwen3-coder           # runs vLLM on :8000 with the right
                                          # --tool-call-parser + guided decoding
```

Leave it running (use a second terminal, or background it). To preview the
command without launching: `factory serve local/qwen3-coder --dry-run`.

## 3. Start the proxy and verify

```bash
factory up --config factory.yaml          # proxy on :8787 (or: --daemon)
factory doctor --config factory.yaml      # health + live tool-call smoke test
```

`factory doctor` must report the model `PASS` before continuing — that confirms
the local model returns a valid tool call through the proxy.

## 4. Wire this project

```bash
factory install                           # writes .claude/settings.local.json
                                          # (ANTHROPIC_BASE_URL + discovery),
                                          # installs local-coder, creates factory.yaml
```

Restart Claude Code so it picks up the new env.

## 5. Delegate a task

In Claude Code, have the orchestrator dispatch the `local-coder` subagent
(with worktree isolation) on a small, well-specified task, e.g.:

> "Add a `clamp(value, lo, hi)` helper to `utils.py` with a docstring and
> three unit tests. Use the local-coder subagent in an isolated worktree."

The subagent returns a ```diff block plus a summary.

## 6. Review and apply

Review the diff. When you accept it, save it and apply:

```bash
# save the returned diff to change.patch, then:
factory apply --check change.patch        # dry run — must say "checked"
factory apply change.patch                # lands it on the main tree
<your project's test command>             # confirm the change is good
```

A conflicting or malformed diff is rejected and the tree is left clean — nothing
the local model produced reaches main until you apply it.
```

- [ ] **Step 4: Commit**

```bash
git add templates/agents/local-coder.md docs/first-run.md
git commit -m "feat: local-coder diff-in-message contract + first-run walkthrough"
```

---

### Task 6: Phase 2a integration verification

**Files:**
- Test: `tests/test_cli.py` (add a help-surface assertion)

**Interfaces:**
- Consumes: the assembled CLI. Confirms the three new commands are registered and the whole suite is green. The live model run is performed manually via `docs/first-run.md` (non-deterministic; not automated).

- [ ] **Step 1: Add a CLI help-surface test (append to tests/test_cli.py)**

```python
# --- append to tests/test_cli.py ---
def test_new_commands_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("install", "serve", "apply", "up", "doctor"):
        assert cmd in result.stdout
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_cli.py::test_new_commands_registered -v`
Expected: PASS

- [ ] **Step 3: Run the entire suite**

Run: `uv run pytest -q`
Expected: PASS (all tests across all files green)

- [ ] **Step 4: Commit**

```bash
git add tests/test_cli.py
git commit -m "test: assert install/serve/apply are registered on the CLI"
```

---

## Self-Review

**Spec coverage (Phase 2a spec sections):**
- §4.1 `factory install` (settings.local.json merge, backup, template copy, factory.yaml init, idempotent) → Task 3 + Task 4 ✓
- §4.2 `factory serve` (vLLM argv from registry, parser mandatory, remote prints, port from base_url) → Task 1 + Task 4 ✓
- §4.3 `factory apply` (--3way, --check dry-run, conflict/empty rejection, changed files) → Task 2 + Task 4 ✓
- §4.4 local-coder diff-in-message contract → Task 5 ✓
- §4.5 docs/first-run.md walkthrough → Task 5 ✓
- §6 safety model (worktree-only writes, apply gate, atomic install) → enforced in Tasks 2/3/5 ✓
- §7 testing strategy (install/serve/apply units + CLI wiring; live run manual) → Tasks 1–4, 6 ✓
- Global constraint: settings.local.json gitignored → Task 3 Step 5 ✓
- Global constraint: never log credentials → no logging of secrets in any new module ✓

**Deferred (correctly out of scope):** `factory models`/`report`, observability/cost split, tool-call repair, the three Phase-1 Minors, remote-backend exercise — remain in `docs/phase2-backlog.md`.

**Placeholder scan:** none — every code step contains complete, runnable code; the only "pin your version here" is an intentional doc instruction for the manual run, not a plan placeholder.

**Type consistency:** verified across tasks — `build_vllm_argv`/`port_from_base_url`/`is_local_host`/`serve_vllm` (serve.py); `apply_diff`/`ApplyResult` (apply.py); `merge_settings_env`/`install_project`/`InstallResult` (install.py). CLI imports these exact names and references them as module attributes (`cli_mod.install_project`, `cli_mod.serve_vllm`, `cli_mod.apply_diff`) so the Task 4 monkeypatches resolve. `ModelConfig`/`BackendConfig`/`FactoryConfig`/`ProxyConfig` fields used match Phase 1 definitions.
