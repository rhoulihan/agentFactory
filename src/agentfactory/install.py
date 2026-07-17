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
