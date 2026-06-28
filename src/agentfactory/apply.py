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
    args = ["git", "apply"]
    if check_only:
        args = ["git", "apply", "--check"]
    proc = _run(args, cwd, stdin=diff_text)
    if proc.returncode != 0:
        return ApplyResult(False, files, proc.stderr.strip() or "apply failed")
    return ApplyResult(True, files, "checked" if check_only else "applied")
