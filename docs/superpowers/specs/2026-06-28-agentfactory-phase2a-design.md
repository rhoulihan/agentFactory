# agentFactory Phase 2a — "Make it real & usable" — Design Spec

**Date:** 2026-06-28
**Status:** Approved for planning
**Author:** Rick Houlihan + Claude
**Builds on:** Phase 1 (merged, PR #1) — see `docs/superpowers/specs/2026-06-27-agentfactory-design.md`

---

## 1. Summary

Phase 1 delivered the routing proxy, translation, and CLI scaffold — all unit-tested but never run against a real local model. Phase 2a turns it into something that **actually runs on the local NVIDIA box**: one command to set a project up, one command to launch the model with the correct tool-calling flags, and a safe review→apply loop — proven by a real end-to-end run.

The slice is deliberately narrow: it makes the **single happy path** (orchestrator → local-coder in an isolated worktree → diff → parent review → apply to main) real and pleasant, and defers everything that widens coverage without deepening that path.

### Goals

- `factory install` — one idempotent command wires a project to the proxy.
- `factory serve <alias>` — launches vLLM with the right flags from the registry.
- `factory apply` — lands a reviewed local diff on the main tree safely.
- A `local-coder` that returns an apply-able diff and never touches the main tree.
- A reproducible **first real run** executed once on the NVIDIA box.

### Non-goals (stay in backlog)

`factory models` / `factory report`, observability + cost split, proxy tool-call repair, the three Phase-1 Minors (streaming usage, error-body sanitize, header forwarding), and exercising the remote Ubuntu backend. See `docs/phase2-backlog.md`.

---

## 2. Decisions (from brainstorming)

- **First-run backend:** localhost vLLM on the NVIDIA box (`http://127.0.0.1:8000/v1`).
- **Merge handoff:** diff-in-message + `factory apply`. The diff survives independent of how Claude Code manages the worktree; `git apply --3way`; conflicts are rejected, main left clean.
- **Install target:** `.claude/settings.local.json` (gitignored, machine-specific). Idempotent merge, back up first, never clobber unrelated keys.
- **Serve helper:** yes — build the vLLM command from the registry so the tool-call parser is always correct.
- **Run proof:** a `docs/first-run.md` walkthrough we execute together, plus a lightweight `factory apply --check` gate; no automated live test in CI (a real model is non-deterministic).

---

## 3. Architecture

Phase 2a adds three thin, single-purpose modules and three CLI commands on top of Phase 1. No change to the proxy/translation core.

```
                    factory CLI (Phase 1: up/down/status/doctor)
                         + install   + serve   + apply        ← new
                           │            │          │
                  ┌────────┘     ┌──────┘     └──────┐
                  ▼              ▼                   ▼
            install.py        serve.py            apply.py
         (settings.local    (build vLLM        (git apply --3way
          merge, template     command from        a reviewed diff
          copy, yaml init)    registry)           to the main tree)

  Real-run loop:
    factory serve qwen   →   vLLM up on :8000 (correct tool-parser)
    factory up           →   proxy on :8787
    factory install      →   project wired to proxy
    Orchestrator dispatches `local-coder` (worktree isolation)
         → local model edits + tests IN THE WORKTREE
         → returns `git diff HEAD` + summary (no commit, main untouched)
    Parent (Opus) reviews the diff
    factory apply --check → factory apply   →   change lands on main
```

### 3.1 New units

| Unit | Responsibility | Depends on | Pure? |
|------|----------------|-----------|-------|
| `install.py` | Wire a project: settings.local.json env merge, template copy, factory.yaml init | `config` (defaults), filesystem, json | No (filesystem) |
| `serve.py` | Build the vLLM argv from a `ModelConfig`; run it (or print for remote) | `config`, subprocess | Arg-building is pure; launch is not |
| `apply.py` | Apply / dry-run a unified diff against the current git repo | git (subprocess) | No (git) |

Each is independently testable: `install` against tmp dirs, `serve`'s argv builder as a pure function, `apply` against a temp git repo.

---

## 4. Component detail

### 4.1 `factory install`

**Behavior (idempotent, safe):**

1. **factory.yaml:** if `./factory.yaml` is absent, copy `factory.example.yaml` → `factory.yaml`. If present, leave it.
2. **settings.local.json:** read `.claude/settings.local.json` (create `{}` if absent). Back up the existing file to `.claude/settings.local.json.bak` before writing. Merge an `env` object, setting exactly:
   - `ANTHROPIC_BASE_URL` = `http://{proxy.host}:{proxy.port}` (from `factory.yaml`)
   - `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY` = `"1"`
   Preserve every other key and every other `env` entry. Re-running changes nothing if the values already match.
3. **template:** copy `templates/agents/local-coder.md` → `.claude/agents/local-coder.md`. If the destination exists, skip unless `--force`.
4. Print a concise next-steps summary (serve, up, doctor).

**Interfaces:**
- `install_project(root: Path, config: FactoryConfig, force: bool = False) -> InstallResult` where `InstallResult` lists actions taken (created/updated/skipped per file). The CLI command prints them.
- Helper `merge_settings_env(existing: dict, host: str, port: int) -> dict` — pure, returns the new settings dict; unit-tested directly.

**Errors:** if `.claude/` cannot be created or files are unwritable, fail with a clear message and make no partial changes to settings (write to a temp file then atomically replace).

### 4.2 `factory serve <alias>`

**Behavior:** look up the alias in the registry → its `ModelConfig` and `BackendConfig`. Build the vLLM command:

```
vllm serve <model> --port <port-from-base_url> --host 127.0.0.1
    --enable-auto-tool-choice --tool-call-parser <tool_parser>
    [--guided-decoding-backend xgrammar  if guided_decoding]
    [--max-model-len <context>  if context set]
```

- The port is parsed from the backend `base_url` (`http://127.0.0.1:8000/v1` → `8000`).
- If the backend host is **not** local (not 127.0.0.1/localhost), do not launch; print the exact command to run on that host instead, and exit 0.
- If `tool_parser` is unset, fail with a clear message (tool calling would silently break — this is the whole point of the helper).
- If the `vllm` executable is not found, print an install hint and exit non-zero.

**Interfaces:**
- `build_vllm_argv(mc: ModelConfig, backend: BackendConfig) -> list[str]` — **pure**, the unit-tested core.
- `serve(config, alias, dry_run=False) -> int` — resolves, builds argv, and either prints (remote/`--dry-run`) or `os.execvp`/subprocess-runs vLLM.

### 4.3 `factory apply [diff]`

**Behavior:** read a unified diff from the path argument or stdin. Operate on the current git repo (error if not one).

- `--check`: run `git apply --check --3way` (dry run). Exit 0 if it would apply cleanly, non-zero with the conflicting paths otherwise. No changes.
- default: `git apply --3way` the diff. On success, print the changed files. On conflict/failure, ensure the working tree is left clean (git apply is atomic — it applies all or nothing) and report the conflicting paths with a non-zero exit.
- Reject an empty diff with a clear message.

**Interfaces:**
- `apply_diff(diff_text: str, check_only: bool = False, cwd: Path | None = None) -> ApplyResult` where `ApplyResult` has `ok: bool`, `changed_files: list[str]`, `message: str`. Unit-tested against a temp git repo.

### 4.4 `local-coder` template refinement

Update `templates/agents/local-coder.md` so its contract makes the diff-in-message loop work:

- It operates **only** inside its assigned worktree (dispatched with worktree isolation).
- After editing and running the relevant tests in the worktree, it runs `git --no-pager diff HEAD` and returns that **exact output inside a single fenced ```diff block**, followed by a one-paragraph summary (what changed, test result).
- It does **not** `git commit` and does **not** touch anything outside the worktree. The diff (against `HEAD`, the shared base) is the only artifact; `factory apply` consumes it on the main tree.

### 4.5 `docs/first-run.md`

A reproducible walkthrough for the NVIDIA box:

1. `cp factory.example.yaml factory.yaml` (edit model/alias if needed)
2. `factory serve local/qwen3-coder` (vLLM up on :8000)
3. `factory up` (proxy on :8787) and `factory doctor` (smoke test passes)
4. `factory install` (project wired)
5. In Claude Code: orchestrator dispatches `local-coder` (worktree isolation) on a small, well-specified task
6. Parent reviews the returned ```diff block
7. Save the diff and `factory apply --check` then `factory apply`
8. Run the project's tests to confirm the change is good

---

## 5. End-to-end flow (acceptance)

The slice is "done" when, on the NVIDIA box, the walkthrough produces: a local Qwen model generating a valid `git diff HEAD` for a small real task, that `factory apply --check` accepts, that `factory apply` lands on main, and that the project's tests pass afterward. We run this once together and capture the result.

---

## 6. Safety model

- The local model writes only inside its worktree; **main is never modified by the model.**
- Nothing reaches main until the parent has reviewed the diff and `factory apply` succeeds.
- `git apply` is all-or-nothing; a conflicting or malformed diff is rejected and main stays clean.
- `factory install` backs up `settings.local.json` and writes atomically; it never clobbers unrelated keys.

---

## 7. Testing strategy

- **`install`:** unit tests against tmp dirs — env merge is idempotent; existing keys preserved; backup created; template copy skips/uses `--force`; factory.yaml created only when absent; `merge_settings_env` pure-function cases.
- **`serve`:** `build_vllm_argv` returns the exact expected argv for a representative `ModelConfig` (parser flag present, port parsed from base_url, guided-decoding flag gated on the config, remote host → print-not-launch); missing `tool_parser` errors.
- **`apply`:** temp git repo — a valid diff applies and changes the right files; `--check` reports cleanliness without changing anything; a conflicting diff is rejected and leaves the tree clean; empty diff errors.
- **CLI wiring:** `install`/`serve --dry-run`/`apply --check` invoke their modules and surface results/exit codes.
- **Live run:** manual, on the box (non-deterministic model output → not in CI).

---

## 8. File structure changes

```
src/agentfactory/install.py    (new)
src/agentfactory/serve.py      (new)
src/agentfactory/apply.py      (new)
src/agentfactory/cli.py        (modify: add install/serve/apply commands)
templates/agents/local-coder.md (modify: diff-in-message contract)
docs/first-run.md              (new)
tests/test_install.py          (new)
tests/test_serve.py            (new)
tests/test_apply.py            (new)
tests/test_cli.py              (modify: wiring for new commands)
```

---

## 9. Open questions (track during implementation, not blockers)

- Exact `git apply` flags for best 3-way behavior on diffs generated in a sibling worktree (verify `--3way` resolves against the shared base as expected).
- Whether `factory serve` should background vLLM or run it in the foreground (default: foreground; the operator backgrounds it). Decide in the plan.
- vLLM flag names can drift between versions — pin the tested vLLM version in `docs/first-run.md`.
