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
