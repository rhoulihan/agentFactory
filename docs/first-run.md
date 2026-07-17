# First real run (local NVIDIA box)

End-to-end walkthrough: a Claude Code orchestrator (Anthropic) dispatches the
`local-coder` subagent, a local Qwen model generates a change in an isolated
worktree, you review the diff, and `factory apply` lands it on the main tree.

> First verified with **vLLM 0.23.0** and **`Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`**
> on an RTX 4070 Ti SUPER (16 GB, WSL2, CUDA driver 591.86 / toolkit 12.0).
> Choose the model to fit your VRAM — see [Troubleshooting](#troubleshooting).

## 1. Configure

```bash
cp factory.example.yaml factory.yaml      # edit model/alias for your GPU if needed
```

## 2. Launch the local model

```bash
factory serve local/qwen-coder            # runs vLLM on :8000 with the correct
                                          # --tool-call-parser for the model
```

Leave it running (use a second terminal, or background it). To preview the
command without launching: `factory serve local/qwen-coder --dry-run`. On WSL or
an older CUDA toolkit, see [Troubleshooting](#troubleshooting) first.

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

## Troubleshooting

### vLLM crashes building a flashinfer kernel (`ninja` / `nvcc` errors)
On a fresh or older-CUDA box (e.g. CUDA toolkit 12.0), vLLM's flashinfer backend
JIT-compiles a sampling kernel at startup and fails with
`FileNotFoundError: 'ninja'` or a CUB compile error
(`... has no member "FlagHeads"`). Fix **without** upgrading CUDA by disabling
flashinfer and using the precompiled FlashAttention backend + native sampler:

```bash
uv tool install ninja                      # only if you hit the 'ninja' error

VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  factory serve local/qwen-coder --config factory.yaml
```

### Model sizing for your VRAM
- **24 GB+** : Qwen2.5-Coder-32B / Qwen3-Coder-30B at 4-bit.
- **16 GB**  : a 7B AWQ coder (e.g. `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`) with
  `extra_args: ["--quantization", "awq", "--gpu-memory-utilization", "0.6"]`
  and `context: 8192`. The 30B will OOM on 16 GB.
- Lower `--gpu-memory-utilization` to leave headroom for other GPU users
  (a desktop/WSL display already consumes a few GB).

### `vllm serve` flag errors
`factory serve` targets modern vLLM (>= 0.23): it emits `--enable-auto-tool-choice`,
`--tool-call-parser`, `--max-model-len`, and your `extra_args`. It does **not**
emit the removed `--guided-decoding-backend` (xgrammar is the default backend).
Pass any version-specific flags via `extra_args` in `factory.yaml`.
