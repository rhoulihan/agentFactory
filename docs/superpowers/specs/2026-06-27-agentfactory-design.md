# agentFactory — Design Spec

**Date:** 2026-06-27
**Status:** Draft for review
**Author:** Rick Houlihan + Claude

---

## 1. Summary

agentFactory lets a Claude Code orchestrator running on a frontier Anthropic model (Opus) **farm out grunt-work code generation to local open-weight models that run as ordinary Claude Code subagents**. The orchestrator plans and reviews; local models do the high-volume mechanical generation; the parent process reviews every result before it touches the real workspace.

The whole system rests on one mechanism: a single **Anthropic-Messages-API-compatible routing proxy** sitting at `ANTHROPIC_BASE_URL` that dispatches each request **by its `model` field** — Claude model names pass through to `api.anthropic.com`, registered `local/*` aliases get translated and forwarded to a local inference backend. Because routing is keyed on the model name, a "local subagent" is nothing more than a normal `.claude/agents/*.md` file whose `model:` points at a local alias. No header-sniffing, no per-session env juggling, no fork of Claude Code.

### Core invariant: local generates, Anthropic reviews

**Local models are used *only* to generate code. Every review, verification, or judgment step runs on the parent Anthropic model — even when that step is delegated to a subagent.** A reviewer or adversarial-review subagent is an ordinary subagent whose `model:` is a Claude alias, so its requests take the passthrough path to `api.anthropic.com` automatically; it is never routed to a local backend. This keeps all quality gating at frontier capability while local models absorb only the grunt-work generation. The routing proxy enforces the split mechanically: `local/*` aliases are generation-only, and any subagent that reviews must carry a `claude-*` model.

### Goals

- Launch local-model subagents **the exact same way** as any Anthropic subagent (native `Agent` dispatch, native UX).
- Let the user/orchestrator **choose a local model per subagent** (frontmatter, per-invocation, or env).
- Keep the orchestrator and any reviewer subagents on Anthropic, automatically.
- Make the parent's **review of local output** a first-class, safe step (isolated git worktrees + diff review).
- Support both a **local NVIDIA workstation** and a **remote Ubuntu host** as backends, behind one abstraction.

### Drivers (all four, unified by "farm out the grunt work")

- **Cost** — spend frontier tokens only on planning + review.
- **Throughput / parallelism** — many concurrent code-gen subagents, no rate limits or per-token cost.
- **Privacy / offline** — source generation stays on local hardware.
- **Experimentation** — a swappable, observable platform for evaluating local models as supervised agents.

### Non-goals (v1)

- No custom inference engine (we serve via vLLM, not our own runtime).
- No fine-tuning or training.
- Not a general multi-provider gateway — only Anthropic passthrough + local backends.
- No GUI.
- LM Studio / MLX native-Anthropic endpoint and docker-compose remote deploy are **later-phase options**, not v1 core.

---

## 2. Background & key research findings

1. **Claude Code has no native per-subagent provider override.** `ANTHROPIC_BASE_URL` is global to a session; the subagent `model:` field nominally expects Anthropic IDs and is validated against an `availableModels` allowlist. Therefore per-subagent local routing is **not** achievable by configuration alone.
2. **Gateway model discovery is the unlock.** With `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`, Claude Code populates its model allowlist from the endpoint at `ANTHROPIC_BASE_URL` via a `/v1/models` discovery call. A proxy can therefore **advertise custom model names** (e.g. `local/qwen3-coder`) that become valid values for a subagent's `model:` field and the `Agent` tool's per-invocation `model` parameter.
3. **Routing by model name is clean and robust.** Since every request (orchestrator and subagent alike) flows through the one proxy and carries a `model` field, the proxy routes deterministically on that field. Orchestrator requests carry a `claude-*` name → passthrough; local subagent requests carry `local/*` → local backend. No fragile request introspection.
4. **Tool-calling reliability is the real bottleneck**, not raw code quality. Local models fail agentic loops by emitting malformed tool JSON, hallucinating tool names, or breaking streamed tool-call deltas. **Schema-constrained / guided decoding** (vLLM xgrammar, llama.cpp GBNF) largely fixes this and is therefore mandatory in our design.
5. **vLLM is the right primary backend** for the target hardware (NVIDIA + remote Ubuntu): best batched throughput for concurrent subagents, most mature tool-parser + guided-decoding stack. Default models: **Qwen3-Coder-30B-A3B** or **Qwen2.5-Coder-32B-Instruct** at Q4_K_M or better. Avoid Ollama on the streaming-tool-call path.
6. **Translation is solved-but-fragile.** Existing proxies (claude-code-router, LiteLLM `/v1/messages`, UniClaudeProxy) demonstrate Anthropic↔OpenAI translation, but every one's weak point is faithfully round-tripping `tool_use`/`tool_result` IDs and streamed tool-argument deltas. We hand-roll a focused translator rather than depend on an experimental passthrough path, and cover it with smoke tests.

### Resolution order Claude Code uses for a subagent's model (for reference)

1. `CLAUDE_CODE_SUBAGENT_MODEL` env var
2. Per-invocation `model` parameter on the `Agent` tool call
3. Subagent definition `model:` frontmatter
4. Main conversation's model

All three of the first three are valid ways for the user/orchestrator to **choose local** — they each just need to name a registered `local/*` alias.

---

## 3. Architecture

```
Claude Code (orchestrator, Anthropic Opus)
        │  ALL model requests → ANTHROPIC_BASE_URL (the proxy)
        ▼
┌──────────────────────────────────────────────────────────┐
│  agentFactory Routing Proxy  (FastAPI, Anthropic Msgs API) │
│                                                            │
│   /v1/models    → discovery: Claude passthrough + local/*  │
│   /v1/messages  → route by request `model` field:          │
│        claude-*   → forward to api.anthropic.com           │
│        local/*    → translate ↔ OpenAI → local backend     │
│                     (guided decoding, tool-call repair)    │
└──────────────────────────────────────────────────────────┘
   claude-* │                              │ local/*
            ▼                              ▼
   api.anthropic.com               vLLM (OpenAI-compatible)
   (orchestrator + reviewers)      on NVIDIA box or remote
                                   Ubuntu host. Qwen3-Coder,
                                   xgrammar guided decoding.
```

### 3.1 Components

| # | Component | Responsibility | Depends on |
|---|-----------|----------------|------------|
| 1 | **Routing proxy** | Anthropic Messages API server; route by model; translate; stream; guided decoding; repair | FastAPI, httpx, Anthropic + OpenAI wire formats |
| 2 | **Model registry** | Declarative config of local aliases → backends; source of `/v1/models` | `factory.yaml` |
| 3 | **`factory` CLI** | Lifecycle + setup + diagnostics | proxy, registry |
| 4 | **Subagent templates** | Ready-made local-model agent definitions | `.claude/agents/` |
| 5 | **Worktree review workflow** | Isolation + diff-review + merge helper | git worktrees, `Agent` isolation |

Each component is independently testable and communicates over explicit interfaces (HTTP for the proxy, YAML for the registry, files for templates, git for the review loop).

---

## 4. Component detail

### 4.1 Routing proxy

**Form:** A FastAPI app (Python) run as a local daemon (default `http://127.0.0.1:8787`). Claude Code points at it via `ANTHROPIC_BASE_URL`.

**Endpoints**

- `POST /v1/messages` — the core. Accepts Anthropic Messages requests (streaming and non-streaming).
- `GET /v1/models` — discovery. Returns the union of (a) a configured Anthropic passthrough model list and (b) all `local/*` aliases from the registry. Enables `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY`.
- `GET /healthz` — liveness + per-backend reachability, used by `factory doctor`.

**Routing logic** (`/v1/messages`):

1. Read `model` from the request body.
2. If it matches a passthrough pattern (default `claude-*`, plus any explicitly configured) → **forward** to `https://api.anthropic.com` (configurable upstream). Preserve method, body, `anthropic-version`, auth headers (`x-api-key` / `Authorization`), and stream the SSE response back verbatim. The orchestrator and any Anthropic-backed reviewer subagents take this path with zero transformation.
3. If it matches a registered `local/*` alias → **translate** and forward to that alias's backend (§4.1.1), then translate the response back to Anthropic shape.
4. Unknown model → `404`-style Anthropic error block with a clear message.

**Auth:** The proxy passes the caller's Anthropic credentials straight through on the passthrough path. Local backends use a per-backend token from the registry (often a dummy like `vllm`). The proxy never logs credentials.

#### 4.1.1 Translation module (Anthropic ↔ OpenAI)

A pure, well-tested module with no I/O. Two directions:

**Request (Anthropic → OpenAI chat/completions):**
- Lift top-level `system` into a leading `{role:"system"}` message.
- Map content blocks: text → string/parts; `tool_use` (assistant) → `assistant.tool_calls[]` with `function.name` and **JSON-stringified** `arguments`; `tool_result` (user) → a `{role:"tool", tool_call_id, content}` message. Preserve/translate the `tool_use_id` ↔ `tool_call_id` pairing exactly.
- Map `tools` (Anthropic `input_schema`) → OpenAI `tools[].function.parameters`.
- Carry `max_tokens`, `temperature`, `stop_sequences` → `stop`.

**Guided decoding:** when `tools` are present, attach the selected tool's JSON schema as a constrained-decoding directive for the backend (vLLM: `guided_json` / `tool_choice` + xgrammar; llama.cpp: GBNF). Tool-call arguments are thereby forced to validate against the schema. The correct **per-model tool-call parser** (e.g. `qwen3_coder`, `hermes`, `llama3_json`) comes from the registry.

**Response (OpenAI → Anthropic):**
- `choices[].message.content` → Anthropic `text` block(s).
- `choices[].message.tool_calls[]` → Anthropic `tool_use` blocks (parse `arguments` JSON string back into an object; mint stable `tool_use` ids).
- `finish_reason` → `stop_reason`: `tool_calls`→`tool_use`, `length`→`max_tokens`, `stop`→`end_turn`, `stop_sequence` preserved. (`tool_calls`↔`tool_use` is the load-bearing mapping for the agent loop.)
- `usage` → Anthropic `input_tokens`/`output_tokens`.

**Streaming (SSE) re-framing:** consume the backend's OpenAI `chat.completion.chunk` stream and emit the Anthropic event sequence: `message_start` → `content_block_start`/`content_block_delta`(text or `input_json_delta` for tool args)/`content_block_stop` → `message_delta`(stop_reason, usage) → `message_stop`. Tool-argument deltas are reassembled per index.

#### 4.1.2 Tool-call repair

When a local backend (despite guided decoding) returns a malformed or unparseable tool call, the proxy attempts a bounded repair before surfacing an error: re-issue the backend call with stricter `tool_choice`/grammar, or coerce/parse-and-fix the arguments. Repairs and their outcomes are logged. A configurable cap prevents loops; on exhaustion the proxy returns a clean Anthropic error so the subagent can recover.

#### 4.1.3 Observability

Structured per-request logs: route taken (passthrough vs `local/<alias>`), backend, latency, token counts, tool-call count + validity, repair attempts. Credentials never logged. A rollup feeds `factory` reporting of the **local-vs-Anthropic spend split** (the cost driver) and tool-call success rate per model.

### 4.2 Model registry (`factory.yaml`)

Declarative. A backend is **just a URL**, so the local NVIDIA box and a remote Ubuntu host are the same abstraction (only the URL differs).

```yaml
proxy:
  host: 127.0.0.1
  port: 8787
  upstream_anthropic: https://api.anthropic.com
  passthrough_models: ["claude-*"]      # everything else must be a known alias

backends:
  local-nvidia:
    engine: vllm
    base_url: http://127.0.0.1:8000/v1
    api_key: vllm
  remote-ubuntu:
    engine: vllm
    base_url: http://10.0.0.20:8000/v1
    api_key: ${REMOTE_VLLM_KEY}

models:
  local/qwen3-coder:                    # the alias Claude Code sees
    backend: local-nvidia
    model: Qwen/Qwen3-Coder-30B-A3B-Instruct
    tool_parser: qwen3_coder
    guided_decoding: true
    context: 65536
  local/qwen-coder-32b:
    backend: remote-ubuntu
    model: Qwen/Qwen2.5-Coder-32B-Instruct
    tool_parser: hermes
    guided_decoding: true
    context: 32768
```

`/v1/models` is generated from `passthrough_models` + the `models` map.

### 4.3 `factory` CLI

| Command | Does |
|---------|------|
| `factory up` / `factory down` | Start/stop the proxy daemon (reads `factory.yaml`). |
| `factory status` | Show proxy + backend health, active models. |
| `factory doctor` | End-to-end check: proxy live, each backend reachable, **a real tool-call smoke test** per local model (round-trip a tiny "call this tool" prompt and assert a valid `tool_use`). |
| `factory models` | `list` / `add` / `remove` / `test <alias>` local models. |
| `factory install` | Configure a project: set `ANTHROPIC_BASE_URL` + `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` in `.claude/settings.json`, and copy local-subagent templates into `.claude/agents/`. |
| `factory report` | Local-vs-Anthropic spend split, tool-call success rates, latency. |

`factory serve` helpers to launch vLLM with the right flags are **optional convenience** (P2); the registry can also point at an already-running vLLM.

### 4.4 Local subagent templates

Ready-made `.claude/agents/*.md` installed by `factory install`. Example `local-coder.md`:

- `model: local/qwen3-coder`
- `description:` scoped to mechanical/high-volume generation so the orchestrator delegates appropriately, and explicit that it is for **well-specified** tasks (local models degrade on long-horizon autonomy).
- Body instructs the subagent to: work only within its assigned worktree, make the change, run the relevant tests/build, and **return a concise diff + summary** rather than a narrative.

Templates are parameterizable (which alias, which task shape). Users can also hand-author agents pointing at any registered alias, or pass `model: local/...` per `Agent` invocation, or set `CLAUDE_CODE_SUBAGENT_MODEL` — all three honor the registry.

### 4.5 Worktree review workflow (parent process review)

The defining workflow. A local subagent's output is **isolated until the parent approves it.**

1. **Dispatch in isolation.** The orchestrator launches a local subagent with worktree isolation (the `Agent` tool's `isolation: "worktree"`), so the subagent operates on an isolated copy of the repo on its own branch.
2. **Generate + self-check.** The local model makes the edit and runs the relevant tests/build inside the worktree. It cannot touch the main tree.
3. **Return a diff.** The subagent returns a unified diff + a short summary (what changed, test results).
4. **Parent review (always Anthropic).** The diff is reviewed against the original spec — correctness, convention adherence, scope — by the parent Anthropic model. This is true whether the orchestrator (Opus) reviews inline or delegates to a reviewer subagent: per the core invariant, any review/adversarial-review subagent carries a `claude-*` model and takes the passthrough path. **Review is never performed by a local model.** This is where frontier judgment is spent.
5. **Decide.** Merge (helper applies the worktree branch to the main tree), iterate (send specific feedback back to the same or a fresh local subagent), or discard (drop the worktree; nothing leaked).

A small **merge helper** (`factory merge <worktree>` or an equivalent the orchestrator calls) encapsulates step 5's apply path and cleans up the worktree. Optional later: a dedicated Anthropic-backed **reviewer subagent** (and adversarial-verification panel) that audits each diff before the parent accepts it (P3) — Anthropic-backed by construction.

---

## 5. End-to-end flow

```
Setup:   factory up               # proxy daemon comes up
         factory doctor           # backends reachable, tool-call smoke test passes
         factory install          # project .claude/ configured, templates installed

Work:    Orchestrator (Opus) plans the task, decomposes into grunt-work units.
         For each unit → Agent(local-coder, isolation: worktree)
              → request hits proxy, model = local/qwen3-coder
              → translate → vLLM generates with guided decoding
              → subagent edits in worktree, runs tests
              → returns diff + summary
         Opus reviews each diff → merge / iterate / discard
         Opus integrates, runs the full suite, reports.
```

---

## 6. Reliability & risks

| Risk | Mitigation |
|------|------------|
| Claude Code request-format / gateway-discovery behavior drifts | Pin observed behavior behind a thin compat layer; `factory doctor` exercises discovery + a live round-trip; integration tests against recorded Claude Code traffic. |
| Per-model tool-call fidelity (malformed JSON, wrong tool names) | Mandatory guided decoding; correct per-model tool parser from registry; bounded tool-call repair; smoke test gates a model as "ready." |
| Streaming SSE edge cases (tool-arg deltas, partial chunks) | Hand-rolled re-framer with a focused unit-test corpus of real chunk sequences; non-streaming fallback if a backend's stream is unreliable. |
| Local model too weak for the task | Scoped subagent descriptions; worktree isolation means failures are discarded safely; orchestrator can escalate a unit to Anthropic. |
| Weak/no native tool calling in a chosen model | Optional ReAct/XML tool-call fallback adapter (P3 escape hatch). |
| Remote backend latency/availability | Per-backend health in `doctor`/`status`; clean Anthropic error on backend failure so the loop degrades gracefully. |
| Credential leakage | Passthrough preserves caller creds without logging; local tokens kept out of logs. |

---

## 7. Phasing

**Phase 1 — Core proves the mechanism**
- Routing proxy: passthrough + one local backend, `/v1/messages` (streaming + non-streaming), `/v1/models` discovery.
- Translation module + guided decoding + SSE re-framing.
- `factory.yaml` registry; `factory up` / `down` / `doctor` (incl. tool-call smoke test).
- One `local-coder` template; manual worktree review by the orchestrator.
- **Exit criteria:** Opus orchestrator dispatches a `local-coder` subagent that generates a correct, tested change in a worktree via a local Qwen model; orchestrator reviews and merges the diff; passthrough requests are unaffected.

**Phase 2 — Usable day-to-day**
- `factory install`, `factory models`, `factory report`.
- Worktree merge helper.
- Multiple local aliases; **remote Ubuntu backend**.
- Tool-call repair; observability + cost split.

**Phase 3 — Robustness & reach**
- docker-compose stack for reproducible remote deploy.
- LM Studio / MLX native-Anthropic endpoint path (zero-translation backend type) for Mac users.
- Dedicated reviewer-subagent + adversarial-verification panel (Anthropic-backed by construction).
- ReAct/XML fallback for weak-tool models.

---

## 8. Testing strategy

- **Translation unit tests** — golden Anthropic↔OpenAI fixtures covering text, single/parallel tool calls, tool_result round-trips, stop-reason mapping.
- **Streaming corpus tests** — recorded OpenAI chunk streams → asserted Anthropic event sequences, including split tool-arg deltas.
- **Proxy routing tests** — `claude-*` passthrough vs `local/*` dispatch vs unknown model.
- **Live smoke tests** (`factory doctor`) — per local model, force a tool call and assert a valid `tool_use`.
- **End-to-end** — scripted orchestrator → local-coder → worktree diff → merge, on a sample repo, asserting tests pass post-merge.

---

## 9. Open questions (track during implementation, not blockers)

- Exact shape Claude Code expects from `/v1/models` discovery (verify empirically and pin).
- Whether `local/` prefix in alias names causes any client-side validation friction; fall back to flat names if so.
- Best default vLLM flags per supported model (tool parser + guided-decoding combo) — establish a tested matrix.
