# Phase 2 Backlog

Carried forward from the Phase 1 build's whole-branch review (deferred, non-blocking for Phase 1). See the spec's §7 phasing for the larger Phase 2/3 scope; this file captures the specific items the review surfaced.

## From the Phase 1 final review (Minor)

- **Propagate streamed token usage.** `streaming.py` emits `message_start` usage `{input_tokens:0, output_tokens:0}` and `message_delta` `{output_tokens:0}` — always zero. Request `stream_options.include_usage` from the backend and thread real counts through so Anthropic clients see accurate streamed usage. (Feeds the Phase 2 cost/observability report.)
- **Sanitize backend error bodies.** `proxy.py` embeds the full raw backend JSON in the client-facing `anthropic_error("backend error: ...")`. No credential leak (api_key never appears in response bodies), but it leaks backend internals — summarize/truncate before returning.
- **Forward more upstream headers on passthrough.** `proxy.py` reconstructs the passthrough `StreamingResponse` using only `content-type`, dropping `request-id`, `anthropic-*`, and rate-limit headers clients may rely on. Forward the safe upstream response headers.

## Coverage gaps to close in Phase 2

- **`stream_upstream` (passthrough path) has no direct unit test.** Covered indirectly via the proxy passthrough test (which monkeypatches it), but the function itself is untested. Add a respx-backed unit test.
- **CLI `up`/`down`/`status` have no direct tests** (only `_pid_path` and `doctor` are exercised). Add coverage when `up` gains the `install` flow.

## Larger Phase 2 scope (from spec §7)

- `factory install`, `factory models`, `factory report`.
- Worktree merge helper.
- Multiple local aliases; exercise the remote Ubuntu backend end-to-end.
- Tool-call repair; observability + local-vs-Anthropic cost split.
