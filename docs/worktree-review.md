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
