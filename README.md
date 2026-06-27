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
