# src/agentfactory/cli.py
from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

import httpx
import typer
import uvicorn

from .config import FactoryConfig
from .proxy import create_app
from .smoke import smoke_test_model

app = typer.Typer(help="agentFactory control CLI")

DEFAULT_CONFIG = "factory.yaml"


def _pid_path() -> Path:
    return Path(".factory") / "proxy.pid"


def _base_url(cfg: FactoryConfig) -> str:
    return f"http://{cfg.proxy.host}:{cfg.proxy.port}"


def _get_health(url: str, timeout: int = 5):
    return httpx.get(url, timeout=timeout)


@app.command()
def up(config: str = DEFAULT_CONFIG, daemon: bool = False) -> None:
    """Start the routing proxy."""
    cfg = FactoryConfig.load(config)
    if daemon:
        pid = os.fork()
        if pid > 0:
            _pid_path().parent.mkdir(parents=True, exist_ok=True)
            _pid_path().write_text(str(pid))
            typer.echo(f"proxy started (pid {pid}) on {_base_url(cfg)}")
            return
    uvicorn.run(create_app(cfg), host=cfg.proxy.host, port=cfg.proxy.port,
                log_level="info")


@app.command()
def down() -> None:
    """Stop a daemonized proxy."""
    p = _pid_path()
    if not p.exists():
        typer.echo("no pid file; proxy not running via daemon")
        raise typer.Exit(0)
    pid = int(p.read_text())
    try:
        os.kill(pid, signal.SIGTERM)
        typer.echo(f"stopped proxy (pid {pid})")
    finally:
        p.unlink(missing_ok=True)


@app.command()
def status(config: str = DEFAULT_CONFIG) -> None:
    """Show proxy health."""
    cfg = FactoryConfig.load(config)
    try:
        resp = _get_health(f"{_base_url(cfg)}/healthz")
    except httpx.HTTPError as exc:
        typer.echo(f"proxy unreachable: {exc}")
        raise typer.Exit(1)
    data = resp.json()
    typer.echo(f"status: {data['status']}")
    typer.echo(f"models: {', '.join(data['models'])}")


@app.command()
def doctor(config: str = DEFAULT_CONFIG) -> None:
    """Check proxy health and run a tool-call smoke test per local model."""
    cfg = FactoryConfig.load(config)
    try:
        health = _get_health(f"{_base_url(cfg)}/healthz")
    except httpx.HTTPError as exc:
        typer.echo(f"proxy unreachable: {exc}")
        raise typer.Exit(1)
    if health.status_code != 200:
        typer.echo(f"proxy unhealthy: http {health.status_code}")
        raise typer.Exit(1)
    typer.echo("proxy: OK")

    all_ok = True
    for alias in cfg.models:
        passed, detail = asyncio.run(smoke_test_model(cfg, alias))
        mark = "PASS" if passed else "FAIL"
        typer.echo(f"  {alias}: {mark} ({detail})")
        all_ok = all_ok and passed

    raise typer.Exit(0 if all_ok else 1)
