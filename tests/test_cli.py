# tests/test_cli.py
from typer.testing import CliRunner
from agentfactory.cli import app, _pid_path

runner = CliRunner()


def test_pid_path_is_under_dot_factory():
    assert _pid_path().as_posix().endswith(".factory/proxy.pid")


def test_doctor_reports_pass(monkeypatch, tmp_path):
    cfg = tmp_path / "factory.yaml"
    cfg.write_text(
        "proxy: {host: 127.0.0.1, port: 8787}\n"
        "backends: {b: {engine: vllm, base_url: 'http://be/v1', api_key: k}}\n"
        "models: {local/x: {backend: b, model: m}}\n"
    )

    import agentfactory.cli as cli_mod

    async def fake_smoke(config, alias, base_url=None):
        return True, "ok"

    def fake_health(url, timeout=5):
        class R:
            status_code = 200
            def json(self_inner):
                return {"status": "ok", "models": ["local/x"]}
        return R()

    monkeypatch.setattr(cli_mod, "smoke_test_model", fake_smoke)
    monkeypatch.setattr(cli_mod, "_get_health", fake_health)

    result = runner.invoke(app, ["doctor", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "local/x" in result.stdout
    assert "PASS" in result.stdout


def test_doctor_fails_when_smoke_fails(monkeypatch, tmp_path):
    cfg = tmp_path / "factory.yaml"
    cfg.write_text(
        "proxy: {host: 127.0.0.1, port: 8787}\n"
        "backends: {b: {engine: vllm, base_url: 'http://be/v1', api_key: k}}\n"
        "models: {local/x: {backend: b, model: m}}\n"
    )

    import agentfactory.cli as cli_mod

    async def fake_smoke(config, alias, base_url=None):
        return False, "no tool_use in response"

    def fake_health(url, timeout=5):
        class R:
            status_code = 200
            def json(self_inner):
                return {"status": "ok", "models": ["local/x"]}
        return R()

    monkeypatch.setattr(cli_mod, "smoke_test_model", fake_smoke)
    monkeypatch.setattr(cli_mod, "_get_health", fake_health)

    result = runner.invoke(app, ["doctor", "--config", str(cfg)])
    assert result.exit_code == 1
    assert "FAIL" in result.stdout


# --- append to tests/test_cli.py ---
import agentfactory.cli as cli_mod
from agentfactory.install import InstallResult
from agentfactory.apply import ApplyResult


def test_install_command_invokes_install_project(monkeypatch, tmp_path):
    called = {}

    def fake_install(root, cfg, force=False, package_root=None):
        called["root"] = root
        called["force"] = force
        return InstallResult(actions=["created factory.yaml", "wrote settings.local.json env"])

    monkeypatch.setattr(cli_mod, "install_project", fake_install)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["install"])
    assert result.exit_code == 0
    assert "settings.local.json" in result.stdout
    assert called["force"] is False


def test_serve_command_returns_serve_vllm_code(monkeypatch, tmp_path):
    cfg = tmp_path / "factory.yaml"
    cfg.write_text(
        "proxy: {host: 127.0.0.1, port: 8787}\n"
        "backends: {b: {engine: vllm, base_url: 'http://127.0.0.1:8000/v1', api_key: k}}\n"
        "models: {local/x: {backend: b, model: m, tool_parser: hermes}}\n"
    )

    def fake_serve(config, alias, dry_run=False):
        assert alias == "local/x"
        assert dry_run is True
        return 0

    monkeypatch.setattr(cli_mod, "serve_vllm", fake_serve)
    result = runner.invoke(app, ["serve", "local/x", "--config", str(cfg), "--dry-run"])
    assert result.exit_code == 0


def test_apply_command_reads_stdin_and_reports(monkeypatch):
    def fake_apply(text, check_only=False, cwd=None):
        assert "diff --git" in text
        assert check_only is True
        return ApplyResult(True, ["a.txt"], "checked")

    monkeypatch.setattr(cli_mod, "apply_diff", fake_apply)
    result = runner.invoke(app, ["apply", "--check"],
                           input="diff --git a/a.txt b/a.txt\n")
    assert result.exit_code == 0
    assert "a.txt" in result.stdout


def test_apply_command_nonzero_on_failure(monkeypatch):
    monkeypatch.setattr(cli_mod, "apply_diff",
                        lambda text, check_only=False, cwd=None: ApplyResult(False, [], "empty diff"))
    result = runner.invoke(app, ["apply"], input="x")
    assert result.exit_code == 1
