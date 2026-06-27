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
