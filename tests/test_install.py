# tests/test_install.py
import json
from pathlib import Path
from agentfactory.config import FactoryConfig, ProxyConfig
from agentfactory.install import merge_settings_env, install_project


def test_merge_settings_env_sets_keys():
    out = merge_settings_env({}, "127.0.0.1", 8787)
    assert out["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert out["env"]["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"


def test_merge_settings_env_preserves_other_keys():
    existing = {"permissions": {"allow": ["Bash"]},
                "env": {"FOO": "bar"}}
    out = merge_settings_env(existing, "127.0.0.1", 8787)
    assert out["permissions"] == {"allow": ["Bash"]}
    assert out["env"]["FOO"] == "bar"
    assert out["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def _pkg(tmp_path: Path) -> Path:
    pkg = tmp_path / "pkg"
    (pkg / "templates" / "agents").mkdir(parents=True)
    (pkg / "factory.example.yaml").write_text("proxy:\n  host: 127.0.0.1\n  port: 8787\n")
    (pkg / "templates" / "agents" / "local-coder.md").write_text("---\nname: local-coder\n---\nbody\n")
    return pkg


def _cfg():
    return FactoryConfig(proxy=ProxyConfig(host="127.0.0.1", port=8787))


def test_install_creates_files(tmp_path):
    pkg = _pkg(tmp_path)
    target = tmp_path / "proj"
    target.mkdir()
    res = install_project(target, _cfg(), package_root=pkg)
    assert (target / "factory.yaml").exists()
    slj = json.loads((target / ".claude" / "settings.local.json").read_text())
    assert slj["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert (target / ".claude" / "agents" / "local-coder.md").exists()
    assert any("settings.local.json" in a for a in res.actions)


def test_install_idempotent_and_backs_up(tmp_path):
    pkg = _pkg(tmp_path)
    target = tmp_path / "proj"
    target.mkdir()
    install_project(target, _cfg(), package_root=pkg)
    # second run: backup created, env still correct, template skipped
    res = install_project(target, _cfg(), package_root=pkg)
    assert (target / ".claude" / "settings.local.json.bak").exists()
    slj = json.loads((target / ".claude" / "settings.local.json").read_text())
    assert slj["env"]["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
    assert any("skipped" in a for a in res.actions)


def test_install_preserves_existing_settings(tmp_path):
    pkg = _pkg(tmp_path)
    target = tmp_path / "proj"
    (target / ".claude").mkdir(parents=True)
    (target / ".claude" / "settings.local.json").write_text(
        json.dumps({"permissions": {"allow": ["Bash"]}}))
    install_project(target, _cfg(), package_root=pkg)
    slj = json.loads((target / ".claude" / "settings.local.json").read_text())
    assert slj["permissions"] == {"allow": ["Bash"]}
    assert slj["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def test_install_does_not_overwrite_existing_factory_yaml(tmp_path):
    pkg = _pkg(tmp_path)
    target = tmp_path / "proj"
    target.mkdir()
    (target / "factory.yaml").write_text("proxy:\n  port: 9999\n")
    install_project(target, _cfg(), package_root=pkg)
    assert "9999" in (target / "factory.yaml").read_text()
