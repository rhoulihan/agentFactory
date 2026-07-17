# tests/test_apply.py
import subprocess
from pathlib import Path
from agentfactory.apply import apply_diff, ApplyResult


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, text=True)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "t@t"], repo)
    _git(["config", "user.name", "t"], repo)
    (repo / "a.txt").write_text("hello\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "init"], repo)
    return repo


def _diff_for_edit(repo: Path) -> str:
    (repo / "a.txt").write_text("hello world\n")
    diff = subprocess.run(["git", "--no-pager", "diff"], cwd=repo,
                          capture_output=True, text=True).stdout
    _git(["checkout", "--", "a.txt"], repo)  # restore
    return diff


def test_apply_valid_diff_changes_file(tmp_path):
    repo = _make_repo(tmp_path)
    diff = _diff_for_edit(repo)
    res = apply_diff(diff, cwd=repo)
    assert res.ok is True
    assert "a.txt" in res.changed_files
    assert (repo / "a.txt").read_text() == "hello world\n"


def test_check_only_does_not_modify(tmp_path):
    repo = _make_repo(tmp_path)
    diff = _diff_for_edit(repo)
    res = apply_diff(diff, check_only=True, cwd=repo)
    assert res.ok is True
    assert (repo / "a.txt").read_text() == "hello\n"  # unchanged


def test_conflicting_diff_rejected_tree_clean(tmp_path):
    repo = _make_repo(tmp_path)
    bogus = (
        "diff --git a/a.txt b/a.txt\n"
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -5,1 +5,1 @@\n"
        "-nonexistent line\n"
        "+changed\n"
    )
    res = apply_diff(bogus, cwd=repo)
    assert res.ok is False
    assert (repo / "a.txt").read_text() == "hello\n"  # untouched


def test_empty_diff_rejected(tmp_path):
    repo = _make_repo(tmp_path)
    res = apply_diff("   \n", cwd=repo)
    assert res.ok is False
    assert "empty" in res.message.lower()


def test_non_git_dir_rejected(tmp_path):
    res = apply_diff("diff --git a/x b/x\n", cwd=tmp_path)
    assert res.ok is False
    assert "git" in res.message.lower()


def test_real_diff_against_diverged_tree_is_rejected_clean(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "a.txt").write_text("hello patched\n")
    patch = subprocess.run(["git", "--no-pager", "diff", "HEAD"], cwd=repo,
                           capture_output=True, text=True).stdout
    _git(["checkout", "--", "a.txt"], repo)
    (repo / "a.txt").write_text("hello diverged\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "diverge"], repo)
    res = apply_diff(patch, cwd=repo)
    assert res.ok is False
    content = (repo / "a.txt").read_text()
    assert content == "hello diverged\n"
    assert "<<<<<<<" not in content
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                            capture_output=True, text=True).stdout
    assert status.strip() == ""


def test_check_predicts_conflict_on_diverged_tree(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "a.txt").write_text("hello patched\n")
    patch = subprocess.run(["git", "--no-pager", "diff", "HEAD"], cwd=repo,
                           capture_output=True, text=True).stdout
    _git(["checkout", "--", "a.txt"], repo)
    (repo / "a.txt").write_text("hello diverged\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "diverge"], repo)
    res = apply_diff(patch, check_only=True, cwd=repo)
    assert res.ok is False
