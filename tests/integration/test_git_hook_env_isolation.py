"""Regression tests for Git repository state inherited from hooks."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from apm_cli.cache.git_cache import GitCache
from apm_cli.cache.url_normalize import cache_shard_key
from apm_cli.deps.bare_cache import (
    bare_clone_with_fallback,
    clone_with_fallback,
    materialize_from_bare,
)
from apm_cli.deps.github_downloader import GitHubPackageDownloader
from apm_cli.deps.github_downloader_validation import AttemptSpec, _path_exists_in_tree_at_ref
from apm_cli.models.dependency.reference import DependencyReference
from apm_cli.utils.git_env import (
    GitUrlRewriteError,
    checkout_git_worktree,
    clone_git_worktree,
    get_git_executable,
    git_subprocess_env,
)

pytestmark = pytest.mark.component


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        [get_git_executable(), "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
        env=git_subprocess_env(),
    )
    return result.stdout.strip()


def _commit(repo: Path, content: str, message: str) -> str:
    (repo / "payload.txt").write_text(content, encoding="ascii")
    _git(repo, "add", "payload.txt")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def test_cache_refresh_ignores_linked_worktree_git_environment(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test")
    _commit(source, "old\n", "old")

    cache = GitCache(tmp_path / "cache")
    bare_dir = cache._db_root / cache_shard_key(str(source))
    subprocess.run(
        [get_git_executable(), "clone", "--bare", str(source), str(bare_dir)],
        check=True,
        capture_output=True,
        env=git_subprocess_env(),
    )
    dependency_sha = _commit(source, "new\n", "new")

    invoking = tmp_path / "invoking"
    invoking.mkdir()
    _git(invoking, "init")
    _git(invoking, "config", "user.email", "test@example.com")
    _git(invoking, "config", "user.name", "Test")
    invoking_sha = _commit(invoking, "invoking\n", "invoking")
    hook_worktree = tmp_path / "hook-worktree"
    _git(invoking, "worktree", "add", "-b", "hook-wt", str(hook_worktree))

    poisoned_env = dict(os.environ)
    poisoned_env["GIT_DIR"] = _git(hook_worktree, "rev-parse", "--absolute-git-dir")
    poisoned_env["GIT_WORK_TREE"] = str(hook_worktree)

    checkout = cache.get_checkout(
        str(source),
        dependency_sha,
        locked_sha=dependency_sha,
        env=poisoned_env,
    )

    assert _git(hook_worktree, "symbolic-ref", "--short", "HEAD") == "hook-wt"
    assert _git(hook_worktree, "rev-parse", "HEAD") == invoking_sha
    assert _git(checkout, "rev-parse", "HEAD") == dependency_sha


def test_fallback_checkout_targets_dependency_worktree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test")
    dependency_sha = _commit(source, "dependency\n", "dependency")

    target = tmp_path / "target"
    subprocess.run(
        [get_git_executable(), "clone", "--no-checkout", str(source), str(target)],
        check=True,
        capture_output=True,
        env=git_subprocess_env(),
    )

    invoking = tmp_path / "invoking"
    invoking.mkdir()
    _git(invoking, "init")
    _git(invoking, "config", "user.email", "test@example.com")
    _git(invoking, "config", "user.name", "Test")
    invoking_sha = _commit(invoking, "invoking\n", "invoking")
    hook_worktree = tmp_path / "hook-worktree"
    _git(invoking, "worktree", "add", "-b", "hook-wt", str(hook_worktree))
    _git(hook_worktree, "fetch", str(source), dependency_sha)

    poisoned_env = dict(os.environ)
    poisoned_env["GIT_DIR"] = _git(hook_worktree, "rev-parse", "--absolute-git-dir")
    poisoned_env["GIT_WORK_TREE"] = str(hook_worktree)

    checkout_git_worktree(target, dependency_sha, env=poisoned_env)

    assert _git(target, "rev-parse", "HEAD") == dependency_sha
    assert _git(hook_worktree, "symbolic-ref", "--short", "HEAD") == "hook-wt"
    assert _git(hook_worktree, "rev-parse", "HEAD") == invoking_sha


def test_materialize_ignores_repository_local_git_config_override(tmp_path: Path) -> None:
    """GIT_CONFIG cannot redirect dependency configuration writes."""
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test")
    dependency_sha = _commit(source, "dependency\n", "dependency")
    bare = tmp_path / "source.git"
    subprocess.run(
        [get_git_executable(), "clone", "--bare", str(source), str(bare)],
        check=True,
        capture_output=True,
        env=git_subprocess_env(),
    )

    invoking = tmp_path / "invoking"
    invoking.mkdir()
    _git(invoking, "init")
    _git(invoking, "config", "user.email", "test@example.com")
    _git(invoking, "config", "user.name", "Test")
    invoking_config = invoking / ".git" / "config"
    original_config = invoking_config.read_bytes()
    poisoned_env = {
        **os.environ,
        "GIT_CONFIG": str(invoking_config),
    }

    materialize_from_bare(
        bare,
        tmp_path / "consumer",
        ref=None,
        env=poisoned_env,
        known_sha=dependency_sha,
    )

    assert invoking_config.read_bytes() == original_config


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX Git hook executable fixture")
def test_dependency_clone_disables_repository_checkout_hook(tmp_path: Path) -> None:
    """A dependency-controlled post-checkout hook cannot execute during clone."""
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test")
    marker = tmp_path / "hook-ran"
    hook = source / ".githooks" / "post-checkout"
    hook.parent.mkdir()
    hook.write_text(
        f"#!/bin/sh\nprintf ran > {shlex.quote(str(marker))}\n",
        encoding="ascii",
    )
    hook.chmod(0o755)
    _commit(source, "dependency\n", "dependency with hook")
    _git(source, "add", ".githooks/post-checkout")
    _git(source, "commit", "-m", "add checkout hook")
    global_config = tmp_path / "gitconfig"
    global_config.write_text("[core]\n\thooksPath = .githooks\n", encoding="ascii")

    clone_git_worktree(
        str(source),
        tmp_path / "consumer",
        env={
            "PATH": os.environ["PATH"],
            "GIT_CONFIG_GLOBAL": str(global_config),
            "GIT_CONFIG_NOSYSTEM": "1",
        },
    )

    assert not marker.exists()


def test_full_clone_fallback_replaces_poisoned_process_environment(tmp_path: Path) -> None:
    """Working-tree clone must replace, not overlay, the Git child environment."""
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test")
    dependency_sha = _commit(source, "dependency\n", "dependency")

    invoking = tmp_path / "invoking"
    invoking.mkdir()
    _git(invoking, "init")
    _git(invoking, "config", "user.email", "test@example.com")
    _git(invoking, "config", "user.name", "Test")
    invoking_sha = _commit(invoking, "invoking\n", "invoking")
    hook_worktree = tmp_path / "hook-worktree"
    _git(invoking, "worktree", "add", "-b", "hook-wt", str(hook_worktree))

    poisoned_env = dict(os.environ)
    poisoned_env["GIT_DIR"] = _git(hook_worktree, "rev-parse", "--absolute-git-dir")
    poisoned_env["GIT_WORK_TREE"] = str(hook_worktree)
    target = tmp_path / "dependency"

    def execute_transport_plan(
        repo_url: str,
        target_path: Path,
        *,
        clone_action: Callable[[str, dict[str, str], Path], None],
        **_kwargs: Any,
    ) -> None:
        clone_action(repo_url, poisoned_env, target_path)

    with patch.dict(os.environ, poisoned_env, clear=True):
        repo = clone_with_fallback(execute_transport_plan, str(source), target)
    repo.close()

    assert _git(target, "rev-parse", "HEAD") == dependency_sha
    assert _git(hook_worktree, "symbolic-ref", "--short", "HEAD") == "hook-wt"
    assert _git(hook_worktree, "rev-parse", "HEAD") == invoking_sha


def test_shallow_fetch_failure_reports_captured_git_stderr(tmp_path: Path) -> None:
    """A real failed Git fetch retains its actionable stderr."""
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test")
    _commit(source, "dependency\n", "dependency")
    logs: list[str] = []
    downloader = GitHubPackageDownloader.__new__(GitHubPackageDownloader)

    exists = _path_exists_in_tree_at_ref(
        downloader,
        DependencyReference(repo_url="owner/repo"),
        "skills/missing",
        "missing-ref",
        logs.append,
        AttemptSpec("local fixture", str(source), git_subprocess_env()),
    )

    assert exists is False
    assert any("couldn't find remote ref" in message for message in logs)


def test_shallow_fetch_failure_keeps_cause_and_redacts_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The validation diagnostic keeps stderr without exposing URL credentials."""
    import apm_cli.deps.github_downloader_validation as validation

    real_run = subprocess.run
    token = "secret-sentinel"

    def fail_fetch(args, **kwargs):
        if "fetch" in args:
            raise subprocess.CalledProcessError(
                128,
                args,
                stderr=f"fatal: denied https://{token}@git.example.test/repo".encode(),
            )
        return real_run(args, **kwargs)

    monkeypatch.setattr(validation.subprocess, "run", fail_fetch)
    logs: list[str] = []
    downloader = GitHubPackageDownloader.__new__(GitHubPackageDownloader)

    exists = _path_exists_in_tree_at_ref(
        downloader,
        DependencyReference(repo_url="owner/repo"),
        "skills/missing",
        "missing-ref",
        logs.append,
        AttemptSpec(
            "fixture",
            "https://git.example.test/repo",
            {
                "PATH": os.environ["PATH"],
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
            },
        ),
    )

    output = "\n".join(logs)
    assert exists is False
    assert "fatal: denied" in output
    assert token not in output


def test_shared_bare_clone_rejects_unsafe_effective_rewrite(tmp_path: Path) -> None:
    """The default shared-bare path cannot bypass URL rewrite validation."""
    remote_url = "https://git.example.test/org/repo"
    unsafe_env = {
        "PATH": os.environ["PATH"],
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "url.http://127.0.0.1:9/.insteadOf",
        "GIT_CONFIG_VALUE_0": remote_url,
    }

    def execute_transport_plan(
        repo_url: str,
        target_path: Path,
        *,
        clone_action: Callable[[str, dict[str, str], Path], None],
        **_kwargs: Any,
    ) -> None:
        clone_action(repo_url, unsafe_env, target_path)

    with pytest.raises(GitUrlRewriteError, match="insecure HTTP"):
        bare_clone_with_fallback(
            execute_transport_plan,
            remote_url,
            tmp_path / "bare.git",
            dep_ref=DependencyReference(repo_url="org/repo", host="git.example.test"),
            ref="main",
            is_commit_sha=False,
        )


def test_shared_bare_sha_rejection_leaves_no_tokenized_config(tmp_path: Path) -> None:
    """Rewrite rejection happens before a token-bearing remote is persisted."""
    token = "bare-config-token"
    remote_url = f"https://{token}@git.example.test/org/repo"
    unsafe_env = {
        "PATH": os.environ["PATH"],
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "url.ssh://git@mirror.example/.insteadOf",
        "GIT_CONFIG_VALUE_0": "https://",
    }
    target = tmp_path / "bare.git"

    def execute_transport_plan(
        repo_url: str,
        target_path: Path,
        *,
        clone_action: Callable[[str, dict[str, str], Path], None],
        **_kwargs: Any,
    ) -> None:
        clone_action(repo_url, unsafe_env, target_path)

    with pytest.raises(GitUrlRewriteError):
        bare_clone_with_fallback(
            execute_transport_plan,
            remote_url,
            target,
            dep_ref=DependencyReference(repo_url="org/repo", host="git.example.test"),
            ref="a" * 40,
            is_commit_sha=True,
        )

    assert not target.exists()


def test_git_cache_ls_remote_rejects_unsafe_effective_rewrite(tmp_path: Path) -> None:
    """Persistent-cache ref resolution uses the same rewrite-safety owner."""
    remote_url = "https://git.example.test/org/repo"
    env = {
        "PATH": os.environ["PATH"],
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "url.https://token@mirror.example/.insteadOf",
        "GIT_CONFIG_VALUE_0": remote_url,
    }

    with pytest.raises(GitUrlRewriteError, match="must not contain credentials"):
        GitCache(tmp_path / "cache")._ls_remote_resolve(remote_url, "main", env=env)


def test_git_cache_fetch_rejects_dependency_bare_local_rewrite(tmp_path: Path) -> None:
    """A cached repository's own config cannot activate an unsafe fetch rewrite."""
    remote_url = "https://git.example.test/org/repo"
    bare = tmp_path / "dependency.git"
    subprocess.run(
        [get_git_executable(), "init", "--bare", str(bare)],
        check=True,
        capture_output=True,
        env=git_subprocess_env(),
    )
    subprocess.run(
        [
            get_git_executable(),
            "--git-dir",
            str(bare),
            "config",
            "url.http://127.0.0.1:9/.insteadOf",
            remote_url,
        ],
        check=True,
        capture_output=True,
        env=git_subprocess_env(),
    )

    with pytest.raises(GitUrlRewriteError, match="insecure HTTP"):
        GitCache(tmp_path / "cache")._fetch_into_bare_locked(
            bare,
            remote_url,
            "a" * 40,
            env={"PATH": os.environ["PATH"]},
        )
