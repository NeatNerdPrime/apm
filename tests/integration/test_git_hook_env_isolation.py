"""Regression tests for Git repository state inherited from hooks."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from apm_cli.cache.git_cache import GitCache
from apm_cli.cache.url_normalize import cache_shard_key
from apm_cli.utils.git_env import (
    checkout_git_worktree,
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
