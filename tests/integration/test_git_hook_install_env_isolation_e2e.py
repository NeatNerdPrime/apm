"""Real CLI install proof for Git-hook repository environment isolation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.utils.isolated_apm_environment import IsolatedApmEnvironment
from tests.utils.local_git_repository import LocalGitRepositoryFactory
from tests.utils.local_package import LocalPackageFactory

pytestmark = [pytest.mark.e2e, pytest.mark.integration]

_SKILL_PATH = "skills/hook-proof"
_SKILL_BYTES = b"---\nname: hook-proof\ndescription: Git hook isolation proof\n---\n# Hook proof\n"


def _git(cwd: Path, env: dict[str, str], *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


@pytest.mark.parametrize(
    "remote_url",
    (
        "https://github.com/acme/hook-proof",
        "https://git.example.com/acme/hook-proof",
    ),
    ids=("github", "generic-host"),
)
def test_apm_install_from_git_hook_preserves_invoking_worktree(
    tmp_path: Path,
    apm_binary_path: Path,
    remote_url: str,
) -> None:
    """A real install cannot redirect Git into the invoking worktree."""
    isolated = IsolatedApmEnvironment.create(
        tmp_path / "scenario",
        base_env=dict(os.environ),
    )
    environment = isolated.subprocess_env()

    package_factory = LocalPackageFactory(isolated.package_root)
    source = package_factory.create("hook-proof-source")
    package_factory.add_skill(source, "hook-proof", _SKILL_BYTES.decode("ascii"))

    repositories = LocalGitRepositoryFactory(isolated.repository_root, env=environment)
    repository = repositories.create("hook-proof", source_tree=source.root)
    repositories.commit(repository, message="seed hook isolation proof")
    child_env = repositories.url_rewrite_subprocess_env(repository, remote_url)

    consumer = LocalPackageFactory(isolated.work_root).create(
        "consumer",
        dependencies=(
            {
                "git": remote_url,
                "path": _SKILL_PATH,
                "ref": "main",
            },
        ),
        targets=("copilot",),
    )
    _git(consumer.root, environment, "init", "--initial-branch=main")
    _git(consumer.root, environment, "config", "user.email", "test@example.com")
    _git(consumer.root, environment, "config", "user.name", "APM Test")
    _git(consumer.root, environment, "add", "apm.yml")
    _git(consumer.root, environment, "commit", "-m", "seed consumer")
    invoking_sha = _git(consumer.root, environment, "rev-parse", "HEAD")

    hook_worktree = isolated.work_root / "hook-worktree"
    _git(
        consumer.root,
        environment,
        "worktree",
        "add",
        "-b",
        "hook-wt",
        str(hook_worktree),
    )
    child_env["GIT_DIR"] = _git(hook_worktree, environment, "rev-parse", "--absolute-git-dir")
    child_env["GIT_WORK_TREE"] = str(hook_worktree)

    result = subprocess.run(
        (
            str(apm_binary_path),
            "install",
            "--target",
            "copilot",
            "--no-policy",
            "--parallel-downloads",
            "0",
        ),
        cwd=hook_worktree,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert _git(hook_worktree, environment, "symbolic-ref", "--short", "HEAD") == "hook-wt"
    assert _git(hook_worktree, environment, "rev-parse", "HEAD") == invoking_sha
    deployed = hook_worktree / ".agents" / "skills" / "hook-proof" / "SKILL.md"
    assert deployed.read_bytes() == _SKILL_BYTES
