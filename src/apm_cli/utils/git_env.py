"""Cached git binary lookup and subprocess environment sanitization.

Ensures that APM's git subprocess calls use a clean environment free
of ambient git state variables that could bias operations (e.g. when
APM is invoked from within a git repository's hook or worktree).

Preserved variables (user-controlled config for proxy/auth):
- GIT_SSH, GIT_SSH_COMMAND, GIT_ASKPASS, SSH_ASKPASS
- GIT_HTTP_USER_AGENT, GIT_TERMINAL_PROMPT
- GIT_CONFIG_GLOBAL, GIT_CONFIG_SYSTEM

Git state variables stripped after external-process sanitization:
- GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE
- GIT_OBJECT_DIRECTORY, GIT_ALTERNATE_OBJECT_DIRECTORIES
- GIT_COMMON_DIR, GIT_NAMESPACE, GIT_INDEX_VERSION
- GIT_CEILING_DIRECTORIES, GIT_DISCOVERY_ACROSS_FILESYSTEM
- GIT_REPLACE_REF_BASE, GIT_GRAFT_FILE, GIT_SHALLOW_FILE
- GIT_IMPLICIT_WORK_TREE, GIT_NO_REPLACE_OBJECTS, GIT_PREFIX
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from apm_cli.utils.subprocess_env import external_process_env

# Module-level cached git executable path (successful resolutions only).
_git_executable: str | None = None

# Variables that represent ambient git state -- strip these to avoid
# biasing APM's git operations when invoked from within another repo
# or when the calling environment uses git's discovery / replacement
# / grafts overrides.
_STRIP_GIT_VARS: frozenset[str] = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
        "GIT_INDEX_VERSION",
        "GIT_CEILING_DIRECTORIES",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_REPLACE_REF_BASE",
        "GIT_GRAFT_FILE",
        "GIT_SHALLOW_FILE",
        "GIT_IMPLICIT_WORK_TREE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_PREFIX",
    }
)


def get_git_executable() -> str:
    """Return the path to the git executable (cached after a successful lookup).

    Uses ``shutil.which("git")`` to locate git on PATH.
    Failed lookups are not cached because PATH can change within a
    long-lived process.

    Returns:
        Absolute or relative path to the git binary.

    Raises:
        FileNotFoundError: If git is not found on PATH.
    """
    global _git_executable
    if _git_executable is not None:
        return _git_executable

    resolved = shutil.which("git")
    if resolved is None:
        raise FileNotFoundError(
            "git executable not found on PATH. Please install git: https://git-scm.com/downloads"
        )
    _git_executable = resolved
    return _git_executable


def git_subprocess_env(overrides: dict[str, object] | None = None) -> dict[str, str]:
    """Return a sanitized environment dict for git subprocesses.

    Restores PyInstaller-managed dynamic-library variables first, then
    strips ambient git state variables while preserving user-controlled
    configuration (proxy, auth, SSH settings). Optional overrides are
    applied through the same state-variable filter.

    Returns:
        An external-process-safe copy of ``os.environ`` with problematic
        git variables removed.
    """
    base = (
        None
        if overrides is None
        else {key: value for key, value in overrides.items() if isinstance(value, str)}
    )
    return {
        key: value
        for key, value in external_process_env(base).items()
        if key not in _STRIP_GIT_VARS
    }


def git_subprocess_error_text(exc: BaseException) -> str:
    """Return captured Git output when a subprocess exception provides it."""
    if isinstance(exc, subprocess.CalledProcessError):
        for stream in (exc.stderr, exc.stdout):
            if isinstance(stream, bytes):
                stream = stream.decode("utf-8", errors="replace")
            if isinstance(stream, str) and stream.strip():
                return stream.strip()
    return str(exc)


def _append_parent_safe_git_config(env: dict[str, str]) -> None:
    """Retain parent config selection and URL rewrites, never auth channels."""
    for name in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM"):
        if name not in env and name in os.environ:
            env[name] = os.environ[name]
    try:
        parent_count = max(0, int(os.environ.get("GIT_CONFIG_COUNT", "0") or "0"))
        target_count = max(0, int(env.get("GIT_CONFIG_COUNT", "0") or "0"))
    except ValueError:
        return

    existing = {
        (env.get(f"GIT_CONFIG_KEY_{index}", ""), env.get(f"GIT_CONFIG_VALUE_{index}", ""))
        for index in range(target_count)
    }
    for index in range(parent_count):
        key = os.environ.get(f"GIT_CONFIG_KEY_{index}", "")
        value = os.environ.get(f"GIT_CONFIG_VALUE_{index}", "")
        normalized = key.lower()
        if not (
            normalized.startswith("url.")
            and normalized.endswith(".insteadof")
            and value
            and (key, value) not in existing
        ):
            continue
        env[f"GIT_CONFIG_KEY_{target_count}"] = key
        env[f"GIT_CONFIG_VALUE_{target_count}"] = value
        target_count += 1
        existing.add((key, value))
    if target_count:
        env["GIT_CONFIG_COUNT"] = str(target_count)


def clone_git_worktree(
    url: str,
    target: Path,
    *,
    env: dict[str, object] | None = None,
    depth: int | None = None,
    branch: str | None = None,
    no_checkout: bool = False,
    extra_options: Sequence[str] = (),
) -> None:
    """Clone a working tree with a complete sanitized child environment."""
    args = [get_git_executable(), "clone"]
    if depth is not None:
        args.extend(("--depth", str(depth)))
    if branch is not None:
        args.extend(("--branch", branch))
    if no_checkout:
        args.append("--no-checkout")
    args.extend(extra_options)
    args.extend(("--", url, str(target)))
    clone_env = git_subprocess_env(env)
    if env is not None:
        _append_parent_safe_git_config(clone_env)
    subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env=clone_env,
    )


def checkout_git_worktree(
    worktree: Path,
    ref: str,
    *,
    env: dict[str, object] | None = None,
) -> None:
    """Check out a ref in an explicitly located worktree."""
    subprocess.run(
        [
            get_git_executable(),
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(worktree),
            "checkout",
            ref,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=git_subprocess_env(env),
    )


def git_worktree_head(
    worktree: Path,
    *,
    env: dict[str, object] | None = None,
) -> str:
    """Return the HEAD commit for an explicitly located worktree."""
    return git_resolve_commit(worktree, "HEAD", env=env)


def git_resolve_commit(
    worktree: Path,
    ref: str,
    *,
    env: dict[str, object] | None = None,
) -> str:
    """Resolve a ref to a commit in an explicitly located worktree."""
    result = subprocess.run(
        [
            get_git_executable(),
            "-C",
            str(worktree),
            "rev-parse",
            "--verify",
            f"{ref}^{{commit}}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=git_subprocess_env(env),
    )
    return result.stdout.strip()


def git_current_branch(
    worktree: Path,
    *,
    env: dict[str, object] | None = None,
) -> str:
    """Return the current branch name for an explicitly located worktree."""
    result = subprocess.run(
        [get_git_executable(), "-C", str(worktree), "symbolic-ref", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=git_subprocess_env(env),
    )
    return result.stdout.strip()


def reset_git_cache() -> None:
    """Reset the cached git executable (for testing purposes only)."""
    global _git_executable
    _git_executable = None


def git_long_paths_args() -> list[str]:
    """Return ``-c core.longpaths=true`` on Windows, ``[]`` elsewhere.

    Windows enforces a 260-character ``MAX_PATH`` limit by default,
    which the GitCache's deeply-nested ``checkouts_v1/<shard>/<sha>/
    <variant>.incomplete.<pid>.<ns>/`` layout can exceed during
    ``git clone`` -- git fails with ``Filename too long`` while
    creating ``.git/hooks/`` files. Setting ``core.longpaths=true``
    via ``-c`` opts that single subprocess into the long-path API
    without mutating the user's global gitconfig. The flag is a
    no-op on POSIX so callers can prepend it unconditionally.
    """
    if os.name == "nt":
        return ["-c", "core.longpaths=true"]
    return []
