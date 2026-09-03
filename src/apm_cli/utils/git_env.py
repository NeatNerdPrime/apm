"""Cached git binary lookup and subprocess environment sanitization.

Ensures that APM's git subprocess calls use a clean environment free
of ambient git state variables that could bias operations (e.g. when
APM is invoked from within a git repository's hook or worktree).

Preserved variables (user-controlled config for proxy/auth):
- GIT_SSH, GIT_SSH_COMMAND, GIT_ASKPASS, SSH_ASKPASS
- GIT_HTTP_USER_AGENT, GIT_TERMINAL_PROMPT
- GIT_CONFIG_GLOBAL, GIT_CONFIG_SYSTEM

Git state variables stripped after external-process sanitization:
- GIT_DIR, GIT_CONFIG, GIT_WORK_TREE, GIT_INDEX_FILE
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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol
from urllib.parse import SplitResult, urlsplit

from apm_cli.utils.subprocess_env import external_process_env

# Module-level cached git executable path (successful resolutions only).
_git_executable: str | None = None
# Keep config inspection independent from tests that replace network subprocess
# execution. Targeted tests can patch this callable directly when exercising
# malformed config output.
_git_config_run = subprocess.run
_git_init_run = subprocess.run

# Variables that represent ambient git state -- strip these to avoid
# biasing APM's git operations when invoked from within another repo
# or when the calling environment uses git's discovery / replacement
# / grafts overrides.
_STRIP_GIT_VARS: frozenset[str] = frozenset(
    {
        "GIT_DIR",
        "GIT_CONFIG",
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

_URL_REWRITE_RECOVERY = (
    "inspect matching rules with "
    "'git config --show-origin --get-regexp ^url\\..*\\.insteadOf$' "
    "and remove the unsafe rule"
)


class GitUrlRewriteError(ValueError):
    """A stable, actionable rejection of an unsafe Git URL rewrite."""

    def __init__(self, reason: str, message: str) -> None:
        """Initialize one rejection with a machine-readable reason."""
        self.reason = reason
        super().__init__(f"{message}; {_URL_REWRITE_RECOVERY}")


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


def git_no_hooks_args() -> tuple[str, str]:
    """Return the canonical per-command fence against repository Git hooks."""
    return "-c", "core.hooksPath=/dev/null"


class _GitProgress(Protocol):
    """Git progress callback consumed by the clone subprocess adapter."""

    def new_message_handler(self) -> Callable[[str], None]:
        """Return a callback that parses one Git progress line."""


def _read_effective_git_url_rewrites(
    env: dict[str, str],
    *,
    git_dir: Path | None = None,
    worktree: Path | None = None,
) -> tuple[tuple[tuple[str, str], ...], bool]:
    """Return visible URL rewrites and whether config injects authorization."""
    if git_dir is not None and worktree is not None:
        raise ValueError("git_dir and worktree are mutually exclusive")
    command = [get_git_executable()]
    probe_env = dict(env)
    probe_cwd: str | None = None
    if git_dir is not None:
        command.extend(("--git-dir", str(git_dir)))
    elif worktree is not None:
        command.extend(("-C", str(worktree)))
    else:
        # `git clone <url>` does not consume the invoking repository's local
        # config. Run from the Git executable's directory so a hook's cwd cannot
        # create false policy failures while system, global, and process-scoped
        # config remain visible.
        probe_cwd = str(Path(get_git_executable()).resolve().parent)
    command.extend(
        (
            "config",
            "--null",
            "--get-regexp",
            r"^(url\..*\.insteadof|http(\..*)?\.extraheader)$",
        )
    )
    try:
        result = _git_config_run(
            command,
            capture_output=True,
            check=False,
            cwd=probe_cwd,
            env=probe_env,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Unable to verify Git URL rewrite safety") from exc
    if result.returncode == 1:
        return (), False
    if result.returncode != 0 or not isinstance(result.stdout, bytes):
        raise ValueError("Unable to verify Git URL rewrite safety")

    rewrites: list[tuple[str, str]] = []
    has_authorization = False
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        if b"\n" not in entry:
            raise ValueError("Unable to verify Git URL rewrite safety")
        key, prefix = entry.split(b"\n", 1)
        try:
            key_text = key.decode("utf-8")
            prefix_text = prefix.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Unable to verify Git URL rewrite safety") from exc
        normalized = key_text.lower()
        if normalized.startswith("http.") or normalized == "http.extraheader":
            if normalized.endswith(".extraheader") and prefix_text.strip():
                has_authorization = True
            continue
        if not normalized.startswith("url.") or not normalized.endswith(".insteadof"):
            raise ValueError("Unable to verify Git URL rewrite safety")
        replacement = key_text[4 : -len(".insteadOf")]
        if not replacement or not prefix_text:
            raise ValueError("Unable to verify Git URL rewrite safety")
        rewrites.append((replacement, prefix_text))
    return tuple(rewrites), has_authorization


def _has_forced_http_authorization(env: dict[str, str]) -> bool:
    """Return whether *env* injects an HTTP authorization header."""
    normalized_env = {key.upper(): value for key, value in env.items()}
    if normalized_env.get("GIT_HTTP_EXTRAHEADER", "").strip():
        return True
    parameters = normalized_env.get("GIT_CONFIG_PARAMETERS", "").lower()
    if "extraheader" in parameters:
        return True
    try:
        count = max(0, int(normalized_env.get("GIT_CONFIG_COUNT", "0") or "0"))
    except ValueError:
        return True
    for index in range(count):
        key = normalized_env.get(f"GIT_CONFIG_KEY_{index}", "").lower()
        value = normalized_env.get(f"GIT_CONFIG_VALUE_{index}", "")
        if key.endswith("extraheader") and value.strip():
            return True
    return False


def _url_origin(url: str) -> tuple[str, str, int | None]:
    """Return a normalized HTTP(S) origin tuple."""
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    port = parsed.port
    if port is None:
        port = {"http": 80, "https": 443}.get(scheme)
    return scheme, (parsed.hostname or "").lower(), port


def _url_contains_credentials(parsed: SplitResult) -> bool:
    """Return whether URL userinfo can carry a credential."""
    return parsed.password is not None or (
        parsed.scheme.lower() in {"http", "https", "file"} and parsed.username is not None
    )


def resolve_git_url_rewrite(
    remote_url: str,
    rewrites: Sequence[tuple[str, str]],
) -> str | None:
    """Apply Git's longest-prefix ``insteadOf`` rule to *remote_url*."""
    matches = tuple(
        (replacement, prefix) for replacement, prefix in rewrites if remote_url.startswith(prefix)
    )
    if not matches:
        return None
    replacement, prefix = max(matches, key=lambda item: len(item[1]))
    return f"{replacement}{remote_url[len(prefix) :]}"


def configured_git_url_policy(
    env: dict[str, object] | None = None,
) -> tuple[tuple[tuple[str, str], ...], bool]:
    """Return repository-neutral rewrites and config authorization state."""
    return _read_effective_git_url_rewrites(git_subprocess_env(env))


def validate_resolved_git_url_rewrite(
    remote_url: str,
    effective_url: str,
    *,
    has_authorization: bool,
) -> None:
    """Validate one effective URL selected by Git's rewrite rules."""
    try:
        source = urlsplit(remote_url)
        target = urlsplit(effective_url)
        source_origin = _url_origin(remote_url)
        target_origin = _url_origin(effective_url)
    except ValueError as exc:
        raise ValueError("Unable to verify Git URL rewrite safety") from exc
    if _url_contains_credentials(target):
        raise GitUrlRewriteError(
            "credentials",
            "Git URL rewrite replacement must not contain credentials",
        )
    source_scheme = source.scheme.lower()
    target_scheme = target.scheme.lower()
    if target_scheme == "http" and source_scheme != "http":
        raise GitUrlRewriteError(
            "https-downgrade" if source_scheme == "https" else "insecure-transport",
            (
                "HTTPS Git remote must not rewrite to insecure HTTP"
                if source_scheme == "https"
                else "Git remote must not rewrite to insecure HTTP"
            ),
        )
    if target_scheme not in {
        "",
        "file",
        "https",
        "ssh",
    }:
        raise GitUrlRewriteError(
            "insecure-transport",
            "HTTPS Git remote must not rewrite to an insecure transport",
        )
    if _url_contains_credentials(source):
        raise GitUrlRewriteError(
            "credential-origin",
            "Credential-bearing Git remote must not be rewritten",
        )
    if target_scheme in {"http", "https"} and has_authorization and source_origin != target_origin:
        raise GitUrlRewriteError(
            "credential-origin",
            f"Authenticated Git remote must not rewrite to a different "
            f"{target_scheme.upper()} origin",
        )


def validate_git_url_rewrite_safety(
    remote_url: str,
    env: dict[str, str],
    *,
    git_dir: Path | None = None,
    worktree: Path | None = None,
) -> str | None:
    """Reject credential-bearing or HTTPS-downgrading effective URL rewrites."""
    try:
        rewrites, config_has_authorization = _read_effective_git_url_rewrites(
            env,
            git_dir=git_dir,
            worktree=worktree,
        )
    except ValueError as exc:
        if str(exc) == "Unable to verify Git URL rewrite safety":
            raise
        raise ValueError("Unable to verify Git URL rewrite safety") from exc

    effective_url = resolve_git_url_rewrite(remote_url, rewrites)
    if effective_url is None:
        return None
    validate_resolved_git_url_rewrite(
        remote_url,
        effective_url,
        has_authorization=config_has_authorization or _has_forced_http_authorization(env),
    )
    return effective_url


def _append_git_url_rewrites(
    env: dict[str, str],
    rewrites: Sequence[tuple[str, str]],
) -> None:
    """Materialize URL rewrites in indexed process configuration."""
    try:
        target_count = max(0, int(env.get("GIT_CONFIG_COUNT", "0") or "0"))
    except ValueError:
        raise ValueError("Unable to verify Git URL rewrite safety") from None

    existing = {
        (env.get(f"GIT_CONFIG_KEY_{index}", ""), env.get(f"GIT_CONFIG_VALUE_{index}", ""))
        for index in range(target_count)
    }
    for replacement, value in rewrites:
        key = f"url.{replacement}.insteadOf"
        if (key, value) in existing:
            continue
        env[f"GIT_CONFIG_KEY_{target_count}"] = key
        env[f"GIT_CONFIG_VALUE_{target_count}"] = value
        target_count += 1
        existing.add((key, value))
    if target_count:
        env["GIT_CONFIG_COUNT"] = str(target_count)


def _append_parent_git_config(env: dict[str, str]) -> None:
    """Retain parent URL rewrites without restoring config auth channels."""
    try:
        parent_rewrites, _ = _read_effective_git_url_rewrites(git_subprocess_env())
    except ValueError:
        raise ValueError("Unable to verify Git URL rewrite safety") from None
    _append_git_url_rewrites(env, parent_rewrites)


def git_network_env(
    remote_url: str,
    overrides: dict[str, object] | None = None,
    *,
    git_dir: Path | None = None,
    worktree: Path | None = None,
) -> dict[str, str]:
    """Return the canonical validated environment for one network Git URL."""
    env = git_subprocess_env(overrides)
    if overrides is not None:
        _append_parent_git_config(env)
    effective_url = validate_git_url_rewrite_safety(
        remote_url,
        env,
        git_dir=git_dir,
        worktree=worktree,
    )
    transport_url = effective_url or remote_url
    if urlsplit(transport_url).scheme.lower() not in {"http", "https"}:
        from apm_cli.core.auth import AuthResolver

        rewrites, _ = _read_effective_git_url_rewrites(
            env,
            git_dir=git_dir,
            worktree=worktree,
        )
        AuthResolver._clear_git_auth_env(env)
        AuthResolver._clear_platform_token_env(env, remove=True)
        _append_git_url_rewrites(env, rewrites)
    return env


def git_clone_env(
    remote_url: str,
    overrides: dict[str, object] | None,
    target: Path,
    *,
    bare: bool = False,
) -> dict[str, str]:
    """Validate clone config in the Git directory the target will activate."""
    target_existed = target.exists()
    git_dir = target if bare else target / ".git"
    probe_created = not (git_dir / "config").exists()
    target_mode = target.stat().st_mode if target_existed else None
    probe_materialized = False
    try:
        if probe_created:
            if target_existed and any(target.iterdir()):
                raise ValueError(f"Git clone target is not empty: {target}")
            probe_materialized = True
            result = _git_init_run(
                [
                    get_git_executable(),
                    "init",
                    *(("--bare",) if bare else ()),
                    "--template=",
                    "--quiet",
                    str(target),
                ],
                capture_output=True,
                check=False,
                env=git_subprocess_env(overrides),
                timeout=30,
            )
            if result.returncode != 0:
                raise ValueError("Unable to prepare Git clone configuration probe")
        return git_network_env(remote_url, overrides, git_dir=git_dir)
    finally:
        if probe_materialized:
            if bare:
                if target.exists():
                    shutil.rmtree(target)
                if target_existed:
                    target.mkdir()
                    if target_mode is not None:
                        os.chmod(target, target_mode)
            else:
                if git_dir.exists():
                    shutil.rmtree(git_dir)
                if not target_existed and target.exists():
                    target.rmdir()


def git_remote_refs(
    remote_url: str,
    *patterns: str,
    env: dict[str, object] | None = None,
    timeout: int = 30,
    check: bool = False,
    options: Sequence[str] = (),
    git_args: Sequence[str] = (),
) -> subprocess.CompletedProcess[str]:
    """Run ``git ls-remote`` with the canonical validated child environment."""
    child_env = git_network_env(remote_url, env)
    # auth-delegated: callers resolve credential and bearer policy before this executor.
    git_executable = get_git_executable()
    command = [git_executable, *git_args, "ls-remote", *options, remote_url, *patterns]
    try:
        return subprocess.run(
            command,
            capture_output=True,
            cwd=str(Path(git_executable).resolve().parent),
            text=True,
            timeout=timeout,
            env=child_env,
            stdin=subprocess.DEVNULL,
            check=check,
        )
    except subprocess.TimeoutExpired:
        # auth-delegated: callers resolve credential and bearer policy before this executor.
        raise subprocess.TimeoutExpired([git_executable, "ls-remote"], timeout) from None


def init_git_remote_worktree(
    worktree: Path,
    remote_url: str,
    env: dict[str, object],
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, str]:
    """Initialize a worktree and add one validated network remote."""
    git_executable = get_git_executable()
    child_env = git_subprocess_env(env)
    result = run(
        [git_executable, "init"],
        cwd=str(worktree),
        env=child_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=True,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    child_env = git_network_env(remote_url, child_env, worktree=worktree)
    result = run(
        [git_executable, "remote", "add", "origin", remote_url],
        cwd=str(worktree),
        env=child_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=True,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    return child_env


def clone_git_worktree(
    url: str,
    target: Path,
    *,
    env: dict[str, object] | None = None,
    depth: int | None = None,
    branch: str | None = None,
    no_checkout: bool = False,
    extra_options: Sequence[str] = (),
    progress: _GitProgress | None = None,
) -> None:
    """Clone a working tree with a complete sanitized child environment."""
    args = [get_git_executable(), *git_no_hooks_args(), "clone"]
    if progress is not None:
        args.append("--progress")
    if depth is not None:
        args.extend(("--depth", str(depth)))
    if branch is not None:
        args.extend(("--branch", branch))
    if no_checkout:
        args.append("--no-checkout")
    args.extend(extra_options)
    args.extend(("--", url, str(target)))
    clone_env = git_clone_env(url, env, target)
    if progress is None:
        try:
            subprocess.run(
                args,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
                env=clone_env,
            )
        except subprocess.TimeoutExpired:
            raise subprocess.TimeoutExpired([get_git_executable(), "clone"], 300) from None
        return

    from git.cmd import handle_process_output

    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=clone_env,
    )
    stdout: list[str] = []
    stderr: list[str] = []
    progress_handler = progress.new_message_handler()

    def capture_stderr(line: str) -> None:
        stderr.append(line)
        progress_handler(line)

    try:
        handle_process_output(
            process,
            stdout.append,
            capture_stderr,
            decode_streams=False,
            kill_after_timeout=300,
        )
        process.wait(timeout=30)
    except (RuntimeError, subprocess.TimeoutExpired):
        process.kill()
        process.wait()
        raise subprocess.TimeoutExpired([get_git_executable(), "clone"], 300) from None
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            args,
            output="".join(stdout),
            stderr="".join(stderr),
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
            *git_no_hooks_args(),
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
