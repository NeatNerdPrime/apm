"""Compatibility upgrade for cached legacy marketplace plugins."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from apm_cli.constants import APM_DIR, APM_YML_FILENAME
from apm_cli.deps.plugin_parser import has_normalized_plugin_skill_sources_receipt
from apm_cli.install.errors import DirectDependencyError
from apm_cli.models.apm_package import APMPackage, PackageType
from apm_cli.models.validation import (
    gather_detection_evidence,
    validate_legacy_marketplace_plugin,
)
from apm_cli.utils.content_hash import compute_package_hash
from apm_cli.utils.path_security import (
    PathTraversalError,
    ensure_path_within,
    has_symlink_component,
)

if TYPE_CHECKING:
    from apm_cli.deps.lockfile import LockFile

_LEGACY_PLUGIN_APM_VERSION = "0.28.0"


def upgrade_cached_legacy_plugin(
    package_path: Path,
    dep_key: str,
    *,
    lockfile: LockFile | None,
    fetched_this_run: bool,
) -> APMPackage | None:
    """Repair receipt-less 0.28 plugin metadata before cached integration."""
    locked_dependency = lockfile.get_dependency(dep_key) if lockfile is not None else None
    if (
        fetched_this_run
        or lockfile is None
        or lockfile.apm_version != _LEGACY_PLUGIN_APM_VERSION
        or locked_dependency is None
        or locked_dependency.package_type != PackageType.MARKETPLACE_PLUGIN.value
    ):
        return None

    apm_yml_path = package_path / APM_YML_FILENAME
    apm_dir = package_path / APM_DIR
    if (
        not apm_yml_path.exists()
        or not apm_dir.is_dir()
        or has_normalized_plugin_skill_sources_receipt(package_path)
    ):
        return None

    evidence = gather_detection_evidence(package_path)
    if not evidence.has_plugin_manifest or evidence.plugin_json_path is None:
        raise _unsafe_upgrade_error(
            dep_key,
            package_path,
            "the locked marketplace plugin manifest is missing or unreadable",
        )

    _reject_unsafe_cache_paths(
        package_path,
        dep_key,
        (apm_yml_path, apm_dir, evidence.plugin_json_path),
    )
    expected_hash = locked_dependency.content_hash
    if not expected_hash:
        raise _unsafe_upgrade_error(
            dep_key, package_path, "the legacy lock entry has no content hash"
        )
    actual_hash = compute_package_hash(package_path)
    if actual_hash != expected_hash:
        raise _unsafe_upgrade_error(
            dep_key,
            package_path,
            f"content hash mismatch (expected {expected_hash}, got {actual_hash})",
        )

    result = validate_legacy_marketplace_plugin(
        package_path,
        evidence.plugin_json_path,
        source_path=package_path,
    )
    if not result.is_valid or result.package is None:
        details = "; ".join(result.errors) or "validator returned no package"
        raise DirectDependencyError(
            f"Cached Claude Plugin '{dep_key}' at '{package_path}' is invalid: {details}. "
            "Remove the cached directory or run 'apm deps clean --yes', then retry."
        )
    return result.package


def _reject_unsafe_cache_paths(
    package_path: Path,
    dep_key: str,
    paths: tuple[Path, ...],
) -> None:
    """Reject metadata paths that could redirect compatibility writes."""
    try:
        unsafe = package_path.is_symlink()
        for path in paths:
            if has_symlink_component(package_path, path):
                unsafe = True
                continue
            ensure_path_within(path, package_path)
    except (OSError, PathTraversalError) as exc:
        raise _unsafe_upgrade_error(
            dep_key, package_path, f"cache metadata path validation failed: {exc}"
        ) from exc
    if unsafe:
        raise _unsafe_upgrade_error(dep_key, package_path, "cache metadata contains a symlink")


def _unsafe_upgrade_error(
    dep_key: str,
    package_path: Path,
    reason: str,
) -> DirectDependencyError:
    return DirectDependencyError(
        f"Cached Claude Plugin '{dep_key}' at '{package_path}' cannot be upgraded safely: "
        f"{reason}. Remove the cached directory or run 'apm deps clean --yes', then retry."
    )
