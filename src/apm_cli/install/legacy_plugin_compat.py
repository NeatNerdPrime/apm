"""Compatibility upgrade for cached legacy marketplace plugins."""

from __future__ import annotations

from pathlib import Path

from apm_cli.constants import APM_YML_FILENAME
from apm_cli.deps.lockfile import LockedDependency
from apm_cli.deps.plugin_parser import has_normalized_plugin_skill_sources_receipt
from apm_cli.install.errors import DirectDependencyError
from apm_cli.models.apm_package import APMPackage, PackageType
from apm_cli.models.validation import (
    gather_detection_evidence,
    validate_legacy_marketplace_plugin,
)

_RECEIPTLESS_PLUGIN_LOCK_VERSIONS = frozenset({"0.28.0"})


def upgrade_cached_legacy_plugin(
    package_path: Path,
    dep_key: str,
    *,
    locked_dependency: LockedDependency | None,
    lockfile_apm_version: str | None,
    content_hash_verified: bool,
    fetched_this_run: bool,
) -> APMPackage | None:
    """Repair receipt-less 0.28 plugin metadata before cached integration."""
    if (
        fetched_this_run
        or not content_hash_verified
        or locked_dependency is None
        or not locked_dependency.content_hash
        or locked_dependency.package_type != PackageType.MARKETPLACE_PLUGIN.value
        or lockfile_apm_version not in _RECEIPTLESS_PLUGIN_LOCK_VERSIONS
    ):
        return None

    evidence = gather_detection_evidence(package_path)
    if (
        not evidence.has_plugin_manifest
        or not (package_path / APM_YML_FILENAME).exists()
        or has_normalized_plugin_skill_sources_receipt(package_path)
    ):
        return None

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
