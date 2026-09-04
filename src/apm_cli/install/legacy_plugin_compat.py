"""Compatibility upgrade for cached legacy marketplace plugins."""

from __future__ import annotations

from pathlib import Path

from apm_cli.constants import APM_YML_FILENAME
from apm_cli.deps.plugin_parser import has_normalized_plugin_skill_sources_receipt
from apm_cli.install.errors import DirectDependencyError
from apm_cli.models.apm_package import APMPackage
from apm_cli.models.validation import (
    gather_detection_evidence,
    validate_legacy_marketplace_plugin,
)


def upgrade_cached_legacy_plugin(package_path: Path, dep_key: str) -> APMPackage | None:
    """Repair receipt-less 0.28 plugin metadata before cached integration."""
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
