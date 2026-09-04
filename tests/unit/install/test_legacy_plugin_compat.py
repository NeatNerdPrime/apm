"""Security boundaries for the cached legacy-plugin compatibility bridge."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apm_cli.deps.lockfile import LockedDependency, LockFile
from apm_cli.install.errors import DirectDependencyError
from apm_cli.install.legacy_plugin_compat import upgrade_cached_legacy_plugin
from apm_cli.models.apm_package import APMPackage
from apm_cli.utils.content_hash import compute_package_hash


def _legacy_cache(root: Path) -> Path:
    package = root / "cached-plugin"
    (package / ".apm" / "skills" / "demo").mkdir(parents=True)
    (package / "apm.yml").write_text("name: cached-plugin\n", encoding="ascii")
    (package / "plugin.json").write_text(
        '{"name": "cached-plugin", "skills": ["./skills/"]}',
        encoding="ascii",
    )
    (package / ".apm" / "skills" / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo\n---\n",
        encoding="ascii",
    )
    return package


def _legacy_lock(package: Path) -> tuple[LockFile, LockedDependency]:
    dependency = LockedDependency(
        repo_url="owner/cached-plugin",
        package_type="marketplace_plugin",
        content_hash=compute_package_hash(package),
    )
    lockfile = LockFile(apm_version="0.28.0")
    lockfile.add_dependency(dependency)
    return lockfile, dependency


@pytest.mark.parametrize(
    ("apm_version", "package_type", "fetched_this_run"),
    [
        ("0.29.0", "marketplace_plugin", False),
        ("0.28.0", "skill_bundle", False),
        pytest.param(
            "0.28.0",
            "apm_package",
            False,
            id="canonical-apm-yml-precedence",
        ),
        ("0.28.0", "marketplace_plugin", True),
    ],
)
def test_upgrade_requires_legacy_lock_provenance(
    tmp_path: Path,
    apm_version: str,
    package_type: str,
    fetched_this_run: bool,
) -> None:
    package = _legacy_cache(tmp_path)
    lockfile, dependency = _legacy_lock(package)
    lockfile.apm_version = apm_version
    dependency.package_type = package_type

    with patch("apm_cli.install.legacy_plugin_compat.gather_detection_evidence") as detect:
        result = upgrade_cached_legacy_plugin(
            package,
            "owner/cached-plugin",
            lockfile=lockfile,
            fetched_this_run=fetched_this_run,
        )

    assert result is None
    detect.assert_not_called()


def test_upgrade_requires_dependency_to_exist_in_legacy_lock(tmp_path: Path) -> None:
    package = _legacy_cache(tmp_path)
    lockfile = LockFile(apm_version="0.28.0")

    with patch("apm_cli.install.legacy_plugin_compat.gather_detection_evidence") as detect:
        result = upgrade_cached_legacy_plugin(
            package,
            "owner/cached-plugin",
            lockfile=lockfile,
            fetched_this_run=False,
        )

    assert result is None
    detect.assert_not_called()


def test_upgrade_rejects_hash_mismatch_before_normalization(tmp_path: Path) -> None:
    package = _legacy_cache(tmp_path)
    lockfile, dependency = _legacy_lock(package)
    dependency.content_hash = "sha256:" + ("0" * 64)

    with (
        patch(
            "apm_cli.install.legacy_plugin_compat.validate_legacy_marketplace_plugin"
        ) as validate,
        pytest.raises(DirectDependencyError, match="content hash mismatch"),
    ):
        upgrade_cached_legacy_plugin(
            package,
            "owner/cached-plugin",
            lockfile=lockfile,
            fetched_this_run=False,
        )

    validate.assert_not_called()


def test_upgrade_rejects_missing_locked_plugin_manifest(tmp_path: Path) -> None:
    package = _legacy_cache(tmp_path)
    (package / "plugin.json").unlink()
    lockfile, dependency = _legacy_lock(package)
    dependency.content_hash = compute_package_hash(package)

    with pytest.raises(DirectDependencyError, match="manifest is missing or unreadable"):
        upgrade_cached_legacy_plugin(
            package,
            "owner/cached-plugin",
            lockfile=lockfile,
            fetched_this_run=False,
        )


def test_upgrade_rejects_missing_hash_before_normalization(tmp_path: Path) -> None:
    package = _legacy_cache(tmp_path)
    lockfile, dependency = _legacy_lock(package)
    dependency.content_hash = None

    with (
        patch(
            "apm_cli.install.legacy_plugin_compat.validate_legacy_marketplace_plugin"
        ) as validate,
        pytest.raises(DirectDependencyError, match="legacy lock entry has no content hash"),
    ):
        upgrade_cached_legacy_plugin(
            package,
            "owner/cached-plugin",
            lockfile=lockfile,
            fetched_this_run=False,
        )

    validate.assert_not_called()


@pytest.mark.parametrize("unsafe_path", ["apm.yml", ".apm"])
def test_upgrade_rejects_symlinked_metadata_without_external_write(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    package = _legacy_cache(tmp_path)
    target = tmp_path / f"external-{unsafe_path.replace('.', 'dot')}"
    source = package / unsafe_path
    if source.is_dir():
        source.rename(target)
        target_is_directory = True
    else:
        target.write_text("external sentinel\n", encoding="ascii")
        source.unlink()
        target_is_directory = False
    try:
        source.symlink_to(target, target_is_directory=target_is_directory)
    except OSError:
        pytest.skip("Symlinks not supported on this platform")
    before = target.read_bytes() if target.is_file() else tuple(target.iterdir())
    lockfile, _dependency = _legacy_lock(package)

    with pytest.raises(DirectDependencyError, match="cache metadata contains a symlink"):
        upgrade_cached_legacy_plugin(
            package,
            "owner/cached-plugin",
            lockfile=lockfile,
            fetched_this_run=False,
        )

    after = target.read_bytes() if target.is_file() else tuple(target.iterdir())
    assert after == before


def test_verified_legacy_cache_routes_through_canonical_validator(tmp_path: Path) -> None:
    package = _legacy_cache(tmp_path)
    lockfile, _dependency = _legacy_lock(package)
    normalized = MagicMock(spec=APMPackage)
    validation = MagicMock(is_valid=True, package=normalized)

    with patch(
        "apm_cli.install.legacy_plugin_compat.validate_legacy_marketplace_plugin",
        return_value=validation,
    ) as validate:
        result = upgrade_cached_legacy_plugin(
            package,
            "owner/cached-plugin",
            lockfile=lockfile,
            fetched_this_run=False,
        )

    assert result is normalized
    validate.assert_called_once()
