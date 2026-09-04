"""Security boundaries for the cached legacy-plugin compatibility bridge."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apm_cli.deps.lockfile import LockedDependency
from apm_cli.install.errors import DirectDependencyError
from apm_cli.install.legacy_plugin_compat import upgrade_cached_legacy_plugin
from apm_cli.models.apm_package import APMPackage


def _legacy_cache(root: Path) -> Path:
    package = root / "cached-plugin"
    (package / ".apm" / "skills" / "demo").mkdir(parents=True)
    (package / "apm.yml").write_text(
        "name: cached-plugin\nversion: 0.1.0\n",
        encoding="ascii",
    )
    (package / "plugin.json").write_text(
        '{"name": "cached-plugin", "skills": ["./skills/"]}',
        encoding="ascii",
    )
    (package / ".apm" / "skills" / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo\n---\n",
        encoding="ascii",
    )
    return package


def _locked_plugin() -> LockedDependency:
    return LockedDependency(
        repo_url="owner/cached-plugin",
        package_type="marketplace_plugin",
        content_hash="sha256:verified",
    )


def _upgrade(
    package: Path,
    *,
    dependency: LockedDependency | None = None,
    apm_version: str = "0.28.0",
    verified: bool = True,
    fetched_this_run: bool = False,
) -> APMPackage | None:
    return upgrade_cached_legacy_plugin(
        package,
        "owner/cached-plugin",
        locked_dependency=dependency,
        lockfile_apm_version=apm_version,
        content_hash_verified=verified,
        fetched_this_run=fetched_this_run,
    )


@pytest.mark.parametrize(
    ("apm_version", "package_type", "verified", "fetched_this_run"),
    [
        ("0.29.0", "marketplace_plugin", True, False),
        ("0.28.0", "skill_bundle", True, False),
        ("0.28.0", "apm_package", True, False),
        ("0.28.0", "marketplace_plugin", False, False),
        ("0.28.0", "marketplace_plugin", True, True),
    ],
)
def test_upgrade_requires_verified_legacy_lock_provenance(
    tmp_path: Path,
    apm_version: str,
    package_type: str,
    verified: bool,
    fetched_this_run: bool,
) -> None:
    package = _legacy_cache(tmp_path)
    dependency = _locked_plugin()
    dependency.package_type = package_type

    with patch("apm_cli.install.legacy_plugin_compat.gather_detection_evidence") as detect:
        result = _upgrade(
            package,
            dependency=dependency,
            apm_version=apm_version,
            verified=verified,
            fetched_this_run=fetched_this_run,
        )

    assert result is None
    detect.assert_not_called()


def test_upgrade_requires_dependency_to_exist_in_legacy_lock(tmp_path: Path) -> None:
    package = _legacy_cache(tmp_path)

    with patch("apm_cli.install.legacy_plugin_compat.gather_detection_evidence") as detect:
        result = _upgrade(package)

    assert result is None
    detect.assert_not_called()


def test_upgrade_requires_locked_content_hash(tmp_path: Path) -> None:
    package = _legacy_cache(tmp_path)
    dependency = _locked_plugin()
    dependency.content_hash = None

    with patch("apm_cli.install.legacy_plugin_compat.gather_detection_evidence") as detect:
        result = _upgrade(package, dependency=dependency)

    assert result is None
    detect.assert_not_called()


def test_upgrade_rejects_missing_locked_plugin_manifest(tmp_path: Path) -> None:
    package = _legacy_cache(tmp_path)
    (package / "plugin.json").unlink()

    with pytest.raises(DirectDependencyError, match="manifest is missing or unreadable"):
        _upgrade(package, dependency=_locked_plugin())


def test_upgrade_rejects_invalid_existing_apm_yml(tmp_path: Path) -> None:
    package = _legacy_cache(tmp_path)
    (package / "apm.yml").write_text("name: [\n", encoding="ascii")

    with pytest.raises(DirectDependencyError, match=r"existing apm.yml is invalid"):
        _upgrade(package, dependency=_locked_plugin())


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

    with pytest.raises(DirectDependencyError, match="cache metadata contains a symlink"):
        _upgrade(package, dependency=_locked_plugin())

    after = target.read_bytes() if target.is_file() else tuple(target.iterdir())
    assert after == before


def test_verified_legacy_cache_routes_through_canonical_validator(tmp_path: Path) -> None:
    package = _legacy_cache(tmp_path)
    normalized = MagicMock(spec=APMPackage)
    validation = MagicMock(is_valid=True, package=normalized)

    with patch(
        "apm_cli.install.legacy_plugin_compat.validate_legacy_marketplace_plugin",
        return_value=validation,
    ) as validate:
        result = _upgrade(package, dependency=_locked_plugin())

    assert result is normalized
    validate.assert_called_once()
