"""Create a deterministic, sibling ZIP archive for a verified release directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .config import RELEASE_NAME, RELEASE_TAG
from .inventory import sha256


ARCHIVE_BASENAME = f"{RELEASE_NAME}-{RELEASE_TAG}"
FIXED_ZIP_TIMESTAMP = (2026, 7, 31, 0, 0, 0)


@dataclass(frozen=True)
class ArchiveArtifacts:
    """Paths to the external upload archive and its integrity sidecar."""

    archive_path: Path
    checksum_path: Path
    sha256: str


def _target_paths(output_root: Path) -> tuple[Path, Path]:
    archive_path = output_root / f"{ARCHIVE_BASENAME}.zip"
    checksum_path = output_root / f"{ARCHIVE_BASENAME}.zip.sha256"
    return archive_path, checksum_path


def _zip_info(relative_path: Path, package_name: str) -> ZipInfo:
    info = ZipInfo(f"{package_name}/{relative_path.as_posix()}", FIXED_ZIP_TIMESTAMP)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def create_archive(
    package_root: Path, output_root: Path, *, replace: bool = False
) -> ArchiveArtifacts:
    """Archive only a verified package, without placing the ZIP inside it.

    Existing archives are protected by default.  Replacement is explicit and
    confined to the two predictable sibling files named by this release.
    """
    package_root = package_root.resolve()
    output_root = output_root.resolve()
    if not package_root.is_dir() or package_root.name != RELEASE_NAME:
        raise ValueError(f"Expected the named release directory, found: {package_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    archive_path, checksum_path = _target_paths(output_root)
    existing = [path for path in (archive_path, checksum_path) if path.exists()]
    if existing and not replace:
        raise FileExistsError(
            "Archive output already exists: " + ", ".join(str(path) for path in existing)
        )
    for path in existing:
        path.unlink()

    files = sorted(path for path in package_root.rglob("*") if path.is_file())
    with ZipFile(
        archive_path,
        mode="w",
        compression=ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for source in files:
            archive.writestr(
                _zip_info(source.relative_to(package_root), package_root.name),
                source.read_bytes(),
            )

    digest = sha256(archive_path)
    checksum_path.write_text(f"{digest}  {archive_path.name}\n", encoding="ascii")
    return ArchiveArtifacts(archive_path=archive_path, checksum_path=checksum_path, sha256=digest)
