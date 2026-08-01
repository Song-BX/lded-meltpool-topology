"""Create the local source-only GitHub upload directory from its allow-list."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .config import RELEASE_NAME
from .documents import write_release_documents
from .inventory import ReleaseFile, build_source_manifest, select_release_files, sha256


def _copy_files(files: list[ReleaseFile], package_root: Path) -> None:
    for item in files:
        destination = package_root / item.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source, destination)


def _validate_copy(manifest: pd.DataFrame, package_root: Path) -> None:
    for row in manifest.itertuples(index=False):
        destination = package_root / row.relative_path
        if not destination.is_file():
            raise FileNotFoundError(f"Staged file is missing: {destination}")
        if destination.stat().st_size != row.size_bytes or sha256(destination) != row.sha256:
            raise ValueError(f"Staged copy differs from source: {row.relative_path}")


def build_release(output_root: Path, *, source_root: Path, replace: bool = False) -> Path:
    """Stage the frozen package without modifying its sources.

    Existing output is intentionally protected unless ``replace`` is passed by
    the caller; this prevents accidentally overwriting a manually prepared
    upload directory.
    """
    package_root = output_root / RELEASE_NAME
    if package_root.exists():
        if not replace:
            raise FileExistsError(
                f"Release output already exists: {package_root}. Use --replace only for a deliberate rebuild."
            )
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=False)
    files = select_release_files(source_root)
    manifest = build_source_manifest(files)
    _copy_files(files, package_root)
    _validate_copy(manifest, package_root)
    write_release_documents(package_root, manifest, files)
    return package_root
