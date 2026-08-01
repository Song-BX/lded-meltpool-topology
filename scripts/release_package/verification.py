"""Validate integrity and scope of the source-only R1 release package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis.point_cloud import COLUMN_MAP

from .config import (
    EXCLUDED_DIRECTORY_NAMES,
    EXCLUDED_SUFFIXES,
    RAW_CSV_COUNT,
    REQUIRED_GENERATED_FILES,
    RETIRED_ROOT_SCRIPTS,
)
from .inventory import build_source_manifest, select_release_files, sha256


FORBIDDEN_SUFFIXES = frozenset(
    {".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".opju", ".tif", ".tiff", ".vsdx"}
)
GENERATED_DIRECTORIES = frozenset({"图", "latex_restructure"})
ALLOWED_CATEGORIES = frozenset({"raw_input", "reference_input", "reproduction_metadata", "code", "test"})


def _fail(message: str) -> None:
    raise ValueError(f"Release verification failed: {message}")


def _read_checksum_file(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Malformed SHA256SUMS entry: {line!r}") from exc
        if len(digest) != 64 or relative in entries:
            raise ValueError(f"Invalid SHA256SUMS entry: {line!r}")
        entries[relative] = digest
    return entries


def _generated_path(relative: Path) -> bool:
    return bool(relative.parts) and relative.parts[0] in GENERATED_DIRECTORIES


def _verify_generated_files(package_root: Path) -> None:
    missing = [name for name in REQUIRED_GENERATED_FILES if not (package_root / name).is_file()]
    if missing:
        _fail(f"generated files are missing: {missing}")


def _verify_hash_sums(package_root: Path, *, allow_generated: bool) -> None:
    entries = _read_checksum_file(package_root / "SHA256SUMS.txt")
    present = {
        path.relative_to(package_root).as_posix(): path
        for path in package_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    immutable = {
        relative: path
        for relative, path in present.items()
        if not _generated_path(Path(relative))
    }
    if set(entries) != set(immutable):
        _fail("SHA256SUMS file list does not match immutable package contents")
    if not allow_generated and len(immutable) != len(present):
        _fail("generated outputs are present in a source-only package")
    mismatches = [relative for relative, path in immutable.items() if sha256(path) != entries[relative]]
    if mismatches:
        _fail(f"SHA-256 mismatch for: {mismatches[:5]}")


def _verify_no_out_of_scope_content(package_root: Path, *, allow_generated: bool) -> None:
    violations: list[str] = []
    for path in package_root.rglob("*"):
        relative = path.relative_to(package_root)
        if not relative.parts:
            continue
        if relative.parts[0] in GENERATED_DIRECTORIES:
            if not allow_generated:
                violations.append(relative.as_posix())
            continue
        if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
            violations.append(relative.as_posix())
            continue
        if path.is_file() and path.suffix.lower() in EXCLUDED_SUFFIXES | FORBIDDEN_SUFFIXES:
            violations.append(relative.as_posix())
            continue
        if path.is_file() and relative.parent == Path("scripts") and path.name in RETIRED_ROOT_SCRIPTS:
            violations.append(relative.as_posix())
    if violations:
        _fail(f"out-of-scope files found: {violations[:10]}")


def _verify_raw_data(package_root: Path) -> None:
    raw_dir = package_root / "raw data"
    inputs = sorted(raw_dir.rglob("*.csv"))
    if len(inputs) != RAW_CSV_COUNT:
        _fail(f"expected {RAW_CSV_COUNT} raw CSV inputs, found {len(inputs)}")
    expected_fields = list(COLUMN_MAP)
    for path in inputs:
        fields = list(pd.read_csv(path, nrows=0).columns)
        if fields != expected_fields:
            _fail(f"unexpected raw-field schema in {path.relative_to(package_root)}")


def _verify_manifest(package_root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(package_root / "RELEASE_CONTENTS.csv")
    required_columns = {"relative_path", "category", "size_bytes", "sha256"}
    if set(manifest.columns) != required_columns:
        _fail(f"unexpected manifest columns: {list(manifest.columns)}")
    categories = set(manifest["category"])
    if not categories <= ALLOWED_CATEGORIES:
        _fail(f"out-of-scope manifest categories: {sorted(categories - ALLOWED_CATEGORIES)}")
    staged_paths = set(manifest["relative_path"].astype(str))
    if any(Path(path).parts[0] in GENERATED_DIRECTORIES for path in staged_paths):
        _fail("generated outputs must not be listed in the source manifest")
    for row in manifest.itertuples(index=False):
        path = package_root / row.relative_path
        if not path.is_file() or path.stat().st_size != row.size_bytes or sha256(path) != row.sha256:
            _fail(f"staged file differs from manifest: {row.relative_path}")
    return manifest


def _verify_source_provenance(package_root: Path, source_root: Path) -> None:
    actual = _verify_manifest(package_root)
    expected = build_source_manifest(select_release_files(source_root))
    columns = ["relative_path", "category", "size_bytes", "sha256"]
    actual = actual[columns].sort_values("relative_path").reset_index(drop=True)
    expected = expected[columns].sort_values("relative_path").reset_index(drop=True)
    if not actual.equals(expected):
        _fail("RELEASE_CONTENTS.csv does not exactly match the source-only allow-list")


def verify_package(
    package_root: Path,
    source_root: Path | None = None,
    *,
    allow_generated: bool = False,
) -> dict[str, int]:
    """Check the source-only package and optionally permit local derived outputs."""
    package_root = package_root.resolve()
    if not package_root.is_dir():
        raise FileNotFoundError(f"Release package not found: {package_root}")
    _verify_generated_files(package_root)
    _verify_no_out_of_scope_content(package_root, allow_generated=allow_generated)
    _verify_hash_sums(package_root, allow_generated=allow_generated)
    manifest = _verify_manifest(package_root)
    _verify_raw_data(package_root)
    if source_root is not None and not allow_generated:
        _verify_source_provenance(package_root, source_root.resolve())
    return {
        "raw_csv_count": RAW_CSV_COUNT,
        "source_files": int(len(manifest)),
        "code_files": int((manifest["category"] == "code").sum()),
    }
