"""Select release files and compute a provenance-preserving inventory."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import RAW_DATA_DIR, RETAINED_SCRIPT_DIRECTORIES, ROOT, SCRIPTS_DIR, TESTS_DIR


@dataclass(frozen=True)
class ReleaseFile:
    """One whitelisted source file and its destination relative path."""

    source: Path
    relative_path: Path
    category: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path, predicate) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and predicate(path):
            yield path


def select_release_files(source_root: Path = ROOT) -> list[ReleaseFile]:
    """Return the fixed allow-list for a source tree with the project layout."""
    source_root = source_root.resolve()
    raw_dir = source_root / RAW_DATA_DIR.relative_to(ROOT)
    scripts_dir = source_root / SCRIPTS_DIR.relative_to(ROOT)
    tests_dir = source_root / TESTS_DIR.relative_to(ROOT)
    chosen: list[ReleaseFile] = []

    for filename in ("requirements.txt", "Flow3D设置.txt", "Flow3D.md", "running.md"):
        path = source_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required release file is missing: {path}")
        chosen.append(ReleaseFile(path, Path(filename), "reproduction_metadata"))

    reference_inputs = (
        (Path("图/3/Aplus_main_metrics_k25.csv"), Path("reference_data/Aplus_main_metrics_k25.csv")),
        (
            Path("图/7/Aplus_Qthreshold_sensitivity_350vs400.csv"),
            Path("reference_data/Aplus_Qthreshold_sensitivity_350vs400.csv"),
        ),
        (
            Path("图/robustness/knn_core_contrasts.csv"),
            Path("reference_data/knn_core_contrasts.csv"),
        ),
        (
            Path("图/spatial_support_audit/source_inputs/Qpoints_350W_k25.csv"),
            Path("reference_data/Qpoints_350W_k25.csv"),
        ),
        (
            Path("图/spatial_support_audit/source_inputs/Qpoints_400W_k25.csv"),
            Path("reference_data/Qpoints_400W_k25.csv"),
        ),
        (
            Path("图/spatial_support_audit/source_inputs/Aplus_extreme_localization_350_400.csv"),
            Path("reference_data/Aplus_extreme_localization_350_400.csv"),
        ),
        (
            Path("scripts/model_fidelity_boundary/prior_model_validation_record.json"),
            Path("reference_data/prior_model_validation_record.json"),
        ),
    )
    for source_relative, destination_relative in reference_inputs:
        path = source_root / source_relative
        if not path.is_file():
            raise FileNotFoundError(f"Required reference input is missing: {path}")
        chosen.append(ReleaseFile(path, destination_relative, "reference_input"))

    # The solver-history mapping template documents an unavailable future input;
    # it is not a 31st scientific point-cloud snapshot and is excluded from the
    # frozen 30-file input grid.  Its absence keeps the health gate audit-only.
    solver_history_dir = raw_dir / "solver_numerics"
    chosen.extend(
        ReleaseFile(path, path.relative_to(source_root), "raw_input")
        for path in _files(
            raw_dir,
            lambda item: item.suffix.lower() == ".csv" and solver_history_dir not in item.parents,
        )
    )
    init_file = scripts_dir / "__init__.py"
    if init_file.is_file():
        chosen.append(ReleaseFile(init_file, init_file.relative_to(source_root), "code"))
    for directory_name in RETAINED_SCRIPT_DIRECTORIES:
        directory = scripts_dir / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Retained module directory is missing: {directory}")
        chosen.extend(
            ReleaseFile(path, path.relative_to(source_root), "code")
            for path in _files(directory, lambda item: item.suffix.lower() == ".py")
        )
    chosen.extend(
        ReleaseFile(path, path.relative_to(source_root), "test")
        for path in _files(tests_dir, lambda item: item.suffix.lower() == ".py")
    )

    by_destination: dict[Path, ReleaseFile] = {}
    for item in chosen:
        if item.relative_path in by_destination:
            raise ValueError(f"Release allow-list contains a duplicate destination: {item.relative_path}")
        by_destination[item.relative_path] = item
    return [by_destination[key] for key in sorted(by_destination, key=lambda value: value.as_posix())]


def build_source_manifest(files: Iterable[ReleaseFile]) -> pd.DataFrame:
    rows = []
    for item in files:
        rows.append(
            {
                "relative_path": item.relative_path.as_posix(),
                "category": item.category,
                "source_path": str(item.source),
                "size_bytes": item.source.stat().st_size,
                "sha256": sha256(item.source),
            }
        )
    return pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)


def raw_csv_headers(files: Iterable[ReleaseFile]) -> pd.DataFrame:
    rows = []
    for item in files:
        if item.category != "raw_input":
            continue
        fields = list(pd.read_csv(item.source, nrows=0).columns)
        rows.append(
            {
                "relative_path": item.relative_path.as_posix(),
                "field_count": len(fields),
                "fields": " | ".join(fields),
            }
        )
    return pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)
