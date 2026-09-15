from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd

from .paths import AUDIT_DIR, INPUT_SPECS, SOURCE_INPUT_DIR, InputSpec, relative_to_root


REQUIRED_POINT_COLUMNS = {"x_m", "y_m", "z_m", "Q", "is_slice_near_symmetry", "power_W"}
REQUIRED_LEGACY_COLUMNS = {"metric", "n_top", "power_W"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_path(spec: InputSpec) -> Path:
    if spec.destination_path.exists():
        return spec.destination_path
    return spec.original_path


def validate_source_availability() -> None:
    missing = [str(spec.original_path) for spec in INPUT_SPECS if not active_path(spec).exists()]
    if missing:
        raise FileNotFoundError(f"Missing spatial-audit inputs: {missing}")


def move_inputs() -> pd.DataFrame:
    """Move the exact CSV inputs once, preserving their hashes and provenance."""
    validate_source_availability()
    SOURCE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for spec in INPUT_SPECS:
        before_path = active_path(spec)
        before_hash = sha256(before_path)
        moved = False
        if before_path == spec.original_path and not spec.destination_path.exists():
            shutil.move(str(spec.original_path), str(spec.destination_path))
            moved = True
        after_path = active_path(spec)
        after_hash = sha256(after_path)
        if before_hash != after_hash:
            raise RuntimeError(f"SHA-256 changed while moving {spec.key}")
        records.append(
            {
                "input_key": spec.key,
                "original_path": relative_to_root(spec.original_path),
                "active_path": relative_to_root(after_path),
                "size_bytes": after_path.stat().st_size,
                "sha256_before_move": before_hash,
                "sha256_after_move": after_hash,
                "moved_this_run": moved,
            }
        )
    return pd.DataFrame(records)


def build_manifest() -> pd.DataFrame:
    validate_source_availability()
    records: list[dict[str, object]] = []
    for spec in INPUT_SPECS:
        path = active_path(spec)
        digest = sha256(path)
        records.append(
            {
                "input_key": spec.key,
                "original_path": relative_to_root(spec.original_path),
                "active_path": relative_to_root(path),
                "size_bytes": path.stat().st_size,
                "sha256_before_move": digest,
                "sha256_after_move": digest,
                "moved_this_run": False,
            }
        )
    return pd.DataFrame(records)


def _read_csv(spec_key: str) -> pd.DataFrame:
    spec = next(item for item in INPUT_SPECS if item.key == spec_key)
    return pd.read_csv(active_path(spec))


def load_point_clouds() -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for spec_key in ("qpoints_350", "qpoints_400"):
        frame = _read_csv(spec_key)
        missing = REQUIRED_POINT_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{spec_key} is missing columns: {sorted(missing)}")
        powers = frame["power_W"].dropna().unique()
        if len(powers) != 1:
            raise ValueError(f"{spec_key} must contain exactly one power, found {powers.tolist()}")
        frames[int(powers[0])] = frame
    return frames


def load_legacy_summary() -> pd.DataFrame:
    frame = _read_csv("legacy_extreme_summary")
    missing = REQUIRED_LEGACY_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"legacy extreme summary is missing columns: {sorted(missing)}")
    return frame


def write_manifest(manifest: pd.DataFrame) -> Path:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIT_DIR / "spatial_support_input_manifest.csv"
    manifest.to_csv(path, index=False, encoding="utf-8-sig")
    return path
