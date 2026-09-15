from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.analysis.point_cloud import COLUMN_MAP

from .config import (
    EXPECTED_POWERS,
    EXPECTED_TIMES,
    OPTIONAL_AUDIT_DIR,
    RAW_DIR,
    ROOT,
    TEMPORAL_DIR,
    TIME_LABELS,
)


@dataclass(frozen=True)
class SnapshotInput:
    time_s: float
    power_W: int
    path: Path
    size_bytes: int
    sha256: str
    schema_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_path(time_s: float, power_W: int) -> Path:
    directory = RAW_DIR if time_s == 0.70 else TEMPORAL_DIR
    return directory / f"{TIME_LABELS[time_s]}s_{power_W}W.csv"


def discover_snapshots() -> list[SnapshotInput]:
    expected_columns = list(COLUMN_MAP)
    schema_hash = hashlib.sha256("\n".join(expected_columns).encode("utf-8")).hexdigest()
    records: list[SnapshotInput] = []
    missing: list[str] = []
    for time_s in EXPECTED_TIMES:
        for power_W in EXPECTED_POWERS:
            path = _expected_path(time_s, power_W)
            if not path.exists():
                missing.append(path.relative_to(ROOT).as_posix())
                continue
            columns = pd.read_csv(path, nrows=0).columns.tolist()
            if columns != expected_columns:
                raise ValueError(
                    f"Schema mismatch for {path.name}: expected {expected_columns}, received {columns}"
                )
            records.append(
                SnapshotInput(
                    time_s=time_s,
                    power_W=power_W,
                    path=path,
                    size_bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                    schema_sha256=schema_hash,
                )
            )
    if missing:
        raise FileNotFoundError(f"Missing required snapshots: {missing}")
    return records


def manifest_frame(records: list[SnapshotInput]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time_s": record.time_s,
                "power_W": record.power_W,
                "path": record.path.relative_to(ROOT).as_posix(),
                "size_bytes": record.size_bytes,
                "sha256": record.sha256,
                "schema_sha256": record.schema_sha256,
            }
            for record in records
        ]
    ).sort_values(["time_s", "power_W"])


def optional_reexport_manifest() -> pd.DataFrame:
    columns = ["path", "size_bytes", "sha256"]
    if not OPTIONAL_AUDIT_DIR.exists():
        return pd.DataFrame(columns=columns)
    rows = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(OPTIONAL_AUDIT_DIR.glob("*.csv"))
    ]
    return pd.DataFrame(rows, columns=columns)
