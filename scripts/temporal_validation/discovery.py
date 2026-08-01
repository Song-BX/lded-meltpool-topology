from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.analysis.point_cloud import COLUMN_MAP

from .config import EXPECTED_POWERS, EXPECTED_TIMES


SNAPSHOT_PATTERN = re.compile(
    r"^(?P<time>\d+(?:\.\d+)?)s_(?P<power>\d+)W\.csv$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SnapshotFile:
    time_s: float
    power_W: int
    path: Path
    sha256: str
    size_bytes: int
    data_rows: int


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _count_data_rows(path: Path) -> int:
    with path.open("rb") as stream:
        return max(sum(1 for _ in stream) - 1, 0)


def _parse_snapshot(path: Path) -> tuple[float, int] | None:
    match = SNAPSHOT_PATTERN.match(path.name)
    if not match:
        return None
    return round(float(match.group("time")), 2), int(match.group("power"))


def discover_snapshots(raw_dir: Path, temporal_dir: Path) -> list[SnapshotFile]:
    candidates = sorted(raw_dir.glob("*.csv")) + sorted(temporal_dir.glob("*.csv"))
    expected = {(time_s, power) for time_s in EXPECTED_TIMES for power in EXPECTED_POWERS}
    selected: dict[tuple[float, int], Path] = {}

    for path in candidates:
        parsed = _parse_snapshot(path)
        if parsed is None or parsed not in expected:
            continue
        if parsed in selected:
            raise ValueError(f"Duplicate snapshot for t={parsed[0]:.2f} s, P={parsed[1]} W")
        selected[parsed] = path

    missing = sorted(expected - set(selected))
    if missing:
        formatted = ", ".join(f"{time_s:.2f}s_{power}W" for time_s, power in missing)
        raise FileNotFoundError(f"Missing required temporal snapshots: {formatted}")

    required_columns = tuple(COLUMN_MAP)
    baseline_columns: tuple[str, ...] | None = None
    records: list[SnapshotFile] = []
    for key in sorted(selected):
        path = selected[key]
        columns = tuple(pd.read_csv(path, nrows=0).columns)
        missing_columns = [column for column in required_columns if column not in columns]
        if missing_columns:
            raise ValueError(f"{path.name} is missing required columns: {missing_columns}")
        if baseline_columns is None:
            baseline_columns = columns
        elif columns != baseline_columns:
            raise ValueError(f"Column order/schema differs from the first snapshot: {path.name}")
        records.append(
            SnapshotFile(
                time_s=key[0],
                power_W=key[1],
                path=path,
                sha256=_hash_file(path),
                size_bytes=path.stat().st_size,
                data_rows=_count_data_rows(path),
            )
        )
    return records


def snapshot_manifest(records: list[SnapshotFile], project_root: Path) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "time_s": record.time_s,
                "power_W": record.power_W,
                "relative_path": record.path.relative_to(project_root).as_posix(),
                "size_bytes": record.size_bytes,
                "data_rows": record.data_rows,
                "sha256": record.sha256,
            }
        )
    return pd.DataFrame(rows).sort_values(["time_s", "power_W"]).reset_index(drop=True)
