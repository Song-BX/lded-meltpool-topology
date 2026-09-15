from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.analysis.point_cloud import COLUMN_MAP

from .config import EXPECTED_POWERS, RAW_DIR, ROOT


@dataclass(frozen=True)
class CaseInput:
    power_W: int
    path: Path
    size_bytes: int
    sha256: str
    schema_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_inputs(raw_dir: Path = RAW_DIR) -> list[CaseInput]:
    records: list[CaseInput] = []
    missing: list[str] = []
    expected_columns = list(COLUMN_MAP)
    expected_signature = hashlib.sha256("\n".join(expected_columns).encode("utf-8")).hexdigest()
    for power in EXPECTED_POWERS:
        path = raw_dir / f"0.7s_{power}W.csv"
        if not path.exists():
            missing.append(path.name)
            continue
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        if columns != expected_columns:
            raise ValueError(
                f"Schema mismatch for {path.name}: expected {expected_columns}, received {columns}"
            )
        records.append(
            CaseInput(
                power_W=power,
                path=path,
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
                schema_sha256=expected_signature,
            )
        )
    if missing:
        raise FileNotFoundError(f"Missing required 0.70 s cases: {missing}")
    return records


def manifest_frame(records: list[CaseInput]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "power_W": record.power_W,
                "path": record.path.relative_to(ROOT).as_posix(),
                "size_bytes": record.size_bytes,
                "sha256": record.sha256,
                "schema_sha256": record.schema_sha256,
            }
            for record in records
        ]
    )
