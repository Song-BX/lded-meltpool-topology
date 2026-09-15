from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.temporal_validation.discovery import SnapshotFile, discover_snapshots, snapshot_manifest

from .config import (
    MAPPING_COLUMNS,
    MAPPING_PATH,
    PEAK_POWERS,
    ROOT,
    SOLVER_DIR,
    SOLVER_ROLES,
    TEMPORAL_DIR,
    RAW_DIR,
)


@dataclass(frozen=True)
class VelocityAuditInputs:
    snapshots: list[SnapshotFile]
    manifest: pd.DataFrame
    solver_mapping: pd.DataFrame
    mapping_issues: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _empty_mapping() -> pd.DataFrame:
    return pd.DataFrame(columns=MAPPING_COLUMNS)


def _mapping_manifest(mapping: pd.DataFrame, issues: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if mapping.empty:
        for power_W in PEAK_POWERS:
            for role in SOLVER_ROLES:
                rows.append(
                    {
                        "input_kind": "solver_history",
                        "time_s": "",
                        "power_W": power_W,
                        "role": role,
                        "relative_path": "",
                        "size_bytes": "",
                        "sha256": "",
                        "availability": "not_available",
                        "validation_detail": "solver_history_mapping.csv is absent",
                    }
                )
        return pd.DataFrame(rows)

    for row in mapping.itertuples(index=False):
        relative = Path(str(row.raw_file))
        path = (SOLVER_DIR / relative).resolve()
        is_within_solver_dir = SOLVER_DIR.resolve() in path.parents or path == SOLVER_DIR.resolve()
        if not is_within_solver_dir:
            rows.append(
                {
                    "input_kind": "solver_history",
                    "time_s": "",
                    "power_W": int(row.power_W),
                    "role": str(row.role),
                    "relative_path": str(row.raw_file),
                    "size_bytes": "",
                    "sha256": "",
                    "availability": "invalid_path",
                    "validation_detail": "raw_file resolves outside raw data/solver_numerics",
                }
            )
        elif path.is_file():
            rows.append(
                {
                    "input_kind": "solver_history",
                    "time_s": "",
                    "power_W": int(row.power_W),
                    "role": str(row.role),
                    "relative_path": path.relative_to(ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "availability": "available",
                    "validation_detail": "mapped raw solver-history file",
                }
            )
        else:
            rows.append(
                {
                    "input_kind": "solver_history",
                    "time_s": "",
                    "power_W": int(row.power_W),
                    "role": str(row.role),
                    "relative_path": str(row.raw_file),
                    "size_bytes": "",
                    "sha256": "",
                    "availability": "not_available",
                    "validation_detail": "mapped raw solver-history file is absent",
                }
            )
    if issues:
        rows.append(
            {
                "input_kind": "solver_history_mapping",
                "time_s": "",
                "power_W": "",
                "role": "",
                "relative_path": MAPPING_PATH.relative_to(ROOT).as_posix(),
                "size_bytes": MAPPING_PATH.stat().st_size if MAPPING_PATH.exists() else "",
                "sha256": sha256_file(MAPPING_PATH) if MAPPING_PATH.exists() else "",
                "availability": "invalid",
                "validation_detail": "; ".join(issues),
            }
        )
    return pd.DataFrame(rows)


def load_solver_mapping() -> tuple[pd.DataFrame, tuple[str, ...]]:
    if not MAPPING_PATH.is_file():
        return _empty_mapping(), ("solver_history_mapping.csv is absent",)
    mapping = pd.read_csv(MAPPING_PATH, dtype=str, keep_default_na=False)
    issues: list[str] = []
    missing = [column for column in MAPPING_COLUMNS if column not in mapping.columns]
    if missing:
        issues.append(f"mapping lacks required columns: {', '.join(missing)}")
        return _empty_mapping(), tuple(issues)
    mapping = mapping.loc[:, list(MAPPING_COLUMNS)].copy()
    mapping["power_W"] = pd.to_numeric(mapping["power_W"], errors="coerce")
    if mapping["power_W"].isna().any() or not set(mapping["power_W"].astype(int)).issubset(PEAK_POWERS):
        issues.append("mapping power_W must be 350 or 400")
    else:
        mapping["power_W"] = mapping["power_W"].astype(int)
    if not set(mapping["role"]).issubset(SOLVER_ROLES):
        issues.append("mapping contains an unrecognised solver-history role")
    if not issues and mapping.duplicated(["power_W", "role"]).any():
        issues.append("mapping has duplicate power_W/role rows")
    expected = {(power_W, role) for power_W in PEAK_POWERS for role in SOLVER_ROLES}
    observed = set(zip(mapping.get("power_W", []), mapping.get("role", [])))
    if not issues and observed != expected:
        missing_pairs = sorted(expected - observed)
        extra_pairs = sorted(observed - expected)
        if missing_pairs:
            issues.append(f"mapping lacks required power/role pairs: {missing_pairs}")
        if extra_pairs:
            issues.append(f"mapping has unexpected power/role pairs: {extra_pairs}")
    return mapping, tuple(issues)


def load_inputs() -> VelocityAuditInputs:
    snapshots = discover_snapshots(RAW_DIR, TEMPORAL_DIR)
    point_manifest = snapshot_manifest(snapshots, ROOT).rename(columns={"relative_path": "relative_path"})
    point_manifest.insert(0, "input_kind", "point_cloud")
    point_manifest["role"] = ""
    point_manifest["availability"] = "available"
    point_manifest["validation_detail"] = "complete 30-cell time-power input grid"
    mapping, mapping_issues = load_solver_mapping()
    solver_manifest = _mapping_manifest(mapping, list(mapping_issues))
    manifest_columns = (
        "input_kind",
        "time_s",
        "power_W",
        "role",
        "relative_path",
        "size_bytes",
        "sha256",
        "availability",
        "validation_detail",
    )
    point_manifest = point_manifest.loc[:, list(manifest_columns)]
    solver_manifest = solver_manifest.reindex(columns=manifest_columns)
    manifest = pd.concat([point_manifest, solver_manifest], ignore_index=True)
    return VelocityAuditInputs(
        snapshots=snapshots,
        manifest=manifest,
        solver_mapping=mapping,
        mapping_issues=mapping_issues,
    )

