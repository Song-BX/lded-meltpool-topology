from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import (
    AGGREGATION_METRICS,
    CANONICAL_METRICS,
    POWERS,
    TEMPORAL_METRICS,
    THERMAL_TAIL_METRICS,
)
from .temperature_median import aggregation_median_temperature_frame


@dataclass(frozen=True)
class AuditInputs:
    canonical: pd.DataFrame
    temporal: pd.DataFrame
    aggregation: pd.DataFrame
    thermal_tail: pd.DataFrame
    median_aggregation: pd.DataFrame
    manifest: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required audit input is missing: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return frame


def _require_powers(frame: pd.DataFrame, column: str, context: str) -> None:
    observed = tuple(sorted(pd.to_numeric(frame[column], errors="raise").astype(int).unique()))
    if observed != POWERS:
        raise ValueError(f"{context} must contain exactly {POWERS}; observed {observed}")


def load_inputs() -> AuditInputs:
    canonical = _read_csv(
        CANONICAL_METRICS,
        {"power_W", "region", "v_mean", "v_max", "T_mean_K", "T_max_K"},
    )
    temporal = _read_csv(
        TEMPORAL_METRICS,
        {
            "time_s",
            "power_W",
            "temperature_mean_all_K",
            "temperature_max_all_K",
            "velocity_max_all_mps",
            "velocity_mean_interface_mps",
        },
    )
    aggregation = _read_csv(
        AGGREGATION_METRICS,
        {
            "aggregation_strategy",
            "power_W",
            "region",
            "v_mean",
            "v_max",
            "T_mean_K",
            "T_max_K",
        },
    )
    thermal_tail = _read_csv(
        THERMAL_TAIL_METRICS,
        {"time_s", "power_W", "representation", "T_median_K", "T_mean_K", "T_max_K"},
    )
    thermal_tail = thermal_tail.loc[thermal_tail["representation"] == "exact_coordinate_mean"].copy()
    if len(thermal_tail) != 30:
        raise ValueError("Thermal-tail audit must provide one exact-coordinate row for each 5x6 snapshot")
    median_aggregation = aggregation_median_temperature_frame()
    _require_powers(canonical, "power_W", "Canonical metrics")
    _require_powers(temporal, "power_W", "Temporal metrics")
    _require_powers(aggregation, "power_W", "Aggregation metrics")

    manifest_rows = []
    for role, path, frame in (
        ("canonical_0.70_s_metrics", CANONICAL_METRICS, canonical),
        ("temporal_metrics", TEMPORAL_METRICS, temporal),
        ("aggregation_sensitivity_metrics", AGGREGATION_METRICS, aggregation),
        ("thermal_tail_metrics", THERMAL_TAIL_METRICS, thermal_tail),
    ):
        manifest_rows.append(
            {
                "input_role": role,
                "relative_path": path.relative_to(path.parents[2]).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(frame),
                "columns": ";".join(frame.columns),
            }
        )
    manifest_rows.append(
        {
            "input_role": "unfiltered_median_aggregation_source",
            "relative_path": "raw data/; raw data/temporal_validation/",
            "sha256": "per-file hashes retained by thermal_fidelity_audit",
            "bytes": "",
            "rows": len(median_aggregation),
            "columns": ";".join(median_aggregation.columns),
        }
    )
    return AuditInputs(canonical, temporal, aggregation, thermal_tail, median_aggregation, pd.DataFrame(manifest_rows))
