from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.point_cloud import deduplicate_points, standardize_columns
from scripts.analysis.wls_q import reconstruct_case

from .config import (
    COORDINATE_TOLERANCE_M,
    FOF_INTERFACE_THRESHOLD,
    K_REFERENCE,
    MIN_NEIGHBOR_DISTANCE_M,
    WLS_CONDITION_CUTOFF,
    WLS_CONDITION_MODE,
    WLS_DISTANCE_EXPONENT,
)
from .discovery import SnapshotFile


def _positive_fraction(values: pd.Series) -> float:
    valid = values.dropna()
    return float((valid > 0).mean()) if len(valid) else np.nan


def compute_snapshot_metrics(record: SnapshotFile) -> dict[str, float | int]:
    raw = pd.read_csv(record.path)
    standardized = standardize_columns(raw)
    deduplicated = deduplicate_points(standardized, eps_c=COORDINATE_TOLERANCE_M)
    reconstructed = reconstruct_case(
        deduplicated,
        k=K_REFERENCE,
        alpha=WLS_DISTANCE_EXPONENT,
        kappa_max=WLS_CONDITION_CUTOFF,
        min_neighbor_distance=MIN_NEIGHBOR_DISTANCE_M,
        condition_on=WLS_CONDITION_MODE,
    )

    valid = reconstructed["chi"] == 1
    interface = valid & (reconstructed["fof"] < FOF_INTERFACE_THRESHOLD)
    if not valid.any():
        raise ValueError(f"No WLS-valid points for {record.path.name}")
    if not interface.any():
        raise ValueError(f"No WLS-valid interface points for {record.path.name}")

    spans = {
        axis: float(deduplicated[axis].max() - deduplicated[axis].min())
        for axis in ("x", "y", "z")
    }
    return {
        "time_s": record.time_s,
        "power_W": record.power_W,
        "raw_points": int(len(standardized)),
        "unique_points": int(len(deduplicated)),
        "dedup_ratio": float((len(standardized) - len(deduplicated)) / len(standardized)),
        "span_x_m": spans["x"],
        "span_y_m": spans["y"],
        "span_z_m": spans["z"],
        "wls_valid_points": int(valid.sum()),
        "wls_valid_fraction": float(valid.mean()),
        "temperature_mean_all_K": float(reconstructed.loc[valid, "T"].mean()),
        "temperature_max_all_K": float(reconstructed.loc[valid, "T"].max()),
        "velocity_max_all_mps": float(reconstructed.loc[valid, "V"].max()),
        "velocity_mean_interface_mps": float(reconstructed.loc[interface, "V"].mean()),
        "q_positive_fraction_all": _positive_fraction(reconstructed.loc[valid, "Q"]),
        "q_positive_fraction_interface": _positive_fraction(
            reconstructed.loc[interface, "Q"]
        ),
    }


def compute_temporal_metrics(records: list[SnapshotFile]) -> pd.DataFrame:
    rows = [compute_snapshot_metrics(record) for record in records]
    return pd.DataFrame(rows).sort_values(["power_W", "time_s"]).reset_index(drop=True)
