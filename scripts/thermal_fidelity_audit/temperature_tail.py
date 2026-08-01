from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.point_cloud import standardize_columns
from scripts.export_diagnostics.aggregation import aggregate_points
from scripts.temporal_validation.discovery import SnapshotFile

from .config import HIGH_TEMPERATURE_THRESHOLD_K, SATURATION_TEMPERATURE_K


def _statistics(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Temperature field has no finite records")
    p25, median, p75, p95, p99 = np.quantile(finite, [0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n_total": int(values.size),
        "n_finite": int(finite.size),
        "finite_fraction": float(finite.size / values.size),
        "T_mean_K": float(np.mean(finite)),
        "T_p25_K": float(p25),
        "T_median_K": float(median),
        "T_p75_K": float(p75),
        "T_p95_K": float(p95),
        "T_p99_K": float(p99),
        "T_max_K": float(np.max(finite)),
        "n_T_ge_Tsat": int(np.sum(finite >= SATURATION_TEMPERATURE_K)),
        "n_T_gt_5000": int(np.sum(finite > HIGH_TEMPERATURE_THRESHOLD_K)),
    }


def _frames(snapshot: SnapshotFile) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = standardize_columns(pd.read_csv(snapshot.path))
    unique = aggregate_points(raw, "mean_all_records")
    return raw, unique


def temperature_tail_metrics(snapshots: list[SnapshotFile]) -> tuple[pd.DataFrame, dict[tuple[float, int], tuple[pd.DataFrame, pd.DataFrame]]]:
    rows: list[dict[str, object]] = []
    cache: dict[tuple[float, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for snapshot in snapshots:
        raw, unique = _frames(snapshot)
        cache[(snapshot.time_s, snapshot.power_W)] = (raw, unique)
        for representation, frame in (("raw_records", raw), ("exact_coordinate_mean", unique)):
            row: dict[str, object] = {
                "time_s": snapshot.time_s,
                "power_W": snapshot.power_W,
                "representation": representation,
                "saturation_temperature_K": SATURATION_TEMPERATURE_K,
                "high_temperature_threshold_K": HIGH_TEMPERATURE_THRESHOLD_K,
            }
            row.update(_statistics(frame["T"].to_numpy(dtype=float)))
            row["interpretation_status"] = "snapshot_local_descriptor" if representation == "exact_coordinate_mean" else "raw_export_audit"
            row["interpretation_boundary"] = "Unfiltered exported numerical temperatures; not an experimental-fidelity, phase-accuracy, or solver-health result."
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["time_s", "power_W", "representation"]).reset_index(drop=True), cache


def temperature_tail_sensitivity(cache: dict[tuple[float, int], tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    conditions = (
        ("unfiltered", lambda values: np.ones(values.shape, dtype=bool)),
        ("exclude_T_gt_5000_K", lambda values: values <= HIGH_TEMPERATURE_THRESHOLD_K),
        ("exclude_T_ge_Tsat", lambda values: values < SATURATION_TEMPERATURE_K),
    )
    for (time_s, power_W), (_, unique) in sorted(cache.items()):
        values = unique["T"].to_numpy(dtype=float)
        finite = np.isfinite(values)
        baseline = _statistics(values[finite])
        for condition, selector in conditions:
            kept = values[finite & selector(values)]
            if kept.size == 0:
                raise ValueError(f"{condition} removed every finite temperature at {time_s}/{power_W}")
            summary = _statistics(kept)
            rows.append(
                {
                    "time_s": time_s,
                    "power_W": power_W,
                    "representation": "exact_coordinate_mean",
                    "sensitivity_condition": condition,
                    "n_retained": int(kept.size),
                    "fraction_retained": float(kept.size / finite.sum()),
                    **summary,
                    "mean_delta_from_unfiltered_K": float(summary["T_mean_K"] - baseline["T_mean_K"]),
                    "mean_delta_from_unfiltered_percent": float((summary["T_mean_K"] / baseline["T_mean_K"] - 1.0) * 100.0),
                    "median_delta_from_unfiltered_K": float(summary["T_median_K"] - baseline["T_median_K"]),
                    "median_delta_from_unfiltered_percent": float((summary["T_median_K"] / baseline["T_median_K"] - 1.0) * 100.0),
                    "interpretation_boundary": "Sensitivity only; unfiltered exact-coordinate values remain canonical and no observation is deleted from the audit.",
                }
            )
    return pd.DataFrame(rows).sort_values(["time_s", "power_W", "sensitivity_condition"]).reset_index(drop=True)


def temperature_extreme_context(cache: dict[tuple[float, int], tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (time_s, power_W), (raw, unique) in sorted(cache.items()):
        maximum = float(unique["T"].max())
        tied = unique.loc[np.isclose(unique["T"], maximum, rtol=0.0, atol=1e-12)]
        for rank, point in enumerate(tied.itertuples(index=False), start=1):
            matching = raw.loc[(raw["x"] == point.x) & (raw["y"] == point.y) & (raw["z"] == point.z)]
            rows.append(
                {
                    "time_s": time_s,
                    "power_W": power_W,
                    "peak_rank": rank,
                    "tied_maximum_exported_coordinates": len(tied),
                    "x_m": float(point.x),
                    "y_m": float(point.y),
                    "z_m": float(point.z),
                    "T_max_K": float(point.T),
                    "fof": float(point.fof),
                    "heat_flux": float(point.heat_flux),
                    "raw_coordinate_multiplicity": int(len(matching)),
                    "n_unique_T_gt_5000": int((unique["T"] > HIGH_TEMPERATURE_THRESHOLD_K).sum()),
                    "n_unique_T_ge_Tsat": int((unique["T"] >= SATURATION_TEMPERATURE_K).sum()),
                    "entity_semantics": "unique exported coordinate after exact-coordinate aggregation; CSV has no native cell identifier",
                    "interpretation_status": "audit_only",
                    "interpretation_boundary": "Peak-level numerical-output context only; does not diagnose a mesh-cell instability or its cause.",
                }
            )
    return pd.DataFrame(rows).sort_values(["time_s", "power_W", "peak_rank"]).reset_index(drop=True)

