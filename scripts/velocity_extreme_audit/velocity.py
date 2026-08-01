from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.point_cloud import standardize_columns
from scripts.export_diagnostics.aggregation import aggregate_points
from scripts.temporal_validation.discovery import SnapshotFile

from .config import (
    AGGREGATION_STRATEGIES,
    CANONICAL_AGGREGATION,
    CANONICAL_METRICS,
    CANONICAL_TIME_S,
    PEAK_POWERS,
    VELOCITY_QUANTILES,
)


def load_standardised(snapshot: SnapshotFile) -> pd.DataFrame:
    return standardize_columns(pd.read_csv(snapshot.path))


def consolidate(frame: pd.DataFrame, strategy: str = CANONICAL_AGGREGATION) -> pd.DataFrame:
    return aggregate_points(frame, strategy)


def vector_speed(frame: pd.DataFrame) -> np.ndarray:
    return np.sqrt(
        frame["u"].to_numpy(dtype=float) ** 2
        + frame["v"].to_numpy(dtype=float) ** 2
        + frame["w"].to_numpy(dtype=float) ** 2
    )


def velocity_quantiles(snapshots: list[SnapshotFile]) -> tuple[pd.DataFrame, dict[tuple[float, int], tuple[pd.DataFrame, pd.DataFrame]]]:
    rows: list[dict[str, object]] = []
    cache: dict[tuple[float, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for snapshot in snapshots:
        raw = load_standardised(snapshot)
        canonical = consolidate(raw)
        cache[(snapshot.time_s, snapshot.power_W)] = (raw, canonical)
        values = canonical["V"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite canonical velocity magnitude at {snapshot.path.name}")
        quantile_values = np.quantile(values, VELOCITY_QUANTILES)
        row: dict[str, object] = {
            "time_s": snapshot.time_s,
            "power_W": snapshot.power_W,
            "aggregation_strategy": CANONICAL_AGGREGATION,
            "n_unique_points": len(canonical),
            "n_raw_rows": len(raw),
            "velocity_min_mps": float(np.min(values)),
            "velocity_max_mps": float(np.max(values)),
        }
        for quantile, value in zip(VELOCITY_QUANTILES, quantile_values):
            row[f"velocity_p{int(quantile * 100):02d}_mps"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["time_s", "power_W"]).reset_index(drop=True), cache


def canonical_reproduction(quantiles: pd.DataFrame) -> pd.DataFrame:
    canonical = pd.read_csv(CANONICAL_METRICS)
    expected = canonical.loc[canonical["region"] == "all", ["power_W", "v_max"]].copy()
    actual = quantiles.loc[
        np.isclose(quantiles["time_s"], CANONICAL_TIME_S), ["power_W", "velocity_max_mps"]
    ]
    merged = expected.merge(actual, on="power_W", validate="one_to_one")
    merged["absolute_difference"] = (merged["v_max"] - merged["velocity_max_mps"]).abs()
    merged["passed"] = np.isclose(
        merged["v_max"], merged["velocity_max_mps"], rtol=1e-10, atol=1e-12
    )
    if not merged["passed"].all():
        raise ValueError("Canonical 0.70 s Vmax values did not reproduce.")
    return merged.sort_values("power_W").reset_index(drop=True)


def closure_rows(
    snapshots: list[SnapshotFile],
    cache: dict[tuple[float, int], tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for snapshot in snapshots:
        raw, canonical = cache[(snapshot.time_s, snapshot.power_W)]
        for level, frame in (("raw_records", raw), ("canonical_mean_all_records", canonical)):
            exported = frame["V"].to_numpy(dtype=float)
            components = vector_speed(frame)
            finite = np.isfinite(exported) & np.isfinite(components)
            if not finite.any():
                raise ValueError(f"No finite speed values for closure at {snapshot.path.name}")
            absolute = np.abs(exported[finite] - components[finite])
            denominator = np.maximum(np.abs(components[finite]), 1e-15)
            rows.append(
                {
                    "time_s": snapshot.time_s,
                    "power_W": snapshot.power_W,
                    "aggregation_strategy": CANONICAL_AGGREGATION,
                    "closure_level": level,
                    "n_total": len(frame),
                    "n_finite": int(finite.sum()),
                    "nonfinite_count": int((~finite).sum()),
                    "max_abs_difference_mps": float(np.max(absolute)),
                    "p99_abs_difference_mps": float(np.quantile(absolute, 0.99)),
                    "mean_abs_difference_mps": float(np.mean(absolute)),
                    "max_relative_difference": float(np.max(absolute / denominator)),
                }
            )
    return pd.DataFrame(rows).sort_values(["time_s", "power_W", "closure_level"]).reset_index(drop=True)


def _direction(value_350: float, value_400: float) -> str:
    if value_350 > value_400:
        return "350>400"
    if value_350 < value_400:
        return "350<400"
    return "tie"


def aggregation_velocity_audit(
    snapshots: list[SnapshotFile],
    cache: dict[tuple[float, int], tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    reference = [snapshot for snapshot in snapshots if np.isclose(snapshot.time_s, CANONICAL_TIME_S)]
    for strategy in AGGREGATION_STRATEGIES:
        for snapshot in reference:
            raw, cached_canonical = cache[(snapshot.time_s, snapshot.power_W)]
            frame = cached_canonical if strategy == CANONICAL_AGGREGATION else consolidate(raw, strategy)
            exported = frame["V"].to_numpy(dtype=float)
            components = vector_speed(frame)
            rows.append(
                {
                    "time_s": snapshot.time_s,
                    "power_W": snapshot.power_W,
                    "aggregation_strategy": strategy,
                    "n_unique_points": len(frame),
                    "vmax_exported_magnitude_mps": float(np.max(exported)),
                    "vmax_component_norm_mps": float(np.max(components)),
                    "max_abs_pointwise_closure_difference_mps": float(np.max(np.abs(exported - components))),
                }
            )
    output = pd.DataFrame(rows)
    for strategy, indices in output.groupby("aggregation_strategy").groups.items():
        subset = output.loc[indices].set_index("power_W")
        output.loc[indices, "direction_exported_magnitude"] = _direction(
            float(subset.loc[350, "vmax_exported_magnitude_mps"]),
            float(subset.loc[400, "vmax_exported_magnitude_mps"]),
        )
        output.loc[indices, "direction_component_norm"] = _direction(
            float(subset.loc[350, "vmax_component_norm_mps"]),
            float(subset.loc[400, "vmax_component_norm_mps"]),
        )
        output.loc[indices, "definition_direction_matches"] = bool(
            output.loc[indices, "direction_exported_magnitude"].iloc[0]
            == output.loc[indices, "direction_component_norm"].iloc[0]
        )
    return output.sort_values(["aggregation_strategy", "power_W"]).reset_index(drop=True)


def peak_provenance(
    quantiles: pd.DataFrame,
    cache: dict[tuple[float, int], tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    canonical = quantiles.loc[np.isclose(quantiles["time_s"], CANONICAL_TIME_S)].set_index("power_W")
    for power_W in PEAK_POWERS:
        raw, frame = cache[(CANONICAL_TIME_S, power_W)]
        vmax = float(frame["V"].max())
        other_power = 400 if power_W == 350 else 350
        other_vmax = float(canonical.loc[other_power, "velocity_max_mps"])
        ties = frame.loc[np.isclose(frame["V"], vmax, rtol=0.0, atol=1e-14)].copy()
        for rank, point in enumerate(ties.itertuples(index=False), start=1):
            coordinate_mask = (
                np.isclose(raw["x"], point.x, rtol=0.0, atol=1e-15)
                & np.isclose(raw["y"], point.y, rtol=0.0, atol=1e-15)
                & np.isclose(raw["z"], point.z, rtol=0.0, atol=1e-15)
            )
            coordinate_rows = raw.loc[coordinate_mask]
            physical_columns = ["fof", "heat_flux", "T", "gradT", "u", "v", "w", "V"]
            rows.append(
                {
                    "time_s": CANONICAL_TIME_S,
                    "power_W": power_W,
                    "peak_rank": rank,
                    "tied_peak_coordinates": len(ties),
                    "x_m": float(point.x),
                    "y_m": float(point.y),
                    "z_m": float(point.z),
                    "velocity_magnitude_mps": float(point.V),
                    "u_mps": float(point.u),
                    "v_mps": float(point.v),
                    "w_mps": float(point.w),
                    "fof": float(point.fof),
                    "temperature_K": float(point.T),
                    "gradT_K_per_m": float(point.gradT),
                    "raw_coordinate_multiplicity": len(coordinate_rows),
                    "raw_unique_physical_states": int(len(coordinate_rows[physical_columns].drop_duplicates())),
                    "raw_coordinate_state_conflict": bool(len(coordinate_rows[physical_columns].drop_duplicates()) > 1),
                    "points_above_other_case_vmax": int((frame["V"] > other_vmax).sum()),
                    "other_case_power_W": other_power,
                    "other_case_vmax_mps": other_vmax,
                }
            )
    return pd.DataFrame(rows).sort_values(["power_W", "peak_rank"]).reset_index(drop=True)

