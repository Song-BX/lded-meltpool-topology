from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CANONICAL_TIME_S, METRICS, POWERS


def _select_region_rows(frame: pd.DataFrame, region: str, context: str) -> pd.DataFrame:
    selected = frame.loc[frame["region"] == region].copy()
    selected["power_W"] = pd.to_numeric(selected["power_W"], errors="raise").astype(int)
    if selected["power_W"].duplicated().any() or tuple(sorted(selected["power_W"])) != POWERS:
        raise ValueError(f"{context}: expected one {region} row for every power")
    return selected.set_index("power_W").loc[list(POWERS)].reset_index()


def _tail_values(tail: pd.DataFrame, time_s: float) -> pd.DataFrame:
    selected = tail.loc[np.isclose(tail["time_s"], time_s), ["power_W", "T_median_K"]].copy()
    selected["power_W"] = pd.to_numeric(selected["power_W"], errors="raise").astype(int)
    if selected["power_W"].duplicated().any() or tuple(sorted(selected["power_W"])) != POWERS:
        raise ValueError(f"Thermal tail audit at {time_s:.2f} s does not contain all sampled powers")
    return selected.set_index("power_W").loc[list(POWERS)].reset_index()


def canonical_metric_frame(canonical: pd.DataFrame, tail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        selected = _tail_values(tail, CANONICAL_TIME_S) if metric.metric_id == "temperature_median_full_pool_K" else _select_region_rows(canonical, metric.region, "canonical metrics")
        for _, row in selected.iterrows():
            rows.append(
                {
                    "time_s": CANONICAL_TIME_S,
                    "power_W": int(row["power_W"]),
                    "metric_id": metric.metric_id,
                    "metric_label": metric.label,
                    "unit": metric.unit,
                    "region": metric.region,
                    "source_column": metric.canonical_column,
                    "value": float(row[metric.canonical_column]),
                }
            )
    return pd.DataFrame(rows).sort_values(["metric_id", "power_W"], ignore_index=True)


def temporal_metric_frame(temporal: pd.DataFrame, tail: pd.DataFrame) -> pd.DataFrame:
    expected_rows = len(POWERS)
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        if metric.metric_id == "temperature_median_full_pool_K":
            for time_s in sorted(tail["time_s"].unique()):
                for _, row in _tail_values(tail, float(time_s)).iterrows():
                    rows.append({"time_s": float(time_s), "power_W": int(row["power_W"]), "metric_id": metric.metric_id, "metric_label": metric.label, "unit": metric.unit, "region": metric.region, "source_column": metric.temporal_column, "value": float(row[metric.temporal_column])})
            continue
        for time_s, block in temporal.groupby("time_s", sort=True):
            current = block.copy()
            current["power_W"] = pd.to_numeric(current["power_W"], errors="raise").astype(int)
            if len(current) != expected_rows or tuple(sorted(current["power_W"])) != POWERS:
                raise ValueError(f"Temporal metrics at {time_s} s do not contain the complete power grid")
            for _, row in current.sort_values("power_W").iterrows():
                rows.append(
                    {
                        "time_s": float(time_s),
                        "power_W": int(row["power_W"]),
                        "metric_id": metric.metric_id,
                        "metric_label": metric.label,
                        "unit": metric.unit,
                        "region": metric.region,
                        "source_column": metric.temporal_column,
                        "value": float(row[metric.temporal_column]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["time_s", "metric_id", "power_W"], ignore_index=True)


def verify_temporal_reproduction(canonical_long: pd.DataFrame, temporal_long: pd.DataFrame) -> pd.DataFrame:
    temporal_snapshot = temporal_long.loc[np.isclose(temporal_long["time_s"], CANONICAL_TIME_S)]
    joined = canonical_long.merge(
        temporal_snapshot[["power_W", "metric_id", "value"]],
        on=["power_W", "metric_id"],
        suffixes=("_canonical", "_temporal"),
        validate="one_to_one",
    )
    joined["absolute_difference"] = np.abs(joined["value_canonical"] - joined["value_temporal"])
    joined["passed"] = np.isclose(
        joined["value_canonical"], joined["value_temporal"], rtol=1e-10, atol=1e-12
    )
    if not joined["passed"].all():
        failed = joined.loc[~joined["passed"], ["power_W", "metric_id", "absolute_difference"]]
        raise ValueError(f"The temporal 0.70 s values do not reproduce canonical metrics:\n{failed}")
    return joined.sort_values(["metric_id", "power_W"], ignore_index=True)
