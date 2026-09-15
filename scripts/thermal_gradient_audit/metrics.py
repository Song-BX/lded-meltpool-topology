from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.export_diagnostics.aggregation import aggregate_points
from scripts.analysis.point_cloud import standardize_columns
from scripts.temporal_validation.discovery import SnapshotFile

from .config import CANONICAL_AGGREGATION, REGIONS, SUPPORT_GATE


def consolidate_snapshot(snapshot: SnapshotFile, aggregation_strategy: str) -> pd.DataFrame:
    raw = pd.read_csv(snapshot.path)
    standardised = standardize_columns(raw)
    return aggregate_points(standardised, aggregation_strategy)


def summarise_gradient(
    consolidated: pd.DataFrame,
    *,
    time_s: float,
    power_W: int,
    aggregation_strategy: str,
) -> list[dict[str, object]]:
    """Summarise raw exported |grad T| without WLS, kNN, or Q reconstruction."""
    rows: list[dict[str, object]] = []
    for region_id, region_label, fof_limit in REGIONS:
        region = consolidated if fof_limit is None else consolidated.loc[consolidated["fof"] < fof_limit]
        values = pd.to_numeric(region["gradT"], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and np.any(finite < 0):
            raise ValueError(
                f"Negative exported temperature-gradient magnitude at t={time_s:.2f}, P={power_W}."
            )
        quantiles = np.quantile(finite, [0.25, 0.50, 0.75, 0.90]) if finite.size else np.full(4, np.nan)
        rows.append(
            {
                "time_s": time_s,
                "power_W": power_W,
                "aggregation_strategy": aggregation_strategy,
                "region": region_id,
                "region_label": region_label,
                "n_total": int(values.size),
                "n_finite": int(finite.size),
                "finite_fraction": float(finite.size / values.size) if values.size else np.nan,
                "support_gate_points": SUPPORT_GATE,
                "support_eligible": bool(finite.size >= SUPPORT_GATE),
                "support_status": "eligible" if finite.size >= SUPPORT_GATE else "audit_context_only",
                "gradT_min_K_per_m": float(np.min(finite)) if finite.size else np.nan,
                "gradT_p25_K_per_m": float(quantiles[0]),
                "gradT_median_K_per_m": float(quantiles[1]),
                "gradT_p75_K_per_m": float(quantiles[2]),
                "gradT_p90_K_per_m": float(quantiles[3]),
                "gradT_max_K_per_m": float(np.max(finite)) if finite.size else np.nan,
            }
        )
    return rows


def attach_discrete_extrema(metrics: pd.DataFrame) -> pd.DataFrame:
    """Label discrete local extrema only at internal sampled powers."""
    result = metrics.copy()
    result["sampled_power_extremum"] = "not_evaluated"
    grouping = ["time_s", "aggregation_strategy", "region"]
    for _, indices in result.groupby(grouping, sort=False).groups.items():
        ordered = result.loc[indices].sort_values("power_W")
        values = ordered["gradT_median_K_per_m"].to_numpy(dtype=float)
        statuses: list[str] = []
        for position, value in enumerate(values):
            if position in {0, len(values) - 1}:
                statuses.append("endpoint_not_tested")
            elif value > values[position - 1] and value > values[position + 1]:
                statuses.append("discrete_local_maximum")
            elif value < values[position - 1] and value < values[position + 1]:
                statuses.append("discrete_local_minimum")
            else:
                statuses.append("not_an_extremum")
        result.loc[ordered.index, "sampled_power_extremum"] = statuses
    return result


def compute_canonical_metrics(snapshots: list[SnapshotFile]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for snapshot in snapshots:
        consolidated = consolidate_snapshot(snapshot, CANONICAL_AGGREGATION)
        rows.extend(
            summarise_gradient(
                consolidated,
                time_s=snapshot.time_s,
                power_W=snapshot.power_W,
                aggregation_strategy=CANONICAL_AGGREGATION,
            )
        )
    metrics = pd.DataFrame(rows).sort_values(["time_s", "power_W", "region"]).reset_index(drop=True)
    return attach_discrete_extrema(metrics)
