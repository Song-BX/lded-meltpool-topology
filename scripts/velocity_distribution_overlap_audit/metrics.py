from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from scripts.velocity_extreme_audit.velocity import consolidate, load_standardised

from .config import (
    CANONICAL_STRATEGY,
    CANONICAL_TIME_S,
    PAIR_POWERS,
    QUANTILE_LEVELS,
    TEMPORAL_CONTEXT_TIMES,
)


def _finite_velocity(frame: pd.DataFrame) -> np.ndarray:
    values = frame["V"].to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("A distribution-audit velocity vector is empty or non-finite.")
    return values


def _quantiles(values: np.ndarray) -> dict[str, float]:
    result = {
        f"velocity_p{int(level * 100):02d}_mps": float(np.quantile(values, level))
        for level in QUANTILE_LEVELS
    }
    result["velocity_max_mps"] = float(np.max(values))
    return result


def calculate_pair_metrics(values_350: Iterable[float], values_400: Iterable[float]) -> dict[str, object]:
    """Return descriptive central-range and tail context without inferential tests."""
    first = np.asarray(list(values_350), dtype=float)
    second = np.asarray(list(values_400), dtype=float)
    if len(first) == 0 or len(second) == 0 or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Both pairwise velocity distributions must be finite and non-empty.")

    first_q = _quantiles(first)
    second_q = _quantiles(second)
    lower = max(first_q["velocity_p25_mps"], second_q["velocity_p25_mps"])
    upper = min(first_q["velocity_p75_mps"], second_q["velocity_p75_mps"])
    overlap_width = max(0.0, upper - lower)
    first_contains_second = bool(
        first_q["velocity_p25_mps"] <= second_q["velocity_p25_mps"]
        and first_q["velocity_p75_mps"] >= second_q["velocity_p75_mps"]
    )
    second_contains_first = bool(
        second_q["velocity_p25_mps"] <= first_q["velocity_p25_mps"]
        and second_q["velocity_p75_mps"] >= first_q["velocity_p75_mps"]
    )
    contained = "400_within_350" if first_contains_second else (
        "350_within_400" if second_contains_first else "neither"
    )

    row: dict[str, object] = {
        "n_unique_points_350": int(len(first)),
        "n_unique_points_400": int(len(second)),
        "iqr_width_350_mps": first_q["velocity_p75_mps"] - first_q["velocity_p25_mps"],
        "iqr_width_400_mps": second_q["velocity_p75_mps"] - second_q["velocity_p25_mps"],
        "iqr_overlap_lower_mps": lower,
        "iqr_overlap_upper_mps": upper,
        "iqr_overlap_width_mps": overlap_width,
        "iqr_overlap_observed": bool(overlap_width > 0.0),
        "one_iqr_contained_in_other": bool(first_contains_second or second_contains_first),
        "contained_iqr": contained,
    }
    for suffix, quantile in (("p25", "velocity_p25_mps"), ("p50", "velocity_p50_mps"), ("p75", "velocity_p75_mps"), ("p90", "velocity_p90_mps"), ("p95", "velocity_p95_mps"), ("p99", "velocity_p99_mps"), ("max", "velocity_max_mps")):
        value_350 = first_q[quantile]
        value_400 = second_q[quantile]
        row[f"{suffix}_350_mps"] = value_350
        row[f"{suffix}_400_mps"] = value_400
        row[f"delta_{suffix}_350_minus_400_mps"] = value_350 - value_400

    for label, cutoff_350, cutoff_400 in (
        ("p99", first_q["velocity_p99_mps"], second_q["velocity_p99_mps"]),
        ("vmax", first_q["velocity_max_mps"], second_q["velocity_max_mps"]),
    ):
        count_350 = int((first > cutoff_400).sum())
        count_400 = int((second > cutoff_350).sum())
        row[f"n_350_gt_400_{label}"] = count_350
        row[f"prop_350_gt_400_{label}"] = count_350 / len(first)
        row[f"n_400_gt_350_{label}"] = count_400
        row[f"prop_400_gt_350_{label}"] = count_400 / len(second)
    return row


def build_audit(snapshots: list[object]) -> pd.DataFrame:
    """Calculate four aggregation and four earlier-time descriptive records."""
    raw_cache: dict[tuple[float, int], pd.DataFrame] = {}
    for snapshot in snapshots:
        raw_cache[(float(snapshot.time_s), int(snapshot.power_W))] = load_standardised(snapshot)

    rows: list[dict[str, object]] = []
    specifications = [
        ("aggregation_sensitivity", CANONICAL_TIME_S, strategy)
        for strategy in ("mean_all_records", "median_all_records", "first_record", "mean_distinct_states")
    ]
    specifications.extend(
        ("temporal_context", time_s, CANONICAL_STRATEGY) for time_s in TEMPORAL_CONTEXT_TIMES
    )
    for context, time_s, strategy in specifications:
        values: dict[int, np.ndarray] = {}
        for power_W in PAIR_POWERS:
            raw = raw_cache[(time_s, power_W)]
            values[power_W] = _finite_velocity(consolidate(raw, strategy))
        row = {
            "audit_context": context,
            "time_s": time_s,
            "aggregation_strategy": strategy,
        }
        row.update(calculate_pair_metrics(values[350], values[400]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["audit_context", "time_s", "aggregation_strategy"], kind="stable"
    ).reset_index(drop=True)
