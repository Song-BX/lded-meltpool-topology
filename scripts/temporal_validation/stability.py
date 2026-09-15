from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import EXPECTED_POWERS, PLATEAU_TIMES, STABILITY_RULES


def _plateau_values(metrics: pd.DataFrame, power: int, metric: str) -> np.ndarray:
    subset = metrics[
        (metrics["power_W"] == power) & metrics["time_s"].isin(PLATEAU_TIMES)
    ].sort_values("time_s")
    values = subset[metric].to_numpy(dtype=float)
    if len(values) != len(PLATEAU_TIMES) or not np.isfinite(values).all():
        raise ValueError(f"Incomplete/non-finite plateau values for {power} W, {metric}")
    return values


def evaluate_stability(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for power in EXPECTED_POWERS:
        for rule in STABILITY_RULES:
            values = _plateau_values(metrics, power, rule.metric)
            median = float(np.median(values))
            if rule.method == "relative_max_deviation":
                if np.isclose(median, 0.0):
                    observed = np.inf
                else:
                    observed = float(np.max(np.abs(values - median)) / abs(median))
            elif rule.method == "absolute_range":
                observed = float(np.max(values) - np.min(values))
            else:
                raise ValueError(f"Unknown stability rule method: {rule.method}")
            rows.append(
                {
                    "power_W": power,
                    "metric": rule.metric,
                    "metric_label": rule.label,
                    "method": rule.method,
                    "threshold": rule.threshold,
                    "observed": observed,
                    "stability_ratio": observed / rule.threshold,
                    "plateau_median": median,
                    "plateau_min": float(np.min(values)),
                    "plateau_max": float(np.max(values)),
                    "plateau_values": json.dumps(values.tolist()),
                    "passed": bool(observed <= rule.threshold),
                }
            )
    detail = pd.DataFrame(rows)
    power_summary = (
        detail.groupby("power_W", as_index=False)
        .agg(passed_metrics=("passed", "sum"), total_metrics=("passed", "size"))
        .sort_values("power_W")
    )
    power_summary["all_metrics_passed"] = (
        power_summary["passed_metrics"] == power_summary["total_metrics"]
    )
    return detail, power_summary


def evaluate_core_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("temperature_max_all_K", "400 W > 350 W", 400, 350),
        ("velocity_max_all_mps", "350 W > 400 W", 350, 400),
        (
            "velocity_mean_interface_mps",
            "350 W > 400 W",
            350,
            400,
        ),
        ("q_positive_fraction_all", "350 W > 400 W", 350, 400),
        ("q_positive_fraction_interface", "350 W > 400 W", 350, 400),
    )
    rows: list[dict[str, object]] = []
    for time_s in PLATEAU_TIMES:
        at_time = metrics[metrics["time_s"] == time_s].set_index("power_W")
        for metric, expectation, higher_power, lower_power in comparisons:
            higher_value = float(at_time.loc[higher_power, metric])
            lower_value = float(at_time.loc[lower_power, metric])
            rows.append(
                {
                    "time_s": time_s,
                    "metric": metric,
                    "expectation": expectation,
                    "higher_power_W": higher_power,
                    "higher_value": higher_value,
                    "lower_power_W": lower_power,
                    "lower_value": lower_value,
                    "difference": higher_value - lower_value,
                    "passed": bool(higher_value > lower_value),
                }
            )
    return pd.DataFrame(rows)


def build_decision(power_summary: pd.DataFrame, contrasts: pd.DataFrame) -> dict[str, object]:
    all_power_pass = bool(power_summary["all_metrics_passed"].all())
    any_power_pass = bool(power_summary["all_metrics_passed"].any())
    core_pass = bool(contrasts["passed"].all())
    if all_power_pass and core_pass:
        status = "temporally_assessed_quasi_steady"
    elif core_pass and any_power_pass:
        status = "partial_temporal_stability"
    else:
        status = "late_time_snapshot"
    return {
        "status": status,
        "all_six_powers_pass": all_power_pass,
        "core_350_400_contrasts_pass": core_pass,
        "passing_powers_W": power_summary.loc[
            power_summary["all_metrics_passed"], "power_W"
        ].astype(int).tolist(),
        "failing_powers_W": power_summary.loc[
            ~power_summary["all_metrics_passed"], "power_W"
        ].astype(int).tolist(),
        "plateau_times_s": list(PLATEAU_TIMES),
    }
