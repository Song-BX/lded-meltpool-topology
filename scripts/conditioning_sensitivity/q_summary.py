from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.robustness.knn_scan import _pooled_thresholds, _region_q

from .config import (
    CORE_POWER_HIGH,
    CORE_POWER_LOW,
    CUTOFF_SPECS,
    FOF_INTERFACE_THRESHOLD,
    HEAT_FLUX_THRESHOLD,
    K_VALUES,
    REGIONS,
    THRESHOLDS,
)


def apply_cutoff(frame: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Apply a condition-number screening mask without recomputing gradients."""
    output = frame.copy()
    kappa = output["kappa"].to_numpy(dtype=float)
    accepted = np.isfinite(kappa)
    if not math.isinf(cutoff):
        accepted &= kappa <= cutoff
    output["chi"] = accepted.astype(int)
    return output


def _thresholds_or_nan(reconstructed: dict[int, pd.DataFrame], region: str) -> dict[str, float]:
    try:
        return _pooled_thresholds(reconstructed, region)
    except ValueError:
        return {threshold: np.nan for threshold in THRESHOLDS}


def summarize_q_metrics(
    reconstructed: dict[tuple[int, int], pd.DataFrame],
) -> pd.DataFrame:
    """Reapply pooled Q thresholds after each pre-specified condition screen."""
    rows: list[dict[str, float | int | str]] = []
    for spec in CUTOFF_SPECS:
        for k in K_VALUES:
            by_power = {
                power: apply_cutoff(reconstructed[(power, k)], spec.value)
                for power in sorted({power for power, current_k in reconstructed if current_k == k})
            }
            thresholds = {
                region: _thresholds_or_nan(by_power, region) for region in REGIONS
            }
            for power, frame in sorted(by_power.items()):
                valid_points = int((frame["chi"] == 1).sum())
                for region in REGIONS:
                    q_values = _region_q(frame, region)
                    for threshold in THRESHOLDS:
                        threshold_value = thresholds[region][threshold]
                        fraction = (
                            float((q_values > threshold_value).mean())
                            if len(q_values) and np.isfinite(threshold_value)
                            else np.nan
                        )
                        rows.append(
                            {
                                "cutoff_label": spec.label,
                                "cutoff_value": spec.value,
                                "kNN": k,
                                "power_W": power,
                                "region": region,
                                "threshold": threshold,
                                "threshold_value": threshold_value,
                                "q_fraction": fraction,
                                "n_region": int(len(q_values)),
                                "wls_valid_points": valid_points,
                                "wls_valid_fraction": float(valid_points / len(frame)),
                            }
                        )
    return pd.DataFrame(rows).sort_values(
        ["cutoff_value", "kNN", "region", "power_W", "threshold"]
    ).reset_index(drop=True)


def build_core_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute the 350 W--400 W contrast for all auditable Q summaries."""
    rows: list[dict[str, float | int | str]] = []
    group_columns = ("cutoff_label", "cutoff_value", "kNN", "region", "threshold")
    for keys, block in metrics.groupby(list(group_columns), sort=True):
        by_power = block.set_index("power_W")
        low = float(by_power.loc[CORE_POWER_LOW, "q_fraction"])
        high = float(by_power.loc[CORE_POWER_HIGH, "q_fraction"])
        if not np.isfinite(low) or not np.isfinite(high):
            direction = "missing"
        elif low > high:
            direction = "350>400"
        elif low < high:
            direction = "350<400"
        else:
            direction = "tie"
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "phi_350": low,
                "phi_400": high,
                "delta_350_400": low - high,
                "direction": direction,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["cutoff_value", "kNN", "region", "threshold"]
    ).reset_index(drop=True)
