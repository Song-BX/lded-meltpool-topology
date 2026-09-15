from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.regions import region_mask
from scripts.robustness.knn_scan import _pooled_thresholds, _region_q

from .config import (
    ALPHA_SPECS,
    CORE_POWER_HIGH,
    CORE_POWER_LOW,
    EXPECTED_POWERS,
    FOF_INTERFACE_THRESHOLD,
    K_VALUES,
    REGIONS,
    THRESHOLDS,
)


def _direction(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value > 0:
        return "350>400"
    if value < 0:
        return "350<400"
    return "tie"


def _thresholds_or_nan(by_power: dict[int, pd.DataFrame], region: str) -> dict[str, float]:
    try:
        return _pooled_thresholds(by_power, region)
    except ValueError:
        return {threshold: np.nan for threshold in THRESHOLDS}


def summarize_q_metrics(
    reconstructed: dict[tuple[str, int, int], pd.DataFrame],
) -> pd.DataFrame:
    """Compute alpha-specific pooled thresholds and all auditable regional Q metrics."""
    rows: list[dict[str, float | int | str]] = []
    for alpha in ALPHA_SPECS:
        for k in K_VALUES:
            by_power = {
                power: reconstructed[(alpha.label, power, k)]
                for power in EXPECTED_POWERS
            }
            pooled = {region: _thresholds_or_nan(by_power, region) for region in REGIONS}
            for power, frame in by_power.items():
                valid_points = int((frame["chi"] == 1).sum())
                for region in REGIONS:
                    q_values = _region_q(frame, region)
                    for threshold in THRESHOLDS:
                        threshold_value = pooled[region][threshold]
                        fraction = (
                            float((q_values > threshold_value).mean())
                            if len(q_values) and np.isfinite(threshold_value)
                            else np.nan
                        )
                        rows.append(
                            {
                                "alpha_label": alpha.label,
                                "alpha": alpha.value,
                                "alpha_role": alpha.role,
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
        ["alpha", "kNN", "region", "power_W", "threshold"]
    ).reset_index(drop=True)


def build_core_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute every alpha-specific 350 W minus 400 W regional contrast."""
    rows: list[dict[str, float | int | str]] = []
    group_columns = ("alpha_label", "alpha", "alpha_role", "kNN", "region", "threshold")
    for keys, block in metrics.groupby(list(group_columns), sort=True):
        by_power = block.set_index("power_W")
        low = float(by_power.loc[CORE_POWER_LOW, "q_fraction"])
        high = float(by_power.loc[CORE_POWER_HIGH, "q_fraction"])
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "phi_350": low,
                "phi_400": high,
                "delta_350_400": low - high,
                "direction": _direction(low - high),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["alpha", "kNN", "region", "threshold"]
    ).reset_index(drop=True)


def common_support_q_metrics(
    reconstructed: dict[tuple[str, int, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat full-pool Q>0 contrasts on points valid under all three alphas."""
    rows: list[dict[str, float | int | str]] = []
    for k in K_VALUES:
        for power in EXPECTED_POWERS:
            frames = {
                alpha.label: reconstructed[(alpha.label, power, k)] for alpha in ALPHA_SPECS
            }
            shared = np.logical_and.reduce(
                [frame["chi"].to_numpy(dtype=int) == 1 for frame in frames.values()]
            )
            for alpha in ALPHA_SPECS:
                values = frames[alpha.label].loc[shared, "Q"].dropna()
                rows.append(
                    {
                        "alpha_label": alpha.label,
                        "alpha": alpha.value,
                        "alpha_role": alpha.role,
                        "kNN": k,
                        "power_W": power,
                        "common_valid_points": int(len(values)),
                        "common_valid_fraction": float(len(values) / len(frames[alpha.label])),
                        "q_positive_fraction": float((values > 0).mean()) if len(values) else np.nan,
                    }
                )
    metrics = pd.DataFrame(rows).sort_values(["alpha", "kNN", "power_W"]).reset_index(drop=True)
    contrasts: list[dict[str, float | int | str]] = []
    for (alpha_label, alpha, alpha_role, k), block in metrics.groupby(
        ["alpha_label", "alpha", "alpha_role", "kNN"], sort=True
    ):
        by_power = block.set_index("power_W")
        low = float(by_power.loc[CORE_POWER_LOW, "q_positive_fraction"])
        high = float(by_power.loc[CORE_POWER_HIGH, "q_positive_fraction"])
        contrasts.append(
            {
                "alpha_label": alpha_label,
                "alpha": alpha,
                "alpha_role": alpha_role,
                "kNN": k,
                "phi_350": low,
                "phi_400": high,
                "delta_350_400": low - high,
                "direction": _direction(low - high),
                "minimum_common_valid_points": int(block["common_valid_points"].min()),
            }
        )
    return metrics, pd.DataFrame(contrasts).sort_values(["alpha", "kNN"]).reset_index(drop=True)

