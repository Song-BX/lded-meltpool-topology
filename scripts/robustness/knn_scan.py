from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.regions import region_mask
from scripts.analysis.wls_q import reconstruct_case

from .config import (
    CORE_POWER_HIGH,
    CORE_POWER_LOW,
    FOF_INTERFACE_THRESHOLD,
    HEAT_FLUX_THRESHOLD,
    K_VALUES,
    PERCENTILES,
    REGIONS,
    THRESHOLDS,
    WLS_CONDITION_CUTOFF,
    WLS_CONDITION_MODE,
    WLS_DISTANCE_EXPONENT,
    WLS_DISTANCE_OFFSET_M,
)


def _region_q(frame: pd.DataFrame, region: str) -> pd.Series:
    mask = region_mask(
        frame,
        region,
        fof_interface_threshold=FOF_INTERFACE_THRESHOLD,
        heat_flux_threshold=HEAT_FLUX_THRESHOLD,
    )
    return frame.loc[mask, "Q"].dropna()


def _pooled_thresholds(reconstructed: dict[int, pd.DataFrame], region: str) -> dict[str, float]:
    positive_parts = [
        values[values > 0]
        for values in (_region_q(frame, region) for frame in reconstructed.values())
    ]
    positive = pd.concat(positive_parts, ignore_index=True)
    if positive.empty:
        raise ValueError(f"No positive Q values for region={region}")
    thresholds = {"Q>0": 0.0}
    thresholds.update(
        {label: float(positive.quantile(percentile)) for label, percentile in PERCENTILES.items()}
    )
    return thresholds


def scan_cases(
    cases: dict[int, pd.DataFrame], *, verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    power_rows: list[dict[str, float | int | str]] = []
    contrast_rows: list[dict[str, float | int | str | bool]] = []

    for k in K_VALUES:
        reconstructed: dict[int, pd.DataFrame] = {}
        for power, frame in sorted(cases.items()):
            reconstructed[power] = reconstruct_case(
                frame,
                k=k,
                alpha=WLS_DISTANCE_EXPONENT,
                eps_w=WLS_DISTANCE_OFFSET_M,
                kappa_max=WLS_CONDITION_CUTOFF,
                condition_on=WLS_CONDITION_MODE,
            )

        thresholds_by_region = {
            region: _pooled_thresholds(reconstructed, region) for region in REGIONS
        }
        metric_lookup: dict[tuple[int, str, str], float] = {}
        for power, frame in sorted(reconstructed.items()):
            valid = frame["chi"] == 1
            finite_kappa = frame.loc[np.isfinite(frame["kappa"]), "kappa"]
            valid_kappa = frame.loc[valid, "kappa"]
            global_metrics = {
                "unique_points": int(len(frame)),
                "wls_valid_points": int(valid.sum()),
                "wls_valid_fraction": float(valid.mean()),
                "kappa_finite_median": float(finite_kappa.median()),
                "kappa_finite_p90": float(finite_kappa.quantile(0.90)),
                "kappa_valid_max": float(valid_kappa.max()),
            }
            for region in REGIONS:
                q_values = _region_q(frame, region)
                for threshold_label in THRESHOLDS:
                    threshold_value = thresholds_by_region[region][threshold_label]
                    fraction = float((q_values > threshold_value).mean()) if len(q_values) else np.nan
                    metric_lookup[(power, region, threshold_label)] = fraction
                    power_rows.append(
                        {
                            "kNN": k,
                            "power_W": power,
                            "region": region,
                            "threshold": threshold_label,
                            "threshold_value": threshold_value,
                            "q_fraction": fraction,
                            "n_region": int(len(q_values)),
                            **global_metrics,
                        }
                    )

        for region in REGIONS:
            for threshold_label in THRESHOLDS:
                phi_low = metric_lookup[(CORE_POWER_LOW, region, threshold_label)]
                phi_high = metric_lookup[(CORE_POWER_HIGH, region, threshold_label)]
                difference = phi_low - phi_high
                if np.isnan(difference):
                    direction = "missing"
                elif difference > 0:
                    direction = "350>400"
                elif difference < 0:
                    direction = "350<400"
                else:
                    direction = "tie"
                contrast_rows.append(
                    {
                        "kNN": k,
                        "region": region,
                        "threshold": threshold_label,
                        "threshold_value": thresholds_by_region[region][threshold_label],
                        "phi_350": phi_low,
                        "phi_400": phi_high,
                        "ratio_350_400": phi_low / phi_high if phi_high != 0 else np.inf,
                        "diff_350_400": difference,
                        "direction": direction,
                    }
                )
        if verbose:
            print(f"completed k={k}: six powers, four regions, four thresholds", flush=True)

    return (
        pd.DataFrame(power_rows).sort_values(["kNN", "region", "power_W", "threshold"]),
        pd.DataFrame(contrast_rows).sort_values(["kNN", "region", "threshold"]),
    )
