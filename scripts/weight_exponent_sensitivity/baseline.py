from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BASELINE_METRICS


def compare_canonical_baseline(metrics: pd.DataFrame) -> pd.DataFrame:
    """Fail closed unless the alpha=0 branch reproduces the retained Q table."""
    expected = pd.read_csv(BASELINE_METRICS)
    actual = metrics[metrics["alpha"] == 0.0].copy()
    keys = ["kNN", "power_W", "region", "threshold"]
    merged = actual.merge(expected, on=keys, suffixes=("_actual", "_expected"), validate="one_to_one")
    rows: list[dict[str, float | int | str | bool]] = []
    for item in merged.itertuples(index=False):
        for metric, actual_value, expected_value, tolerance in (
            ("q_fraction", item.q_fraction_actual, item.q_fraction_expected, 1.0e-12),
            ("n_region", item.n_region_actual, item.n_region_expected, 0.0),
            ("wls_valid_points", item.wls_valid_points_actual, item.wls_valid_points_expected, 0.0),
            ("wls_valid_fraction", item.wls_valid_fraction_actual, item.wls_valid_fraction_expected, 1.0e-12),
        ):
            rows.append(
                {
                    "check": "canonical_alpha_0",
                    "kNN": item.kNN,
                    "power_W": item.power_W,
                    "region": item.region,
                    "threshold": item.threshold,
                    "metric": metric,
                    "actual": actual_value,
                    "expected": expected_value,
                    "absolute_difference": abs(float(actual_value) - float(expected_value)),
                    "passed": bool(np.isclose(actual_value, expected_value, rtol=0.0, atol=tolerance)),
                }
            )
    return pd.DataFrame(rows)

