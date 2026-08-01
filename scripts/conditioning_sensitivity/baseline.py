from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BASELINE_METRICS, WLS_CONDITION_CUTOFF


def compare_canonical_baseline(
    metrics: pd.DataFrame, point_audit: pd.DataFrame
) -> pd.DataFrame:
    """Require the Comment 6 kappa=100 branch to reproduce retained outputs."""
    baseline = pd.read_csv(BASELINE_METRICS)
    current = metrics[metrics["cutoff_value"] == WLS_CONDITION_CUTOFF].copy()
    merged = current.merge(
        baseline,
        on=["kNN", "power_W", "region", "threshold"],
        suffixes=("_actual", "_expected"),
        validate="one_to_one",
    )
    rows: list[dict[str, float | int | str | bool]] = []
    for row in merged.itertuples(index=False):
        for metric, actual, expected, tolerance in (
            ("q_fraction", row.q_fraction_actual, row.q_fraction_expected, 1.0e-12),
            ("n_region", row.n_region_actual, row.n_region_expected, 0.0),
        ):
            rows.append(
                {
                    "check": "canonical_kappa_100",
                    "kNN": row.kNN,
                    "power_W": row.power_W,
                    "region": row.region,
                    "threshold": row.threshold,
                    "metric": metric,
                    "actual": actual,
                    "expected": expected,
                    "absolute_difference": abs(float(actual) - float(expected)),
                    "passed": bool(np.isclose(actual, expected, rtol=0.0, atol=tolerance)),
                }
            )

    current_audit = point_audit[point_audit["cutoff_value"] == WLS_CONDITION_CUTOFF]
    expected_valid = baseline[["kNN", "power_W", "wls_valid_points", "wls_valid_fraction"]].drop_duplicates()
    merged_audit = current_audit.merge(
        expected_valid,
        on=["kNN", "power_W"],
        suffixes=("_actual", "_expected"),
        validate="one_to_one",
    )
    for row in merged_audit.itertuples(index=False):
        for metric, actual, expected, tolerance in (
            ("wls_valid_points", row.retained_points, row.wls_valid_points, 0.0),
            ("wls_valid_fraction", row.retained_fraction, row.wls_valid_fraction, 1.0e-12),
        ):
            rows.append(
                {
                    "check": "canonical_kappa_100",
                    "kNN": row.kNN,
                    "power_W": row.power_W,
                    "region": "all",
                    "threshold": "Q>0",
                    "metric": metric,
                    "actual": actual,
                    "expected": expected,
                    "absolute_difference": abs(float(actual) - float(expected)),
                    "passed": bool(np.isclose(actual, expected, rtol=0.0, atol=tolerance)),
                }
            )
    return pd.DataFrame(rows)
