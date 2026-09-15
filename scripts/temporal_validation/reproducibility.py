from __future__ import annotations

import numpy as np
import pandas as pd


COMPARISONS = (
    ("temperature_mean_all_K", "all", "T_mean_K"),
    ("temperature_max_all_K", "all", "T_max_K"),
    ("velocity_max_all_mps", "all", "v_max"),
    ("velocity_mean_interface_mps", "interface", "v_mean"),
    ("q_positive_fraction_all", "all", "Q_pos_frac"),
    ("q_positive_fraction_interface", "interface", "Q_pos_frac"),
)


def compare_baseline(metrics: pd.DataFrame, canonical_csv: pd.DataFrame) -> pd.DataFrame:
    at_baseline = metrics[np.isclose(metrics["time_s"], 0.70)].set_index("power_W")
    rows: list[dict[str, object]] = []
    for metric, region, canonical_column in COMPARISONS:
        reference = canonical_csv[canonical_csv["region"] == region].set_index("power_W")
        for power in sorted(at_baseline.index):
            actual = float(at_baseline.loc[power, metric])
            expected = float(reference.loc[power, canonical_column])
            absolute_difference = abs(actual - expected)
            relative_difference = absolute_difference / max(abs(expected), 1e-15)
            rows.append(
                {
                    "power_W": int(power),
                    "metric": metric,
                    "canonical_region": region,
                    "actual": actual,
                    "canonical": expected,
                    "absolute_difference": absolute_difference,
                    "relative_difference": relative_difference,
                    "passed": bool(np.isclose(actual, expected, rtol=1e-6, atol=1e-10)),
                }
            )
    return pd.DataFrame(rows)
