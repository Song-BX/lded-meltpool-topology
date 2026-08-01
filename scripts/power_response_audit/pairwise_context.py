from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .config import CANONICAL_TIME_S, METRICS, POWERS


_REQUIRED_COLUMNS = {
    "time_s",
    "power_W",
    "metric_id",
    "metric_label",
    "unit",
    "region",
    "value",
}


def _validate_canonical_metric_frame(canonical: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS - set(canonical.columns))
    if missing:
        raise ValueError(f"Canonical metric frame lacks required columns: {missing}")
    expected_rows = len(METRICS) * len(POWERS)
    if len(canonical) != expected_rows:
        raise ValueError(f"Expected {expected_rows} canonical metric rows, found {len(canonical)}")
    if not np.allclose(canonical["time_s"].to_numpy(dtype=float), CANONICAL_TIME_S):
        raise ValueError("Pairwise context requires only the canonical 0.70 s snapshot")

    expected_metric_ids = {metric.metric_id for metric in METRICS}
    if set(canonical["metric_id"]) != expected_metric_ids:
        raise ValueError("Canonical metric frame does not contain the configured descriptor set")
    for metric in METRICS:
        block = canonical.loc[canonical["metric_id"] == metric.metric_id]
        observed_powers = tuple(sorted(pd.to_numeric(block["power_W"], errors="raise").astype(int)))
        if len(block) != len(POWERS) or observed_powers != POWERS:
            raise ValueError(
                f"{metric.metric_id}: expected one value at every configured sampled power"
            )


def _direction(lower_value: float, higher_value: float) -> str:
    if np.isclose(higher_value, lower_value, rtol=0.0, atol=1e-12):
        return "tie"
    return "higher_power_greater" if higher_value > lower_value else "lower_power_greater"


def build_pairwise_snapshot_context(canonical: pd.DataFrame) -> pd.DataFrame:
    """Build a descriptive ledger for every unordered sampled-power pair.

    The ledger contains no inferential quantities and makes no assertion beyond
    the six observed 0.70 s solver-export descriptors.
    """

    _validate_canonical_metric_frame(canonical)
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        block = canonical.loc[canonical["metric_id"] == metric.metric_id].copy()
        value_by_power = block.set_index("power_W")["value"].astype(float)
        for lower_power, higher_power in combinations(POWERS, 2):
            lower_value = float(value_by_power.loc[lower_power])
            higher_value = float(value_by_power.loc[higher_power])
            rows.append(
                {
                    "snapshot_time_s": CANONICAL_TIME_S,
                    "lower_power_W": lower_power,
                    "higher_power_W": higher_power,
                    "metric_id": metric.metric_id,
                    "metric_label": metric.label,
                    "unit": metric.unit,
                    "region": metric.region,
                    "lower_value": lower_value,
                    "higher_value": higher_value,
                    "delta_higher_minus_lower": higher_value - lower_value,
                    "direction": _direction(lower_value, higher_value),
                    "interpretation_status": metric.interpretation_status,
                    "interpretation_boundary": metric.interpretation_boundary,
                }
            )

    context = pd.DataFrame(rows).sort_values(
        ["metric_id", "lower_power_W", "higher_power_W"], ignore_index=True
    )
    expected_rows = len(METRICS) * len(POWERS) * (len(POWERS) - 1) // 2
    if len(context) != expected_rows:
        raise ValueError(f"Expected {expected_rows} pairwise rows, found {len(context)}")
    if context.duplicated(["metric_id", "lower_power_W", "higher_power_W"]).any():
        raise ValueError("Pairwise context contains duplicate metric-pair rows")
    if context[["lower_power_W", "higher_power_W"]].drop_duplicates().shape[0] != 15:
        raise ValueError("Pairwise context does not contain every unordered sampled-power pair")
    return context
