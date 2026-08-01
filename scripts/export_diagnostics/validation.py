from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    AGGREGATION_STRATEGIES,
    EXPECTED_POWERS,
    EXPECTED_TIMES,
    K_VALUES,
    PHYSICAL_COLUMNS,
    REGIONS,
    THRESHOLDS,
)
from .discovery import SnapshotInput


def validate_complete_grid(records: list[SnapshotInput]) -> None:
    observed = {(record.time_s, record.power_W) for record in records}
    expected = {(time_s, power_W) for time_s in EXPECTED_TIMES for power_W in EXPECTED_POWERS}
    if observed != expected:
        raise AssertionError(f"Incomplete time-power grid: missing={sorted(expected - observed)}")


def validate_group_accounting(summary: pd.DataFrame) -> None:
    accounted = (
        summary["unique_coordinate_representatives"]
        + summary["additional_distinct_state_rows"]
        + summary["exact_repeated_rows"]
    )
    if not bool((accounted == summary["raw_points"]).all()):
        raise AssertionError("Export row categories do not sum to raw point counts")


def validate_spot_checks(
    checks: pd.DataFrame, frames: dict[tuple[float, int], pd.DataFrame]
) -> None:
    for _, check in checks.iterrows():
        frame = frames[(float(check["time_s"]), int(check["power_W"]))]
        mask = (
            (frame["x"] == check["x"])
            & (frame["y"] == check["y"])
            & (frame["z"] == check["z"])
        )
        group = frame.loc[mask]
        if len(group) != int(check["multiplicity"]):
            raise AssertionError("Spot-check multiplicity does not match the source CSV")
        states = len(group[list(PHYSICAL_COLUMNS)].drop_duplicates())
        if states != int(check["unique_physical_states"]):
            raise AssertionError("Spot-check physical-state count does not match the source CSV")
        variable = check["variable"]
        if variable != "all":
            observed_range = float(group[variable].max() - group[variable].min())
            if not np.isclose(observed_range, float(check["observed_range"]), rtol=0, atol=1e-15):
                raise AssertionError("Spot-check variable range does not match the source CSV")


def validate_sensitivity(outputs: dict[str, pd.DataFrame]) -> None:
    expected_k25 = len(AGGREGATION_STRATEGIES) * len(EXPECTED_POWERS) * len(REGIONS)
    expected_power = (
        len(AGGREGATION_STRATEGIES)
        * len(K_VALUES)
        * len(EXPECTED_POWERS)
        * len(REGIONS)
        * len(THRESHOLDS)
    )
    expected_contrasts = (
        len(AGGREGATION_STRATEGIES) * len(K_VALUES) * len(REGIONS) * len(THRESHOLDS)
    )
    if len(outputs["k25_metrics"]) != expected_k25:
        raise AssertionError("Incomplete k=25 aggregation-strategy grid")
    if len(outputs["knn_power_metrics"]) != expected_power:
        raise AssertionError("Incomplete power-k-region-threshold aggregation grid")
    if len(outputs["knn_contrasts"]) != expected_contrasts:
        raise AssertionError("Incomplete core-contrast aggregation grid")
    reproducibility = outputs["baseline_reproducibility"]
    if reproducibility.empty or not bool(reproducibility["passed"].all()):
        failed = reproducibility.loc[~reproducibility["passed"]]
        raise RuntimeError(
            "Canonical baseline reproduction failed; no export-diagnostic outputs were written.\n"
            + failed.to_string(index=False)
        )

