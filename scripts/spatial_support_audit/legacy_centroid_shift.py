"""Archive, but do not reinterpret, the withdrawn historical centroid shift."""

from __future__ import annotations

from math import hypot

import pandas as pd

from .legacy_shift_config import (
    COMPARISON_POWER_W,
    HISTORICAL_METRIC,
    NOMINAL_GRID_SPACING_MM,
    REFERENCE_POWER_W,
)


REQUIRED_LEGACY_SHIFT_COLUMNS = {
    "metric",
    "power_W",
    "centroid_x_mm",
    "centroid_z_mm",
}


def _historical_row(legacy: pd.DataFrame, power_w: int) -> pd.Series:
    selected = legacy.loc[
        (legacy["metric"] == HISTORICAL_METRIC) & (legacy["power_W"] == power_w)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one {HISTORICAL_METRIC} row for {power_w} W, found {len(selected)}."
        )
    return selected.iloc[0]


def calculate_legacy_centroid_shift_context(
    legacy: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate an archival shift from existing historical rows only.

    This routine deliberately does not rebuild Q, centroid, radius, or uncertainty.
    The nominal grid spacing is reported as a scale reference, never as a noise estimate.
    """
    missing_legacy = REQUIRED_LEGACY_SHIFT_COLUMNS - set(legacy.columns)
    if missing_legacy:
        raise ValueError(f"Legacy summary is missing columns: {sorted(missing_legacy)}")
    required_reconciliation = {"power_W", "legacy_metric", "reconciliation_status"}
    missing_reconciliation = required_reconciliation - set(reconciliation.columns)
    if missing_reconciliation:
        raise ValueError(
            f"Reconciliation table is missing columns: {sorted(missing_reconciliation)}"
        )

    reference = _historical_row(legacy, REFERENCE_POWER_W)
    comparison = _historical_row(legacy, COMPARISON_POWER_W)
    comparison_status = reconciliation.loc[
        (reconciliation["power_W"] == COMPARISON_POWER_W)
        & (reconciliation["legacy_metric"] == HISTORICAL_METRIC),
        "reconciliation_status",
    ]
    if len(comparison_status) != 1:
        raise ValueError(
            "Expected exactly one reconciliation status for the 400 W historical Qpos_top10 row."
        )
    if comparison_status.iloc[0] != "unreconciled_legacy_summary":
        raise ValueError("The 400 W historical Qpos_top10 row must remain unreconciled.")

    reference_x = round(float(reference["centroid_x_mm"]), 9)
    reference_z = round(float(reference["centroid_z_mm"]), 9)
    comparison_x = round(float(comparison["centroid_x_mm"]), 9)
    comparison_z = round(float(comparison["centroid_z_mm"]), 9)
    delta_x = round(comparison_x - reference_x, 9)
    delta_z = round(comparison_z - reference_z, 9)
    absolute_x = abs(delta_x)
    absolute_z = abs(delta_z)
    euclidean_shift = round(hypot(delta_x, delta_z), 9)

    return pd.DataFrame(
        [
            {
                "source_metric": HISTORICAL_METRIC,
                "reference_power_W": REFERENCE_POWER_W,
                "comparison_power_W": COMPARISON_POWER_W,
                "reference_centroid_x_mm": reference_x,
                "reference_centroid_z_mm": reference_z,
                "comparison_centroid_x_mm": comparison_x,
                "comparison_centroid_z_mm": comparison_z,
                "delta_x_mm": delta_x,
                "delta_z_mm": delta_z,
                "absolute_delta_x_mm": absolute_x,
                "absolute_delta_z_mm": absolute_z,
                "euclidean_shift_mm": euclidean_shift,
                "nominal_grid_spacing_mm": NOMINAL_GRID_SPACING_MM,
                "absolute_delta_x_nominal_grid_cells": round(
                    absolute_x / NOMINAL_GRID_SPACING_MM, 9
                ),
                "absolute_delta_z_nominal_grid_cells": round(
                    absolute_z / NOMINAL_GRID_SPACING_MM, 9
                ),
                "euclidean_shift_nominal_grid_cells": round(
                    euclidean_shift / NOMINAL_GRID_SPACING_MM, 9
                ),
                "comparison_reconciliation_status": comparison_status.iloc[0],
                "evidence_status": "archival_context_only",
                "interpretation": (
                    "not_distinguishable_from_reconstruction_noise_with_available_data; "
                    "nominal grid spacing is not a displacement uncertainty or reconstruction-noise estimate"
                ),
            }
        ]
    )
