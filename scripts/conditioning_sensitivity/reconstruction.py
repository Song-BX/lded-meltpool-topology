from __future__ import annotations

import math

import pandas as pd

from scripts.analysis.wls_q import reconstruct_case
from scripts.analysis.point_cloud import deduplicate_points, standardize_columns
from scripts.robustness.discovery import CaseInput

from .config import (
    COORDINATE_TOLERANCE_M,
    K_VALUES,
    WLS_CONDITION_MODE,
    WLS_DISTANCE_EXPONENT,
    WLS_DISTANCE_OFFSET_M,
)


def load_cases(records: list[CaseInput]) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Load the six exports and apply the canonical exact-coordinate consolidation."""
    cases: dict[int, pd.DataFrame] = {}
    rows: list[dict[str, int]] = []
    for record in records:
        raw = pd.read_csv(record.path)
        standardized = standardize_columns(raw)
        consolidated = deduplicate_points(
            standardized, eps_c=COORDINATE_TOLERANCE_M
        )
        cases[record.power_W] = consolidated
        rows.append(
            {
                "power_W": record.power_W,
                "raw_rows": len(raw),
                "unique_coordinate_rows": len(consolidated),
            }
        )
    return cases, pd.DataFrame(rows).sort_values("power_W")


def reconstruct_without_finite_cutoff(
    cases: dict[int, pd.DataFrame],
) -> dict[tuple[int, int], pd.DataFrame]:
    """Reconstruct all finite-condition points once for the cutoff audit."""
    reconstructed: dict[tuple[int, int], pd.DataFrame] = {}
    for k in K_VALUES:
        for power, frame in sorted(cases.items()):
            reconstructed[(power, k)] = reconstruct_case(
                frame,
                k=k,
                alpha=WLS_DISTANCE_EXPONENT,
                eps_w=WLS_DISTANCE_OFFSET_M,
                kappa_max=math.inf,
                condition_on=WLS_CONDITION_MODE,
            )
    return reconstructed
