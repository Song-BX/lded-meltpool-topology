from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.analysis.wls_q import nearest_neighbor_indices, reconstruct_case, reconstruct_case_from_neighbor_sets
from scripts.export_diagnostics.aggregation import aggregate_points
from scripts.gradient_validation.quadratic import reconstruct_quadratic_case

from .config import K_REFERENCE, K_VALUES, WLS_CONDITION_CUTOFF, WLS_CONDITION_MODE, WLS_DISTANCE_OFFSET_M


def reconstruct_grid(
    cases: dict[int, pd.DataFrame],
    *,
    alpha: float = 0.0,
    kappa_max: float = WLS_CONDITION_CUTOFF,
    k_values: tuple[int, ...] = K_VALUES,
) -> dict[tuple[int, int], pd.DataFrame]:
    """Run the shared first-order WLS implementation over a fixed power-k grid."""
    return {
        (power, k): reconstruct_case(
            frame,
            k=k,
            alpha=alpha,
            eps_w=WLS_DISTANCE_OFFSET_M,
            kappa_max=kappa_max,
            condition_on=WLS_CONDITION_MODE,
        )
        for power, frame in sorted(cases.items())
        for k in k_values
    }


def screen_condition_number(frame: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Apply a condition-number inclusion screen without recomputing gradients."""
    screened = frame.copy()
    valid = np.isfinite(screened["kappa"].to_numpy(dtype=float))
    if math.isfinite(cutoff):
        valid &= screened["kappa"].to_numpy(dtype=float) <= cutoff
    screened["chi"] = valid.astype(int)
    return screened


def reconstruct_aggregation_grid(
    raw_cases: dict[int, pd.DataFrame], strategy: str
) -> dict[tuple[int, int], pd.DataFrame]:
    aggregated = {
        power: aggregate_points(frame, strategy)
        for power, frame in sorted(raw_cases.items())
    }
    return reconstruct_grid(aggregated, k_values=(K_REFERENCE,))


def reconstruct_quadratic_grid(
    cases: dict[int, pd.DataFrame], k_values: tuple[int, ...]
) -> dict[tuple[int, int], pd.DataFrame]:
    """Use the existing scaled second-order comparator on the requested k grid."""
    return {
        (power, k): reconstruct_quadratic_case(
            frame, k=k, kappa_max=WLS_CONDITION_CUTOFF
        )
        for power, frame in sorted(cases.items())
        for k in k_values
    }


def fixed_neighbour_sets(cases: dict[int, pd.DataFrame]) -> dict[int, np.ndarray]:
    return {
        power: nearest_neighbor_indices(frame[["x", "y", "z"]].to_numpy(dtype=float), k=K_REFERENCE)[:, 1:]
        for power, frame in sorted(cases.items())
    }


def sample_neighbour_sets(base_sets: np.ndarray, generator: np.random.Generator, subset_size: int) -> np.ndarray:
    return np.vstack([generator.choice(row, size=subset_size, replace=False) for row in base_sets])


def reconstruct_from_sets(
    frame: pd.DataFrame, neighbour_sets: np.ndarray, *, alpha: float
) -> pd.DataFrame:
    return reconstruct_case_from_neighbor_sets(
        frame,
        neighbour_sets,
        k=K_REFERENCE,
        alpha=alpha,
        eps_w=WLS_DISTANCE_OFFSET_M,
        kappa_max=WLS_CONDITION_CUTOFF,
        condition_on=WLS_CONDITION_MODE,
    )
