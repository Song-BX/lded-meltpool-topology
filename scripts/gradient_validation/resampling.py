from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.regions import region_mask
from scripts.analysis.wls_q import (
    nearest_neighbor_indices,
    reconstruct_case,
    reconstruct_case_from_neighbor_sets,
)

from .config import (
    FOF_INTERFACE_THRESHOLD,
    K_REFERENCE,
    REGIONS,
    RESAMPLE_COUNT,
    RESAMPLE_NEIGHBOURS,
    RESAMPLE_SEED,
    WLS_CONDITION_CUTOFF,
    WLS_CONDITION_MODE,
    WLS_DISTANCE_EXPONENT,
    WLS_DISTANCE_OFFSET_M,
)


def _direction(value: float) -> str:
    if value > 0:
        return "350>400"
    if value < 0:
        return "350<400"
    return "tie"


def _sample_neighbour_sets(
    base_sets: np.ndarray, generator: np.random.Generator
) -> np.ndarray:
    return np.vstack(
        [generator.choice(row, size=RESAMPLE_NEIGHBOURS, replace=False) for row in base_sets]
    )


def run_neighbour_resampling(
    cases: dict[int, pd.DataFrame], *, alpha: float = WLS_DISTANCE_EXPONENT
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perturb 20-of-25 nearest-neighbour memberships without claiming replication."""
    generator = np.random.default_rng(RESAMPLE_SEED)
    baseline: dict[int, pd.DataFrame] = {}
    neighbour_sets: dict[int, np.ndarray] = {}
    for power, frame in sorted(cases.items()):
        baseline[power] = reconstruct_case(
            frame,
            k=K_REFERENCE,
            alpha=alpha,
            eps_w=WLS_DISTANCE_OFFSET_M,
            kappa_max=WLS_CONDITION_CUTOFF,
            condition_on=WLS_CONDITION_MODE,
        )
        points = frame[["x", "y", "z"]].to_numpy(dtype=float)
        neighbour_sets[power] = nearest_neighbor_indices(points, k=K_REFERENCE)[:, 1:]

    rows: list[dict[str, object]] = []
    for replicate in range(1, RESAMPLE_COUNT + 1):
        reconstructed = {
            power: reconstruct_case_from_neighbor_sets(
                cases[power],
                _sample_neighbour_sets(neighbour_sets[power], generator),
                k=K_REFERENCE,
                alpha=alpha,
                eps_w=WLS_DISTANCE_OFFSET_M,
                kappa_max=WLS_CONDITION_CUTOFF,
                condition_on=WLS_CONDITION_MODE,
            )
            for power in sorted(cases)
        }
        for power, frame in reconstructed.items():
            for region in REGIONS:
                mask = region_mask(
                    frame, region, fof_interface_threshold=FOF_INTERFACE_THRESHOLD
                ).to_numpy(dtype=bool)
                baseline_mask = region_mask(
                    baseline[power], region, fof_interface_threshold=FOF_INTERFACE_THRESHOLD
                ).to_numpy(dtype=bool)
                shared = mask & baseline_mask
                sign_agreement = (
                    float(
                        (
                            (frame.loc[shared, "Q"].to_numpy() > 0)
                            == (baseline[power].loc[shared, "Q"].to_numpy() > 0)
                        ).mean()
                    )
                    if shared.any()
                    else np.nan
                )
                values = frame.loc[mask, "Q"].dropna()
                rows.append(
                    {
                        "alpha": alpha,
                        "replicate": replicate,
                        "power_W": power,
                        "region": region,
                        "subset_neighbours": RESAMPLE_NEIGHBOURS,
                        "parent_kNN": K_REFERENCE,
                        "valid_points": int(len(values)),
                        "q_positive_fraction": float((values > 0).mean()) if len(values) else np.nan,
                        "point_sign_agreement_with_k25": sign_agreement,
                    }
                )
    replicates = pd.DataFrame(rows)

    core = replicates.pivot_table(
        index=["alpha", "replicate", "region"], columns="power_W", values="q_positive_fraction", aggfunc="first"
    ).reset_index()
    core["delta_350_400"] = core[350] - core[400]
    core["direction_350_400"] = core["delta_350_400"].map(_direction)
    summary = (
        replicates.groupby(["alpha", "power_W", "region"], as_index=False)
        .agg(
            q_fraction_min=("q_positive_fraction", "min"),
            q_fraction_max=("q_positive_fraction", "max"),
            q_fraction_median=("q_positive_fraction", "median"),
            sign_agreement_min=("point_sign_agreement_with_k25", "min"),
            sign_agreement_median=("point_sign_agreement_with_k25", "median"),
        )
        .sort_values(["region", "power_W"])
    )
    return replicates, core, summary
