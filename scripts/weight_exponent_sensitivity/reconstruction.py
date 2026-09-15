from __future__ import annotations

import pandas as pd

from scripts.analysis.wls_q import reconstruct_case

from .config import (
    ALPHA_SPECS,
    K_VALUES,
    WLS_CONDITION_CUTOFF,
    WLS_CONDITION_MODE,
    WLS_DISTANCE_OFFSET_M,
)


def reconstruct_grid(cases: dict[int, pd.DataFrame]) -> dict[tuple[str, int, int], pd.DataFrame]:
    """Run the fixed alpha × power × k grid on unchanged neighbour definitions."""
    reconstructed: dict[tuple[str, int, int], pd.DataFrame] = {}
    for alpha in ALPHA_SPECS:
        for k in K_VALUES:
            for power, frame in sorted(cases.items()):
                reconstructed[(alpha.label, power, k)] = reconstruct_case(
                    frame,
                    k=k,
                    alpha=alpha.value,
                    eps_w=WLS_DISTANCE_OFFSET_M,
                    kappa_max=WLS_CONDITION_CUTOFF,
                    condition_on=WLS_CONDITION_MODE,
                )
    return reconstructed

