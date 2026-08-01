from __future__ import annotations

import pandas as pd

from scripts.gradient_validation.resampling import run_neighbour_resampling

from .config import ALPHA_SPECS


def run_alpha_resampling(cases: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reuse the fixed 20-of-25 subset diagnostic under every fixed exponent."""
    result_sets = [run_neighbour_resampling(cases, alpha=alpha.value) for alpha in ALPHA_SPECS]
    replicates = pd.concat([result[0] for result in result_sets], ignore_index=True)
    core = pd.concat([result[1] for result in result_sets], ignore_index=True)
    summary = pd.concat([result[2] for result in result_sets], ignore_index=True)
    labels = pd.DataFrame(
        {
            "alpha": [spec.value for spec in ALPHA_SPECS],
            "alpha_label": [spec.label for spec in ALPHA_SPECS],
            "alpha_role": [spec.role for spec in ALPHA_SPECS],
        }
    )
    return (
        replicates.merge(labels, on="alpha", validate="many_to_one").sort_values(
            ["alpha", "replicate", "power_W", "region"]
        ),
        core.merge(labels, on="alpha", validate="many_to_one").sort_values(
            ["alpha", "replicate", "region"]
        ),
        summary.merge(labels, on="alpha", validate="many_to_one").sort_values(
            ["alpha", "power_W", "region"]
        ),
    )

