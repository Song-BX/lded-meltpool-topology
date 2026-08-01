from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.export_diagnostics.config import AGGREGATION_STRATEGIES

from .config import (
    ALPHA_SPECS,
    CUTOFF_SPECS,
    K_REFERENCE,
    QUADRATIC_K_VALUES,
    RESAMPLE_COUNT,
    RESAMPLE_NEIGHBOURS,
    RESAMPLE_SEED,
)
from .reconstruction import (
    fixed_neighbour_sets,
    reconstruct_aggregation_grid,
    reconstruct_from_sets,
    reconstruct_grid,
    reconstruct_quadratic_grid,
    sample_neighbour_sets,
    screen_condition_number,
)
from .summaries import core_contrasts, summarize_grid


def _combine(results: list[tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.concat([item[0] for item in results], ignore_index=True),
        pd.concat([item[1] for item in results], ignore_index=True),
    )


def aggregation_sensitivity(raw_cases: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    for strategy in AGGREGATION_STRATEGIES:
        grid = reconstruct_aggregation_grid(raw_cases, strategy)
        results.append(
            summarize_grid(
                grid,
                context="aggregation_sensitivity",
                metadata={"aggregation_strategy": strategy},
            )
        )
    return _combine(results)


def exponent_sensitivity(
    cases: dict[int, pd.DataFrame], canonical_grid: dict[tuple[int, int], pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    for alpha_spec in ALPHA_SPECS:
        grid = canonical_grid if alpha_spec.value == 0.0 else reconstruct_grid(cases, alpha=alpha_spec.value)
        results.append(
            summarize_grid(
                grid,
                context="weight_exponent_sensitivity",
                metadata={"alpha_label": alpha_spec.label, "alpha": alpha_spec.value},
            )
        )
    return _combine(results)


def conditioning_sensitivity(cases: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    uncapped = reconstruct_grid(cases, kappa_max=float("inf"))
    results = []
    for cutoff_spec in CUTOFF_SPECS:
        screened = {
            key: screen_condition_number(frame, cutoff_spec.value)
            for key, frame in uncapped.items()
        }
        results.append(
            summarize_grid(
                screened,
                context="conditioning_sensitivity",
                metadata={"cutoff_label": cutoff_spec.label, "cutoff": cutoff_spec.value},
            )
        )
    return _combine(results)


def model_order_sensitivity(
    cases: dict[int, pd.DataFrame], canonical_grid: dict[tuple[int, int], pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first = {key: frame for key, frame in canonical_grid.items() if key[1] in QUADRATIC_K_VALUES}
    second = reconstruct_quadratic_grid(cases, QUADRATIC_K_VALUES)
    first_metrics, first_agreement = summarize_grid(
        first, context="model_order_sensitivity", metadata={"method": "first_order"}
    )
    second_metrics, second_agreement = summarize_grid(
        second, context="model_order_sensitivity", metadata={"method": "second_order"}
    )
    joined = core_contrasts(pd.concat([first_metrics, second_metrics], ignore_index=True))
    return (
        pd.concat([first_metrics, second_metrics], ignore_index=True),
        pd.concat([first_agreement, second_agreement], ignore_index=True),
        joined,
    )


def neighbour_subset_sensitivity(
    cases: dict[int, pd.DataFrame], *, alpha_specs=ALPHA_SPECS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse the fixed-seed 20-of-25 membership diagnostic for all three descriptors."""
    base_sets = fixed_neighbour_sets(cases)
    results = []
    for alpha_spec in alpha_specs:
        generator = np.random.default_rng(RESAMPLE_SEED)
        for replicate in range(1, RESAMPLE_COUNT + 1):
            grid = {
                (power, K_REFERENCE): reconstruct_from_sets(
                    cases[power],
                    sample_neighbour_sets(base_sets[power], generator, RESAMPLE_NEIGHBOURS),
                    alpha=alpha_spec.value,
                )
                for power in sorted(cases)
            }
            results.append(
                summarize_grid(
                    grid,
                    context="neighbour_subset_sensitivity",
                    metadata={
                        "alpha_label": alpha_spec.label,
                        "alpha": alpha_spec.value,
                        "replicate": replicate,
                        "subset_neighbours": RESAMPLE_NEIGHBOURS,
                    },
                )
            )
        print(f"completed complementary neighbour-subset audit at alpha={alpha_spec.label}", flush=True)
    return _combine(results)
