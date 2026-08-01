from __future__ import annotations

import pandas as pd

from .config import (
    AGGREGATION_STRATEGIES,
    ALL_STANDARD_COLUMNS,
    COORDINATE_COLUMNS,
    PHYSICAL_COLUMNS,
)


def aggregate_points(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Collapse exact-coordinate classes using one pre-specified strategy."""
    if strategy not in AGGREGATION_STRATEGIES:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")
    missing = [column for column in ALL_STANDARD_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing standardized columns: {missing}")

    grouped = frame.groupby(list(COORDINATE_COLUMNS), sort=False, dropna=False)
    # Preserve the canonical geometry calculation used by deduplicate_points.
    # Even exact stored coordinates can differ at the last floating-point bit
    # after a mean reduction, which can alter tie-breaking among equidistant kNNs.
    coordinate_values = grouped[list(COORDINATE_COLUMNS)].mean().reset_index(drop=True)
    if strategy == "mean_all_records":
        output = grouped[list(ALL_STANDARD_COLUMNS)].mean().reset_index(drop=True)
    elif strategy == "median_all_records":
        physical = grouped[list(PHYSICAL_COLUMNS)].median().reset_index(drop=True)
        output = pd.concat([coordinate_values, physical], axis=1)
    elif strategy == "first_record":
        physical = grouped[list(PHYSICAL_COLUMNS)].first().reset_index(drop=True)
        output = pd.concat([coordinate_values, physical], axis=1)
    else:
        rows: list[dict[str, float]] = []
        for _, group in grouped:
            distinct_states = group[list(PHYSICAL_COLUMNS)].drop_duplicates()
            rows.append(distinct_states.mean().to_dict())
        physical = pd.DataFrame(rows)
        output = pd.concat([coordinate_values, physical], axis=1)
    return output.loc[:, list(ALL_STANDARD_COLUMNS)].reset_index(drop=True)
