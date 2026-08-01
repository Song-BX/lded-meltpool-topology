from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .config import COORDINATE_COLUMNS, PHYSICAL_COLUMNS
from .discovery import SnapshotInput


def analyse_snapshot(
    record: SnapshotInput, frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = frame.groupby(list(COORDINATE_COLUMNS), sort=False, dropna=False)
    group_sizes = grouped.size()
    state_counts = pd.Series(
        {
            coordinates: len(group[list(PHYSICAL_COLUMNS)].drop_duplicates())
            for coordinates, group in grouped
        }
    )

    raw_points = int(len(frame))
    unique_coordinates = int(len(group_sizes))
    unique_full_rows = int(len(frame.drop_duplicates()))
    additional_distinct_state_rows = unique_full_rows - unique_coordinates
    exact_repeated_rows = raw_points - unique_full_rows
    conflicting_groups = int((state_counts > 1).sum())
    modes = group_sizes.mode()

    summary = pd.DataFrame(
        [
            {
                "time_s": record.time_s,
                "power_W": record.power_W,
                "raw_points": raw_points,
                "unique_coordinates": unique_coordinates,
                "unique_full_rows": unique_full_rows,
                "coordinate_duplicate_rows": raw_points - unique_coordinates,
                "coordinate_duplicate_ratio": (raw_points - unique_coordinates) / raw_points,
                "unique_coordinate_representatives": unique_coordinates,
                "additional_distinct_state_rows": additional_distinct_state_rows,
                "exact_repeated_rows": exact_repeated_rows,
                "exact_full_row_duplicate_ratio": exact_repeated_rows / raw_points,
                "conflicting_coordinate_groups": conflicting_groups,
                "conflicting_coordinate_group_fraction": conflicting_groups / unique_coordinates,
                "multiplicity_min": int(group_sizes.min()),
                "multiplicity_median": float(group_sizes.median()),
                "multiplicity_p90": float(group_sizes.quantile(0.90)),
                "multiplicity_mode": int(modes.iloc[0]),
                "multiplicity_max": int(group_sizes.max()),
                "raw_points_mod_12": raw_points % 12,
            }
        ]
    )

    multiplicity_rows = [
        {
            "time_s": record.time_s,
            "power_W": record.power_W,
            "multiplicity": multiplicity,
            "coordinate_groups": count,
            "coordinate_group_fraction": count / unique_coordinates,
        }
        for multiplicity, count in sorted(Counter(group_sizes.tolist()).items())
    ]
    multiplicity = pd.DataFrame(multiplicity_rows)

    variable_rows: list[dict[str, float | int | str]] = []
    range_cache: dict[str, pd.Series] = {}
    for variable in PHYSICAL_COLUMNS:
        ranges = grouped[variable].max() - grouped[variable].min()
        range_cache[variable] = ranges
        nonzero = ranges[ranges > 0]
        global_span = float(frame[variable].max() - frame[variable].min())
        variable_rows.append(
            {
                "time_s": record.time_s,
                "power_W": record.power_W,
                "variable": variable,
                "coordinate_groups": unique_coordinates,
                "conflict_groups": int((ranges > 0).sum()),
                "conflict_group_fraction": float((ranges > 0).mean()),
                "nonzero_range_median": float(nonzero.median()) if len(nonzero) else 0.0,
                "nonzero_range_p90": float(nonzero.quantile(0.90)) if len(nonzero) else 0.0,
                "maximum_group_range": float(ranges.max()),
                "case_global_span": global_span,
                "maximum_range_relative_to_case_span": (
                    float(ranges.max()) / global_span if global_span else np.nan
                ),
            }
        )
    variable_consistency = pd.DataFrame(variable_rows)

    ordered_groups = list(grouped)
    deterministic_index = (int(round(record.time_s * 100)) + record.power_W) % len(ordered_groups)
    checks: list[dict[str, float | int | str]] = []
    selected: list[tuple[str, tuple[float, float, float], pd.DataFrame, str]] = []
    random_key, random_group = ordered_groups[deterministic_index]
    selected.append(("deterministic_sample", random_key, random_group, "all"))
    max_mult_key = group_sizes.idxmax()
    selected.append(("maximum_multiplicity", max_mult_key, grouped.get_group(max_mult_key), "all"))
    for variable in ("T", "V", "fof"):
        key = range_cache[variable].idxmax()
        selected.append(("maximum_variable_range", key, grouped.get_group(key), variable))
    for check_type, key, group, variable in selected:
        checks.append(
            {
                "time_s": record.time_s,
                "power_W": record.power_W,
                "check_type": check_type,
                "variable": variable,
                "x": key[0],
                "y": key[1],
                "z": key[2],
                "multiplicity": len(group),
                "unique_physical_states": len(group[list(PHYSICAL_COLUMNS)].drop_duplicates()),
                "observed_range": (
                    0.0 if variable == "all" else float(group[variable].max() - group[variable].min())
                ),
            }
        )
    return summary, multiplicity, variable_consistency, pd.DataFrame(checks)
