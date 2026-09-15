from __future__ import annotations

import pandas as pd

from scripts.analysis.point_cloud import standardize_columns
from scripts.export_diagnostics.aggregation import aggregate_points
from scripts.temporal_validation.discovery import discover_snapshots

from .config import AGGREGATION_STRATEGIES, CANONICAL_TIME_S, RAW_DIR, TEMPORAL_DIR


def aggregation_median_temperature_frame() -> pd.DataFrame:
    """Compute unfiltered full-pool median temperature for the four fixed aggregations.

    This is a descriptive source transformation only.  It neither filters hot
    records nor changes the canonical mean-all-records coordinate rule.
    """
    rows: list[dict[str, object]] = []
    snapshots = [item for item in discover_snapshots(RAW_DIR, TEMPORAL_DIR) if item.time_s == CANONICAL_TIME_S]
    for snapshot in snapshots:
        raw = standardize_columns(pd.read_csv(snapshot.path))
        for strategy in AGGREGATION_STRATEGIES:
            values = aggregate_points(raw, strategy)["T"]
            rows.append(
                {
                    "aggregation_strategy": strategy,
                    "power_W": snapshot.power_W,
                    "region": "all",
                    "T_median_K": float(values.median()),
                    "n_unique_points": int(len(values)),
                    "source_snapshot": snapshot.path.name,
                    "temperature_filter": "none_unfiltered",
                }
            )
    return pd.DataFrame(rows).sort_values(["aggregation_strategy", "power_W"]).reset_index(drop=True)
