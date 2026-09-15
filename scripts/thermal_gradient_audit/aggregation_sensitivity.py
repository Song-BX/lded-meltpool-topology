from __future__ import annotations

import pandas as pd

from scripts.temporal_validation.discovery import SnapshotFile

from .config import AGGREGATION_STRATEGIES, CANONICAL_TIME_S
from .metrics import attach_discrete_extrema, consolidate_snapshot, summarise_gradient


def compute_aggregation_sensitivity(snapshots: list[SnapshotFile]) -> pd.DataFrame:
    """Apply all pre-specified coordinate aggregations to the 0.70 s gradient export."""
    rows: list[dict[str, object]] = []
    target = [snapshot for snapshot in snapshots if abs(snapshot.time_s - CANONICAL_TIME_S) < 1e-12]
    if len(target) != 6:
        raise ValueError(f"Expected six t={CANONICAL_TIME_S:.2f} snapshots, found {len(target)}.")
    for strategy in AGGREGATION_STRATEGIES:
        for snapshot in target:
            consolidated = consolidate_snapshot(snapshot, strategy)
            rows.extend(
                summarise_gradient(
                    consolidated,
                    time_s=snapshot.time_s,
                    power_W=snapshot.power_W,
                    aggregation_strategy=strategy,
                )
            )
    frame = pd.DataFrame(rows).sort_values(["aggregation_strategy", "power_W", "region"]).reset_index(drop=True)
    return attach_discrete_extrema(frame)

