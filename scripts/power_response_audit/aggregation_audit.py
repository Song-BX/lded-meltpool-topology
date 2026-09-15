from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AGGREGATION_STRATEGIES, CANONICAL_AGGREGATION, METRICS, POWERS
from .local_extrema import classify_discrete_extrema


def aggregation_metric_frame(aggregation: pd.DataFrame, median_aggregation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        selected = median_aggregation.copy() if metric.metric_id == "temperature_median_full_pool_K" else aggregation.loc[aggregation["region"] == metric.region].copy()
        for strategy in AGGREGATION_STRATEGIES:
            block = selected.loc[selected["aggregation_strategy"] == strategy].copy()
            block["power_W"] = pd.to_numeric(block["power_W"], errors="raise").astype(int)
            if block["power_W"].duplicated().any() or tuple(sorted(block["power_W"])) != POWERS:
                raise ValueError(f"{strategy}/{metric.metric_id} does not contain the complete power grid")
            for _, row in block.sort_values("power_W").iterrows():
                rows.append(
                    {
                        "aggregation_strategy": strategy,
                        "power_W": int(row["power_W"]),
                        "metric_id": metric.metric_id,
                        "metric_label": metric.label,
                        "unit": metric.unit,
                        "region": metric.region,
                        "source_column": metric.aggregation_column,
                        "value": float(row[metric.aggregation_column]),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["aggregation_strategy", "metric_id", "power_W"], ignore_index=True
    )


def audit_aggregation(canonical_long: pd.DataFrame, aggregation_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical_branch = aggregation_long.loc[
        aggregation_long["aggregation_strategy"] == CANONICAL_AGGREGATION
    ]
    comparison = canonical_long.merge(
        canonical_branch[["power_W", "metric_id", "value"]],
        on=["power_W", "metric_id"],
        suffixes=("_canonical", "_aggregation"),
        validate="one_to_one",
    )
    comparison["absolute_difference"] = np.abs(
        comparison["value_canonical"] - comparison["value_aggregation"]
    )
    comparison["passed"] = np.isclose(
        comparison["value_canonical"], comparison["value_aggregation"], rtol=1e-10, atol=1e-12
    )
    if not comparison["passed"].all():
        raise ValueError("The canonical aggregation branch does not reproduce the retained metrics")

    extrema = classify_discrete_extrema(
        aggregation_long, ["aggregation_strategy", "metric_id"]
    )
    statuses = extrema.pivot_table(
        index=["metric_id", "power_W"],
        columns="aggregation_strategy",
        values="extremum_status",
        aggfunc="first",
    ).reindex(columns=AGGREGATION_STRATEGIES)
    consistency = statuses.apply(lambda row: row.nunique(dropna=False) == 1, axis=1)
    extrema = extrema.merge(
        consistency.rename("status_consistent_all_strategies"),
        on=["metric_id", "power_W"],
        validate="many_to_one",
    )
    return extrema, comparison.sort_values(["metric_id", "power_W"], ignore_index=True)
