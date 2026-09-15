from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .config import AGGREGATION_STRATEGIES, CANONICAL_TIME_S, METRICS
from .local_extrema import LOCAL_MAXIMUM


def build_summary(
    snapshot_extrema: pd.DataFrame,
    aggregation_extrema: pd.DataFrame,
    temporal_extrema: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    canonical_status = snapshot_extrema.set_index(["metric_id", "power_W"])["extremum_status"]
    aggregation = aggregation_extrema.pivot_table(
        index=["metric_id", "power_W"],
        columns="aggregation_strategy",
        values="extremum_status",
        aggfunc="first",
    )
    temporal = temporal_extrema.pivot_table(
        index=["metric_id", "power_W"],
        columns="time_s",
        values="extremum_status",
        aggfunc="first",
    )

    rows: list[dict[str, object]] = []
    for metric in METRICS:
        for power in (200, 250, 300, 350, 400, 450):
            status = canonical_status.loc[(metric.metric_id, power)]
            aggregation_statuses = aggregation.loc[(metric.metric_id, power)].reindex(
                AGGREGATION_STRATEGIES
            )
            temporal_statuses = temporal.loc[(metric.metric_id, power)]
            temporal_match_count = int((temporal_statuses == status).sum())
            rows.append(
                {
                    "metric_id": metric.metric_id,
                    "metric_label": metric.label,
                    "power_W": power,
                    "snapshot_time_s": CANONICAL_TIME_S,
                    "snapshot_extremum_status": status,
                    "aggregation_status_consistent": bool(aggregation_statuses.nunique() == 1),
                    "aggregation_statuses": ";".join(
                        f"{strategy}:{aggregation_statuses[strategy]}"
                        for strategy in AGGREGATION_STRATEGIES
                    ),
                    "temporal_matching_status_count": temporal_match_count,
                    "temporal_status_changes": bool(temporal_match_count < len(temporal_statuses)),
                    "temporal_statuses": ";".join(
                        f"{float(time_s):.2f}:{temporal_statuses[time_s]}"
                        for time_s in sorted(temporal_statuses.index)
                    ),
                }
            )
    summary = pd.DataFrame(rows).sort_values(["metric_id", "power_W"], ignore_index=True)

    maxima = summary.loc[
        summary["snapshot_extremum_status"] == LOCAL_MAXIMUM,
        ["metric_id", "power_W", "aggregation_status_consistent", "temporal_status_changes"],
    ]
    decision = {
        "decision": "no_physical_inflection_claim",
        "analysis_scope": "six discrete single-simulation L-DED/FLOW-3D cases at one configuration",
        "canonical_snapshot_time_s": CANONICAL_TIME_S,
        "continuous_power_interpolation_performed": False,
        "derivative_or_inflection_estimation_performed": False,
        "independent_statistical_replicates": False,
        "q_used_as_physical_or_comparative_evidence": False,
        "endpoint_rule": "200 W and 450 W are displayed but not tested as local extrema",
        "local_extremum_rule": "An internal sampled power must be strictly greater or strictly smaller than both adjacent 50 W cases.",
        "snapshot_local_maxima": maxima.to_dict(orient="records"),
        "interpretation": (
            "The audit identifies only sampled-power, snapshot-level extrema. It cannot distinguish "
            "a local export/transient anomaly from a physically meaningful continuous-power feature."
        ),
    }
    return summary, decision

