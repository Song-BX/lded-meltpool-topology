from __future__ import annotations

import pandas as pd


def reconcile_legacy_summary(legacy: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    support_by_power = support.set_index("power_W")
    rows: list[dict[str, object]] = []
    for row in legacy.itertuples(index=False):
        power = int(row.power_W)
        support_row = support_by_power.loc[power]
        metric = str(row.metric)
        requested_n = int(row.n_top)
        is_positive_q = metric.startswith("Qpos_")
        available = int(support_row.positive_Q_points if is_positive_q else support_row.valid_slice_points)
        enough_requested_points = available >= requested_n
        if metric == "Qpos_top10" and not enough_requested_points:
            status = "unreconciled_legacy_summary"
            note = (
                f"Legacy {metric} records n_top={requested_n}, but the current canonical XZ slice "
                f"contains only {available} positive-Q points."
            )
        else:
            status = "not_revalidated_geometry_withdrawn"
            note = "Geometry is withdrawn because the complete XZ slice fails the 100-point support policy."
        rows.append(
            {
                "power_W": power,
                "legacy_metric": metric,
                "legacy_n_top": requested_n,
                "current_available_points": available,
                "requested_top_n_available": enough_requested_points,
                "support_status": str(support_row.evidence_status),
                "reconciliation_status": status,
                "reconciliation_note": note,
            }
        )
    return pd.DataFrame(rows).sort_values(["power_W", "legacy_metric"]).reset_index(drop=True)
