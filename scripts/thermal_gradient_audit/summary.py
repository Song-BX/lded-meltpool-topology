from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CANONICAL_AGGREGATION, CANONICAL_TIME_S, SUPPORT_GATE


def _direction(first: float, second: float) -> str:
    if np.isclose(first, second, rtol=0.0, atol=0.0):
        return "tie"
    return "350_gt_400" if first > second else "350_lt_400"


def _ordered_powers(frame: pd.DataFrame) -> str:
    ranked = frame.sort_values("gradT_median_K_per_m", ascending=False)["power_W"].astype(int).tolist()
    return ">".join(str(power) for power in ranked)


def build_summary(metrics: pd.DataFrame, aggregation: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    canonical = metrics.loc[
        (np.isclose(metrics["time_s"], CANONICAL_TIME_S))
        & (metrics["aggregation_strategy"] == CANONICAL_AGGREGATION)
    ].copy()
    if len(canonical) != 12:
        raise ValueError(f"Expected 12 canonical gradient summaries, found {len(canonical)}.")

    summary_rows: list[dict[str, object]] = []
    canonical_snapshot: dict[str, object] = {}
    aggregation_status: dict[str, object] = {}
    temporal_status: dict[str, object] = {}
    for region, group in canonical.groupby("region", sort=False):
        group = group.sort_values("power_W")
        maxima = group.loc[group["sampled_power_extremum"] == "discrete_local_maximum", "power_W"].astype(int).tolist()
        canonical_snapshot[region] = {
            "power_order_descending": _ordered_powers(group),
            "sampled_power_local_maxima_W": maxima,
            "support_eligible_all_powers": bool(group["support_eligible"].all()),
        }
        summary_rows.append(
            {
                "summary_type": "canonical_snapshot",
                "region": region,
                "power_W": np.nan,
                "time_s": CANONICAL_TIME_S,
                "value": np.nan,
                "status": "exploratory_late_time_snapshot",
                "detail": f"median-G order={_ordered_powers(group)}; local maxima={maxima}",
            }
        )

        strategy_orders: dict[str, str] = {}
        median_matches: dict[str, bool] = {}
        extrema_match: dict[str, bool] = {}
        relative_deviations: dict[str, float] = {}
        canonical_maxima = group.loc[
            group["sampled_power_extremum"] == "discrete_local_maximum", "power_W"
        ].astype(int).tolist()
        for strategy, strategy_group in aggregation.loc[aggregation["region"] == region].groupby("aggregation_strategy", sort=False):
            strategy_group = strategy_group.sort_values("power_W")
            baseline = group.set_index("power_W")["gradT_median_K_per_m"]
            tested = strategy_group.set_index("power_W")["gradT_median_K_per_m"].reindex(baseline.index)
            median_matches[strategy] = bool(np.allclose(tested, baseline, rtol=1e-12, atol=0.0, equal_nan=True))
            strategy_orders[strategy] = _ordered_powers(strategy_group)
            extrema_match[strategy] = (
                strategy_group.loc[
                    strategy_group["sampled_power_extremum"] == "discrete_local_maximum", "power_W"
                ].astype(int).tolist()
                == canonical_maxima
            )
            relative_deviations[strategy] = float(
                np.max(np.abs(tested.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)) / baseline.to_numpy(dtype=float))
            )
        ordering_matches = all(order == _ordered_powers(group) for order in strategy_orders.values())
        extrema_all_match = all(extrema_match.values())
        max_relative_deviation = max(relative_deviations.values())
        aggregation_status[region] = {
            "all_strategy_medians_match_canonical": bool(all(median_matches.values())),
            "power_ordering_matches_canonical": ordering_matches,
            "sampled_power_local_maxima_match_canonical": extrema_all_match,
            "maximum_relative_median_deviation": max_relative_deviation,
            "strategy_orders_descending": strategy_orders,
        }
        summary_rows.append(
            {
                "summary_type": "aggregation_sensitivity",
                "region": region,
                "power_W": np.nan,
                "time_s": CANONICAL_TIME_S,
                "value": np.nan,
                "status": (
                    "all_strategy_medians_match"
                    if all(median_matches.values())
                    else "numerical_values_differ_without_order_change"
                    if ordering_matches and extrema_all_match
                    else "strategy_changes_order_or_extrema"
                ),
                "detail": "; ".join(
                    f"{name}: order={strategy_orders[name]}, max-relative-median-deviation={relative_deviations[name]:.6g}"
                    for name in strategy_orders
                ),
            }
        )

        late_window = metrics.loc[
            (metrics["region"] == region)
            & (metrics["aggregation_strategy"] == CANONICAL_AGGREGATION)
            & metrics["time_s"].isin([0.60, 0.65, 0.70])
        ]
        for power, power_group in late_window.groupby("power_W", sort=True):
            values = power_group["gradT_median_K_per_m"].to_numpy(dtype=float)
            reference = float(np.median(values))
            deviation = float(np.max(np.abs(values - reference) / reference)) if reference else np.nan
            summary_rows.append(
                {
                    "summary_type": "post_hoc_late_window_context",
                    "region": region,
                    "power_W": int(power),
                    "time_s": np.nan,
                    "value": deviation,
                    "status": "within_5pct" if deviation <= 0.05 else "exceeds_5pct",
                    "detail": "Post-hoc Comment 9 context; not part of the pre-specified Comment 1 decision.",
                }
            )

        directions: list[dict[str, object]] = []
        for time_s, time_group in metrics.loc[
            (metrics["region"] == region) & (metrics["aggregation_strategy"] == CANONICAL_AGGREGATION)
        ].groupby("time_s", sort=True):
            values = time_group.set_index("power_W")["gradT_median_K_per_m"]
            directions.append(
                {
                    "time_s": float(time_s),
                    "direction": _direction(float(values.loc[350]), float(values.loc[400])),
                }
            )
        temporal_status[region] = {
            "pairwise_350_vs_400": directions,
            "direction_consistent_over_five_snapshots": len({item["direction"] for item in directions}) == 1,
            "context_role": "post_hoc_comment9_context",
        }

    support = metrics.groupby("region", sort=False)["n_finite"].agg(["min", "max"]).to_dict(orient="index")
    decision: dict[str, object] = {
        "analysis_scope": "30 FLOW-3D CSV exports; direct scalar temperature-gradient magnitude only",
        "primary_descriptor_status": "direct_exported_temperature_gradient_magnitude",
        "source_field": "Temperature Gradient At Tgrdout",
        "uses_wls_knn_q_or_conditioning": False,
        "canonical_aggregation": CANONICAL_AGGREGATION,
        "snapshot_status": "exploratory_late_time_snapshot",
        "temporal_context_status": "post_hoc_comment9_context",
        "marangoni_status": "not_identifiable_from_available_csv",
        "marangoni_missing_inputs": [
            "temperature-gradient vector components",
            "free-surface normals or geometry",
            "tangential surface temperature gradient",
            "surface-tension temperature coefficient dgamma/dT",
            "direct Marangoni stress or force output",
        ],
        "support_gate_points": SUPPORT_GATE,
        "support_by_region": support,
        "canonical_snapshot": canonical_snapshot,
        "aggregation_sensitivity": aggregation_status,
        "temporal_pairwise_context": temporal_status,
        "expected_rows": {
            "thermal_gradient_metrics": 60,
            "thermal_gradient_aggregation_sensitivity": 48,
            "thermal_gradient_temporal_context": 60,
        },
    }
    return pd.DataFrame(summary_rows), decision
