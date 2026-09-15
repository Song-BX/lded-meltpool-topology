from __future__ import annotations

import pandas as pd

from .config import CANONICAL_STRATEGY, CANONICAL_TIME_S


def _canonical_row(audit: pd.DataFrame) -> pd.Series:
    rows = audit.loc[
        (audit["audit_context"] == "aggregation_sensitivity")
        & (audit["time_s"] == CANONICAL_TIME_S)
        & (audit["aggregation_strategy"] == CANONICAL_STRATEGY)
    ]
    if len(rows) != 1:
        raise ValueError("The canonical 0.70 s mean-all-records overlap row is missing or duplicated.")
    return rows.iloc[0]


def build_summary(audit: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    canonical = _canonical_row(audit)
    strategy_rows = audit.loc[audit["audit_context"] == "aggregation_sensitivity"]
    temporal_rows = audit.loc[audit["audit_context"] == "temporal_context"]
    all_strategy_overlap = bool(strategy_rows["iqr_overlap_observed"].all())
    all_strategy_containment = bool(strategy_rows["one_iqr_contained_in_other"].all())
    central_status = (
        "not_supported_by_central_distribution"
        if bool(canonical["iqr_overlap_observed"])
        else "not_assessed_as_distributional_separation"
    )
    summary = pd.DataFrame(
        [
            {
                "metric": "canonical_iqr_350W_mps",
                "value": f"{canonical.p25_350_mps:.6f}--{canonical.p75_350_mps:.6f}",
                "interpretation": "central 50% of the 0.70 s canonical full-pool distribution",
            },
            {
                "metric": "canonical_iqr_400W_mps",
                "value": f"{canonical.p25_400_mps:.6f}--{canonical.p75_400_mps:.6f}",
                "interpretation": "central 50% of the 0.70 s canonical full-pool distribution",
            },
            {
                "metric": "canonical_iqr_relation",
                "value": str(canonical.contained_iqr),
                "interpretation": "descriptive central-range relation; not an inferential distributional test",
            },
            {
                "metric": "canonical_p99_delta_350_minus_400_mps",
                "value": float(canonical.delta_p99_350_minus_400_mps),
                "interpretation": "upper-tail context for the maximum-velocity audit",
            },
            {
                "metric": "canonical_350_points_above_400_vmax",
                "value": int(canonical.n_350_gt_400_vmax),
                "interpretation": "sparse unique-coordinate support for the maximum contrast",
            },
        ]
    )
    decision = {
        "claim_id": "full_pool_velocity_distribution_separation_350_400",
        "analysis_scope": "descriptive central-distribution and tail audit of matched point-cloud snapshots; no p values, bootstrap, spatial-point independence, or continuous-power inference",
        "canonical_time_s": CANONICAL_TIME_S,
        "canonical_central_distribution": {
            "n_unique_points_350": int(canonical.n_unique_points_350),
            "n_unique_points_400": int(canonical.n_unique_points_400),
            "iqr_350_mps": [float(canonical.p25_350_mps), float(canonical.p75_350_mps)],
            "iqr_400_mps": [float(canonical.p25_400_mps), float(canonical.p75_400_mps)],
            "iqr_overlap_observed": bool(canonical.iqr_overlap_observed),
            "one_iqr_contained_in_other": bool(canonical.one_iqr_contained_in_other),
            "contained_iqr": str(canonical.contained_iqr),
            "p99_350_mps": float(canonical.p99_350_mps),
            "p99_400_mps": float(canonical.p99_400_mps),
            "vmax_350_mps": float(canonical.max_350_mps),
            "vmax_400_mps": float(canonical.max_400_mps),
            "n_350_gt_400_vmax": int(canonical.n_350_gt_400_vmax),
            "prop_350_gt_400_vmax": float(canonical.prop_350_gt_400_vmax),
        },
        "aggregation_sensitivity": {
            "strategy_count": int(len(strategy_rows)),
            "all_fixed_strategies_iqr_overlap": all_strategy_overlap,
            "all_fixed_strategies_one_iqr_contained": all_strategy_containment,
        },
        "temporal_context": {
            "serial_snapshot_count": int(len(temporal_rows) + 1),
            "role": "descriptive serial context only; not independent repetition",
        },
        "whole_pool_distribution_separation": central_status,
        "vmax_role": "sparse_peak_audit_only",
        "allowed_interpretation": "The maximum is a sparse peak-level audit record, and the observed IQR relation describes only the central ranges of the current exported snapshot.",
        "prohibited_interpretation": [
            "whole-pool velocity-distribution separation",
            "robust structural signal",
            "whole-pool flow-strength decrease",
            "comparative evidence",
            "numerical convergence",
            "physical mechanism or causal explanation",
        ],
    }
    return summary, decision
