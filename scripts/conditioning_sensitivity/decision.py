from __future__ import annotations

import math

import pandas as pd

from .config import CUTOFF_SPECS, K_VALUES, WLS_CONDITION_CUTOFF


def summarize_cutoffs(metrics: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-specified support and directional decision rule."""
    canonical = core[
        (core["cutoff_value"] == WLS_CONDITION_CUTOFF)
        & (core["region"] == "all")
        & (core["threshold"] == "Q>0")
    ].sort_values("kNN")
    directions = set(canonical["direction"])
    if len(canonical) != len(K_VALUES) or len(directions) != 1 or "tie" in directions:
        raise RuntimeError("The canonical kappa=100 full-pool Q>0 direction is not unique.")
    canonical_direction = directions.pop()

    rows: list[dict[str, float | int | str | bool]] = []
    for spec in CUTOFF_SPECS:
        support = metrics[
            (metrics["cutoff_value"] == spec.value)
            & (metrics["region"] == "all")
            & (metrics["threshold"] == "Q>0")
        ]
        contrast = core[
            (core["cutoff_value"] == spec.value)
            & (core["region"] == "all")
            & (core["threshold"] == "Q>0")
        ].sort_values("kNN")
        support_complete = len(support) == 6 * len(K_VALUES)
        all_supported = support_complete and bool((support["n_region"] >= 100).all())
        direction_complete = len(contrast) == len(K_VALUES)
        direction_matches = int((contrast["direction"] == canonical_direction).sum())
        direction_consistent = direction_complete and direction_matches == len(K_VALUES)
        if not all_supported:
            status = "insufficient_support"
        elif direction_consistent:
            status = "support_and_direction_consistent"
        else:
            status = "support_but_direction_changed"
        rows.append(
            {
                "cutoff_label": spec.label,
                "cutoff_value": spec.value,
                "canonical_direction": canonical_direction,
                "fullpool_cells": len(support),
                "fullpool_cells_with_support": int((support["n_region"] >= 100).sum()),
                "all_cells_support_qualified": all_supported,
                "direction_match_count": direction_matches,
                "direction_positive_count": int((contrast["direction"] == "350>400").sum()),
                "direction_negative_count": int((contrast["direction"] == "350<400").sum()),
                "direction_tie_count": int((contrast["direction"] == "tie").sum()),
                "direction_consistent": direction_consistent,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def decision_payload(
    distribution: pd.DataFrame, point_audit: pd.DataFrame, summary: pd.DataFrame
) -> dict[str, object]:
    supported = summary[summary["all_cells_support_qualified"]]
    any_direction_change = bool(
        (supported["status"] == "support_but_direction_changed").any()
    )
    legacy = point_audit[point_audit["cutoff_label"] == "1e12"]
    infinite = point_audit[point_audit["cutoff_label"] == "inf"]
    return {
        "analysis_scope": "six 0.70 s exports, exact-coordinate consolidation, k=8-50",
        "conditioning_definition": "cond(sqrt(W) A) with alpha=0",
        "canonical_cutoff": WLS_CONDITION_CUTOFF,
        "legacy_cutoff": 1.0e12,
        "cutoffs": [spec.label for spec in CUTOFF_SPECS],
        "near_cutoff_definition": "0.5*kappa_max < kappa <= kappa_max for finite cutoffs",
        "legacy_finite_points_above_1e12": int(legacy["exceeded_points"].sum()),
        "legacy_nonfinite_points": int(legacy["nonfinite_points"].sum()),
        "legacy_and_infinite_retained_point_difference": int(
            infinite["retained_points"].sum() - legacy["retained_points"].sum()
        ),
        "max_finite_kappa": float(distribution["kappa_max"].max()),
        "final_q_claim_status": (
            "conditioning_cutoff_dependent"
            if any_direction_change
            else "directionally_consistent_over_supported_predefined_cutoffs"
        ),
        "cutoff_summary": summary.to_dict(orient="records"),
        "interpretation_boundary": "Condition-number screening does not establish solver-gradient fidelity, CFD physical truth, or temporal persistence.",
    }
