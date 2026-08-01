from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import AGGREGATION_STRATEGIES, CANONICAL_STRATEGY


def decision_payload(
    summary: pd.DataFrame,
    sensitivity: dict[str, pd.DataFrame],
    optional_reexports: pd.DataFrame,
) -> dict[str, object]:
    baseline = summary[summary["time_s"] == 0.70]
    k25_core = sensitivity["k25_core_contrasts"]
    orderings = sensitivity["power_orderings"]
    changes = sensitivity["knn_direction_changes"]
    mismatch_counts = {
        strategy: int((changes["aggregation_strategy"] == strategy).sum())
        for strategy in AGGREGATION_STRATEGIES
        if strategy != CANONICAL_STRATEGY
    }
    return {
        "analysis_scope": "30 FLOW-3D snapshots (five times by six powers)",
        "analysis_date": "2026-07-28",
        "coordinate_matching": "exact equality in x, y, and z",
        "row_structure": {
            "coordinate_duplicate_ratio_all_files": [
                float(summary["coordinate_duplicate_ratio"].min()),
                float(summary["coordinate_duplicate_ratio"].max()),
            ],
            "exact_full_row_duplicate_ratio_all_files": [
                float(summary["exact_full_row_duplicate_ratio"].min()),
                float(summary["exact_full_row_duplicate_ratio"].max()),
            ],
            "coordinate_duplicate_ratio_at_0p70s": [
                float(baseline["coordinate_duplicate_ratio"].min()),
                float(baseline["coordinate_duplicate_ratio"].max()),
            ],
            "exact_full_row_duplicate_ratio_at_0p70s": [
                float(baseline["exact_full_row_duplicate_ratio"].min()),
                float(baseline["exact_full_row_duplicate_ratio"].max()),
            ],
            "conflicting_coordinate_group_fraction_at_0p70s": [
                float(baseline["conflicting_coordinate_group_fraction"].min()),
                float(baseline["conflicting_coordinate_group_fraction"].max()),
            ],
            "all_files_multiplicity_median_is_six": bool(
                (summary["multiplicity_median"] == 6).all()
            ),
            "all_files_multiplicity_mode_is_six": bool(
                (summary["multiplicity_mode"] == 6).all()
            ),
            "all_raw_row_counts_divisible_by_12": bool(
                (summary["raw_points_mod_12"] == 0).all()
            ),
        },
        "aggregation_sensitivity": {
            "strategies": list(AGGREGATION_STRATEGIES),
            "k25_core_directions_all_match_canonical": bool(
                k25_core["matches_canonical_direction"].all()
            ),
            "six_power_orderings_all_match_canonical": bool(
                orderings["matches_canonical_order"].all()
            ),
            "direction_mismatches_among_688_knn_region_threshold_cells": mismatch_counts,
            "all_direction_mismatches_are_p90": bool(
                changes.empty or (changes["threshold"] == "Q>posP90").all()
            ),
        },
        "upstream_reexport_audit": {
            "available_csv_files": int(len(optional_reexports)),
            "specific_flow3d_setting_attribution": "unresolved",
        },
        "manuscript_interpretation": {
            "preferred_term": "systematic export-level redundancy",
            "not_claimed": [
                "normal FLOW-3D behaviour",
                "a pathological FLOW-3D solver state",
                "a benefit or achievement of the post-processing framework",
                "identification of a specific export option from CSV data alone",
            ],
        },
    }


def write_decision(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

