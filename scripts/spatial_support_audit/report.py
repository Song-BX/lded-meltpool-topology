from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .legacy_shift_config import NOMINAL_GRID_SPACING_MM
from .paths import AUDIT_DIR
from .slice_support import MIN_VALID_POINTS, REQUESTED_TOP_N


def write_outputs(
    support: pd.DataFrame,
    reconciliation: pd.DataFrame,
    legacy_shift_context: pd.DataFrame,
) -> dict[str, Path]:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    support_path = AUDIT_DIR / "slice_support_audit.csv"
    reconciliation_path = AUDIT_DIR / "legacy_extreme_summary_reconciliation.csv"
    legacy_shift_path = AUDIT_DIR / "legacy_centroid_shift_context.csv"
    decision_path = AUDIT_DIR / "spatial_exclusion_decision.json"
    support.to_csv(support_path, index=False, encoding="utf-8-sig")
    reconciliation.to_csv(reconciliation_path, index=False, encoding="utf-8-sig")
    legacy_shift_context.to_csv(legacy_shift_path, index=False, encoding="utf-8-sig")
    shift_record = legacy_shift_context.iloc[0]
    decision = {
        "decision": "exclude_spatial_geometric_comparisons",
        "all_spatial_geometric_comparisons_eligible": False,
        "support_policy": {
            "minimum_valid_points": MIN_VALID_POINTS,
            "requested_positive_Q_top_n": REQUESTED_TOP_N,
            "policy_origin": "Reviewer #1 Comment 4 evidence-support eligibility",
        },
        "per_power": support.to_dict(orient="records"),
        "legacy_summary_status": sorted(reconciliation["reconciliation_status"].unique().tolist()),
        "comment_11_historical_centroid_shift": {
            "status": "archival_context_only",
            "source_metric": str(shift_record.source_metric),
            "reference_power_W": int(shift_record.reference_power_W),
            "comparison_power_W": int(shift_record.comparison_power_W),
            "nominal_grid_spacing_mm": NOMINAL_GRID_SPACING_MM,
            "comparison_reconciliation_status": str(shift_record.comparison_reconciliation_status),
            "not_distinguishable_from_reconstruction_noise_with_available_data": True,
            "nominal_grid_spacing_is_not": "a displacement uncertainty or reconstruction-noise estimate",
            "prohibited_rescue_methods": [
                "bootstrap or p values",
                "smoothing",
                "wider slice",
                "different neighbourhood or spatial estimator",
                "centroid or RMS replacement",
            ],
        },
        "prohibited_rescue_methods": [
            "centroid or RMS replacement",
            "smoothing",
            "bootstrap or p values",
            "wider slice",
            "different neighbourhood or spatial estimator",
        ],
    }
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return {
        "support": support_path,
        "reconciliation": reconciliation_path,
        "legacy_shift_context": legacy_shift_path,
        "decision": decision_path,
    }
