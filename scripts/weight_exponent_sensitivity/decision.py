from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import (
    AFFINE_NUMERICAL_TOLERANCE,
    ALPHA_SPECS,
    EXPECTED_POWERS,
    K_VALUES,
    MIN_REGION_POINTS,
)


def _alpha_summary(metrics: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for alpha in ALPHA_SPECS:
        primary = metrics[
            (metrics["alpha"] == alpha.value)
            & (metrics["region"] == "all")
            & (metrics["threshold"] == "Q>0")
        ]
        primary_contrasts = contrasts[
            (contrasts["alpha"] == alpha.value)
            & (contrasts["region"] == "all")
            & (contrasts["threshold"] == "Q>0")
        ]
        support_passed = bool(
            len(primary) == len(EXPECTED_POWERS) * len(K_VALUES)
            and (primary["n_region"] >= MIN_REGION_POINTS).all()
        )
        directions = primary_contrasts["direction"].tolist()
        rows.append(
            {
                "alpha_label": alpha.label,
                "alpha": alpha.value,
                "alpha_role": alpha.role,
                "power_k_cells": int(len(primary)),
                "support_passing_cells": int((primary["n_region"] >= MIN_REGION_POINTS).sum()),
                "support_passed_all_cells": support_passed,
                "contrast_k_count": int(len(primary_contrasts)),
                "direction_350_gt_400_count": int((primary_contrasts["direction"] == "350>400").sum()),
                "direction_350_lt_400_count": int((primary_contrasts["direction"] == "350<400").sum()),
                "tie_or_missing_count": int((~primary_contrasts["direction"].eq("350>400") & ~primary_contrasts["direction"].eq("350<400")).sum()),
                "all_directions_350_gt_400": bool(
                    len(directions) == len(K_VALUES) and all(direction == "350>400" for direction in directions)
                ),
                "delta_min": float(primary_contrasts["delta_350_400"].min()) if len(primary_contrasts) else np.nan,
                "delta_max": float(primary_contrasts["delta_350_400"].max()) if len(primary_contrasts) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def _common_support_summary(common_core: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for alpha in ALPHA_SPECS:
        block = common_core[common_core["alpha"] == alpha.value]
        rows.append(
            {
                "alpha_label": alpha.label,
                "alpha": alpha.value,
                "common_k_count": int(len(block)),
                "minimum_common_valid_points": int(block["minimum_common_valid_points"].min()) if len(block) else 0,
                "all_directions_350_gt_400": bool(
                    len(block) == len(K_VALUES) and (block["direction"] == "350>400").all()
                ),
                "delta_min": float(block["delta_350_400"].min()) if len(block) else np.nan,
                "delta_max": float(block["delta_350_400"].max()) if len(block) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def decision_payload(
    baseline: pd.DataFrame,
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    common_core: pd.DataFrame,
    manufactured: pd.DataFrame,
    resampling_core: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply pre-specified support and direction rules without post-hoc pruning."""
    alpha_summary = _alpha_summary(metrics, contrasts)
    common_summary = _common_support_summary(common_core)
    baseline_passed = bool(not baseline.empty and baseline["passed"].all())
    affine = manufactured[manufactured["field_class"] == "affine"]
    affine_max_error = float(affine["gradient_nrmse"].max()) if len(affine) else np.inf
    affine_passed = bool(len(affine) and affine_max_error <= AFFINE_NUMERICAL_TOLERANCE)
    support_passed = bool(alpha_summary["support_passed_all_cells"].all())
    direction_passed = bool(alpha_summary["all_directions_350_gt_400"].all())
    common_support_passed = bool(common_summary["all_directions_350_gt_400"].all())

    if not baseline_passed:
        status = "canonical_alpha_0_reproduction_failed"
    elif not affine_passed:
        status = "withdraw_all_q_comparisons_due_to_affine_implementation_failure"
    elif not support_passed:
        status = "insufficient_support_under_alpha"
    elif not direction_passed or not common_support_passed:
        status = "distance_exponent_dependent"
    else:
        status = "directionally_consistent_over_predefined_weight_exponents"

    resampling_rows = []
    for alpha in ALPHA_SPECS:
        block = resampling_core[(resampling_core["alpha"] == alpha.value) & (resampling_core["region"] == "all")]
        directions = sorted(block["direction_350_400"].dropna().unique().tolist())
        resampling_rows.append(
            {
                "alpha_label": alpha.label,
                "alpha": alpha.value,
                "replicates": int(len(block)),
                "directions": directions,
                "all_directions_350_gt_400": bool(len(block) and all(value == "350>400" for value in directions)),
            }
        )

    payload = {
        "analysis_scope": "six 0.70 s exports after exact-coordinate consolidation; first-order WLS, k=8-50, kappa<=100",
        "predefined_alpha_values": [spec.value for spec in ALPHA_SPECS],
        "canonical_alpha": 0.0,
        "baseline_reproduction_passed": baseline_passed,
        "affine_exactness": {
            "max_gradient_nrmse": affine_max_error,
            "tolerance": AFFINE_NUMERICAL_TOLERANCE,
            "passed": affine_passed,
        },
        "support_rule": f"at least {MIN_REGION_POINTS} full-pool valid points in every power-k cell",
        "alpha_specific_summary": alpha_summary.to_dict(orient="records"),
        "common_support_summary": common_summary.to_dict(orient="records"),
        "resampling_summary": resampling_rows,
        "final_q_claim_status": status,
        "required_manuscript_boundary": "This audit tests a fixed distance-weight exponent range. It does not select a universally optimal exponent, validate unavailable solver gradients, establish CFD physical truth, or remove the temporal limitation of the 0.70 s snapshot.",
    }
    return alpha_summary.merge(common_summary, on=["alpha_label", "alpha"], how="left"), payload


def write_json(path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

