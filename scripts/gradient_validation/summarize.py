from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AFFINE_NUMERICAL_TOLERANCE, EXPECTED_POWERS, K_VALUES, RESAMPLE_COUNT


def validate_outputs(
    manifest: pd.DataFrame,
    manufactured: pd.DataFrame,
    geometry: pd.DataFrame,
    model_metrics: pd.DataFrame,
    resamples: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []
    checks.append(
        {
            "check": "input_manifest",
            "expected": len(EXPECTED_POWERS),
            "observed": len(manifest),
            "passed": len(manifest) == len(EXPECTED_POWERS),
        }
    )
    expected_manufactured = len(EXPECTED_POWERS) * len(K_VALUES) * 9 * 2
    checks.append(
        {
            "check": "manufactured_grid",
            "expected": expected_manufactured,
            "observed": len(manufactured),
            "passed": len(manufactured) == expected_manufactured,
        }
    )
    expected_geometry = int(manifest["unique_coordinate_rows"].sum()) * len(K_VALUES)
    checks.append(
        {
            "check": "geometry_grid",
            "expected": expected_geometry,
            "observed": len(geometry),
            "passed": len(geometry) == expected_geometry,
        }
    )
    expected_model_metrics = len(EXPECTED_POWERS) * 36 * 2 * 4 * 2
    checks.append(
        {
            "check": "model_order_grid",
            "expected": expected_model_metrics,
            "observed": len(model_metrics),
            "passed": len(model_metrics) == expected_model_metrics,
        }
    )
    expected_resamples = len(EXPECTED_POWERS) * 2 * RESAMPLE_COUNT
    checks.append(
        {
            "check": "resampling_grid",
            "expected": expected_resamples,
            "observed": len(resamples),
            "passed": len(resamples) == expected_resamples,
        }
    )
    return pd.DataFrame(checks)


def manufactured_summary(manufactured: pd.DataFrame) -> pd.DataFrame:
    return (
        manufactured.groupby(["field_id", "field_class", "feature_scale_mm", "region"], dropna=False, as_index=False)
        .agg(
            gradient_nrmse_median=("gradient_nrmse", "median"),
            gradient_nrmse_p90=("gradient_nrmse", lambda value: value.quantile(0.90)),
            q_nrmse_median=("q_nrmse", "median"),
            q_sign_accuracy_margin_median=("q_sign_accuracy_margin", "median"),
            q_sign_accuracy_margin_min=("q_sign_accuracy_margin", "min"),
            valid_fraction_min=("valid_fraction", "min"),
        )
        .sort_values(["field_class", "field_id", "feature_scale_mm", "region"])
    )


def decision_payload(
    manufactured: pd.DataFrame,
    model_summary: pd.DataFrame,
    resampling_core: pd.DataFrame,
    native_status: pd.DataFrame,
) -> dict[str, object]:
    affine = manufactured[manufactured["field_class"] == "affine"]
    affine_max_error = float(affine["gradient_nrmse"].max())
    affine_exact = bool(affine_max_error <= AFFINE_NUMERICAL_TOLERANCE)
    eligible_model = model_summary[model_summary["evidence_eligible"]]
    model_dependent = bool((eligible_model["status"] == "model_order_dependent").any())
    model_not_comparable = bool((eligible_model["status"] == "not_comparable").any())
    resampling_rows = []
    direction_stable = True
    for region, block in resampling_core.groupby("region", sort=True):
        directions = sorted(block["direction_350_400"].unique())
        direction_stable = direction_stable and len(directions) == 1
        resampling_rows.append(
            {
                "region": region,
                "replicates": int(len(block)),
                "directions": directions,
                "direction_stable": len(directions) == 1,
            }
        )
    native_available = bool((native_status.get("status", pd.Series(dtype=str)) == "matched").any())
    if not affine_exact:
        q_claim_status = "withdraw_all_q_comparisons_due_to_affine_implementation_failure"
    elif model_dependent or model_not_comparable:
        q_claim_status = "withdraw_affected_q_comparisons_due_to_model_order_dependence"
    elif not direction_stable:
        q_claim_status = "withdraw_affected_q_comparisons_due_to_neighbour_subset_sensitivity"
    else:
        q_claim_status = "retain_as_scale_limited_reconstruction_proxy_only"
    return {
        "analysis_scope": "six 0.70 s exports after exact-coordinate consolidation",
        "native_solver_gradient_reference_available": native_available,
        "affine_numerical_exactness": {
            "max_gradient_nrmse": affine_max_error,
            "tolerance": AFFINE_NUMERICAL_TOLERANCE,
            "passed": affine_exact,
        },
        "nonlinear_interpretation": "Manufactured fields test the sampled point geometry and algorithm. They do not validate the original CFD gradients, free-surface physics, or temporal persistence.",
        "model_order_results": model_summary.to_dict(orient="records"),
        "neighbour_subset_results": resampling_rows,
        "q_claim_status": q_claim_status,
        "required_manuscript_boundary": "Q remains a reconstruction-dependent topology proxy; direct fidelity to solver gradients is not established from the available CSV exports.",
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
