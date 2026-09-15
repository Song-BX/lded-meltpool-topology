from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BASELINE_Q_METRICS, TEMPORAL_DECISION, WEIGHT_DECISION


def baseline_reproduction(metrics: pd.DataFrame) -> pd.DataFrame:
    expected = pd.read_csv(BASELINE_Q_METRICS)
    expected = expected[(expected["region"] == "all") & (expected["threshold"] == "Q>0")][
        ["power_W", "kNN", "q_fraction"]
    ].rename(columns={"q_fraction": "expected_q_fraction"})
    observed = metrics[
        (metrics["context"] == "canonical")
        & (metrics["region"] == "all")
        & (metrics["descriptor"] == "Q")
    ][["power_W", "kNN", "positive_fraction"]].rename(
        columns={"positive_fraction": "observed_q_fraction"}
    )
    result = expected.merge(observed, on=["power_W", "kNN"], how="outer", validate="one_to_one")
    result["absolute_difference"] = np.abs(result["expected_q_fraction"] - result["observed_q_fraction"])
    result["passed"] = result["absolute_difference"].le(1e-12)
    return result.sort_values(["power_W", "kNN"]).reset_index(drop=True)


def decision_payload(
    manifest: pd.DataFrame,
    baseline: pd.DataFrame,
    canonical_metrics: pd.DataFrame,
    canonical_agreement: pd.DataFrame,
    manufactured: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    weight = json.loads(WEIGHT_DECISION.read_text(encoding="utf-8"))
    temporal = json.loads(TEMPORAL_DECISION.read_text(encoding="utf-8"))
    q_omega = canonical_agreement[
        (canonical_agreement["first_descriptor"] == "Q")
        & (canonical_agreement["second_descriptor"] == "omega_normalized")
    ]
    q_lambda = canonical_agreement[
        (canonical_agreement["first_descriptor"] == "Q")
        & (canonical_agreement["second_descriptor"] == "lambda2")
    ]
    affine = manufactured[manufactured["field_class"] == "affine"]
    summary = pd.DataFrame(
        [
            {"check": "input_manifest", "expected": 6, "observed": len(manifest), "passed": len(manifest) == 6},
            {"check": "canonical_q_reproduction", "expected": 258, "observed": int(baseline["passed"].sum()), "passed": bool(baseline["passed"].all())},
            {"check": "canonical_q_omega_exact_classification", "expected": len(q_omega), "observed": int(np.isclose(q_omega["agreement_fraction"], 1.0, rtol=0.0, atol=0.0).sum()), "passed": bool(np.isclose(q_omega["agreement_fraction"], 1.0, rtol=0.0, atol=0.0).all())},
            {"check": "manufactured_descriptor_grid", "expected": 3 * 6 * 9 * 43 * 2 * 3, "observed": len(manufactured), "passed": len(manufactured) == 3 * 6 * 9 * 43 * 2 * 3},
            {"check": "affine_descriptor_records", "expected": 3 * 6 * 3 * 43 * 2 * 3, "observed": len(affine), "passed": len(affine) == 3 * 6 * 3 * 43 * 2 * 3},
        ]
    )
    payload = {
        "analysis_scope": "six 0.70 s exports after exact-coordinate consolidation; shared first-order WLS tensor, k=8-50",
        "descriptor_definitions": {
            "lambda2": "middle ordered eigenvalue of S^2 + Omega^2; lambda2 < 0 is the sign classification",
            "omega_normalized": "Omega_norm2 / (Omega_norm2 + S_norm2) for nonzero tensor energy; finite zero tensors are neutral at 0.5",
            "q_omega_identity": "Omega_N - 0.5 = Q / (Omega_norm2 + S_norm2) for finite nonzero tensors",
        },
        "baseline_reproduction_passed": bool(baseline["passed"].all()),
        "q_omega_exact_agreement": {
            "all_cells_passed": bool(np.isclose(q_omega["agreement_fraction"], 1.0, rtol=0.0, atol=0.0).all()),
            "max_identity_abs_error": float(q_omega["q_omega_identity_max_abs_error"].max()),
        },
        "q_lambda_agreement": {
            "minimum": float(q_lambda["agreement_fraction"].min()),
            "maximum": float(q_lambda["agreement_fraction"].max()),
            "is_independent_validation": False,
        },
        "existing_blocking_gates": {
            "weight_exponent_affine_exactness_passed": bool(weight["affine_exactness"]["passed"]),
            "weight_exponent_affine_max_gradient_nrmse": weight["affine_exactness"]["max_gradient_nrmse"],
            "temporal_pairwise_persistence_passed": bool(temporal["core_350_400_contrasts_pass"]),
            "native_solver_gradient_reference_available": False,
        },
        "final_claim_status": "audit_only",
        "required_interpretation": "The descriptors share one reconstructed WLS tensor. Agreement is an implementation and reporting check, not independent cross-validation or solver/physical validation.",
        "prohibited_interpretations": [
            "validated vortex structure",
            "independent cross-validation",
            "comparative vortex evidence",
            "physical mechanism evidence",
        ],
    }
    return summary, payload


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
