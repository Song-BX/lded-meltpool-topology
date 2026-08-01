from __future__ import annotations

import unittest
import json
from pathlib import Path

import pandas as pd

from scripts.claim_classification.classifier import (
    classify_q_status,
    classify_spatial_status,
    classify_velocity_extreme_status,
)
from scripts.claim_classification.gates import Gate


def _gate(gate_id: str, passed: bool) -> Gate:
    return Gate("test", gate_id, True, passed, "test", "test", str(passed), "failed")


class ClaimClassificationTests(unittest.TestCase):
    def _gates(self, **overrides: bool) -> list[Gate]:
        defaults = {
            "support_eligibility": True,
            "canonical_reproduction": True,
            "aggregation_consistency": True,
            "knn_directional_stability": True,
            "model_order_consistency": True,
            "conditioning_consistency": True,
            "weight_exponent_affine_exactness": True,
            "temporal_pairwise_persistence": True,
        }
        defaults.update(overrides)
        return [_gate(gate_id, passed) for gate_id, passed in defaults.items()]

    def test_support_failure_is_insufficient_support(self) -> None:
        self.assertEqual(
            classify_q_status(self._gates(support_eligibility=False), "insufficient_support", "excluded"),
            "insufficient_support",
        )

    def test_sparse_audit_region_is_retained_as_audit_only(self) -> None:
        self.assertEqual(
            classify_q_status(self._gates(support_eligibility=False), "insufficient_support", "audit_only"),
            "audit_only",
        )

    def test_alpha_failure_blocks_comparative_promotion(self) -> None:
        self.assertEqual(
            classify_q_status(self._gates(weight_exponent_affine_exactness=False), "evidence_eligible", "primary_evidence"),
            "audit_only",
        )

    def test_temporal_failure_leaves_snapshot_descriptor(self) -> None:
        self.assertEqual(
            classify_q_status(self._gates(temporal_pairwise_persistence=False), "evidence_eligible", "primary_evidence"),
            "snapshot_local_descriptor",
        )

    def test_all_required_gates_allow_comparative_evidence(self) -> None:
        self.assertEqual(
            classify_q_status(self._gates(), "evidence_eligible", "primary_evidence"),
            "comparative_evidence",
        )

    def test_explicit_spatial_exclusion_has_priority(self) -> None:
        self.assertEqual(
            classify_spatial_status("exclude_spatial_geometric_comparisons"),
            "excluded",
        )

    def test_velocity_extreme_never_promotes_beyond_snapshot_descriptor(self) -> None:
        gates = [_gate("canonical_vmax_reproduction", True), _gate("native_solver_history_health", True)]
        self.assertEqual(classify_velocity_extreme_status(gates), "snapshot_local_descriptor")
        self.assertEqual(
            classify_velocity_extreme_status([_gate("canonical_vmax_reproduction", True), _gate("native_solver_history_health", False)]),
            "audit_only",
        )

    def test_central_velocity_separation_is_not_supported(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = pd.read_csv(root / "图" / "claim_classification" / "claim_input_manifest.csv")
        registry = pd.read_csv(root / "图" / "claim_classification" / "claim_registry.csv")
        decision = json.loads(
            (root / "图" / "claim_classification" / "claim_classification_decision.json").read_text(
                encoding="utf-8"
            )
        )
        row = registry.loc[registry["claim_id"] == "velocity_distribution_separation"].iloc[0]
        self.assertIn("velocity_distribution", set(manifest["input_key"]))
        self.assertEqual(row["final_status"], "not_supported")
        self.assertIn("central_iqr_separation", row["failed_gates"])
        self.assertEqual(decision["input_validation"]["source_count"], 18)

    def test_thermal_fidelity_records_are_bounded(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = pd.read_csv(root / "图" / "claim_classification" / "claim_registry.csv")
        config = registry.loc[registry["claim_id"] == "native_phase_model_configuration"].iloc[0]
        high_tail = registry.loc[registry["claim_id"] == "temperature_high_tail"].iloc[0]
        temperature = registry.loc[registry["claim_id"] == "current_temperature_field_physical_fidelity"].iloc[0]
        self.assertEqual(config["final_status"], "direct_observation")
        self.assertEqual(high_tail["final_status"], "audit_only")
        self.assertEqual(temperature["final_status"], "not_supported")

    def test_six_case_pairwise_context_is_hashed_and_scope_bound(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = pd.read_csv(root / "图" / "claim_classification" / "claim_input_manifest.csv")
        registry = pd.read_csv(root / "图" / "claim_classification" / "claim_registry.csv")
        pairwise = manifest.loc[manifest["input_key"] == "power_response_pairwise_context"].iloc[0]
        response = registry.loc[registry["claim_id"] == "six_case_response"].iloc[0]
        self.assertEqual(int(pairwise["expected_rows"]), 60)
        self.assertTrue(bool(pairwise["validation_passed"]))
        self.assertEqual(response["final_status"], "snapshot_local_descriptor")
        self.assertIn("450 W", response["prohibited_interpretation"])

    def test_cross_context_applicability_is_not_supported(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = pd.read_csv(root / "图" / "claim_classification" / "claim_registry.csv")
        row = registry.loc[registry["claim_id"] == "beyond_lded_applicability"].iloc[0]
        self.assertEqual(row["final_status"], "not_supported")
        self.assertIn("external", row["failed_gates"])

    def test_current_cfd_fidelity_is_not_supported_but_prior_context_is_recorded(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = pd.read_csv(root / "图" / "claim_classification" / "claim_registry.csv")
        prior = registry.loc[registry["claim_id"] == "prior_model_validation_context"].iloc[0]
        current = registry.loc[registry["claim_id"] == "current_cfd_physical_fidelity"].iloc[0]
        self.assertEqual(prior["final_status"], "direct_observation")
        self.assertEqual(prior["retention_role"], "context_only")
        self.assertEqual(current["final_status"], "not_supported")
        self.assertIn("current_matched_experimental_observable", current["failed_gates"])


if __name__ == "__main__":
    unittest.main()
