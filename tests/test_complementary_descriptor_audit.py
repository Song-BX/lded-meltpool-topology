from __future__ import annotations

import json
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "图" / "complementary_descriptor_audit"


class ComplementaryDescriptorAuditTests(unittest.TestCase):
    def test_frozen_audit_outputs_satisfy_required_invariants(self) -> None:
        required = {
            "descriptor_input_manifest.csv",
            "descriptor_metrics.csv",
            "descriptor_agreement.csv",
            "descriptor_sensitivity.csv",
            "descriptor_manufactured_metrics.csv",
            "complementary_descriptor_summary.csv",
            "complementary_descriptor_decision.json",
        }
        self.assertTrue(all((AUDIT_DIR / name).is_file() for name in required))

        decision = json.loads((AUDIT_DIR / "complementary_descriptor_decision.json").read_text(encoding="utf-8"))
        summary = pd.read_csv(AUDIT_DIR / "complementary_descriptor_summary.csv")

        self.assertTrue(bool(summary["passed"].all()))
        self.assertTrue(decision["baseline_reproduction_passed"])
        self.assertTrue(decision["q_omega_exact_agreement"]["all_cells_passed"])
        self.assertLessEqual(decision["q_omega_exact_agreement"]["max_identity_abs_error"], 2e-15)
        self.assertFalse(decision["q_lambda_agreement"]["is_independent_validation"])
        self.assertEqual(decision["final_claim_status"], "audit_only")
        self.assertFalse(decision["existing_blocking_gates"]["weight_exponent_affine_exactness_passed"])
        self.assertFalse(decision["existing_blocking_gates"]["temporal_pairwise_persistence_passed"])
        self.assertFalse(decision["existing_blocking_gates"]["native_solver_gradient_reference_available"])


if __name__ == "__main__":
    unittest.main()
