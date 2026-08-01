from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.transferability_scope_audit.controls import build_context_bound_controls
from scripts.transferability_scope_audit.gates import build_external_gate_audit
from scripts.transferability_scope_audit.run_analysis import run
from scripts.transferability_scope_audit.sources import load_sources
from scripts.transferability_scope_audit.summary import decision_payload


class TransferabilityScopeAuditTests(unittest.TestCase):
    def test_configuration_sources_are_hashed_and_complete(self) -> None:
        values, manifest_rows = load_sources()
        self.assertEqual(len(manifest_rows), 4)
        self.assertTrue(all(row["validation_passed"] for row in manifest_rows))
        self.assertTrue(all(len(str(row["sha256"])) == 64 for row in manifest_rows))
        self.assertIn("COLUMN_MAP", values["point_cloud_schema"])
        self.assertIn("K_VALUES", values["robustness_configuration"])

    def test_all_required_controls_are_context_bound(self) -> None:
        values, _ = load_sources()
        controls = build_context_bound_controls(values)
        expected = {
            "flow3d_csv_schema_and_units",
            "coordinate_consolidation_and_aggregation",
            "neighbourhood_support_scale",
            "wls_model_weight_and_conditioning",
            "region_mask_semantics",
            "support_and_pooled_threshold_policy",
            "temporal_and_power_case_design",
            "audit_and_claim_governance",
        }
        self.assertEqual({str(row["control_id"]) for row in controls}, expected)
        self.assertTrue(all(not bool(row["portable_default"]) for row in controls))

    def test_external_gates_block_cross_context_applicability(self) -> None:
        values, _ = load_sources()
        controls = build_context_bound_controls(values)
        gates = build_external_gate_audit()
        decision = decision_payload(controls, gates)
        self.assertEqual(decision["cross_context_applicability"], "not_supported")
        self.assertEqual(decision["portable_parameter_defaults"], [])
        self.assertTrue(all(not bool(row["passed"]) for row in gates))
        self.assertEqual(len(decision["failed_external_gates"]), len(gates))

    def test_runner_writes_the_complete_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run(output)
            manifest = pd.read_csv(output / "transferability_input_manifest.csv")
            controls = pd.read_csv(output / "context_bound_controls.csv")
            gates = pd.read_csv(output / "transferability_gate_audit.csv")
            decision = json.loads((output / "transferability_decision.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest), 4)
        self.assertEqual(len(controls), 8)
        self.assertEqual(len(gates), 5)
        self.assertEqual(decision["cross_context_applicability"], "not_supported")


if __name__ == "__main__":
    unittest.main()
