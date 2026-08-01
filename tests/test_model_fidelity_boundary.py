from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.model_fidelity_boundary.alignment import build_alignment
from scripts.model_fidelity_boundary.gates import build_gate_audit
from scripts.model_fidelity_boundary.run_analysis import run
from scripts.model_fidelity_boundary.sources import load_inputs
from scripts.model_fidelity_boundary.summary import decision_payload


class ModelFidelityBoundaryTests(unittest.TestCase):
    def test_inputs_and_prior_context_are_auditable(self) -> None:
        values, manifest = load_inputs()
        record = values["prior_validation_record"]
        self.assertEqual(len(manifest), 3)
        self.assertTrue(all(row["validation_passed"] for row in manifest))
        self.assertEqual(record["doi"], "10.1016/j.ijthermalsci.2020.106579")
        self.assertTrue(record["metadata_verification"]["title_and_doi_verified"])

    def test_alignment_does_not_promote_undocumented_matches(self) -> None:
        values, _ = load_inputs()
        alignment = build_alignment(values["prior_validation_record"])
        statuses = {str(row["alignment_status"]) for row in alignment}
        self.assertEqual(len(alignment), 9)
        self.assertIn("partial_match", statuses)
        self.assertIn("not_documented", statuses)
        self.assertNotIn("exact_match", statuses)

    def test_missing_current_evidence_blocks_physical_fidelity(self) -> None:
        values, _ = load_inputs()
        record = values["prior_validation_record"]
        alignment = build_alignment(record)
        gates = build_gate_audit(record)
        decision = decision_payload(record, alignment, gates)
        current = [row for row in gates if row["required_for_current_fidelity"]]
        self.assertTrue(gates[0]["passed"])
        self.assertTrue(all(not bool(row["passed"]) for row in current))
        self.assertEqual(decision["current_cfd_physical_fidelity"], "not_supported")
        self.assertEqual(len(decision["failed_current_fidelity_gates"]), 4)

    def test_runner_writes_all_public_audit_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run(output)
            manifest = pd.read_csv(output / "model_fidelity_input_manifest.csv")
            alignment = pd.read_csv(output / "model_alignment_audit.csv")
            gates = pd.read_csv(output / "cfd_fidelity_gate_audit.csv")
            decision = json.loads((output / "model_fidelity_decision.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest), 3)
        self.assertEqual(len(alignment), 9)
        self.assertEqual(len(gates), 5)
        self.assertEqual(decision["current_cfd_physical_fidelity"], "not_supported")


if __name__ == "__main__":
    unittest.main()
