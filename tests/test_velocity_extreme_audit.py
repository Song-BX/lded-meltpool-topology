from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.velocity_extreme_audit.health import evaluate_health
from scripts.velocity_extreme_audit.velocity import vector_speed


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "图" / "velocity_extreme_audit"


class VelocityExtremeAuditTests(unittest.TestCase):
    def test_vector_speed_closure_formula(self) -> None:
        frame = pd.DataFrame({"u": [3.0, 0.0], "v": [4.0, 0.0], "w": [0.0, 5.0]})
        np.testing.assert_allclose(vector_speed(frame), [5.0, 5.0])

    def test_missing_native_history_cannot_pass_health_gate(self) -> None:
        output = evaluate_health(pd.DataFrame(), ("solver-history mapping missing",))
        self.assertTrue((output["status"] == "not_available").all())
        self.assertFalse(output["passed"].any())

    def test_frozen_velocity_extreme_outputs(self) -> None:
        required = {
            "velocity_extreme_input_manifest.csv",
            "velocity_quantiles.csv",
            "canonical_vmax_reproduction.csv",
            "peak_provenance.csv",
            "velocity_closure_audit.csv",
            "velocity_temporal_aggregation_audit.csv",
            "normalised_solver_history.csv",
            "solver_health_gate_audit.csv",
            "velocity_extreme_summary.csv",
            "velocity_extreme_decision.json",
        }
        self.assertTrue(all((AUDIT_DIR / name).is_file() for name in required))
        manifest = pd.read_csv(AUDIT_DIR / "velocity_extreme_input_manifest.csv")
        quantiles = pd.read_csv(AUDIT_DIR / "velocity_quantiles.csv")
        reproduction = pd.read_csv(AUDIT_DIR / "canonical_vmax_reproduction.csv")
        provenance = pd.read_csv(AUDIT_DIR / "peak_provenance.csv")
        aggregation = pd.read_csv(AUDIT_DIR / "velocity_temporal_aggregation_audit.csv")
        health = pd.read_csv(AUDIT_DIR / "solver_health_gate_audit.csv")
        decision = json.loads((AUDIT_DIR / "velocity_extreme_decision.json").read_text(encoding="utf-8"))

        self.assertEqual(int((manifest["input_kind"] == "point_cloud").sum()), 30)
        self.assertEqual(len(quantiles), 30)
        self.assertTrue(reproduction["passed"].all())
        self.assertTrue(aggregation["direction_exported_magnitude"].eq("350>400").all())
        self.assertTrue(aggregation["definition_direction_matches"].all())
        self.assertEqual(int(provenance.loc[provenance["power_W"] == 350, "tied_peak_coordinates"].max()), 3)
        self.assertTrue(health["status"].eq("not_available").all())
        self.assertEqual(decision["final_status"], "audit_only")
        self.assertTrue(decision["gates"]["solver_history_not_available"])
        self.assertAlmostEqual(decision["canonical_values"]["relative_drop_350_to_400_pct"], 49.1399200269, places=8)


if __name__ == "__main__":
    unittest.main()

