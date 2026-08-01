from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.velocity_distribution_overlap_audit.metrics import calculate_pair_metrics


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "图" / "velocity_distribution_overlap_audit"


class VelocityDistributionOverlapAuditTests(unittest.TestCase):
    def test_pair_metrics_report_iqr_containment_and_tail_counts(self) -> None:
        metrics = calculate_pair_metrics([0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 1.5, 2.0, 2.5, 3.0])
        self.assertTrue(metrics["iqr_overlap_observed"])
        self.assertTrue(metrics["one_iqr_contained_in_other"])
        self.assertEqual(metrics["contained_iqr"], "400_within_350")
        self.assertEqual(metrics["n_350_gt_400_vmax"], 1)
        self.assertEqual(metrics["n_400_gt_350_vmax"], 0)

    def test_frozen_overlap_outputs(self) -> None:
        required = {
            "velocity_distribution_input_manifest.csv",
            "velocity_distribution_overlap_audit.csv",
            "velocity_distribution_overlap_summary.csv",
            "velocity_distribution_overlap_decision.json",
        }
        self.assertTrue(all((AUDIT_DIR / name).is_file() for name in required))
        manifest = pd.read_csv(AUDIT_DIR / "velocity_distribution_input_manifest.csv")
        audit = pd.read_csv(AUDIT_DIR / "velocity_distribution_overlap_audit.csv")
        decision = json.loads(
            (AUDIT_DIR / "velocity_distribution_overlap_decision.json").read_text(encoding="utf-8")
        )
        canonical = audit.loc[
            (audit["audit_context"] == "aggregation_sensitivity")
            & (np.isclose(audit["time_s"], 0.70))
            & (audit["aggregation_strategy"] == "mean_all_records")
        ].iloc[0]

        self.assertEqual(len(manifest), 30)
        self.assertEqual(len(audit), 8)
        self.assertTrue(bool(canonical["iqr_overlap_observed"]))
        self.assertTrue(bool(canonical["one_iqr_contained_in_other"]))
        self.assertEqual(canonical["contained_iqr"], "400_within_350")
        self.assertAlmostEqual(float(canonical["p25_350_mps"]), 0.00554508, places=8)
        self.assertAlmostEqual(float(canonical["p75_400_mps"]), 0.0447076, places=8)
        self.assertAlmostEqual(float(canonical["p99_350_mps"]), 0.154952, places=8)
        self.assertEqual(int(canonical["n_350_gt_400_vmax"]), 3)
        self.assertEqual(
            decision["whole_pool_distribution_separation"],
            "not_supported_by_central_distribution",
        )
        self.assertEqual(decision["vmax_role"], "sparse_peak_audit_only")


if __name__ == "__main__":
    unittest.main()
