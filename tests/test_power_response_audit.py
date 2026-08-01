from __future__ import annotations

import unittest

from scripts.power_response_audit.inputs import load_inputs
from scripts.power_response_audit.pairwise_context import build_pairwise_snapshot_context
from scripts.power_response_audit.snapshot_metrics import canonical_metric_frame


class PowerResponseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_inputs()
        cls.canonical = canonical_metric_frame(cls.inputs.canonical, cls.inputs.thermal_tail)

    def test_pairwise_context_contains_complete_unordered_grid(self) -> None:
        context = build_pairwise_snapshot_context(self.canonical)
        self.assertEqual(len(context), 60)
        self.assertEqual(
            len(context[["lower_power_W", "higher_power_W"]].drop_duplicates()), 15
        )
        self.assertEqual(context["metric_id"].nunique(), 4)
        self.assertFalse(context["metric_id"].str.contains("Q", case=False).any())

    def test_200_400_context_reports_mixed_snapshot_descriptors(self) -> None:
        context = build_pairwise_snapshot_context(self.canonical)
        pair = context.loc[
            (context["lower_power_W"] == 200) & (context["higher_power_W"] == 400)
        ].set_index("metric_id")
        self.assertAlmostEqual(pair.loc["temperature_median_full_pool_K", "lower_value"], 1763.09)
        self.assertAlmostEqual(pair.loc["temperature_median_full_pool_K", "higher_value"], 1798.53)
        self.assertEqual(pair.loc["temperature_median_full_pool_K", "direction"], "higher_power_greater")
        self.assertAlmostEqual(
            pair.loc["temperature_mean_full_pool_K", "lower_value"], 1987.651643, places=6
        )
        self.assertAlmostEqual(
            pair.loc["temperature_mean_full_pool_K", "higher_value"], 2133.419023, places=6
        )
        self.assertEqual(pair.loc["temperature_mean_full_pool_K", "direction"], "higher_power_greater")
        self.assertAlmostEqual(pair.loc["velocity_max_full_pool_mps", "lower_value"], 0.230529)
        self.assertAlmostEqual(pair.loc["velocity_max_full_pool_mps", "higher_value"], 0.185829)
        self.assertEqual(pair.loc["velocity_max_full_pool_mps", "direction"], "lower_power_greater")
        self.assertEqual(pair.loc["velocity_max_full_pool_mps", "interpretation_status"], "audit_only")
        self.assertAlmostEqual(
            pair.loc["velocity_mean_interface_mps", "lower_value"], 0.0412544601, places=9
        )
        self.assertAlmostEqual(
            pair.loc["velocity_mean_interface_mps", "higher_value"], 0.0538486173, places=9
        )
        self.assertEqual(pair.loc["velocity_mean_interface_mps", "direction"], "higher_power_greater")

    def test_pairwise_context_rejects_an_incomplete_power_grid(self) -> None:
        incomplete = self.canonical.iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "Expected 24 canonical metric rows"):
            build_pairwise_snapshot_context(incomplete)


if __name__ == "__main__":
    unittest.main()
