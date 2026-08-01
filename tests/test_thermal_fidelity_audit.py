from __future__ import annotations

import unittest

from scripts.thermal_fidelity_audit.gates import decision_payload, gate_audit
from scripts.thermal_fidelity_audit.running_log import parse_running_log
from scripts.thermal_fidelity_audit.sources import load_inputs
from scripts.thermal_fidelity_audit.temperature_tail import temperature_extreme_context, temperature_tail_metrics, temperature_tail_sensitivity


class ThermalFidelityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_inputs()
        cls.progress, cls.events, cls.log_summary = parse_running_log()
        cls.tail, cls.cache = temperature_tail_metrics(cls.inputs.snapshots)
        cls.sensitivity = temperature_tail_sensitivity(cls.cache)
        cls.gates = gate_audit(cls.inputs.phase_configuration, cls.progress, cls.events, cls.log_summary)
        cls.decision = decision_payload(cls.inputs.phase_configuration, cls.progress, cls.events, cls.log_summary, cls.tail, cls.gates)

    def test_input_manifest_and_phase_configuration_are_complete(self) -> None:
        self.assertEqual(len(self.inputs.manifest), 32)
        values = self.inputs.phase_configuration.set_index("field_id")["value"]
        self.assertEqual(int(values["phase_change_enabled"]), 1)
        self.assertEqual(float(values["saturation_temperature_K"]), 3134.0)
        self.assertEqual(int(values["recoil_pressure_enabled"]), 0)

    def test_native_300w_log_facts_are_reproduced_without_acceptance_inference(self) -> None:
        summary = self.log_summary.set_index("metric")["value"]
        self.assertEqual(len(self.progress), 171)
        self.assertEqual(len(self.events), 2)
        self.assertTrue(bool(self.events["restart_with_smaller_timestep_reported"].all()))
        self.assertEqual(int(summary["completion_cycle"]), 18495)
        self.assertTrue(bool(summary["normal_completion_reported"]))
        residual_gate = self.gates.loc[self.gates["gate_id"] == "configured_residual_acceptance_available", "passed"].iloc[0]
        self.assertFalse(bool(residual_gate))

    def test_sparse_extreme_and_tail_sensitivity_remain_audit_only(self) -> None:
        context = temperature_extreme_context(self.cache)
        maximum = context.loc[(context["time_s"] == 0.50) & (context["power_W"] == 350)].iloc[0]
        self.assertAlmostEqual(float(maximum["T_max_K"]), 9981.54)
        self.assertEqual(int(maximum["tied_maximum_exported_coordinates"]), 1)
        self.assertEqual(int(maximum["n_unique_T_gt_5000"]), 8)
        screen = self.sensitivity.loc[(self.sensitivity["time_s"] == 0.50) & (self.sensitivity["power_W"] == 350) & (self.sensitivity["sensitivity_condition"] == "exclude_T_gt_5000_K")].iloc[0]
        self.assertAlmostEqual(float(screen["mean_delta_from_unfiltered_percent"]), -6.09784487, places=6)
        self.assertAlmostEqual(float(screen["median_delta_from_unfiltered_percent"]), -0.5404931, places=6)
        self.assertEqual(self.decision["temperature_tail"]["status"], "audit_only")
        self.assertEqual(self.decision["current_temperature_field_physical_fidelity"], "not_supported")


if __name__ == "__main__":
    unittest.main()
