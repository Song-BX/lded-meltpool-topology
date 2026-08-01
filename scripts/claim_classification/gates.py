from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .inputs import LoadedInputs


@dataclass(frozen=True)
class Gate:
    claim_id: str
    gate_id: str
    required_for_promotion: bool
    passed: bool
    source_artifact: str
    source_pointer: str
    observed_value: str
    failure_reason: str

    def row(self) -> dict[str, object]:
        return asdict(self)


def _gate(
    claim_id: str,
    gate_id: str,
    passed: bool,
    source_artifact: str,
    source_pointer: str,
    observed_value: object,
    failure_reason: str,
    required: bool = True,
) -> Gate:
    return Gate(
        claim_id=claim_id,
        gate_id=gate_id,
        required_for_promotion=required,
        passed=bool(passed),
        source_artifact=source_artifact,
        source_pointer=source_pointer,
        observed_value=str(observed_value),
        failure_reason="" if passed else failure_reason,
    )


def _model_order_lookup(inputs: LoadedInputs) -> dict[tuple[str, str], dict[str, Any]]:
    records = inputs.json_data["gradient_validation"]["model_order_results"]
    return {(str(row["region"]), str(row["threshold"])): row for row in records}


def q_gate_records(row: pd.Series, inputs: LoadedInputs) -> list[Gate]:
    claim_id = f"q_{row.region}_{row.threshold}".replace(">", "gt").replace("_", "-")
    export = inputs.json_data["export"]
    temporal = inputs.json_data["temporal"]
    weight = inputs.json_data["weight_exponent"]
    conditioning = inputs.json_data["conditioning"]
    model_order = _model_order_lookup(inputs).get((str(row.region), str(row.threshold)))
    support = bool(row.evidence_eligible)
    directionally_stable = str(row.classification) == "directionally_stable"
    model_order_passed = bool(model_order and model_order["status"] == "order_consistent_over_compared_k")
    conditioning_passed = bool(
        row.region == "all"
        and row.threshold == "Q>0"
        and conditioning["final_q_claim_status"]
        == "directionally_consistent_over_supported_predefined_cutoffs"
    )
    alpha_passed = bool(weight["affine_exactness"]["passed"])
    canonical_passed = bool(
        temporal["baseline_reproducibility_passed"] and weight["baseline_reproduction_passed"]
    )
    aggregation_passed = bool(
        export["aggregation_sensitivity"]["k25_core_directions_all_match_canonical"]
    )
    return [
        _gate(
            claim_id,
            "support_eligibility",
            support,
            "knn_evidence_eligibility.csv",
            f"{row.region}/{row.threshold}/evidence_eligible",
            row.evidence_status,
            str(row.exclusion_reason) or "The 43-k support policy was not passed.",
        ),
        _gate(
            claim_id,
            "canonical_reproduction",
            canonical_passed,
            "temporal_validation_decision.json; weight_exponent_decision.json",
            "baseline_reproducibility_passed; baseline_reproduction_passed",
            canonical_passed,
            "A retained canonical baseline did not reproduce.",
        ),
        _gate(
            claim_id,
            "aggregation_consistency",
            aggregation_passed,
            "export_diagnostics_decision.json",
            "aggregation_sensitivity.k25_core_directions_all_match_canonical",
            aggregation_passed,
            "A fixed coordinate-aggregation strategy changed a canonical core direction.",
        ),
        _gate(
            claim_id,
            "knn_directional_stability",
            directionally_stable,
            "knn_robustness_summary.csv",
            f"{row.region}/{row.threshold}/classification",
            row.classification,
            "The direction is k-dependent over the pre-specified 43 values.",
        ),
        _gate(
            claim_id,
            "model_order_consistency",
            model_order_passed,
            "gradient_validation_decision.json",
            f"model_order_results/{row.region}/{row.threshold}",
            model_order["status"] if model_order else "not evaluated",
            "The comparison is model-order-dependent or was not a primary model-order comparison.",
        ),
        _gate(
            claim_id,
            "conditioning_consistency",
            conditioning_passed,
            "conditioning_decision.json",
            "final_q_claim_status",
            conditioning["final_q_claim_status"],
            "Conditioning sensitivity was not directionally consistent for this exact comparison.",
        ),
        _gate(
            claim_id,
            "weight_exponent_affine_exactness",
            alpha_passed,
            "weight_exponent_decision.json",
            "affine_exactness.passed",
            alpha_passed,
            "The alpha=2 affine manufactured-field branch failed its numerical-exactness gate.",
        ),
        _gate(
            claim_id,
            "temporal_pairwise_persistence",
            bool(temporal["core_350_400_contrasts_pass"]),
            "temporal_validation_decision.json",
            "core_350_400_contrasts_pass",
            temporal["core_350_400_contrasts_pass"],
            "The pre-specified 350 W--400 W directions did not persist over 0.60--0.70 s.",
        ),
    ]


def non_q_gate_records(inputs: LoadedInputs) -> list[Gate]:
    export = inputs.json_data["export"]
    temporal = inputs.json_data["temporal"]
    power = inputs.json_data["power_response"]
    spatial = inputs.json_data["spatial_support"]
    thermal = inputs.json_data["thermal_gradient"]
    complementary = inputs.json_data["complementary_descriptor"]
    velocity = inputs.json_data["velocity_extreme"]
    velocity_distribution = inputs.json_data["velocity_distribution"]
    transferability = inputs.json_data["transferability_scope"]
    model_fidelity = inputs.json_data["model_fidelity"]
    thermal_fidelity = inputs.json_data["thermal_fidelity"]
    thermal_aggregation = thermal["aggregation_sensitivity"]
    return [
        _gate("export_redundancy", "row_structure_audit", True, "export_diagnostics_decision.json", "row_structure", "30-file audit completed", "The export audit is incomplete."),
        _gate("export_redundancy", "aggregation_audit", bool(export["aggregation_sensitivity"]["k25_core_directions_all_match_canonical"]), "export_diagnostics_decision.json", "aggregation_sensitivity.k25_core_directions_all_match_canonical", export["aggregation_sensitivity"]["k25_core_directions_all_match_canonical"], "Canonical core directions were not stable across fixed aggregation strategies."),
        _gate("quasi_steadiness", "all_six_power_temporal_pass", bool(temporal["all_six_powers_pass"]), "temporal_validation_decision.json", "all_six_powers_pass", temporal["all_six_powers_pass"], "None of the six powers passed all 11 pre-specified temporal criteria."),
        _gate("quasi_steadiness", "pairwise_temporal_persistence", bool(temporal["core_350_400_contrasts_pass"]), "temporal_validation_decision.json", "core_350_400_contrasts_pass", temporal["core_350_400_contrasts_pass"], "The pre-specified 350 W--400 W directions reversed within the decision window."),
        _gate("thermal_gradient", "direct_export_field", thermal["primary_descriptor_status"] == "direct_exported_temperature_gradient_magnitude", "thermal_gradient_decision.json", "primary_descriptor_status", thermal["primary_descriptor_status"], "The direct exported temperature-gradient field is unavailable."),
        _gate("thermal_gradient", "support_all_powers", bool(thermal["canonical_snapshot"]["full_pool"]["support_eligible_all_powers"] and thermal["canonical_snapshot"]["interface_proxy"]["support_eligible_all_powers"]), "thermal_gradient_decision.json", "canonical_snapshot.*.support_eligible_all_powers", "full-pool and interface-proxy support", "At least one direct-gradient region lacks all-power support."),
        _gate("thermal_gradient", "aggregation_order_consistency", bool(thermal_aggregation["full_pool"]["power_ordering_matches_canonical"] and thermal_aggregation["interface_proxy"]["power_ordering_matches_canonical"]), "thermal_gradient_decision.json", "aggregation_sensitivity.*.power_ordering_matches_canonical", "full-pool and interface-proxy orders", "A fixed aggregation strategy changed a six-power gradient order."),
        _gate("thermal_gradient", "temporal_persistence", False, "temporal_validation_decision.json; thermal_gradient_decision.json", "status; temporal_pairwise_context", "late_time_snapshot; post-hoc context", "The temporal assessment does not support persistent interpretation."),
        _gate("six_case_response", "canonical_reproduction", bool(power["canonical_reproduction_passed"]), "power_response_decision.json", "canonical_reproduction_passed", power["canonical_reproduction_passed"], "The six-case canonical metrics did not reproduce."),
        _gate("six_case_response", "aggregation_reproduction", bool(power["aggregation_reproduction_passed"]), "power_response_decision.json", "aggregation_reproduction_passed", power["aggregation_reproduction_passed"], "The sampled-power extrema did not reproduce across fixed aggregation strategies."),
        _gate("six_case_response", "complete_pairwise_snapshot_context", bool(power["pairwise_snapshot_context"]["row_count"] == 60 and power["pairwise_snapshot_context"]["unordered_power_pair_count"] == 15 and power["pairwise_snapshot_context"]["metric_count"] == 4), "power_response_decision.json; pairwise_snapshot_context.csv", "pairwise_snapshot_context", power["pairwise_snapshot_context"], "The complete 60-row, 15-pair descriptor ledger is unavailable or incomplete."),
        _gate("six_case_response", "observed_domain_only", bool(power["observed_power_domain"] == "200--450 W" and power["no_extrapolation_beyond_observed_power_domain"]), "power_response_decision.json", "observed_power_domain; no_extrapolation_beyond_observed_power_domain", power["observed_power_domain"], "The observed sampled-power domain or no-extrapolation boundary was not retained."),
        _gate("six_case_response", "higher_power_regime_not_assessed", not bool(power["higher_power_regime_assessed"]), "power_response_decision.json", "higher_power_regime_assessed", power["higher_power_regime_assessed"], "No additional higher-power regime was assessed in this study."),
        _gate("six_case_response", "q_not_used_for_power_context", not bool(power["pairwise_snapshot_context"]["q_used"]), "power_response_decision.json", "pairwise_snapshot_context.q_used", power["pairwise_snapshot_context"]["q_used"], "Q was used in the six-power physical context, contrary to the stated audit boundary."),
        _gate("six_case_response", "no_continuous_interpolation", not bool(power["continuous_power_interpolation_performed"]), "power_response_decision.json", "continuous_power_interpolation_performed", power["continuous_power_interpolation_performed"], "A continuous-power inference was introduced."),
        _gate("six_case_response", "temporal_persistence", False, "temporal_validation_decision.json; power_response_decision.json", "status; snapshot_local_maxima", "late_time_snapshot; extrema change over time", "The discrete extrema are not temporally persistent."),
        _gate("velocity_extreme", "canonical_vmax_reproduction", bool(velocity["gates"]["canonical_vmax_reproduction"]), "velocity_extreme_decision.json", "gates.canonical_vmax_reproduction", velocity["gates"]["canonical_vmax_reproduction"], "The 0.70 s Vmax descriptor did not reproduce the canonical output."),
        _gate("velocity_extreme", "aggregation_direction_consistency", bool(velocity["gates"]["aggregation_direction_consistency"]), "velocity_extreme_decision.json", "gates.aggregation_direction_consistency", velocity["gates"]["aggregation_direction_consistency"], "A fixed coordinate aggregation changed the 350 W--400 W exported-Vmax direction."),
        _gate("velocity_extreme", "velocity_definition_direction_consistency", bool(velocity["gates"]["velocity_definition_direction_consistency"]), "velocity_extreme_decision.json", "gates.velocity_definition_direction_consistency", velocity["gates"]["velocity_definition_direction_consistency"], "The exported velocity magnitude and component-norm definitions changed the pairwise direction."),
        _gate("velocity_extreme", "native_solver_history_health", bool(velocity["gates"]["solver_health_all_gates_passed"]), "velocity_extreme_decision.json; solver_health_gate_audit.csv", "gates.solver_health_all_gates_passed", velocity["gates"]["solver_health_all_gates_passed"], "Native configuration, completion, residual, stability, or conservation records are missing or did not pass their documented gate."),
        _gate("velocity_distribution_separation", "central_iqr_separation", False, "velocity_distribution_overlap_decision.json", "canonical_central_distribution.iqr_overlap_observed", velocity_distribution["canonical_central_distribution"]["iqr_overlap_observed"], "The canonical 350 W and 400 W IQRs overlap; the 400 W IQR is contained within the 350 W IQR."),
        _gate("velocity_distribution_separation", "fixed_aggregation_overlap_context", not bool(velocity_distribution["aggregation_sensitivity"]["all_fixed_strategies_iqr_overlap"]), "velocity_distribution_overlap_decision.json", "aggregation_sensitivity.all_fixed_strategies_iqr_overlap", velocity_distribution["aggregation_sensitivity"]["all_fixed_strategies_iqr_overlap"], "The central IQR overlap remains under all four fixed aggregation strategies."),
        _gate("spatial_geometry", "spatial_support_eligibility", bool(spatial["all_spatial_geometric_comparisons_eligible"]), "spatial_exclusion_decision.json", "all_spatial_geometric_comparisons_eligible", spatial["all_spatial_geometric_comparisons_eligible"], "The XZ branch is excluded for insufficient support and an unreconciled legacy summary."),
        _gate("spatial_geometry", "explicit_exclusion", spatial["decision"] != "exclude_spatial_geometric_comparisons", "spatial_exclusion_decision.json", "decision", spatial["decision"], "The spatial branch is explicitly excluded."),
        _gate("marangoni_mechanism", "required_mechanistic_inputs", False, "thermal_gradient_decision.json", "marangoni_missing_inputs", "; ".join(thermal["marangoni_missing_inputs"]), "Required vector-gradient, surface, material, and stress inputs are unavailable."),
        _gate("configuration_binding", "scope_audit_complete", bool(transferability["all_controls_context_bound"]), "transferability_decision.json", "all_controls_context_bound", transferability["all_controls_context_bound"], "The declared configuration boundary was not completely audited."),
        _gate("configuration_binding", "no_portable_parameter_defaults", len(transferability["portable_parameter_defaults"]) == 0, "transferability_decision.json", "portable_parameter_defaults", transferability["portable_parameter_defaults"], "At least one current numerical or semantic control was incorrectly marked as portable."),
        _gate(
            "prior_model_validation_context",
            "prior_published_validation_context",
            model_fidelity["prior_model_validation_context"]["status"] == "direct_observation",
            "model_fidelity_decision.json",
            "prior_model_validation_context.status",
            model_fidelity["prior_model_validation_context"]["doi"],
            "The prior publication metadata or its context-only use was not documented.",
        ),
        _gate(
            "native_phase_model_configuration",
            "native_300W_phase_model_record",
            thermal_fidelity["phase_model"]["status"] == "direct_observation_300W_configuration",
            "thermal_fidelity_decision.json; phase_model_configuration.csv",
            "phase_model.status",
            thermal_fidelity["phase_model"]["status"],
            "The supplied 300 W native phase-model configuration was not parsed.",
        ),
        _gate(
            "solver_execution_record_300W",
            "normal_completion_recorded",
            thermal_fidelity["solver_execution_record"]["status"] == "direct_observation" and bool(thermal_fidelity["solver_execution_record"]["normal_completion_reported"]),
            "thermal_fidelity_decision.json; running_log_summary.csv",
            "solver_execution_record.status; normal_completion_reported",
            thermal_fidelity["solver_execution_record"],
            "The supplied 300 W run record did not establish normal completion as a directly observed log fact.",
        ),
        _gate(
            "adaptive_stability_events_300W",
            "adaptive_restart_events_recorded",
            thermal_fidelity["adaptive_stability_events"]["status"] == "direct_observation" and int(thermal_fidelity["adaptive_stability_events"]["event_count"]) == 2 and bool(thermal_fidelity["adaptive_stability_events"]["restart_with_smaller_timestep_reported_all_events"]),
            "thermal_fidelity_decision.json; running_log_events.csv",
            "adaptive_stability_events",
            thermal_fidelity["adaptive_stability_events"],
            "The two supplied 300 W adaptive restart events were not reproducibly parsed.",
        ),
        _gate(
            "temperature_high_tail",
            "complete_unfiltered_tail_audit",
            thermal_fidelity["temperature_tail"]["status"] == "audit_only" and int(thermal_fidelity["temperature_tail"]["canonical_070_snapshot_rows"]) == 6,
            "thermal_fidelity_decision.json; temperature_tail_metrics.csv",
            "temperature_tail.status; canonical_070_snapshot_rows",
            thermal_fidelity["temperature_tail"],
            "The complete unfiltered high-temperature-tail audit is unavailable.",
        ),
        _gate(
            "temperature_high_tail",
            "unfiltered_values_remain_canonical",
            thermal_fidelity["temperature_tail"]["canonical_representation"] == "unfiltered exact_coordinate_mean",
            "thermal_fidelity_decision.json; temperature_tail_sensitivity.csv",
            "temperature_tail.canonical_representation",
            thermal_fidelity["temperature_tail"]["canonical_representation"],
            "A tail-screened result replaced the unfiltered canonical export.",
        ),
        *[
            _gate(
                "current_temperature_field_physical_fidelity",
                gate_id,
                False,
                "thermal_fidelity_decision.json; thermal_fidelity_gate_audit.csv",
                "failed_current_fidelity_gates",
                gate_id,
                "The supplied native material does not contain the evidence required to validate the current numerical temperature field.",
            )
            for gate_id in thermal_fidelity["failed_current_fidelity_gates"]
        ],
        *[
            _gate(
                "current_cfd_physical_fidelity",
                gate_id,
                False,
                "model_fidelity_decision.json; cfd_fidelity_gate_audit.csv",
                "failed_current_fidelity_gates",
                gate_id,
                "The current numerical exports do not contain the evidence required to validate their physical fidelity.",
            )
            for gate_id in model_fidelity["failed_current_fidelity_gates"]
        ],
        *[
            _gate(
                "beyond_lded_applicability",
                gate_id,
                False,
                "transferability_decision.json",
                "failed_external_gates",
                gate_id,
                "The current study does not contain the external evidence required for cross-context applicability.",
            )
            for gate_id in transferability["failed_external_gates"]
        ],
        _gate("complementary_tensor_descriptors", "canonical_reproduction", bool(complementary["baseline_reproduction_passed"]), "complementary_descriptor_decision.json", "baseline_reproduction_passed", complementary["baseline_reproduction_passed"], "The canonical Q baseline did not reproduce in the shared-tensor audit."),
        _gate("complementary_tensor_descriptors", "q_omega_identity", bool(complementary["q_omega_exact_agreement"]["all_cells_passed"]), "complementary_descriptor_decision.json", "q_omega_exact_agreement.all_cells_passed", complementary["q_omega_exact_agreement"]["all_cells_passed"], "The expected Q--Omega algebraic identity was not reproduced."),
        _gate("complementary_tensor_descriptors", "shared_tensor_independence", False, "complementary_descriptor_decision.json", "q_lambda_agreement.is_independent_validation", complementary["q_lambda_agreement"]["is_independent_validation"], "All three descriptors are computed from the same reconstructed WLS tensor and are not independent validation."),
        _gate("complementary_tensor_descriptors", "weight_exponent_affine_exactness", bool(complementary["existing_blocking_gates"]["weight_exponent_affine_exactness_passed"]), "complementary_descriptor_decision.json", "existing_blocking_gates.weight_exponent_affine_exactness_passed", complementary["existing_blocking_gates"]["weight_exponent_affine_exactness_passed"], "The alpha=2 affine numerical-exactness gate remains failed."),
        _gate("complementary_tensor_descriptors", "temporal_pairwise_persistence", bool(complementary["existing_blocking_gates"]["temporal_pairwise_persistence_passed"]), "complementary_descriptor_decision.json", "existing_blocking_gates.temporal_pairwise_persistence_passed", complementary["existing_blocking_gates"]["temporal_pairwise_persistence_passed"], "The 350 W--400 W direction does not persist over the pre-specified temporal window."),
        _gate("complementary_tensor_descriptors", "native_solver_gradient_reference", bool(complementary["existing_blocking_gates"]["native_solver_gradient_reference_available"]), "complementary_descriptor_decision.json", "existing_blocking_gates.native_solver_gradient_reference_available", complementary["existing_blocking_gates"]["native_solver_gradient_reference_available"], "No compatible native solver velocity-gradient reference is available."),
    ]
