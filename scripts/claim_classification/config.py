from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "图" / "claim_classification"
TABLE_DIR = ROOT / "latex_restructure" / "tables"


@dataclass(frozen=True)
class InputSpec:
    key: str
    relative_path: Path
    kind: str
    required_keys: tuple[str, ...] = ()
    expected_rows: int | None = None


INPUT_SPECS = (
    InputSpec(
        "export",
        Path("图/export_diagnostics/export_diagnostics_decision.json"),
        "json",
        ("analysis_scope", "row_structure", "aggregation_sensitivity", "manuscript_interpretation"),
    ),
    InputSpec(
        "robustness",
        Path("图/robustness/knn_decision.json"),
        "json",
        ("k_values", "support_policy", "evidence_eligible_count", "insufficient_support_count", "combinations"),
    ),
    InputSpec(
        "gradient_validation",
        Path("图/gradient_validation/gradient_validation_decision.json"),
        "json",
        ("affine_numerical_exactness", "model_order_results", "q_claim_status"),
    ),
    InputSpec(
        "weight_exponent",
        Path("图/weight_exponent_sensitivity/weight_exponent_decision.json"),
        "json",
        ("baseline_reproduction_passed", "affine_exactness", "final_q_claim_status"),
    ),
    InputSpec(
        "conditioning",
        Path("图/conditioning_sensitivity/conditioning_decision.json"),
        "json",
        ("cutoffs", "final_q_claim_status", "cutoff_summary"),
    ),
    InputSpec(
        "temporal",
        Path("图/s4/temporal_validation_decision.json"),
        "json",
        ("status", "all_six_powers_pass", "core_350_400_contrasts_pass", "baseline_reproducibility_passed"),
    ),
    InputSpec(
        "power_response",
        Path("图/power_response_audit/power_response_decision.json"),
        "json",
        (
            "decision",
            "canonical_reproduction_passed",
            "aggregation_reproduction_passed",
            "snapshot_local_maxima",
            "observed_power_domain",
            "higher_power_regime_assessed",
            "no_extrapolation_beyond_observed_power_domain",
            "pairwise_snapshot_context",
        ),
    ),
    InputSpec(
        "power_response_pairwise_context",
        Path("图/power_response_audit/pairwise_snapshot_context.csv"),
        "csv",
        (
            "snapshot_time_s",
            "lower_power_W",
            "higher_power_W",
            "metric_id",
            "metric_label",
            "unit",
            "region",
            "lower_value",
            "higher_value",
            "delta_higher_minus_lower",
            "direction",
            "interpretation_status",
            "interpretation_boundary",
        ),
        60,
    ),
    InputSpec(
        "spatial_support",
        Path("图/spatial_support_audit/spatial_exclusion_decision.json"),
        "json",
        ("decision", "all_spatial_geometric_comparisons_eligible", "legacy_summary_status"),
    ),
    InputSpec(
        "thermal_gradient",
        Path("图/thermal_gradient_audit/thermal_gradient_decision.json"),
        "json",
        ("primary_descriptor_status", "canonical_snapshot", "aggregation_sensitivity", "marangoni_status"),
    ),
    InputSpec(
        "complementary_descriptor",
        Path("图/complementary_descriptor_audit/complementary_descriptor_decision.json"),
        "json",
        ("analysis_scope", "baseline_reproduction_passed", "q_omega_exact_agreement", "q_lambda_agreement", "existing_blocking_gates", "final_claim_status"),
    ),
    InputSpec(
        "velocity_extreme",
        Path("图/velocity_extreme_audit/velocity_extreme_decision.json"),
        "json",
        ("canonical_values", "peak_provenance", "gates", "final_status", "allowed_interpretation", "prohibited_interpretation"),
    ),
    InputSpec(
        "velocity_distribution",
        Path("图/velocity_distribution_overlap_audit/velocity_distribution_overlap_decision.json"),
        "json",
        (
            "canonical_central_distribution",
            "aggregation_sensitivity",
            "temporal_context",
            "whole_pool_distribution_separation",
            "vmax_role",
            "prohibited_interpretation",
        ),
    ),
    InputSpec(
        "transferability_scope",
        Path("图/transferability_scope_audit/transferability_decision.json"),
        "json",
        (
            "analysis_scope",
            "control_count",
            "all_controls_context_bound",
            "cross_context_applicability",
            "portable_parameter_defaults",
            "failed_external_gates",
            "allowed_statement",
            "prohibited_interpretations",
        ),
    ),
    InputSpec(
        "model_fidelity",
        Path("图/model_fidelity_boundary/model_fidelity_decision.json"),
        "json",
        (
            "analysis_scope",
            "prior_model_validation_context",
            "alignment_statuses",
            "current_cfd_physical_fidelity",
            "failed_current_fidelity_gates",
            "allowed_statement",
            "prohibited_interpretations",
        ),
    ),
    InputSpec(
        "thermal_fidelity",
        Path("图/thermal_fidelity_audit/thermal_fidelity_decision.json"),
        "json",
        (
            "phase_model",
            "solver_execution_record",
            "adaptive_stability_events",
            "temperature_tail",
            "high_temperature_cause_350_400W",
            "all_six_solver_health",
            "current_temperature_field_physical_fidelity",
            "central_temperature_descriptors",
            "failed_current_fidelity_gates",
        ),
    ),
    InputSpec(
        "q_eligibility",
        Path("图/robustness/knn_evidence_eligibility.csv"),
        "csv",
        ("region", "threshold", "evidence_status", "analysis_role", "evidence_eligible"),
        16,
    ),
    InputSpec(
        "q_robustness",
        Path("图/robustness/knn_robustness_summary.csv"),
        "csv",
        ("region", "threshold", "classification", "positive_count", "negative_count"),
        16,
    ),
)


STATUS_PRECEDENCE = (
    "excluded",
    "insufficient_support",
    "audit_only",
    "not_supported",
    "snapshot_local_descriptor",
    "comparative_evidence",
    "physical_mechanism_evidence",
)

Q_REQUIRED_GATES = (
    "support_eligibility",
    "canonical_reproduction",
    "aggregation_consistency",
    "knn_directional_stability",
    "model_order_consistency",
    "conditioning_consistency",
    "weight_exponent_affine_exactness",
    "temporal_pairwise_persistence",
)


Q_THRESHOLD_LABELS = {
    "Q>0": r"$Q>0$",
    "Q>posP50": r"$Q>P_{50}(Q^+)$",
    "Q>posP75": r"$Q>P_{75}(Q^+)$",
    "Q>posP90": r"$Q>P_{90}(Q^+)$",
}


Q_REGION_LABELS = {
    "all": "full-pool",
    "interface": "interface",
    "heated": "heated",
    "interface_heated": "interface-heated",
}
