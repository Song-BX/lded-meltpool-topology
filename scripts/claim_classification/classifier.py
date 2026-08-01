from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd

from .config import Q_REQUIRED_GATES
from .gates import Gate, non_q_gate_records, q_gate_records
from .inputs import LoadedInputs


def classify_q_status(
    gates: Iterable[Gate], support_status: str, analysis_role: str
) -> str:
    by_id = {gate.gate_id: gate for gate in gates}
    if not by_id["support_eligibility"].passed:
        return "audit_only" if analysis_role == "audit_only" else "insufficient_support"
    numerical = (
        "canonical_reproduction",
        "aggregation_consistency",
        "knn_directional_stability",
        "model_order_consistency",
        "conditioning_consistency",
        "weight_exponent_affine_exactness",
    )
    if not all(by_id[gate_id].passed for gate_id in numerical):
        return "audit_only"
    if not by_id["temporal_pairwise_persistence"].passed:
        return "snapshot_local_descriptor"
    return "comparative_evidence"


def classify_spatial_status(spatial_decision: str) -> str:
    """An explicit exclusion remains blocking regardless of any other result."""
    if spatial_decision == "exclude_spatial_geometric_comparisons":
        return "excluded"
    return "comparative_evidence"


def classify_velocity_extreme_status(gates: Iterable[Gate]) -> str:
    """Keep a Vmax difference below comparative evidence even if every audit gate passes."""
    return "snapshot_local_descriptor" if all(gate.passed for gate in gates) else "audit_only"


def _failed_gates(gates: Iterable[Gate]) -> str:
    failed: list[str] = []
    for gate in gates:
        if gate.required_for_promotion and not gate.passed and gate.gate_id not in failed:
            failed.append(gate.gate_id)
    return "; ".join(failed)


def _registry_row(
    claim_id: str,
    claim: str,
    final_status: str,
    support_status: str,
    retention_role: str,
    promotion_rule: str,
    allowed_interpretation: str,
    prohibited_interpretation: str,
    evidence_paths: str,
    gates: list[Gate],
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "claim": claim,
        "final_status": final_status,
        "support_status": support_status,
        "retention_role": retention_role,
        "promotion_rule": promotion_rule,
        "failed_gates": _failed_gates(gates),
        "allowed_interpretation": allowed_interpretation,
        "prohibited_interpretation": prohibited_interpretation,
        "evidence_paths": evidence_paths,
    }


def build_registry(inputs: LoadedInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    gates = non_q_gate_records(inputs)
    spatial_status = classify_spatial_status(inputs.json_data["spatial_support"]["decision"])
    transferability = inputs.json_data["transferability_scope"]
    model_fidelity = inputs.json_data["model_fidelity"]
    thermal_fidelity = inputs.json_data["thermal_fidelity"]
    grouped: dict[str, list[Gate]] = defaultdict(list)
    for gate in gates:
        grouped[gate.claim_id].append(gate)
    velocity_extreme_status = classify_velocity_extreme_status(grouped["velocity_extreme"])

    registry = [
        _registry_row(
            "export_redundancy",
            "Systematic export-level redundancy is present in the available files.",
            "direct_observation",
            "not_applicable",
            "descriptive_only",
            "Complete row-structure and aggregation audits are required for this file-level observation.",
            "A directly observed export-level feature requiring audit.",
            "A specific FLOW-3D setting, solver pathology, or physical mechanism.",
            "export_diagnostics_decision.json",
            grouped["export_redundancy"],
        ),
        _registry_row(
            "configuration_binding",
            "The retained analysis configuration is bound to one L-DED/FLOW-3D export context.",
            "direct_observation",
            "not_applicable",
            "configuration_record",
            "A complete configuration-source audit must show that no numerical, semantic, or study-design control is labelled as a portable default.",
            "A transparent record of the current context-bound controls and a future re-specification checklist.",
            "A solver-independent, material-independent, or directly transferable parameter default.",
            "transferability_scope_audit/transferability_decision.json; context_bound_controls.csv",
            grouped["configuration_binding"],
        ),
        _registry_row(
            "beyond_lded_applicability",
            "The workflow is empirically applicable beyond the current L-DED/FLOW-3D configuration.",
            "not_supported",
            "not_applicable",
            "not_retained",
            "Independent external contexts, semantic export mapping, target-geometry recalibration, target mask/support validation, and a complete external audit are all required.",
            "No cross-context applicability conclusion is retained.",
            "Applicable beyond L-DED, a general-purpose framework, solver-independent performance, or transferable parameter defaults.",
            "transferability_scope_audit/transferability_decision.json; transferability_gate_audit.csv",
            grouped["beyond_lded_applicability"],
        ),
        _registry_row(
            "prior_model_validation_context",
            "A published prior model-validation context is documented for the model family.",
            "direct_observation",
            "not_applicable",
            "context_only",
            "A verified citation and a record limiting its use to model provenance are required.",
            "Published model provenance and limited validation background only.",
            "Validation of the current six power cases, the exported point fields, velocity gradients, Q descriptors, numerical convergence, or a physical mechanism.",
            "model_fidelity_boundary/model_fidelity_decision.json; model_alignment_audit.csv",
            grouped["prior_model_validation_context"],
        ),
        _registry_row(
            "current_cfd_physical_fidelity",
            "The six current solver-exported fields are physically validated representations of an actual L-DED melt pool.",
            str(model_fidelity["current_cfd_physical_fidelity"]),
            "not_applicable",
            "not_retained",
            "Case-matched experiment, complete solver history, mesh/timestep convergence, and entity-compatible native fields are all required.",
            "Only solver-exported numerical-field descriptors are retained.",
            "An experimentally validated melt-pool representation, numerical convergence, validated topology, physical mechanism, or experimentally supported comparison.",
            "model_fidelity_boundary/model_fidelity_decision.json; cfd_fidelity_gate_audit.csv",
            grouped["current_cfd_physical_fidelity"],
        ),
        _registry_row(
            "native_phase_model_configuration",
            "The supplied 300 W native file records configured liquid--vapour phase-change settings.",
            "direct_observation",
            "not_applicable",
            "configuration_record",
            "A parseable supplied 300 W configuration record is required.",
            "A native-configuration fact for the supplied 300 W file only.",
            "Accurate liquid--vapour physics, validated evaporation, output fidelity, or verification of all six configurations.",
            "thermal_fidelity_audit/phase_model_configuration.csv",
            grouped["native_phase_model_configuration"],
        ),
        _registry_row(
            "solver_execution_record_300W",
            "The supplied 300 W run record reports normal completion at its documented end time and cycle.",
            "direct_observation",
            "not_applicable",
            "context_only",
            "The supplied 300 W log must contain the parsed completion record.",
            "A 300 W execution-record fact, including its reported end time and cycle.",
            "All-six solver health, configured residual acceptance, numerical convergence, or physical fidelity.",
            "thermal_fidelity_audit/running_log_summary.csv",
            grouped["solver_execution_record_300W"],
        ),
        _registry_row(
            "adaptive_stability_events_300W",
            "The supplied 300 W log records two convective-flux stability-limit events followed by smaller-step restarts.",
            "direct_observation",
            "not_applicable",
            "context_only",
            "The two supplied event records and their restart messages must parse reproducibly.",
            "A native 300 W adaptive-time-step log fact.",
            "A failure diagnosis, output-cause explanation, or attribution to a 350/400 W temperature coordinate.",
            "thermal_fidelity_audit/running_log_events.csv",
            grouped["adaptive_stability_events_300W"],
        ),
        _registry_row(
            "temperature_high_tail",
            "High-temperature exported values, including Tmax, are retained as an unfiltered numerical-output tail audit.",
            "audit_only",
            "support_checked",
            "audit_only",
            "A complete unfiltered 30-snapshot tail audit and explicit preservation of its canonical values are required; no physical-fidelity promotion path is available.",
            "Transparent peak and tail context, including sensitivity to two reported screens.",
            "Correct physical vapour temperature, solver instability diagnosis, a mesh-cell cause, convergence, or replacement of canonical values by screened values.",
            "thermal_fidelity_audit/temperature_tail_metrics.csv; temperature_tail_sensitivity.csv; Fig. S11",
            grouped["temperature_high_tail"],
        ),
        _registry_row(
            "current_temperature_field_physical_fidelity",
            "The current six exported numerical temperature fields are physically faithful representations of L-DED melt-pool temperatures.",
            str(thermal_fidelity["current_temperature_field_physical_fidelity"]),
            "not_applicable",
            "not_retained",
            "Case-matched experiments, all-six native histories with documented acceptance criteria, mesh/time-step convergence, and output-field identity are required.",
            "Only bounded numerical-export temperature descriptors are retained.",
            "Temperature-field fidelity, validated phase-change accuracy, convergence, or a physical explanation for an individual high-temperature coordinate.",
            "thermal_fidelity_audit/thermal_fidelity_decision.json; thermal_fidelity_gate_audit.csv",
            grouped["current_temperature_field_physical_fidelity"],
        ),
        _registry_row(
            "quasi_steadiness",
            "The 0.70 s field is quasi-steady.",
            "not_supported",
            "not_applicable",
            "not_retained",
            "All six powers must pass all 11 pre-specified temporal criteria; pairwise persistence is required for the 350 W--400 W contrast.",
            "The field is a late-time snapshot.",
            "Quasi-steadiness or temporal persistence.",
            "s4/temporal_validation_decision.json",
            grouped["quasi_steadiness"],
        ),
        _registry_row(
            "thermal_gradient",
            "Direct exported temperature-gradient magnitude describes the 0.70 s numerical export.",
            "snapshot_local_descriptor",
            "support_checked",
            "descriptive_snapshot",
            "A direct field, all-power support, and stable discrete ordering across the four fixed aggregation strategies are required. Temporal persistence is additionally required for a persistent descriptor.",
            "A primary, aggregation-robust descriptor of the 0.70 s numerical export.",
            "A persistent ranking, Marangoni stress, causal mechanism, or experimental melt-pool validation.",
            "thermal_gradient_audit/thermal_gradient_decision.json",
            grouped["thermal_gradient"],
        ),
        _registry_row(
            "six_case_response",
            "The six sampled numerical exports provide a complete 0.70 s descriptor ledger for every unordered pair within 200--450 W.",
            "snapshot_local_descriptor",
            "not_applicable",
            "descriptive_snapshot",
            "Canonical and aggregation reproduction, a complete 60-row pairwise ledger, and an explicit observed-domain boundary are required. Temporal persistence is additionally required before any persistent-response statement.",
            "Aggregation-robust descriptor context within one late-time numerical export set and the observed 200--450 W range only.",
            "A continuous response, inflection, transition power, physical anomaly source, behavior above 450 W, a distinct physical regime, or experimental confirmation.",
            "power_response_audit/power_response_decision.json; power_response_audit/pairwise_snapshot_context.csv",
            grouped["six_case_response"],
        ),
        _registry_row(
            "velocity_extreme",
            "The 350 W--400 W full-pool Vmax difference is a sparse peak-level audit record.",
            velocity_extreme_status,
            "native solver history unavailable",
            "audit_only" if velocity_extreme_status == "audit_only" else "descriptive_snapshot",
            "Canonical reproduction, fixed aggregation, exported-magnitude/component-norm direction agreement, and all documented native solver-history health gates are required. Passing every gate still permits only a snapshot descriptor.",
            "Peak-level audit record at 0.70 s only; it does not separate the central full-pool distributions." if velocity_extreme_status == "snapshot_local_descriptor" else "Sparse peak-level audit record only; central IQR overlap is observed and native solver history is unavailable or fails a documented health gate.",
            "A whole-pool flow-strength decrease, robust structural signal, numerical convergence, comparative evidence, physical mechanism, or causal explanation.",
            "velocity_extreme_audit/velocity_extreme_decision.json; velocity_distribution_overlap_audit/velocity_distribution_overlap_decision.json; Figs. S1 and S10",
            grouped["velocity_extreme"],
        ),
        _registry_row(
            "velocity_distribution_separation",
            "The 350 W and 400 W full-pool velocity distributions are centrally separated.",
            "not_supported",
            "central IQR overlap observed",
            "not_retained",
            "A non-overlapping central distribution would still be descriptive only; the observed IQR overlap prevents a Vmax-based whole-pool separation claim.",
            "The IQR relation is transparent distributional context for a single late-time snapshot.",
            "A robust structural signal, a whole-pool flow-strength difference, comparative evidence, numerical convergence, or a physical mechanism.",
            "velocity_distribution_overlap_audit/velocity_distribution_overlap_decision.json; Figs. S1 and S10",
            grouped["velocity_distribution_separation"],
        ),
        _registry_row(
            "spatial_geometry",
            "XZ geometric comparisons and centroid-derived structure.",
            spatial_status,
            "insufficient_support",
            "not_retained",
            "Spatial support must pass before any geometric comparison; an explicit exclusion decision blocks all promotion.",
            "Only the exclusion and its support audit.",
            "Centroid, RMS radius, spatial coherence, or inter-power displacement claims.",
            "spatial_support_audit/spatial_exclusion_decision.json",
            grouped["spatial_geometry"],
        ),
        _registry_row(
            "marangoni_mechanism",
            "A Marangoni mechanism explains the sampled-power descriptors.",
            "not_supported",
            "not_applicable",
            "requires_future_protocol",
            "A separate protocol must provide compatible vector gradients, free-surface geometry, tangential gradients, material surface-tension data, stress outputs, and independent physical validation.",
            "General physical background only.",
            "A causal Marangoni explanation or physical-mechanism evidence.",
            "thermal_gradient_audit/thermal_gradient_decision.json",
            grouped["marangoni_mechanism"],
        ),
        _registry_row(
            "complementary_tensor_descriptors",
            r"Agreement among Q, $\lambda_2$, and normalized $\Omega_N$ descriptors from the shared reconstructed tensor.",
            "audit_only",
            "support_checked",
            "audit_only",
            "Canonical reproduction and descriptor checks are required, but shared-tensor descriptors cannot independently validate one another; all existing numerical, temporal, and native-reference gates also remain binding.",
            "A transparent, reconstruction-dependent consistency audit at one late-time snapshot.",
            "Independent cross-validation, validated vortex structure, comparative vortex evidence, or physical-mechanism evidence.",
            "complementary_descriptor_audit/complementary_descriptor_decision.json; Fig. S9",
            grouped["complementary_tensor_descriptors"],
        ),
    ]

    q_rows = inputs.q_eligibility.merge(
        inputs.q_robustness[["region", "threshold", "classification", "positive_count", "negative_count"]],
        on=["region", "threshold"],
        suffixes=("", "_robustness"),
        validate="one_to_one",
    ).sort_values(["region", "threshold"])
    q_statuses: list[str] = []
    for row in q_rows.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        q_gates = q_gate_records(row_series, inputs)
        q_status = classify_q_status(q_gates, str(row.evidence_status), str(row.analysis_role))
        q_statuses.append(q_status)
        claim_id = q_gates[0].claim_id
        grouped[claim_id] = q_gates
        registry.append(
            _registry_row(
                claim_id,
                f"Q-proxy contrast: {row.region}, {row.threshold}.",
                q_status,
                str(row.evidence_status),
                "audit_only" if q_status == "audit_only" else str(row.analysis_role),
                "All support, canonical, aggregation, kNN, model-order, conditioning, distance-weight affine, and temporal gates must pass for comparative evidence.",
                "A transparent audit record at the stated operational status.",
                "A comparative Q conclusion, solver-gradient validation, vortex-strength measurement, or physical mechanism claim.",
                "robustness/knn_evidence_eligibility.csv; robustness/knn_robustness_summary.csv; gradient_validation_decision.json; weight_exponent_decision.json; conditioning_decision.json; temporal_validation_decision.json",
                q_gates,
            )
        )

    q_overall_gates = grouped["q-all-Qgt0"]
    registry.insert(
        4,
        _registry_row(
            "q_proxy_comparisons",
            "Reconstructed Q-proxy comparisons between 350 W and 400 W.",
            "audit_only",
            "7 evidence_eligible; 9 insufficient_support",
            "audit_only",
            "A Q comparison can become comparative evidence only if every applicable boolean gate passes; evidence eligibility alone is not a conclusion. Across the branch, $\\alpha=2$ affine exactness fails, six support-qualified combinations are model-order-dependent, nine lack support, and the pairwise temporal gate fails.",
            "Reconstruction-dependent snapshot audit descriptors.",
            "Any comparative or physical Q claim.",
            "robustness; gradient_validation; weight_exponent_sensitivity; conditioning_sensitivity; temporal_validation",
            q_overall_gates,
        ),
    )

    registry_frame = pd.DataFrame(registry)
    audit_rows: list[dict[str, object]] = []
    final_status = dict(zip(registry_frame["claim_id"], registry_frame["final_status"]))
    for claim_id, claim_gates in grouped.items():
        for gate in claim_gates:
            row = gate.row()
            row["final_status"] = final_status.get(claim_id, "audit_only")
            audit_rows.append(row)
    return registry_frame, pd.DataFrame(audit_rows)


def decision_payload(registry: pd.DataFrame, inputs: LoadedInputs) -> dict[str, object]:
    q_records = registry[registry["claim_id"].str.startswith("q-")]
    return {
        "policy_scope": "revision-stage deterministic classification of completed Reviewer 1 audits and Reviewer 2 Comments 1--3; Comment 14 release packaging remains deferred",
        "policy_type": "all-required-gates boolean rules; not an ordinal evidence score",
        "status_precedence": ["excluded", "insufficient_support", "audit_only", "not_supported", "snapshot_local_descriptor", "comparative_evidence", "physical_mechanism_evidence"],
        "q_promotion_rule": list(Q_REQUIRED_GATES),
        "input_validation": {
            "all_manifest_rows_passed": bool(inputs.manifest["validation_passed"].all()),
            "source_count": int(len(inputs.manifest)),
            "q_combination_count": int(len(q_records)),
            "q_support_status_counts": q_records["support_status"].value_counts().to_dict(),
            "q_final_status_counts": q_records["final_status"].value_counts().to_dict(),
        },
        "current_outcomes": registry[["claim_id", "final_status", "support_status", "failed_gates"]].to_dict(orient="records"),
            "physical_mechanism_evidence": {
            "current_status": "unreachable_from_available_csv",
            "requires": [
                "a separate pre-specified validation protocol",
                "compatible native fields and surface quantities",
                "independent physical validation",
            ],
        },
    }
