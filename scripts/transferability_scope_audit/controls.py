from __future__ import annotations

import json
from typing import Any


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_context_bound_controls(values: dict[str, dict[str, Any]]) -> list[dict[str, object]]:
    cloud = values["point_cloud_schema"]
    robust = values["robustness_configuration"]
    temporal = values["temporal_configuration"]
    manufactured = values["manufactured_field_configuration"]
    return [
        {
            "control_id": "flow3d_csv_schema_and_units",
            "control_class": "current_export_semantic_control",
            "source_keys": "point_cloud_schema",
            "current_value": _compact(cloud["COLUMN_MAP"]),
            "current_scope": "The exact FLOW-3D CSV headers and their coordinate, velocity, FOF, heat-flux, temperature, and gradT meanings.",
            "future_requirement": "Map source entities, units, point/cell semantics, and missing fields before any new analysis.",
            "portable_default": False,
        },
        {
            "control_id": "coordinate_consolidation_and_aggregation",
            "control_class": "current_export_semantic_control",
            "source_keys": "point_cloud_schema; robustness_configuration",
            "current_value": _compact({"coordinate_tolerance_m": robust["COORDINATE_TOLERANCE_M"], "canonical_aggregation": "mean_all_records"}),
            "current_scope": "Exact stored-coordinate equality and all-record mean aggregation address the audited duplicate structure of these exports.",
            "future_requirement": "Audit duplicate classes and justify coordinate identity and aggregation for the target export.",
            "portable_default": False,
        },
        {
            "control_id": "neighbourhood_support_scale",
            "control_class": "current_geometry_calibrated_numerical_control",
            "source_keys": "robustness_configuration",
            "current_value": _compact({"k_values": robust["K_VALUES"], "reference_k": robust["K_REFERENCE"], "grid_spacing_mm": robust["GRID_SPACING_MM"]}),
            "current_scope": "The k=8--50 span was assessed against the current cloud density and approximately 0.1 mm nominal grid spacing.",
            "future_requirement": "Choose and report a target-geometry support range in physical units; rerun the full scale and sensitivity audit.",
            "portable_default": False,
        },
        {
            "control_id": "wls_model_weight_and_conditioning",
            "control_class": "current_geometry_calibrated_numerical_control",
            "source_keys": "robustness_configuration; manufactured_field_configuration",
            "current_value": _compact({"model": "first_order", "alpha": robust["WLS_DISTANCE_EXPONENT"], "epsilon_w_m": robust["WLS_DISTANCE_OFFSET_M"], "condition_mode": robust["WLS_CONDITION_MODE"], "condition_cutoff": robust["WLS_CONDITION_CUTOFF"], "affine_tolerance": manufactured["AFFINE_NUMERICAL_TOLERANCE"]}),
            "current_scope": "The canonical branch and its manufactured-field checks were evaluated only on these observed point geometries.",
            "future_requirement": "Reassess model order, weighting, conditioning, and manufactured-field error on the target geometry; do not carry over a cutoff or exponent by default.",
            "portable_default": False,
        },
        {
            "control_id": "region_mask_semantics",
            "control_class": "current_export_semantic_control",
            "source_keys": "robustness_configuration",
            "current_value": _compact({"interface_fof_threshold": robust["FOF_INTERFACE_THRESHOLD"], "heated_heat_flux_threshold": robust["HEAT_FLUX_THRESHOLD"]}),
            "current_scope": "FOF and heat-flux masks refer to the available FLOW-3D variables; heat-related masks remain audit only.",
            "future_requirement": "Define target-region semantics from documented source variables and independently audit numerical support.",
            "portable_default": False,
        },
        {
            "control_id": "support_and_pooled_threshold_policy",
            "control_class": "current_geometry_calibrated_numerical_control",
            "source_keys": "robustness_configuration",
            "current_value": _compact({"minimum_region_points": robust["MIN_REGION_POINTS"], "minimum_pooled_strict_exceedances": robust["MIN_POOLED_EXCEEDANCES"], "threshold_pool": "six current power cases"}),
            "current_scope": "The support gates and pooled Q thresholds are revision-stage rules for this six-case analysis, not universal statistical limits.",
            "future_requirement": "Pre-specify and audit target-specific support and threshold rules without treating spatial points as independent replicates.",
            "portable_default": False,
        },
        {
            "control_id": "temporal_and_power_case_design",
            "control_class": "current_study_design_control",
            "source_keys": "robustness_configuration; temporal_configuration",
            "current_value": _compact({"powers_W": robust["EXPECTED_POWERS"], "times_s": temporal["EXPECTED_TIMES"], "late_window_s": temporal["PLATEAU_TIMES"]}),
            "current_scope": "Six discrete powers and five serial exports per power define snapshot and temporal interpretations in this study.",
            "future_requirement": "Define independent target cases and temporal coverage; do not infer a continuous response from inherited case labels.",
            "portable_default": False,
        },
        {
            "control_id": "audit_and_claim_governance",
            "control_class": "procedural_audit_step",
            "source_keys": "all configuration sources",
            "current_value": _compact({"source_hash_manifest": True, "claim_promotion": "all-required-gates"}),
            "current_scope": "Hash validation, export audit, support audit, reconstruction checks, sensitivity checks, and claim promotion organise this revision's evidence.",
            "future_requirement": "Repeat the procedural sequence with a new declared configuration; completing the sequence alone does not establish cross-context performance.",
            "portable_default": False,
        },
    ]

