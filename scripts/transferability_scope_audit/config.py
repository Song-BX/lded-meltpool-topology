from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "图" / "transferability_scope_audit"


@dataclass(frozen=True)
class SourceSpec:
    key: str
    relative_path: Path
    module_name: str
    required_attributes: tuple[str, ...]


SOURCE_SPECS = (
    SourceSpec(
        "point_cloud_schema",
        Path("scripts/analysis/point_cloud.py"),
        "scripts.analysis.point_cloud",
        ("COLUMN_MAP",),
    ),
    SourceSpec(
        "robustness_configuration",
        Path("scripts/robustness/config.py"),
        "scripts.robustness.config",
        (
            "EXPECTED_POWERS",
            "K_VALUES",
            "K_REFERENCE",
            "COORDINATE_TOLERANCE_M",
            "FOF_INTERFACE_THRESHOLD",
            "HEAT_FLUX_THRESHOLD",
            "WLS_DISTANCE_EXPONENT",
            "WLS_DISTANCE_OFFSET_M",
            "WLS_CONDITION_MODE",
            "WLS_CONDITION_CUTOFF",
            "GRID_SPACING_MM",
            "MIN_REGION_POINTS",
            "MIN_POOLED_EXCEEDANCES",
        ),
    ),
    SourceSpec(
        "temporal_configuration",
        Path("scripts/temporal_validation/config.py"),
        "scripts.temporal_validation.config",
        ("EXPECTED_TIMES", "PLATEAU_TIMES", "STABILITY_RULES"),
    ),
    SourceSpec(
        "manufactured_field_configuration",
        Path("scripts/gradient_validation/config.py"),
        "scripts.gradient_validation.config",
        ("FIELD_SPECS", "QUADRATIC_K_VALUES", "AFFINE_NUMERICAL_TOLERANCE"),
    ),
)


CURRENT_CONTEXT = {
    "process": "L-DED",
    "solver_export": "FLOW-3D CSV point-cloud export",
    "material_scan_mesh_context": "one material, one scan strategy, one mesh configuration",
    "cases": "six discrete powers and 30 time-power exports",
    "external_context_runs": 0,
}


EXTERNAL_GENERALISATION_GATES = (
    (
        "independent_external_context",
        "At least one independently generated target context is available for a separate audit.",
        "No additional process, material, scan-strategy, mesh, solver, or experiment context was analysed.",
    ),
    (
        "cross_solver_schema_and_entity_mapping",
        "The target export has a documented semantic and unit mapping for coordinates, velocity, masks, and field entities.",
        "No non-FLOW-3D export schema or point/cell/entity mapping is available.",
    ),
    (
        "external_geometry_recalibration",
        "Neighbourhood scale, conditioning, model order, weighting, and manufactured-field checks are re-specified on the target geometry.",
        "All numerical settings were assessed only on the current FLOW-3D point geometries.",
    ),
    (
        "external_mask_and_support_validation",
        "Region semantics, support gates, and threshold construction are redefined and audited for the target fields.",
        "FOF, heat-flux, support, and pooled-threshold definitions were specified for the current export semantics.",
    ),
    (
        "external_end_to_end_audit",
        "The complete diagnostic and claim-promotion sequence is run and reported for the target context.",
        "No end-to-end audit has been run outside the current L-DED/FLOW-3D configuration.",
    ),
)

