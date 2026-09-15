from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.analysis.release_paths import reference_input


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "图" / "model_fidelity_boundary"
TABLE_DIR = ROOT / "latex_restructure" / "tables"
EVIDENCE_RECORD_PATH = reference_input(
    ROOT,
    "prior_model_validation_record.json",
    Path("scripts/model_fidelity_boundary/prior_model_validation_record.json"),
).relative_to(ROOT)


@dataclass(frozen=True)
class SourceSpec:
    key: str
    relative_path: Path
    required_text: str | None = None


SOURCE_SPECS = (
    SourceSpec("current_settings_note", Path("Flow3D设置.txt"), "Symmetry"),
    SourceSpec("current_phase_model_record", Path("Flow3D.md"), "if_phchg"),
    SourceSpec("prior_validation_record", EVIDENCE_RECORD_PATH),
)


ALIGNMENT_ITEMS = (
    (
        "model_lineage",
        "Model lineage",
        "The authors report that the earlier published study used the same model family.",
        "Author-provided relationship and the cited prior publication.",
    ),
    (
        "material",
        "Material and thermophysical-property definition",
        "316L stainless steel with the current FLOW-3D material definition.",
        "Flow3D设置.txt; Table 1.",
    ),
    (
        "process_conditions",
        "Laser, scan, powder-feed, and geometry conditions",
        "Six powers with the current scan, powder-feed, and domain settings.",
        "Table 1.",
    ),
    (
        "heat_source_absorptivity",
        "Heat source and absorptivity",
        "Current Gaussian free-surface source and stated energy efficiency.",
        "Table 1.",
    ),
    (
        "phase_interface_models",
        "Phase change, interface tracking, and surface forcing",
        "VOF, solid-liquid and liquid-vapour phase change, and surface tension are listed.",
        "Table 1.",
    ),
    (
        "boundary_and_symmetry",
        "Boundary and symmetry conditions",
        "The current domain uses the documented wall, pressure, and Ymin symmetry boundaries.",
        "Table 1; Flow3D设置.txt.",
    ),
    (
        "mesh_and_timestep",
        "Mesh and timestep/convergence evidence",
        "A 0.1 mm nominal grid is listed; timestep and convergence evidence are unavailable.",
        "Table 1; current audit inventory.",
    ),
    (
        "experimental_observable",
        "Experiment-simulation observable",
        "No experiment matched to the six current simulations is available.",
        "Current-study inventory.",
    ),
    (
        "postprocessing_entity_semantics",
        "Native field identity and export semantics",
        "The CSV exports lack entity identifiers and native velocity-gradient or solver-Q fields.",
        "Current export and gradient audits.",
    ),
)


CURRENT_FIDELITY_GATES = (
    (
        "current_matched_experimental_observable",
        "A direct experiment-simulation comparison under conditions matched to the six current cases.",
        "No melt-pool experiment was performed for the current six simulations.",
    ),
    (
        "current_solver_configuration_and_history",
        "Complete current configuration plus run, residual, stability, and conservation histories.",
        "Current solver-history and conservation records are unavailable.",
    ),
    (
        "grid_and_timestep_convergence",
        "A documented mesh and timestep convergence study for the current cases.",
        "No mesh or timestep convergence study is available for the current cases.",
    ),
    (
        "native_field_identity",
        "Entity-compatible native fields linking the post-processed points to solver quantities.",
        "CSV exports lack point/cell identifiers, native velocity gradients, and solver Q values.",
    ),
)
