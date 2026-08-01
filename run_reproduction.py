"""Run the fixed R1 analysis and figure-generation workflow."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "scripts" / "figures"
sys.dont_write_bytecode = True


def _run(label: str, command: list[str], *, cwd: Path = ROOT) -> None:
    print(f"\n==> {label}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["R1_RELEASE_NON_TIFF"] = "1"
    prior_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(ROOT) if not prior_pythonpath else str(ROOT) + os.pathsep + prior_pythonpath
    subprocess.run(command, cwd=cwd, check=True, env=environment)


def main() -> None:
    module_steps = (
        ("export-structure audit", "scripts.export_diagnostics.run_analysis"),
        ("dense kNN/support audit", "scripts.robustness.run_analysis"),
        ("gradient-validation audit", "scripts.gradient_validation.run_analysis"),
        ("conditioning-cutoff audit", "scripts.conditioning_sensitivity.run_analysis"),
        ("distance-exponent audit", "scripts.weight_exponent_sensitivity.run_analysis"),
        ("temporal validation", "scripts.temporal_validation.run_analysis"),
        ("thermal-fidelity audit", "scripts.thermal_fidelity_audit.run_analysis"),
        ("six-power response audit", "scripts.power_response_audit.run_analysis"),
        ("thermal-gradient audit", "scripts.thermal_gradient_audit.run_analysis"),
        ("spatial-support audit", "scripts.spatial_support_audit.run"),
        ("complementary-tensor audit", "scripts.complementary_descriptor_audit.run_analysis"),
        ("velocity-extreme audit", "scripts.velocity_extreme_audit.run_analysis"),
        ("velocity-distribution overlap audit", "scripts.velocity_distribution_overlap_audit.run_analysis"),
        ("model-fidelity boundary", "scripts.model_fidelity_boundary.run_analysis"),
        ("cross-context scope audit", "scripts.transferability_scope_audit.run_analysis"),
        ("claim classification", "scripts.claim_classification.run_analysis"),
    )
    for label, module in module_steps:
        _run(label, [sys.executable, "-m", module])
    for filename in (
        "build_nature_figures.py",
        "build_export_diagnostics_figure.py",
        "build_knn_robustness_figure.py",
        "build_gradient_validation_figure.py",
        "build_temporal_validation_figure.py",
        "build_conditioning_sensitivity_figure.py",
        "build_weight_exponent_sensitivity_figure.py",
        "build_power_response_audit_figure.py",
        "build_thermal_gradient_audit_figure.py",
        "build_complementary_descriptor_audit_figure.py",
        "build_velocity_distribution_overlap_figure.py",
        "build_velocity_extreme_audit_figure.py",
        "build_thermal_fidelity_audit_figure.py",
    ):
        _run(f"figure builder: {filename}", [sys.executable, filename], cwd=FIGURE_DIR)


if __name__ == "__main__":
    main()
