"""Immutable selection rules for the source-only R1 upload package."""

from __future__ import annotations

from pathlib import Path


RELEASE_NAME = "lded-meltpool-topology-r1-reproducibility"
RELEASE_TAG = "r1-review-2026-07-31"
RELEASE_DESCRIPTION = "Source data, code, and reproduction workflow for the R1 revision"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "release"

RAW_DATA_DIR = ROOT / "raw data"
SCRIPTS_DIR = ROOT / "scripts"
TESTS_DIR = ROOT / "tests"

RETAINED_SCRIPT_DIRECTORIES = (
    "analysis",
    "claim_classification",
    "complementary_descriptor_audit",
    "conditioning_sensitivity",
    "export_diagnostics",
    "figures",
    "gradient_validation",
    "model_fidelity_boundary",
    "power_response_audit",
    "release_package",
    "robustness",
    "spatial_support_audit",
    "temporal_validation",
    "thermal_gradient_audit",
    "thermal_fidelity_audit",
    "transferability_scope_audit",
    "velocity_distribution_overlap_audit",
    "velocity_extreme_audit",
    "weight_exponent_sensitivity",
)
RETIRED_ROOT_SCRIPTS = frozenset(
    {
        "plot_figures.py",
        "preprocess.py",
        "reconstruct_q.py",
        "run_all.py",
        "statistics.py",
    }
)

EXCLUDED_SUFFIXES = frozenset({".tiff", ".tif", ".pyc", ".opju", ".vsdx"})
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {".codex", ".venv", "__pycache__", "release", "revision", "tmp", "图", "latex_restructure"}
)

RAW_CSV_COUNT = 30
REQUIRED_GENERATED_FILES = (
    "README.md",
    "REPRODUCTION.md",
    "DATA_DICTIONARY.md",
    "RELEASE_CONTENTS.csv",
    "SHA256SUMS.txt",
    "LICENSE-CODE",
    "LICENSE-DATA",
    "CITATION.cff",
    ".gitignore",
    "run_reproduction.py",
    "verify_release.py",
)
