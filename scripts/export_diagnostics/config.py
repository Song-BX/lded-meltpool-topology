from __future__ import annotations

from pathlib import Path

from scripts.analysis.release_paths import reference_input

from scripts.robustness.config import (
    FOF_INTERFACE_THRESHOLD,
    HEAT_FLUX_THRESHOLD,
    K_REFERENCE,
    K_VALUES,
    REGIONS,
    THRESHOLDS,
    WLS_CONDITION_CUTOFF,
    WLS_CONDITION_MODE,
    WLS_DISTANCE_EXPONENT,
    WLS_DISTANCE_OFFSET_M,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "raw data"
TEMPORAL_DIR = RAW_DIR / "temporal_validation"
OPTIONAL_AUDIT_DIR = RAW_DIR / "export_audit"
OUTPUT_DIR = ROOT / "图" / "export_diagnostics"
CANONICAL_METRICS = reference_input(
    ROOT, "Aplus_main_metrics_k25.csv", Path("图/3/Aplus_main_metrics_k25.csv")
)
CANONICAL_KNN_CONTRASTS = reference_input(
    ROOT, "knn_core_contrasts.csv", Path("图/robustness/knn_core_contrasts.csv")
)

EXPECTED_POWERS = (200, 250, 300, 350, 400, 450)
EXPECTED_TIMES = (0.50, 0.55, 0.60, 0.65, 0.70)
TIME_LABELS = {
    0.50: "0.5",
    0.55: "0.55",
    0.60: "0.6",
    0.65: "0.65",
    0.70: "0.7",
}

COORDINATE_COLUMNS = ("x", "y", "z")
PHYSICAL_COLUMNS = ("fof", "heat_flux", "T", "gradT", "u", "v", "w", "V")
ALL_STANDARD_COLUMNS = COORDINATE_COLUMNS + PHYSICAL_COLUMNS

AGGREGATION_STRATEGIES = (
    "mean_all_records",
    "median_all_records",
    "first_record",
    "mean_distinct_states",
)
CANONICAL_STRATEGY = "mean_all_records"
