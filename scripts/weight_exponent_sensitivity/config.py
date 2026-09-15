from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.gradient_validation.config import (
    AFFINE_NUMERICAL_TOLERANCE,
    FIELD_SPECS,
    Q_MARGIN_FRACTION,
    REGIONS as VALIDATION_REGIONS,
    RESAMPLE_COUNT,
    RESAMPLE_NEIGHBOURS,
    RESAMPLE_SEED,
)
from scripts.robustness.config import (
    CORE_POWER_HIGH,
    CORE_POWER_LOW,
    EXPECTED_POWERS,
    FOF_INTERFACE_THRESHOLD,
    K_REFERENCE,
    K_VALUES,
    MIN_REGION_POINTS,
    REGIONS,
    THRESHOLDS,
    WLS_CONDITION_CUTOFF,
    WLS_CONDITION_MODE,
    WLS_DISTANCE_OFFSET_M,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "图" / "weight_exponent_sensitivity"
BASELINE_METRICS = ROOT / "图" / "robustness" / "knn_power_metrics.csv"


@dataclass(frozen=True)
class AlphaSpec:
    label: str
    value: float
    role: str


ALPHA_SPECS = (
    AlphaSpec("0", 0.0, "canonical_equal_weight"),
    AlphaSpec("1", 1.0, "unjustified_original_manuscript_setting"),
    AlphaSpec("2", 2.0, "reviewer_requested_comparator"),
)

