from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.robustness.config import (
    COORDINATE_TOLERANCE_M,
    CORE_POWER_HIGH,
    CORE_POWER_LOW,
    EXPECTED_POWERS,
    FOF_INTERFACE_THRESHOLD,
    HEAT_FLUX_THRESHOLD,
    K_REFERENCE,
    K_VALUES,
    PERCENTILES,
    REGIONS,
    THRESHOLDS,
    WLS_CONDITION_CUTOFF,
    WLS_CONDITION_MODE,
    WLS_DISTANCE_EXPONENT,
    WLS_DISTANCE_OFFSET_M,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "raw data"
OUTPUT_DIR = ROOT / "图" / "conditioning_sensitivity"
BASELINE_METRICS = ROOT / "图" / "robustness" / "knn_power_metrics.csv"


@dataclass(frozen=True)
class CutoffSpec:
    label: str
    value: float

    @property
    def finite(self) -> bool:
        return self.value != float("inf")


CUTOFF_SPECS = (
    CutoffSpec("10", 10.0),
    CutoffSpec("30", 30.0),
    CutoffSpec("100", 100.0),
    CutoffSpec("300", 300.0),
    CutoffSpec("1e3", 1.0e3),
    CutoffSpec("1e6", 1.0e6),
    CutoffSpec("1e12", 1.0e12),
    CutoffSpec("inf", float("inf")),
)

# The bins are fixed before analysis and span the current, legacy, and
# effectively unbounded screening regimes used in the reviewer audit.
CONDITION_BINS = (
    ("kappa_lt_10", 0.0, 10.0),
    ("kappa_10_to_30", 10.0, 30.0),
    ("kappa_30_to_100", 30.0, 100.0),
    ("kappa_100_to_300", 100.0, 300.0),
    ("kappa_300_to_1e3", 300.0, 1.0e3),
    ("kappa_1e3_to_1e6", 1.0e3, 1.0e6),
    ("kappa_1e6_to_1e12", 1.0e6, 1.0e12),
    ("kappa_ge_1e12", 1.0e12, float("inf")),
)
