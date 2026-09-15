from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.conditioning_sensitivity.config import CUTOFF_SPECS
from scripts.gradient_validation.config import FIELD_SPECS, QUADRATIC_K_VALUES, RESAMPLE_COUNT, RESAMPLE_NEIGHBOURS, RESAMPLE_SEED
from scripts.robustness.config import (
    CORE_POWER_HIGH,
    CORE_POWER_LOW,
    EXPECTED_POWERS,
    FOF_INTERFACE_THRESHOLD,
    K_REFERENCE,
    K_VALUES,
    MIN_REGION_POINTS,
    WLS_CONDITION_CUTOFF,
    WLS_CONDITION_MODE,
    WLS_DISTANCE_OFFSET_M,
)
from scripts.weight_exponent_sensitivity.config import ALPHA_SPECS


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "图" / "complementary_descriptor_audit"
BASELINE_Q_METRICS = ROOT / "图" / "robustness" / "knn_power_metrics.csv"
WEIGHT_DECISION = ROOT / "图" / "weight_exponent_sensitivity" / "weight_exponent_decision.json"
TEMPORAL_DECISION = ROOT / "图" / "s4" / "temporal_validation_decision.json"

REGIONS = ("all", "interface")
DESCRIPTORS = ("Q", "lambda2", "omega_normalized")


@dataclass(frozen=True)
class DescriptorSpec:
    name: str
    label: str
    relation: str
    threshold: float


DESCRIPTOR_SPECS = (
    DescriptorSpec("Q", r"$Q>0$", "greater", 0.0),
    DescriptorSpec("lambda2", r"$\lambda_2<0$", "less", 0.0),
    DescriptorSpec("omega_normalized", r"$\Omega_N>0.5$", "greater", 0.5),
)
