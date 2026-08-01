from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.robustness.config import (
    EXPECTED_POWERS,
    FOF_INTERFACE_THRESHOLD,
    K_REFERENCE,
    K_VALUES,
    WLS_CONDITION_CUTOFF,
    WLS_CONDITION_MODE,
    WLS_DISTANCE_EXPONENT,
    WLS_DISTANCE_OFFSET_M,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "raw data"
OUTPUT_DIR = ROOT / "图" / "gradient_validation"
NATIVE_REFERENCE_DIR = RAW_DIR / "native_gradient_validation"

REGIONS = ("all", "interface")
QUADRATIC_K_VALUES = tuple(value for value in K_VALUES if value >= 15)
RESAMPLE_COUNT = 200
RESAMPLE_NEIGHBOURS = 20
RESAMPLE_SEED = 20260728
Q_MARGIN_FRACTION = 0.05
AFFINE_NUMERICAL_TOLERANCE = 1e-10


@dataclass(frozen=True)
class FieldSpec:
    field_id: str
    field_class: str
    scale_m: float | None


AFFINE_FIELDS = (
    FieldSpec("affine_rotation", "affine", None),
    FieldSpec("affine_strain", "affine", None),
    FieldSpec("simple_shear_zero_q", "affine", None),
)
NONLINEAR_FIELDS = tuple(
    FieldSpec(field_id, "nonlinear", scale)
    for field_id in ("gaussian_vortex", "tanh_shear")
    for scale in (1.0e-4, 2.0e-4, 3.0e-4)
)
FIELD_SPECS = AFFINE_FIELDS + NONLINEAR_FIELDS
