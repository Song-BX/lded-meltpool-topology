from __future__ import annotations

from dataclasses import dataclass


EXPECTED_TIMES = (0.50, 0.55, 0.60, 0.65, 0.70)
PLATEAU_TIMES = (0.60, 0.65, 0.70)
EXPECTED_POWERS = (200, 250, 300, 350, 400, 450)
K_REFERENCE = 25
# The canonical manuscript tables consolidate repeated exported coordinates
# exactly; a non-zero tolerance is available in preprocess.py for sensitivity use.
COORDINATE_TOLERANCE_M = 0.0
FOF_INTERFACE_THRESHOLD = 0.99
# These settings reproduce the canonical k=25 tables used in the submitted
# manuscript. Distance-weight sensitivity is handled separately under R1-7.
WLS_DISTANCE_EXPONENT = 0.0
MIN_NEIGHBOR_DISTANCE_M = 0.0
WLS_CONDITION_MODE = "design"
WLS_CONDITION_CUTOFF = 100.0


@dataclass(frozen=True)
class StabilityRule:
    metric: str
    label: str
    method: str
    threshold: float


STABILITY_RULES = (
    StabilityRule("unique_points", "Unique support", "relative_max_deviation", 0.05),
    StabilityRule("span_x_m", "x span", "relative_max_deviation", 0.05),
    StabilityRule("span_y_m", "y span", "relative_max_deviation", 0.05),
    StabilityRule("span_z_m", "z span", "relative_max_deviation", 0.05),
    StabilityRule("temperature_mean_all_K", "Mean temperature", "relative_max_deviation", 0.05),
    StabilityRule("temperature_max_all_K", "Maximum temperature", "relative_max_deviation", 0.10),
    StabilityRule("velocity_max_all_mps", "Maximum velocity", "relative_max_deviation", 0.10),
    StabilityRule(
        "velocity_mean_interface_mps",
        "Interface mean velocity",
        "relative_max_deviation",
        0.05,
    ),
    StabilityRule("wls_valid_fraction", "WLS-valid fraction", "absolute_range", 0.05),
    StabilityRule("q_positive_fraction_all", "Positive-Q fraction, full-pool", "absolute_range", 0.05),
    StabilityRule(
        "q_positive_fraction_interface",
        "Positive-Q fraction, interface",
        "absolute_range",
        0.05,
    ),
)
