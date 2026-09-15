from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "raw data"
TEMPORAL_DIR = RAW_DIR / "temporal_validation"
OUTPUT_DIR = ROOT / "图" / "thermal_gradient_audit"

POWERS = (200, 250, 300, 350, 400, 450)
TIMES = (0.50, 0.55, 0.60, 0.65, 0.70)
CANONICAL_TIME_S = 0.70
CANONICAL_AGGREGATION = "mean_all_records"
AGGREGATION_STRATEGIES = (
    "mean_all_records",
    "median_all_records",
    "first_record",
    "mean_distinct_states",
)

INTERFACE_FOF_THRESHOLD = 0.99
SUPPORT_GATE = 100
GRADIENT_SOURCE_FIELD = "Temperature Gradient At Tgrdout"
GRADIENT_STANDARD_FIELD = "gradT"

REGIONS = (
    ("full_pool", "full-pool", None),
    ("interface_proxy", "FOF < 0.99 interface proxy", INTERFACE_FOF_THRESHOLD),
)

