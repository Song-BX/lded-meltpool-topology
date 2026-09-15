from __future__ import annotations

from pathlib import Path

from scripts.analysis.release_paths import reference_input


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "raw data"
OUTPUT_DIR = ROOT / "图" / "robustness"
CANONICAL_METRICS = reference_input(
    ROOT, "Aplus_main_metrics_k25.csv", Path("图/3/Aplus_main_metrics_k25.csv")
)
RETAINED_SENSITIVITY = reference_input(
    ROOT,
    "Aplus_Qthreshold_sensitivity_350vs400.csv",
    Path("图/7/Aplus_Qthreshold_sensitivity_350vs400.csv"),
)

EXPECTED_POWERS = (200, 250, 300, 350, 400, 450)
K_VALUES = tuple(range(8, 51))
K_REFERENCE = 25
REGIONS = ("all", "interface", "heated", "interface_heated")
THRESHOLDS = ("Q>0", "Q>posP50", "Q>posP75", "Q>posP90")
PERCENTILES = {"Q>posP50": 0.50, "Q>posP75": 0.75, "Q>posP90": 0.90}

# Reviewer #1 Comment 4: conservative evidence-support gates.  These gates
# control numerical resolution only; they do not imply spatial independence.
MIN_REGION_POINTS = 100
MIN_POOLED_EXCEEDANCES = 10
AUDIT_ONLY_REGIONS = ("heated", "interface_heated")

COORDINATE_TOLERANCE_M = 0.0
FOF_INTERFACE_THRESHOLD = 0.99
HEAT_FLUX_THRESHOLD = 0.0
WLS_DISTANCE_EXPONENT = 0.0
WLS_DISTANCE_OFFSET_M = 1e-12
WLS_CONDITION_MODE = "design"
WLS_CONDITION_CUTOFF = 100.0
GRID_SPACING_MM = 0.1

CORE_POWER_LOW = 350
CORE_POWER_HIGH = 400
