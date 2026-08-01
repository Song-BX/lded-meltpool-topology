from __future__ import annotations

from pathlib import Path

from scripts.analysis.release_paths import reference_input


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "raw data"
TEMPORAL_DIR = RAW_DIR / "temporal_validation"
SOLVER_DIR = RAW_DIR / "solver_numerics"
MAPPING_PATH = SOLVER_DIR / "solver_history_mapping.csv"
OUTPUT_DIR = ROOT / "图" / "velocity_extreme_audit"
CANONICAL_METRICS = reference_input(
    ROOT, "Aplus_main_metrics_k25.csv", Path("图/3/Aplus_main_metrics_k25.csv")
)

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
VELOCITY_QUANTILES = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
PEAK_POWERS = (350, 400)

# A native-history mapping is deliberately explicit.  The data exported by different
# FLOW-3D versions vary, so unlabelled columns must never be guessed as solver health
# variables.  Each power-role pair must be represented exactly once in a supplied map.
SOLVER_ROLES = (
    "configuration",
    "run_history",
    "residual",
    "stability",
    "conservation",
)
MAPPING_COLUMNS = (
    "power_W",
    "role",
    "raw_file",
    "time_column",
    "iteration_column",
    "timestep_column",
    "status_column",
    "variable_column",
    "value_column",
    "target_column",
    "acceptance_column",
)
NORMALIZED_HISTORY_COLUMNS = (
    "power_W",
    "role",
    "time_s",
    "iteration",
    "timestep_s",
    "run_status",
    "variable",
    "value",
    "target",
    "accepted",
    "source_file",
    "source_row",
)
POWER_CONFIGURATION_KEYS = frozenset(
    {"laser_power", "laser_power_w", "laser power", "power", "power_w", "power w"}
)
HEALTH_WINDOW = (0.50, 0.70)
