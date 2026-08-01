from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.analysis.release_paths import reference_input


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "图" / "power_response_audit"
CANONICAL_METRICS = reference_input(
    ROOT, "Aplus_main_metrics_k25.csv", Path("图/3/Aplus_main_metrics_k25.csv")
)
TEMPORAL_METRICS = ROOT / "图" / "s4" / "temporal_metrics.csv"
AGGREGATION_METRICS = ROOT / "图" / "export_diagnostics" / "aggregation_k25_metrics.csv"
THERMAL_TAIL_METRICS = ROOT / "图" / "thermal_fidelity_audit" / "temperature_tail_metrics.csv"
RAW_DIR = ROOT / "raw data"
TEMPORAL_DIR = RAW_DIR / "temporal_validation"

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


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    label: str
    unit: str
    region: str
    canonical_column: str
    temporal_column: str
    aggregation_column: str
    interpretation_status: str
    interpretation_boundary: str


METRICS = (
    MetricSpec(
        "temperature_median_full_pool_K",
        "Full-pool median temperature (unfiltered)",
        "K",
        "all",
        "T_median_K",
        "T_median_K",
        "T_median_K",
        "snapshot_local_descriptor",
        "An unfiltered central-temperature descriptor of one numerical export; not a continuous response, regime boundary, or physical-fidelity result.",
    ),
    MetricSpec(
        "temperature_mean_full_pool_K",
        "Full-pool mean temperature",
        "K",
        "all",
        "T_mean_K",
        "temperature_mean_all_K",
        "T_mean_K",
        "snapshot_local_descriptor",
        "A discrete late-time thermal descriptor; not a continuous response, regime boundary, or physical-fidelity result.",
    ),
    MetricSpec(
        "velocity_max_full_pool_mps",
        "Full-pool maximum velocity",
        "m s^-1",
        "all",
        "v_max",
        "velocity_max_all_mps",
        "v_max",
        "audit_only",
        "A sparse peak-level audit descriptor; it does not represent central full-pool velocity separation or a physical regime.",
    ),
    MetricSpec(
        "velocity_mean_interface_mps",
        "Interface mean velocity",
        "m s^-1",
        "interface",
        "v_mean",
        "velocity_mean_interface_mps",
        "v_mean",
        "snapshot_local_descriptor",
        "A discrete late-time interface-proxy velocity descriptor; not a continuous response, regime boundary, or physical-fidelity result.",
    ),
)

METRIC_BY_ID = {metric.metric_id: metric for metric in METRICS}
