from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "raw data"
TEMPORAL_DIR = RAW_DIR / "temporal_validation"
FLOW3D_PATH = ROOT / "Flow3D.md"
RUNNING_PATH = ROOT / "running.md"
OUTPUT_DIR = ROOT / "图" / "thermal_fidelity_audit"
TABLE_DIR = ROOT / "latex_restructure" / "tables"

SATURATION_TEMPERATURE_K = 3134.0
HIGH_TEMPERATURE_THRESHOLD_K = 5000.0
CANONICAL_TIME_S = 0.70
EXPECTED_PROGRESS_ROWS = 171
EXPECTED_STABILITY_EVENTS = 2
EXPECTED_FINAL_CYCLE = 18495
EXPECTED_FINAL_TIME_S = 0.700031

# The dictionary order becomes the audit-table order.  Values are parsed from
# the native 300 W input rather than inferred from prose or output fields.
PHASE_CONFIGURATION = (
    ("project", "project", "string", "300 W project identifier"),
    ("phase_change_enabled", "if_phchg", "integer", "native phase-change switch"),
    ("liquidus_temperature_K", "tl1", "float", "native liquidus setting"),
    ("solidus_temperature_K", "ts1", "float", "native solidus setting"),
    ("saturation_pressure_Pa", "pv1", "float", "native saturation-pressure setting"),
    ("saturation_temperature_K", "tv1", "float", "native saturation-temperature setting"),
    ("latent_heat_vaporization_J_per_kg", "clhv1", "float", "native vaporization latent heat"),
    ("recoil_pressure_enabled", "if_prsrecoil", "integer", "native recoil-pressure switch"),
    ("simulation_finish_time_s", "twfin", "float", "configured finish time"),
    ("minimum_timestep_s", "dtmin", "float", "configured minimum time step"),
    ("laser_power_W", "powlbm(1, 1)", "float", "native 300 W heat-source power"),
)

