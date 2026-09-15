from __future__ import annotations

from pathlib import Path

from scripts.velocity_extreme_audit.config import (
    AGGREGATION_STRATEGIES,
    CANONICAL_AGGREGATION,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "raw data"
TEMPORAL_DIR = RAW_DIR / "temporal_validation"
OUTPUT_DIR = ROOT / "图" / "velocity_distribution_overlap_audit"

CANONICAL_TIME_S = 0.70
CANONICAL_STRATEGY = CANONICAL_AGGREGATION
PAIR_POWERS = (350, 400)
TEMPORAL_CONTEXT_TIMES = (0.50, 0.55, 0.60, 0.65)
QUANTILE_LEVELS = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

# The 0.70 s canonical result appears once in the strategy panel.  Earlier
# snapshots provide serial context only and do not duplicate that row.
EXPECTED_AUDIT_ROWS = len(AGGREGATION_STRATEGIES) + len(TEMPORAL_CONTEXT_TIMES)
