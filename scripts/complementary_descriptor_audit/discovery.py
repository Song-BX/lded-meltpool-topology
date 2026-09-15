from __future__ import annotations

import pandas as pd

from scripts.analysis.point_cloud import deduplicate_points, standardize_columns
from scripts.robustness.discovery import CaseInput, discover_inputs, manifest_frame


def load_canonical_cases(records: list[CaseInput]) -> dict[int, pd.DataFrame]:
    """Load and exactly consolidate the six frozen 0.70 s inputs."""
    cases: dict[int, pd.DataFrame] = {}
    for record in records:
        raw = pd.read_csv(record.path)
        cases[record.power_W] = deduplicate_points(standardize_columns(raw), eps_c=0.0)
    return cases


def load_raw_cases(records: list[CaseInput]) -> dict[int, pd.DataFrame]:
    """Load standardized, unconsolidated inputs for the fixed aggregation audit."""
    return {
        record.power_W: standardize_columns(pd.read_csv(record.path))
        for record in records
    }


__all__ = ["discover_inputs", "manifest_frame", "load_canonical_cases", "load_raw_cases"]
