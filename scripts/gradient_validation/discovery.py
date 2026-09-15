from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scripts.analysis.point_cloud import deduplicate_points, standardize_columns
from scripts.robustness.discovery import CaseInput, discover_inputs, manifest_frame


@dataclass(frozen=True)
class LoadedCase:
    record: CaseInput
    frame: pd.DataFrame
    raw_row_count: int


def load_cases() -> tuple[list[CaseInput], dict[int, LoadedCase]]:
    """Load the retained six 0.70 s cases using canonical exact-coordinate consolidation."""
    records = discover_inputs()
    cases: dict[int, LoadedCase] = {}
    for record in records:
        raw = pd.read_csv(record.path)
        standardized = standardize_columns(raw)
        deduplicated = deduplicate_points(standardized, eps_c=0.0)
        if len(deduplicated) >= len(raw):
            raise ValueError(
                f"Expected duplicate-coordinate consolidation for {record.path.name}"
            )
        cases[record.power_W] = LoadedCase(
            record=record,
            frame=deduplicated,
            raw_row_count=len(raw),
        )
    return records, cases


def validation_manifest(
    records: list[CaseInput], cases: dict[int, LoadedCase]
) -> pd.DataFrame:
    manifest = manifest_frame(records).copy()
    manifest["raw_rows"] = [cases[int(power)].raw_row_count for power in manifest["power_W"]]
    manifest["unique_coordinate_rows"] = [
        len(cases[int(power)].frame) for power in manifest["power_W"]
    ]
    manifest["coordinate_consolidation"] = "exact equality (epsilon_c=0)"
    return manifest
