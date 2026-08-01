from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scripts.analysis.point_cloud import COLUMN_MAP
from scripts.temporal_validation.discovery import SnapshotFile, discover_snapshots, snapshot_manifest

from .config import GRADIENT_SOURCE_FIELD, RAW_DIR, ROOT, TEMPORAL_DIR


@dataclass(frozen=True)
class GradientInputs:
    snapshots: list[SnapshotFile]
    manifest: pd.DataFrame


def discover_gradient_inputs() -> GradientInputs:
    """Discover the complete temporal grid and confirm its direct gradient field."""
    if GRADIENT_SOURCE_FIELD not in COLUMN_MAP:
        raise ValueError(f"The standardisation map does not define {GRADIENT_SOURCE_FIELD!r}.")

    snapshots = discover_snapshots(RAW_DIR, TEMPORAL_DIR)
    manifest = snapshot_manifest(snapshots, ROOT)
    headers: list[dict[str, object]] = []
    for snapshot in snapshots:
        columns = set(pd.read_csv(snapshot.path, nrows=0).columns)
        if GRADIENT_SOURCE_FIELD not in columns:
            raise ValueError(f"{snapshot.path.name} lacks {GRADIENT_SOURCE_FIELD!r}.")
        headers.append(
            {
                "time_s": snapshot.time_s,
                "power_W": snapshot.power_W,
                "gradient_source_field": GRADIENT_SOURCE_FIELD,
                "gradient_field_present": True,
            }
        )
    manifest = manifest.merge(pd.DataFrame(headers), on=["time_s", "power_W"], validate="one_to_one")
    return GradientInputs(snapshots=snapshots, manifest=manifest.sort_values(["time_s", "power_W"]).reset_index(drop=True))
