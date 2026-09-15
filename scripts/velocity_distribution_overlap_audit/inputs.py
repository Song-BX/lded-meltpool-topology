from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scripts.temporal_validation.discovery import (
    SnapshotFile,
    discover_snapshots,
    snapshot_manifest,
)

from .config import RAW_DIR, ROOT, TEMPORAL_DIR


@dataclass(frozen=True)
class DistributionAuditInputs:
    snapshots: list[SnapshotFile]
    manifest: pd.DataFrame


def load_inputs() -> DistributionAuditInputs:
    """Discover the complete matched point-cloud grid and preserve its hashes."""
    snapshots = discover_snapshots(RAW_DIR, TEMPORAL_DIR)
    manifest = snapshot_manifest(snapshots, ROOT)
    if len(manifest) != 30:
        raise ValueError("The velocity-distribution audit requires all 30 time-power snapshots.")
    return DistributionAuditInputs(snapshots=snapshots, manifest=manifest)
