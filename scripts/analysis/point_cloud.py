"""Shared point-cloud input normalisation used by the retained R1 analyses.

This module intentionally contains only import-format normalisation and
coordinate consolidation.  It is separated from the retired root-level
workflow scripts so the reproducibility release can run without them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


COLUMN_MAP = {
    "Points_0": "x",
    "Points_1": "y",
    "Points_2": "z",
    "Fraction Of Fluid": "fof",
    "Heat Flux Spatial Distribution": "heat_flux",
    "Temperature": "T",
    "Temperature Gradient At Tgrdout": "gradT",
    "Velocity_0": "u",
    "Velocity_1": "v",
    "Velocity_2": "w",
    "Velocity_Magnitude": "V",
}


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Select and rename the required FLOW-3D CSV fields."""
    missing = [column for column in COLUMN_MAP if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return frame.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].copy()


def deduplicate_points(frame: pd.DataFrame, eps_c: float = 1e-9) -> pd.DataFrame:
    """Consolidate equal coordinates by a per-variable arithmetic mean.

    ``eps_c=0`` performs exact-coordinate consolidation.  A positive value
    creates stable rounded coordinate keys before aggregation.
    """
    work = frame.copy()
    if eps_c <= 0:
        group_columns = ["x", "y", "z"]
    else:
        coordinates = frame[["x", "y", "z"]].to_numpy()
        keys = np.round(coordinates / eps_c).astype(np.int64)
        work[["kx", "ky", "kz"]] = keys
        group_columns = ["kx", "ky", "kz"]
    return (
        work.groupby(group_columns, sort=False)[list(COLUMN_MAP.values())]
        .mean()
        .reset_index(drop=True)
    )
