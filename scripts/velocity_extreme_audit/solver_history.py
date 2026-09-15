from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import NORMALIZED_HISTORY_COLUMNS, SOLVER_DIR


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if not name:
        return pd.Series(np.nan, index=frame.index)
    if name not in frame.columns:
        raise ValueError(f"mapped source column {name!r} is absent")
    return frame[name]


def _boolean(values: pd.Series) -> pd.Series:
    truthy = {"1", "true", "yes", "pass", "passed", "accept", "accepted", "success"}
    return values.astype(str).str.strip().str.lower().isin(truthy)


def normalise_solver_history(mapping: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Normalise mapped native CSV logs without guessing FLOW-3D field semantics."""
    if mapping.empty:
        return pd.DataFrame(columns=NORMALIZED_HISTORY_COLUMNS), ("no valid solver-history mapping",)
    chunks: list[pd.DataFrame] = []
    issues: list[str] = []
    for map_row in mapping.itertuples(index=False):
        path = (SOLVER_DIR / str(map_row.raw_file)).resolve()
        if SOLVER_DIR.resolve() not in path.parents or not path.is_file():
            issues.append(f"missing or unsafe mapped solver file for {map_row.power_W} W {map_row.role}")
            continue
        try:
            raw = pd.read_csv(path)
            normalized = pd.DataFrame(
                {
                    "power_W": int(map_row.power_W),
                    "role": str(map_row.role),
                    "time_s": pd.to_numeric(_column(raw, str(map_row.time_column)), errors="coerce"),
                    "iteration": pd.to_numeric(_column(raw, str(map_row.iteration_column)), errors="coerce"),
                    "timestep_s": pd.to_numeric(_column(raw, str(map_row.timestep_column)), errors="coerce"),
                    "run_status": _column(raw, str(map_row.status_column)).astype(str),
                    "variable": _column(raw, str(map_row.variable_column)).astype(str),
                    "value": pd.to_numeric(_column(raw, str(map_row.value_column)), errors="coerce"),
                    "target": pd.to_numeric(_column(raw, str(map_row.target_column)), errors="coerce"),
                    "accepted": _boolean(_column(raw, str(map_row.acceptance_column))),
                    "source_file": path.relative_to(SOLVER_DIR).as_posix(),
                    "source_row": np.arange(len(raw), dtype=int) + 2,
                }
            )
            chunks.append(normalized)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            issues.append(f"could not normalise {map_row.power_W} W {map_row.role}: {error}")
    if not chunks:
        return pd.DataFrame(columns=NORMALIZED_HISTORY_COLUMNS), tuple(issues)
    return pd.concat(chunks, ignore_index=True).loc[:, list(NORMALIZED_HISTORY_COLUMNS)], tuple(issues)

