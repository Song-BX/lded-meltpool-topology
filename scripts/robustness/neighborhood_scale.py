from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GRID_SPACING_MM, K_VALUES


def _sorted_neighbor_distances(points: np.ndarray) -> np.ndarray:
    deltas = points[:, None, :] - points[None, :, :]
    squared = np.einsum("ijk,ijk->ij", deltas, deltas)
    np.fill_diagonal(squared, np.inf)
    return np.sqrt(np.sort(squared, axis=1, kind="mergesort"))


def compute_neighborhood_scales(cases: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for power, frame in sorted(cases.items()):
        points = frame[["x", "y", "z"]].to_numpy(dtype=float)
        if max(K_VALUES) >= len(points):
            raise ValueError(
                f"{power} W has {len(points)} unique points, insufficient for k={max(K_VALUES)}"
            )
        sorted_distances = _sorted_neighbor_distances(points)
        for k in K_VALUES:
            radii_m = sorted_distances[:, k - 1]
            median_mm = float(np.median(radii_m) * 1000)
            p90_mm = float(np.quantile(radii_m, 0.90) * 1000)
            rows.append(
                {
                    "power_W": power,
                    "kNN": k,
                    "unique_points": len(frame),
                    "radius_median_mm": median_mm,
                    "radius_p90_mm": p90_mm,
                    "radius_min_mm": float(np.min(radii_m) * 1000),
                    "radius_max_mm": float(np.max(radii_m) * 1000),
                    "radius_median_grid_spacings": median_mm / GRID_SPACING_MM,
                    "radius_p90_grid_spacings": p90_mm / GRID_SPACING_MM,
                }
            )
    return pd.DataFrame(rows).sort_values(["power_W", "kNN"]).reset_index(drop=True)
