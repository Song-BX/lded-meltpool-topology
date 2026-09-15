from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.wls_q import nearest_neighbor_indices

from .config import FOF_INTERFACE_THRESHOLD, K_VALUES


def neighbourhood_geometry(frame: pd.DataFrame, power_W: int) -> pd.DataFrame:
    """Describe local support geometry for every centre point and tested k."""
    points = frame[["x", "y", "z"]].to_numpy(dtype=float)
    max_k = max(K_VALUES)
    indices = nearest_neighbor_indices(points, k=max_k)[:, 1:]
    rows: list[dict[str, float | int | bool]] = []
    interface = frame["fof"].to_numpy(dtype=float) < FOF_INTERFACE_THRESHOLD
    for k in K_VALUES:
        neighbour_indices = indices[:, :k]
        offsets = points[neighbour_indices] - points[:, None, :]
        radii = np.linalg.norm(offsets, axis=2)
        for index, (local_offsets, local_radii) in enumerate(zip(offsets, radii)):
            covariance = local_offsets.T @ local_offsets / float(k)
            eigenvalues = np.linalg.eigvalsh(covariance)
            max_eigenvalue = float(eigenvalues[-1])
            min_eigenvalue = float(eigenvalues[0])
            rows.append(
                {
                    "power_W": power_W,
                    "kNN": k,
                    "point_index": index,
                    "is_interface": bool(interface[index]),
                    "kth_radius_mm": float(np.max(local_radii) * 1_000.0),
                    "condition_design": float(np.linalg.cond(local_offsets)),
                    "covariance_eigenvalue_min": min_eigenvalue,
                    "covariance_eigenvalue_max": max_eigenvalue,
                    "eigenvalue_ratio_min_max": (
                        min_eigenvalue / max_eigenvalue if max_eigenvalue > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)
