from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.tensor_metrics import tensor_metrics_from_velocity_gradients
from scripts.analysis.wls_q import nearest_neighbor_indices


def _quadratic_design(normalized_offsets: np.ndarray) -> np.ndarray:
    x, y, z = normalized_offsets.T
    return np.column_stack(
        (x, y, z, 0.5 * x**2, x * y, x * z, 0.5 * y**2, y * z, 0.5 * z**2)
    )


def reconstruct_quadratic_case(
    frame: pd.DataFrame,
    k: int,
    *,
    kappa_max: float,
) -> pd.DataFrame:
    """Second-order local comparison fit with scaled offsets and no regularization."""
    if k < 9:
        raise ValueError("A three-dimensional quadratic local fit requires at least nine neighbours")
    points = frame[["x", "y", "z"]].to_numpy(dtype=float)
    indices = nearest_neighbor_indices(points, k=k)[:, 1:]
    gradients = np.full((len(frame), 3, 3), np.nan)
    kappa = np.full(len(frame), np.nan)
    valid = np.zeros(len(frame), dtype=int)
    values = frame[["u", "v", "w"]].to_numpy(dtype=float)

    for centre, neighbours in enumerate(indices):
        offsets = points[neighbours] - points[centre]
        support_radius = float(np.max(np.linalg.norm(offsets, axis=1)))
        if not np.isfinite(support_radius) or support_radius <= 0:
            continue
        design = _quadratic_design(offsets / support_radius)
        condition_number = float(np.linalg.cond(design))
        kappa[centre] = condition_number
        if not np.isfinite(condition_number) or condition_number > kappa_max:
            continue
        coefficients = np.linalg.pinv(design) @ (values[neighbours] - values[centre])
        gradients[centre] = (coefficients[:3, :] / support_radius).T
        valid[centre] = 1

    output = frame.copy()
    output["k"] = k
    output["kappa"] = kappa
    output["chi"] = valid
    output[["du_dx", "du_dy", "du_dz"]] = gradients[:, 0, :]
    output[["dv_dx", "dv_dy", "dv_dz"]] = gradients[:, 1, :]
    output[["dw_dx", "dw_dy", "dw_dz"]] = gradients[:, 2, :]
    tensor_metrics = tensor_metrics_from_velocity_gradients(gradients)
    for name, values in tensor_metrics.items():
        output[name] = values
    return output
