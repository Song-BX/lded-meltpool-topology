from __future__ import annotations

import numpy as np
import pandas as pd

from .tensor_metrics import tensor_metrics_from_velocity_gradients

def nearest_neighbor_indices(points: np.ndarray, k: int) -> np.ndarray:
    """Return each point followed by its k nearest neighbours.

    The retained point clouds contain only a few hundred coordinates.  A full,
    stable NumPy ordering is therefore inexpensive and intentionally preferred
    to tree queries: exact-distance ties at regular-grid locations are broken
    by input index, rather than by a SciPy-version-dependent tree traversal.
    """
    if k >= len(points):
        raise ValueError(f"k={k} requires at least {k + 1} points; received {len(points)}")
    deltas = points[:, None, :] - points[None, :, :]
    squared_distances = np.einsum("ijk,ijk->ij", deltas, deltas)
    return np.argsort(squared_distances, axis=1, kind="mergesort")[:, : k + 1]


def reconstruct_case(
    frame: pd.DataFrame,
    k: int,
    *,
    alpha: float = 0.0,
    eps_w: float = 1e-12,
    kappa_max: float = 100.0,
    min_neighbor_distance: float = 0.0,
    condition_on: str = "design",
) -> pd.DataFrame:
    """Reconstruct velocity gradients and Q on one deduplicated point cloud.

    The default arguments are the canonical retained-manuscript baseline:
    unweighted local least squares (alpha=0) and
    cond(sqrt(W) A) <= 100.
    """
    points = frame[["x", "y", "z"]].to_numpy(dtype=float)
    indices = nearest_neighbor_indices(points, k=k)[:, 1:]
    return reconstruct_case_from_neighbor_sets(
        frame,
        indices,
        k=k,
        alpha=alpha,
        eps_w=eps_w,
        kappa_max=kappa_max,
        min_neighbor_distance=min_neighbor_distance,
        condition_on=condition_on,
    )


def reconstruct_case_from_neighbor_sets(
    frame: pd.DataFrame,
    neighbour_sets: np.ndarray,
    *,
    k: int | None = None,
    alpha: float = 0.0,
    eps_w: float = 1e-12,
    kappa_max: float = 100.0,
    min_neighbor_distance: float = 0.0,
    condition_on: str = "design",
) -> pd.DataFrame:
    """Reconstruct a first-order field from explicit, centre-excluded neighbour sets.

    This is the canonical WLS implementation used both by the retained kNN scan
    and by geometry-resampling diagnostics.  ``neighbour_sets`` has one row per
    centre point and contains exactly the neighbours selected for that centre.
    """
    points = frame[["x", "y", "z"]].to_numpy(dtype=float)
    neighbours_by_point = np.asarray(neighbour_sets, dtype=int)
    if neighbours_by_point.ndim != 2 or neighbours_by_point.shape[0] != len(frame):
        raise ValueError(
            "neighbour_sets must have shape (number of points, number of neighbours)"
        )
    if neighbours_by_point.shape[1] < 3:
        raise ValueError("At least three neighbours are required for a 3D first-order fit")
    if (neighbours_by_point < 0).any() or (neighbours_by_point >= len(frame)).any():
        raise ValueError("neighbour_sets contains an out-of-range point index")
    if np.any(neighbours_by_point == np.arange(len(frame))[:, None]):
        raise ValueError("neighbour_sets must exclude each centre point")

    gradients = {
        "u": np.full((len(frame), 3), np.nan),
        "v": np.full((len(frame), 3), np.nan),
        "w": np.full((len(frame), 3), np.nan),
    }
    kappa = np.full(len(frame), np.nan)
    valid = np.zeros(len(frame), dtype=int)
    values = {component: frame[component].to_numpy(dtype=float) for component in gradients}

    for i in range(len(frame)):
        neighbours = neighbours_by_point[i]
        design = points[neighbours] - points[i]
        distances = np.linalg.norm(design, axis=1)
        if float(np.min(distances)) <= min_neighbor_distance:
            continue

        weights = 1.0 / np.power(distances + eps_w, alpha)
        sqrt_weights = np.sqrt(weights)
        weighted_design = sqrt_weights[:, None] * design
        if condition_on == "design":
            condition_matrix = weighted_design
        elif condition_on == "normal":
            condition_matrix = design.T @ (weights[:, None] * design)
        else:
            raise ValueError(f"Unknown condition_on mode: {condition_on}")

        try:
            condition_number = float(np.linalg.cond(condition_matrix))
        except np.linalg.LinAlgError:
            condition_number = np.inf
        kappa[i] = condition_number
        if not np.isfinite(condition_number) or condition_number > kappa_max:
            continue

        valid[i] = 1
        pseudo_inverse = np.linalg.pinv(weighted_design)
        for component, target in gradients.items():
            differences = values[component][neighbours] - values[component][i]
            target[i] = pseudo_inverse @ (sqrt_weights * differences)

    output = frame.copy()
    output["k"] = int(k if k is not None else neighbours_by_point.shape[1])
    output["kappa"] = kappa
    output["chi"] = valid
    output[["du_dx", "du_dy", "du_dz"]] = gradients["u"]
    output[["dv_dx", "dv_dy", "dv_dz"]] = gradients["v"]
    output[["dw_dx", "dw_dy", "dw_dz"]] = gradients["w"]

    velocity_gradients = np.stack(
        [gradients["u"], gradients["v"], gradients["w"]], axis=1
    )
    tensor_metrics = tensor_metrics_from_velocity_gradients(velocity_gradients)
    for name, values in tensor_metrics.items():
        output[name] = values
    return output
