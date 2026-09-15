from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.wls_q import nearest_neighbor_indices

from .config import ALPHA_SPECS, K_VALUES, WLS_DISTANCE_OFFSET_M


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if len(values) else np.nan


def summarize_weight_geometry(
    cases: dict[int, pd.DataFrame],
    reconstructed: dict[tuple[str, int, int], pd.DataFrame],
) -> pd.DataFrame:
    """Summarize support, conditioning, and distance-weight concentration."""
    rows: list[dict[str, float | int | str]] = []
    for power, frame in sorted(cases.items()):
        points = frame[["x", "y", "z"]].to_numpy(dtype=float)
        for k in K_VALUES:
            indices = nearest_neighbor_indices(points, k=k)[:, 1:]
            offsets = points[indices] - points[:, None, :]
            distances = np.linalg.norm(offsets, axis=2)
            kth_radius_mm = distances.max(axis=1) * 1_000.0
            for alpha in ALPHA_SPECS:
                weights = 1.0 / np.power(distances + WLS_DISTANCE_OFFSET_M, alpha.value)
                normalized = weights / weights.sum(axis=1, keepdims=True)
                effective_neighbours = 1.0 / np.square(normalized).sum(axis=1)
                max_weight = normalized.max(axis=1)
                result = reconstructed[(alpha.label, power, k)]
                kappa = result["kappa"].to_numpy(dtype=float)
                finite_kappa = kappa[np.isfinite(kappa)]
                valid = result["chi"].to_numpy(dtype=int) == 1
                rows.append(
                    {
                        "alpha_label": alpha.label,
                        "alpha": alpha.value,
                        "alpha_role": alpha.role,
                        "power_W": power,
                        "kNN": k,
                        "points": len(frame),
                        "wls_valid_points": int(valid.sum()),
                        "wls_valid_fraction": float(valid.mean()),
                        "kth_radius_mm_median": _quantile(kth_radius_mm, 0.50),
                        "kth_radius_mm_p90": _quantile(kth_radius_mm, 0.90),
                        "max_normalized_weight_median": _quantile(max_weight, 0.50),
                        "max_normalized_weight_p90": _quantile(max_weight, 0.90),
                        "effective_neighbours_median": _quantile(effective_neighbours, 0.50),
                        "effective_neighbours_p10": _quantile(effective_neighbours, 0.10),
                        "kappa_finite_count": int(len(finite_kappa)),
                        "kappa_p50": _quantile(finite_kappa, 0.50),
                        "kappa_p90": _quantile(finite_kappa, 0.90),
                        "kappa_p95": _quantile(finite_kappa, 0.95),
                        "kappa_p99": _quantile(finite_kappa, 0.99),
                        "kappa_max": float(finite_kappa.max()) if len(finite_kappa) else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(["alpha", "power_W", "kNN"]).reset_index(drop=True)

