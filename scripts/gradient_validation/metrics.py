from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.regions import region_mask

from .config import FOF_INTERFACE_THRESHOLD, Q_MARGIN_FRACTION, REGIONS
from .manufactured_fields import ManufacturedField


GRADIENT_COLUMNS = (
    ("du_dx", "du_dy", "du_dz"),
    ("dv_dx", "dv_dy", "dv_dz"),
    ("dw_dx", "dw_dy", "dw_dz"),
)


def gradients_from_reconstruction(frame: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [frame[list(component_columns)].to_numpy(dtype=float) for component_columns in GRADIENT_COLUMNS],
        axis=1,
    )


def _balanced_accuracy(truth_positive: np.ndarray, predicted_positive: np.ndarray) -> float:
    positive = truth_positive
    negative = ~truth_positive
    if not positive.any() or not negative.any():
        return np.nan
    sensitivity = float(predicted_positive[positive].mean())
    specificity = float((~predicted_positive[negative]).mean())
    return 0.5 * (sensitivity + specificity)


def manufactured_metrics(
    reconstructed: pd.DataFrame,
    truth: ManufacturedField,
    *,
    power_W: int,
    k: int,
) -> pd.DataFrame:
    """Compare reconstructed and analytic gradient/Q fields in the retained regions."""
    estimated_gradients = gradients_from_reconstruction(reconstructed)
    true_gradients = truth.true_gradients
    estimated_q = reconstructed["Q"].to_numpy(dtype=float)
    true_q = truth.true_q
    rows: list[dict[str, object]] = []

    for region in REGIONS:
        mask = region_mask(
            reconstructed,
            region,
            fof_interface_threshold=FOF_INTERFACE_THRESHOLD,
        ).to_numpy(dtype=bool)
        valid_count = int(mask.sum())
        finite = mask & np.isfinite(estimated_gradients).all(axis=(1, 2))
        compared_count = int(finite.sum())
        true_subset = true_gradients[finite]
        estimated_subset = estimated_gradients[finite]
        gradient_difference = estimated_subset - true_subset
        true_norm = np.linalg.norm(true_subset.reshape(compared_count, -1), axis=1)
        error_norm = np.linalg.norm(gradient_difference.reshape(compared_count, -1), axis=1)
        gradient_nrmse = (
            float(np.sqrt(np.mean(error_norm**2)) / np.sqrt(np.mean(true_norm**2)))
            if compared_count and np.any(true_norm > 0)
            else np.nan
        )
        pointwise_relative = error_norm / np.maximum(true_norm, 1e-12)

        true_q_subset = true_q[finite]
        estimated_q_subset = estimated_q[finite]
        q_difference = estimated_q_subset - true_q_subset
        q_denominator = float(np.sqrt(np.mean(true_q_subset**2))) if compared_count else np.nan
        q_nrmse = (
            float(np.sqrt(np.mean(q_difference**2)) / q_denominator)
            if np.isfinite(q_denominator) and q_denominator > 0
            else np.nan
        )
        truth_positive = true_q_subset > 0
        predicted_positive = estimated_q_subset > 0
        q_sign_accuracy = (
            float((truth_positive == predicted_positive).mean()) if compared_count else np.nan
        )
        q_balanced_accuracy = _balanced_accuracy(truth_positive, predicted_positive)
        q_scale = float(np.max(np.abs(true_q_subset))) if compared_count else np.nan
        margin = Q_MARGIN_FRACTION * q_scale if np.isfinite(q_scale) else np.nan
        margin_mask = np.abs(true_q_subset) >= margin if np.isfinite(margin) else np.zeros(compared_count, dtype=bool)
        margin_count = int(margin_mask.sum())
        q_sign_accuracy_margin = (
            float((truth_positive[margin_mask] == predicted_positive[margin_mask]).mean())
            if margin_count
            else np.nan
        )
        q_balanced_accuracy_margin = (
            _balanced_accuracy(truth_positive[margin_mask], predicted_positive[margin_mask])
            if margin_count
            else np.nan
        )
        finite_kappa = reconstructed.loc[mask & np.isfinite(reconstructed["kappa"]), "kappa"]
        rows.append(
            {
                "power_W": power_W,
                "kNN": k,
                "region": region,
                "field_id": truth.spec.field_id,
                "field_class": truth.spec.field_class,
                "feature_scale_mm": (
                    np.nan if truth.spec.scale_m is None else truth.spec.scale_m * 1_000.0
                ),
                "valid_points": valid_count,
                "compared_points": compared_count,
                "valid_fraction": valid_count / len(reconstructed),
                "gradient_nrmse": gradient_nrmse,
                "gradient_relative_error_median": (
                    float(np.median(pointwise_relative)) if compared_count else np.nan
                ),
                "gradient_relative_error_p90": (
                    float(np.quantile(pointwise_relative, 0.90)) if compared_count else np.nan
                ),
                "q_nrmse": q_nrmse,
                "q_sign_accuracy": q_sign_accuracy,
                "q_balanced_accuracy": q_balanced_accuracy,
                "q_margin_fraction": Q_MARGIN_FRACTION,
                "q_margin_points": margin_count,
                "q_sign_accuracy_margin": q_sign_accuracy_margin,
                "q_balanced_accuracy_margin": q_balanced_accuracy_margin,
                "kappa_median": float(finite_kappa.median()) if len(finite_kappa) else np.nan,
                "kappa_p90": float(finite_kappa.quantile(0.90)) if len(finite_kappa) else np.nan,
            }
        )
    return pd.DataFrame(rows)
