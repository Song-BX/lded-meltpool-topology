from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from scripts.analysis.regions import region_mask

from .config import DESCRIPTOR_SPECS, FOF_INTERFACE_THRESHOLD, MIN_REGION_POINTS, REGIONS


def _is_positive(values: np.ndarray, relation: str, threshold: float) -> np.ndarray:
    return values > threshold if relation == "greater" else values < threshold


def _descriptor_spec(name: str):
    return next(spec for spec in DESCRIPTOR_SPECS if spec.name == name)


def summarize_grid(
    grid: Mapping[tuple[int, int], pd.DataFrame],
    *,
    context: str,
    metadata: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize descriptor distributions and within-tensor classification agreement."""
    metadata = dict(metadata or {})
    metric_rows: list[dict[str, object]] = []
    agreement_rows: list[dict[str, object]] = []
    for (power, k), frame in sorted(grid.items()):
        for region in REGIONS:
            mask = region_mask(frame, region, fof_interface_threshold=FOF_INTERFACE_THRESHOLD).to_numpy(dtype=bool)
            values_by_descriptor = {
                spec.name: frame.loc[mask, spec.name].to_numpy(dtype=float)
                for spec in DESCRIPTOR_SPECS
            }
            zero = frame.loc[mask, "zero_tensor"].to_numpy(dtype=bool)
            for spec in DESCRIPTOR_SPECS:
                values = values_by_descriptor[spec.name]
                finite = np.isfinite(values)
                selected = values[finite]
                positive = _is_positive(selected, spec.relation, spec.threshold)
                metric_rows.append(
                    {
                        "context": context,
                        **metadata,
                        "power_W": power,
                        "kNN": k,
                        "region": region,
                        "descriptor": spec.name,
                        "classification": spec.label,
                        "valid_points": int(len(selected)),
                        "support_eligible": bool(len(selected) >= MIN_REGION_POINTS),
                        "positive_count": int(positive.sum()),
                        "positive_fraction": float(positive.mean()) if len(positive) else np.nan,
                        "zero_tensor_count": int(zero[finite].sum()),
                        "value_mean": float(np.mean(selected)) if len(selected) else np.nan,
                        "value_p25": float(np.quantile(selected, 0.25)) if len(selected) else np.nan,
                        "value_median": float(np.median(selected)) if len(selected) else np.nan,
                        "value_p75": float(np.quantile(selected, 0.75)) if len(selected) else np.nan,
                        "value_p90": float(np.quantile(selected, 0.90)) if len(selected) else np.nan,
                        "value_min": float(np.min(selected)) if len(selected) else np.nan,
                        "value_max": float(np.max(selected)) if len(selected) else np.nan,
                    }
                )

            finite_all = mask & np.isfinite(frame[[spec.name for spec in DESCRIPTOR_SPECS]].to_numpy(dtype=float)).all(axis=1)
            nonzero = finite_all & ~frame["zero_tensor"].to_numpy(dtype=bool)
            tensor_energy = frame.loc[nonzero, "tensor_energy"].to_numpy(dtype=float)
            q = frame.loc[nonzero, "Q"].to_numpy(dtype=float)
            omega = frame.loc[nonzero, "omega_normalized"].to_numpy(dtype=float)
            identity_error = np.abs(omega - (0.5 + q / tensor_energy)) if len(q) else np.array([])
            sign_values = {
                spec.name: _is_positive(
                    frame.loc[nonzero, spec.name].to_numpy(dtype=float), spec.relation, spec.threshold
                )
                for spec in DESCRIPTOR_SPECS
            }
            for first, second in (("Q", "lambda2"), ("Q", "omega_normalized"), ("lambda2", "omega_normalized")):
                a = sign_values[first]
                b = sign_values[second]
                agreement_rows.append(
                    {
                        "context": context,
                        **metadata,
                        "power_W": power,
                        "kNN": k,
                        "region": region,
                        "first_descriptor": first,
                        "second_descriptor": second,
                        "comparable_nonzero_points": int(len(a)),
                        "agreement_fraction": float((a == b).mean()) if len(a) else np.nan,
                        "both_positive_count": int((a & b).sum()),
                        "first_only_count": int((a & ~b).sum()),
                        "second_only_count": int((~a & b).sum()),
                        "both_negative_count": int((~a & ~b).sum()),
                        "q_omega_identity_max_abs_error": float(identity_error.max()) if len(identity_error) else np.nan,
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(agreement_rows)


def core_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive 350 W--400 W fraction contrasts without inference."""
    excluded = {
        "power_W", "positive_count", "positive_fraction", "valid_points", "zero_tensor_count",
        "value_mean", "value_p25", "value_median", "value_p75", "value_p90", "value_min", "value_max", "support_eligible",
    }
    index_columns = [column for column in metrics.columns if column not in excluded]
    wide = metrics.pivot_table(
        index=index_columns,
        columns="power_W",
        values="positive_fraction",
        aggfunc="first",
    ).reset_index()
    if 350 not in wide or 400 not in wide:
        return pd.DataFrame()
    wide["phi_350"] = wide[350]
    wide["phi_400"] = wide[400]
    wide["delta_350_400"] = wide[350] - wide[400]
    wide["direction_350_400"] = np.select(
        [wide["delta_350_400"] > 0, wide["delta_350_400"] < 0],
        ["350>400", "350<400"],
        default="tie_or_missing",
    )
    power_columns = [column for column in wide.columns if isinstance(column, (int, np.integer))]
    return wide.drop(columns=power_columns)
