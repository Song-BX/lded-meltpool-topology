from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONDITION_BINS, CUTOFF_SPECS


def _finite_values(frame: pd.DataFrame) -> np.ndarray:
    values = frame["kappa"].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def condition_distribution(
    reconstructed: dict[tuple[int, int], pd.DataFrame],
) -> pd.DataFrame:
    """Summarize the unfiltered condition-number distribution by power and k."""
    rows: list[dict[str, float | int]] = []
    for (power, k), frame in sorted(reconstructed.items()):
        values = _finite_values(frame)
        row: dict[str, float | int] = {
            "power_W": power,
            "kNN": k,
            "total_points": len(frame),
            "finite_points": len(values),
            "nonfinite_points": int(len(frame) - len(values)),
        }
        for quantile, label in ((0.50, "p50"), (0.90, "p90"), (0.95, "p95"), (0.99, "p99")):
            row[f"kappa_{label}"] = float(np.quantile(values, quantile)) if len(values) else np.nan
        row["kappa_max"] = float(values.max()) if len(values) else np.nan
        for label, lower, upper in CONDITION_BINS:
            if np.isinf(upper):
                count = int((values >= lower).sum())
            else:
                count = int(((values >= lower) & (values < upper)).sum())
            row[label] = count
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["kNN", "power_W"]).reset_index(drop=True)


def cutoff_point_audit(
    reconstructed: dict[tuple[int, int], pd.DataFrame],
) -> pd.DataFrame:
    """Quantify retained, near-cutoff, and rejected points for every fixed cutoff."""
    rows: list[dict[str, float | int | str]] = []
    for spec in CUTOFF_SPECS:
        for (power, k), frame in sorted(reconstructed.items()):
            values = frame["kappa"].to_numpy(dtype=float)
            finite = np.isfinite(values)
            accepted = finite if not spec.finite else finite & (values <= spec.value)
            exceeded = finite & ~accepted
            near = (
                finite & (values > 0.5 * spec.value) & (values <= spec.value)
                if spec.finite
                else np.zeros(len(frame), dtype=bool)
            )
            rows.append(
                {
                    "cutoff_label": spec.label,
                    "cutoff_value": spec.value,
                    "power_W": power,
                    "kNN": k,
                    "total_points": len(frame),
                    "finite_points": int(finite.sum()),
                    "nonfinite_points": int((~finite).sum()),
                    "retained_points": int(accepted.sum()),
                    "retained_fraction": float(accepted.mean()),
                    "near_cutoff_points": int(near.sum()),
                    "near_cutoff_fraction": float(near.mean()),
                    "exceeded_points": int(exceeded.sum()),
                    "exceeded_fraction": float(exceeded.mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["cutoff_value", "kNN", "power_W"]).reset_index(drop=True)
