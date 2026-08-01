from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.wls_q import reconstruct_case
from scripts.gradient_validation.manufactured_fields import build_manufactured_field
from scripts.gradient_validation.metrics import manufactured_metrics

from .config import (
    ALPHA_SPECS,
    FIELD_SPECS,
    K_VALUES,
    WLS_CONDITION_CUTOFF,
    WLS_CONDITION_MODE,
    WLS_DISTANCE_OFFSET_M,
)


def run_manufactured_field_audit(cases: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all pre-specified exponents on analytic fields at observed geometries."""
    blocks: list[pd.DataFrame] = []
    for alpha in ALPHA_SPECS:
        for power, frame in sorted(cases.items()):
            for spec in FIELD_SPECS:
                truth = build_manufactured_field(frame, spec)
                for k in K_VALUES:
                    reconstructed = reconstruct_case(
                        truth.frame,
                        k=k,
                        alpha=alpha.value,
                        eps_w=WLS_DISTANCE_OFFSET_M,
                        kappa_max=WLS_CONDITION_CUTOFF,
                        condition_on=WLS_CONDITION_MODE,
                    )
                    metric = manufactured_metrics(reconstructed, truth, power_W=power, k=k)
                    metric.insert(0, "alpha_label", alpha.label)
                    metric.insert(1, "alpha", alpha.value)
                    metric.insert(2, "alpha_role", alpha.role)
                    blocks.append(metric)
                print(
                    f"completed manufactured field {spec.field_id} for {power} W at alpha={alpha.label}",
                    flush=True,
                )
    metrics = pd.concat(blocks, ignore_index=True).sort_values(
        ["alpha", "power_W", "field_id", "kNN", "region"]
    ).reset_index(drop=True)
    summary = (
        metrics.groupby(
            ["alpha_label", "alpha", "alpha_role", "field_id", "field_class", "feature_scale_mm", "region"],
            dropna=False,
            as_index=False,
        )
        .agg(
            gradient_nrmse_median=("gradient_nrmse", "median"),
            gradient_nrmse_p90=("gradient_nrmse", lambda value: value.quantile(0.90)),
            q_nrmse_median=("q_nrmse", "median"),
            q_sign_accuracy_margin_median=("q_sign_accuracy_margin", "median"),
            q_sign_accuracy_margin_min=("q_sign_accuracy_margin", "min"),
            valid_fraction_min=("valid_fraction", "min"),
        )
        .sort_values(["alpha", "field_class", "field_id", "feature_scale_mm", "region"])
        .reset_index(drop=True)
    )
    return metrics, summary
