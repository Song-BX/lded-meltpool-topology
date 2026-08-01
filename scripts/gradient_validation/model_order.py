from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.regions import region_mask
from scripts.analysis.wls_q import reconstruct_case
from scripts.robustness.knn_scan import _pooled_thresholds

from .config import (
    FOF_INTERFACE_THRESHOLD,
    K_REFERENCE,
    QUADRATIC_K_VALUES,
    REGIONS,
    WLS_CONDITION_CUTOFF,
    WLS_CONDITION_MODE,
    WLS_DISTANCE_EXPONENT,
    WLS_DISTANCE_OFFSET_M,
)
from .quadratic import reconstruct_quadratic_case


THRESHOLDS = ("Q>0", "Q>posP50", "Q>posP75", "Q>posP90")


def _direction(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value > 0:
        return "350>400"
    if value < 0:
        return "350<400"
    return "tie"


def _fraction_rows(
    reconstructed: dict[int, pd.DataFrame], *, method: str, k: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    thresholds = {region: _pooled_thresholds(reconstructed, region) for region in REGIONS}
    for power, frame in sorted(reconstructed.items()):
        for region in REGIONS:
            mask = region_mask(
                frame, region, fof_interface_threshold=FOF_INTERFACE_THRESHOLD
            )
            values = frame.loc[mask, "Q"].dropna()
            for threshold in THRESHOLDS:
                threshold_value = thresholds[region][threshold]
                rows.append(
                    {
                        "method": method,
                        "kNN": k,
                        "power_W": power,
                        "region": region,
                        "threshold": threshold,
                        "threshold_value": threshold_value,
                        "q_fraction": float((values > threshold_value).mean()) if len(values) else np.nan,
                        "n_region": int(len(values)),
                    }
                )
    return pd.DataFrame(rows)


def run_model_order_comparison(
    cases: dict[int, pd.DataFrame], eligibility: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare canonical first-order and scaled second-order local fits."""
    metric_blocks: list[pd.DataFrame] = []
    for k in QUADRATIC_K_VALUES:
        first = {
            power: reconstruct_case(
                frame,
                k=k,
                alpha=WLS_DISTANCE_EXPONENT,
                eps_w=WLS_DISTANCE_OFFSET_M,
                kappa_max=WLS_CONDITION_CUTOFF,
                condition_on=WLS_CONDITION_MODE,
            )
            for power, frame in sorted(cases.items())
        }
        quadratic = {
            power: reconstruct_quadratic_case(
                frame, k=k, kappa_max=WLS_CONDITION_CUTOFF
            )
            for power, frame in sorted(cases.items())
        }
        metric_blocks.extend(
            [_fraction_rows(first, method="first_order", k=k), _fraction_rows(quadratic, method="second_order", k=k)]
        )
    metrics = pd.concat(metric_blocks, ignore_index=True).sort_values(
        ["method", "kNN", "region", "power_W", "threshold"]
    )

    wide = metrics.pivot_table(
        index=["kNN", "region", "threshold", "power_W"],
        columns="method",
        values="q_fraction",
        aggfunc="first",
    ).reset_index()
    contrasts: list[dict[str, object]] = []
    for (k, region, threshold), block in wide.groupby(["kNN", "region", "threshold"]):
        record: dict[str, object] = {"kNN": int(k), "region": region, "threshold": threshold}
        for method in ("first_order", "second_order"):
            lookup = block.set_index("power_W")[method]
            low = float(lookup.get(350, np.nan))
            high = float(lookup.get(400, np.nan))
            difference = low - high
            record[f"phi350_{method}"] = low
            record[f"phi400_{method}"] = high
            record[f"direction_{method}"] = _direction(difference)
            record[f"delta_{method}"] = difference
        contrasts.append(record)
    contrast_frame = pd.DataFrame(contrasts).merge(
        eligibility[["region", "threshold", "evidence_eligible", "evidence_status"]],
        on=["region", "threshold"],
        how="left",
        validate="many_to_one",
    )
    contrast_frame["direction_matches"] = (
        contrast_frame["direction_first_order"] == contrast_frame["direction_second_order"]
    )
    contrast_frame["comparable"] = ~contrast_frame[
        ["direction_first_order", "direction_second_order"]
    ].isin(["missing"]).any(axis=1)

    summary_rows: list[dict[str, object]] = []
    for (region, threshold), block in contrast_frame.groupby(["region", "threshold"], sort=True):
        eligible = bool(block["evidence_eligible"].iloc[0])
        comparable = block[block["comparable"]]
        mismatch_count = int((~comparable["direction_matches"]).sum())
        if not eligible:
            status = "not_primary_evidence"
        elif comparable.empty:
            status = "not_comparable"
        elif mismatch_count:
            status = "model_order_dependent"
        else:
            status = "order_consistent_over_compared_k"
        summary_rows.append(
            {
                "region": region,
                "threshold": threshold,
                "evidence_eligible": eligible,
                "k_values_tested": len(block),
                "comparable_k_count": int(len(comparable)),
                "direction_mismatch_count": mismatch_count,
                "status": status,
                "first_order_reference_k25_direction": (
                    block.loc[block["kNN"] == K_REFERENCE, "direction_first_order"].iloc[0]
                    if (block["kNN"] == K_REFERENCE).any()
                    else "not_tested"
                ),
            }
        )
    return metrics.reset_index(drop=True), contrast_frame.reset_index(drop=True), pd.DataFrame(summary_rows)
