from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    AUDIT_ONLY_REGIONS,
    K_VALUES,
    MIN_POOLED_EXCEEDANCES,
    MIN_REGION_POINTS,
    REGIONS,
    THRESHOLDS,
)


def _integer_count(fraction: pd.Series, sample_size: pd.Series) -> pd.Series:
    return np.rint(
        fraction.to_numpy(dtype=float) * sample_size.to_numpy(dtype=int)
    ).astype(int)


def build_support_audit(
    power_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify region-threshold combinations without discarding raw metrics."""
    required = {"kNN", "power_W", "region", "threshold", "q_fraction", "n_region"}
    missing = sorted(required - set(power_metrics.columns))
    if missing:
        raise ValueError(f"Support audit is missing columns: {missing}")

    expected = {
        (k, region, threshold)
        for k in K_VALUES
        for region in REGIONS
        for threshold in THRESHOLDS
    }
    observed = set(
        power_metrics[["kNN", "region", "threshold"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed != expected:
        missing_cells = sorted(expected - observed)
        extra_cells = sorted(observed - expected)
        raise ValueError(
            f"Incomplete support grid; missing={missing_cells[:5]}, extra={extra_cells[:5]}"
        )

    q_zero = power_metrics[power_metrics["threshold"] == "Q>0"].copy()
    q_zero["positive_count"] = _integer_count(q_zero["q_fraction"], q_zero["n_region"])
    pooled_positive = (
        q_zero.groupby(["kNN", "region"], as_index=False)["positive_count"]
        .sum()
        .rename(columns={"positive_count": "pooled_positive_n"})
    )

    rows: list[dict[str, object]] = []
    for (k, region, threshold), block in power_metrics.groupby(
        ["kNN", "region", "threshold"], sort=True
    ):
        n_min = int(block["n_region"].min())
        n_max = int(block["n_region"].max())
        pooled_positive_n = int(
            pooled_positive.loc[
                (pooled_positive["kNN"] == k)
                & (pooled_positive["region"] == region),
                "pooled_positive_n",
            ].iloc[0]
        )
        pooled_exceedance_n = int(
            _integer_count(block["q_fraction"], block["n_region"]).sum()
        )
        point_support_pass = n_min >= MIN_REGION_POINTS
        tail_support_pass = (
            True
            if threshold == "Q>0"
            else pooled_exceedance_n >= MIN_POOLED_EXCEEDANCES
        )
        failures: list[str] = []
        if not point_support_pass:
            failures.append(f"minimum regional support {n_min} < {MIN_REGION_POINTS}")
        if not tail_support_pass:
            failures.append(
                f"pooled strict exceedances {pooled_exceedance_n} < "
                f"{MIN_POOLED_EXCEEDANCES}"
            )
        rows.append(
            {
                "kNN": int(k),
                "region": str(region),
                "threshold": str(threshold),
                "n_min_power": n_min,
                "n_max_power": n_max,
                "max_single_point_fraction_step": 1.0 / n_min,
                "pooled_positive_n": pooled_positive_n,
                "pooled_exceedance_n": pooled_exceedance_n,
                "point_support_pass": bool(point_support_pass),
                "tail_support_pass": bool(tail_support_pass),
                "eligible_at_k": bool(point_support_pass and tail_support_pass),
                "failure_reason": "; ".join(failures),
            }
        )

    detail = pd.DataFrame(rows).sort_values(["region", "threshold", "kNN"])
    summary_rows: list[dict[str, object]] = []
    for (region, threshold), block in detail.groupby(["region", "threshold"], sort=True):
        failed = block.loc[~block["eligible_at_k"]]
        evidence_eligible = failed.empty
        failure_reasons = sorted(
            {reason for reason in failed["failure_reason"].astype(str) if reason}
        )
        summary_rows.append(
            {
                "region": region,
                "threshold": threshold,
                "evidence_status": (
                    "evidence_eligible" if evidence_eligible else "insufficient_support"
                ),
                "analysis_role": (
                    "primary_evidence"
                    if evidence_eligible
                    else "audit_only"
                    if region in AUDIT_ONLY_REGIONS
                    else "excluded"
                ),
                "evidence_eligible": bool(evidence_eligible),
                "failed_k_count": int(len(failed)),
                "failed_k_values": ",".join(str(int(value)) for value in failed["kNN"]),
                "point_support_failed_k_count": int((~block["point_support_pass"]).sum()),
                "tail_support_failed_k_count": int((~block["tail_support_pass"]).sum()),
                "minimum_n_across_power_k": int(block["n_min_power"].min()),
                "maximum_single_point_fraction_step": float(
                    block["max_single_point_fraction_step"].max()
                ),
                "minimum_pooled_positive_n": int(block["pooled_positive_n"].min()),
                "maximum_pooled_positive_n": int(block["pooled_positive_n"].max()),
                "minimum_pooled_exceedance_n": int(block["pooled_exceedance_n"].min()),
                "maximum_pooled_exceedance_n": int(block["pooled_exceedance_n"].max()),
                "exclusion_reason": " | ".join(failure_reasons),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["region", "threshold"])
    return detail.reset_index(drop=True), summary.reset_index(drop=True)


def attach_eligibility(
    robustness_summary: pd.DataFrame, eligibility: pd.DataFrame
) -> pd.DataFrame:
    return robustness_summary.merge(
        eligibility,
        on=["region", "threshold"],
        how="left",
        validate="one_to_one",
    )
