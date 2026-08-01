from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.regions import region_mask
from scripts.analysis.wls_q import reconstruct_case
from scripts.robustness.knn_scan import scan_cases

from .aggregation import aggregate_points
from .config import (
    AGGREGATION_STRATEGIES,
    CANONICAL_KNN_CONTRASTS,
    CANONICAL_METRICS,
    CANONICAL_STRATEGY,
    FOF_INTERFACE_THRESHOLD,
    HEAT_FLUX_THRESHOLD,
    K_REFERENCE,
    REGIONS,
    THRESHOLDS,
    WLS_CONDITION_CUTOFF,
    WLS_CONDITION_MODE,
    WLS_DISTANCE_EXPONENT,
    WLS_DISTANCE_OFFSET_M,
)


def _reconstruct_reference(frame: pd.DataFrame) -> pd.DataFrame:
    return reconstruct_case(
        frame,
        k=K_REFERENCE,
        alpha=WLS_DISTANCE_EXPONENT,
        eps_w=WLS_DISTANCE_OFFSET_M,
        kappa_max=WLS_CONDITION_CUTOFF,
        condition_on=WLS_CONDITION_MODE,
    )


def _k25_metrics(
    cases: dict[int, pd.DataFrame], strategy: str
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for power_W, frame in sorted(cases.items()):
        reconstructed = _reconstruct_reference(frame)
        for region in REGIONS:
            mask = region_mask(
                reconstructed,
                region,
                fof_interface_threshold=FOF_INTERFACE_THRESHOLD,
                heat_flux_threshold=HEAT_FLUX_THRESHOLD,
            )
            subset = reconstructed.loc[mask]
            rows.append(
                {
                    "aggregation_strategy": strategy,
                    "power_W": power_W,
                    "region": region,
                    "n": int(len(subset)),
                    "v_mean": float(subset["V"].mean()),
                    "v_p95": float(subset["V"].quantile(0.95)),
                    "v_max": float(subset["V"].max()),
                    "T_mean_K": float(subset["T"].mean()),
                    "T_max_K": float(subset["T"].max()),
                    "Q_pos_frac": float((subset["Q"] > 0).mean()),
                    "wls_valid_fraction": float((reconstructed["chi"] == 1).mean()),
                }
            )
    return pd.DataFrame(rows)


def _compare_canonical_k25(k25_metrics: pd.DataFrame) -> pd.DataFrame:
    canonical = pd.read_csv(CANONICAL_METRICS)
    actual = k25_metrics[k25_metrics["aggregation_strategy"] == CANONICAL_STRATEGY]
    merged = actual.merge(
        canonical,
        on=["power_W", "region"],
        suffixes=("_actual", "_canonical"),
        validate="one_to_one",
    )
    rows: list[dict[str, float | int | str | bool]] = []
    for _, row in merged.iterrows():
        for metric in ("n", "v_mean", "v_p95", "v_max", "T_mean_K", "T_max_K", "Q_pos_frac"):
            actual_value = float(row[f"{metric}_actual"])
            canonical_value = float(row[f"{metric}_canonical"])
            rows.append(
                {
                    "check": "canonical_k25_metric",
                    "power_W": int(row["power_W"]),
                    "region": row["region"],
                    "kNN": K_REFERENCE,
                    "threshold": "",
                    "metric": metric,
                    "actual": actual_value,
                    "canonical": canonical_value,
                    "absolute_difference": abs(actual_value - canonical_value),
                    "passed": bool(
                        np.isclose(actual_value, canonical_value, rtol=1e-6, atol=1e-10)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _compare_canonical_knn(knn_contrasts: pd.DataFrame) -> pd.DataFrame:
    canonical = pd.read_csv(CANONICAL_KNN_CONTRASTS)
    actual = knn_contrasts[
        knn_contrasts["aggregation_strategy"] == CANONICAL_STRATEGY
    ]
    merged = actual.merge(
        canonical,
        on=["kNN", "region", "threshold"],
        suffixes=("_actual", "_canonical"),
        validate="one_to_one",
    )
    return pd.DataFrame(
        {
            "check": "canonical_knn_contrast",
            "power_W": "",
            "region": merged["region"],
            "kNN": merged["kNN"],
            "threshold": merged["threshold"],
            "metric": "diff_350_400",
            "actual": merged["diff_350_400_actual"],
            "canonical": merged["diff_350_400_canonical"],
            "absolute_difference": (
                merged["diff_350_400_actual"] - merged["diff_350_400_canonical"]
            ).abs(),
            "passed": np.isclose(
                merged["diff_350_400_actual"],
                merged["diff_350_400_canonical"],
                rtol=1e-6,
                atol=1e-10,
            ),
        }
    )


def _direction(value: float) -> str:
    if value > 0:
        return "350>400"
    if value < 0:
        return "350<400"
    return "tie"


def _k25_core_contrasts(k25_metrics: pd.DataFrame) -> pd.DataFrame:
    definitions = (
        ("T_mean_K_full_pool", "all", "T_mean_K"),
        ("T_max_K_full_pool", "all", "T_max_K"),
        ("V_max_full_pool", "all", "v_max"),
        ("V_mean_interface", "interface", "v_mean"),
        ("Q_positive_fraction_full_pool", "all", "Q_pos_frac"),
        ("Q_positive_fraction_interface", "interface", "Q_pos_frac"),
    )
    rows: list[dict[str, float | str]] = []
    for strategy in AGGREGATION_STRATEGIES:
        subset = k25_metrics[k25_metrics["aggregation_strategy"] == strategy]
        for label, region, metric in definitions:
            indexed = subset[subset["region"] == region].set_index("power_W")
            value_350 = float(indexed.loc[350, metric])
            value_400 = float(indexed.loc[400, metric])
            difference = value_350 - value_400
            rows.append(
                {
                    "aggregation_strategy": strategy,
                    "metric": label,
                    "region": region,
                    "value_350": value_350,
                    "value_400": value_400,
                    "diff_350_400": difference,
                    "direction": _direction(difference),
                }
            )
    output = pd.DataFrame(rows)
    canonical = output[output["aggregation_strategy"] == CANONICAL_STRATEGY].set_index(
        "metric"
    )["direction"]
    output["matches_canonical_direction"] = output.apply(
        lambda row: row["direction"] == canonical.loc[row["metric"]], axis=1
    )
    return output


def _power_orderings(k25_metrics: pd.DataFrame) -> pd.DataFrame:
    definitions = (
        ("T_mean_K", "all", "T_mean_K"),
        ("T_max_K", "all", "T_max_K"),
        ("V_max", "all", "v_max"),
        ("V_mean_interface", "interface", "v_mean"),
        ("Q_positive_fraction_full_pool", "all", "Q_pos_frac"),
        ("Q_positive_fraction_interface", "interface", "Q_pos_frac"),
    )
    rows: list[dict[str, str | bool]] = []
    for strategy in AGGREGATION_STRATEGIES:
        subset = k25_metrics[k25_metrics["aggregation_strategy"] == strategy]
        for label, region, metric in definitions:
            values = subset[subset["region"] == region].set_index("power_W")[metric]
            ordering = ">".join(
                str(power) for power in values.sort_values(ascending=False, kind="mergesort").index
            )
            rows.append(
                {
                    "aggregation_strategy": strategy,
                    "metric": label,
                    "region": region,
                    "descending_power_order": ordering,
                }
            )
    output = pd.DataFrame(rows)
    canonical = output[output["aggregation_strategy"] == CANONICAL_STRATEGY].set_index(
        "metric"
    )["descending_power_order"]
    output["matches_canonical_order"] = output.apply(
        lambda row: row["descending_power_order"] == canonical.loc[row["metric"]], axis=1
    )
    return output


def _summarize_knn(knn_contrasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = knn_contrasts[
        knn_contrasts["aggregation_strategy"] == CANONICAL_STRATEGY
    ].set_index(["kNN", "region", "threshold"])
    summary_rows: list[dict[str, float | int | str]] = []
    mismatch_parts: list[pd.DataFrame] = []
    for strategy in AGGREGATION_STRATEGIES:
        subset = knn_contrasts[knn_contrasts["aggregation_strategy"] == strategy].copy()
        indexed = subset.set_index(["kNN", "region", "threshold"])
        mismatch = indexed["direction"] != canonical["direction"]
        if strategy != CANONICAL_STRATEGY and mismatch.any():
            detail = indexed.loc[mismatch].reset_index()
            detail = detail.merge(
                canonical[["direction", "diff_350_400"]].reset_index(),
                on=["kNN", "region", "threshold"],
                suffixes=("", "_canonical"),
            )
            mismatch_parts.append(detail)
        for region in REGIONS:
            for threshold in THRESHOLDS:
                block = subset[
                    (subset["region"] == region) & (subset["threshold"] == threshold)
                ]
                key_index = block.set_index(["kNN", "region", "threshold"]).index
                block_mismatch = int(
                    (
                        block.set_index(["kNN", "region", "threshold"])["direction"]
                        != canonical.loc[key_index, "direction"]
                    ).sum()
                )
                summary_rows.append(
                    {
                        "aggregation_strategy": strategy,
                        "region": region,
                        "threshold": threshold,
                        "tested_k": len(block),
                        "count_350_gt_400": int((block["direction"] == "350>400").sum()),
                        "count_350_lt_400": int((block["direction"] == "350<400").sum()),
                        "count_tie": int((block["direction"] == "tie").sum()),
                        "direction_mismatches_vs_canonical": block_mismatch,
                        "median_diff_350_400": float(block["diff_350_400"].median()),
                        "minimum_diff_350_400": float(block["diff_350_400"].min()),
                        "maximum_diff_350_400": float(block["diff_350_400"].max()),
                    }
                )
    mismatch_details = (
        pd.concat(mismatch_parts, ignore_index=True)
        if mismatch_parts
        else pd.DataFrame(
            columns=[
                "kNN",
                "region",
                "threshold",
                "aggregation_strategy",
                "direction",
                "diff_350_400",
                "direction_canonical",
                "diff_350_400_canonical",
            ]
        )
    )
    return pd.DataFrame(summary_rows), mismatch_details


def run_aggregation_sensitivity(
    baseline_frames: dict[int, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    k25_parts: list[pd.DataFrame] = []
    power_parts: list[pd.DataFrame] = []
    contrast_parts: list[pd.DataFrame] = []
    for strategy in AGGREGATION_STRATEGIES:
        cases = {
            power_W: aggregate_points(frame, strategy)
            for power_W, frame in baseline_frames.items()
        }
        k25_parts.append(_k25_metrics(cases, strategy))
        power_metrics, contrasts = scan_cases(cases, verbose=False)
        power_metrics.insert(0, "aggregation_strategy", strategy)
        contrasts.insert(0, "aggregation_strategy", strategy)
        power_parts.append(power_metrics)
        contrast_parts.append(contrasts)

    k25 = pd.concat(k25_parts, ignore_index=True)
    knn_power = pd.concat(power_parts, ignore_index=True)
    knn_contrasts = pd.concat(contrast_parts, ignore_index=True)
    reproducibility = pd.concat(
        [_compare_canonical_k25(k25), _compare_canonical_knn(knn_contrasts)],
        ignore_index=True,
    )
    summary, mismatch_details = _summarize_knn(knn_contrasts)
    return {
        "k25_metrics": k25,
        "k25_core_contrasts": _k25_core_contrasts(k25),
        "power_orderings": _power_orderings(k25),
        "knn_power_metrics": knn_power,
        "knn_contrasts": knn_contrasts,
        "knn_summary": summary,
        "knn_direction_changes": mismatch_details,
        "baseline_reproducibility": reproducibility,
    }
