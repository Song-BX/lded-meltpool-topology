from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import K_REFERENCE, K_VALUES, REGIONS, THRESHOLDS


def compare_canonical_k25(power_metrics: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    actual = power_metrics[
        (power_metrics["kNN"] == K_REFERENCE) & (power_metrics["threshold"] == "Q>0")
    ].set_index(["power_W", "region"])
    rows: list[dict[str, object]] = []
    for row in canonical.itertuples(index=False):
        key = (int(row.power_W), str(row.region))
        if key not in actual.index:
            continue
        current = actual.loc[key]
        checks = (
            ("n_region", float(current["n_region"]), float(row.n), 0.0),
            ("q_fraction", float(current["q_fraction"]), float(row.Q_pos_frac), 1e-12),
        )
        for metric, observed, expected, tolerance in checks:
            rows.append(
                {
                    "check": "canonical_k25",
                    "power_W": key[0],
                    "region": key[1],
                    "kNN": K_REFERENCE,
                    "threshold": "Q>0",
                    "metric": metric,
                    "actual": observed,
                    "expected": expected,
                    "absolute_difference": abs(observed - expected),
                    "passed": bool(np.isclose(observed, expected, rtol=0.0, atol=tolerance)),
                }
            )
    return pd.DataFrame(rows)


def compare_retained_scan(core: pd.DataFrame, retained: pd.DataFrame) -> pd.DataFrame:
    lookup = core.set_index(["kNN", "region", "threshold"])
    rows: list[dict[str, object]] = []
    for row in retained.itertuples(index=False):
        key = (int(row.kNN), str(row.region), str(row.threshold))
        if key not in lookup.index:
            continue
        current = lookup.loc[key]
        for metric in ("ratio_350_400", "diff_350_400"):
            observed = float(current[metric])
            expected = float(getattr(row, metric))
            rows.append(
                {
                    "check": "retained_five_k_scan",
                    "power_W": "350-400",
                    "region": key[1],
                    "kNN": key[0],
                    "threshold": key[2],
                    "metric": metric,
                    "actual": observed,
                    "expected": expected,
                    "absolute_difference": abs(observed - expected),
                    "passed": bool(np.isclose(observed, expected, rtol=1e-10, atol=1e-12)),
                }
            )
    return pd.DataFrame(rows)


def _power_order(subset: pd.DataFrame) -> tuple[int, ...]:
    ordered = subset.sort_values(["q_fraction", "power_W"], ascending=[False, True])
    return tuple(int(power) for power in ordered["power_W"])


def _rank_correlation(reference: pd.DataFrame, observed: pd.DataFrame) -> float:
    ref = reference.set_index("power_W")["q_fraction"].rank(method="average")
    obs = observed.set_index("power_W")["q_fraction"].rank(method="average")
    common = ref.index.intersection(obs.index)
    if len(common) < 2:
        return np.nan
    return float(np.corrcoef(ref.loc[common], obs.loc[common])[0, 1])


def summarize_robustness(
    power_metrics: pd.DataFrame, core_contrasts: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region in REGIONS:
        for threshold in THRESHOLDS:
            core = core_contrasts[
                (core_contrasts["region"] == region)
                & (core_contrasts["threshold"] == threshold)
            ].sort_values("kNN")
            reference_direction = str(
                core.loc[core["kNN"] == K_REFERENCE, "direction"].iloc[0]
            )
            directions = core["direction"]
            direction_matches = int((directions == reference_direction).sum())
            positive = int((core["diff_350_400"] > 0).sum())
            negative = int((core["diff_350_400"] < 0).sum())
            ties = int((core["diff_350_400"] == 0).sum())

            metric_subset = power_metrics[
                (power_metrics["region"] == region)
                & (power_metrics["threshold"] == threshold)
            ]
            reference = metric_subset[metric_subset["kNN"] == K_REFERENCE]
            reference_order = _power_order(reference)
            ordering_matches = 0
            correlations: list[float] = []
            for k in K_VALUES:
                observed = metric_subset[metric_subset["kNN"] == k]
                ordering_matches += int(_power_order(observed) == reference_order)
                correlations.append(_rank_correlation(reference, observed))

            rows.append(
                {
                    "region": region,
                    "threshold": threshold,
                    "reference_direction_k25": reference_direction,
                    "direction_match_count": direction_matches,
                    "positive_count": positive,
                    "negative_count": negative,
                    "tie_count": ties,
                    "total": len(K_VALUES),
                    "success_rate": positive / len(K_VALUES),
                    "median_delta": float(core["diff_350_400"].median()),
                    "min_delta": float(core["diff_350_400"].min()),
                    "max_delta": float(core["diff_350_400"].max()),
                    "delta_range": float(core["diff_350_400"].max() - core["diff_350_400"].min()),
                    "classification": (
                        "directionally_stable"
                        if direction_matches == len(K_VALUES) and ties == 0
                        else "k_dependent"
                    ),
                    "reference_power_order_k25": ">".join(map(str, reference_order)),
                    "power_order_match_count": ordering_matches,
                    "minimum_rank_correlation": float(np.nanmin(correlations)),
                }
            )
    return pd.DataFrame(rows)


def decision_payload(summary: pd.DataFrame, scale: pd.DataFrame) -> dict[str, object]:
    median_by_k = scale.groupby("kNN")["radius_median_mm"].median()
    combinations = []
    for row in summary.itertuples(index=False):
        combinations.append(
            {
                "region": row.region,
                "threshold": row.threshold,
                "classification": row.classification,
                "reference_direction_k25": row.reference_direction_k25,
                "direction_match_count": int(row.direction_match_count),
                "positive_count": int(row.positive_count),
                "negative_count": int(row.negative_count),
                "tie_count": int(row.tie_count),
                "power_order_match_count": int(row.power_order_match_count),
                "evidence_status": row.evidence_status,
                "analysis_role": row.analysis_role,
                "failed_k_count": int(row.failed_k_count),
                "exclusion_reason": row.exclusion_reason,
            }
        )
    eligible_count = int((summary["evidence_status"] == "evidence_eligible").sum())
    return {
        "analysis_scope": "six 0.70 s L-DED/FLOW-3D power cases",
        "empirical_transferability": "not_evaluated",
        "k_values": list(K_VALUES),
        "reference_k": K_REFERENCE,
        "median_support_radius_mm": {
            "k8": float(median_by_k.loc[8]),
            "k25": float(median_by_k.loc[25]),
            "k50": float(median_by_k.loc[50]),
        },
        "decision_rule": "directional stability is assessed only after all 43 k values pass the evidence-support gates",
        "support_policy": {
            "minimum_regional_points_per_power_k": 100,
            "maximum_single_point_fraction_step": 0.01,
            "minimum_pooled_strict_exceedances": 10,
            "spatial_independence_claimed": False,
        },
        "evidence_eligible_count": eligible_count,
        "insufficient_support_count": int(len(summary) - eligible_count),
        "combinations": combinations,
    }


def write_decision(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
