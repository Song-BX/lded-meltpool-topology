from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.analysis.point_cloud import deduplicate_points, standardize_columns

from .config import (
    CANONICAL_METRICS,
    COORDINATE_TOLERANCE_M,
    OUTPUT_DIR,
    RETAINED_SENSITIVITY,
    ROOT,
)
from .discovery import discover_inputs, manifest_frame
from .knn_scan import scan_cases
from .neighborhood_scale import compute_neighborhood_scales
from .support_policy import attach_eligibility, build_support_audit
from .summarize import (
    compare_canonical_k25,
    compare_retained_scan,
    decision_payload,
    summarize_robustness,
    write_decision,
)


def _load_cases(records) -> dict[int, pd.DataFrame]:
    cases: dict[int, pd.DataFrame] = {}
    for record in records:
        raw = pd.read_csv(record.path)
        standardized = standardize_columns(raw)
        cases[record.power_W] = deduplicate_points(
            standardized, eps_c=COORDINATE_TOLERANCE_M
        )
    return cases


def _write_compatibility_outputs(
    power_metrics: pd.DataFrame, core: pd.DataFrame, summary: pd.DataFrame
) -> None:
    compatibility_dirs = (ROOT / "图" / "7", ROOT / "图" / "8")
    for directory in compatibility_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    raw = power_metrics.rename(
        columns={"power_W": "power", "threshold_value": "thr_value", "q_fraction": "frac_above", "n_region": "n"}
    )
    raw[["kNN", "region", "power", "threshold", "thr_value", "frac_above", "n"]].to_csv(
        ROOT / "图" / "7" / "Aplus_Qthreshold_sensitivity_raw.csv", index=False
    )
    core.to_csv(ROOT / "图" / "7" / "Aplus_Qthreshold_sensitivity_350vs400.csv", index=False)
    table = summary.rename(columns={"positive_count": "success"})
    table.to_csv(ROOT / "图" / "8" / "Aplus_Qtrend_robustness_summary.csv", index=False)


def run(output_dir: Path = OUTPUT_DIR) -> None:
    records = discover_inputs()
    retained = pd.read_csv(RETAINED_SENSITIVITY)
    canonical = pd.read_csv(CANONICAL_METRICS)
    cases = _load_cases(records)
    scale = compute_neighborhood_scales(cases)
    power_metrics, core = scan_cases(cases)

    reproducibility = pd.concat(
        [
            compare_canonical_k25(power_metrics, canonical),
            compare_retained_scan(core, retained),
        ],
        ignore_index=True,
    )
    if reproducibility.empty or not bool(reproducibility["passed"].all()):
        failed = reproducibility.loc[~reproducibility["passed"]]
        raise RuntimeError(
            "Canonical baseline reproduction failed; no analysis outputs were written.\n"
            + failed.to_string(index=False)
        )

    support_audit, eligibility = build_support_audit(power_metrics)
    summary = attach_eligibility(
        summarize_robustness(power_metrics, core), eligibility
    )
    payload = decision_payload(summary, scale)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_frame(records).to_csv(output_dir / "knn_input_manifest.csv", index=False)
    scale.to_csv(output_dir / "knn_neighborhood_scale.csv", index=False)
    power_metrics.to_csv(output_dir / "knn_power_metrics.csv", index=False)
    core.to_csv(output_dir / "knn_core_contrasts.csv", index=False)
    support_audit.to_csv(output_dir / "knn_support_audit.csv", index=False)
    eligibility.to_csv(output_dir / "knn_evidence_eligibility.csv", index=False)
    summary.to_csv(output_dir / "knn_robustness_summary.csv", index=False)
    reproducibility.to_csv(output_dir / "baseline_reproducibility.csv", index=False)
    write_decision(output_dir / "knn_decision.json", payload)
    _write_compatibility_outputs(power_metrics, core, summary)
    print(f"wrote robustness outputs to {output_dir}")
    print(f"baseline checks passed: {len(reproducibility)}/{len(reproducibility)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dense kNN robustness analysis.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
