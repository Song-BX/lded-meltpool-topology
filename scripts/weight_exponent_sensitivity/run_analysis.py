from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .baseline import compare_canonical_baseline
from .config import (
    ALPHA_SPECS,
    EXPECTED_POWERS,
    FIELD_SPECS,
    K_VALUES,
    OUTPUT_DIR,
    RESAMPLE_COUNT,
    VALIDATION_REGIONS,
)
from .decision import decision_payload, write_json
from .discovery import discover_inputs, load_cases, manifest_frame
from .manufactured import run_manufactured_field_audit
from .q_summary import build_core_contrasts, common_support_q_metrics, summarize_q_metrics
from .reconstruction import reconstruct_grid
from .resampling import run_alpha_resampling
from .weight_geometry import summarize_weight_geometry


def _checks(
    manifest: pd.DataFrame,
    geometry: pd.DataFrame,
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    common_metrics: pd.DataFrame,
    common_core: pd.DataFrame,
    manufactured: pd.DataFrame,
    resampling: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    expected_q = len(ALPHA_SPECS) * len(K_VALUES) * len(EXPECTED_POWERS) * 4 * 4
    expected_core = len(ALPHA_SPECS) * len(K_VALUES) * 4 * 4
    expected_common = len(ALPHA_SPECS) * len(K_VALUES) * len(EXPECTED_POWERS)
    expected_manufactured = len(ALPHA_SPECS) * len(EXPECTED_POWERS) * len(FIELD_SPECS) * len(K_VALUES) * len(VALIDATION_REGIONS)
    expected_resampling = len(ALPHA_SPECS) * len(EXPECTED_POWERS) * len(VALIDATION_REGIONS) * RESAMPLE_COUNT
    checks = [
        ("input_manifest", len(EXPECTED_POWERS), len(manifest)),
        ("reconstruction_geometry_grid", len(ALPHA_SPECS) * len(EXPECTED_POWERS) * len(K_VALUES), len(geometry)),
        ("q_metrics_grid", expected_q, len(metrics)),
        ("core_contrast_grid", expected_core, len(contrasts)),
        ("common_support_metric_grid", expected_common, len(common_metrics)),
        ("common_support_contrast_grid", len(ALPHA_SPECS) * len(K_VALUES), len(common_core)),
        ("manufactured_field_grid", expected_manufactured, len(manufactured)),
        ("neighbour_resampling_grid", expected_resampling, len(resampling)),
        ("canonical_alpha_0_reproduction", len(baseline), int(baseline["passed"].sum()) if len(baseline) else 0),
    ]
    return pd.DataFrame(
        [{"check": name, "expected": expected, "observed": observed, "passed": expected == observed} for name, expected, observed in checks]
    )


def run(output_dir: Path = OUTPUT_DIR) -> None:
    records = discover_inputs()
    cases, case_counts = load_cases(records)
    manifest = manifest_frame(records).merge(case_counts, on="power_W", validate="one_to_one")

    reconstructed = reconstruct_grid(cases)
    geometry = summarize_weight_geometry(cases, reconstructed)
    metrics = summarize_q_metrics(reconstructed)
    contrasts = build_core_contrasts(metrics)
    common_metrics, common_core = common_support_q_metrics(reconstructed)
    baseline = compare_canonical_baseline(metrics)
    if baseline.empty or not bool(baseline["passed"].all()):
        raise RuntimeError("Canonical alpha=0 reproduction failed; no Comment 7 outputs were written.")

    manufactured, manufactured_summary = run_manufactured_field_audit(cases)
    resampling, resampling_core, resampling_summary = run_alpha_resampling(cases)
    checks = _checks(
        manifest,
        geometry,
        metrics,
        contrasts,
        common_metrics,
        common_core,
        manufactured,
        resampling,
        baseline,
    )
    if not bool(checks["passed"].all()):
        raise RuntimeError("Weight-exponent audit completeness checks failed.\n" + checks.to_string(index=False))
    summary, payload = decision_payload(
        baseline, metrics, contrasts, common_core, manufactured, resampling_core
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "alpha_input_manifest.csv", index=False)
    geometry.to_csv(output_dir / "alpha_weight_geometry.csv", index=False)
    metrics.to_csv(output_dir / "alpha_q_metrics.csv", index=False)
    contrasts.to_csv(output_dir / "alpha_core_contrasts.csv", index=False)
    common_metrics.to_csv(output_dir / "alpha_common_support_metrics.csv", index=False)
    common_core.to_csv(output_dir / "alpha_common_support_contrasts.csv", index=False)
    manufactured.to_csv(output_dir / "alpha_manufactured_field_metrics.csv", index=False)
    manufactured_summary.to_csv(output_dir / "alpha_manufactured_field_summary.csv", index=False)
    resampling.to_csv(output_dir / "alpha_neighbour_subset_resampling.csv", index=False)
    resampling_core.to_csv(output_dir / "alpha_neighbour_subset_core_contrasts.csv", index=False)
    resampling_summary.to_csv(output_dir / "alpha_neighbour_subset_summary.csv", index=False)
    baseline.to_csv(output_dir / "canonical_alpha0_reproducibility.csv", index=False)
    summary.to_csv(output_dir / "alpha_summary.csv", index=False)
    checks.to_csv(output_dir / "alpha_analysis_checks.csv", index=False)
    write_json(output_dir / "weight_exponent_decision.json", payload)
    print(f"wrote Comment 7 weight-exponent outputs to {output_dir}")
    print(f"canonical alpha=0 checks passed: {len(baseline)}/{len(baseline)}")
    print(f"final Q claim status: {payload['final_q_claim_status']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reviewer #1 Comment 7 WLS exponent sensitivity audit.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()

