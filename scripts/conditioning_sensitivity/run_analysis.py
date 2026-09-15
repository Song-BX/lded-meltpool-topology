from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import compare_canonical_baseline
from .condition_metrics import condition_distribution, cutoff_point_audit
from .config import CUTOFF_SPECS, K_VALUES, OUTPUT_DIR
from .decision import decision_payload, summarize_cutoffs
from .discovery import discover_inputs, manifest_frame
from .q_summary import build_core_contrasts, summarize_q_metrics
from .reconstruction import load_cases, reconstruct_without_finite_cutoff


def _validate_shapes(point_audit, metrics, core) -> None:
    expected_audit = 6 * len(K_VALUES) * len(CUTOFF_SPECS)
    expected_metrics = expected_audit * 4 * 4
    expected_core = len(CUTOFF_SPECS) * len(K_VALUES) * 4 * 4
    if len(point_audit) != expected_audit:
        raise RuntimeError(f"Expected {expected_audit} point-audit rows, found {len(point_audit)}")
    if len(metrics) != expected_metrics:
        raise RuntimeError(f"Expected {expected_metrics} Q-metric rows, found {len(metrics)}")
    if len(core) != expected_core:
        raise RuntimeError(f"Expected {expected_core} core-contrast rows, found {len(core)}")


def run(output_dir: Path = OUTPUT_DIR) -> None:
    records = discover_inputs()
    cases, case_counts = load_cases(records)
    reconstructed = reconstruct_without_finite_cutoff(cases)
    if len(reconstructed) != 6 * len(K_VALUES):
        raise RuntimeError("Incomplete unfiltered reconstruction grid.")

    distribution = condition_distribution(reconstructed)
    point_audit = cutoff_point_audit(reconstructed)
    metrics = summarize_q_metrics(reconstructed)
    core = build_core_contrasts(metrics)
    _validate_shapes(point_audit, metrics, core)

    reproducibility = compare_canonical_baseline(metrics, point_audit)
    if reproducibility.empty or not bool(reproducibility["passed"].all()):
        failed = reproducibility.loc[~reproducibility["passed"]]
        raise RuntimeError(
            "Canonical kappa=100 reproduction failed; no outputs were written.\n"
            + failed.to_string(index=False)
        )

    summary = summarize_cutoffs(metrics, core)
    payload = decision_payload(distribution, point_audit, summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_frame(records).merge(case_counts, on="power_W", validate="one_to_one")
    manifest.to_csv(output_dir / "condition_input_manifest.csv", index=False)
    distribution.to_csv(output_dir / "condition_distribution.csv", index=False)
    point_audit.to_csv(output_dir / "cutoff_point_audit.csv", index=False)
    metrics.to_csv(output_dir / "cutoff_q_metrics.csv", index=False)
    core.to_csv(output_dir / "cutoff_core_contrasts.csv", index=False)
    summary.to_csv(output_dir / "cutoff_summary.csv", index=False)
    reproducibility.to_csv(output_dir / "canonical_reproducibility.csv", index=False)
    (output_dir / "conditioning_decision.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote conditioning sensitivity outputs to {output_dir}")
    print(f"canonical reproduction checks passed: {len(reproducibility)}/{len(reproducibility)}")
    print(f"final Q claim status: {payload['final_q_claim_status']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the sensitivity of Q results to the WLS condition-number cutoff."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
