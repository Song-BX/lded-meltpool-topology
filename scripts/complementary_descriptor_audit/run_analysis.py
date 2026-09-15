from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from .config import ALPHA_SPECS, OUTPUT_DIR
from .decision import baseline_reproduction, decision_payload, write_json
from .discovery import discover_inputs, load_canonical_cases, load_raw_cases, manifest_frame
from .manufactured import run_manufactured_audit
from .reconstruction import reconstruct_grid
from .sensitivity import (
    aggregation_sensitivity,
    conditioning_sensitivity,
    exponent_sensitivity,
    model_order_sensitivity,
    neighbour_subset_sensitivity,
)
from .summaries import core_contrasts, summarize_grid


def _staging_dir(output_dir: Path) -> Path:
    return output_dir / ".staging"


def _write_stage(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run_core_stage(output_dir: Path) -> None:
    records = discover_inputs()
    manifest = manifest_frame(records)
    cases = load_canonical_cases(records)
    raw_cases = load_raw_cases(records)

    canonical_grid = reconstruct_grid(cases)
    canonical_metrics, canonical_agreement = summarize_grid(canonical_grid, context="canonical")
    baseline = baseline_reproduction(canonical_metrics)
    if baseline.empty or not bool(baseline["passed"].all()):
        raise RuntimeError("Canonical Q reproduction failed; no complementary-descriptor outputs were written.")

    aggregation_metrics, aggregation_agreement = aggregation_sensitivity(raw_cases)
    alpha_metrics, alpha_agreement = exponent_sensitivity(cases, canonical_grid)
    cutoff_metrics, cutoff_agreement = conditioning_sensitivity(cases)
    model_metrics, model_agreement, model_contrasts = model_order_sensitivity(cases, canonical_grid)
    staging = _staging_dir(output_dir)
    _write_stage(staging / "manifest.csv", manifest)
    _write_stage(staging / "canonical_metrics.csv", canonical_metrics)
    _write_stage(staging / "canonical_agreement.csv", canonical_agreement)
    _write_stage(staging / "baseline.csv", baseline)
    _write_stage(staging / "aggregation_metrics.csv", aggregation_metrics)
    _write_stage(staging / "aggregation_agreement.csv", aggregation_agreement)
    _write_stage(staging / "alpha_metrics.csv", alpha_metrics)
    _write_stage(staging / "alpha_agreement.csv", alpha_agreement)
    _write_stage(staging / "cutoff_metrics.csv", cutoff_metrics)
    _write_stage(staging / "cutoff_agreement.csv", cutoff_agreement)
    _write_stage(staging / "model_metrics.csv", model_metrics)
    _write_stage(staging / "model_agreement.csv", model_agreement)
    _write_stage(staging / "model_contrasts.csv", model_contrasts)
    print(f"wrote Comment 15 core stage to {staging}")


def run_resampling_stage(output_dir: Path, alpha_label: str) -> None:
    alpha_spec = next(spec for spec in ALPHA_SPECS if spec.label == alpha_label)
    cases = load_canonical_cases(discover_inputs())
    metrics, agreement = neighbour_subset_sensitivity(cases, alpha_specs=(alpha_spec,))
    staging = _staging_dir(output_dir)
    _write_stage(staging / f"resample_metrics_alpha{alpha_label}.csv", metrics)
    _write_stage(staging / f"resample_agreement_alpha{alpha_label}.csv", agreement)
    print(f"wrote Comment 15 resampling stage for alpha={alpha_label}")


def run_manufactured_stage(output_dir: Path, alpha_label: str) -> None:
    alpha_spec = next(spec for spec in ALPHA_SPECS if spec.label == alpha_label)
    cases = load_canonical_cases(discover_inputs())
    manufactured = run_manufactured_audit(cases, (alpha_spec,))
    _write_stage(_staging_dir(output_dir) / f"manufactured_alpha{alpha_label}.csv", manufactured)
    print(f"wrote Comment 15 manufactured-field stage for alpha={alpha_label}")


def finalize_stage(output_dir: Path) -> None:
    staging = _staging_dir(output_dir)
    required = [
        "manifest.csv", "canonical_metrics.csv", "canonical_agreement.csv", "baseline.csv",
        "aggregation_metrics.csv", "aggregation_agreement.csv", "alpha_metrics.csv", "alpha_agreement.csv",
        "cutoff_metrics.csv", "cutoff_agreement.csv", "model_metrics.csv", "model_agreement.csv", "model_contrasts.csv",
        *[f"resample_metrics_alpha{spec.label}.csv" for spec in ALPHA_SPECS],
        *[f"resample_agreement_alpha{spec.label}.csv" for spec in ALPHA_SPECS],
        *[f"manufactured_alpha{spec.label}.csv" for spec in ALPHA_SPECS],
    ]
    missing = [name for name in required if not (staging / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot finalize Comment 15 audit; missing stage files: {missing}")
    read = lambda name: pd.read_csv(staging / name)
    manifest = read("manifest.csv")
    canonical_metrics = read("canonical_metrics.csv")
    canonical_agreement = read("canonical_agreement.csv")
    baseline = read("baseline.csv")
    aggregation_metrics, alpha_metrics, cutoff_metrics, model_metrics = [
        read(name) for name in ("aggregation_metrics.csv", "alpha_metrics.csv", "cutoff_metrics.csv", "model_metrics.csv")
    ]
    aggregation_agreement, alpha_agreement, cutoff_agreement, model_agreement = [
        read(name) for name in ("aggregation_agreement.csv", "alpha_agreement.csv", "cutoff_agreement.csv", "model_agreement.csv")
    ]
    model_contrasts = read("model_contrasts.csv")
    resample_metrics = pd.concat([read(f"resample_metrics_alpha{spec.label}.csv") for spec in ALPHA_SPECS], ignore_index=True)
    resample_agreement = pd.concat([read(f"resample_agreement_alpha{spec.label}.csv") for spec in ALPHA_SPECS], ignore_index=True)
    manufactured = pd.concat([read(f"manufactured_alpha{spec.label}.csv") for spec in ALPHA_SPECS], ignore_index=True)

    descriptor_metrics = pd.concat(
        [canonical_metrics, aggregation_metrics, alpha_metrics, cutoff_metrics, model_metrics, resample_metrics],
        ignore_index=True,
    )
    descriptor_agreement = pd.concat(
        [canonical_agreement, aggregation_agreement, alpha_agreement, cutoff_agreement, model_agreement, resample_agreement],
        ignore_index=True,
    )
    descriptor_sensitivity = pd.concat(
        [
            core_contrasts(aggregation_metrics),
            core_contrasts(alpha_metrics),
            core_contrasts(cutoff_metrics),
            model_contrasts,
            core_contrasts(resample_metrics),
        ],
        ignore_index=True,
    )
    checks, payload = decision_payload(
        manifest, baseline, canonical_metrics, canonical_agreement, manufactured
    )
    if not bool(checks["passed"].all()):
        raise RuntimeError("Complementary descriptor audit completeness checks failed.\n" + checks.to_string(index=False))

    manifest.to_csv(output_dir / "descriptor_input_manifest.csv", index=False, encoding="utf-8-sig")
    descriptor_metrics.to_csv(output_dir / "descriptor_metrics.csv", index=False, encoding="utf-8-sig")
    descriptor_agreement.to_csv(output_dir / "descriptor_agreement.csv", index=False, encoding="utf-8-sig")
    descriptor_sensitivity.to_csv(output_dir / "descriptor_sensitivity.csv", index=False, encoding="utf-8-sig")
    manufactured.to_csv(output_dir / "descriptor_manufactured_metrics.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(output_dir / "complementary_descriptor_summary.csv", index=False, encoding="utf-8-sig")
    baseline.to_csv(output_dir / "canonical_q_reproducibility.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "complementary_descriptor_decision.json", payload)
    # The staged CSVs make long-running substeps restartable, but they are not
    # scientific outputs.  Remove them only after every final output is written
    # so release verification remains closed over the documented file inventory.
    shutil.rmtree(staging)
    print(f"wrote Comment 15 complementary descriptor audit to {output_dir}")
    print(f"canonical Q reproduction: {int(baseline['passed'].sum())}/{len(baseline)}")
    print("final claim status: audit_only")


def run(output_dir: Path = OUTPUT_DIR) -> None:
    run_core_stage(output_dir)
    for alpha_spec in ALPHA_SPECS:
        run_resampling_stage(output_dir, alpha_spec.label)
        run_manufactured_stage(output_dir, alpha_spec.label)
    finalize_stage(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reviewer #1 Comment 15 complementary tensor-descriptor audit.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--stage",
        choices=("all", "core", "resample-0", "resample-1", "resample-2", "manufactured-0", "manufactured-1", "manufactured-2", "finalize"),
        default="all",
    )
    args = parser.parse_args()
    if args.stage == "all":
        run(args.output_dir)
    elif args.stage == "core":
        run_core_stage(args.output_dir)
    elif args.stage.startswith("resample-"):
        run_resampling_stage(args.output_dir, args.stage.rsplit("-", maxsplit=1)[1])
    elif args.stage.startswith("manufactured-"):
        run_manufactured_stage(args.output_dir, args.stage.rsplit("-", maxsplit=1)[1])
    else:
        finalize_stage(args.output_dir)


if __name__ == "__main__":
    main()
