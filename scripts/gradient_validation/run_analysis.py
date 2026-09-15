from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.analysis.wls_q import reconstruct_case
from scripts.robustness.config import (
    WLS_CONDITION_CUTOFF,
    WLS_CONDITION_MODE,
    WLS_DISTANCE_EXPONENT,
    WLS_DISTANCE_OFFSET_M,
)

from .config import FIELD_SPECS, K_VALUES, OUTPUT_DIR
from .discovery import load_cases, validation_manifest
from .geometry import neighbourhood_geometry
from .manufactured_fields import build_manufactured_field
from .metrics import manufactured_metrics
from .model_order import run_model_order_comparison
from .native_reference import compare_native_reference
from .resampling import run_neighbour_resampling
from .summarize import (
    decision_payload,
    manufactured_summary,
    validate_outputs,
    write_json,
)


def _reconstruction(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    return reconstruct_case(
        frame,
        k=k,
        alpha=WLS_DISTANCE_EXPONENT,
        eps_w=WLS_DISTANCE_OFFSET_M,
        kappa_max=WLS_CONDITION_CUTOFF,
        condition_on=WLS_CONDITION_MODE,
    )


def run(output_dir: str | None = None) -> None:
    output_path = OUTPUT_DIR if output_dir is None else Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records, loaded = load_cases()
    cases = {power: item.frame for power, item in loaded.items()}
    manifest = validation_manifest(records, loaded)

    geometry = pd.concat(
        [neighbourhood_geometry(item.frame, power) for power, item in sorted(loaded.items())],
        ignore_index=True,
    )
    geometry_blocks = []
    for region, subset in (
        ("all", geometry),
        ("interface", geometry[geometry["is_interface"]]),
    ):
        grouped = subset.groupby(["power_W", "kNN"], as_index=False).agg(
            kth_radius_mm_median=("kth_radius_mm", "median"),
            kth_radius_mm_p90=("kth_radius_mm", lambda value: value.quantile(0.90)),
            condition_design_p90=("condition_design", lambda value: value.quantile(0.90)),
            eigenvalue_ratio_median=("eigenvalue_ratio_min_max", "median"),
        )
        grouped["region"] = region
        geometry_blocks.append(grouped)
    geometry_summary = pd.concat(geometry_blocks, ignore_index=True)

    manufactured_blocks: list[pd.DataFrame] = []
    for power, frame in sorted(cases.items()):
        for spec in FIELD_SPECS:
            truth = build_manufactured_field(frame, spec)
            for k in K_VALUES:
                manufactured_blocks.append(
                    manufactured_metrics(_reconstruction(truth.frame, k), truth, power_W=power, k=k)
                )
            print(f"completed manufactured field {spec.field_id} for {power} W", flush=True)
    manufactured = pd.concat(manufactured_blocks, ignore_index=True)

    eligibility = pd.read_csv(
        OUTPUT_DIR.parent / "robustness" / "knn_evidence_eligibility.csv"
    )
    model_metrics, model_contrasts, model_summary = run_model_order_comparison(cases, eligibility)
    resamples, resampling_core, resampling_summary = run_neighbour_resampling(cases)
    native_status, native_comparison = compare_native_reference(cases)
    checks = validate_outputs(manifest, manufactured, geometry, model_metrics, resamples)
    if not bool(checks["passed"].all()):
        raise ValueError(f"Gradient-validation completeness checks failed: {checks.to_dict(orient='records')}")
    decision = decision_payload(manufactured, model_summary, resampling_core, native_status)

    manifest.to_csv(output_path / "gradient_validation_input_manifest.csv", index=False)
    geometry.to_csv(output_path / "neighbourhood_geometry.csv", index=False)
    geometry_summary.to_csv(output_path / "neighbourhood_geometry_summary.csv", index=False)
    manufactured.to_csv(output_path / "manufactured_field_metrics.csv", index=False)
    manufactured_summary(manufactured).to_csv(output_path / "manufactured_field_summary.csv", index=False)
    model_metrics.to_csv(output_path / "model_order_metrics.csv", index=False)
    model_contrasts.to_csv(output_path / "model_order_core_contrasts.csv", index=False)
    model_summary.to_csv(output_path / "model_order_summary.csv", index=False)
    resamples.to_csv(output_path / "neighbour_subset_resampling.csv", index=False)
    resampling_core.to_csv(output_path / "neighbour_subset_core_contrasts.csv", index=False)
    resampling_summary.to_csv(output_path / "neighbour_subset_summary.csv", index=False)
    native_status.to_csv(output_path / "native_reference_status.csv", index=False)
    native_comparison.to_csv(output_path / "native_reference_comparison.csv", index=False)
    checks.to_csv(output_path / "gradient_validation_checks.csv", index=False)
    write_json(output_path / "gradient_validation_decision.json", decision)
    print(f"wrote Comment 5 gradient-validation outputs to {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reviewer #1 Comment 5 gradient validation.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
