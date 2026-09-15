from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.analysis.point_cloud import standardize_columns

from .aggregation_sensitivity import run_aggregation_sensitivity
from .config import OUTPUT_DIR
from .discovery import discover_snapshots, manifest_frame, optional_reexport_manifest
from .group_diagnostics import analyse_snapshot
from .summarize import decision_payload, write_decision
from .validation import (
    validate_complete_grid,
    validate_group_accounting,
    validate_sensitivity,
    validate_spot_checks,
)


def run(output_dir: Path = OUTPUT_DIR) -> None:
    records = discover_snapshots()
    validate_complete_grid(records)
    frames: dict[tuple[float, int], pd.DataFrame] = {}
    summary_parts: list[pd.DataFrame] = []
    multiplicity_parts: list[pd.DataFrame] = []
    variable_parts: list[pd.DataFrame] = []
    check_parts: list[pd.DataFrame] = []

    for record in records:
        frame = standardize_columns(pd.read_csv(record.path))
        frames[(record.time_s, record.power_W)] = frame
        summary, multiplicity, variables, checks = analyse_snapshot(record, frame)
        summary_parts.append(summary)
        multiplicity_parts.append(multiplicity)
        variable_parts.append(variables)
        check_parts.append(checks)

    summary = pd.concat(summary_parts, ignore_index=True).sort_values(["time_s", "power_W"])
    multiplicity = pd.concat(multiplicity_parts, ignore_index=True).sort_values(
        ["time_s", "power_W", "multiplicity"]
    )
    variables = pd.concat(variable_parts, ignore_index=True).sort_values(
        ["time_s", "power_W", "variable"]
    )
    checks = pd.concat(check_parts, ignore_index=True).sort_values(
        ["time_s", "power_W", "check_type", "variable"]
    )
    validate_group_accounting(summary)
    validate_spot_checks(checks, frames)

    baseline_frames = {
        power_W: frames[(0.70, power_W)]
        for power_W in sorted({power_W for time_s, power_W in frames if time_s == 0.70})
    }
    sensitivity = run_aggregation_sensitivity(baseline_frames)
    validate_sensitivity(sensitivity)
    reexports = optional_reexport_manifest()
    payload = decision_payload(summary, sensitivity, reexports)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_frame(records).to_csv(output_dir / "export_input_manifest.csv", index=False)
    summary.to_csv(output_dir / "duplicate_group_summary.csv", index=False)
    multiplicity.to_csv(output_dir / "duplicate_multiplicity_distribution.csv", index=False)
    variables.to_csv(output_dir / "duplicate_variable_consistency.csv", index=False)
    checks.to_csv(output_dir / "duplicate_group_spot_checks.csv", index=False)
    reexports.to_csv(output_dir / "optional_reexport_manifest.csv", index=False)
    sensitivity["k25_metrics"].to_csv(output_dir / "aggregation_k25_metrics.csv", index=False)
    sensitivity["k25_core_contrasts"].to_csv(
        output_dir / "aggregation_k25_core_contrasts.csv", index=False
    )
    sensitivity["power_orderings"].to_csv(
        output_dir / "aggregation_power_orderings.csv", index=False
    )
    sensitivity["knn_power_metrics"].to_csv(
        output_dir / "aggregation_knn_power_metrics.csv", index=False
    )
    sensitivity["knn_contrasts"].to_csv(
        output_dir / "aggregation_knn_core_contrasts.csv", index=False
    )
    sensitivity["knn_summary"].to_csv(
        output_dir / "aggregation_sensitivity_summary.csv", index=False
    )
    sensitivity["knn_direction_changes"].to_csv(
        output_dir / "aggregation_direction_changes.csv", index=False
    )
    sensitivity["baseline_reproducibility"].to_csv(
        output_dir / "baseline_reproducibility.csv", index=False
    )
    write_decision(output_dir / "export_diagnostics_decision.json", payload)

    baseline = summary[summary["time_s"] == 0.70]
    print(f"wrote export diagnostics to {output_dir}")
    print(
        "0.70 s exact-row redundancy: "
        f"{baseline['exact_full_row_duplicate_ratio'].min() * 100:.1f}-"
        f"{baseline['exact_full_row_duplicate_ratio'].max() * 100:.1f}%"
    )
    print(
        "0.70 s conflicting-coordinate groups: "
        f"{baseline['conflicting_coordinate_group_fraction'].min() * 100:.1f}-"
        f"{baseline['conflicting_coordinate_group_fraction'].max() * 100:.1f}%"
    )
    print(
        "canonical baseline checks passed: "
        f"{int(sensitivity['baseline_reproducibility']['passed'].sum())}/"
        f"{len(sensitivity['baseline_reproducibility'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit export redundancy and coordinate-aggregation sensitivity."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
