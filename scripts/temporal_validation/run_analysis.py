from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.analysis.release_paths import reference_input

from .discovery import discover_snapshots, snapshot_manifest
from .metrics import compute_temporal_metrics
from .reproducibility import compare_baseline
from .stability import build_decision, evaluate_core_contrasts, evaluate_stability


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Validate late-time temporal stability.")
    parser.add_argument("--raw-dir", type=Path, default=root / "raw data")
    parser.add_argument(
        "--temporal-dir", type=Path, default=root / "raw data" / "temporal_validation"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "图" / "s4")
    parser.add_argument(
        "--canonical-metrics",
        type=Path,
        default=reference_input(
            root, "Aplus_main_metrics_k25.csv", Path("图/3/Aplus_main_metrics_k25.csv")
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = discover_snapshots(args.raw_dir, args.temporal_dir)
    manifest = snapshot_manifest(snapshots, project_root())
    manifest.to_csv(args.output_dir / "temporal_input_manifest.csv", index=False, encoding="utf-8-sig")

    metrics = compute_temporal_metrics(snapshots)
    metrics.to_csv(args.output_dir / "temporal_metrics.csv", index=False, encoding="utf-8-sig")

    stability, power_summary = evaluate_stability(metrics)
    contrasts = evaluate_core_contrasts(metrics)
    canonical = pd.read_csv(args.canonical_metrics)
    reproducibility = compare_baseline(metrics, canonical)
    if not reproducibility["passed"].all():
        failures = reproducibility.loc[~reproducibility["passed"], ["power_W", "metric"]]
        reproducibility.to_csv(
            args.output_dir / "baseline_reproducibility.csv", index=False, encoding="utf-8-sig"
        )
        raise RuntimeError(f"0.70 s metrics did not reproduce canonical values:\n{failures}")

    decision = build_decision(power_summary, contrasts)
    decision["snapshot_count"] = len(snapshots)
    decision["baseline_reproducibility_passed"] = True

    stability.to_csv(
        args.output_dir / "temporal_stability_summary.csv", index=False, encoding="utf-8-sig"
    )
    power_summary.to_csv(
        args.output_dir / "temporal_power_summary.csv", index=False, encoding="utf-8-sig"
    )
    contrasts.to_csv(
        args.output_dir / "temporal_core_contrasts.csv", index=False, encoding="utf-8-sig"
    )
    reproducibility.to_csv(
        args.output_dir / "baseline_reproducibility.csv", index=False, encoding="utf-8-sig"
    )
    (args.output_dir / "temporal_validation_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
