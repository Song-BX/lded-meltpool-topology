from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregation_sensitivity import compute_aggregation_sensitivity
from .config import OUTPUT_DIR
from .discovery import discover_gradient_inputs
from .metrics import compute_canonical_metrics
from .summary import build_summary
from .temporal_audit import build_temporal_context


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    inputs = discover_gradient_inputs()
    metrics = compute_canonical_metrics(inputs.snapshots)
    aggregation = compute_aggregation_sensitivity(inputs.snapshots)
    temporal_context = build_temporal_context(metrics)
    summary, decision = build_summary(metrics, aggregation)

    observed = {
        "thermal_gradient_metrics": len(metrics),
        "thermal_gradient_aggregation_sensitivity": len(aggregation),
        "thermal_gradient_temporal_context": len(temporal_context),
    }
    if observed != decision["expected_rows"]:
        raise ValueError(f"Unexpected thermal-gradient audit dimensions: {observed}")
    if not (metrics["finite_fraction"] == 1.0).all():
        raise ValueError("Non-finite exported temperature-gradient magnitudes were found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    inputs.manifest.to_csv(output_dir / "thermal_gradient_input_manifest.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "thermal_gradient_metrics.csv", index=False, encoding="utf-8-sig")
    aggregation.to_csv(output_dir / "thermal_gradient_aggregation_sensitivity.csv", index=False, encoding="utf-8-sig")
    temporal_context.to_csv(output_dir / "thermal_gradient_temporal_context.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "thermal_gradient_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "thermal_gradient_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit direct exported temperature-gradient magnitudes without WLS/Q reconstruction."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
