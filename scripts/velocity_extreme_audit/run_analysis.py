from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import OUTPUT_DIR
from .health import evaluate_health
from .inputs import load_inputs
from .solver_history import normalise_solver_history
from .summary import build_summary
from .velocity import (
    aggregation_velocity_audit,
    canonical_reproduction,
    closure_rows,
    peak_provenance,
    velocity_quantiles,
)


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    inputs = load_inputs()
    quantiles, cache = velocity_quantiles(inputs.snapshots)
    reproduction = canonical_reproduction(quantiles)
    closure = closure_rows(inputs.snapshots, cache)
    aggregation = aggregation_velocity_audit(inputs.snapshots, cache)
    provenance = peak_provenance(quantiles, cache)
    normalized_history, normalization_issues = normalise_solver_history(inputs.solver_mapping)
    health = evaluate_health(normalized_history, tuple(inputs.mapping_issues) + tuple(normalization_issues))
    summary, decision = build_summary(quantiles, reproduction, aggregation, provenance, health)

    if len(quantiles) != 30 or len(closure) != 60 or len(aggregation) != 24:
        raise ValueError("Velocity-extreme audit dimensions do not match the pre-specified input grid.")
    if not reproduction["passed"].all():
        raise ValueError("Canonical Vmax reproduction gate failed.")

    output_dir.mkdir(parents=True, exist_ok=True)
    inputs.manifest.to_csv(output_dir / "velocity_extreme_input_manifest.csv", index=False, encoding="utf-8-sig")
    quantiles.to_csv(output_dir / "velocity_quantiles.csv", index=False, encoding="utf-8-sig")
    reproduction.to_csv(output_dir / "canonical_vmax_reproduction.csv", index=False, encoding="utf-8-sig")
    provenance.to_csv(output_dir / "peak_provenance.csv", index=False, encoding="utf-8-sig")
    closure.to_csv(output_dir / "velocity_closure_audit.csv", index=False, encoding="utf-8-sig")
    aggregation.to_csv(output_dir / "velocity_temporal_aggregation_audit.csv", index=False, encoding="utf-8-sig")
    normalized_history.to_csv(output_dir / "normalised_solver_history.csv", index=False, encoding="utf-8-sig")
    health.to_csv(output_dir / "solver_health_gate_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "velocity_extreme_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "velocity_extreme_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit peak-velocity provenance and native solver-history health.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

