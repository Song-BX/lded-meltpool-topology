from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregation_audit import aggregation_metric_frame, audit_aggregation
from .config import OUTPUT_DIR
from .inputs import load_inputs
from .local_extrema import classify_discrete_extrema
from .pairwise_context import build_pairwise_snapshot_context
from .snapshot_metrics import (
    canonical_metric_frame,
    temporal_metric_frame,
    verify_temporal_reproduction,
)
from .summary import build_summary


def run(output_dir: Path = OUTPUT_DIR) -> None:
    inputs = load_inputs()
    canonical = canonical_metric_frame(inputs.canonical, inputs.thermal_tail)
    temporal = temporal_metric_frame(inputs.temporal, inputs.thermal_tail)
    temporal_reproduction = verify_temporal_reproduction(canonical, temporal)
    aggregation = aggregation_metric_frame(inputs.aggregation, inputs.median_aggregation)
    aggregation_extrema, aggregation_reproduction = audit_aggregation(canonical, aggregation)
    pairwise_context = build_pairwise_snapshot_context(canonical)

    snapshot_extrema = classify_discrete_extrema(canonical, ["metric_id"])
    temporal_extrema = classify_discrete_extrema(temporal, ["time_s", "metric_id"])
    summary, decision = build_summary(snapshot_extrema, aggregation_extrema, temporal_extrema)
    decision["canonical_reproduction_passed"] = bool(temporal_reproduction["passed"].all())
    decision["aggregation_reproduction_passed"] = bool(aggregation_reproduction["passed"].all())
    decision.update(
        {
            "observed_power_domain": "200--450 W",
            "observed_power_domain_W": {"minimum": 200, "maximum": 450},
            "higher_power_regime_assessed": False,
            "no_extrapolation_beyond_observed_power_domain": True,
            "pairwise_snapshot_context": {
                "row_count": len(pairwise_context),
                "unordered_power_pair_count": int(
                    pairwise_context[["lower_power_W", "higher_power_W"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "metric_count": int(pairwise_context["metric_id"].nunique()),
                "q_used": False,
                "purpose": "descriptive context for all observed unordered sampled-power pairs",
                "boundary": (
                    "The ledger does not identify a continuous response, a regime, or behavior "
                    "outside 200--450 W."
                ),
            },
        }
    )
    decision["expected_rows"] = {
        "snapshot_power_metrics": 24,
        "local_extremum_audit": 24,
        "aggregation_local_extrema": 96,
        "temporal_local_extrema": 120,
        "pairwise_snapshot_context": 60,
    }

    expected_rows = decision["expected_rows"]
    observed = {
        "snapshot_power_metrics": len(canonical),
        "local_extremum_audit": len(snapshot_extrema),
        "aggregation_local_extrema": len(aggregation_extrema),
        "temporal_local_extrema": len(temporal_extrema),
        "pairwise_snapshot_context": len(pairwise_context),
    }
    if observed != expected_rows:
        raise ValueError(f"Unexpected audit output dimensions: {observed}")

    output_dir.mkdir(parents=True, exist_ok=True)
    inputs.manifest.to_csv(output_dir / "power_response_input_manifest.csv", index=False, encoding="utf-8-sig")
    canonical.to_csv(output_dir / "snapshot_power_metrics.csv", index=False, encoding="utf-8-sig")
    snapshot_extrema.to_csv(output_dir / "local_extremum_audit.csv", index=False, encoding="utf-8-sig")
    aggregation_extrema.to_csv(
        output_dir / "aggregation_local_extrema.csv", index=False, encoding="utf-8-sig"
    )
    temporal_extrema.to_csv(
        output_dir / "temporal_local_extrema.csv", index=False, encoding="utf-8-sig"
    )
    pairwise_context.to_csv(
        output_dir / "pairwise_snapshot_context.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(output_dir / "power_response_summary.csv", index=False, encoding="utf-8-sig")
    temporal_reproduction.to_csv(
        output_dir / "temporal_snapshot_reproducibility.csv", index=False, encoding="utf-8-sig"
    )
    aggregation_reproduction.to_csv(
        output_dir / "aggregation_snapshot_reproducibility.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "power_response_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit six discrete powers without inferring a continuous response."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
