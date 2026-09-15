from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import EXPECTED_AUDIT_ROWS, OUTPUT_DIR
from .inputs import load_inputs
from .metrics import build_audit
from .summary import build_summary


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    inputs = load_inputs()
    audit = build_audit(inputs.snapshots)
    summary, decision = build_summary(audit)
    if len(audit) != EXPECTED_AUDIT_ROWS:
        raise ValueError("Velocity-distribution overlap audit does not contain the required 8 records.")
    if not bool(
        audit.loc[
            (audit["audit_context"] == "aggregation_sensitivity")
            & (audit["aggregation_strategy"] == "mean_all_records"),
            "iqr_overlap_observed",
        ].all()
    ):
        raise ValueError("Canonical IQR overlap was not reproduced from the current point clouds.")

    output_dir.mkdir(parents=True, exist_ok=True)
    inputs.manifest.to_csv(output_dir / "velocity_distribution_input_manifest.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output_dir / "velocity_distribution_overlap_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "velocity_distribution_overlap_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "velocity_distribution_overlap_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit central velocity-distribution overlap for the 350 W--400 W pair.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
