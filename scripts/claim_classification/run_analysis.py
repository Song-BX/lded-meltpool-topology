from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classifier import build_registry, decision_payload
from .config import OUTPUT_DIR, TABLE_DIR
from .inputs import load_inputs
from .render_latex import (
    render_claim_boundary_table,
    render_diagnostic_hierarchy,
    render_robustness_table,
)


def run(output_dir: Path = OUTPUT_DIR) -> None:
    inputs = load_inputs()
    registry, gates = build_registry(inputs)
    payload = decision_payload(registry, inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    inputs.manifest.to_csv(output_dir / "claim_input_manifest.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(output_dir / "claim_gate_audit.csv", index=False, encoding="utf-8-sig")
    registry.to_csv(output_dir / "claim_registry.csv", index=False, encoding="utf-8-sig")
    (output_dir / "claim_classification_decision.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    render_diagnostic_hierarchy(TABLE_DIR / "table_diagnostic_inference_hierarchy.tex")
    render_robustness_table(registry, gates, TABLE_DIR / "table2_robustness.tex")
    render_claim_boundary_table(registry, TABLE_DIR / "table3_claim_evidence_boundary.tex")
    print(f"wrote claim classification outputs to {output_dir}")
    print("comparative evidence count:", int((registry["final_status"] == "comparative_evidence").sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify manuscript claims from existing audit outputs.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
