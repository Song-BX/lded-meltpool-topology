from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .alignment import build_alignment
from .config import OUTPUT_DIR, TABLE_DIR
from .gates import build_gate_audit
from .render_latex import render_supplementary_table
from .sources import load_inputs
from .summary import build_summary, decision_payload


def run(output_dir: Path = OUTPUT_DIR) -> None:
    values, manifest = load_inputs()
    record = values["prior_validation_record"]
    alignment = build_alignment(record)
    gates = build_gate_audit(record)
    summary = build_summary(record, alignment, gates)
    decision = decision_payload(record, alignment, gates)
    output_dir.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(output_dir / "model_fidelity_input_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(alignment).to_csv(output_dir / "model_alignment_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gates).to_csv(output_dir / "cfd_fidelity_gate_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(output_dir / "model_fidelity_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "model_fidelity_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_supplementary_table(alignment, TABLE_DIR / "table_s1_model_validation_context.tex")
    print(f"wrote model-fidelity boundary audit to {output_dir}")
    print("current_cfd_physical_fidelity:", decision["current_cfd_physical_fidelity"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the physical-fidelity boundary of current CFD exports.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
