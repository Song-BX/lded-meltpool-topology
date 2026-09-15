from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import OUTPUT_DIR
from .controls import build_context_bound_controls
from .gates import build_external_gate_audit
from .sources import load_sources
from .summary import build_summary, decision_payload


def run(output_dir: Path = OUTPUT_DIR) -> None:
    values, manifest_rows = load_sources()
    controls = build_context_bound_controls(values)
    gate_rows = build_external_gate_audit()
    summary = build_summary(controls, gate_rows)
    payload = decision_payload(controls, gate_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(
        output_dir / "transferability_input_manifest.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(controls).to_csv(
        output_dir / "context_bound_controls.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(gate_rows).to_csv(
        output_dir / "transferability_gate_audit.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summary).to_csv(
        output_dir / "transferability_summary.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "transferability_decision.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote transferability scope audit to {output_dir}")
    print("cross_context_applicability:", payload["cross_context_applicability"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the configuration boundary of the current L-DED/FLOW-3D workflow."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
