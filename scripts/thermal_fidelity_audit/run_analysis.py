from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import OUTPUT_DIR, TABLE_DIR
from .gates import decision_payload, gate_audit
from .render_latex import render_table_s2
from .running_log import parse_running_log
from .sources import load_inputs
from .summary import build_summary
from .temperature_tail import temperature_extreme_context, temperature_tail_metrics, temperature_tail_sensitivity


def run(output_dir: Path = OUTPUT_DIR) -> None:
    inputs = load_inputs()
    progress, events, log_summary = parse_running_log()
    tail, cache = temperature_tail_metrics(inputs.snapshots)
    sensitivity = temperature_tail_sensitivity(cache)
    extremes = temperature_extreme_context(cache)
    gates = gate_audit(inputs.phase_configuration, progress, events, log_summary)
    decision = decision_payload(inputs.phase_configuration, progress, events, log_summary, tail, gates)
    summary = build_summary(tail, sensitivity, decision)
    output_dir.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    inputs.manifest.to_csv(output_dir / "thermal_fidelity_input_manifest.csv", index=False, encoding="utf-8-sig")
    inputs.phase_configuration.to_csv(output_dir / "phase_model_configuration.csv", index=False, encoding="utf-8-sig")
    progress.to_csv(output_dir / "running_log_progress.csv", index=False, encoding="utf-8-sig")
    events.to_csv(output_dir / "running_log_events.csv", index=False, encoding="utf-8-sig")
    log_summary.to_csv(output_dir / "running_log_summary.csv", index=False, encoding="utf-8-sig")
    tail.to_csv(output_dir / "temperature_tail_metrics.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(output_dir / "temperature_tail_sensitivity.csv", index=False, encoding="utf-8-sig")
    extremes.to_csv(output_dir / "temperature_extreme_context.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(output_dir / "thermal_fidelity_gate_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "thermal_fidelity_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "thermal_fidelity_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_table_s2(inputs.phase_configuration, log_summary, TABLE_DIR / "table_s2_thermal_fidelity_context.tex")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit supplied native 300 W records and exported temperature tails.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
