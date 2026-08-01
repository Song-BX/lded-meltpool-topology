from __future__ import annotations

from pathlib import Path

import pandas as pd


def _cell(text: str) -> str:
    return rf"\Rone{{{text}}}"


def render_table_s2(configuration: pd.DataFrame, log_summary: pd.DataFrame, output: Path) -> None:
    values = configuration.set_index("field_id")["value"]
    summary = log_summary.set_index("metric")["value"]
    rows = [
        ("Native 300 W phase model", f"$if\\_phchg=1$; $T_{{sat}}={float(values['saturation_temperature_K']):.0f}$ K; $L_v={float(values['latent_heat_vaporization_J_per_kg']):.3g}$ J kg$^{{-1}}$", "Documents configured liquid--vapour phase change; does not validate phase-change accuracy."),
        ("Surface-force setting", f"$if\\_prsrecoil={int(values['recoil_pressure_enabled'])}$", "Documents disabled recoil pressure; does not establish an evaporation or recoil mechanism."),
        ("300 W run record", f"normal completion at $t={float(summary['completion_time_s']):.6f}$ s, cycle {int(summary['completion_cycle'])}; {int(summary['progress_record_count'])} progress rows", "One supplied 300 W execution record only; not a six-case health audit."),
        ("Adaptive events", f"{int(summary['stability_event_count'])} convective-flux stability-limit messages, each followed by a smaller-step restart", "Reported log events, not a failure diagnosis or a cause of another case's temperature peak."),
        ("Printed diagnostics", r"pressure/heat-transfer \texttt{res/epsi} and fluid \texttt{\%loss} retained with native labels", "No configured acceptance thresholds or mass/energy-balance interpretation were supplied."),
        ("High-temperature outputs", "unfiltered $T_{max}$ and tail sensitivity retained in Fig. S11", "Peak-level numerical-output audit only; not a physical-fidelity result."),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{\Rone{Native 300 W phase-model and run-record context for the high-temperature-output audit.}}",
        r"\label{tab:s2-thermal-fidelity}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.22\linewidth}>{\raggedright\arraybackslash}p{0.35\linewidth}>{\raggedright\arraybackslash}p{0.35\linewidth}}",
        r"\toprule",
        _cell("Record") + " & " + _cell("Observed setting or log fact") + " & " + _cell("Evidence boundary") + r" \\",
        r"\midrule",
        *[" & ".join(_cell(str(value)) for value in row) + r" \\" for row in rows],
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
