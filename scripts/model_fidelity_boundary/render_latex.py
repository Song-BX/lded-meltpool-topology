from __future__ import annotations

from pathlib import Path


def _escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("&", r"\&")


def _display_evidence(value: object) -> str:
    text = str(value).replace("_", " ")
    text = text.replace("experiment simulation", "experiment-simulation")
    return text


def render_supplementary_table(alignment: list[dict[str, object]], output: Path) -> None:
    rows = []
    for row in alignment:
        rows.append(
            " & ".join(
                [
                    r"\Rone{" + _escape(row["comparison_item"]) + "}",
                    r"\Rone{" + _escape(str(row["alignment_status"]).replace("_", " ")) + "}",
                    r"\Rone{" + _escape(_display_evidence(row["prior_study_evidence"])) + "}",
                    r"\Rone{" + _escape(row["interpretation"]) + "}",
                ]
            )
            + r" \\"
        )
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\caption{\Rone{Prior model-validation context and physical-fidelity boundary. The cited prior publication is retained only as model provenance. No row establishes experimental validation of the six current FLOW-3D cases.}}",
        r"\label{tab:s1-model-validation-context}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.22\linewidth}>{\raggedright\arraybackslash}p{0.13\linewidth}>{\raggedright\arraybackslash}p{0.29\linewidth}>{\raggedright\arraybackslash}p{0.28\linewidth}}",
        r"\toprule",
        r"\Rone{Comparison item} & \Rone{Alignment status} & \Rone{Prior-study evidence available here} & \Rone{Boundary for the current study} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
