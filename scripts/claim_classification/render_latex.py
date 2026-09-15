from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Q_REGION_LABELS, Q_THRESHOLD_LABELS


def _marked(value: str) -> str:
    return rf"\Rone{{{value}}}"


def _row(*values: str) -> str:
    return " & ".join(_marked(value) for value in values) + r" \\"


def render_diagnostic_hierarchy(output: Path) -> None:
    columns = r"{>{\raggedright\arraybackslash}p{0.22\linewidth}>{\raggedright\arraybackslash}p{0.27\linewidth}>{\raggedright\arraybackslash}p{0.25\linewidth}>{\raggedright\arraybackslash}p{0.20\linewidth}}"
    header = "Inference layer & Operational test & Evidence produced & Interpretation boundary " + r"\\"
    rows = [
        _row("Export structure and support", "Hash-validated row structure, coordinate aggregation, and regional point counts.", "Direct observations and supported denominators.", "Does not identify a solver setting, physical cause, or field fidelity."),
        _row("Direct snapshot descriptors", "Six powers, direct thermal and velocity fields, fixed aggregation checks, and five matched times.", "Descriptors of the sampled numerical exports.", "Temporal failure limits them to snapshot-local interpretation; sampled endpoints do not define regimes."),
        _row("Gradient reconstruction", "Manufactured fields, model order, neighbourhood size, conditioning, and distance weighting.", "The numerical operating range of the WLS reconstruction.", "Without native gradients, reconstruction tests do not validate the solver field."),
        _row("Reconstructed tensor descriptors", r"Support-qualified Q, $\lambda_2$, and normalized $\Omega_N$ calculations.", "Reconstruction-dependent topology descriptors.", "A failed numerical or temporal gate retains these quantities for audit only."),
        _row("Physical fidelity and persistence", "Five-snapshot tests, prior-model context, native records, and missing experiment/convergence evidence.", "Explicit limits on persistent and physical interpretation.", "No current claim reaches comparative or physical-mechanism evidence."),
        _row("External use", "Configuration manifest and external-transfer requirements.", "A reproducible record of the present case-study controls.", "Fields, masks, neighbourhoods, and thresholds are not portable defaults."),
    ]

    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.90}",
        r"\captionsetup{hypcap=false}",
        r"\caption{\Rone{Diagnostic progression from numerical export to bounded interpretation.}}\label{tab:diagnostic-hierarchy}",
        r"\begin{tabular}" + columns,
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def render_robustness_table(registry: pd.DataFrame, gates: pd.DataFrame, output: Path) -> None:
    row = registry.loc[registry["claim_id"] == "q-all-Qgt0"].iloc[0]
    core_gates = gates[gates["claim_id"] == "q-all-Qgt0"].set_index("gate_id")
    support = "evidence-eligible" if bool(core_gates.loc["support_eligibility", "passed"]) else "insufficient support"
    direction = "43/43 k directionally stable" if bool(core_gates.loc["knn_directional_stability", "passed"]) else "k-dependent"
    alpha = "failed: $\\alpha=2$ affine exactness" if not bool(core_gates.loc["weight_exponent_affine_exactness", "passed"]) else "passed"
    temporal = "failed: 350 W--400 W direction not persistent" if not bool(core_gates.loc["temporal_pairwise_persistence", "passed"]) else "passed"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\caption{\Rone{Deterministic promotion-gate audit for the reconstructed full-pool 350 W--400 W $Q>0$ contrast. Evidence eligibility is a necessary support condition, not a comparative conclusion.}}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.15\linewidth}>{\raggedright\arraybackslash}p{0.14\linewidth}>{\raggedright\arraybackslash}p{0.17\linewidth}>{\raggedright\arraybackslash}p{0.22\linewidth}>{\raggedright\arraybackslash}p{0.24\linewidth}}",
        r"\toprule",
        " & ".join(_marked(item) for item in ("Region", "Threshold", "Support / kNN", "Numerical and temporal gates", "Final classification")) + r" \\",
        r"\midrule",
        _row("full-pool", "$Q>0$", f"{support}; {direction}", f"conditioning passed; {alpha}; {temporal}", str(row.final_status).replace("_", " ") + ": no comparative Q conclusion"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def render_claim_boundary_table(registry: pd.DataFrame, output: Path) -> None:
    selected_ids = [
        "export_redundancy",
        "configuration_binding",
        "beyond_lded_applicability",
        "prior_model_validation_context",
        "current_cfd_physical_fidelity",
        "native_phase_model_configuration",
        "solver_execution_record_300W",
        "adaptive_stability_events_300W",
        "temperature_high_tail",
        "current_temperature_field_physical_fidelity",
        "quasi_steadiness",
        "thermal_gradient",
        "six_case_response",
        "velocity_extreme",
        "velocity_distribution_separation",
        "q_proxy_comparisons",
        "complementary_tensor_descriptors",
        "spatial_geometry",
        "marangoni_mechanism",
    ]
    rows = registry.set_index("claim_id").loc[selected_ids]
    columns = r"{>{\raggedright\arraybackslash}p{0.18\linewidth}>{\raggedright\arraybackslash}p{0.22\linewidth}>{\raggedright\arraybackslash}p{0.14\linewidth}>{\raggedright\arraybackslash}p{0.25\linewidth}>{\raggedright\arraybackslash}p{0.13\linewidth}}"
    header = " & ".join(
        _marked(item)
        for item in ("Claim", "Primary evidence", "Operational status", "Promotion rule / failed gate", "Boundary / not claimed")
    ) + r" \\"
    evidence = {
        "export_redundancy": "Fig.~\\ref{fig:point-quality}; 30-file structure and aggregation audits.",
        "configuration_binding": "Configuration-source hash manifest and context-bound-control audit.",
        "beyond_lded_applicability": "External-transfer gate audit; no external context was analysed.",
        "prior_model_validation_context": r"Supplementary Table~\ref{tab:s1-model-validation-context}; verified prior-publication record.",
        "current_cfd_physical_fidelity": "Model-fidelity gate audit; no current case-matched experiment, solver history, convergence study, or native-field identity.",
        "native_phase_model_configuration": "Supplementary Table~\\ref{tab:s2-thermal-fidelity}; supplied 300 W native configuration record.",
        "solver_execution_record_300W": "Supplementary Table~\\ref{tab:s2-thermal-fidelity}; 300 W native completion record.",
        "adaptive_stability_events_300W": "Fig.~\\ref{fig:s11}; supplied 300 W adaptive restart log events.",
        "temperature_high_tail": "Fig.~\\ref{fig:s11}; unfiltered 30-snapshot exported-temperature-tail audit and sensitivity screens.",
        "current_temperature_field_physical_fidelity": "Fig.~\\ref{fig:s11}; temperature-fidelity gate audit.",
        "quasi_steadiness": "Fig.~\\ref{fig:s4}; 11 fixed temporal tests.",
        "thermal_gradient": "Figs.~\\ref{fig:thermal-flow}, \\ref{fig:s3}; direct $G$ and aggregation audit.",
        "six_case_response": "Figs.~\\ref{fig:thermal-flow}, \\ref{fig:s8}; six sampled powers and the 60-row all-pair context ledger.",
        "velocity_extreme": "Figs.~\\ref{fig:thermal-flow}, \\ref{fig:s1}, \\ref{fig:s10}; Vmax provenance, central-range context, and solver-history health gate.",
        "velocity_distribution_separation": "Figs.~\\ref{fig:s1}, \\ref{fig:s10}; canonical IQR and tail-overlap audit.",
        "q_proxy_comparisons": "Figs.~\\ref{fig:q-metrics}, \\ref{fig:sensitivity}, \\ref{fig:s5}--\\ref{fig:s7}; Table~\\ref{tab:robustness}.",
        "complementary_tensor_descriptors": "Fig.~\\ref{fig:s9}; shared-tensor Q, $\\lambda_2$, $\\Omega_N$ audit.",
        "spatial_geometry": "XZ support audit and legacy-summary reconciliation.",
        "marangoni_mechanism": "Direct scalar $G$ audit and documented missing inputs.",
    }
    promotion = {
        "export_redundancy": "Complete structure and aggregation audit. Failed: none within this descriptive scope.",
        "configuration_binding": "All retained controls must be declared and none may be a portable default. Failed: none within this descriptive scope.",
        "beyond_lded_applicability": "Independent external context, semantic mapping, recalibration, mask/support validation, and end-to-end audit are required. Failed: all external-transfer gates.",
        "prior_model_validation_context": "Verified citation and context-only record are required. Failed: none within this limited provenance scope.",
        "current_cfd_physical_fidelity": "Matched experiment, solver history, mesh/timestep convergence, and native-field identity are required. Failed: all current-fidelity gates.",
        "native_phase_model_configuration": "Parseable supplied 300 W configuration record required. Failed: none within this configuration-record scope.",
        "solver_execution_record_300W": "Parsed supplied 300 W completion record required. Failed: none within this execution-record scope.",
        "adaptive_stability_events_300W": "Both supplied 300 W restart events must parse. Failed: none within this log-record scope.",
        "temperature_high_tail": "Complete unfiltered tail audit required. No physical-fidelity promotion path exists; retained as audit only.",
        "current_temperature_field_physical_fidelity": "Matched experiment, all-six histories with acceptance criteria, mesh/timestep convergence, and native-field identity are required. Failed: all temperature-fidelity gates.",
        "quasi_steadiness": "All six powers must pass 11 tests and pairwise persistence. Failed: temporal tests and persistence.",
        "thermal_gradient": "Direct export, support, and aggregation are required. Failed: temporal persistence for a persistent descriptor.",
        "six_case_response": "Canonical/aggregation reproduction, complete all-pair context, and an explicit 200--450 W boundary are required. Failed: temporal persistence for a persistent response; no higher-power regime was assessed.",
        "velocity_extreme": "Canonical, aggregation, velocity-definition, and native-history gates are required. Failed: central IQR overlap and native solver history unavailable.",
        "velocity_distribution_separation": "Central-range separation is not supported. Failed: 400 W IQR is contained within the 350 W IQR.",
        "q_proxy_comparisons": "All Q gates must pass. Failed: $\\alpha=2$, temporal persistence; six model-order-dependent and nine unsupported combinations.",
        "complementary_tensor_descriptors": "Canonical and descriptor checks are required, but the shared tensor prevents independent promotion. Failed: shared tensor, $\\alpha=2$, temporal persistence, native reference.",
        "spatial_geometry": "100-point spatial support is required. Failed: support and explicit exclusion.",
        "marangoni_mechanism": "Compatible vector, surface, material-stress, and independent validation inputs are required. Failed: required inputs.",
    }
    rendered_rows: list[str] = []
    for claim_id, row in rows.iterrows():
        final_status = str(row.final_status).replace("_", " ")
        rendered_rows.append(
            _row(str(row.claim), evidence[claim_id], final_status, promotion[claim_id], str(row.prohibited_interpretation))
        )

    def table_block(block_rows: list[str], *, continuation: bool) -> list[str]:
        caption = (
            r"\caption*{\Rone{Table~\ref{tab:claim-boundary} (continued).}}"
            if continuation
            else r"\caption{\Rone{Complete claim-promotion registry for the revised manuscript.}}\label{tab:claim-boundary}"
        )
        return [
            r"\begin{table}[p]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2pt}",
            r"\renewcommand{\arraystretch}{0.88}",
            r"\captionsetup{hypcap=false}",
            caption,
            r"\begin{tabular}" + columns,
            r"\toprule",
            header,
            r"\midrule",
            *block_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]

    # The five-column registry is intentionally dense.  A fourth continuation
    # keeps the last promotion-rule rows within a single page at normal size.
    blocks = (rendered_rows[:5], rendered_rows[5:10], rendered_rows[10:15], rendered_rows[15:])
    lines: list[str] = []
    for index, block in enumerate(blocks):
        lines.extend(table_block(block, continuation=index > 0))
    output.write_text("\n".join(lines), encoding="utf-8")
