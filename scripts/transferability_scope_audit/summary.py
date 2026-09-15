from __future__ import annotations

from typing import Any


def build_summary(
    controls: list[dict[str, object]], gate_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    return [
        {
            "claim_id": "configuration_binding",
            "final_status": "direct_observation",
            "observed_scope": "All retained numerical, semantic, and study-design controls are declared for one L-DED/FLOW-3D configuration.",
            "allowed_interpretation": "A context-bound configuration record and future adaptation checklist.",
            "prohibited_interpretation": "A solver-independent, material-independent, or directly transferable parameter default.",
            "control_count": len(controls),
            "failed_external_gate_count": 0,
        },
        {
            "claim_id": "beyond_lded_applicability",
            "final_status": "not_supported",
            "observed_scope": "No external process, solver, material, scan strategy, mesh, or export-schema audit exists.",
            "allowed_interpretation": "No cross-context applicability conclusion is retained.",
            "prohibited_interpretation": "Applicable beyond L-DED; general-purpose workflow; transferable parameter defaults; solver-independent performance.",
            "control_count": len(controls),
            "failed_external_gate_count": sum(not bool(row["passed"]) for row in gate_rows),
        },
    ]


def decision_payload(
    controls: list[dict[str, object]], gate_rows: list[dict[str, object]]
) -> dict[str, Any]:
    failed_gates = [str(row["gate_id"]) for row in gate_rows if not bool(row["passed"])]
    return {
        "analysis_scope": "one material, one scan strategy, one mesh configuration, six discrete L-DED powers, and FLOW-3D CSV point-cloud exports",
        "control_count": len(controls),
        "all_controls_context_bound": all(not bool(row["portable_default"]) for row in controls),
        "cross_context_applicability": "not_supported",
        "portable_parameter_defaults": [],
        "failed_external_gates": failed_gates,
        "allowed_statement": "The code documents a context-bound diagnostic sequence that must be re-specified and revalidated before use in another target context.",
        "prohibited_interpretations": [
            "applicable beyond L-DED",
            "general-purpose framework",
            "solver-independent performance",
            "transferable numerical or mask defaults",
        ],
        "release_package_status": "Comment 14 package rebuilt locally after all reviewer comments; public access remains pending manual GitHub upload.",
    }
