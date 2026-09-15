from __future__ import annotations

from typing import Any


def build_summary(
    record: dict[str, Any], alignment: list[dict[str, object]], gates: list[dict[str, object]]
) -> list[dict[str, object]]:
    current_failed = [row for row in gates if row["required_for_current_fidelity"] and not row["passed"]]
    return [
        {
            "claim_id": "prior_model_validation_context",
            "final_status": "direct_observation",
            "retention_role": "context_only",
            "observed_scope": "A prior publication and the authors' reported experiment-simulation context are documented without importing its result values.",
            "allowed_interpretation": "Model provenance and limited validation background only.",
            "prohibited_interpretation": record["validation_context"]["prohibited_use"],
            "alignment_counts": "; ".join(
                f"{status}={sum(row['alignment_status'] == status for row in alignment)}"
                for status in ("exact_match", "partial_match", "different", "not_documented")
            ),
            "failed_gate_count": 0,
        },
        {
            "claim_id": "current_cfd_physical_fidelity",
            "final_status": "not_supported",
            "retention_role": "not_retained",
            "observed_scope": "The current study contains numerical exports only; it has no case-matched experimental comparison, solver-history archive, convergence study, or compatible native fields.",
            "allowed_interpretation": "The reported quantities are descriptors of the available solver-exported numerical fields.",
            "prohibited_interpretation": "An accurate representation of an actual L-DED melt pool; numerical convergence; validated flow topology; physical mechanism; or experimental support for the current cases.",
            "alignment_counts": "not applicable",
            "failed_gate_count": len(current_failed),
        },
    ]


def decision_payload(
    record: dict[str, Any], alignment: list[dict[str, object]], gates: list[dict[str, object]]
) -> dict[str, Any]:
    failed_current = [row["gate_id"] for row in gates if row["required_for_current_fidelity"] and not row["passed"]]
    statuses = {row["alignment_status"] for row in alignment}
    return {
        "analysis_scope": "provenance and physical-fidelity boundary only; no CFD field is recalculated",
        "prior_model_validation_context": {
            "status": "direct_observation",
            "retention_role": "context_only",
            "citation_key": record["citation_key"],
            "doi": record["doi"],
            "metadata_verification": record["metadata_verification"],
            "permitted_use": record["validation_context"]["permitted_use"],
            "prohibited_use": record["validation_context"]["prohibited_use"],
        },
        "alignment_statuses": sorted(statuses),
        "current_cfd_physical_fidelity": "not_supported",
        "failed_current_fidelity_gates": failed_current,
        "allowed_statement": "The analysis reports solver-exported numerical-field descriptors and an auditable boundary around their physical interpretation.",
        "prohibited_interpretations": [
            "the current six exported fields accurately represent an actual L-DED melt pool",
            "k, mask, threshold, aggregation, or manufactured-field checks validate CFD physical fidelity",
            "numerical convergence or solver health for the current cases",
            "a physical mechanism or experimentally supported comparative conclusion",
        ],
    }
