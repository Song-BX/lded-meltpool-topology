from __future__ import annotations

from typing import Any

from .config import CURRENT_FIDELITY_GATES


def build_gate_audit(record: dict[str, Any]) -> list[dict[str, object]]:
    context = record["validation_context"]
    prior_context_passed = bool(
        record["metadata_verification"]["title_and_doi_verified"]
        and context["experimental_comparison_reported_by_authors"]
        and context["status"] == "published_model_context_only"
    )
    rows = [
        {
            "gate_id": "prior_published_validation_context",
            "claim_id": "prior_model_validation_context",
            "required_for_current_fidelity": False,
            "passed": prior_context_passed,
            "required_evidence": "A verified citation and a documented, strictly context-only record of the authors' prior model-validation publication.",
            "observed_value": record["doi"],
            "failure_reason": "The publication metadata or its restricted use is not documented.",
        }
    ]
    rows.extend(
        {
            "gate_id": gate_id,
            "claim_id": "current_cfd_physical_fidelity",
            "required_for_current_fidelity": True,
            "passed": False,
            "required_evidence": required_evidence,
            "observed_value": "not available in the current revision workspace",
            "failure_reason": failure_reason,
        }
        for gate_id, required_evidence, failure_reason in CURRENT_FIDELITY_GATES
    )
    return rows
