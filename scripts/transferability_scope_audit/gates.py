from __future__ import annotations

from .config import EXTERNAL_GENERALISATION_GATES


def build_external_gate_audit() -> list[dict[str, object]]:
    return [
        {
            "gate_id": gate_id,
            "required_for_cross_context_applicability": True,
            "passed": False,
            "required_evidence": required_evidence,
            "observed_value": "not available in the current single-context study",
            "failure_reason": failure_reason,
        }
        for gate_id, required_evidence, failure_reason in EXTERNAL_GENERALISATION_GATES
    ]

