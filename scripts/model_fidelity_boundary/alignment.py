from __future__ import annotations

from typing import Any

from .config import ALIGNMENT_ITEMS


def _status(item_id: str, prior_value: str) -> str:
    if item_id == "model_lineage" and prior_value == "authors_reported_same_model_family":
        return "partial_match"
    if prior_value == "not_documented_in_current_workspace":
        return "not_documented"
    if "not documented" in prior_value:
        return "not_documented"
    return "different"


def build_alignment(record: dict[str, Any]) -> list[dict[str, object]]:
    prior_evidence = record["prior_alignment_evidence"]
    rows: list[dict[str, object]] = []
    for item_id, item_name, current_value, current_source in ALIGNMENT_ITEMS:
        prior_value = str(prior_evidence[item_id])
        status = _status(item_id, prior_value)
        rows.append(
            {
                "alignment_id": item_id,
                "comparison_item": item_name,
                "current_study_value": current_value,
                "current_source": current_source,
                "prior_study_evidence": prior_value,
                "alignment_status": status,
                "interpretation": (
                    "Model-family relationship is recorded, but exact current-case equivalence is not established."
                    if item_id == "model_lineage"
                    else "The prior-study condition cannot be treated as matched to the current six cases."
                ),
            }
        )
    return rows
