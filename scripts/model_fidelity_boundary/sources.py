from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import ROOT, SOURCE_SPECS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs() -> tuple[dict[str, Any], list[dict[str, object]]]:
    """Load only the documented provenance inputs, never numerical field results."""
    values: dict[str, Any] = {}
    manifest_rows: list[dict[str, object]] = []
    for spec in SOURCE_SPECS:
        path = ROOT / spec.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing model-fidelity input: {path}")
        content = path.read_text(encoding="utf-8")
        text_requirement_passed = spec.required_text is None or spec.required_text in content
        if not text_requirement_passed:
            raise ValueError(f"Required evidence marker is absent from {path}: {spec.required_text}")
        if spec.key == "prior_validation_record":
            record = json.loads(content)
            required_keys = {
                "citation_key",
                "doi",
                "title",
                "metadata_verification",
                "relationship_to_current_study",
                "validation_context",
                "prior_alignment_evidence",
            }
            missing = sorted(required_keys - set(record))
            if missing:
                raise ValueError(f"Prior-validation record is incomplete: {missing}")
            values[spec.key] = record
        else:
            values[spec.key] = content
        manifest_rows.append(
            {
                "input_key": spec.key,
                "relative_path": spec.relative_path.as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "required_marker": spec.required_text or "structured JSON record",
                "validation_passed": True,
            }
        )
    return values, manifest_rows
