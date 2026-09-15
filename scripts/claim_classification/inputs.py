from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import INPUT_SPECS, ROOT


@dataclass(frozen=True)
class LoadedInputs:
    json_data: dict[str, dict[str, Any]]
    q_eligibility: pd.DataFrame
    q_robustness: pd.DataFrame
    manifest: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_input(path: Path, kind: str) -> tuple[Any, int | None, set[str]]:
    if kind == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        return payload, None, set(payload)
    if kind == "csv":
        frame = pd.read_csv(path)
        return frame, len(frame), set(frame.columns)
    raise ValueError(f"Unsupported input kind: {kind}")


def load_inputs() -> LoadedInputs:
    json_data: dict[str, dict[str, Any]] = {}
    csv_data: dict[str, pd.DataFrame] = {}
    manifest_rows: list[dict[str, object]] = []

    for spec in INPUT_SPECS:
        path = ROOT / spec.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required claim-classification input is missing: {path}")
        payload, observed_rows, observed_keys = _read_input(path, spec.kind)
        missing = sorted(set(spec.required_keys) - observed_keys)
        row_match = spec.expected_rows is None or observed_rows == spec.expected_rows
        valid = not missing and row_match
        manifest_rows.append(
            {
                "input_key": spec.key,
                "relative_path": spec.relative_path.as_posix(),
                "kind": spec.kind,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "expected_rows": spec.expected_rows,
                "observed_rows": observed_rows,
                "required_keys": ";".join(spec.required_keys),
                "missing_keys": ";".join(missing),
                "validation_passed": valid,
            }
        )
        if not valid:
            raise ValueError(
                f"Input validation failed for {path}: missing_keys={missing}, "
                f"expected_rows={spec.expected_rows}, observed_rows={observed_rows}"
            )
        if spec.kind == "json":
            json_data[spec.key] = payload
        else:
            csv_data[spec.key] = payload

    return LoadedInputs(
        json_data=json_data,
        q_eligibility=csv_data["q_eligibility"],
        q_robustness=csv_data["q_robustness"],
        manifest=pd.DataFrame(manifest_rows),
    )
