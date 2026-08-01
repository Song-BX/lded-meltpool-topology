from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.temporal_validation.discovery import SnapshotFile, discover_snapshots, snapshot_manifest

from .config import FLOW3D_PATH, PHASE_CONFIGURATION, RAW_DIR, ROOT, RUNNING_PATH, TEMPORAL_DIR


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


@dataclass(frozen=True)
class ThermalFidelityInputs:
    snapshots: list[SnapshotFile]
    manifest: pd.DataFrame
    phase_configuration: pd.DataFrame


def _line_value(lines: list[str], parameter: str, kind: str) -> tuple[object, int, str]:
    escaped = re.escape(parameter).replace(r"\ ", r"\s*")
    if kind == "string":
        pattern = re.compile(rf"^\s*{escaped}\s*=\s*'(?P<value>[^']+)'", re.IGNORECASE)
    else:
        pattern = re.compile(rf"^\s*{escaped}\s*=\s*(?P<value>[-+0-9.eEdD]+)", re.IGNORECASE)
    for number, line in enumerate(lines, start=1):
        match = pattern.search(line)
        if match:
            raw = match.group("value")
            if kind == "string":
                return raw, number, line.strip()
            numeric = float(raw.replace("D", "E").replace("d", "e"))
            return int(numeric) if kind == "integer" else numeric, number, line.strip()
    raise ValueError(f"Native configuration parameter {parameter!r} was not found in {FLOW3D_PATH.name}")


def parse_phase_configuration(path: Path = FLOW3D_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing native 300 W configuration record: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: list[dict[str, object]] = []
    for field_id, parameter, kind, meaning in PHASE_CONFIGURATION:
        value, source_line, source_text = _line_value(lines, parameter, kind)
        rows.append(
            {
                "field_id": field_id,
                "native_parameter": parameter,
                "value": value,
                "unit": "K" if field_id.endswith("temperature_K") else ("Pa" if field_id.endswith("pressure_Pa") else ("J kg^-1" if field_id.endswith("J_per_kg") else ("s" if field_id.endswith("_s") else ("W" if field_id.endswith("_W") else "")))),
                "meaning": meaning,
                "source_file": path.relative_to(ROOT).as_posix(),
                "source_line": source_line,
                "source_text": source_text,
                "evidence_status": "direct_observation_300W_configuration",
                "interpretation_boundary": "Native 300 W configuration record only; not a validation of output fidelity.",
            }
        )
    rows.append(
        {
            "field_id": "other_power_difference_statement",
            "native_parameter": "author statement",
            "value": "only laser power differs for the other five cases",
            "unit": "",
            "meaning": "scope statement for the unavailable five configuration exports",
            "source_file": "author statement",
            "source_line": "",
            "source_text": "Not independently verified from five separate native project files.",
            "evidence_status": "author_attested_not_independently_verified",
            "interpretation_boundary": "Must not be treated as a five-case native configuration audit.",
        }
    )
    return pd.DataFrame(rows)


def build_input_manifest(snapshots: list[SnapshotFile]) -> pd.DataFrame:
    point_rows = snapshot_manifest(snapshots, ROOT)
    point_rows.insert(0, "input_kind", "point_cloud_csv")
    point_rows["availability"] = "available"
    point_rows["validation_detail"] = "complete 5-time by 6-power point-cloud grid"
    rows = point_rows.to_dict(orient="records")
    for kind, path, detail in (
        ("native_300W_configuration", FLOW3D_PATH, "provided 300 W FLOW-3D input/configuration export"),
        ("native_300W_run_record", RUNNING_PATH, "provided 300 W FLOW-3D run record"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required thermal-fidelity input is missing: {path}")
        rows.append(
            {
                "input_kind": kind,
                "time_s": "",
                "power_W": 300,
                "relative_path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "data_rows": "",
                "sha256": sha256_file(path),
                "availability": "available",
                "validation_detail": detail,
            }
        )
    return pd.DataFrame(rows).sort_values(["input_kind", "time_s", "power_W"], kind="stable").reset_index(drop=True)


def load_inputs() -> ThermalFidelityInputs:
    snapshots = discover_snapshots(RAW_DIR, TEMPORAL_DIR)
    return ThermalFidelityInputs(
        snapshots=snapshots,
        manifest=build_input_manifest(snapshots),
        phase_configuration=parse_phase_configuration(),
    )

