from __future__ import annotations

import hashlib
import importlib
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .config import ROOT, SOURCE_SPECS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialise(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialise(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _serialise(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_serialise(item) for item in value]
    return value


def load_sources() -> tuple[dict[str, dict[str, Any]], list[dict[str, object]]]:
    values: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, object]] = []
    for spec in SOURCE_SPECS:
        path = ROOT / spec.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing Comment 18 configuration source: {path}")
        module = importlib.import_module(spec.module_name)
        missing = [name for name in spec.required_attributes if not hasattr(module, name)]
        if missing:
            raise ValueError(f"Missing attributes in {spec.module_name}: {missing}")
        values[spec.key] = {
            name: _serialise(getattr(module, name)) for name in spec.required_attributes
        }
        manifest_rows.append(
            {
                "source_key": spec.key,
                "relative_path": spec.relative_path.as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "required_attributes": ";".join(spec.required_attributes),
                "missing_attributes": "",
                "validation_passed": True,
            }
        )
    return values, manifest_rows

