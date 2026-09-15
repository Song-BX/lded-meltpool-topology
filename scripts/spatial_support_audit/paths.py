from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.analysis.release_paths import reference_input


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "图"
AUDIT_DIR = DATA_DIR / "spatial_support_audit"
SOURCE_INPUT_DIR = AUDIT_DIR / "source_inputs"


@dataclass(frozen=True)
class InputSpec:
    key: str
    original_path: Path
    destination_name: str

    @property
    def destination_path(self) -> Path:
        return SOURCE_INPUT_DIR / self.destination_name


INPUT_SPECS = (
    InputSpec(
        "qpoints_350",
        reference_input(
            ROOT,
            "Qpoints_350W_k25.csv",
            Path("图/spatial_support_audit/source_inputs/Qpoints_350W_k25.csv"),
        ),
        "Qpoints_350W_k25.csv",
    ),
    InputSpec(
        "qpoints_400",
        reference_input(
            ROOT,
            "Qpoints_400W_k25.csv",
            Path("图/spatial_support_audit/source_inputs/Qpoints_400W_k25.csv"),
        ),
        "Qpoints_400W_k25.csv",
    ),
    InputSpec(
        "legacy_extreme_summary",
        reference_input(
            ROOT,
            "Aplus_extreme_localization_350_400.csv",
            Path("图/spatial_support_audit/source_inputs/Aplus_extreme_localization_350_400.csv"),
        ),
        "Aplus_extreme_localization_350_400.csv",
    ),
)


def relative_to_root(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()
