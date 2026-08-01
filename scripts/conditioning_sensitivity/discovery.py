from __future__ import annotations

import pandas as pd

from scripts.robustness.discovery import CaseInput, discover_inputs as _discover_inputs
from scripts.robustness.discovery import manifest_frame as _manifest_frame


def discover_inputs() -> list[CaseInput]:
    """Reuse the canonical six-file schema and hash audit."""
    return _discover_inputs()


def manifest_frame(records: list[CaseInput]) -> pd.DataFrame:
    return _manifest_frame(records)
