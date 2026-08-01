"""Resolve compact reference inputs in a release package or development tree."""

from __future__ import annotations

from pathlib import Path


def reference_input(root: Path, filename: str, development_fallback: Path) -> Path:
    """Prefer the source-only package input, then retain the development path."""
    packaged = root / "reference_data" / filename
    return packaged if packaged.is_file() else root / development_fallback
