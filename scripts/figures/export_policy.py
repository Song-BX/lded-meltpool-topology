"""Central export-format policy for manuscript and repository figure builds."""

from __future__ import annotations

import os


def figure_extensions() -> tuple[str, ...]:
    """Return output extensions, omitting TIFF in the review-release workflow."""
    formats = ("pdf", "svg", "png", "tiff")
    if os.environ.get("R1_RELEASE_NON_TIFF") == "1":
        return formats[:-1]
    return formats


def figure_suffixes() -> tuple[str, ...]:
    """Return :func:`figure_extensions` with a leading period for ``Path`` APIs."""
    return tuple(f".{extension}" for extension in figure_extensions())
