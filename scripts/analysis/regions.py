from __future__ import annotations

import pandas as pd


REGION_ORDER = ("all", "interface", "heated", "interface_heated")
REGION_LABELS = {
    "all": "full-pool",
    "interface": "interface",
    "heated": "heated",
    "interface_heated": "interface-heated",
}

_ALIASES = {
    "R_all": "all",
    "R_int": "interface",
    "R_heat": "heated",
    "R_(int∩heat)": "interface_heated",
    "R_(int鈭﹉eat)": "interface_heated",
}


def canonical_region(region: str) -> str:
    canonical = _ALIASES.get(region, region)
    if canonical not in REGION_ORDER:
        raise ValueError(f"Unknown region: {region}")
    return canonical


def region_mask(
    frame: pd.DataFrame,
    region: str,
    *,
    fof_interface_threshold: float = 0.99,
    heat_flux_threshold: float = 0.0,
) -> pd.Series:
    """Return the manuscript region mask on WLS-valid points."""
    canonical = canonical_region(region)
    valid = frame["chi"] == 1
    if canonical == "all":
        return valid
    if canonical == "interface":
        return valid & (frame["fof"] < fof_interface_threshold)
    if canonical == "heated":
        return valid & (frame["heat_flux"] > heat_flux_threshold)
    return (
        valid
        & (frame["fof"] < fof_interface_threshold)
        & (frame["heat_flux"] > heat_flux_threshold)
    )
