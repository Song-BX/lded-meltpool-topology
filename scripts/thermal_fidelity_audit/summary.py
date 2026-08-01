from __future__ import annotations

import pandas as pd


def build_summary(tail: pd.DataFrame, sensitivity: pd.DataFrame, decision: dict[str, object]) -> pd.DataFrame:
    canonical = tail.loc[(tail["time_s"] == 0.70) & (tail["representation"] == "exact_coordinate_mean")].copy()
    canonical["summary_group"] = "canonical_070_unfiltered_temperature"
    canonical["status"] = "snapshot_local_descriptor"
    special = sensitivity.loc[(sensitivity["time_s"] == 0.50) & (sensitivity["power_W"] == 350)].copy()
    special["summary_group"] = "350W_050s_tail_sensitivity"
    special["status"] = "audit_only"
    output = pd.concat([canonical, special], ignore_index=True, sort=False)
    output["overall_physical_fidelity_status"] = decision["current_temperature_field_physical_fidelity"]
    return output

