from __future__ import annotations

import pandas as pd

from .config import POWERS


ENDPOINT_STATUS = "endpoint_not_tested"
LOCAL_MAXIMUM = "discrete_local_maximum"
LOCAL_MINIMUM = "discrete_local_minimum"
NOT_EXTREMUM = "not_local_extremum"


def classify_discrete_extrema(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    required = set(group_columns) | {"power_W", "metric_id", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cannot classify extrema; missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for group_key, block in frame.groupby(group_columns, dropna=False, sort=True):
        block = block.copy()
        block["power_W"] = pd.to_numeric(block["power_W"], errors="raise").astype(int)
        block = block.sort_values("power_W")
        if tuple(block["power_W"]) != POWERS or block["power_W"].duplicated().any():
            raise ValueError(f"Incomplete or duplicated power grid for group {group_key}")
        values = block.set_index("power_W")["value"].astype(float)
        for _, row in block.iterrows():
            power = int(row["power_W"])
            result = row.to_dict()
            result["left_power_W"] = power - 50 if power != POWERS[0] else pd.NA
            result["right_power_W"] = power + 50 if power != POWERS[-1] else pd.NA
            result["left_value"] = float(values.loc[power - 50]) if power != POWERS[0] else pd.NA
            result["right_value"] = float(values.loc[power + 50]) if power != POWERS[-1] else pd.NA
            if power in (POWERS[0], POWERS[-1]):
                result["extremum_status"] = ENDPOINT_STATUS
            elif values.loc[power] > values.loc[power - 50] and values.loc[power] > values.loc[power + 50]:
                result["extremum_status"] = LOCAL_MAXIMUM
            elif values.loc[power] < values.loc[power - 50] and values.loc[power] < values.loc[power + 50]:
                result["extremum_status"] = LOCAL_MINIMUM
            else:
                result["extremum_status"] = NOT_EXTREMUM
            rows.append(result)
    return pd.DataFrame(rows).sort_values(group_columns + ["power_W"], ignore_index=True)

