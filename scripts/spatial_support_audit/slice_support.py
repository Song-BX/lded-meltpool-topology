from __future__ import annotations

import numpy as np
import pandas as pd


MIN_VALID_POINTS = 100
REQUESTED_TOP_N = 10


def _unique_count(frame: pd.DataFrame, columns: list[str]) -> int:
    return int(frame.loc[:, columns].drop_duplicates().shape[0])


def audit_slice_support(point_clouds: dict[int, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for power, frame in sorted(point_clouds.items()):
        slice_frame = frame.loc[frame["is_slice_near_symmetry"].astype(int) == 1].copy()
        coordinates = slice_frame[["x_m", "y_m", "z_m"]].apply(pd.to_numeric, errors="coerce")
        q_values = pd.to_numeric(slice_frame["Q"], errors="coerce")
        finite_coordinates = np.isfinite(coordinates).all(axis=1)
        finite_q = np.isfinite(q_values)
        valid = slice_frame.loc[finite_coordinates & finite_q].copy()
        positive = valid.loc[pd.to_numeric(valid["Q"], errors="coerce") > 0].copy()
        reasons = [
            f"valid_slice_points={len(valid)} is below the existing {MIN_VALID_POINTS}-point evidence threshold",
        ]
        if len(positive) < REQUESTED_TOP_N:
            reasons.append(f"positive_Q_points={len(positive)} is below requested top-{REQUESTED_TOP_N}")
        records.append(
            {
                "power_W": power,
                "slice_rows_flagged": len(slice_frame),
                "valid_slice_points": len(valid),
                "finite_Q_points": int(finite_q.sum()),
                "positive_Q_points": len(positive),
                "unique_3d_coordinates": _unique_count(valid, ["x_m", "y_m", "z_m"]),
                "unique_projected_XZ_coordinates": _unique_count(valid, ["x_m", "z_m"]),
                "requested_top_n": REQUESTED_TOP_N,
                "positive_Q_top_n_available": len(positive),
                "Qpos_top10_possible": len(positive) >= REQUESTED_TOP_N,
                "evidence_threshold_valid_points": MIN_VALID_POINTS,
                "evidence_status": "insufficient_support",
                "exclusion_reason": "; ".join(reasons),
            }
        )
    return pd.DataFrame(records).sort_values("power_W").reset_index(drop=True)
