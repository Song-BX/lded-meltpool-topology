from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.regions import region_mask
from scripts.analysis.tensor_metrics import q_from_velocity_gradients
from scripts.analysis.wls_q import reconstruct_case

from .config import (
    EXPECTED_POWERS,
    FOF_INTERFACE_THRESHOLD,
    K_REFERENCE,
    NATIVE_REFERENCE_DIR,
    WLS_CONDITION_CUTOFF,
    WLS_CONDITION_MODE,
    WLS_DISTANCE_EXPONENT,
    WLS_DISTANCE_OFFSET_M,
)
from .metrics import gradients_from_reconstruction


NATIVE_GRADIENT_ALIASES = {
    "du_dx": ("du_dx", "VelocityGradient_0_0", "Velocity Gradient 0 0"),
    "du_dy": ("du_dy", "VelocityGradient_0_1", "Velocity Gradient 0 1"),
    "du_dz": ("du_dz", "VelocityGradient_0_2", "Velocity Gradient 0 2"),
    "dv_dx": ("dv_dx", "VelocityGradient_1_0", "Velocity Gradient 1 0"),
    "dv_dy": ("dv_dy", "VelocityGradient_1_1", "Velocity Gradient 1 1"),
    "dv_dz": ("dv_dz", "VelocityGradient_1_2", "Velocity Gradient 1 2"),
    "dw_dx": ("dw_dx", "VelocityGradient_2_0", "Velocity Gradient 2 0"),
    "dw_dy": ("dw_dy", "VelocityGradient_2_1", "Velocity Gradient 2 1"),
    "dw_dz": ("dw_dz", "VelocityGradient_2_2", "Velocity Gradient 2 2"),
}
COORDINATE_ALIASES = {
    "x": ("x", "Points_0"),
    "y": ("y", "Points_1"),
    "z": ("z", "Points_2"),
}


def _match_columns(columns: list[str], aliases: dict[str, tuple[str, ...]]) -> dict[str, str] | None:
    selected: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        match = next((candidate for candidate in candidates if candidate in columns), None)
        if match is None:
            return None
        selected[canonical] = match
    return selected


def compare_native_reference(cases: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use a future compatible native-gradient export when one is supplied.

    Missing or semantically incompatible exports are reported as unavailable rather
    than silently converted into a surrogate reference.
    """
    status_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    if not NATIVE_REFERENCE_DIR.exists():
        return (
            pd.DataFrame(
                [
                    {
                        "status": "not_available",
                        "reason": "No native-gradient export directory was supplied.",
                    }
                ]
            ),
            pd.DataFrame(),
        )

    for power in EXPECTED_POWERS:
        candidates = sorted(NATIVE_REFERENCE_DIR.glob(f"*{power}W*.csv"))
        if len(candidates) != 1:
            status_rows.append(
                {
                    "power_W": power,
                    "status": "not_available",
                    "reason": f"Expected one native-gradient CSV, found {len(candidates)}.",
                }
            )
            continue
        native = pd.read_csv(candidates[0])
        coordinate_columns = _match_columns(native.columns.tolist(), COORDINATE_ALIASES)
        gradient_columns = _match_columns(native.columns.tolist(), NATIVE_GRADIENT_ALIASES)
        if coordinate_columns is None or gradient_columns is None:
            status_rows.append(
                {
                    "power_W": power,
                    "status": "schema_incompatible",
                    "reason": "Coordinates or nine velocity-gradient components could not be identified.",
                }
            )
            continue
        normalized = native[
            list(coordinate_columns.values()) + list(gradient_columns.values())
        ].rename(columns={**{v: k for k, v in coordinate_columns.items()}, **{v: k for k, v in gradient_columns.items()}})
        grouped = normalized.groupby(["x", "y", "z"], as_index=False).mean(numeric_only=True)
        canonical = cases[power].copy()
        canonical["point_index"] = np.arange(len(canonical))
        merged = canonical.merge(grouped, on=["x", "y", "z"], how="left", validate="one_to_one")
        matched = merged[list(NATIVE_GRADIENT_ALIASES)].notna().all(axis=1)
        status_rows.append(
            {
                "power_W": power,
                "status": "matched" if matched.all() else "partial_match",
                "reason": "Exact-coordinate map after independent native export grouping.",
                "canonical_points": len(canonical),
                "matched_points": int(matched.sum()),
            }
        )
        if not matched.any():
            continue
        reconstructed_frame = reconstruct_case(
            canonical,
            k=K_REFERENCE,
            alpha=WLS_DISTANCE_EXPONENT,
            eps_w=WLS_DISTANCE_OFFSET_M,
            kappa_max=WLS_CONDITION_CUTOFF,
            condition_on=WLS_CONDITION_MODE,
        )
        reconstructed = gradients_from_reconstruction(reconstructed_frame)
        native_tensor = np.stack(
            [merged.loc[:, ["du_dx", "du_dy", "du_dz"]].to_numpy(), merged.loc[:, ["dv_dx", "dv_dy", "dv_dz"]].to_numpy(), merged.loc[:, ["dw_dx", "dw_dy", "dw_dz"]].to_numpy()],
            axis=1,
        )
        _, _, native_q = q_from_velocity_gradients(native_tensor)
        for region in ("all", "interface"):
            region_membership = region_mask(
                reconstructed_frame,
                region,
                fof_interface_threshold=FOF_INTERFACE_THRESHOLD,
            ).to_numpy(dtype=bool)
            usable = matched.to_numpy(dtype=bool) & region_membership & (reconstructed_frame["chi"].to_numpy() == 1)
            if not usable.any():
                continue
            estimated = reconstructed[usable]
            reference = native_tensor[usable]
            error = estimated - reference
            error_norm = np.linalg.norm(error.reshape(len(error), -1), axis=1)
            reference_norm = np.linalg.norm(reference.reshape(len(reference), -1), axis=1)
            estimated_q = reconstructed_frame.loc[usable, "Q"].to_numpy(dtype=float)
            reference_q = native_q[usable]
            comparison_rows.append(
                {
                    "power_W": power,
                    "region": region,
                    "matched_valid_points": int(usable.sum()),
                    "gradient_nrmse": float(
                        np.sqrt(np.mean(error_norm**2)) / np.sqrt(np.mean(reference_norm**2))
                    ) if np.any(reference_norm > 0) else np.nan,
                    "q_nrmse": float(
                        np.sqrt(np.mean((estimated_q - reference_q) ** 2))
                        / np.sqrt(np.mean(reference_q**2))
                    ) if np.any(reference_q != 0) else np.nan,
                    "q_sign_accuracy": float(((estimated_q > 0) == (reference_q > 0)).mean()),
                    "semantic_status": "requires_user_confirmation_of_point_or_cell_association",
                }
            )
    return pd.DataFrame(status_rows), pd.DataFrame(comparison_rows)
