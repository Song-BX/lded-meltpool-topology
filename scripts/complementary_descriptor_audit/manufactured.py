from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.regions import region_mask
from scripts.analysis.wls_q import reconstruct_case
from scripts.gradient_validation.manufactured_fields import build_manufactured_field

from .config import DESCRIPTOR_SPECS, FIELD_SPECS, FOF_INTERFACE_THRESHOLD, K_VALUES, REGIONS, WLS_CONDITION_CUTOFF, WLS_CONDITION_MODE, WLS_DISTANCE_OFFSET_M


def _truth_values(truth, descriptor: str) -> np.ndarray:
    if descriptor == "Q":
        return truth.true_q
    if descriptor == "lambda2":
        return truth.true_lambda2
    return truth.true_omega_normalized


def run_manufactured_audit(cases: dict[int, pd.DataFrame], alpha_specs) -> pd.DataFrame:
    """Evaluate all descriptors on existing analytic fields at observed geometries."""
    rows: list[dict[str, object]] = []
    for alpha_spec in alpha_specs:
        for power, frame in sorted(cases.items()):
            for field_spec in FIELD_SPECS:
                truth = build_manufactured_field(frame, field_spec)
                for k in K_VALUES:
                    reconstructed = reconstruct_case(
                        truth.frame,
                        k=k,
                        alpha=alpha_spec.value,
                        eps_w=WLS_DISTANCE_OFFSET_M,
                        kappa_max=WLS_CONDITION_CUTOFF,
                        condition_on=WLS_CONDITION_MODE,
                    )
                    for region in REGIONS:
                        mask = region_mask(reconstructed, region, fof_interface_threshold=FOF_INTERFACE_THRESHOLD).to_numpy(dtype=bool)
                        for descriptor_spec in DESCRIPTOR_SPECS:
                            estimated = reconstructed.loc[mask, descriptor_spec.name].to_numpy(dtype=float)
                            true = _truth_values(truth, descriptor_spec.name)[mask]
                            finite = np.isfinite(estimated) & np.isfinite(true)
                            estimated = estimated[finite]
                            true = true[finite]
                            denominator = float(np.sqrt(np.mean(true**2))) if len(true) else np.nan
                            nrmse = (
                                float(np.sqrt(np.mean((estimated - true) ** 2)) / denominator)
                                if np.isfinite(denominator) and denominator > 0 else np.nan
                            )
                            true_class = true > descriptor_spec.threshold if descriptor_spec.relation == "greater" else true < descriptor_spec.threshold
                            estimated_class = estimated > descriptor_spec.threshold if descriptor_spec.relation == "greater" else estimated < descriptor_spec.threshold
                            rows.append(
                                {
                                    "alpha_label": alpha_spec.label,
                                    "alpha": alpha_spec.value,
                                    "power_W": power,
                                    "kNN": k,
                                    "region": region,
                                    "field_id": field_spec.field_id,
                                    "field_class": field_spec.field_class,
                                    "feature_scale_mm": np.nan if field_spec.scale_m is None else field_spec.scale_m * 1000.0,
                                    "descriptor": descriptor_spec.name,
                                    "compared_points": int(len(true)),
                                    "value_nrmse": nrmse,
                                    "value_mae": float(np.mean(np.abs(estimated - true))) if len(true) else np.nan,
                                    "classification_agreement": float((true_class == estimated_class).mean()) if len(true) else np.nan,
                                }
                            )
                print(f"completed complementary manufactured field {field_spec.field_id} for {power} W at alpha={alpha_spec.label}", flush=True)
    return pd.DataFrame(rows)
