from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CANONICAL_TIME_S


def build_summary(
    quantiles: pd.DataFrame,
    canonical_reproduction: pd.DataFrame,
    aggregation: pd.DataFrame,
    provenance: pd.DataFrame,
    health: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    snapshot = quantiles.loc[np.isclose(quantiles["time_s"], CANONICAL_TIME_S)].set_index("power_W")
    value_350 = float(snapshot.loc[350, "velocity_max_mps"])
    value_400 = float(snapshot.loc[400, "velocity_max_mps"])
    p95_350 = float(snapshot.loc[350, "velocity_p95_mps"])
    p95_400 = float(snapshot.loc[400, "velocity_p95_mps"])
    relative_drop = (value_350 - value_400) / value_350 * 100.0
    aggregation_passed = bool(
        aggregation["direction_exported_magnitude"].eq("350>400").all()
    )
    definition_passed = bool(aggregation["definition_direction_matches"].all())
    health_all_passed = bool(health["passed"].all())
    missing_native_history = bool(health["status"].eq("not_available").any())
    final_status = "snapshot_local_descriptor" if (
        bool(canonical_reproduction["passed"].all())
        and aggregation_passed
        and definition_passed
        and health_all_passed
    ) else "audit_only"
    peak_count = int(provenance.loc[provenance["power_W"] == 350, "tied_peak_coordinates"].max())
    summary = pd.DataFrame(
        [
            {
                "metric": "canonical_350W_vmax_mps",
                "value": value_350,
                "unit": "m s^-1",
                "interpretation": "0.70 s full-pool peak-level snapshot descriptor",
            },
            {
                "metric": "canonical_400W_vmax_mps",
                "value": value_400,
                "unit": "m s^-1",
                "interpretation": "0.70 s full-pool peak-level snapshot descriptor",
            },
            {
                "metric": "relative_vmax_drop_350_to_400_pct",
                "value": relative_drop,
                "unit": "%",
                "interpretation": "adjacent sampled-power peak difference; not a whole-pool or causal effect",
            },
            {
                "metric": "p95_velocity_350W_mps",
                "value": p95_350,
                "unit": "m s^-1",
                "interpretation": "distributional context for the peak descriptor",
            },
            {
                "metric": "p95_velocity_400W_mps",
                "value": p95_400,
                "unit": "m s^-1",
                "interpretation": "distributional context for the peak descriptor",
            },
            {
                "metric": "tied_350W_peak_coordinates",
                "value": peak_count,
                "unit": "unique coordinates",
                "interpretation": "peak provenance support count",
            },
        ]
    )
    decision = {
        "claim_id": "adjacent_vmax_350_400",
        "analysis_scope": "six discrete late-time snapshot descriptors; no continuous-power, physical-mechanism, grid-convergence, or timestep-convergence inference",
        "canonical_time_s": CANONICAL_TIME_S,
        "canonical_values": {
            "vmax_350W_mps": value_350,
            "vmax_400W_mps": value_400,
            "relative_drop_350_to_400_pct": relative_drop,
            "p95_350W_mps": p95_350,
            "p95_400W_mps": p95_400,
        },
        "peak_provenance": {
            "tied_350W_unique_coordinates": peak_count,
            "description": "The 350 W maximum is a sparse peak-level descriptor; quantile context is reported separately.",
        },
        "gates": {
            "canonical_vmax_reproduction": bool(canonical_reproduction["passed"].all()),
            "aggregation_direction_consistency": aggregation_passed,
            "velocity_definition_direction_consistency": definition_passed,
            "solver_health_all_gates_passed": health_all_passed,
            "solver_history_not_available": missing_native_history,
        },
        "final_status": final_status,
        "allowed_interpretation": (
            "Peak-level 0.70 s velocity descriptor only."
            if final_status == "snapshot_local_descriptor"
            else "Transparent audit record only; a native solver-history health check is unavailable or failed."
        ),
        "prohibited_interpretation": [
            "comparative_evidence",
            "physical_mechanism_evidence",
            "whole-pool flow-strength decrease",
            "grid or timestep convergence",
            "causal explanation for the velocity peak",
        ],
    }
    return summary, decision

