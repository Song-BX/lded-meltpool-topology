from __future__ import annotations

import pandas as pd

from .config import (
    EXPECTED_FINAL_CYCLE,
    EXPECTED_FINAL_TIME_S,
    EXPECTED_PROGRESS_ROWS,
    EXPECTED_STABILITY_EVENTS,
    SATURATION_TEMPERATURE_K,
)


def gate_audit(configuration: pd.DataFrame, progress: pd.DataFrame, events: pd.DataFrame, log_summary: pd.DataFrame) -> pd.DataFrame:
    by_field = configuration.set_index("field_id")
    by_metric = log_summary.set_index("metric")["value"]
    tests = (
        ("phase_model_configured", int(by_field.loc["phase_change_enabled", "value"]) == 1, "if_phchg=1 is directly present in 300 W configuration", "No native phase-change switch was located."),
        ("saturation_temperature_configured", float(by_field.loc["saturation_temperature_K", "value"]) == SATURATION_TEMPERATURE_K, "tv1=3134 K is directly present in 300 W configuration", "Expected saturation-temperature entry was not verified."),
        ("recoil_pressure_disabled", int(by_field.loc["recoil_pressure_enabled", "value"]) == 0, "if_prsrecoil=0 is directly present in 300 W configuration", "Expected recoil-pressure switch was not verified."),
        ("progress_record_complete", len(progress) == EXPECTED_PROGRESS_ROWS, f"{len(progress)} parsed progress rows", "Progress-row parser did not reproduce the supplied record."),
        ("adaptive_events_recorded", len(events) == EXPECTED_STABILITY_EVENTS and bool(events["restart_with_smaller_timestep_reported"].all()), f"{len(events)} events with reported smaller-step restarts", "Expected adaptive restart events were not reproduced."),
        ("normal_completion_recorded", bool(by_metric["normal_completion_reported"]) and int(by_metric["completion_cycle"]) == EXPECTED_FINAL_CYCLE and abs(float(by_metric["completion_time_s"]) - EXPECTED_FINAL_TIME_S) < 1e-9, "normal completion at supplied end time/cycle", "Completion line was absent or mismatched."),
        ("no_nan_inf_tokens_recorded", int(by_metric["nan_or_inf_token_lines"]) == 0, "zero textual NaN/Inf tokens", "At least one textual NaN/Inf token occurred."),
        ("configured_residual_acceptance_available", False, "not available in supplied log", "The log supplies printed res/epsi values but no configured acceptance target."),
        ("all_six_solver_histories_available", False, "only 300 W running record supplied", "No native run records were supplied for the other five powers."),
        ("mesh_timestep_convergence_available", False, "not supplied", "No mesh or time-step convergence study was supplied."),
        ("case_matched_experimental_temperature_available", False, "not supplied", "No case-matched melt-pool temperature experiment was supplied."),
    )
    return pd.DataFrame(
        [
            {
                "gate_id": gate_id,
                "passed": passed,
                "observed_value": observed,
                "failure_reason": "" if passed else failure,
                "scope": "300W_native_record" if gate_id in {"phase_model_configured", "saturation_temperature_configured", "recoil_pressure_disabled", "progress_record_complete", "adaptive_events_recorded", "normal_completion_recorded", "no_nan_inf_tokens_recorded", "configured_residual_acceptance_available"} else "current_six_case_fidelity",
            }
            for gate_id, passed, observed, failure in tests
        ]
    )


def decision_payload(configuration: pd.DataFrame, progress: pd.DataFrame, events: pd.DataFrame, log_summary: pd.DataFrame, tail: pd.DataFrame, gate_frame: pd.DataFrame) -> dict[str, object]:
    canonical = tail.loc[(tail["time_s"] == 0.70) & (tail["representation"] == "exact_coordinate_mean")]
    tail_350_050 = tail.loc[(tail["time_s"] == 0.50) & (tail["power_W"] == 350) & (tail["representation"] == "exact_coordinate_mean")].iloc[0]
    failed_current = gate_frame.loc[(gate_frame["scope"] == "current_six_case_fidelity") & ~gate_frame["passed"], "gate_id"].tolist()
    return {
        "analysis_scope": "Native 300 W configuration and run-record audit plus unfiltered 30-snapshot exported-temperature-tail audit; no CFD rerun or physical-fidelity validation.",
        "phase_model": {
            "status": "direct_observation_300W_configuration",
            "phase_change_enabled": int(configuration.set_index("field_id").loc["phase_change_enabled", "value"]),
            "saturation_temperature_K": float(configuration.set_index("field_id").loc["saturation_temperature_K", "value"]),
            "recoil_pressure_enabled": int(configuration.set_index("field_id").loc["recoil_pressure_enabled", "value"]),
            "other_power_configuration_status": "author_attested_power_only_difference_not_independently_verified",
        },
        "solver_execution_record": {
            "status": "direct_observation",
            "power_W": 300,
            "progress_records": int(len(progress)),
            "normal_completion_reported": bool(log_summary.set_index("metric").loc["normal_completion_reported", "value"]),
            "completion_time_s": float(log_summary.set_index("metric").loc["completion_time_s", "value"]),
            "completion_cycle": int(log_summary.set_index("metric").loc["completion_cycle", "value"]),
        },
        "adaptive_stability_events": {
            "status": "direct_observation",
            "power_W": 300,
            "event_count": int(len(events)),
            "restart_with_smaller_timestep_reported_all_events": bool(events["restart_with_smaller_timestep_reported"].all()),
            "boundary": "Events are native 300 W log facts, not a failure diagnosis or explanation of high-temperature coordinates in another case.",
        },
        "temperature_tail": {
            "status": "audit_only",
            "canonical_representation": "unfiltered exact_coordinate_mean",
            "canonical_070_snapshot_rows": int(len(canonical)),
            "all_snapshot_unique_coordinates_T_ge_3134K": int(tail.loc[tail["representation"] == "exact_coordinate_mean", "n_T_ge_Tsat"].sum()),
            "all_snapshot_unique_coordinates_T_gt_5000K": int(tail.loc[tail["representation"] == "exact_coordinate_mean", "n_T_gt_5000"].sum()),
            "maximum_exported_temperature_K": float(tail["T_max_K"].max()),
            "350W_050s_maximum_context": {
                "T_max_K": float(tail_350_050["T_max_K"]),
                "n_unique_T_gt_5000": int(tail_350_050["n_T_gt_5000"]),
                "n_unique_T_ge_3134": int(tail_350_050["n_T_ge_Tsat"]),
            },
        },
        "high_temperature_cause_350_400W": "not_supported",
        "all_six_solver_health": "not_supported",
        "current_temperature_field_physical_fidelity": "not_supported",
        "central_temperature_descriptors": "snapshot_local_descriptor",
        "failed_current_fidelity_gates": failed_current,
        "prohibited_interpretations": [
            "300 W normal completion validates the other five runs",
            "the two 300 W restarts explain the 350 W 9981.54 K exported coordinate",
            "printed res/epsi values establish convergence without configured acceptance targets",
            "a sparse maximum establishes physical temperature fidelity or numerical stability",
            "temperature filtering replaces the canonical unfiltered numerical export",
        ],
    }

