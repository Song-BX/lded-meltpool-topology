from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HEALTH_WINDOW, PEAK_POWERS, POWER_CONFIGURATION_KEYS, SOLVER_ROLES


def _gate(gate_id: str, status: str, observed: str, requirement: str) -> dict[str, object]:
    return {
        "gate_id": gate_id,
        "status": status,
        "passed": status == "passed",
        "observed": observed,
        "requirement": requirement,
    }


def _role_available(history: pd.DataFrame, role: str) -> bool:
    return all(
        not history.loc[(history["power_W"] == power_W) & (history["role"] == role)].empty
        for power_W in PEAK_POWERS
    )


def _configuration_gate(history: pd.DataFrame) -> dict[str, object]:
    if not _role_available(history, "configuration"):
        return _gate("matching_configuration", "not_available", "configuration logs unavailable", "Both configurations must be mapped and match except laser power.")
    values: dict[int, dict[str, str]] = {}
    for power_W in PEAK_POWERS:
        block = history.loc[(history["power_W"] == power_W) & (history["role"] == "configuration")]
        pairs = {
            str(row.variable).strip().lower(): str(row.value)
            for row in block.itertuples(index=False)
            if str(row.variable).strip().lower() not in POWER_CONFIGURATION_KEYS
        }
        values[power_W] = pairs
    passed = bool(values[350]) and values[350] == values[400]
    return _gate(
        "matching_configuration",
        "passed" if passed else "failed",
        "identical non-power configuration" if passed else "non-power configuration differs or is empty",
        "Both configurations must match except laser power.",
    )


def _run_completion_gate(history: pd.DataFrame) -> dict[str, object]:
    if not _role_available(history, "run_history"):
        return _gate("normal_completion", "not_available", "run-history logs unavailable", "Both runs must complete normally through 0.70 s without abort, failure, or restart.")
    failed_tokens = ("abort", "fail", "fatal", "restart", "interrupt", "error")
    complete_tokens = ("complete", "completed", "finish", "finished", "success")
    outcomes: list[bool] = []
    for power_W in PEAK_POWERS:
        block = history.loc[(history["power_W"] == power_W) & (history["role"] == "run_history")]
        status = " ".join(block["run_status"].dropna().astype(str).str.lower())
        reaches_final_time = bool(pd.to_numeric(block["time_s"], errors="coerce").max() >= HEALTH_WINDOW[1])
        normal_status = not any(token in status for token in failed_tokens) and any(token in status for token in complete_tokens)
        outcomes.append(reaches_final_time and normal_status)
    return _gate(
        "normal_completion",
        "passed" if all(outcomes) else "failed",
        "both runs complete through 0.70 s" if all(outcomes) else "one or both runs lack a normal-completion record through 0.70 s",
        "Both runs must complete normally through 0.70 s without abort, failure, or restart.",
    )


def _residual_gate(history: pd.DataFrame) -> dict[str, object]:
    if not _role_available(history, "residual"):
        return _gate("residual_targets", "not_available", "residual logs or configured targets unavailable", "Every reported equation must meet its configured residual target from 0.50 to 0.70 s.")
    lower, upper = HEALTH_WINDOW
    block = history.loc[(history["role"] == "residual") & history["time_s"].between(lower, upper)].copy()
    valid = bool(len(block)) and block[["time_s", "value", "target"]].apply(np.isfinite).all().all()
    passed = valid and bool((block["value"] <= block["target"]).all())
    return _gate(
        "residual_targets",
        "passed" if passed else "failed",
        "all reported residuals meet mapped targets" if passed else "missing, non-finite, or above-target residual record in 0.50--0.70 s",
        "Every reported equation must meet its configured residual target from 0.50 to 0.70 s.",
    )


def _finite_gate(history: pd.DataFrame) -> dict[str, object]:
    if history.empty:
        return _gate("finite_solver_values", "not_available", "no normalised solver history", "No mapped solver-history value may be non-finite.")
    numeric = history[["time_s", "iteration", "timestep_s", "value", "target"]]
    present = numeric.notna()
    finite = pd.DataFrame(
        {column: np.isfinite(numeric[column]) | ~present[column] for column in numeric.columns}
    )
    passed = bool(finite.all().all())
    return _gate(
        "finite_solver_values",
        "passed" if passed else "failed",
        "all mapped numeric values finite" if passed else "at least one mapped numeric value is non-finite",
        "No mapped solver-history value may be non-finite.",
    )


def _acceptance_gate(history: pd.DataFrame, role: str, gate_id: str, label: str) -> dict[str, object]:
    if not _role_available(history, role):
        return _gate(gate_id, "not_available", f"{label} logs or software acceptance criteria unavailable", f"All reported {label} variables must satisfy their mapped software-recorded acceptance criteria from 0.50 to 0.70 s.")
    lower, upper = HEALTH_WINDOW
    block = history.loc[(history["role"] == role) & history["time_s"].between(lower, upper)].copy()
    passed = bool(len(block)) and bool(block["accepted"].all()) and bool(np.isfinite(block["value"]).all())
    return _gate(
        gate_id,
        "passed" if passed else "failed",
        f"all mapped {label} records accepted" if passed else f"missing, non-finite, or non-accepted {label} record in 0.50--0.70 s",
        f"All reported {label} variables must satisfy their mapped software-recorded acceptance criteria from 0.50 to 0.70 s.",
    )


def evaluate_health(history: pd.DataFrame, mapping_issues: tuple[str, ...]) -> pd.DataFrame:
    """Apply only documented solver-history gates; absent native records cannot pass."""
    if mapping_issues:
        return pd.DataFrame(
            [
                _gate(
                    "native_history_availability",
                    "not_available",
                    "; ".join(mapping_issues),
                    "A complete, validated native solver-history mapping is required before health gates can be evaluated.",
                )
            ]
            + [
                _gate(gate, "not_available", "native solver-history mapping unavailable", requirement)
                for gate, requirement in (
                    ("matching_configuration", "Both configurations must match except laser power."),
                    ("normal_completion", "Both runs must complete normally through 0.70 s."),
                    ("residual_targets", "Every reported equation must meet its configured residual target."),
                    ("finite_solver_values", "No mapped solver-history value may be non-finite."),
                    ("stability_acceptance", "Reported CFL/equivalent stability variables must be accepted."),
                    ("conservation_acceptance", "Reported mass, energy, and VOF variables must be accepted."),
                )
            ]
        )
    gates = [
        _configuration_gate(history),
        _run_completion_gate(history),
        _residual_gate(history),
        _finite_gate(history),
        _acceptance_gate(history, "stability", "stability_acceptance", "stability"),
        _acceptance_gate(history, "conservation", "conservation_acceptance", "mass, energy, and VOF conservation"),
    ]
    return pd.DataFrame(gates)

