from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RUNNING_PATH, ROOT


_PROGRESS = re.compile(
    r"^\s*(?P<time_s>[-+0-9.Ee]+)\s+(?P<cycle>\d+)\s+(?P<delt_s>[-+0-9.Ee]+)\s+(?P<dt_stbl_code>\S+)"
    r"\s+(?P<pressure_iterations>\d+)\s+(?P<pressure_res_epsi>[-+0-9.Ee]+)"
    r"\s+(?P<heat_transfer_iterations>\d+)\s+(?P<heat_transfer_res_epsi>[-+0-9.Ee]+)"
    r"\s+(?P<fluid_volume>[-+0-9.Ee]+)\s+(?P<fluid_percent_loss>[-+0-9.Ee]+)\s+(?P<fluid_fraction>[-+0-9.Ee]+)"
    r"\s+(?P<solid_volume>[-+0-9.Ee]+)\s+(?P<solidification_fraction>[-+0-9.Ee]+)\s+(?P<elapsed>\d\d:\d\d:\d\d)"
    r"\s+(?P<percent_pe>\d+)\s+(?P<clock>\d\d:\d\d:\d\d)\s+(?P<estimated_remaining>\S+)",
    re.IGNORECASE,
)
_STABILITY = re.compile(
    r"at t=\s*(?P<time_s>[-+0-9.Ee]+),\s*cycle=\s*(?P<cycle>\d+),\s*iter=\s*(?P<iteration>\d+),\s*delt=\s*(?P<delt_s>[-+0-9.Ee]+)",
    re.IGNORECASE,
)
_COMPLETE = re.compile(r"end of calculation at\s+t\s*=\s*(?P<time_s>[-+0-9.Ee]+),\s*cycle\s*=\s*(?P<cycle>\d+)", re.IGNORECASE)


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def parse_running_log(path: Path = RUNNING_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing native 300 W run record: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    progress_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for number, line in enumerate(lines, start=1):
        matched = _PROGRESS.match(line)
        if matched:
            row = matched.groupdict()
            for key in ("time_s", "delt_s", "pressure_res_epsi", "heat_transfer_res_epsi", "fluid_volume", "fluid_percent_loss", "fluid_fraction", "solid_volume", "solidification_fraction"):
                row[key] = _float(str(row[key]))
            for key in ("cycle", "pressure_iterations", "heat_transfer_iterations", "percent_pe"):
                row[key] = int(str(row[key]))
            row.update({"power_W": 300, "source_file": path.relative_to(ROOT).as_posix(), "source_line": number})
            progress_rows.append(row)
        if "convective flux exceeded stability limit" in line.lower():
            detail = _STABILITY.search(lines[number] if number < len(lines) else "")
            if detail is None:
                raise ValueError(f"Could not parse stability event following line {number}")
            restart_line = lines[number + 1] if number + 1 < len(lines) else ""
            event_rows.append(
                {
                    "power_W": 300,
                    "event_type": "convective_flux_exceeded_stability_limit",
                    "time_s": _float(detail.group("time_s")),
                    "cycle": int(detail.group("cycle")),
                    "iteration": int(detail.group("iteration")),
                    "reported_delt_s": _float(detail.group("delt_s")),
                    "restart_with_smaller_timestep_reported": "restarting cycle with smaller time step" in restart_line.lower(),
                    "source_file": path.relative_to(ROOT).as_posix(),
                    "event_source_line": number,
                    "detail_source_line": number + 1,
                    "restart_source_line": number + 2,
                    "interpretation_boundary": "A reported adaptive time-step restart; not a solver-failure diagnosis or output-cause attribution.",
                }
            )
    completion_row: dict[str, object] | None = None
    for number, line in enumerate(lines, start=1):
        match = _COMPLETE.search(line)
        if match:
            following = lines[number] if number < len(lines) else ""
            completion_row = {
                "power_W": 300,
                "completion_time_s": _float(match.group("time_s")),
                "completion_cycle": int(match.group("cycle")),
                "normal_completion_reported": "normal completion" in following.lower(),
                "completion_reason": following.strip(),
                "completion_source_line": number,
                "status_source_line": number + 1,
            }
            break
    if completion_row is None:
        raise ValueError("No end-of-calculation line found in native 300 W run record")
    progress = pd.DataFrame(progress_rows)
    events = pd.DataFrame(event_rows)
    if progress.empty:
        raise ValueError("No progress rows found in native 300 W run record")
    nonfinite_tokens = sum(bool(re.search(r"\b(?:nan|inf)\b", line, flags=re.IGNORECASE)) for line in lines)
    final = progress.sort_values(["time_s", "cycle"]).iloc[-1]
    summary_rows = [
        ("progress_record_count", len(progress), "rows", "direct_observation"),
        ("stability_event_count", len(events), "events", "direct_observation"),
        ("final_progress_time_s", float(final.time_s), "s", "direct_observation"),
        ("final_progress_cycle", int(final.cycle), "cycle", "direct_observation"),
        ("normal_completion_reported", bool(completion_row["normal_completion_reported"]), "", "direct_observation"),
        ("completion_time_s", float(completion_row["completion_time_s"]), "s", "direct_observation"),
        ("completion_cycle", int(completion_row["completion_cycle"]), "cycle", "direct_observation"),
        ("nan_or_inf_token_lines", nonfinite_tokens, "lines", "direct_observation"),
        ("delt_min_s", float(progress["delt_s"].min()), "s", "direct_observation"),
        ("delt_max_s", float(progress["delt_s"].max()), "s", "direct_observation"),
        ("pressure_res_epsi_max", float(progress["pressure_res_epsi"].max()), "printed res/epsi", "direct_observation"),
        ("pressure_res_epsi_final", float(final.pressure_res_epsi), "printed res/epsi", "direct_observation"),
        ("heat_transfer_res_epsi_max", float(progress["heat_transfer_res_epsi"].max()), "printed res/epsi", "direct_observation"),
        ("heat_transfer_res_epsi_final", float(final.heat_transfer_res_epsi), "printed res/epsi", "direct_observation"),
        ("fluid_percent_loss_min", float(progress["fluid_percent_loss"].min()), "printed %loss", "direct_observation"),
        ("fluid_percent_loss_max", float(progress["fluid_percent_loss"].max()), "printed %loss", "direct_observation"),
        ("fluid_percent_loss_final", float(final.fluid_percent_loss), "printed %loss", "direct_observation"),
    ]
    summary = pd.DataFrame(summary_rows, columns=["metric", "value", "software_label_or_unit", "evidence_status"])
    summary["power_W"] = 300
    summary["interpretation_boundary"] = (
        "Printed labels are retained verbatim; no configured acceptance threshold, mass balance, energy balance, or convergence conclusion is inferred."
    )
    return progress, events, summary

