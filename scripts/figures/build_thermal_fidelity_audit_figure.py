"""Build Fig. S11 from the bounded native-configuration and temperature-tail audit.

Figure contract
---------------
Core conclusion: the supplied 300 W native files document a configured phase
model and a normally completed, adaptively stepped run, while the sparse
high-temperature output tail remains an audit-only numerical-output record
without a cross-power causal or physical-fidelity interpretation.
Evidence chain: panel a records settings; b records the two native log events;
c shows the full 30-snapshot unfiltered tail; d makes the canonical/filtered
boundary and failed fidelity gates explicit.  Archetype: quantitative grid.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="thermal_fidelity_fig_mpl_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

try:
    from .export_policy import figure_suffixes
except ImportError:  # Direct script execution.
    from export_policy import figure_suffixes


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "图" / "thermal_fidelity_audit"
FIG_DIR = ROOT / "latex_restructure" / "figures"
TRACE_PATH = ROOT / "图" / "figure_traceability.csv"
STEM = "FigS11_phase_model_solver_record_temperature_tail_audit"
MM = 1 / 25.4
DOUBLE_W = 183 * MM
BLUE = "#0F4D92"
ORANGE = "#D46A1F"
RED = "#B64342"
TEAL = "#42949E"
GRAY = "#4D4D4D"
LIGHT = "#F2F2F2"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "legend.fontsize": 5.0,
        "xtick.labelsize": 5.1,
        "ytick.labelsize": 5.1,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _clean(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", pad=1.4)


def _panel(axis: plt.Axes, label: str) -> None:
    axis.text(-0.15, 1.04, label, transform=axis.transAxes, fontsize=8, fontweight="bold")


def _configuration_panel(axis: plt.Axes, configuration: pd.DataFrame) -> None:
    values = configuration.set_index("field_id")["value"]
    axis.axis("off")
    axis.set_title("Supplied native 300 W phase-model record", loc="left", pad=3)
    records = [
        ("Phase change", r"$if_{phchg}=1$", "configured"),
        (r"$T_{sat}$", f"{float(values['saturation_temperature_K']):.0f} K", "configured"),
        (r"$L_v$", f"{float(values['latent_heat_vaporization_J_per_kg']):.3g} J kg$^{{-1}}$", "configured"),
        ("Recoil pressure", r"$if_{prsrecoil}=0$", "disabled"),
        ("Laser power", f"{float(values['laser_power_W']):.0f} W", "300 W file"),
    ]
    for index, (label, value, note) in enumerate(records):
        y = 0.90 - index * 0.145
        axis.add_patch(FancyBboxPatch((0.03, y - 0.06), 0.94, 0.105, boxstyle="round,pad=0.008", facecolor="#EAF2F8", edgecolor="#8AA9C4", linewidth=0.5))
        axis.text(0.06, y, label, va="center", ha="left", fontsize=5.8)
        axis.text(0.58, y, value, va="center", ha="left", fontsize=5.7, fontweight="bold")
        axis.text(0.95, y, note, va="center", ha="right", fontsize=5.0, color=GRAY)
    axis.text(0.03, 0.045, "The other-power power-only difference is an author statement,\nnot a five-file native configuration verification.", fontsize=5.0, color=RED, va="bottom")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)


def _timeline_panel(axis: plt.Axes, progress: pd.DataFrame, events: pd.DataFrame, decision: dict[str, object]) -> None:
    axis.plot(progress["time_s"], progress["delt_s"] * 1e5, color=BLUE, lw=0.8, label="reported delt")
    for item in events.itertuples(index=False):
        axis.axvline(float(item.time_s), color=ORANGE, linestyle="--", linewidth=0.9, zorder=2)
        axis.scatter(float(item.time_s), 4.3, s=22, color=ORANGE, edgecolor="white", linewidth=0.35, zorder=3)
        axis.annotate(f"cycle {int(item.cycle)}\nsmaller-step restart", (float(item.time_s), 4.3), xytext=(4, 4), textcoords="offset points", fontsize=4.8, color=ORANGE)
    axis.set_xlim(0, float(decision["solver_execution_record"]["completion_time_s"]))
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel(r"Reported $delt$ ($10^{-5}$ s)")
    axis.set_title("300 W native log: adaptive time-step events", loc="left", pad=3)
    axis.text(0.02, 0.95, f"normal completion: cycle {decision['solver_execution_record']['completion_cycle']}\n{decision['solver_execution_record']['progress_records']} progress records; no textual NaN/Inf", transform=axis.transAxes, va="top", fontsize=5.0, bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F7F7F7", "edgecolor": "none"})
    _clean(axis)


def _tail_panel(axis: plt.Axes, tail: pd.DataFrame) -> None:
    data = tail.loc[tail["representation"] == "exact_coordinate_mean"].copy()
    colors = {200: "#4C78A8", 250: "#72B7B2", 300: "#54A24B", 350: "#F58518", 400: "#E45756", 450: "#B279A2"}
    for power, block in data.groupby("power_W", sort=True):
        axis.scatter(block["time_s"], block["T_max_K"], s=17, color=colors[int(power)], edgecolor="white", linewidth=0.3, label=f"{int(power)} W", zorder=3)
    axis.axhline(3134, color=TEAL, lw=0.8, linestyle="--", label=r"configured $T_{sat}$")
    axis.axhline(5000, color=RED, lw=0.8, linestyle=":", label="tail screen")
    axis.set_xlim(0.495, 0.705)
    axis.set_xticks([0.50, 0.55, 0.60, 0.65, 0.70])
    axis.set_xlabel("Snapshot time (s)")
    axis.set_ylabel(r"Unfiltered exported $T_{max}$ (K)")
    axis.set_title("30-snapshot high-temperature numerical-output tail", loc="left", pad=3)
    axis.legend(ncol=2, loc="upper left", columnspacing=0.65, handletextpad=0.25)
    _clean(axis)


def _sensitivity_gate_panel(axis: plt.Axes, sensitivity: pd.DataFrame, gates: pd.DataFrame) -> None:
    block = sensitivity.loc[(np.isclose(sensitivity["time_s"], 0.50)) & (sensitivity["power_W"] == 350)].set_index("sensitivity_condition")
    order = ["unfiltered", "exclude_T_gt_5000_K", "exclude_T_ge_Tsat"]
    labels = ["unfiltered", r"exclude $T>5000$", r"exclude $T\geq T_{sat}$"]
    med = [float(block.loc[key, "T_median_K"]) for key in order]
    mean = [float(block.loc[key, "T_mean_K"]) for key in order]
    x = np.arange(3)
    axis.bar(x - 0.17, med, 0.32, color=BLUE, label="median")
    axis.bar(x + 0.17, mean, 0.32, color=ORANGE, label="mean")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=14, ha="right")
    axis.set_ylabel("Temperature (K)")
    axis.set_title("350 W, 0.50 s tail sensitivity and blocking gates", loc="left", pad=3)
    axis.legend(loc="upper right")
    gate_names = ["all_six_solver_histories_available", "mesh_timestep_convergence_available", "case_matched_experimental_temperature_available"]
    gates.set_index("gate_id").loc[gate_names]
    axis.text(
        0.03,
        0.065,
        "Unavailable gates: all-six histories; mesh/time-step convergence;\ncase-matched experimental temperature.",
        transform=axis.transAxes,
        fontsize=4.6,
        va="bottom",
        color=RED,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
    )
    _clean(axis)


def _update_traceability(decision: dict[str, object]) -> None:
    existing = pd.read_csv(TRACE_PATH) if TRACE_PATH.exists() else pd.DataFrame()
    existing = existing.loc[existing["figure_id"] != "Fig. S11"] if len(existing) else existing
    rows = pd.DataFrame(
        [
            {"figure_id": "Fig. S11", "panel_id": "a", "source_csv": "图/thermal_fidelity_audit/phase_model_configuration.csv", "metric_name": "native 300 W phase-model configuration", "reported_value": "phase-change switch, configured saturation temperature/latent heat, recoil-pressure switch, and 300 W power", "verified": "yes"},
            {"figure_id": "Fig. S11", "panel_id": "b", "source_csv": "图/thermal_fidelity_audit/running_log_progress.csv; 图/thermal_fidelity_audit/running_log_events.csv", "metric_name": "300 W native run-record timeline", "reported_value": f"{decision['solver_execution_record']['progress_records']} progress rows; {decision['adaptive_stability_events']['event_count']} reported smaller-step restarts; normal completion at cycle {decision['solver_execution_record']['completion_cycle']}", "verified": "yes"},
            {"figure_id": "Fig. S11", "panel_id": "c", "source_csv": "图/thermal_fidelity_audit/temperature_tail_metrics.csv", "metric_name": "unfiltered 30-snapshot exported-temperature maxima", "reported_value": f"maximum exported temperature={decision['temperature_tail']['maximum_exported_temperature_K']:.2f} K; audit only", "verified": "yes"},
            {"figure_id": "Fig. S11", "panel_id": "d", "source_csv": "图/thermal_fidelity_audit/temperature_tail_sensitivity.csv; 图/thermal_fidelity_audit/thermal_fidelity_gate_audit.csv", "metric_name": "350 W 0.50 s tail sensitivity and physical-fidelity gate boundary", "reported_value": "unfiltered canonical values retained; three current-fidelity gates unavailable", "verified": "yes"},
        ]
    )
    pd.concat([existing, rows], ignore_index=True).to_csv(TRACE_PATH, index=False, encoding="utf-8-sig")


def build_and_update_traceability() -> Path:
    configuration = pd.read_csv(AUDIT_DIR / "phase_model_configuration.csv")
    progress = pd.read_csv(AUDIT_DIR / "running_log_progress.csv")
    events = pd.read_csv(AUDIT_DIR / "running_log_events.csv")
    tail = pd.read_csv(AUDIT_DIR / "temperature_tail_metrics.csv")
    sensitivity = pd.read_csv(AUDIT_DIR / "temperature_tail_sensitivity.csv")
    gates = pd.read_csv(AUDIT_DIR / "thermal_fidelity_gate_audit.csv")
    decision = json.loads((AUDIT_DIR / "thermal_fidelity_decision.json").read_text(encoding="utf-8"))
    if decision["current_temperature_field_physical_fidelity"] != "not_supported" or decision["temperature_tail"]["status"] != "audit_only":
        raise ValueError("Fig. S11 is defined for the bounded thermal-fidelity decision state.")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_W, 126 * MM))
    _configuration_panel(axes[0, 0], configuration)
    _timeline_panel(axes[0, 1], progress, events, decision)
    _tail_panel(axes[1, 0], tail)
    _sensitivity_gate_panel(axes[1, 1], sensitivity, gates)
    for axis, label in zip(axes.ravel(), "abcd"):
        _panel(axis, label)
    fig.text(0.5, 0.008, "Configuration and log records document supplied 300 W inputs only. Tail screens are sensitivity audits; they do not replace unfiltered values or establish numerical convergence, cross-power health, or physical fidelity.", ha="center", va="bottom", fontsize=4.7, color=GRAY)
    fig.tight_layout(rect=[0, 0.04, 1, 1], h_pad=1.7, w_pad=1.8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in figure_suffixes():
        kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix in {".png", ".tiff"}:
            kwargs["dpi"] = 600
        if suffix == ".tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(FIG_DIR / f"{STEM}{suffix}", **kwargs)
    plt.close(fig)
    _update_traceability(decision)
    return FIG_DIR / f"{STEM}.pdf"


def main() -> None:
    print(f"Generated {build_and_update_traceability()}")


if __name__ == "__main__":
    main()
