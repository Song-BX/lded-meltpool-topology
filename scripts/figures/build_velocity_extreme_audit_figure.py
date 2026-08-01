"""Build Fig. S10 from the frozen Comment 16 velocity-extreme audit outputs.

Figure contract
---------------
Core conclusion: the 350 W--400 W central velocity ranges overlap, while the
large Vmax difference is carried by sparse peak-level support and cannot be
treated as a whole-pool structural signal or numerically explained without
native solver histories.
Archetype: quantitative grid.  Panel a gives all-six-power distributional
context; b is the peak-provenance hero; c establishes the time-snapshot boundary;
d records the deliberately blocking native-history health gate.  No panel implies
replication, continuous-power interpolation, or a physical mechanism.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="velocity_extreme_fig_mpl_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

try:
    from .export_policy import figure_extensions
except ImportError:  # Direct script execution.
    from export_policy import figure_extensions


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "图" / "velocity_extreme_audit"
OVERLAP_DIR = ROOT / "图" / "velocity_distribution_overlap_audit"
FIGURE_DIR = ROOT / "latex_restructure" / "figures"
TRACE_PATH = ROOT / "图" / "figure_traceability.csv"
STEM = "FigS10_velocity_extreme_solver_health_audit"
MM = 1 / 25.4
DOUBLE_WIDTH = 183 * MM

BLUE = "#0F4D92"
ORANGE = "#D46A1F"
TEAL = "#42949E"
GRAY = "#4D4D4D"
MID_GRAY = "#7A7A7A"
LIGHT = "#D8D8D8"
VERY_LIGHT = "#F2F2F2"
RED = "#B64342"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "legend.fontsize": 5.0,
        "xtick.labelsize": 5.2,
        "ytick.labelsize": 5.2,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.5,
    }
)


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", pad=1.5)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.15, 1.04, label, transform=axis.transAxes, fontsize=8, fontweight="bold")


def _quantile_panel(axis: plt.Axes, quantiles: pd.DataFrame) -> None:
    block = quantiles.loc[np.isclose(quantiles["time_s"], 0.70)].sort_values("power_W")
    powers = block["power_W"].to_numpy(dtype=int)
    axis.vlines(
        powers,
        block["velocity_p25_mps"],
        block["velocity_p75_mps"],
        color="#A7C9C9",
        linewidth=2.0,
        label="P25--P75",
        zorder=2,
    )
    styles = [
        ("velocity_p50_mps", "P50", LIGHT, "o", 14),
        ("velocity_p90_mps", "P90", TEAL, "^", 16),
        ("velocity_p95_mps", "P95", MID_GRAY, "D", 14),
        ("velocity_p99_mps", "P99", "#6585A9", "v", 16),
        ("velocity_max_mps", r"$V_{max}$", BLUE, "o", 23),
    ]
    for column, label, color, marker, size in styles:
        axis.scatter(
            powers,
            block[column],
            s=size,
            marker=marker,
            color=color,
            edgecolor="white" if column != "velocity_p50_mps" else GRAY,
            linewidth=0.35,
            label=label,
            zorder=3,
        )
    axis.axvspan(340, 410, color="#E7F0F8", zorder=0)
    axis.set_xlim(190, 460)
    axis.set_xticks(powers)
    axis.set_xlabel("Laser power (W)")
    axis.set_ylabel(r"Velocity descriptor (m s$^{-1}$)")
    axis.set_title("Six-case 0.70 s central-range and tail context", loc="left", pad=3)
    axis.legend(ncol=3, loc="upper left", handletextpad=0.25, columnspacing=0.6)
    _clean_axis(axis)


def _provenance_panel(
    axis: plt.Axes,
    quantiles: pd.DataFrame,
    provenance: pd.DataFrame,
    overlap: pd.Series,
) -> None:
    block = quantiles.loc[
        np.isclose(quantiles["time_s"], 0.70) & quantiles["power_W"].isin([350, 400])
    ].set_index("power_W")
    labels = ["IQR", "P99", r"$V_{max}$"]
    y = np.arange(len(labels))
    for power_W, color, offset in ((350, BLUE, -0.10), (400, ORANGE, 0.10)):
        q25 = float(block.loc[power_W, "velocity_p25_mps"])
        q75 = float(block.loc[power_W, "velocity_p75_mps"])
        p99 = float(block.loc[power_W, "velocity_p99_mps"])
        vmax = float(block.loc[power_W, "velocity_max_mps"])
        axis.hlines(y[0] + offset, q25, q75, color=color, linewidth=2.5, alpha=0.85)
        axis.hlines(y[1:] + offset, 0, [p99, vmax], color=color, linewidth=1.0, alpha=0.78)
        axis.scatter([p99, vmax], y[1:] + offset, s=22, color=color, edgecolor="white", linewidth=0.35, zorder=3, label=f"{power_W} W")
    peak_350 = provenance.loc[provenance["power_W"] == 350]
    peak_400 = provenance.loc[provenance["power_W"] == 400]
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.set_ylim(-0.55, 2.55)
    axis.set_xlim(0, max(block["velocity_max_mps"]) * 1.17)
    axis.set_xlabel(r"Velocity (m s$^{-1}$)")
    axis.set_title("Central-range overlap and peak-tail provenance", loc="left", pad=3)
    axis.legend(loc="lower right", handletextpad=0.3)
    axis.text(
        0.02,
        0.94,
        "400 W IQR is contained within the 350 W IQR\n"
        f"IQR overlap: {float(overlap.iqr_overlap_width_mps):.6f} m s$^{{-1}}$\n"
        f"350 W max: {len(peak_350)} tied unique coordinates\n"
        f"{int(peak_350['points_above_other_case_vmax'].iloc[0])} unique 350 W points exceed 400 W $V_{{max}}$\n"
        f"400 W max: {len(peak_400)} tied unique coordinate",
        transform=axis.transAxes,
        va="top",
        fontsize=5.1,
        color=RED,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F9E8E6", "edgecolor": "none"},
    )
    _clean_axis(axis)


def _temporal_panel(axis: plt.Axes, quantiles: pd.DataFrame) -> None:
    styles = [
        ("velocity_p95_mps", "P95", "--"),
        ("velocity_p99_mps", "P99", ":"),
        ("velocity_max_mps", r"$V_{max}$", "-"),
    ]
    for power_W, color in ((350, BLUE), (400, ORANGE)):
        block = quantiles.loc[quantiles["power_W"] == power_W].sort_values("time_s")
        for column, label, line_style in styles:
            axis.plot(
                block["time_s"],
                block[column],
                color=color,
                linestyle=line_style,
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label=f"{power_W} W {label}",
            )
    axis.axvspan(0.60, 0.70, color=VERY_LIGHT, zorder=0)
    axis.set_xlim(0.495, 0.705)
    axis.set_xticks([0.50, 0.55, 0.60, 0.65, 0.70])
    axis.set_xlabel("Snapshot time (s)")
    axis.set_ylabel(r"Velocity (m s$^{-1}$)")
    axis.set_title("350/400 W peak and upper-tail trajectories", loc="left", pad=3)
    axis.legend(ncol=2, loc="upper left", handlelength=1.7, columnspacing=0.8)
    _clean_axis(axis)


def _health_panel(axis: plt.Axes, health: pd.DataFrame) -> None:
    axis.axis("off")
    axis.set_title("Native solver-health gate", loc="left", pad=3)
    y_positions = np.linspace(0.88, 0.26, len(health))
    for y, row in zip(y_positions, health.itertuples(index=False)):
        status = str(row.status)
        color = "#E2E2E2" if status == "not_available" else ("#DDF3DE" if status == "passed" else "#F6CFCB")
        edge = "#A0A0A0" if status == "not_available" else ("#5C9F61" if status == "passed" else RED)
        axis.add_patch(
            FancyBboxPatch(
                (0.02, y - 0.039),
                0.96,
                0.076,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                linewidth=0.5,
                facecolor=color,
                edgecolor=edge,
            )
        )
        axis.text(0.04, y, str(row.gate_id).replace("_", " "), ha="left", va="center", fontsize=5.0)
        axis.text(0.95, y, status.replace("_", " "), ha="right", va="center", fontsize=5.0, color=edge, fontweight="bold")
    axis.text(
        0.02,
        0.01,
        "Native residual, stability, and conservation histories were not available.\n"
        "The peak difference therefore remains audit-only, not numerically explained.",
        fontsize=5.0,
        va="bottom",
        color=RED,
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)


def _update_traceability(
    quantiles: pd.DataFrame,
    provenance: pd.DataFrame,
    decision: dict[str, object],
    overlap: pd.Series,
) -> None:
    existing = pd.read_csv(TRACE_PATH) if TRACE_PATH.exists() else pd.DataFrame()
    if len(existing):
        existing = existing.loc[existing["figure_id"] != "Fig. S10"]
    canonical = quantiles.loc[np.isclose(quantiles["time_s"], 0.70)].set_index("power_W")
    peak_count = int(provenance.loc[provenance["power_W"] == 350, "tied_peak_coordinates"].max())
    rows = pd.DataFrame(
        [
            {"figure_id": "Fig. S10", "panel_id": "a", "source_csv": "图/velocity_extreme_audit/velocity_quantiles.csv", "metric_name": "six-case 0.70 s velocity central-range and tail context", "reported_value": "P25--P75, P50, P90--P99, and Vmax shown for all six discrete powers", "verified": "yes"},
            {"figure_id": "Fig. S10", "panel_id": "b", "source_csv": "图/velocity_distribution_overlap_audit/velocity_distribution_overlap_audit.csv; 图/velocity_extreme_audit/peak_provenance.csv", "metric_name": "350/400 W central-range overlap and peak provenance", "reported_value": f"350 W IQR={overlap.p25_350_mps:.6f}--{overlap.p75_350_mps:.6f}; 400 W IQR={overlap.p25_400_mps:.6f}--{overlap.p75_400_mps:.6f}; 350 W tied coordinates={peak_count}", "verified": "yes"},
            {"figure_id": "Fig. S10", "panel_id": "c", "source_csv": "图/velocity_extreme_audit/velocity_quantiles.csv", "metric_name": "350/400 W temporal upper-tail trajectories", "reported_value": "five serial snapshots; P95, P99, and Vmax", "verified": "yes"},
            {"figure_id": "Fig. S10", "panel_id": "d", "source_csv": "图/velocity_extreme_audit/solver_health_gate_audit.csv", "metric_name": "native solver-health gates", "reported_value": f"final status={decision['final_status']}; native history available={not decision['gates']['solver_history_not_available']}", "verified": "yes"},
        ]
    )
    pd.concat([existing, rows], ignore_index=True).to_csv(TRACE_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    quantiles = pd.read_csv(AUDIT_DIR / "velocity_quantiles.csv")
    provenance = pd.read_csv(AUDIT_DIR / "peak_provenance.csv")
    health = pd.read_csv(AUDIT_DIR / "solver_health_gate_audit.csv")
    decision = json.loads((AUDIT_DIR / "velocity_extreme_decision.json").read_text(encoding="utf-8"))
    overlap_audit = pd.read_csv(OVERLAP_DIR / "velocity_distribution_overlap_audit.csv")
    overlap_decision = json.loads((OVERLAP_DIR / "velocity_distribution_overlap_decision.json").read_text(encoding="utf-8"))
    overlap = overlap_audit.loc[
        (overlap_audit["audit_context"] == "aggregation_sensitivity")
        & np.isclose(overlap_audit["time_s"], 0.70)
        & (overlap_audit["aggregation_strategy"] == "mean_all_records")
    ]
    if decision["final_status"] != "audit_only":
        raise ValueError("Fig. S10 must be updated if the native solver-history decision changes.")
    if len(overlap) != 1 or overlap_decision["whole_pool_distribution_separation"] != "not_supported_by_central_distribution":
        raise ValueError("Fig. S10 requires the frozen Comment 17 overlap decision.")
    overlap_row = overlap.iloc[0]

    figure, axes = plt.subplots(2, 2, figsize=(DOUBLE_WIDTH, 122 * MM))
    _quantile_panel(axes[0, 0], quantiles)
    _provenance_panel(axes[0, 1], quantiles, provenance, overlap_row)
    _temporal_panel(axes[1, 0], quantiles)
    _health_panel(axes[1, 1], health)
    for axis, label in zip(axes.ravel(), "abcd"):
        _panel_label(axis, label)
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.095, top=0.95, hspace=0.42, wspace=0.34)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in figure_extensions():
        options: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            options["dpi"] = 450
        elif extension == "tiff":
            options["dpi"] = 600
            options["pil_kwargs"] = {"compression": "tiff_lzw"}
        figure.savefig(FIGURE_DIR / f"{STEM}.{extension}", **options)
    plt.close(figure)
    _update_traceability(quantiles, provenance, decision, overlap_row)
    print(f"Generated {STEM} in PDF/SVG/PNG/TIFF")


if __name__ == "__main__":
    main()
