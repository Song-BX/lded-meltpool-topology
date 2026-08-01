"""Build Fig. S1 from the frozen Comment 17 velocity-distribution audit.

Figure contract
---------------
Core conclusion: at 0.70 s, the 350 W and 400 W full-pool central velocity
ranges overlap, with the 400 W IQR contained within the 350 W IQR.  The Vmax
contrast is therefore a sparse peak-level audit record, not evidence of a
whole-pool structural signal.
Archetype: quantitative grid.  Panel a supplies the six-power velocity context
and the 350/400 W IQR inset. Panel b preserves the six-power temperature
background. No panel conveys replicate uncertainty, inferential separation,
continuous-power interpolation, or a physical mechanism.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="velocity_distribution_fig_mpl_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .export_policy import figure_extensions
except ImportError:  # Direct script execution.
    from export_policy import figure_extensions


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "图"
AUDIT_DIR = DATA_DIR / "velocity_distribution_overlap_audit"
FIGURE_DIR = ROOT / "latex_restructure" / "figures"
TRACE_PATH = DATA_DIR / "figure_traceability.csv"
STEM = "FigS1_distribution_checks"
MM = 1 / 25.4
DOUBLE_WIDTH = 183 * MM

BLUE = "#0F4D92"
ORANGE = "#D46A1F"
DARK = "#1A1A1A"
GRAY = "#5C5C5C"
LIGHT = "#DDECF7"
OVERLAP = "#D8E9E2"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "axes.linewidth": 0.5,
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


def _boxplot(axis: plt.Axes, data: list[pd.Series], powers: list[int], ylabel: str, title: str) -> None:
    plot = axis.boxplot(data, tick_labels=powers, showfliers=False, patch_artist=True, widths=0.55)
    for power, patch in zip(powers, plot["boxes"]):
        patch.set_facecolor(BLUE if power == 350 else (ORANGE if power == 400 else LIGHT))
        patch.set_alpha(0.72 if power in (350, 400) else 1.0)
        patch.set_edgecolor(DARK)
        patch.set_linewidth(0.5)
    for element in ("whiskers", "caps", "medians"):
        for artist in plot[element]:
            artist.set_color(DARK)
            artist.set_linewidth(0.5)
    for index, values in enumerate(data, start=1):
        axis.text(index, axis.get_ylim()[1], f"n={len(values)}", ha="center", va="top", fontsize=5.0, color=GRAY)
    axis.set_xlabel("Laser power (W)")
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", pad=3)
    _clean_axis(axis)


def _velocity_iqr_inset(axis: plt.Axes, canonical: pd.Series) -> None:
    inset = axis.inset_axes([0.42, 0.58, 0.53, 0.30])
    lower = float(canonical.iqr_overlap_lower_mps)
    upper = float(canonical.iqr_overlap_upper_mps)
    inset.axvspan(lower, upper, color=OVERLAP, zorder=0)
    values = (
        (350, BLUE, float(canonical.p25_350_mps), float(canonical.p50_350_mps), float(canonical.p75_350_mps)),
        (400, ORANGE, float(canonical.p25_400_mps), float(canonical.p50_400_mps), float(canonical.p75_400_mps)),
    )
    for power, color, q25, q50, q75 in values:
        inset.hlines(power, q25, q75, color=color, linewidth=2.2, zorder=2)
        inset.scatter(q50, power, color="white", edgecolor=color, linewidth=0.9, s=16, zorder=3)
    inset.set_xlim(0.0, 0.060)
    inset.set_ylim(430, 320)
    inset.set_yticks([350, 400])
    inset.set_yticklabels(["350 W", "400 W"])
    inset.set_xlabel(r"IQR velocity (m s$^{-1}$)", fontsize=4.9, labelpad=1)
    inset.tick_params(labelsize=4.7, length=2, pad=1)
    inset.set_title("400 W IQR within 350 W IQR", fontsize=4.9, loc="left", pad=1.5)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)


def _update_traceability(canonical: pd.Series) -> None:
    existing = pd.read_csv(TRACE_PATH) if TRACE_PATH.exists() else pd.DataFrame()
    if len(existing):
        existing = existing.loc[existing["figure_id"] != "Fig. S1"]
    rows = pd.DataFrame(
        [
            {
                "figure_id": "Fig. S1",
                "panel_id": "a",
                "source_csv": "图/s1/FigS1_distributions_longtable.csv; 图/velocity_distribution_overlap_audit/velocity_distribution_overlap_audit.csv",
                "metric_name": "full-pool velocity distribution and central IQR relation",
                "reported_value": f"350 W IQR={canonical.p25_350_mps:.6f}--{canonical.p75_350_mps:.6f}; 400 W IQR={canonical.p25_400_mps:.6f}--{canonical.p75_400_mps:.6f}; {canonical.contained_iqr}",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S1",
                "panel_id": "b",
                "source_csv": "图/s1/FigS1_distributions_longtable.csv",
                "metric_name": "full-pool temperature distribution",
                "reported_value": "six discrete powers; box plots are within-cloud summaries, not replicate uncertainty",
                "verified": "yes",
            },
        ]
    )
    pd.concat([existing, rows], ignore_index=True).to_csv(TRACE_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    distributions = pd.read_csv(DATA_DIR / "s1" / "FigS1_distributions_longtable.csv")
    audit = pd.read_csv(AUDIT_DIR / "velocity_distribution_overlap_audit.csv")
    decision = json.loads((AUDIT_DIR / "velocity_distribution_overlap_decision.json").read_text(encoding="utf-8"))
    canonical = audit.loc[
        (audit["audit_context"] == "aggregation_sensitivity")
        & np.isclose(audit["time_s"], 0.70)
        & (audit["aggregation_strategy"] == "mean_all_records")
    ]
    if len(canonical) != 1 or decision["whole_pool_distribution_separation"] != "not_supported_by_central_distribution":
        raise ValueError("Fig. S1 requires the frozen Comment 17 overlap decision.")
    canonical_row = canonical.iloc[0]
    all_pool = distributions.loc[distributions["region"] == "all"].copy()
    powers = sorted(all_pool["power_W"].astype(int).unique())

    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_WIDTH, 72 * MM))
    velocity_data = [all_pool.loc[all_pool["power_W"].astype(int) == power, "Vmag"].dropna() for power in powers]
    temperature_data = [all_pool.loc[all_pool["power_W"].astype(int) == power, "Temperature_K"].dropna() for power in powers]
    _boxplot(axes[0], velocity_data, powers, r"Velocity magnitude (m s$^{-1}$)", "Full-pool velocity distribution")
    _velocity_iqr_inset(axes[0], canonical_row)
    axes[0].text(
        0.02,
        0.04,
        "Boxplots show within-cloud distributional context.\n"
        "Extrema are audited separately in Fig. S10.",
        transform=axes[0].transAxes,
        fontsize=4.9,
        color=GRAY,
        va="bottom",
    )
    _boxplot(axes[1], temperature_data, powers, "Temperature (K)", "Full-pool temperature distribution")
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    figure.text(
        0.5,
        0.01,
        "Boxes and whiskers summarise one exported point cloud per discrete simulation, not replicate uncertainty. No inferential distributional test is shown.",
        ha="center",
        va="bottom",
        fontsize=5.0,
        color=GRAY,
    )
    figure.tight_layout(rect=[0, 0.05, 1, 1], w_pad=1.8)

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
    _update_traceability(canonical_row)
    print(f"Generated {STEM} in PDF/SVG/PNG/TIFF")


if __name__ == "__main__":
    main()
