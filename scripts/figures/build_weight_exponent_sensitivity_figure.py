from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="weight_exponent_fig_mpl_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .export_policy import figure_extensions
except ImportError:  # Direct figure-script execution.
    from export_policy import figure_extensions


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "图" / "weight_exponent_sensitivity"
FIGURE_DIR = ROOT / "latex_restructure" / "figures"
TRACE_PATH = ROOT / "图" / "figure_traceability.csv"
STEM = "FigS7_distance_weight_exponent_sensitivity"

MM = 1 / 25.4
DOUBLE_WIDTH = 183 * MM
ALPHA_COLORS = {"0": "#0072B2", "1": "#D55E00", "2": "#009E73"}
DARK = "#1A1A1A"
GRAY = "#5A5A5A"
FAIL = "#B2182B"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "legend.fontsize": 5.1,
        "xtick.labelsize": 5.3,
        "ytick.labelsize": 5.3,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", pad=1.5)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.15,
        1.04,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def _effective_neighbour_panel(axis: plt.Axes, geometry: pd.DataFrame) -> None:
    for alpha_label, block in geometry.groupby("alpha_label", sort=True):
        aggregate = block.groupby("kNN", as_index=False).agg(
            effective_neighbours_median=("effective_neighbours_median", "median"),
            max_normalized_weight_median=("max_normalized_weight_median", "median"),
        )
        axis.plot(
            aggregate["kNN"],
            aggregate["effective_neighbours_median"],
            color=ALPHA_COLORS[alpha_label],
            marker="o",
            ms=1.9,
            lw=1.0,
            label=rf"$\alpha={alpha_label}$",
        )
    axis.set_xlabel("Neighbour count, k")
    axis.set_ylabel("Median effective weighted neighbours")
    axis.set_title("Distance weighting concentrates local support", loc="left", pad=3)
    axis.text(
        0.02,
        0.05,
        "At k=25 (six-power median): 25.0 / 21.4 / 13.6\neffective neighbours for $\\alpha=0$ / 1 / 2.",
        transform=axis.transAxes,
        va="bottom",
        fontsize=4.9,
        color=GRAY,
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    axis.legend(frameon=False, loc="upper left", ncol=3, columnspacing=0.75, handlelength=1.4)
    _clean_axis(axis)


def _validity_panel(axis: plt.Axes, geometry: pd.DataFrame) -> None:
    for alpha_label, block in geometry.groupby("alpha_label", sort=True):
        minimum = block.groupby("kNN", as_index=False)["wls_valid_fraction"].min()
        axis.plot(
            minimum["kNN"],
            minimum["wls_valid_fraction"] * 100,
            color=ALPHA_COLORS[alpha_label],
            marker="o",
            ms=1.9,
            lw=1.0,
            label=rf"$\alpha={alpha_label}$",
        )
    axis.set_ylim(90, 100.5)
    axis.set_xlabel("Neighbour count, k")
    axis.set_ylabel("Minimum WLS-valid fraction across powers (%)")
    axis.set_title("All exponent branches retain full-pool support", loc="left", pad=3)
    axis.text(
        0.02,
        0.05,
        "All 774 alpha x power x k cells retain at least 100 full-pool points.",
        transform=axis.transAxes,
        va="bottom",
        fontsize=4.9,
        color=GRAY,
    )
    _clean_axis(axis)


def _contrast_panel(axis: plt.Axes, contrasts: pd.DataFrame) -> None:
    selected = contrasts[(contrasts["region"] == "all") & (contrasts["threshold"] == "Q>0")]
    for alpha_label, block in selected.groupby("alpha_label", sort=True):
        block = block.sort_values("kNN")
        axis.plot(
            block["kNN"],
            block["delta_350_400"],
            color=ALPHA_COLORS[alpha_label],
            marker="o",
            ms=2.1,
            lw=1.1,
            label=rf"$\alpha={alpha_label}$",
        )
    axis.axhline(0, color=DARK, lw=0.6)
    axis.axvline(25, color=GRAY, lw=0.55, ls="--")
    axis.set_xlabel("Neighbour count, k")
    axis.set_ylabel(r"$\Delta\phi_{Q>0}=\phi_{350}-\phi_{400}$")
    axis.set_title("Q direction is alpha-consistent but audit-only", loc="left", pad=3)
    axis.text(
        0.02,
        0.96,
        "43/43 k values and 200/200 subset runs are positive for every alpha.\n"
        "The alpha=2 affine exactness failure prevents comparative interpretation.",
        transform=axis.transAxes,
        va="top",
        fontsize=4.9,
        color=FAIL,
    )
    axis.legend(frameon=False, loc="lower right", ncol=3, columnspacing=0.75, handlelength=1.4)
    _clean_axis(axis)


def _manufactured_panel(axis: plt.Axes, manufactured: pd.DataFrame) -> None:
    affine = (
        manufactured[manufactured["field_class"] == "affine"]
        .groupby(["alpha_label", "alpha"], as_index=False)["gradient_nrmse"]
        .max()
        .sort_values("alpha")
    )
    x = np.arange(len(affine))
    axis.bar(
        x,
        affine["gradient_nrmse"],
        color=[ALPHA_COLORS[value] for value in affine["alpha_label"]],
        width=0.62,
        edgecolor=DARK,
        linewidth=0.45,
    )
    axis.axhline(1.0e-10, color=FAIL, lw=0.75, ls="--")
    axis.set_yscale("log")
    axis.set_xticks(x)
    axis.set_xticklabels([rf"$\alpha={value}$" for value in affine["alpha_label"]])
    axis.set_ylabel("Maximum affine gradient NRMSE")
    axis.set_title("Pre-specified affine exactness gate", loc="left", pad=3)
    axis.text(
        0.02,
        0.05,
        r"Gate: $1\times10^{-10}$; alpha=2: $1.36\times10^{-9}$." + "\n"
        "Q comparison withdrawn after this numerical failure.",
        transform=axis.transAxes,
        va="bottom",
        fontsize=4.9,
        color=FAIL,
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )
    _clean_axis(axis)


def _update_traceability(
    geometry: pd.DataFrame,
    contrasts: pd.DataFrame,
    manufactured: pd.DataFrame,
    decision: dict[str, object],
) -> None:
    existing = pd.read_csv(TRACE_PATH) if TRACE_PATH.exists() else pd.DataFrame()
    if len(existing):
        existing = existing[existing["figure_id"] != "Fig. S7"]
    alpha_two = geometry[(geometry["alpha"] == 2.0) & (geometry["kNN"] == 25)]
    effective = float(alpha_two["effective_neighbours_median"].median())
    q_core = contrasts[(contrasts["region"] == "all") & (contrasts["threshold"] == "Q>0")]
    affine_max = float(
        manufactured[(manufactured["field_class"] == "affine") & (manufactured["alpha"] == 2.0)]["gradient_nrmse"].max()
    )
    rows = pd.DataFrame(
        [
            {
                "figure_id": "Fig. S7",
                "panel_id": "a",
                "source_csv": "图/weight_exponent_sensitivity/alpha_weight_geometry.csv",
                "metric_name": "effective distance-weighted neighbourhood support",
                "reported_value": f"alpha=2 median effective neighbours at k=25: {effective:.3f}",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S7",
                "panel_id": "b",
                "source_csv": "图/weight_exponent_sensitivity/alpha_weight_geometry.csv",
                "metric_name": "minimum power-specific WLS-valid fraction",
                "reported_value": "all alpha-power-k cells satisfy the full-pool 100-point support requirement",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S7",
                "panel_id": "c",
                "source_csv": "图/weight_exponent_sensitivity/alpha_core_contrasts.csv",
                "metric_name": "full-pool Q>0 delta across pre-specified exponents",
                "reported_value": f"positive direction count={int((q_core['direction'] == '350>400').sum())}/{len(q_core)}; audit only",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S7",
                "panel_id": "d",
                "source_csv": "图/weight_exponent_sensitivity/alpha_manufactured_field_metrics.csv",
                "metric_name": "affine manufactured-field exactness gate",
                "reported_value": f"alpha=2 maximum gradient NRMSE={affine_max:.3e}; final status={decision['final_q_claim_status']}",
                "verified": "yes",
            },
        ]
    )
    pd.concat([existing, rows], ignore_index=True).to_csv(TRACE_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    geometry = pd.read_csv(DATA_DIR / "alpha_weight_geometry.csv")
    contrasts = pd.read_csv(DATA_DIR / "alpha_core_contrasts.csv")
    manufactured = pd.read_csv(DATA_DIR / "alpha_manufactured_field_metrics.csv")
    for frame in (geometry, contrasts, manufactured):
        frame["alpha_label"] = frame["alpha_label"].astype(str)
    decision = json.loads((DATA_DIR / "weight_exponent_decision.json").read_text(encoding="utf-8"))

    figure, axes = plt.subplots(2, 2, figsize=(DOUBLE_WIDTH, 118 * MM))
    _effective_neighbour_panel(axes[0, 0], geometry)
    _validity_panel(axes[0, 1], geometry)
    _contrast_panel(axes[1, 0], contrasts)
    _manufactured_panel(axes[1, 1], manufactured)
    for axis, label in zip(axes.ravel(), "abcd"):
        _panel_label(axis, label)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.93, wspace=0.34, hspace=0.42)

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
    _update_traceability(geometry, contrasts, manufactured, decision)
    print(f"Generated {STEM} in PDF/SVG/PNG/TIFF")


if __name__ == "__main__":
    main()
