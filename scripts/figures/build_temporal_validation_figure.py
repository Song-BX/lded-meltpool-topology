from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="temporal_fig_mpl_"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .export_policy import figure_extensions
except ImportError:  # Direct figure-script execution.
    from export_policy import figure_extensions
from matplotlib.colors import LinearSegmentedColormap, Normalize


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "图" / "s4"
FIGURE_DIR = ROOT / "latex_restructure" / "figures"
TRACEABILITY_PATH = ROOT / "图" / "figure_traceability.csv"
FIGURE_STEM = "FigS4_temporal_stability_assessment"

MM = 1 / 25.4
DOUBLE_COLUMN_WIDTH = 183 * MM
COLORS = {
    200: "#666666",
    250: "#56B4E9",
    300: "#009E73",
    350: "#0072B2",
    400: "#D55E00",
    450: "#CC79A7",
}
KEY_HEATMAP_METRICS = (
    "unique_points",
    "span_x_m",
    "span_y_m",
    "span_z_m",
    "temperature_mean_all_K",
    "temperature_max_all_K",
    "velocity_max_all_mps",
    "velocity_mean_interface_mps",
    "wls_valid_fraction",
    "q_positive_fraction_all",
    "q_positive_fraction_interface",
)
HEATMAP_LABELS = (
    "Support",
    "x span",
    "y span",
    "z span",
    r"$T_{mean}$",
    r"$T_{max}$",
    r"$V_{max}$",
    r"$V_{int}$",
    "WLS valid",
    r"$\phi_Q$, all",
    r"$\phi_Q$, int.",
)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "legend.fontsize": 5.4,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "savefig.dpi": 450,
    }
)


def panel_label(ax: plt.Axes, label: str, x: float = -0.14) -> None:
    ax.text(
        x,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.5)
    ax.axvspan(0.60, 0.70, color="#F0F0F0", zorder=-10)
    ax.set_xlim(0.49, 0.71)
    ax.set_xticks([0.50, 0.55, 0.60, 0.65, 0.70])
    ax.set_xlabel("Simulation time (s)")


def plot_series(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    metric: str,
    ylabel: str,
    normalize_to_final: bool,
) -> None:
    for power, group in metrics.groupby("power_W", sort=True):
        group = group.sort_values("time_s")
        values = group[metric].to_numpy(dtype=float)
        if normalize_to_final:
            values = values / values[-1]
        central = int(power) in (350, 400)
        ax.plot(
            group["time_s"],
            values,
            color=COLORS[int(power)],
            lw=1.55 if central else 0.9,
            marker="o",
            ms=3.2 if central else 2.5,
            label=f"{int(power)} W",
            zorder=4 if central else 2,
        )
    if normalize_to_final:
        ax.axhline(1.0, color="#888888", lw=0.6, ls="--", zorder=0)
    ax.set_ylabel(ylabel)
    clean_axis(ax)


def plot_stability_heatmap(ax: plt.Axes, stability: pd.DataFrame) -> None:
    matrix = np.full((6, len(KEY_HEATMAP_METRICS)), np.nan)
    powers = sorted(stability["power_W"].astype(int).unique())
    for row_index, power in enumerate(powers):
        indexed = stability[stability["power_W"] == power].set_index("metric")
        for column_index, metric in enumerate(KEY_HEATMAP_METRICS):
            matrix[row_index, column_index] = float(indexed.loc[metric, "stability_ratio"])

    cmap = LinearSegmentedColormap.from_list(
        "stability",
        ["#2E9E44", "#F7F7F7", "#E69F00", "#B64342"],
        N=256,
    )
    shown = np.clip(matrix, 0.0, 2.5)
    image = ax.imshow(shown, cmap=cmap, norm=Normalize(0, 2.5), aspect="auto")
    ax.set_xticks(range(len(HEATMAP_LABELS)))
    ax.set_xticklabels(HEATMAP_LABELS, rotation=32, ha="right")
    ax.set_yticks(range(len(powers)))
    ax.set_yticklabels([f"{power} W" for power in powers])
    ax.tick_params(length=0, pad=1.5)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text_color = "white" if value >= 1.65 else "black"
            ax.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=4.8,
                color=text_color,
            )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.032, pad=0.025)
    colorbar.set_label("Observed / predefined limit", labelpad=3)
    colorbar.set_ticks([0, 1, 2, 2.5])
    colorbar.set_ticklabels(["0", "1 (limit)", "2", r"$\geq$2.5"])
    for spine in ax.spines.values():
        spine.set_visible(False)


def update_traceability(decision: dict[str, object]) -> None:
    if TRACEABILITY_PATH.exists():
        trace = pd.read_csv(TRACEABILITY_PATH)
        trace = trace[trace["figure_id"] != "Fig. S4"]
    else:
        trace = pd.DataFrame(
            columns=[
                "figure_id",
                "panel_id",
                "source_csv",
                "metric_name",
                "reported_value",
                "verified",
            ]
        )
    rows = pd.DataFrame(
        [
            {
                "figure_id": "Fig. S4",
                "panel_id": "a",
                "source_csv": "图/s4/temporal_metrics.csv",
                "metric_name": "maximum temperature time series",
                "reported_value": "six powers; 0.50-0.70 s",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S4",
                "panel_id": "b",
                "source_csv": "图/s4/temporal_metrics.csv",
                "metric_name": "maximum velocity time series",
                "reported_value": "six powers; 0.50-0.70 s",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S4",
                "panel_id": "c",
                "source_csv": "图/s4/temporal_metrics.csv",
                "metric_name": "full-pool positive-Q fraction time series",
                "reported_value": "six powers; k=25",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S4",
                "panel_id": "d",
                "source_csv": "图/s4/temporal_stability_summary.csv",
                "metric_name": "predefined stability-limit ratio",
                "reported_value": str(decision["status"]),
                "verified": "yes",
            },
        ]
    )
    pd.concat([trace, rows], ignore_index=True).to_csv(
        TRACEABILITY_PATH, index=False, encoding="utf-8-sig"
    )


def main() -> None:
    metrics = pd.read_csv(DATA_DIR / "temporal_metrics.csv")
    stability = pd.read_csv(DATA_DIR / "temporal_stability_summary.csv")
    decision = json.loads((DATA_DIR / "temporal_validation_decision.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 132 * MM))
    grid = fig.add_gridspec(2, 3, hspace=0.58, wspace=0.42, height_ratios=[1.0, 0.92])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, :])

    plot_series(
        ax_a,
        metrics,
        "temperature_max_all_K",
        r"Normalized $T_{max}$",
        normalize_to_final=True,
    )
    plot_series(
        ax_b,
        metrics,
        "velocity_max_all_mps",
        r"Normalized $V_{max}$",
        normalize_to_final=True,
    )
    plot_series(
        ax_c,
        metrics,
        "q_positive_fraction_all",
        "Positive-Q fraction",
        normalize_to_final=False,
    )
    plot_stability_heatmap(ax_d, stability)

    ax_a.set_title("Thermal evolution", loc="left", pad=3)
    ax_b.set_title("Velocity evolution", loc="left", pad=3)
    ax_c.set_title("Topology-proxy evolution", loc="left", pad=3)
    ax_d.set_title("Late-window stability assessment", loc="left", pad=3)
    for axis, label in zip((ax_a, ax_b, ax_c), "abc"):
        panel_label(axis, label)
    panel_label(ax_d, "d", x=-0.045)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.005),
        ncol=6,
        columnspacing=1.1,
        handlelength=1.8,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.89, bottom=0.10, left=0.08, right=0.95)
    for extension in figure_extensions():
        path = FIGURE_DIR / f"{FIGURE_STEM}.{extension}"
        kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = 450
        elif extension == "tiff":
            kwargs["dpi"] = 600
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
    plt.close(fig)
    update_traceability(decision)
    print(f"Generated {FIGURE_STEM} in four formats; decision={decision['status']}")


if __name__ == "__main__":
    main()
