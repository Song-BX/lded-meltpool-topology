from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="conditioning_fig_mpl_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from scripts.conditioning_sensitivity.config import CUTOFF_SPECS
from scripts.figures.export_policy import figure_extensions


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "图" / "conditioning_sensitivity"
FIGURE_DIR = ROOT / "latex_restructure" / "figures"
TRACE_PATH = ROOT / "图" / "figure_traceability.csv"
STEM = "FigS6_conditioning_cutoff_sensitivity"

MM = 1 / 25.4
DOUBLE_WIDTH = 183 * MM
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
DARK = "#1A1A1A"
GRAY = "#666666"
LIGHT = "#E8E8E8"

CUTOFF_LABELS = [spec.label for spec in CUTOFF_SPECS]
CUTOFF_COLORS = {
    "10": "#6A6A6A",
    "30": "#56B4E9",
    "100": BLUE,
    "300": "#009E73",
    "1e3": "#CC79A7",
    "1e6": "#E69F00",
    "1e12": ORANGE,
    "inf": "#303030",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "legend.fontsize": 4.9,
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


def _at_cutoff(frame: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    values = frame["cutoff_value"].to_numpy(dtype=float)
    matches = np.isinf(values) if np.isinf(cutoff) else np.isclose(values, cutoff)
    return frame.loc[matches]


def _condition_distribution_panel(axis: plt.Axes, distribution: pd.DataFrame) -> None:
    columns = [
        "kappa_lt_10",
        "kappa_10_to_30",
        "kappa_30_to_100",
        "kappa_100_to_300",
        "kappa_300_to_1e3",
        "kappa_1e3_to_1e6",
        "kappa_1e6_to_1e12",
        "kappa_ge_1e12",
    ]
    labels = ["<10", "10–30", "30–100", "100–300", "300–1e3", "1e3–1e6", "1e6–1e12", "≥1e12"]
    colors = ["#C9DDEE", "#8DBBD8", "#4E99C7", BLUE, GREEN, "#E7B76B", ORANGE, "#A23B2A"]
    total = float(distribution["total_points"].sum())
    values = [float(distribution[column].sum()) / total * 100 for column in columns]
    axis.bar(np.arange(len(values)), values, color=colors, width=0.78)
    axis.set_xticks(np.arange(len(values)))
    axis.set_xticklabels(labels, rotation=31, ha="right")
    axis.set_ylabel("Reconstruction events (%)")
    axis.set_title("Condition-number distribution before screening", loc="left", pad=3)
    axis.text(
        0.01,
        0.96,
        "Binned regimes include current κ=100 and legacy κ=1e12",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=4.9,
        color=GRAY,
    )
    _clean_axis(axis)


def _retention_panel(axis: plt.Axes, point_audit: pd.DataFrame) -> None:
    current = _at_cutoff(point_audit, 100.0)
    matrix = current.pivot(index="power_W", columns="kNN", values="retained_fraction").sort_index()
    image = axis.imshow(matrix.to_numpy() * 100, cmap="cividis", norm=Normalize(90, 100), aspect="auto")
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels([f"{value} W" for value in matrix.index])
    xticks = [0, 8, 17, 26, 34, 42]
    axis.set_xticks(xticks)
    axis.set_xticklabels(matrix.columns.to_numpy()[xticks])
    axis.set_xlabel("kNN")
    axis.set_title("κ=100 retains most reconstructed points", loc="left", pad=3)
    cbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
    cbar.set_label("retained points (%)")
    cbar.ax.tick_params(labelsize=5.0, width=0.4, length=2)
    _clean_axis(axis)


def _contrast_panel(axis: plt.Axes, core: pd.DataFrame, summary: pd.DataFrame) -> None:
    selected = core[(core["region"] == "all") & (core["threshold"] == "Q>0")]
    for spec in CUTOFF_SPECS:
        label = spec.label
        block = _at_cutoff(selected, spec.value).sort_values("kNN")
        status = str(_at_cutoff(summary, spec.value)["status"].iloc[0])
        supported = status == "support_and_direction_consistent"
        axis.plot(
            block["kNN"],
            block["delta_350_400"],
            color=CUTOFF_COLORS[label],
            lw=1.1 if label == "100" else 0.75,
            ls="-" if supported else "--",
            marker="o" if label == "100" else None,
            ms=2.0,
            label=f"κ={label}",
        )
    axis.axhline(0, color=DARK, lw=0.55)
    axis.set_xlabel("kNN")
    axis.set_ylabel(r"$\Delta\phi_{Q>0}=\phi_{350}-\phi_{400}$")
    axis.set_title("Full-pool Q>0 contrast across cutoff screens", loc="left", pad=3)
    axis.legend(frameon=False, loc="upper right", ncol=2, columnspacing=0.7, handlelength=1.6)
    _clean_axis(axis)


def _decision_panel(axis: plt.Axes, point_audit: pd.DataFrame, summary: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for spec in CUTOFF_SPECS:
        row = _at_cutoff(summary, spec.value).iloc[0]
        audit = _at_cutoff(point_audit, spec.value)
        retained = float(audit["retained_points"].sum() / audit["total_points"].sum() * 100)
        state = (
            "consistent"
            if row["status"] == "support_and_direction_consistent"
            else str(row["status"]).replace("_", " ")
        )
        rows.append(
            [
                f"κ={spec.label}",
                f"{retained:.2f}",
                f"{int(row['fullpool_cells_with_support'])}/258",
                f"{int(row['direction_match_count'])}/43",
                state,
            ]
        )
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["screen", "retained\n(%)", "support", "direction", "status"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.07, 1.0, 0.83],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(4.7)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        cell.set_edgecolor("#B0B0B0")
        if row == 0:
            cell.set_facecolor("#E9EEF2")
            cell.set_text_props(weight="bold")
        elif rows[row - 1][0] == "κ=100":
            cell.set_facecolor("#DCECF7")
    axis.set_title("Pre-specified cutoff decision audit", loc="left", pad=3)
    axis.text(
        0.0,
        0.0,
        "Direction is not an inferential test or a solver-gradient validation.",
        transform=axis.transAxes,
        fontsize=4.8,
        color=GRAY,
        va="bottom",
    )


def _update_traceability(distribution: pd.DataFrame, audit: pd.DataFrame, summary: pd.DataFrame) -> None:
    existing = pd.read_csv(TRACE_PATH) if TRACE_PATH.exists() else pd.DataFrame()
    if len(existing):
        existing = existing[existing["figure_id"] != "Fig. S6"]
    current = _at_cutoff(audit, 100.0)
    retained = float(current["retained_points"].sum() / current["total_points"].sum())
    legacy = _at_cutoff(audit, 1.0e12)
    rows = pd.DataFrame(
        [
            {
                "figure_id": "Fig. S6",
                "panel_id": "a",
                "source_csv": "图/conditioning_sensitivity/condition_distribution.csv",
                "metric_name": "unfiltered condition-number bands",
                "reported_value": f"maximum finite kappa={float(distribution['kappa_max'].max()):.3g}",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S6",
                "panel_id": "b",
                "source_csv": "图/conditioning_sensitivity/cutoff_point_audit.csv",
                "metric_name": "kappa=100 retained-point fraction",
                "reported_value": f"pooled retained fraction={retained:.6f}",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S6",
                "panel_id": "c",
                "source_csv": "图/conditioning_sensitivity/cutoff_core_contrasts.csv",
                "metric_name": "full-pool Q>0 delta across cutoffs and kNN",
                "reported_value": "350 W > 400 W direction evaluated at every cutoff and kNN",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S6",
                "panel_id": "d",
                "source_csv": "图/conditioning_sensitivity/cutoff_summary.csv",
                "metric_name": "support and directional decision",
                "reported_value": f"legacy kappa=1e12 exceeded events={int(legacy['exceeded_points'].sum())}",
                "verified": "yes",
            },
        ]
    )
    pd.concat([existing, rows], ignore_index=True).to_csv(TRACE_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    distribution = pd.read_csv(DATA_DIR / "condition_distribution.csv")
    point_audit = pd.read_csv(DATA_DIR / "cutoff_point_audit.csv")
    core = pd.read_csv(DATA_DIR / "cutoff_core_contrasts.csv")
    summary = pd.read_csv(DATA_DIR / "cutoff_summary.csv")

    figure, axes = plt.subplots(2, 2, figsize=(DOUBLE_WIDTH, 118 * MM))
    _condition_distribution_panel(axes[0, 0], distribution)
    _retention_panel(axes[0, 1], point_audit)
    _contrast_panel(axes[1, 0], core, summary)
    _decision_panel(axes[1, 1], point_audit, summary)
    for axis, label in zip(axes.ravel(), "abcd"):
        _panel_label(axis, label)
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.93, wspace=0.34, hspace=0.42)

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
    _update_traceability(distribution, point_audit, summary)
    print(f"Generated {STEM} in PDF/SVG/PNG/TIFF")


if __name__ == "__main__":
    main()
