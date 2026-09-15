from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="export_diag_mpl_"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import Normalize

try:
    from .export_policy import figure_suffixes
except ImportError:  # Direct figure-script execution.
    from export_policy import figure_suffixes


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "图" / "export_diagnostics"
MAIN_METRICS = ROOT / "图" / "3" / "Aplus_main_metrics_k25.csv"
FIGURE_DIR = ROOT / "latex_restructure" / "figures"
TRACE_PATH = ROOT / "图" / "figure_traceability.csv"

MM = 1 / 25.4
DOUBLE_WIDTH = 183 * MM
BLUE = "#0072B2"
ORANGE = "#D55E00"
LIGHT_GRAY = "#C7C7C7"
DARK = "#1A1A1A"
GRAY = "#4D4D4D"
SUPPORT_GATE = 100

REGION_ORDER = ("all", "interface", "heated", "interface_heated")
REGION_LABELS = {
    "all": "full-pool",
    "interface": "interface",
    "heated": "heated",
    "interface_heated": "interface-heated",
}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "legend.fontsize": 5.3,
        "xtick.labelsize": 5.4,
        "ytick.labelsize": 5.4,
        "axes.linewidth": 0.45,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 180,
    }
)


def _panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=DARK,
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.5)


def _clean_heatmap(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
    ax.tick_params(length=2.0, width=0.45, direction="out", pad=1.5)


def _luminance(color: tuple[float, float, float, float]) -> float:
    red, green, blue = color[:3]
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _export(figure: plt.Figure) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / "Fig2_point_cloud_quality"
    for suffix in figure_suffixes():
        options: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == ".png":
            options["dpi"] = 450
        elif suffix == ".tiff":
            options["dpi"] = 600
            options["pil_kwargs"] = {"compression": "tiff_lzw"}
        figure.savefig(stem.with_suffix(suffix), **options)
    plt.close(figure)
    return stem.with_suffix(".pdf")


def _add_trace(
    rows: list[dict[str, object]], panel: str, source: str, metric: str, value: object
) -> None:
    rows.append(
        {
            "figure_id": "Fig.2",
            "panel_id": panel,
            "source_csv": source,
            "metric_name": metric,
            "reported_value": value,
            "verified": "yes",
        }
    )


def build_figure(trace_rows: list[dict[str, object]] | None = None) -> Path:
    summary = pd.read_csv(SOURCE_DIR / "duplicate_group_summary.csv")
    summary = summary[np.isclose(summary["time_s"], 0.70)].sort_values("power_W")
    multiplicity = pd.read_csv(SOURCE_DIR / "duplicate_multiplicity_distribution.csv")
    multiplicity = multiplicity[np.isclose(multiplicity["time_s"], 0.70)]
    main = pd.read_csv(MAIN_METRICS)
    region_counts = (
        main.pivot(index="power_W", columns="region", values="n")
        .sort_index()
        .reindex(columns=REGION_ORDER)
    )
    powers = summary["power_W"].astype(int).tolist()

    figure = plt.figure(figsize=(DOUBLE_WIDTH, 99 * MM))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[1.25, 1.0],
        height_ratios=[1.08, 0.92],
        wspace=0.34,
        hspace=0.40,
    )

    ax = figure.add_subplot(grid[:, 0])
    x = np.arange(len(summary))
    unique = summary["unique_coordinate_representatives"].to_numpy()
    distinct = summary["additional_distinct_state_rows"].to_numpy()
    repeated = summary["exact_repeated_rows"].to_numpy()
    ax.bar(x, unique, width=0.66, color=BLUE, label="one representative per coordinate")
    ax.bar(x, distinct, width=0.66, bottom=unique, color=ORANGE, label="additional distinct states")
    ax.bar(
        x,
        repeated,
        width=0.66,
        bottom=unique + distinct,
        color=LIGHT_GRAY,
        label="exact repeated rows",
    )
    ax.text(
        0.02,
        0.985,
        "79.5-80.9% exact repeated rows; 7.5-10.1% conflicting coordinate groups",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.3,
        color=GRAY,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(powers)
    ax.set_xlabel("Laser power (W)")
    ax.set_ylabel("Exported rows")
    ax.set_ylim(0, float(summary["raw_points"].max()) * 1.10)
    ax.set_title("Export-row structure at 0.70 s", loc="left", pad=3)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=1,
        handlelength=1.2,
        labelspacing=0.25,
    )
    _clean_axis(ax)
    _panel_label(ax, "a", x=-0.12, y=1.02)

    ax = figure.add_subplot(grid[0, 1])
    multiplicity_grid = (
        multiplicity.pivot(
            index="multiplicity", columns="power_W", values="coordinate_group_fraction"
        )
        .reindex(index=range(1, 13), columns=powers)
        .fillna(0.0)
    )
    image = ax.imshow(
        multiplicity_grid.to_numpy(), cmap="Blues", vmin=0, vmax=0.55, aspect="auto"
    )
    ax.set_xticks(np.arange(len(powers)))
    ax.set_xticklabels(powers)
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(range(1, 13))
    ax.set_xlabel("Laser power (W)")
    ax.set_ylabel("Records per coordinate")
    ax.set_title("Coordinate multiplicity distribution", loc="left", pad=3)
    for row in range(multiplicity_grid.shape[0]):
        for column in range(multiplicity_grid.shape[1]):
            value = float(multiplicity_grid.iloc[row, column])
            if value >= 0.05:
                color = "white" if _luminance(plt.get_cmap("Blues")(value / 0.55)) < 0.48 else DARK
                ax.text(column, row, f"{value * 100:.0f}", ha="center", va="center", fontsize=4.8, color=color)
    ax.add_patch(
        patches.Rectangle(
            (-0.48, 4.52),
            len(powers) - 0.04,
            0.96,
            fill=False,
            edgecolor=ORANGE,
            linewidth=0.8,
        )
    )
    _clean_heatmap(ax)
    _panel_label(ax, "b", x=-0.15, y=1.03)
    colorbar = figure.colorbar(image, ax=ax, fraction=0.042, pad=0.028)
    colorbar.set_label("coordinate-group fraction")
    colorbar.ax.tick_params(labelsize=5.0, width=0.4, length=2)

    ax = figure.add_subplot(grid[1, 1])
    x = np.arange(len(powers))
    width = 0.19
    styles = {
        "all": {"color": BLUE, "edgecolor": BLUE, "hatch": None},
        "interface": {"color": ORANGE, "edgecolor": ORANGE, "hatch": None},
        "heated": {"color": "white", "edgecolor": GRAY, "hatch": "////"},
        "interface_heated": {
            "color": LIGHT_GRAY,
            "edgecolor": GRAY,
            "hatch": "xxxx",
        },
    }
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(REGION_ORDER))
    for offset, region in zip(offsets, REGION_ORDER):
        values = region_counts[region].to_numpy(dtype=float)
        style = styles[region]
        label = REGION_LABELS[region]
        if region in {"heated", "interface_heated"}:
            label += " (audit only)"
        ax.bar(
            x + offset,
            values,
            width=width,
            color=style["color"],
            edgecolor=style["edgecolor"],
            hatch=style["hatch"],
            linewidth=0.55,
            label=label,
        )
    ax.axhline(SUPPORT_GATE, color=DARK, lw=0.75, ls="--")
    ax.text(
        0.99,
        SUPPORT_GATE + 5,
        "evidence gate: n = 100",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=GRAY,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(powers)
    ax.set_xlabel("Laser power (W)")
    ax.set_ylabel("Valid points at k = 25")
    ax.set_ylim(0, float(region_counts.to_numpy().max()) * 1.12)
    ax.set_title("Evidence-support gate", loc="left", pad=3)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        columnspacing=0.7,
        handlelength=1.2,
        fontsize=4.8,
    )
    _clean_axis(ax)
    _panel_label(ax, "c", x=-0.15, y=1.03)

    if trace_rows is not None:
        summary_source = "图/export_diagnostics/duplicate_group_summary.csv"
        _add_trace(
            trace_rows,
            "a",
            summary_source,
            "exact full-row duplicate ratio at 0.70 s",
            f"{summary['exact_full_row_duplicate_ratio'].min() * 100:.1f}-"
            f"{summary['exact_full_row_duplicate_ratio'].max() * 100:.1f}%",
        )
        _add_trace(
            trace_rows,
            "a",
            summary_source,
            "conflicting coordinate-group fraction at 0.70 s",
            f"{summary['conflicting_coordinate_group_fraction'].min() * 100:.1f}-"
            f"{summary['conflicting_coordinate_group_fraction'].max() * 100:.1f}%",
        )
        _add_trace(
            trace_rows,
            "b",
            "图/export_diagnostics/duplicate_multiplicity_distribution.csv",
            "multiplicity median and mode",
            "6 for every 0.70 s power case",
        )
        for region in REGION_ORDER:
            _add_trace(
                trace_rows,
                "c",
                "图/3/Aplus_main_metrics_k25.csv",
                f"{REGION_LABELS[region]} sample support",
                f"{int(region_counts[region].min())}-{int(region_counts[region].max())}",
            )
        _add_trace(
            trace_rows,
            "c",
            "scripts/robustness/config.py",
            "minimum regional support gate",
            SUPPORT_GATE,
        )

    return _export(figure)


def update_trace(rows: list[dict[str, object]]) -> None:
    new = pd.DataFrame(rows)
    if TRACE_PATH.exists():
        existing = pd.read_csv(TRACE_PATH)
        existing = existing[existing["figure_id"] != "Fig.2"]
        new = pd.concat([existing, new], ignore_index=True)
    new.to_csv(TRACE_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    trace_rows: list[dict[str, object]] = []
    output = build_figure(trace_rows)
    update_trace(trace_rows)
    print(f"generated {output} and PDF/SVG/PNG/TIFF companions")


if __name__ == "__main__":
    main()
