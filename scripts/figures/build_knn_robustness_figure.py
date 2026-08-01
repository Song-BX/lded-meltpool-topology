from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="knn_fig_mpl_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

try:
    from .export_policy import figure_extensions
except ImportError:  # Direct figure-script execution.
    from export_policy import figure_extensions


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "图" / "robustness"
FIG_DIR = ROOT / "latex_restructure" / "figures"
TRACE_PATH = ROOT / "图" / "figure_traceability.csv"

MM = 1 / 25.4
DOUBLE_WIDTH = 183 * MM
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GRAY = "#5A5A5A"
DARK = "#1A1A1A"
LIGHT = "#EFEFEF"

REGIONS = ("all", "interface", "heated", "interface_heated")
REGION_LABELS = {
    "all": "full-pool",
    "interface": "interface",
    "heated": "heated",
    "interface_heated": "interface-heated",
}
THRESHOLDS = ("Q>0", "Q>posP50", "Q>posP75", "Q>posP90")
FINAL_REGION = "all"
FINAL_THRESHOLD = "Q>0"
THRESHOLD_LABELS = {
    "Q>0": r"$Q>0$",
    "Q>posP50": r"$Q>P_{50}(Q^+)$",
    "Q>posP75": r"$Q>P_{75}(Q^+)$",
    "Q>posP90": r"$Q>P_{90}(Q^+)$",
}
POWER_COLORS = {
    200: "#4C78A8",
    250: "#72B7B2",
    300: "#54A24B",
    350: "#0072B2",
    400: "#D55E00",
    450: "#B279A2",
}


def _eligible_thresholds(eligibility: pd.DataFrame, region: str) -> list[str]:
    allowed = set(
        eligibility.loc[
            (eligibility["region"] == region)
            & (eligibility["evidence_status"] == "evidence_eligible"),
            "threshold",
        ]
    )
    return [threshold for threshold in THRESHOLDS if threshold in allowed]


def _gradient_qualified_pairs(model_order: pd.DataFrame) -> pd.DataFrame:
    """Return the support-qualified comparisons that also pass order sensitivity."""
    qualified = model_order.loc[
        model_order["evidence_eligible"]
        & (model_order["status"] == "order_consistent_over_compared_k"),
        ["region", "threshold"],
    ].copy()
    expected = {(FINAL_REGION, FINAL_THRESHOLD)}
    observed = set(map(tuple, qualified.to_records(index=False)))
    if observed != expected:
        raise ValueError(
            "The model-order audit scope must be exactly full-pool Q>0; "
            f"observed {sorted(observed)}"
        )
    return qualified


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "legend.fontsize": 5.2,
        "xtick.labelsize": 5.3,
        "ytick.labelsize": 5.3,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 450,
    }
)


def panel_label(axis: plt.Axes, label: str, *, x: float = -0.13, y: float = 1.04) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", pad=1.5)


def export_figure(figure: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in figure_extensions():
        path = FIG_DIR / f"{stem}.{suffix}"
        options: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix in {"png", "tiff"}:
            options["dpi"] = 450
        figure.savefig(path, **options)
        paths.append(path)
    plt.close(figure)
    return paths


def build_fig5(core: pd.DataFrame, qualified: pd.DataFrame) -> list[Path]:
    evidence = core.merge(qualified, on=["region", "threshold"], validate="many_to_one")
    figure, axis = plt.subplots(figsize=(DOUBLE_WIDTH, 46 * MM))
    maximum = float(np.abs(evidence["diff_350_400"]).max())
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    selected_ticks = [8, 15, 25, 35, 50]
    matrix = evidence.pivot(index="threshold", columns="kNN", values="diff_350_400").reindex(
        index=[FINAL_THRESHOLD], columns=range(8, 51)
    )
    image = axis.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
    reference_x = list(matrix.columns).index(25)
    axis.axvline(reference_x, color=DARK, lw=0.8)
    axis.scatter(reference_x, -0.72, marker="v", s=12, color=DARK, clip_on=False)
    axis.set_title("Audit-only full-pool Q contrast", loc="left", pad=3)
    axis.set_xlabel("Neighbour count, k")
    axis.set_ylabel("Reconstructed Q metric")
    axis.set_xticks([list(matrix.columns).index(k) for k in selected_ticks])
    axis.set_xticklabels(selected_ticks)
    axis.set_yticks([0])
    axis.set_yticklabels([r"$Q>0$"])
    axis.tick_params(direction="out", pad=1.5)
    panel_label(axis, "a")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    colorbar.set_label(r"$\Delta\phi=\phi_{350}-\phi_{400}$")
    colorbar.ax.tick_params(width=0.4, length=2)
    figure.subplots_adjust(left=0.10, right=0.89, bottom=0.24, top=0.89)
    return export_figure(figure, "Fig5_threshold_knn_sensitivity")


def build_fig6(summary: pd.DataFrame, qualified: pd.DataFrame) -> list[Path]:
    ordered = summary.merge(qualified, on=["region", "threshold"], validate="one_to_one").reset_index(drop=True)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_WIDTH, 82 * MM),
        gridspec_kw={"width_ratios": [1.0, 1.18]},
    )
    y_positions = np.arange(len(ordered))[::-1]
    threshold_colors = dict(zip(THRESHOLDS, [BLUE, GREEN, ORANGE, PURPLE]))
    bar_colors = [threshold_colors[value] for value in ordered["threshold"]]
    axes[0].barh(y_positions, ordered["positive_count"], color=bar_colors, height=0.62)
    axes[0].axvline(43, color=DARK, lw=0.5)
    axes[0].set_xlim(0, 46)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(["full-pool  $Q>0$"])
    axes[0].set_xlabel("k values with 350 W > 400 W (of 43)")
    axes[0].set_title("Direction count after support and order audits", loc="left", pad=3)
    for y, row in zip(y_positions, ordered.itertuples(index=False)):
        axes[0].text(
            min(int(row.positive_count) + 0.7, 44.2),
            y,
            f"{int(row.positive_count)}/43",
            ha="left",
            va="center",
            fontsize=5.3,
        )
    clean_axis(axes[0])
    panel_label(axes[0], "a")

    for y, row in zip(y_positions, ordered.itertuples(index=False)):
        color = threshold_colors[row.threshold]
        axes[1].plot([row.min_delta, row.max_delta], [y, y], color=color, lw=1.25, solid_capstyle="round")
        axes[1].scatter(row.median_delta, y, s=15, color=color, edgecolor="white", linewidth=0.35, zorder=3)
    axes[1].axvline(0, color=DARK, lw=0.65)
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel(r"Median and full range of $\Delta\phi$")
    axes[1].set_title("Magnitude envelope over k = 8-50", loc="left", pad=3)
    clean_axis(axes[1])
    panel_label(axes[1], "b", x=-0.16)
    figure.subplots_adjust(left=0.24, right=0.98, bottom=0.15, top=0.92, wspace=0.20)
    return export_figure(figure, "Fig6_robustness_summary")


def _power_lines(axis: plt.Axes, frame: pd.DataFrame, y_column: str) -> None:
    for power in sorted(frame["power_W"].unique()):
        subset = frame[frame["power_W"] == power].sort_values("kNN")
        axis.plot(
            subset["kNN"],
            subset[y_column],
            color=POWER_COLORS[int(power)],
            lw=0.9,
            label=f"{int(power)} W",
        )


def build_figs2(
    scale: pd.DataFrame,
    metrics: pd.DataFrame,
    core: pd.DataFrame,
    qualified: pd.DataFrame,
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(DOUBLE_WIDTH, 117 * MM))

    scale_grouped = scale.groupby("kNN")["radius_median_mm"].agg(["min", "median", "max"])
    axes[0, 0].fill_between(
        scale_grouped.index,
        scale_grouped["min"],
        scale_grouped["max"],
        color=SKY,
        alpha=0.22,
        linewidth=0,
        label="six-power range",
    )
    axes[0, 0].plot(scale_grouped.index, scale_grouped["median"], color=BLUE, lw=1.1, label="six-power median")
    axes[0, 0].axvline(25, color=DARK, lw=0.65, ls=":")
    axes[0, 0].set_xlabel("Neighbour count, k")
    axes[0, 0].set_ylabel("kth-neighbour radius (mm)")
    axes[0, 0].set_title("Physical support of the kNN scan", loc="left", pad=3)
    selected_ticks = [8, 15, 25, 35, 50]
    axes[0, 0].set_xticks(selected_ticks)
    top_axis = axes[0, 0].secondary_xaxis("top")
    top_axis.set_xticks(selected_ticks)
    top_axis.set_xticklabels([f"{scale_grouped.loc[k, 'median']:.2f}" for k in selected_ticks])
    top_axis.set_xlabel("Median support radius (mm)", labelpad=2)
    axes[0, 0].legend(frameon=False, loc="lower right")
    clean_axis(axes[0, 0])
    panel_label(axes[0, 0], "a")

    unique_metrics = metrics.drop_duplicates(["kNN", "power_W"])
    _power_lines(axes[0, 1], unique_metrics, "wls_valid_fraction")
    axes[0, 1].axvline(25, color=DARK, lw=0.65, ls=":")
    axes[0, 1].set_xlabel("Neighbour count, k")
    axes[0, 1].set_ylabel("WLS-valid fraction")
    axes[0, 1].set_ylim(0.90, 1.005)
    axes[0, 1].set_title("Numerically admissible reconstruction", loc="left", pad=3)
    clean_axis(axes[0, 1])
    panel_label(axes[0, 1], "b")

    q_zero = metrics[(metrics["region"] == "all") & (metrics["threshold"] == "Q>0")]
    _power_lines(axes[1, 0], q_zero, "q_fraction")
    axes[1, 0].axvline(25, color=DARK, lw=0.65, ls=":")
    axes[1, 0].set_xlabel("Neighbour count, k")
    axes[1, 0].set_ylabel("Full-pool positive-Q fraction")
    axes[1, 0].set_title("All six analysed powers", loc="left", pad=3)
    clean_axis(axes[1, 0])
    panel_label(axes[1, 0], "c")

    final = core.merge(qualified, on=["region", "threshold"], validate="many_to_one").sort_values("kNN")
    axes[1, 1].plot(final["kNN"], final["diff_350_400"], color=BLUE, lw=1.1)
    axes[1, 1].axhline(0, color=DARK, lw=0.55)
    axes[1, 1].axvline(25, color=DARK, lw=0.65, ls=":")
    axes[1, 1].set_xlabel("Neighbour count, k")
    axes[1, 1].set_ylabel(r"$\Delta\phi=\phi_{350}-\phi_{400}$")
    axes[1, 1].set_title("Audit-only full-pool Q contrast", loc="left", pad=3)
    clean_axis(axes[1, 1])
    panel_label(axes[1, 1], "d")

    power_handles = [
        plt.Line2D([0], [0], color=POWER_COLORS[power], lw=1.1, label=f"{power} W")
        for power in sorted(POWER_COLORS)
    ]
    figure.legend(
        power_handles,
        [handle.get_label() for handle in power_handles],
        frameon=False,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.005),
        columnspacing=1.1,
    )
    axes[1, 1].legend(
        [plt.Line2D([0], [0], color=BLUE, lw=1.1, label="full-pool $Q>0$")],
        ["full-pool $Q>0$"],
        frameon=False,
        ncol=2,
        loc="upper right",
        columnspacing=0.8,
        handlelength=2.2,
    )
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.91, wspace=0.29, hspace=0.36)
    return export_figure(figure, "FigS2_knn_sensitivity_curves")


def update_traceability(
    core: pd.DataFrame,
    summary: pd.DataFrame,
    scale: pd.DataFrame,
    metrics: pd.DataFrame,
    qualified: pd.DataFrame,
) -> None:
    existing = pd.read_csv(TRACE_PATH)
    existing = existing[~existing["figure_id"].isin(["Fig.5", "Fig.6", "Fig.7", "Fig.8", "Fig. S2"])]
    rows: list[dict[str, str]] = []
    subset = core.merge(qualified, on=["region", "threshold"], validate="many_to_one")
    rows.append(
        {
            "figure_id": "Fig.5",
            "panel_id": "a",
            "source_csv": "图/robustness/knn_core_contrasts.csv",
            "metric_name": "audit-only full-pool Q>0 43-k contrast grid",
            "reported_value": f"min={subset.diff_350_400.min():.6f}; max={subset.diff_350_400.max():.6f}",
            "verified": "yes",
        }
    )
    rows.extend(
        [
            {
                "figure_id": "Fig.6",
                "panel_id": "a",
                "source_csv": "图/robustness/knn_robustness_summary.csv",
                "metric_name": "350 W > 400 W counts",
                "reported_value": "one support- and order-audited combination; denominator=43; no comparative claim",
                "verified": "yes",
            },
            {
                "figure_id": "Fig.6",
                "panel_id": "b",
                "source_csv": "图/robustness/knn_robustness_summary.csv",
                "metric_name": "median and full delta envelope",
                "reported_value": "all 43 pre-specified k values",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S2",
                "panel_id": "a",
                "source_csv": "图/robustness/knn_neighborhood_scale.csv",
                "metric_name": "physical support radius",
                "reported_value": f"k8={scale[scale.kNN==8].radius_median_mm.median():.4f} mm; k50={scale[scale.kNN==50].radius_median_mm.median():.4f} mm",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S2",
                "panel_id": "b/c",
                "source_csv": "图/robustness/knn_power_metrics.csv",
                "metric_name": "six-power WLS validity and positive-Q fractions",
                "reported_value": f"valid fraction={metrics.wls_valid_fraction.min():.4f}-{metrics.wls_valid_fraction.max():.4f}",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S2",
                "panel_id": "d",
                "source_csv": "图/robustness/knn_core_contrasts.csv",
                "metric_name": "audit-only full-pool core contrast",
                "reported_value": "full-pool Q>0 only; k=8-50; no comparative claim",
                "verified": "yes",
            },
        ]
    )
    pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).to_csv(TRACE_PATH, index=False)


def make_contact_sheet(png_paths: list[Path]) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    for axis, path in zip(axes, png_paths):
        axis.imshow(plt.imread(path))
        axis.set_title(path.name, fontsize=8)
        axis.set_axis_off()
    figure.tight_layout()
    qa_dir = Path(tempfile.mkdtemp(prefix="knn_figure_qa_"))
    output = qa_dir / "knn_robustness_contact_sheet.png"
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    core = pd.read_csv(DATA_DIR / "knn_core_contrasts.csv")
    summary = pd.read_csv(DATA_DIR / "knn_robustness_summary.csv")
    scale = pd.read_csv(DATA_DIR / "knn_neighborhood_scale.csv")
    metrics = pd.read_csv(DATA_DIR / "knn_power_metrics.csv")
    model_order = pd.read_csv(ROOT / "图" / "gradient_validation" / "model_order_summary.csv")
    qualified = _gradient_qualified_pairs(model_order)
    figure_paths = [
        build_fig5(core, qualified),
        build_fig6(summary, qualified),
        build_figs2(scale, metrics, core, qualified),
    ]
    update_traceability(core, summary, scale, metrics, qualified)
    contact = make_contact_sheet([paths[2] for paths in figure_paths])
    for paths in figure_paths:
        print(paths[0])
    print(f"QA contact sheet: {contact}")


if __name__ == "__main__":
    main()
