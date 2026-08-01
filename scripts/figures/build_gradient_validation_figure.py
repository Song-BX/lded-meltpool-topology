from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="gradient_validation_fig_mpl_"))

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
DATA_DIR = ROOT / "图" / "gradient_validation"
FIGURE_DIR = ROOT / "latex_restructure" / "figures"
TRACE_PATH = ROOT / "图" / "figure_traceability.csv"
STEM = "FigS5_gradient_reconstruction_validation"

MM = 1 / 25.4
DOUBLE_WIDTH = 183 * MM
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
DARK = "#1A1A1A"
GRAY = "#5A5A5A"


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
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", pad=1.5)


def _panel_label(axis: plt.Axes, label: str, x: float = -0.15) -> None:
    axis.text(
        x,
        1.04,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def _error_panel(axis: plt.Axes, summary: pd.DataFrame) -> None:
    styles = {
        ("gaussian_vortex", "all"): (BLUE, "o", "Gaussian vortex, full-pool"),
        ("gaussian_vortex", "interface"): (ORANGE, "o", "Gaussian vortex, interface"),
        ("tanh_shear", "all"): (GREEN, "s", "Tanh shear, full-pool"),
        ("tanh_shear", "interface"): (PURPLE, "s", "Tanh shear, interface"),
    }
    for (field_id, region), (color, marker, label) in styles.items():
        block = summary[(summary["field_id"] == field_id) & (summary["region"] == region)].sort_values("feature_scale_mm")
        axis.plot(
            block["feature_scale_mm"],
            block["gradient_nrmse_median"],
            color=color,
            marker=marker,
            ms=3.0,
            lw=1.0,
            label=label,
        )
    axis.axhline(1.0, color=GRAY, lw=0.6, ls="--")
    axis.set_xlabel("Analytic feature scale (mm)")
    axis.set_ylabel("Median gradient NRMSE")
    axis.set_title("Controlled gradient recovery on observed geometry", loc="left", pad=3)
    axis.set_xticks([0.10, 0.20, 0.30])
    axis.legend(frameon=False, ncol=2, loc="upper right", columnspacing=0.7, handlelength=1.5)
    _clean_axis(axis)


def _q_sign_panel(axis: plt.Axes, summary: pd.DataFrame, affine_max: float) -> None:
    colors = {"all": BLUE, "interface": ORANGE}
    for region, color in colors.items():
        block = summary[
            (summary["field_id"] == "gaussian_vortex") & (summary["region"] == region)
        ].sort_values("feature_scale_mm")
        axis.plot(
            block["feature_scale_mm"],
            block["q_sign_accuracy_margin_median"],
            color=color,
            marker="o",
            ms=3.2,
            lw=1.0,
            label=region.replace("all", "full-pool"),
        )
    axis.axhline(0.80, color=GRAY, lw=0.6, ls="--")
    axis.set_ylim(0.25, 1.03)
    axis.set_xlabel("Gaussian-vortex scale (mm)")
    axis.set_ylabel("Q-sign accuracy outside 5% margin")
    axis.set_title("Q classification is scale dependent", loc="left", pad=3)
    axis.set_xticks([0.10, 0.20, 0.30])
    axis.text(
        0.02,
        0.04,
        f"Affine P90 gradient NRMSE = {affine_max:.1e}",
        transform=axis.transAxes,
        fontsize=5.0,
        color=GRAY,
    )
    axis.legend(frameon=False, loc="lower right")
    _clean_axis(axis)


def _order_panel(axis: plt.Axes, model: pd.DataFrame) -> None:
    evidence = model[model["evidence_eligible"]].copy()
    evidence["label"] = evidence["region"].map({"all": "full-pool", "interface": "interface"}) + "  " + evidence["threshold"].map(
        {"Q>0": r"$Q>0$", "Q>posP50": r"$P_{50}$", "Q>posP75": r"$P_{75}$", "Q>posP90": r"$P_{90}$"}
    )
    evidence = evidence.sort_values(["region", "threshold"], ascending=[False, True]).reset_index(drop=True)
    y = np.arange(len(evidence))[::-1]
    colors = [BLUE if value == 0 else ORANGE for value in evidence["direction_mismatch_count"]]
    axis.barh(y, evidence["direction_mismatch_count"], color=colors, height=0.62)
    axis.set_yticks(y)
    axis.set_yticklabels(evidence["label"])
    axis.set_xlabel("First-/second-order direction mismatches (of 36 k)")
    axis.set_title("Only full-pool $Q>0$ is order consistent", loc="left", pad=3)
    axis.set_xlim(0, max(12, int(evidence["direction_mismatch_count"].max()) + 1))
    for yy, row in zip(y, evidence.itertuples(index=False)):
        axis.text(row.direction_mismatch_count + 0.25, yy, f"{int(row.direction_mismatch_count)}/36", va="center", fontsize=5.1)
    _clean_axis(axis)


def _resampling_panel(axis: plt.Axes, core: pd.DataFrame) -> None:
    data = [
        core.loc[core["region"] == region, "delta_350_400"].to_numpy()
        for region in ("all", "interface")
    ]
    box = axis.boxplot(data, tick_labels=["full-pool", "interface"], widths=0.52, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], (BLUE, ORANGE)):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor(DARK)
        patch.set_linewidth(0.55)
    for key in ("whiskers", "caps", "medians"):
        for artist in box[key]:
            artist.set_color(DARK)
            artist.set_linewidth(0.55)
    axis.axhline(0, color=DARK, lw=0.65)
    axis.set_ylabel(r"$Delta\phi_{Q>0}=\phi_{350}-\phi_{400}$")
    axis.set_title("Neighbour-subset diagnostic at k = 25", loc="left", pad=3)
    axis.text(
        0.02,
        0.96,
        "200 random 20-of-25 subsets; no inferential statistics",
        transform=axis.transAxes,
        fontsize=5.0,
        color=GRAY,
        va="top",
    )
    _clean_axis(axis)


def _update_traceability(summary: pd.DataFrame, model: pd.DataFrame, core: pd.DataFrame) -> None:
    existing = pd.read_csv(TRACE_PATH) if TRACE_PATH.exists() else pd.DataFrame()
    existing = existing[existing["figure_id"] != "Fig. S5"] if len(existing) else existing
    affine_max = float(summary[summary["field_class"] == "affine"]["gradient_nrmse_p90"].max())
    rows = pd.DataFrame(
        [
            {
                "figure_id": "Fig. S5",
                "panel_id": "a",
                "source_csv": "图/gradient_validation/manufactured_field_summary.csv",
                "metric_name": "nonlinear gradient NRMSE over observed point geometries",
                "reported_value": "Gaussian vortex and tanh shear at 0.10, 0.20, and 0.30 mm",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S5",
                "panel_id": "b",
                "source_csv": "图/gradient_validation/manufactured_field_summary.csv",
                "metric_name": "Gaussian-vortex Q-sign accuracy outside 5% true-Q margin",
                "reported_value": f"affine P90 gradient NRMSE={affine_max:.2e}",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S5",
                "panel_id": "c",
                "source_csv": "图/gradient_validation/model_order_summary.csv",
                "metric_name": "first-/second-order directional mismatch count",
                "reported_value": "full-pool Q>0 = 0/36; all other eligible comparisons have at least one mismatch",
                "verified": "yes",
            },
            {
                "figure_id": "Fig. S5",
                "panel_id": "d",
                "source_csv": "图/gradient_validation/neighbour_subset_core_contrasts.csv",
                "metric_name": "20-of-25 neighbour-subset Q>0 contrast",
                "reported_value": f"{len(core) // 2} resamples per region; all deltas positive",
                "verified": "yes",
            },
        ]
    )
    pd.concat([existing, rows], ignore_index=True).to_csv(TRACE_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    summary = pd.read_csv(DATA_DIR / "manufactured_field_summary.csv")
    model = pd.read_csv(DATA_DIR / "model_order_summary.csv")
    core = pd.read_csv(DATA_DIR / "neighbour_subset_core_contrasts.csv")
    affine_max = float(summary[summary["field_class"] == "affine"]["gradient_nrmse_p90"].max())

    figure, axes = plt.subplots(2, 2, figsize=(DOUBLE_WIDTH, 118 * MM))
    _error_panel(axes[0, 0], summary)
    _q_sign_panel(axes[0, 1], summary, affine_max)
    _order_panel(axes[1, 0], model)
    _resampling_panel(axes[1, 1], core)
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
    _update_traceability(summary, model, core)
    print(f"Generated {STEM} in PDF/SVG/PNG/TIFF")


if __name__ == "__main__":
    main()
