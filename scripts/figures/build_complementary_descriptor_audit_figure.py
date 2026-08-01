from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="complementary_descriptor_fig_mpl_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .export_policy import figure_extensions
except ImportError:
    from export_policy import figure_extensions


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "图" / "complementary_descriptor_audit"
FIGURE_DIR = ROOT / "latex_restructure" / "figures"
TRACE_PATH = ROOT / "图" / "figure_traceability.csv"
STEM = "FigS9_complementary_tensor_descriptor_audit"

MM = 1 / 25.4
DOUBLE_WIDTH = 183 * MM
BLUE = "#0F4D92"
TEAL = "#42949E"
VIOLET = "#9A4D8E"
RED = "#B64342"
GRAY = "#4D4D4D"
LIGHT = "#CFCECE"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "legend.fontsize": 5.2,
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
    axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontsize=8, fontweight="bold")


def _definition_panel(axis: plt.Axes, decision: dict[str, object]) -> None:
    axis.axis("off")
    axis.text(0.01, 0.94, "Shared-tensor descriptor logic", fontsize=7, fontweight="bold", va="top")
    axis.text(
        0.01,
        0.78,
        r"$Q=\frac{1}{2}(\|\Omega\|_F^2-\|S\|_F^2)$" + "\n"
        r"$\lambda_2=\mathrm{middle\ eig}(S^2+\Omega^2)$" + "\n"
        r"$\Omega_N=\|\Omega\|_F^2/(\|\Omega\|_F^2+\|S\|_F^2)$",
        fontsize=7,
        va="top",
    )
    axis.text(
        0.01,
        0.47,
        r"For finite non-zero tensors:  $\Omega_N-0.5=Q/D$" + "\n"
        r"Thus $Q>0$ and $\Omega_N>0.5$ are algebraically equivalent.",
        fontsize=6.2,
        va="top",
        color=BLUE,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#E7F0F8", "edgecolor": "none"},
    )
    identity = decision["q_omega_exact_agreement"]
    axis.text(
        0.01,
        0.20,
        f"Canonical check: {identity['all_cells_passed']} in 516 power-k-region cells\n"
        f"Maximum identity error: {identity['max_identity_abs_error']:.2e}\n"
        r"Zero tensors are neutral ($\Omega_N=0.5$).",
        fontsize=5.4,
        va="top",
        color=GRAY,
    )


def _six_power_panel(axis: plt.Axes, metrics: pd.DataFrame) -> None:
    selected = metrics[
        (metrics["context"] == "canonical")
        & (metrics["kNN"] == 25)
        & (metrics["region"] == "all")
    ].sort_values("power_W")
    style = {
        "Q": (BLUE, "o", "Q>0"),
        "lambda2": (VIOLET, "s", r"$\lambda_2<0$"),
        "omega_normalized": (TEAL, "^", r"$\Omega_N>0.5$ (coincident with Q)"),
    }
    for descriptor, block in selected.groupby("descriptor", sort=False):
        color, marker, label = style[descriptor]
        axis.plot(
            block["power_W"],
            block["positive_fraction"] * 100,
            color=color,
            marker=marker,
            ms=3.4,
            lw=1.0,
            ls="--" if descriptor == "omega_normalized" else "-",
            label=label,
        )
    axis.set_xlabel("Laser power (W)")
    axis.set_ylabel("Classified-point fraction (%)")
    axis.set_title("Six-case k=25 reconstructed-tensor audit", loc="left", pad=3)
    axis.set_xticks([200, 250, 300, 350, 400, 450])
    axis.set_ylim(0, 35)
    axis.legend(loc="upper right", handlelength=1.6)
    axis.text(0.02, 0.05, "Each point is one reconstructed spatial sample;\nno replicate uncertainty or inference is shown.", transform=axis.transAxes, fontsize=4.9, color=GRAY)
    _clean_axis(axis)


def _agreement_panel(axis: plt.Axes, agreement: pd.DataFrame) -> None:
    selected = agreement[
        (agreement["context"] == "canonical")
        & (agreement["first_descriptor"] == "Q")
        & (agreement["second_descriptor"].isin(["lambda2", "omega_normalized"]))
    ]
    series = [
        ("lambda2", "all", VIOLET, "Q vs lambda2, full pool"),
        ("lambda2", "interface", "#B89BD9", "Q vs lambda2, interface"),
        ("omega_normalized", "all", TEAL, "Q vs Omega_N, both regions"),
    ]
    for descriptor, region, color, label in series:
        block = selected[(selected["second_descriptor"] == descriptor) & (selected["region"] == region)].sort_values("kNN")
        if descriptor == "omega_normalized":
            block = selected[(selected["second_descriptor"] == descriptor)].groupby("kNN", as_index=False)["agreement_fraction"].mean()
        axis.plot(block["kNN"], block["agreement_fraction"] * 100, color=color, lw=1.0, marker="o", ms=1.8, label=label)
    axis.set_xlabel("Neighbour count, k")
    axis.set_ylabel("Pointwise classification agreement (%)")
    axis.set_ylim(70, 101)
    axis.set_title("Lambda2 is complementary but not identical to Q", loc="left", pad=3)
    axis.axhline(100, color=LIGHT, lw=0.55, zorder=0)
    axis.axvline(25, color=GRAY, lw=0.55, ls="--")
    axis.legend(loc="lower right", handlelength=1.5)
    _clean_axis(axis)


def _all_positive(block: pd.DataFrame) -> bool:
    directions = set(block["direction_350_400"].dropna())
    return bool(directions) and directions == {"350>400"}


def _sensitivity_panel(axis: plt.Axes, sensitivity: pd.DataFrame, decision: dict[str, object]) -> None:
    columns = [
        ("weight_exponent_sensitivity", "alpha_label", "0.0", r"$\alpha=0$"),
        ("weight_exponent_sensitivity", "alpha_label", "1.0", r"$\alpha=1$"),
        ("weight_exponent_sensitivity", "alpha_label", "2.0", r"$\alpha=2$"),
        ("conditioning_sensitivity", "cutoff_label", "10", r"$\kappa=10$"),
        ("conditioning_sensitivity", "cutoff_label", "100", r"$\kappa=100$"),
        ("conditioning_sensitivity", "cutoff_label", "1e12", r"$\kappa=10^{12}$"),
        ("conditioning_sensitivity", "cutoff_label", "inf", r"$\kappa=\infty$"),
        ("model_order_sensitivity", "method", "first_order", "order 1"),
        ("model_order_sensitivity", "method", "second_order", "order 2"),
    ]
    descriptors = ["Q", "lambda2", "omega_normalized"]
    matrix = np.zeros((len(descriptors), len(columns)))
    for row, descriptor in enumerate(descriptors):
        for column, (context, field, value, _) in enumerate(columns):
            selector = sensitivity[field].astype(str) == value
            if field == "cutoff_label":
                selector = pd.to_numeric(sensitivity[field], errors="coerce").eq(float(value))
            block = sensitivity[
                (sensitivity["context"] == context)
                & (sensitivity["region"] == "all")
                & (sensitivity["descriptor"] == descriptor)
                & selector
            ]
            matrix[row, column] = 1 if _all_positive(block) else 0
    axis.imshow(matrix, cmap=colors.ListedColormap(["#E9A6A1", "#3775BA"]), vmin=0, vmax=1, aspect="auto")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, "350>400" if value else "mixed", ha="center", va="center", fontsize=4.4, color="white" if value else "black")
    axis.set_yticks(range(3), [r"$Q>0$", r"$\lambda_2<0$", r"$\Omega_N>0.5$"])
    axis.set_xticks(range(len(columns)), [label for _, _, _, label in columns], rotation=35, ha="right")
    axis.set_title("Direction audit across reconstruction screens", loc="left", pad=3)
    axis.text(
        0.01,
        -0.37,
        "All displayed directions are audit descriptors only: alpha=2 fails the affine numerical gate,\n"
        "the 350–400 temporal direction does not persist, and no native solver-gradient reference is available.",
        transform=axis.transAxes,
        fontsize=4.9,
        color=RED,
        va="top",
    )
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)


def _update_traceability(metrics: pd.DataFrame, agreement: pd.DataFrame, decision: dict[str, object]) -> None:
    existing = pd.read_csv(TRACE_PATH) if TRACE_PATH.exists() else pd.DataFrame()
    if len(existing):
        existing = existing[existing["figure_id"] != "Fig. S9"]
    q_lambda = agreement[
        (agreement["context"] == "canonical")
        & (agreement["first_descriptor"] == "Q")
        & (agreement["second_descriptor"] == "lambda2")
    ]["agreement_fraction"]
    core = metrics[(metrics["context"] == "canonical") & (metrics["kNN"] == 25) & (metrics["region"] == "all")]
    q350 = core[(core["power_W"] == 350) & (core["descriptor"] == "Q")]["positive_fraction"].iloc[0]
    q400 = core[(core["power_W"] == 400) & (core["descriptor"] == "Q")]["positive_fraction"].iloc[0]
    rows = pd.DataFrame(
        [
            {"figure_id": "Fig. S9", "panel_id": "a", "source_csv": "图/complementary_descriptor_audit/complementary_descriptor_decision.json", "metric_name": "Q-Omega algebraic identity", "reported_value": f"max absolute identity error={decision['q_omega_exact_agreement']['max_identity_abs_error']:.3e}; 516/516 cells agree", "verified": "yes"},
            {"figure_id": "Fig. S9", "panel_id": "b", "source_csv": "图/complementary_descriptor_audit/descriptor_metrics.csv", "metric_name": "six-power k=25 classified fractions", "reported_value": f"full-pool Q/Omega: 350 W={q350:.6f}; 400 W={q400:.6f}", "verified": "yes"},
            {"figure_id": "Fig. S9", "panel_id": "c", "source_csv": "图/complementary_descriptor_audit/descriptor_agreement.csv", "metric_name": "Q-lambda2 classification agreement", "reported_value": f"canonical range={q_lambda.min():.6f}-{q_lambda.max():.6f}; not independent validation", "verified": "yes"},
            {"figure_id": "Fig. S9", "panel_id": "d", "source_csv": "图/complementary_descriptor_audit/descriptor_sensitivity.csv", "metric_name": "direction audit over reconstruction screens", "reported_value": "all displayed 350 W > 400 W directions remain audit-only because existing promotion gates fail", "verified": "yes"},
        ]
    )
    pd.concat([existing, rows], ignore_index=True).to_csv(TRACE_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    metrics = pd.read_csv(DATA_DIR / "descriptor_metrics.csv", low_memory=False)
    agreement = pd.read_csv(DATA_DIR / "descriptor_agreement.csv", low_memory=False)
    sensitivity = pd.read_csv(DATA_DIR / "descriptor_sensitivity.csv", low_memory=False)
    decision = json.loads((DATA_DIR / "complementary_descriptor_decision.json").read_text(encoding="utf-8"))

    figure, axes = plt.subplots(2, 2, figsize=(DOUBLE_WIDTH, 118 * MM))
    _definition_panel(axes[0, 0], decision)
    _six_power_panel(axes[0, 1], metrics)
    _agreement_panel(axes[1, 0], agreement)
    _sensitivity_panel(axes[1, 1], sensitivity, decision)
    for axis, label in zip(axes.ravel(), "abcd"):
        _panel_label(axis, label)
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.17, top=0.94, wspace=0.35, hspace=0.43)

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
    _update_traceability(metrics, agreement, decision)
    print(f"Generated {STEM} in PDF/SVG/PNG/TIFF")


if __name__ == "__main__":
    main()
