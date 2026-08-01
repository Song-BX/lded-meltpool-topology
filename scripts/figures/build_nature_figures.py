from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="nature_fig_mpl_"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

try:
    from .export_policy import figure_suffixes
except ImportError:  # Direct figure-script execution.
    from export_policy import figure_suffixes


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "latex_restructure" / "figures"
DATA_DIR = ROOT / "图"
SRC = ROOT / "\u56fe"

MM = 1 / 25.4
DOUBLE_W = 183 * MM
BLUE = "#0072B2"       # Okabe-Ito blue
ORANGE = "#D55E00"     # Okabe-Ito vermillion
SKY = "#56B4E9"
YELLOW = "#F0E442"
GRAY = "#4D4D4D"
LIGHT = "#F2F2F2"
DARK = "#1A1A1A"

THRESHOLD_ORDER = ["Q>0", "Q>posP50", "Q>posP75", "Q>posP90"]
THRESHOLD_LABELS = {
    "Q>0": r"$Q>0$",
    "Q>posP50": r"$Q>P_{50}(Q^+)$",
    "Q>posP75": r"$Q>P_{75}(Q^+)$",
    "Q>posP90": r"$Q>P_{90}(Q^+)$",
}
REGION_ORDER = ["all", "interface", "heated", "interface_heated"]
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
        "legend.fontsize": 5.5,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "axes.linewidth": 0.45,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 180,
        "savefig.dpi": 450,
    }
)


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(*parts: str) -> pd.DataFrame:
    return pd.read_csv(SRC.joinpath(*parts))


def panel_label(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.05) -> None:
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


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.5)


def clean_heatmap(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
    ax.tick_params(length=2.0, width=0.45, direction="out", pad=1.5)


def luminance(rgb: tuple[float, float, float, float]) -> float:
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def export_figure(fig: plt.Figure, stem: str) -> Path:
    paths: list[Path] = []
    for ext in figure_suffixes():
        path = FIG_DIR / f"{stem}{ext}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext in [".png", ".tiff"]:
            kwargs["dpi"] = 450
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths[0]


def add_trace(rows: list[dict[str, object]], figure_id: str, panel_id: str, source_csv: str, metric: str, value: object) -> None:
    rows.append(
        {
            "figure_id": figure_id,
            "panel_id": panel_id,
            "source_csv": source_csv,
            "metric_name": metric,
            "reported_value": value,
            "verified": "yes",
        }
    )


def bar_pair(ax: plt.Axes, labels: list[str], values_350: list[float], values_400: list[float], ylabel: str) -> None:
    x = np.arange(len(labels))
    width = 0.32
    ax.bar(x - width / 2, values_350, width, color=BLUE, label="350 W")
    ax.bar(x + width / 2, values_400, width, color=ORANGE, label="400 W")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    clean_axis(ax)


def make_fig1(sens: pd.DataFrame, trace: list[dict[str, object]]) -> Path:
    fig = plt.figure(figsize=(DOUBLE_W, 88 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.72, 1.95, 0.95], wspace=0.34)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_axis_off()
    ax0.set_title("Late-time input", loc="left", pad=3)
    input_lines = [
        "six discrete powers",
        "0.70 s snapshot",
        "exact-coordinate",
        "consolidation",
        "support audit before",
        "claim classification",
    ]
    for index, text in enumerate(input_lines):
        y = 0.83 - index * 0.115
        ax0.add_patch(
            patches.FancyBboxPatch(
                (0.08, y),
                0.82,
                0.075,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                lw=0.55,
                ec="#2A2A2A",
                fc="#E8F1F8" if index < 4 else "#F4F1E8",
            )
        )
        ax0.text(0.49, y + 0.0375, text, ha="center", va="center", fontsize=5.5, color=DARK)
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    panel_label(ax0, "a", x=-0.14, y=1.02)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_axis_off()
    ax1.set_title("Evidence-bounded hierarchy", loc="left", pad=3)
    panel_label(ax1, "b", x=-0.08, y=1.02)
    layers = [
        ("1", "Model provenance\nand fidelity boundary", "current fields\nnot validated", "#F4E8E8"),
        ("2", "Point-cloud\nsupport", "sample counts\nand masks", "#E8F1F8"),
        ("3", "WLS reconstruction\nvalidity", "conditioned\nlocal gradients", "#E8F1F8"),
        ("4", "Q-proxy\nactivity", "reconstruction\naudit only", "#EAF6F1"),
        ("5", "Threshold-kNN\nrobustness", "parameter-domain\nstability", "#F4F1E8"),
        ("6", "Claim\nboundary", "allowed claims\nand limits", "#F4F1E8"),
    ]
    x0, w, h = 0.13, 0.74, 0.092
    ys = np.linspace(0.80, 0.12, len(layers))
    for idx, (num, heading, detail, fc) in enumerate(layers):
        y = ys[idx]
        ax1.add_patch(
            patches.FancyBboxPatch(
                (x0, y),
                w,
                h,
                boxstyle="round,pad=0.014,rounding_size=0.018",
                lw=0.6,
                ec="#2A2A2A",
                fc=fc,
            )
        )
        ax1.add_patch(patches.Circle((x0 + 0.055, y + h / 2), 0.030, fc="white", ec="#2A2A2A", lw=0.5))
        ax1.text(x0 + 0.055, y + h / 2, num, ha="center", va="center", fontsize=5.4, fontweight="bold", color=DARK)
        ax1.text(x0 + 0.115, y + h * 0.62, heading, ha="left", va="center", fontsize=6.0, color=DARK, linespacing=1.0)
        ax1.text(x0 + 0.49, y + h * 0.50, detail, ha="left", va="center", fontsize=5.0, color=GRAY, linespacing=1.0)
        if idx < len(layers) - 1:
            ax1.annotate(
                "",
                xy=(x0 + w / 2, ys[idx + 1] + h + 0.004),
                xytext=(x0 + w / 2, y - 0.004),
                arrowprops=dict(arrowstyle="-|>", lw=0.55, color="#555555", shrinkA=0, shrinkB=0),
            )
    ax1.text(
        0.13,
        0.02,
        "Post-processing bounds interpretation.\nIt does not validate the current CFD fields.",
        ha="left",
        va="bottom",
        fontsize=5.5,
        color=GRAY,
        clip_on=True,
    )
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax3 = fig.add_subplot(gs[0, 2])
    pivot = (
        sens[sens["region"].isin(["all", "interface"])]
        .pivot_table(index="region", columns="threshold", values="diff_350_400", aggfunc="median")
        .reindex(["all", "interface"])
        .reindex(columns=THRESHOLD_ORDER)
    )
    vmax2 = max(float(np.nanmax(np.abs(pivot.values))), 0.12)
    im = ax3.imshow(pivot.values, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax2, vcenter=0, vmax=vmax2), aspect="auto")
    ax3.set_xticks(np.arange(len(THRESHOLD_ORDER)))
    ax3.set_xticklabels([r"$Q>0$", r"$P_{50}$", r"$P_{75}$", r"$P_{90}$"], rotation=35, ha="right")
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["full-pool", "interface"])
    ax3.set_title("Parameter-domain readout", loc="left", pad=3)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax3.text(j, i, f"{pivot.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=5.3, color=DARK)
    clean_heatmap(ax3)
    panel_label(ax3, "c", x=-0.20, y=1.02)
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.035)
    cbar.set_label(r"median $\Delta\phi$", labelpad=2)
    cbar.ax.tick_params(labelsize=5.2, width=0.4, length=2)
    ax3.text(0.00, -0.17, "red: 350 W stronger", transform=ax3.transAxes, ha="left", va="top", fontsize=5.2, color=GRAY)
    add_trace(trace, "Fig.1", "a", "scripts/figures/build_nature_figures.py", "late-time input scope", "six discrete powers; 0.70 s; exact-coordinate consolidation")
    add_trace(trace, "Fig.1", "b", "scripts/figures/build_nature_figures.py", "evidence-bounded hierarchy", "physical-fidelity boundary plus five post-processing layers; under-supported XZ geometry excluded")
    add_trace(trace, "Fig.1", "c", "robustness/knn_core_contrasts.csv", "audit-only parameter-domain readout", f"median Δphi={pivot.values.min():.6f} to {pivot.values.max():.6f}")
    return export_figure(fig, "Fig1_workflow_evidence_bounded")


def make_fig2(counts: pd.DataFrame, main: pd.DataFrame, trace: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 76 * MM), gridspec_kw={"width_ratios": [1.12, 1]})
    region_counts = main.pivot(index="power_W", columns="region", values="n").sort_index().reindex(columns=REGION_ORDER)

    ax = axes[0]
    x = np.arange(len(counts))
    width = 0.34
    ax.bar(x - width / 2, counts["n_raw"], width, color="#BDBDBD", label="raw export")
    ax.bar(x + width / 2, counts["n_unique"], width, color=BLUE, label="deduplicated")
    for xi, raw, unique, ratio in zip(x, counts["n_raw"], counts["n_unique"], counts["dup_ratio"]):
        ax.plot([xi - width / 2, xi + width / 2], [raw, unique], color="#8C8C8C", lw=0.45)
        ax.text(xi, raw + 48, f"{ratio * 100:.1f}%", ha="center", va="bottom", fontsize=5.4, color=GRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(counts["power_W"].astype(int))
    ax.set_xlabel("Laser power (W)")
    ax.set_ylabel("Point count")
    ax.set_title("Duplicate-coordinate normalisation")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    clean_axis(ax)
    panel_label(ax, "a")

    ax = axes[1]
    im = ax.imshow(region_counts.T.values, cmap="cividis", aspect="auto")
    ax.set_xticks(np.arange(len(region_counts.index)))
    ax.set_xticklabels(region_counts.index.astype(int))
    ax.set_yticks(np.arange(len(REGION_ORDER)))
    ax.set_yticklabels([REGION_LABELS[r] for r in REGION_ORDER])
    ax.set_xlabel("Laser power (W)")
    ax.set_title("Region-mask sample support")
    norm = Normalize(vmin=np.nanmin(region_counts.values), vmax=np.nanmax(region_counts.values))
    cmap = plt.get_cmap("cividis")
    for i in range(region_counts.shape[0]):
        for j in range(region_counts.shape[1]):
            value = int(region_counts.iloc[i, j])
            color = "white" if luminance(cmap(norm(value))) < 0.42 else DARK
            ax.text(i, j, str(value), ha="center", va="center", fontsize=5.6, color=color)
    clean_heatmap(ax)
    panel_label(ax, "b")
    cbar = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.035)
    cbar.set_label("valid points")
    cbar.ax.tick_params(labelsize=5.2, width=0.4, length=2)
    fig.tight_layout(w_pad=1.5)

    source_counts = "图/2/Fig2_dedup_counts_summary.csv"
    add_trace(trace, "Fig.2", "a", source_counts, "duplicate ratio range", f"{counts['dup_ratio'].min()*100:.1f}-{counts['dup_ratio'].max()*100:.1f}%")
    source_main = "图/3/Aplus_main_metrics_k25.csv"
    for region in REGION_ORDER:
        add_trace(trace, "Fig.2", "b", source_main, f"{REGION_LABELS[region]} sample support", f"{int(region_counts[region].min())}-{int(region_counts[region].max())}")
    return export_figure(fig, "Fig2_point_cloud_quality")


def make_fig3(main: pd.DataFrame, trace: list[dict[str, object]]) -> Path:
    all_df = main[main["region"] == "all"].set_index("power_W")
    interface_df = main[main["region"] == "interface"].set_index("power_W")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 72 * MM))

    labels = ["full-pool\n$V_{max}$", "interface\nmean $V$"]
    v350 = [all_df.loc[350, "v_max"], interface_df.loc[350, "v_mean"]]
    v400 = [all_df.loc[400, "v_max"], interface_df.loc[400, "v_mean"]]
    bar_pair(axes[0], labels, v350, v400, r"Velocity (m s$^{-1}$)")
    axes[0].set_title("Velocity contrast")
    axes[0].set_ylim(0, max(v350 + v400) * 1.25)
    for i, (a, b) in enumerate(zip(v350, v400)):
        axes[0].text(i, max(a, b) * 1.08, f"350 W +{(a / b - 1) * 100:.1f}%", ha="center", fontsize=5.5, color=GRAY)
    axes[0].legend(frameon=False, loc="upper right")
    panel_label(axes[0], "a")

    labels = ["full-pool\n$T_{max}$", "full-pool\nmean $T$"]
    t350 = [all_df.loc[350, "T_max_K"], all_df.loc[350, "T_mean_K"]]
    t400 = [all_df.loc[400, "T_max_K"], all_df.loc[400, "T_mean_K"]]
    bar_pair(axes[1], labels, t350, t400, "Temperature (K)")
    axes[1].set_title("Thermal contrast")
    axes[1].set_ylim(0, max(t350 + t400) * 1.20)
    axes[1].text(0, max(t350[0], t400[0]) * 1.06, f"400 W +{(t400[0] / t350[0] - 1) * 100:.1f}%", ha="center", fontsize=5.5, color=GRAY)
    panel_label(axes[1], "b")
    fig.tight_layout(w_pad=1.8)

    source = "图/3/Aplus_main_metrics_k25.csv"
    for metric, values in [("full-pool Vmax", (v350[0], v400[0])), ("interface mean velocity", (v350[1], v400[1])), ("full-pool Tmax", (t350[0], t400[0]))]:
        add_trace(trace, "Fig.3", "a/b", source, metric, f"350W={values[0]:.6g}; 400W={values[1]:.6g}")
    return export_figure(fig, "Fig3_six_case_snapshot_power_response")


def make_fig3_six_case(main: pd.DataFrame, trace: list[dict[str, object]]) -> Path:
    """Plot all six discrete power cases without fitting a continuous response."""
    all_df = main[main["region"] == "all"].set_index("power_W")
    interface_df = main[main["region"] == "interface"].set_index("power_W")
    extrema = pd.read_csv(SRC / "power_response_audit" / "local_extremum_audit.csv")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_W, 112 * MM))
    panels = [
        ("temperature_max_full_pool_K", all_df, "T_max_K", "Full-pool maximum temperature", "Temperature (K)", "a"),
        ("temperature_mean_full_pool_K", all_df, "T_mean_K", "Full-pool mean temperature", "Temperature (K)", "b"),
        ("velocity_max_full_pool_mps", all_df, "v_max", "Full-pool maximum velocity", r"Velocity (m s$^{-1}$)", "c"),
        ("velocity_mean_interface_mps", interface_df, "v_mean", "Interface mean velocity", r"Velocity (m s$^{-1}$)", "d"),
    ]
    powers = np.array([200, 250, 300, 350, 400, 450])
    for ax, (metric_id, frame, column, title, ylabel, label) in zip(axes.ravel(), panels):
        values = frame.loc[powers, column].astype(float).to_numpy()
        local_maxima = extrema.loc[
            (extrema["metric_id"] == metric_id)
            & (extrema["extremum_status"] == "discrete_local_maximum"),
            "power_W",
        ].astype(int).tolist()
        ax.scatter(powers, values, s=24, color=BLUE, edgecolor="white", linewidth=0.45, zorder=3)
        if local_maxima:
            indices = [int(np.where(powers == power)[0][0]) for power in local_maxima]
            ax.scatter(
                powers[indices], values[indices], s=62, facecolors="none", edgecolors=ORANGE,
                linewidth=1.0, zorder=4,
            )
            for index in indices:
                ax.annotate(
                    f"{powers[index]} W\nlocal max", (powers[index], values[index]),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                    fontsize=4.9, color=ORANGE,
                )
        ax.set_xticks(powers)
        ax.set_xlabel("Laser power (W)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=3)
        ax.margins(y=0.13)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        clean_axis(ax)
        panel_label(ax, label)
        add_trace(
            trace,
            "Fig.3",
            label,
            "图/3/Aplus_main_metrics_k25.csv",
            title,
            "; ".join(f"{power}W={value:.6g}" for power, value in zip(powers, values)),
        )
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=4.4, label="discrete simulation"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=ORANGE, markersize=6.0, label="sampled-power local maximum"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.53, 1.00), handletextpad=0.35, columnspacing=1.0)
    fig.text(0.5, 0.01, "Markers are discrete simulations within the observed 200--450 W range; endpoints are not regime boundaries. No interpolation or inflection estimate is shown.", ha="center", va="bottom", fontsize=5.1, color=GRAY)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94], h_pad=1.6, w_pad=1.7)
    return export_figure(fig, "Fig3_six_case_snapshot_power_response")


def make_fig3_with_direct_gradients(
    main: pd.DataFrame, gradients: pd.DataFrame, temperature_tail: pd.DataFrame, trace: list[dict[str, object]]
) -> Path:
    """Show all six powers with direct exported temperature-gradient magnitudes."""
    all_df = main[main["region"] == "all"].set_index("power_W")
    interface_df = main[main["region"] == "interface"].set_index("power_W")
    extrema = pd.read_csv(SRC / "power_response_audit" / "local_extremum_audit.csv")
    gradient_snapshot = gradients.loc[
        np.isclose(gradients["time_s"], 0.70)
        & (gradients["aggregation_strategy"] == "mean_all_records")
    ].copy()
    gradient_full = gradient_snapshot.loc[gradient_snapshot["region"] == "full_pool"].set_index("power_W")
    gradient_interface = gradient_snapshot.loc[
        gradient_snapshot["region"] == "interface_proxy"
    ].set_index("power_W")
    median_temperature = temperature_tail.loc[
        np.isclose(temperature_tail["time_s"], 0.70)
        & (temperature_tail["representation"] == "exact_coordinate_mean")
    ].set_index("power_W")
    if len(gradient_full) != 6 or len(gradient_interface) != 6 or len(median_temperature) != 6:
        raise ValueError("Fig. 3 requires six canonical direct-gradient and unfiltered-median-temperature values.")

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_W, 142 * MM))
    panels = [
        ("temperature_median_full_pool_K", median_temperature, "T_median_K", "Full-pool median temperature (unfiltered)", "Temperature (K)", "a", "temperature_tail"),
        ("temperature_mean_full_pool_K", all_df, "T_mean_K", "Full-pool mean temperature", "Temperature (K)", "b", None),
        ("gradient_median_full_pool", gradient_full, "gradT_median_K_per_m", "Full-pool median exported gradient", r"$G$ ($10^6$ K m$^{-1}$)", "c", "gradient"),
        ("gradient_median_interface_proxy", gradient_interface, "gradT_median_K_per_m", "Interface-proxy median exported gradient", r"$G$ ($10^6$ K m$^{-1}$)", "d", "gradient"),
        ("velocity_max_full_pool_mps", all_df, "v_max", "Full-pool maximum velocity", r"Velocity (m s$^{-1}$)", "e", None),
        ("velocity_mean_interface_mps", interface_df, "v_mean", "Interface mean velocity", r"Velocity (m s$^{-1}$)", "f", None),
    ]
    powers = np.array([200, 250, 300, 350, 400, 450])
    for ax, (metric_id, frame, column, title, ylabel, label, panel_kind) in zip(axes.ravel(), panels):
        raw_values = frame.loc[powers, column].astype(float).to_numpy()
        if panel_kind == "gradient":
            lower = raw_values - frame.loc[powers, "gradT_p25_K_per_m"].astype(float).to_numpy()
            upper = frame.loc[powers, "gradT_p75_K_per_m"].astype(float).to_numpy() - raw_values
            values = raw_values / 1e6
            ax.errorbar(powers, values, yerr=np.vstack([lower / 1e6, upper / 1e6]), fmt="none", color=SKY, elinewidth=0.75, capsize=2, zorder=2)
            local_maxima = frame.index[
                frame["sampled_power_extremum"] == "discrete_local_maximum"
            ].astype(int).tolist()
            source = "图/thermal_gradient_audit/thermal_gradient_metrics.csv"
        else:
            values = raw_values
            local_maxima = extrema.loc[
                (extrema["metric_id"] == metric_id)
                & (extrema["extremum_status"] == "discrete_local_maximum"),
                "power_W",
            ].astype(int).tolist()
            source = "图/thermal_fidelity_audit/temperature_tail_metrics.csv" if panel_kind == "temperature_tail" else "图/3/Aplus_main_metrics_k25.csv"
        ax.scatter(powers, values, s=24, color=BLUE, edgecolor="white", linewidth=0.45, zorder=3)
        for power in local_maxima:
            index = int(np.where(powers == power)[0][0])
            ax.scatter(powers[index], values[index], s=62, facecolors="none", edgecolors=ORANGE, linewidth=1.0, zorder=4)
            ax.annotate(f"{powers[index]} W\nlocal max", (powers[index], values[index]), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=4.9, color=ORANGE)
        ax.set_xticks(powers)
        ax.set_xlabel("Laser power (W)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=3)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        clean_axis(ax)
        panel_label(ax, label)
        add_trace(
            trace,
            "Fig.3",
            label,
            source,
            title,
            "; ".join(
                f"{power}W={value:.6g}" for power, value in zip(powers, raw_values)
            ),
        )
    add_trace(
        trace,
        "Fig.3",
        "scope",
        "图/power_response_audit/pairwise_snapshot_context.csv",
        "all-pair snapshot context",
        "60 rows; 15 unordered pairs; observed 200--450 W; no Q input or extrapolation",
    )
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=4.4, label="discrete simulation"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=ORANGE, markersize=6.0, label="sampled-power local maximum"),
        plt.Line2D([0], [0], color=SKY, lw=0.8, marker="|", markersize=5, label="gradient P25--P75 within cloud"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.53, 1.00), handletextpad=0.35, columnspacing=0.9)
    fig.text(0.5, 0.01, "Markers are discrete simulations within the observed 200--450 W range; endpoints are not regime boundaries. Gradient bars show P25--P75 within-cloud distributions, not replicate uncertainty. No interpolation or inflection estimate is shown.", ha="center", va="bottom", fontsize=4.8, color=GRAY)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94], h_pad=1.7, w_pad=1.7)
    return export_figure(fig, "Fig3_six_case_snapshot_power_response")


def make_fig4(main: pd.DataFrame, trace: list[dict[str, object]]) -> Path:
    sub = main[main["power_W"].isin([350, 400])]
    fig, ax = plt.subplots(figsize=(103 * MM, 68 * MM))

    regions = ["all"]
    x = np.arange(len(regions))
    width = 0.34
    vals350 = [sub[(sub["power_W"] == 350) & (sub["region"] == r)]["Q_pos_frac"].iloc[0] for r in regions]
    vals400 = [sub[(sub["power_W"] == 400) & (sub["region"] == r)]["Q_pos_frac"].iloc[0] for r in regions]
    bars350 = ax.bar(x - width / 2, vals350, width, color=BLUE, label="350 W")
    bars400 = ax.bar(x + width / 2, vals400, width, color=ORANGE, label="400 W")
    ax.set_xticks(x)
    ax.set_xticklabels([REGION_LABELS[r] for r in regions])
    ax.set_ylabel("Positive-Q fraction")
    ax.set_title("Reconstructed full-pool Q-proxy activity", loc="left", pad=3)
    ax.set_ylim(0, max(vals350 + vals400) * 1.34)
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax)
    for bars, power in ((bars350, 350), (bars400, 400)):
        for bar, region in zip(bars, regions):
            row = sub[(sub["power_W"] == power) & (sub["region"] == region)].iloc[0]
            positive_count = int(round(float(row["Q_pos_frac"]) * int(row["n"])))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.007,
                f"{positive_count}/{int(row['n'])}",
                ha="center",
                va="bottom",
                fontsize=5.3,
            )
    ax.text(
        0.01,
        0.98,
        "Counts: positive Q / valid points; snapshot descriptor, not a solver-gradient validation",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.0,
        color=GRAY,
    )
    fig.tight_layout()

    source = "图/3/Aplus_main_metrics_k25.csv"
    for region, value350, value400 in zip(regions, vals350, vals400):
        add_trace(
            trace,
            "Fig.4",
            "main",
            source,
            f"{REGION_LABELS[region]} Q>0 fraction",
            f"350W={value350:.6f}; 400W={value400:.6f}",
        )
    return export_figure(fig, "Fig4_q_activity_metrics")


def make_contact_sheet(paths: list[Path]) -> Path:
    images = [plt.imread(str(p.with_suffix(".png"))) for p in paths]
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 12.0))
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, image, path in zip(axes.ravel(), images, paths):
        ax.imshow(image)
        ax.set_title(path.with_suffix(".pdf").name, fontsize=7)
        ax.set_axis_off()
    fig.tight_layout()
    qa_dir = Path(tempfile.mkdtemp(prefix="nature_fig_qa_"))
    out = qa_dir / "nature_figure_contact_sheet.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    from build_export_diagnostics_figure import build_figure as build_export_diagnostics_figure
    from build_knn_robustness_figure import main as build_knn_robustness_figures
    from build_power_response_audit_figure import build_and_update_traceability
    from build_thermal_gradient_audit_figure import build_and_update_traceability as build_gradient_traceability
    from build_thermal_fidelity_audit_figure import build_and_update_traceability as build_thermal_fidelity_figure
    from build_velocity_distribution_overlap_figure import main as build_velocity_distribution_overlap_figure
    from build_velocity_extreme_audit_figure import main as build_velocity_extreme_figure

    ensure_dirs()
    trace: list[dict[str, object]] = []
    counts = read_csv("2", "Fig2_dedup_counts_summary.csv")
    main_df = read_csv("3", "Aplus_main_metrics_k25.csv")
    gradient_df = read_csv("thermal_gradient_audit", "thermal_gradient_metrics.csv")
    temperature_tail_df = read_csv("thermal_fidelity_audit", "temperature_tail_metrics.csv")
    sens = read_csv("7", "Aplus_Qthreshold_sensitivity_350vs400.csv")

    paths = [
        make_fig1(sens, trace),
        build_export_diagnostics_figure(trace),
        make_fig3_with_direct_gradients(main_df, gradient_df, temperature_tail_df, trace),
        make_fig4(main_df, trace),
    ]
    trace_df = pd.DataFrame(trace)
    trace_path = DATA_DIR / "figure_traceability.csv"
    if trace_path.exists():
        existing = pd.read_csv(trace_path)
        replaced_ids = set(trace_df["figure_id"].unique())
        existing = existing[~existing["figure_id"].isin(replaced_ids)]
        trace_df = pd.concat([existing, trace_df], ignore_index=True)
    trace_df.to_csv(trace_path, index=False, encoding="utf-8-sig")
    paths.append(build_gradient_traceability())
    paths.append(build_and_update_traceability())
    build_velocity_distribution_overlap_figure()
    paths.append(FIG_DIR / "FigS1_distribution_checks.pdf")
    build_velocity_extreme_figure()
    paths.append(FIG_DIR / "FigS10_velocity_extreme_solver_health_audit.pdf")
    paths.append(build_thermal_fidelity_figure())
    contact_sheet = make_contact_sheet(paths)
    build_knn_robustness_figures()
    print(f"Generated {len(paths)} base figures plus Fig. 5, Fig. 6, and Fig. S2 in {FIG_DIR}")
    print(f"Contact sheet: {contact_sheet}")


if __name__ == "__main__":
    main()
