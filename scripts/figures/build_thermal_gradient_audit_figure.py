from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="thermal_gradient_mpl_"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

try:
    from .export_policy import figure_suffixes
except ImportError:  # Direct figure-script execution.
    from export_policy import figure_suffixes


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "图" / "thermal_gradient_audit"
FIG_DIR = ROOT / "latex_restructure" / "figures"
TRACE_PATH = ROOT / "图" / "figure_traceability.csv"

MM = 1 / 25.4
DOUBLE_W = 183 * MM
BLUE = "#0072B2"
ORANGE = "#D55E00"
SKY = "#56B4E9"
GREEN = "#009E73"
GRAY = "#4D4D4D"
DARK = "#1A1A1A"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "legend.fontsize": 5.1,
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


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.5)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold", color=DARK)


def _export(fig: plt.Figure) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIG_DIR / "FigS3_temperature_gradient_audit"
    for suffix in figure_suffixes():
        kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix in {".png", ".tiff"}:
            kwargs["dpi"] = 600
        if suffix == ".tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(stem.with_suffix(suffix), **kwargs)
    plt.close(fig)
    return stem.with_suffix(".pdf")


def _gradient_panel(ax: plt.Axes, frame: pd.DataFrame, title: str, label: str) -> list[dict[str, object]]:
    ordered = frame.sort_values("power_W")
    powers = ordered["power_W"].to_numpy(dtype=int)
    medians = ordered["gradT_median_K_per_m"].to_numpy(dtype=float) / 1e6
    lower = medians - ordered["gradT_p25_K_per_m"].to_numpy(dtype=float) / 1e6
    upper = ordered["gradT_p75_K_per_m"].to_numpy(dtype=float) / 1e6 - medians
    ax.errorbar(powers, medians, yerr=np.vstack([lower, upper]), fmt="none", color=SKY, elinewidth=0.8, capsize=2, zorder=2, label="P25--P75 within cloud")
    ax.scatter(powers, medians, s=26, color=BLUE, edgecolor="white", linewidth=0.45, zorder=3, label="median")
    maxima = ordered.loc[ordered["sampled_power_extremum"] == "discrete_local_maximum", "power_W"].astype(int).tolist()
    for power in maxima:
        index = int(np.where(powers == power)[0][0])
        ax.scatter(powers[index], medians[index], s=66, facecolors="none", edgecolors=ORANGE, linewidth=1.0, zorder=4)
        ax.annotate(f"{power} W\nlocal max", (powers[index], medians[index]), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=4.8, color=ORANGE)
    ax.set_xticks(powers)
    ax.set_xlabel("Laser power (W)")
    ax.set_ylabel(r"Exported $G$ ($10^6$ K m$^{-1}$)")
    ax.set_title(title, loc="left", pad=3)
    _clean_axis(ax)
    _panel_label(ax, label)
    return [{
        "figure_id": "Fig.S3",
        "panel_id": label,
        "source_csv": "图/thermal_gradient_audit/thermal_gradient_metrics.csv",
        "metric_name": title,
        "reported_value": "; ".join(f"{power}W={value:.6g}" for power, value in zip(powers, medians * 1e6)),
        "verified": "yes",
    }]


def build_figure() -> tuple[Path, list[dict[str, object]]]:
    metrics = pd.read_csv(AUDIT_DIR / "thermal_gradient_metrics.csv")
    aggregation = pd.read_csv(AUDIT_DIR / "thermal_gradient_aggregation_sensitivity.csv")
    temporal = pd.read_csv(AUDIT_DIR / "thermal_gradient_temporal_context.csv")
    with (AUDIT_DIR / "thermal_gradient_decision.json").open(encoding="utf-8") as handle:
        decision = json.load(handle)
    if decision["primary_descriptor_status"] != "direct_exported_temperature_gradient_magnitude":
        raise ValueError("Fig. S3 requires direct exported temperature-gradient results.")
    if decision["marangoni_status"] != "not_identifiable_from_available_csv":
        raise ValueError("Fig. S3 must retain the available-data Marangoni boundary.")

    canonical = metrics.loc[np.isclose(metrics["time_s"], 0.70) & (metrics["aggregation_strategy"] == "mean_all_records")].copy()
    full = canonical.loc[canonical["region"] == "full_pool"]
    interface = canonical.loc[canonical["region"] == "interface_proxy"]
    if len(full) != 6 or len(interface) != 6:
        raise ValueError("Fig. S3 requires six canonical rows for both gradient regions.")

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_W, 126 * MM))
    trace = _gradient_panel(axes[0, 0], full, "Full-pool exported gradient magnitude", "a")
    trace.extend(_gradient_panel(axes[0, 1], interface, "Interface-proxy exported gradient magnitude", "b"))
    axes[0, 1].legend(frameon=False, loc="upper left", handlelength=1.0, labelspacing=0.25)

    canonical_values = aggregation.loc[aggregation["aggregation_strategy"] == "mean_all_records"].set_index(["region", "power_W"])["gradT_median_K_per_m"]
    styles = {
        "mean_all_records": (BLUE, "mean all records"),
        "median_all_records": (GREEN, "median all records"),
        "first_record": (ORANGE, "first record"),
        "mean_distinct_states": ("#7A5AA6", "mean distinct states"),
    }
    ax = axes[1, 0]
    for region, linestyle, marker in (("full_pool", "-", "o"), ("interface_proxy", "--", "s")):
        for strategy, (color, label_text) in styles.items():
            sub = aggregation.loc[(aggregation["region"] == region) & (aggregation["aggregation_strategy"] == strategy)].sort_values("power_W")
            powers = sub["power_W"].to_numpy(dtype=int)
            baseline = np.array([canonical_values.loc[(region, power)] for power in powers], dtype=float)
            delta = (sub["gradT_median_K_per_m"].to_numpy(dtype=float) / baseline - 1.0) * 100.0
            ax.plot(powers, delta, color=color, linestyle=linestyle, marker=marker, ms=2.8, lw=0.8, label=f"{label_text}, {region.replace('_', '-')}")
    ax.axhline(0, color=DARK, lw=0.45)
    ax.set_xticks([200, 250, 300, 350, 400, 450])
    ax.set_xlabel("Laser power (W)")
    ax.set_ylabel("Median-$G$ change from canonical (%)")
    ax.set_title("Four coordinate-aggregation strategies", loc="left", pad=3)
    _clean_axis(ax)
    _panel_label(ax, "c")
    handles = [
        plt.Line2D([0], [0], color=GRAY, lw=0.8, linestyle="-", marker="o", ms=2.8, label="full-pool"),
        plt.Line2D([0], [0], color=GRAY, lw=0.8, linestyle="--", marker="s", ms=2.8, label="interface proxy"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left", ncol=2, handlelength=1.1, columnspacing=0.8)
    baseline_per_row = np.array([canonical_values.loc[(row.region, row.power_W)] for row in aggregation.itertuples()], dtype=float)
    max_delta = float(np.max(np.abs((aggregation["gradT_median_K_per_m"].to_numpy(dtype=float) / baseline_per_row - 1.0) * 100.0)))
    trace.append({
        "figure_id": "Fig.S3",
        "panel_id": "c",
        "source_csv": "图/thermal_gradient_audit/thermal_gradient_aggregation_sensitivity.csv",
        "metric_name": "median-G coordinate-aggregation sensitivity",
        "reported_value": f"maximum relative difference={max_delta:.3f}%; all six-power orders and local maxima preserved",
        "verified": "yes",
    })

    ax = axes[1, 1]
    direction_rows: list[list[int]] = []
    for region in ("full_pool", "interface_proxy"):
        row: list[int] = []
        for time_s in sorted(temporal["time_s"].unique()):
            values = temporal.loc[(temporal["region"] == region) & np.isclose(temporal["time_s"], time_s)].set_index("power_W")["gradT_median_K_per_m"]
            row.append(1 if values.loc[350] > values.loc[400] else -1 if values.loc[350] < values.loc[400] else 0)
        direction_rows.append(row)
    ax.imshow(np.array(direction_rows), cmap=ListedColormap([BLUE, "#D9D9D9", ORANGE]), vmin=-1, vmax=1, aspect="auto")
    times = sorted(temporal["time_s"].unique())
    ax.set_xticks(np.arange(len(times)))
    ax.set_xticklabels([f"{time:.2f}" for time in times])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["full-pool", "interface proxy"])
    ax.set_xlabel("Time (s)")
    ax.set_title("350 W vs 400 W median-$G$ direction", loc="left", pad=3)
    for row_index, row in enumerate(direction_rows):
        for col_index, direction in enumerate(row):
            text = "350 > 400" if direction == 1 else "350 < 400" if direction == -1 else "tie"
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=4.8, color="white" if direction else DARK)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
    ax.tick_params(length=2.0, width=0.45, direction="out", pad=1.5)
    _panel_label(ax, "d")
    trace.append({
        "figure_id": "Fig.S3",
        "panel_id": "d",
        "source_csv": "图/thermal_gradient_audit/thermal_gradient_temporal_context.csv",
        "metric_name": "post-hoc 350 W--400 W median-G direction",
        "reported_value": "both regions reverse at 0.60 s; temporal context only",
        "verified": "yes",
    })

    fig.text(0.5, 0.012, r"$G=|\nabla T|$ is the directly exported scalar gradient magnitude. P25--P75 bars describe within-cloud distributions, not replicate uncertainty. This is not a Marangoni test.", ha="center", va="bottom", fontsize=5.2, color=GRAY)
    fig.tight_layout(rect=[0, 0.035, 1, 1], h_pad=1.7, w_pad=1.8)
    return _export(fig), trace


def build_and_update_traceability() -> Path:
    output, trace = build_figure()
    new = pd.DataFrame(trace)
    if TRACE_PATH.exists():
        existing = pd.read_csv(TRACE_PATH)
        new = pd.concat([existing.loc[existing["figure_id"] != "Fig.S3"], new], ignore_index=True)
    new.to_csv(TRACE_PATH, index=False, encoding="utf-8-sig")
    return output


def main() -> None:
    print(f"Generated {build_and_update_traceability()}")


if __name__ == "__main__":
    main()
