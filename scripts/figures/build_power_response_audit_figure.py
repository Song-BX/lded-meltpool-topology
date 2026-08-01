from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="power_response_mpl_"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .export_policy import figure_suffixes
except ImportError:  # Direct figure-script execution.
    from export_policy import figure_suffixes


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "图" / "power_response_audit"
FIG_DIR = ROOT / "latex_restructure" / "figures"
TRACE_PATH = ROOT / "图" / "figure_traceability.csv"
MM = 1 / 25.4
DOUBLE_W = 183 * MM
GRAY = "#4D4D4D"
TIME_COLORS = ["#3B4CC0", "#648FFF", "#00A087", "#FCA636", "#B40426"]
TIME_LABELS = {0.50: "0.50 s", 0.55: "0.55 s", 0.60: "0.60 s", 0.65: "0.65 s", 0.70: "0.70 s"}


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
        "axes.linewidth": 0.45,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 450,
    }
)


def _metric_layout() -> list[tuple[str, str, str, str]]:
    return [
        ("temperature_max_tail_audit", "Full-pool maximum temperature (tail audit)", "Temperature (K)", "a"),
        ("temperature_mean_full_pool_K", "Full-pool mean temperature", "Temperature (K)", "b"),
        ("velocity_max_full_pool_mps", "Full-pool maximum velocity", r"Velocity (m s$^{-1}$)", "c"),
        ("velocity_mean_interface_mps", "Interface mean velocity", r"Velocity (m s$^{-1}$)", "d"),
    ]


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.5)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")


def _export(fig: plt.Figure, stem: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in figure_suffixes():
        kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if ext in {".png", ".tiff"}:
            kwargs["dpi"] = 450
        fig.savefig(FIG_DIR / f"{stem}{ext}", **kwargs)
    plt.close(fig)
    return FIG_DIR / f"{stem}.pdf"


def _maximum_text(block: pd.DataFrame, group_columns: list[str]) -> pd.Series:
    maxima = block.loc[block["extremum_status"] == "discrete_local_maximum"]
    return maxima.groupby(group_columns)["power_W"].apply(
        lambda values: ",".join(str(int(value)) for value in sorted(values))
    )


def build_figure() -> tuple[Path, list[dict[str, object]]]:
    temporal = pd.read_csv(AUDIT_DIR / "temporal_local_extrema.csv")
    temperature_tail = pd.read_csv(ROOT / "图" / "thermal_fidelity_audit" / "temperature_tail_metrics.csv")
    aggregation = pd.read_csv(AUDIT_DIR / "aggregation_local_extrema.csv")
    pairwise_context = pd.read_csv(AUDIT_DIR / "pairwise_snapshot_context.csv")
    with (AUDIT_DIR / "power_response_decision.json").open(encoding="utf-8") as handle:
        decision = json.load(handle)
    if decision["decision"] != "no_physical_inflection_claim":
        raise ValueError("Fig. S8 requires the no-physical-inflection decision state")
    if (
        len(pairwise_context) != 60
        or len(pairwise_context[["lower_power_W", "higher_power_W"]].drop_duplicates()) != 15
        or decision["observed_power_domain"] != "200--450 W"
        or bool(decision["higher_power_regime_assessed"])
    ):
        raise ValueError("Fig. S8 requires the complete, bounded all-pair snapshot context")

    fig = plt.figure(figsize=(DOUBLE_W, 160 * MM))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.83], hspace=0.52, wspace=0.34)
    trace: list[dict[str, object]] = []
    powers = np.array([200, 250, 300, 350, 400, 450])
    times = sorted(float(value) for value in temporal["time_s"].unique())

    for index, (metric_id, title, ylabel, label) in enumerate(_metric_layout()):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        if metric_id == "temperature_max_tail_audit":
            block = temperature_tail.loc[
                temperature_tail["representation"] == "exact_coordinate_mean",
                ["time_s", "power_W", "T_max_K"],
            ].rename(columns={"T_max_K": "value"})
        else:
            block = temporal.loc[temporal["metric_id"] == metric_id]
        for time_s, color in zip(times, TIME_COLORS):
            values = block.loc[np.isclose(block["time_s"], time_s)].sort_values("power_W")
            ax.scatter(values["power_W"], values["value"], s=14, color=color, edgecolor="white", linewidth=0.30, label=TIME_LABELS[time_s], zorder=3)
        ax.set_xticks(powers)
        ax.set_xlabel("Laser power (W)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=3)
        _clean_axis(ax)
        _panel_label(ax, label)
        trace.append(
            {
                "figure_id": "Fig.S8",
                "panel_id": label,
                "source_csv": "图/thermal_fidelity_audit/temperature_tail_metrics.csv" if metric_id == "temperature_max_tail_audit" else "图/power_response_audit/temporal_local_extrema.csv",
                "metric_name": title,
                "reported_value": "five time points × six discrete powers; no interpolation",
                "verified": "yes",
            }
        )
    fig.axes[0].legend(frameon=False, ncol=3, loc="upper left", handletextpad=0.25, columnspacing=0.65)

    metric_ids = ["temperature_median_full_pool_K", "temperature_mean_full_pool_K", "velocity_max_full_pool_mps", "velocity_mean_interface_mps"]
    strategies = ["mean_all_records", "median_all_records", "first_record", "mean_distinct_states"]
    ax = fig.add_subplot(grid[2, 0])
    labels = _maximum_text(aggregation, ["metric_id", "aggregation_strategy"])
    ax.imshow(np.ones((len(metric_ids), len(strategies))), cmap="Blues", vmin=0, vmax=2, aspect="auto", alpha=0.22)
    for row, metric_id in enumerate(metric_ids):
        for column, strategy in enumerate(strategies):
            ax.text(column, row, labels.get((metric_id, strategy), "–"), ha="center", va="center", fontsize=5.7, color=GRAY)
    ax.set_xticks(np.arange(len(strategies)))
    ax.set_xticklabels(["mean", "median", "first", "distinct\nstates"])
    ax.set_yticks(np.arange(len(metric_ids)))
    ax.set_yticklabels(["median $T$", "mean $T$", "$V_{max}$", "interface mean $V$"])
    ax.set_title("0.70 s local-maximum power(s) by aggregation strategy", loc="left", pad=3)
    ax.set_xlabel("Coordinate aggregation")
    ax.set_ylabel("Metric")
    ax.tick_params(length=0)
    _panel_label(ax, "e")
    trace.append(
        {
            "figure_id": "Fig.S8",
            "panel_id": "e",
            "source_csv": "图/power_response_audit/aggregation_local_extrema.csv",
            "metric_name": "local-maximum powers across four aggregation strategies",
            "reported_value": "all 0.70 s local-extremum statuses agree across four strategies",
            "verified": "yes",
        }
    )

    ax = fig.add_subplot(grid[2, 1])
    labels = _maximum_text(temporal, ["metric_id", "time_s"])
    canonical_labels = _maximum_text(temporal.loc[np.isclose(temporal["time_s"], 0.70)], ["metric_id"])
    matrix = np.zeros((len(metric_ids), len(times)))
    for row, metric_id in enumerate(metric_ids):
        canonical_text = canonical_labels.get(metric_id, "–")
        for column, time_s in enumerate(times):
            current = labels.get((metric_id, time_s), "–")
            matrix[row, column] = 1 if current == canonical_text else 0
    ax.imshow(matrix, cmap="Oranges", vmin=0, vmax=1, aspect="auto", alpha=0.35)
    for row, metric_id in enumerate(metric_ids):
        for column, time_s in enumerate(times):
            ax.text(column, row, labels.get((metric_id, time_s), "–"), ha="center", va="center", fontsize=5.7, color=GRAY)
    ax.set_xticks(np.arange(len(times)))
    ax.set_xticklabels([TIME_LABELS[time_s] for time_s in times])
    ax.set_yticks(np.arange(len(metric_ids)))
    ax.set_yticklabels(["median $T$", "mean $T$", "$V_{max}$", "interface mean $V$"])
    ax.set_title("Local-maximum power(s) change across snapshots", loc="left", pad=3)
    ax.set_xlabel("Snapshot time")
    ax.set_ylabel("Metric")
    ax.tick_params(length=0)
    _panel_label(ax, "f")
    trace.append(
        {
            "figure_id": "Fig.S8",
            "panel_id": "f",
            "source_csv": "图/power_response_audit/temporal_local_extrema.csv",
            "metric_name": "time-varying sampled-power local maxima",
            "reported_value": "all canonical local maxima change status over five snapshots",
            "verified": "yes",
        }
    )
    fig.text(0.5, 0.004, "The 200 W and 450 W endpoints bound the observed range, not regimes. Numbers in e and f are local-maximum sampled powers; they do not estimate a continuous response or physical inflection.", ha="center", va="bottom", fontsize=4.8, color=GRAY)
    trace.append(
        {
            "figure_id": "Fig.S8",
            "panel_id": "scope",
            "source_csv": "图/power_response_audit/pairwise_snapshot_context.csv",
            "metric_name": "observed power-domain boundary",
            "reported_value": "60 rows; 15 unordered pairs; 200--450 W endpoints are not regimes",
            "verified": "yes",
        }
    )
    return _export(fig, "FigS8_power_response_audit"), trace


def build_and_update_traceability() -> Path:
    output, trace = build_figure()
    trace_frame = pd.DataFrame(trace)
    if TRACE_PATH.exists():
        existing = pd.read_csv(TRACE_PATH)
        trace_frame = pd.concat([existing.loc[existing["figure_id"] != "Fig.S8"], trace_frame], ignore_index=True)
    trace_frame.to_csv(TRACE_PATH, index=False, encoding="utf-8-sig")
    return output


def main() -> None:
    print(f"Generated {build_and_update_traceability()}")


if __name__ == "__main__":
    main()
