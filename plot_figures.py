from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess import project_root


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure2(processed_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(processed_dir / "point_count_summary.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    x = df["power_W"].to_numpy()
    ax.plot(x, df["raw_points"], marker="o", label="Raw points")
    ax.plot(x, df["unique_points"], marker="s", label="Unique points")
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Count")
    ax.set_title("Figure 2")
    ax.legend()
    savefig(fig, out_dir / "figure2_counts.png")


def plot_figure3(processed_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(processed_dir / "scalar_metrics.csv")
    x = df["power_W"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, df["velocity_max_all"], marker="o", label="Velocity max (all)")
    ax.plot(x, df["velocity_mean_int_heat"], marker="s", label="Velocity mean (interface-heated)")
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Figure 3a")
    ax.legend()
    savefig(fig, out_dir / "figure3a_velocity.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, df["temperature_max_all"], marker="o", label="Max temperature")
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Figure 3b")
    ax.legend()
    savefig(fig, out_dir / "figure3b_temperature.png")


def plot_figure4(processed_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(processed_dir / "q_fraction_metrics.csv")
    pivot = df.pivot(index="power_W", columns="region", values="positive_q_fraction")
    fig, ax = plt.subplots(figsize=(6, 4))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", label=col)
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Positive-Q fraction")
    ax.set_title("Figure 4")
    ax.legend()
    savefig(fig, out_dir / "figure4_q_fraction.png")


def plot_figure5(processed_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, power in zip(axes, [350, 400]):
        df = pd.read_csv(processed_dir / f"reconstructed_{power}W_k25.csv")
        sub = df[(df["chi"] == 1) & (df["y"] <= 2e-5)].copy()
        sc = ax.scatter(sub["x"] * 1e3, sub["z"] * 1e3, c=sub["Q"], s=18)
        ax.set_title(f"{power} W")
        ax.set_xlabel("x (mm)")
    axes[0].set_ylabel("z (mm)")
    fig.colorbar(sc, ax=axes, label="Q (s$^{-2}$)")
    savefig(fig, out_dir / "figure5_q_slice.png")


def plot_figure6(processed_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(processed_dir / "extreme_geometry_metrics.csv")
    labels = ["Qpos_top10", "Qpos_top5", "Vmag_top10", "Vmag_top5"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    width = 0.35
    for i, power in enumerate([350, 400]):
        sub = df[df["power_W"] == power].set_index("set_label").loc[labels]
        ax.bar(x + (i - 0.5) * width, sub["rms_radius"], width=width, label=f"{power} W")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("RMS radius (m)")
    ax.set_title("Figure 6a")
    ax.legend()
    savefig(fig, out_dir / "figure6a_rms_radius.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    for power in [350, 400]:
        sub = df[df["power_W"] == power]
        ax.scatter(sub["span_x"] * 1e3, sub["span_z"] * 1e3, label=f"{power} W")
        for row in sub.itertuples(index=False):
            ax.annotate(f"{row.set_label}_{row.power_W}", (row.span_x * 1e3, row.span_z * 1e3), fontsize=8)
    ax.set_xlabel("Span in x (mm)")
    ax.set_ylabel("Span in z (mm)")
    ax.set_title("Figure 6b")
    ax.legend()
    savefig(fig, out_dir / "figure6b_spans.png")


def plot_figure7(processed_dir: Path, out_dir: Path) -> None:
    for region in ["R_all", "R_int"]:
        df = pd.read_csv(processed_dir / f"heatmap_{region}.csv", index_col=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(df.to_numpy(dtype=float), aspect="auto")
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        ax.set_xlabel("k")
        ax.set_ylabel("Threshold")
        ax.set_title(f"Figure 7 ({region})")
        fig.colorbar(im, ax=ax, label="ΔΦ")
        savefig(fig, out_dir / f"figure7_{region}.png")


def export_table1(processed_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(processed_dir / "table1_success_rates.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "table1_success_rates.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the main figures and table.")
    parser.add_argument("--processed-dir", type=Path, default=project_root() / "data" / "processed")
    parser.add_argument("--figures-dir", type=Path, default=project_root() / "figures")
    parser.add_argument("--tables-dir", type=Path, default=project_root() / "tables")
    args = parser.parse_args()

    plot_figure2(args.processed_dir, args.figures_dir)
    plot_figure3(args.processed_dir, args.figures_dir)
    plot_figure4(args.processed_dir, args.figures_dir)
    plot_figure5(args.processed_dir, args.figures_dir)
    plot_figure6(args.processed_dir, args.figures_dir)
    plot_figure7(args.processed_dir, args.figures_dir)
    export_table1(args.processed_dir, args.tables_dir)
    print(f"Saved figures to {args.figures_dir} and tables to {args.tables_dir}")


if __name__ == "__main__":
    main()
