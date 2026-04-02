from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from preprocess import project_root

K_REF = 25
K_VALUES = [15, 20, 25, 30, 35]
THRESHOLD_LABELS = ["Q>0", "Q>posP50", "Q>posP75", "Q>posP90"]
SLICE_Y_MAX = 2e-5
F_TH = 0.99
H_TH = 0.0


def load_reconstructed(processed_dir: Path, power: int, k: int) -> pd.DataFrame:
    path = processed_dir / f"reconstructed_{power}W_k{k}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing reconstructed file: {path}")
    return pd.read_csv(path)


def region_mask(df: pd.DataFrame, region: str) -> pd.Series:
    base = df["chi"] == 1
    if region == "R_all":
        return base
    if region == "R_int":
        return base & (df["fof"] < F_TH)
    if region == "R_heat":
        return base & (df["heat_flux"] > H_TH)
    if region == "R_(int∩heat)":
        return base & (df["fof"] < F_TH) & (df["heat_flux"] > H_TH)
    raise ValueError(f"Unknown region: {region}")


def slice_mask(df: pd.DataFrame, y_max: float = SLICE_Y_MAX) -> pd.Series:
    return (df["chi"] == 1) & (df["y"] <= y_max)


def positive_q_fraction(df: pd.DataFrame, region: str) -> float:
    mask = region_mask(df, region)
    sub = df.loc[mask, "Q"].dropna()
    return float((sub > 0).mean()) if len(sub) else np.nan


def compute_scalar_metrics(processed_dir: Path, powers: Iterable[int]) -> pd.DataFrame:
    rows = []
    for p in powers:
        df = load_reconstructed(processed_dir, p, K_REF)
        all_mask = region_mask(df, "R_all")
        int_heat = region_mask(df, "R_(int∩heat)")
        rows.append(
            {
                "power_W": p,
                "velocity_max_all": float(df.loc[all_mask, "V"].max()),
                "velocity_mean_int_heat": float(df.loc[int_heat, "V"].mean()) if int_heat.any() else np.nan,
                "temperature_max_all": float(df.loc[all_mask, "T"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("power_W")


def compute_q_fraction_metrics(processed_dir: Path, powers: Iterable[int]) -> pd.DataFrame:
    rows = []
    for p in powers:
        df = load_reconstructed(processed_dir, p, K_REF)
        rows.append(
            {
                "power_W": p,
                "region": "R_all",
                "positive_q_fraction": positive_q_fraction(df, "R_all"),
            }
        )
        rows.append(
            {
                "power_W": p,
                "region": "R_int",
                "positive_q_fraction": positive_q_fraction(df, "R_int"),
            }
        )
    return pd.DataFrame(rows)


def top_m_set(df: pd.DataFrame, value_col: str, m: int, require_positive_q: bool = False) -> pd.DataFrame:
    sub = df.loc[slice_mask(df)].copy()
    if require_positive_q:
        sub = sub.loc[sub["Q"] > 0].copy()
    sub = sub.sort_values(value_col, ascending=False).head(m)
    return sub


def geometry_metrics(sub: pd.DataFrame) -> dict:
    coords = sub[["x", "z"]].to_numpy()
    if len(coords) == 0:
        return {"rms_radius": np.nan, "span_x": np.nan, "span_z": np.nan}
    centroid = coords.mean(axis=0)
    rms = float(np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1))))
    span_x = float(sub["x"].max() - sub["x"].min())
    span_z = float(sub["z"].max() - sub["z"].min())
    return {"rms_radius": rms, "span_x": span_x, "span_z": span_z}


def compute_extreme_geometry(processed_dir: Path, powers: Iterable[int] = (350, 400)) -> pd.DataFrame:
    rows = []
    definitions = [
        ("Qpos_top10", "Q", 10, True),
        ("Qpos_top5", "Q", 5, True),
        ("Vmag_top10", "V", 10, False),
        ("Vmag_top5", "V", 5, False),
    ]
    for p in powers:
        df = load_reconstructed(processed_dir, p, K_REF)
        for label, col, m, positive_only in definitions:
            sub = top_m_set(df, col, m, require_positive_q=positive_only)
            geom = geometry_metrics(sub)
            rows.append({"power_W": p, "set_label": label, **geom})
    return pd.DataFrame(rows)


def threshold_for_label(qvals: pd.Series, label: str) -> float:
    if label == "Q>0":
        return 0.0
    pos = qvals[qvals > 0]
    if len(pos) == 0:
        return np.nan
    if label == "Q>posP50":
        return float(pos.quantile(0.50))
    if label == "Q>posP75":
        return float(pos.quantile(0.75))
    if label == "Q>posP90":
        return float(pos.quantile(0.90))
    raise ValueError(label)


def phi_metric(df: pd.DataFrame, region: str, label: str) -> float:
    mask = region_mask(df, region)
    sub = df.loc[mask, "Q"].dropna()
    if len(sub) == 0:
        return np.nan
    theta = threshold_for_label(sub, label)
    if np.isnan(theta):
        return np.nan
    return float((sub > theta).mean())


def compute_heatmaps_and_table(processed_dir: Path) -> tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    heatmaps: Dict[str, pd.DataFrame] = {}
    regions_for_heatmap = ["R_all", "R_int"]
    table_regions = ["R_all", "R_heat", "R_int", "R_(int∩heat)"]
    table_rows = []

    for region in regions_for_heatmap:
        mat = pd.DataFrame(index=THRESHOLD_LABELS, columns=K_VALUES, dtype=float)
        for k in K_VALUES:
            df350 = load_reconstructed(processed_dir, 350, k)
            df400 = load_reconstructed(processed_dir, 400, k)
            for label in THRESHOLD_LABELS:
                phi350 = phi_metric(df350, region, label)
                phi400 = phi_metric(df400, region, label)
                mat.loc[label, k] = phi350 - phi400
        heatmaps[region] = mat

    for region in table_regions:
        for label in THRESHOLD_LABELS:
            success = 0
            total = 0
            for k in K_VALUES:
                df350 = load_reconstructed(processed_dir, 350, k)
                df400 = load_reconstructed(processed_dir, 400, k)
                phi350 = phi_metric(df350, region, label)
                phi400 = phi_metric(df400, region, label)
                if np.isnan(phi350) or np.isnan(phi400):
                    continue
                total += 1
                if (phi350 - phi400) > 0:
                    success += 1
            table_rows.append(
                {
                    "region": region,
                    "threshold": label,
                    "success": success,
                    "total": total,
                    "success_rate": success / total if total else np.nan,
                }
            )
    return heatmaps, pd.DataFrame(table_rows)


def save_outputs(processed_dir: Path) -> None:
    cases_df = pd.read_csv(processed_dir / "cases.csv")
    powers = cases_df["power_W"].tolist()

    scalar = compute_scalar_metrics(processed_dir, powers)
    scalar.to_csv(processed_dir / "scalar_metrics.csv", index=False)

    qfrac = compute_q_fraction_metrics(processed_dir, powers)
    qfrac.to_csv(processed_dir / "q_fraction_metrics.csv", index=False)

    geom = compute_extreme_geometry(processed_dir)
    geom.to_csv(processed_dir / "extreme_geometry_metrics.csv", index=False)

    heatmaps, table1 = compute_heatmaps_and_table(processed_dir)
    heatmaps["R_all"].to_csv(processed_dir / "heatmap_R_all.csv")
    heatmaps["R_int"].to_csv(processed_dir / "heatmap_R_int.csv")
    table1.to_csv(processed_dir / "table1_success_rates.csv", index=False)

    print(f"Saved statistics to {processed_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute statistics used in the paper figures and table.")
    parser.add_argument("--processed-dir", type=Path, default=project_root() / "data" / "processed")
    args = parser.parse_args()
    save_outputs(args.processed_dir)


if __name__ == "__main__":
    main()
