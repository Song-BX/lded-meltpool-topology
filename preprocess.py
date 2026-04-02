from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

COLUMN_MAP = {
    "Points_0": "x",
    "Points_1": "y",
    "Points_2": "z",
    "Fraction Of Fluid": "fof",
    "Heat Flux Spatial Distribution": "heat_flux",
    "Temperature": "T",
    "Temperature Gradient At Tgrdout": "gradT",
    "Velocity_0": "u",
    "Velocity_1": "v",
    "Velocity_2": "w",
    "Velocity_Magnitude": "V",
}


def project_root() -> Path:
    return Path(__file__).resolve().parent


def find_case_files(raw_dir: Path) -> Dict[int, Path]:
    """Find CSV files and map them to integer power levels in watts."""
    files = sorted(raw_dir.glob("*.csv"))
    case_files: Dict[int, Path] = {}
    for path in files:
        m = re.search(r"(\d+)W", path.stem)
        if m:
            case_files[int(m.group(1))] = path
    return dict(sorted(case_files.items()))


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].copy()
    return out


def deduplicate_points(df: pd.DataFrame, eps_c: float = 1e-9) -> pd.DataFrame:
    coords = df[["x", "y", "z"]].to_numpy()
    key = np.round(coords / eps_c).astype(np.int64)
    work = df.copy()
    work[["kx", "ky", "kz"]] = key
    grouped = work.groupby(["kx", "ky", "kz"], sort=False)
    dedup = grouped[["x", "y", "z", "fof", "heat_flux", "T", "gradT", "u", "v", "w", "V"]].mean().reset_index(drop=True)
    return dedup


def preprocess_case(file_path: Path, eps_c: float) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(file_path)
    std = standardize_columns(raw)
    dedup = deduplicate_points(std, eps_c=eps_c)
    meta = {
        "raw_points": int(len(std)),
        "unique_points": int(len(dedup)),
        "dedup_ratio": float((len(std) - len(dedup)) / len(std)),
        "duplication_factor": float(len(std) / len(dedup)),
        "x_min": float(dedup["x"].min()),
        "x_max": float(dedup["x"].max()),
        "y_min": float(dedup["y"].min()),
        "y_max": float(dedup["y"].max()),
        "z_min": float(dedup["z"].min()),
        "z_max": float(dedup["z"].max()),
    }
    return dedup, meta


def build_cases_table(case_files: Dict[int, Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"power_W": power, "filename": path.name} for power, path in case_files.items()]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate raw Flow3D point-cloud data.")
    parser.add_argument("--raw-dir", type=Path, default=project_root() / "data" / "raw")
    parser.add_argument("--processed-dir", type=Path, default=project_root() / "data" / "processed")
    parser.add_argument("--eps-c", type=float, default=1e-9, help="Coordinate tolerance for deduplication.")
    args = parser.parse_args()

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    case_files = find_case_files(args.raw_dir)
    if not case_files:
        raise FileNotFoundError(f"No raw CSV files found in {args.raw_dir}")

    cases_df = build_cases_table(case_files)
    cases_df.to_csv(args.processed_dir / "cases.csv", index=False)

    summaries: List[dict] = []
    for power, file_path in case_files.items():
        dedup, meta = preprocess_case(file_path, eps_c=args.eps_c)
        dedup.to_csv(args.processed_dir / f"dedup_{power}W.csv", index=False)
        meta["power_W"] = power
        summaries.append(meta)
        print(f"Processed {power} W: raw={meta['raw_points']} unique={meta['unique_points']}")

    summary_df = pd.DataFrame(summaries).sort_values("power_W")
    summary_df.to_csv(args.processed_dir / "point_count_summary.csv", index=False)
    print(f"Saved processed files to: {args.processed_dir}")


if __name__ == "__main__":
    main()
