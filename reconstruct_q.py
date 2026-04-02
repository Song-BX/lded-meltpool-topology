from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from preprocess import project_root


DEFAULT_K_VALUES = [15, 20, 25, 30, 35]


def load_dedup_cases(processed_dir: Path) -> dict[int, pd.DataFrame]:
    cases = {}
    for path in sorted(processed_dir.glob("dedup_*W.csv")):
        power = int(path.stem.split("_")[1].replace("W", ""))
        cases[power] = pd.read_csv(path)
    if not cases:
        raise FileNotFoundError("No deduplicated case files found. Run preprocess.py first.")
    return cases


def weighted_ls_gradient(points: np.ndarray, values: np.ndarray, neighbor_idx: np.ndarray, alpha: float, eps_w: float) -> tuple[np.ndarray, float]:
    x0 = points[neighbor_idx[0]] * 0 + points[neighbor_idx[0]]  # safe shape helper
    A = points[neighbor_idx] - x0
    # x0 above is wrong anchor for arbitrary point; caller passes shifted indices instead.
    raise RuntimeError("weighted_ls_gradient should not be called directly.")


def reconstruct_case(df: pd.DataFrame, k: int, alpha: float = 1.0, eps_w: float = 1e-12, kappa_max: float = 1e12) -> pd.DataFrame:
    pts = df[["x", "y", "z"]].to_numpy()
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)

    dudx = np.full((len(df), 3), np.nan)
    dvdx = np.full((len(df), 3), np.nan)
    dwdx = np.full((len(df), 3), np.nan)
    kappa = np.full(len(df), np.nan)
    chi = np.zeros(len(df), dtype=int)

    uvals = df["u"].to_numpy()
    vvals = df["v"].to_numpy()
    wvals = df["w"].to_numpy()

    for i in range(len(df)):
        nbrs = idx[i, 1:]
        A = pts[nbrs] - pts[i]
        dist = np.linalg.norm(A, axis=1)
        weights = 1.0 / np.power(dist + eps_w, alpha)
        W = np.diag(weights)
        ATA = A.T @ W @ A
        try:
            kappa_i = float(np.linalg.cond(ATA))
        except np.linalg.LinAlgError:
            kappa_i = np.inf
        kappa[i] = kappa_i
        if not np.isfinite(kappa_i) or kappa_i > kappa_max:
            continue

        chi[i] = 1
        WA = np.sqrt(W) @ A
        for vals, target in [(uvals, dudx), (vvals, dvdx), (wvals, dwdx)]:
            b = vals[nbrs] - vals[i]
            Wb = np.sqrt(W) @ b
            grad = np.linalg.pinv(WA) @ Wb
            target[i] = grad

    out = df.copy()
    out["k"] = k
    out["kappa"] = kappa
    out["chi"] = chi
    out[["du_dx", "du_dy", "du_dz"]] = dudx
    out[["dv_dx", "dv_dy", "dv_dz"]] = dvdx
    out[["dw_dx", "dw_dy", "dw_dz"]] = dwdx

    # Tensor decomposition and Q
    S_norm2 = []
    O_norm2 = []
    Q_vals = []
    for row in out.itertuples(index=False):
        if row.chi != 1:
            S_norm2.append(np.nan)
            O_norm2.append(np.nan)
            Q_vals.append(np.nan)
            continue
        grad_u = np.array([
            [row.du_dx, row.du_dy, row.du_dz],
            [row.dv_dx, row.dv_dy, row.dv_dz],
            [row.dw_dx, row.dw_dy, row.dw_dz],
        ], dtype=float)
        S = 0.5 * (grad_u + grad_u.T)
        O = 0.5 * (grad_u - grad_u.T)
        s2 = float(np.trace(S.T @ S))
        o2 = float(np.trace(O.T @ O))
        q = 0.5 * (o2 - s2)
        S_norm2.append(s2)
        O_norm2.append(o2)
        Q_vals.append(q)

    out["S_norm2"] = S_norm2
    out["Omega_norm2"] = O_norm2
    out["Q"] = Q_vals
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct velocity gradients and compute Q.")
    parser.add_argument("--processed-dir", type=Path, default=project_root() / "data" / "processed")
    parser.add_argument("--k-values", type=int, nargs="*", default=DEFAULT_K_VALUES)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eps-w", type=float, default=1e-12)
    parser.add_argument("--kappa-max", type=float, default=1e12)
    args = parser.parse_args()

    cases = load_dedup_cases(args.processed_dir)
    for power, df in cases.items():
        for k in args.k_values:
            recon = reconstruct_case(df, k=k, alpha=args.alpha, eps_w=args.eps_w, kappa_max=args.kappa_max)
            recon.to_csv(args.processed_dir / f"reconstructed_{power}W_k{k}.csv", index=False)
            valid = int((recon["chi"] == 1).sum())
            print(f"Reconstructed {power} W, k={k}: valid={valid}/{len(recon)}")


if __name__ == "__main__":
    main()
