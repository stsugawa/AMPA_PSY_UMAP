#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replicate MATLAB run_WB2_random_perm.m in Python using umap-learn.

- reduction_with_labels.csv に (id, umap1, umap2, label, perm_id) を保存
- 図 (png) を:
    * 真のラベル → 1枚保存
    * permutation → 10回ごとに保存
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt


def fill_missing_nearest(df: pd.DataFrame) -> pd.DataFrame:
    """Column-wise nearest interpolation (like MATLAB fillmissing(...,'nearest')).
    If edge NaNs remain, fill with column mean."""
    out = df.copy()
    for col in out.columns:
        s = out[col]
        s = s.interpolate(method="nearest", limit_direction="both")
        if s.isna().any():
            s = s.fillna(s.mean())
        out[col] = s
    return out


def prepare_data(path_csv: str) -> (np.ndarray, np.ndarray):
    """
    Match MATLAB slicing:
      Rows:
        ASD: 1:35   -> 0:35
        BIP: 36:72  -> 35:72
        DEP: 73:107 -> 72:107
        HC :108:177 -> 107:177
        SCH:178:219 -> 177:219 (end-exclusive)
      Cols: 8:82 (inclusive in MATLAB) -> iloc[:, 7:82] (0-based, end-exclusive)
    Returns:
      X: (N, D) float64 normalized (z-score per column)
      y_true: (N,) int labels in {1,2,3,4,5}
    """
    M = pd.read_csv(path_csv)
    c0, c1 = 7, 82  # columns

    # row blocks
    r_asd = (0, 35)
    r_bip = (35, 72)
    r_dep = (72, 107)
    r_hc = (107, 177)
    r_sch = (177, 219)

    WB_ASD = M.iloc[r_asd[0]:r_asd[1], c0:c1]
    WB_BIP = M.iloc[r_bip[0]:r_bip[1], c0:c1]
    WB_DEP = M.iloc[r_dep[0]:r_dep[1], c0:c1]
    WB_HC = M.iloc[r_hc[0]:r_hc[1], c0:c1]
    WB_SCH = M.iloc[r_sch[0]:r_sch[1], c0:c1]

    # fill missing
    WB_ASD = fill_missing_nearest(WB_ASD)
    WB_BIP = fill_missing_nearest(WB_BIP)
    WB_DEP = fill_missing_nearest(WB_DEP)
    WB_HC = fill_missing_nearest(WB_HC)
    WB_SCH = fill_missing_nearest(WB_SCH)

    exactD_table = pd.concat([WB_ASD, WB_BIP, WB_DEP, WB_HC, WB_SCH],
                             axis=0, ignore_index=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(exactD_table.values)

    y_true = np.concatenate([
        np.full(len(WB_ASD), 0, dtype=int),
        np.full(len(WB_BIP), 1, dtype=int),
        np.full(len(WB_DEP), 2, dtype=int),
        np.full(len(WB_HC), 3, dtype=int),
        np.full(len(WB_SCH), 4, dtype=int),
    ]) + 1

    return X, y_true


def run_umap_supervised(X: np.ndarray, y: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Supervised UMAP to mirror MATLAB run_umap with target_weight=0.005 etc."""
    reducer = umap.UMAP(
        n_neighbors=50,
        min_dist=1.0,
        n_components=2,
        spread=10.0,
        target_weight=0.005,
        target_metric="categorical",
        random_state=random_state,
        verbose=False,
    )
    emb = reducer.fit_transform(X, y=y)
    return emb


def run_umap_unsupervised(X: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Unsupervised UMAP (no labels). Uses same geometry params as supervised."""
    reducer = umap.UMAP(
        n_neighbors=50,
        min_dist=1.0,
        n_components=2,
        spread=10.0,
        random_state=random_state,
        verbose=False,
    )
    emb = reducer.fit_transform(X)
    return emb


def save_umap_plot(embedding: np.ndarray, labels: np.ndarray,
                   title: str, out_path: str):
    """Save UMAP scatter plot."""
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                          c=labels, cmap="tab10", s=15)
    plt.colorbar(scatter, label="Label")
    plt.title(title)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default="./data/HCPSY219_WBSUVR_Dataset.csv")
    parser.add_argument("--nperm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_root", type=str, default="./result")
    args = parser.parse_args()

    currentDate = datetime.now().strftime("%Y%m%d")
    saveDir = os.path.join(args.out_root, currentDate,
                           "normalizedWB_label_by_random", "2d")
    os.makedirs(saveDir, exist_ok=True)

    X, y_true = prepare_data(args.data)
    N = X.shape[0]
    rng = np.random.default_rng(args.seed)

    # ---- True labels
    reduction_true = run_umap_supervised(X, y_true, random_state=args.seed)
    df_true = pd.DataFrame({
        "id": np.arange(1, N + 1),
        "umap1": reduction_true[:, 0],
        "umap2": reduction_true[:, 1],
        "label": y_true,
        "perm_id": 0,
    })

    # ---- Unsupervised (no-label) embedding
    reduction_nolabel = run_umap_unsupervised(X, random_state=args.seed)
    df_nolabel = pd.DataFrame({
        "id": np.arange(1, N + 1),
        "umap1": reduction_nolabel[:, 0],
        "umap2": reduction_nolabel[:, 1],
        "label": y_true,    # reference labels only; NOT used in training
        "perm_id": -1       # denote no-label run with -1
    })
    # Save no-label outputs
    out_csv_nl = os.path.join(saveDir, "reduction_no_label.csv")
    df_nolabel.to_csv(out_csv_nl, index=False, encoding="utf-8")
    save_umap_plot(reduction_nolabel, y_true,
                   "UMAP (no-label / unsupervised)",
                   os.path.join(saveDir, "umap_no_label.png"))

    # Save true labels figure
    save_umap_plot(reduction_true, y_true,
                   "UMAP (true labels)",
                   os.path.join(saveDir, "umap_true_labels.png"))

    rows = [df_true]

    # ---- Permutations
    for b in range(1, args.nperm + 1):
        perm_idx = rng.permutation(N)
        y_perm = y_true[perm_idx]
        reduction_perm = run_umap_supervised(X, y_perm, random_state=args.seed)

        rows.append(pd.DataFrame({
            "id": np.arange(1, N + 1),
            "umap1": reduction_perm[:, 0],
            "umap2": reduction_perm[:, 1],
            "label": y_perm,
            "perm_id": b,
        }))

        # Save every 10th permutation figure
        if b % 10 == 0:
            save_umap_plot(reduction_perm, y_perm,
                           f"UMAP (perm {b})",
                           os.path.join(saveDir, f"umap_perm_{b:04d}.png"))

    # ---- Save all results
    all_results = pd.concat(rows, axis=0, ignore_index=True)
    out_csv = os.path.join(saveDir, "reduction_with_labels.csv")
    all_results.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved CSV: {out_csv}")
    print(f"Figures saved in: {saveDir}")


if __name__ == "__main__":
    main()