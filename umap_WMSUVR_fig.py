#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMAP pipeline for white-matter–corrected SUVR (WMSUVR) with optional age/sex residualization.

Inputs:
  - CSV with columns at least: MID, group, age, sex, and multiple region columns (e.g., 'ManualWM_*').
    Example: HCPSY219_WMSUVR_Dataset.csv

Outputs (under --out_root/YYYYMMDD/wmsuvr_label_by_random/2d):
  - reduction_with_labels.csv   : id, umap1, umap2, label(1..K), perm_id (0=true, 1..B=permutation)
  - reduction_no_label.csv      : same columns with perm_id=-1 (unsupervised UMAP)
  - umap_true_labels.png        : scatter colored by true labels
  - umap_perm_XXXX.png          : every 10th permutation scatter
  - umap_no_label.png           : unsupervised UMAP scatter

Usage:
  python umap_WMSUVR_fig.py --data /path/to/HCPSY219_WMSUVR_Dataset.csv --nperm 1000 --residualize 1
"""

import argparse
import os
from datetime import datetime
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import umap
import matplotlib.pyplot as plt


GROUP_ORDER = ["ASD", "BIP", "DEP", "HC", "SCH"]  # map to 1..5


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Pick regional columns. Prefer 'ManualWM_*'; otherwise, all numeric except meta."""
    meta = {"MID", "site", "group", "site_number", "group_number", "age", "sex"}
    manual_cols = [c for c in df.columns if c.startswith("ManualWM_")]
    if manual_cols:
        return manual_cols
    # fallback: numeric columns not in meta
    num_cols = [c for c in df.columns if c not in meta and np.issubdtype(df[c].dtype, np.number)]
    return num_cols


def residualize_by_age_sex(X_df: pd.DataFrame, age: np.ndarray, sex: np.ndarray) -> pd.DataFrame:
    """Per-feature OLS: feature ~ 1 + age + sex; return residuals."""
    model = LinearRegression()
    A = np.column_stack([age.astype(float), sex.astype(float)])
    res = {}
    for col in X_df.columns:
        y = X_df[col].values.astype(float)
        model.fit(A, y)
        y_hat = model.predict(A)
        res[col] = y - y_hat
    return pd.DataFrame(res, index=X_df.index)


def encode_labels(groups: pd.Series) -> np.ndarray:
    """Map group strings to 1..K using GROUP_ORDER; unseen labels get increasing codes."""
    order = list(GROUP_ORDER)
    unseen = [g for g in groups.unique().tolist() if g not in order]
    order += unseen
    mapping = {g: i+1 for i, g in enumerate(order)}
    return groups.map(mapping).astype(int).values


def run_umap_supervised(X: np.ndarray, y: np.ndarray, random_state: int = 0) -> np.ndarray:
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
    return reducer.fit_transform(X, y=y)


def run_umap_unsupervised(X: np.ndarray, random_state: int = 0) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=50,
        min_dist=1.0,
        n_components=2,
        spread=10.0,
        random_state=random_state,
        verbose=False,
    )
    return reducer.fit_transform(X)


def save_umap_plot(embedding: np.ndarray, labels: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=15)
    plt.colorbar(scatter, label="Label")
    plt.title(title)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def prepare_matrix(
    df: pd.DataFrame,
    residualize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create X (N, D) and integer labels y without imputation."""

    if len(df) != EXPECTED_N:
        raise ValueError(
            f"Input dataset contains {len(df)} participants; "
            f"expected {EXPECTED_N}."
        )

    feat_cols = select_feature_columns(df)

    if len(feat_cols) != EXPECTED_D:
        raise ValueError(
            f"Detected {len(feat_cols)} regional feature columns; "
            f"expected {EXPECTED_D}."
        )

    X_raw = require_complete_numeric(
        df[feat_cols].copy(),
        name="SUVR-WM feature matrix",
        expected_shape=(EXPECTED_N, EXPECTED_D),
    )

    if df["group"].isna().any():
        raise ValueError(
            "Diagnostic group labels contain missing values."
        )

    if residualize:
        covars = require_complete_numeric(
            df[["age", "sex"]].copy(),
            name="Age/sex covariates",
            expected_shape=(EXPECTED_N, 2),
        )

        X_adj = residualize_by_age_sex(
            X_raw,
            covars["age"].to_numpy(dtype=float),
            covars["sex"].to_numpy(dtype=float),
        )
    else:
        X_adj = X_raw

    scaler = StandardScaler()
    X = scaler.fit_transform(
        X_adj.to_numpy(dtype=float)
    )

    y_true = encode_labels(df["group"])

    return X, y_true

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./HCPSY219_WMSUVR_Dataset.csv")
    parser.add_argument("--nperm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_root", type=str, default="./result")
    parser.add_argument("--residualize", type=int, default=0, help="1: use age/sex residuals, 0: raw values")
    args = parser.parse_args()

    date_str = datetime.now().strftime("%Y%m%d")
    label_mode = "resid" if bool(args.residualize) else "raw"
    save_dir = os.path.join(args.out_root, date_str, f"wmsuvr_{label_mode}_label_by_random", "2d")
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data)
    X, y_true = prepare_matrix(df, residualize=bool(args.residualize))
    N = X.shape[0]
    rng = np.random.default_rng(args.seed)

    # ---- True labels (supervised UMAP)
    emb_true = run_umap_supervised(X, y_true, random_state=args.seed)
    df_true = pd.DataFrame({
        "id": np.arange(1, N+1),
        "umap1": emb_true[:, 0],
        "umap2": emb_true[:, 1],
        "label": y_true,
        "perm_id": 0,
    })
    save_umap_plot(emb_true, y_true, "UMAP (true labels)", os.path.join(save_dir, "umap_true_labels.png"))

    # ---- Unsupervised (no-label) UMAP
    emb_nl = run_umap_unsupervised(X, random_state=args.seed)
    df_nl = pd.DataFrame({
        "id": np.arange(1, N+1),
        "umap1": emb_nl[:, 0],
        "umap2": emb_nl[:, 1],
        "label": y_true,   # reference only
        "perm_id": -1,
    })
    df_nl.to_csv(os.path.join(save_dir, "reduction_no_label.csv"), index=False, encoding="utf-8")
    save_umap_plot(emb_nl, y_true, "UMAP (no-label / unsupervised)", os.path.join(save_dir, "umap_no_label.png"))

    # ---- Permutations (labels randomized)
    rows = [df_true]
    for b in range(1, args.nperm + 1):
        y_perm = rng.permutation(y_true)
        emb_perm = run_umap_supervised(X, y_perm, random_state=args.seed)
        rows.append(pd.DataFrame({
            "id": np.arange(1, N+1),
            "umap1": emb_perm[:, 0],
            "umap2": emb_perm[:, 1],
            "label": y_perm,
            "perm_id": b,
        }))
        if b % 10 == 0:
            save_umap_plot(emb_perm, y_perm, f"UMAP (perm {b})", os.path.join(save_dir, f"umap_perm_{b:04d}.png"))

    all_results = pd.concat(rows, axis=0, ignore_index=True)
    out_csv = os.path.join(save_dir, "reduction_with_labels.csv")
    all_results.to_csv(out_csv, index=False, encoding="utf-8")

    print("Saved CSV:", out_csv)
    print("Saved no-label CSV:", os.path.join(save_dir, "reduction_no_label.csv"))
    print("Figures saved in:", save_dir)


if __name__ == "__main__":
    main()
