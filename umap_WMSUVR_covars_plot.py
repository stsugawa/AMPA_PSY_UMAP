#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge UMAP coordinates with covariates and plot (no mean SUVR recomputation).

What this script does
---------------------
1) Loads UMAP result: reduction_with_labels.csv produced by umap_WMSUVR_fig.py
2) Loads covariates CSV (which already contains mean SUVR in the *last* column)
   e.g. /Users/sakiko/Dropbox/研究/22_AMPA_MultiVariateAnalysis/UMAP/data/HCPSY219_WMSUVR_Dataset_withMaskMean.csv
3) Renames the last column to `meanSUVR` and keeps {MID, meanSUVR, age, sex, site}
4) Merges on MID and saves to:
   {out_root}/{date}/wmsuvr_label_by_random/2d/reduction_no_label_with_covars.csv
5) Makes simple UMAP scatter plots colored by meanSUVR, age, sex, site (if present).

Usage example
-------------
python umap_WMSUVR_covars_plot.py \
  --out_root /path/to/out_root \
  --date 20250101 \
  --emb_dirname wmsuvr_label_by_random/2d \
  --covars_csv "/Users/sakiko/Dropbox/研究/22_AMPA_MultiVariateAnalysis/UMAP/data/HCPSY219_WMSUVR_Dataset_withMaskMean.csv"

Notes
-----
- This script does NOT compute mean SUVR from features. It uses the last column in the
  covariates CSV (column name typically 'MaskMeanSUVR_output01') and renames it to 'meanSUVR'.
"""

import os
import argparse
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_umap_csv(emb_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(emb_csv_path):
        raise FileNotFoundError(f"UMAP CSV not found: {emb_csv_path}")
    df = pd.read_csv(emb_csv_path)
    # Expected columns: id, MID, group, umap1, umap2, (possibly others)
    # Ensure UMAP coords exist
    for c in ("umap1", "umap2"):
        if c not in df.columns:
            raise ValueError(f"UMAP CSV missing column: {c}. Columns found: {list(df.columns)}")
    # Keep 'id' if present (1-based row index from generation), 'MID' may or may not be present
    return df


def load_covars_csv(covars_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(covars_csv_path):
        raise FileNotFoundError(f"Covariates CSV not found: {covars_csv_path}")
    covar_df = pd.read_csv(covars_csv_path)
    # Last column is mean SUVR
    last_col = covar_df.columns[-1]
    covar_df = covar_df.rename(columns={last_col: "meanSUVR"})
    # Keep only relevant columns if available
    keep_cols = ["MID", "meanSUVR", "age", "sex", "site"]
    present = [c for c in keep_cols if c in covar_df.columns]
    if "MID" not in present:
        raise ValueError(f"'MID' column not found in covariates CSV: {covars_csv_path}")
    covar_df = covar_df[present].copy()
    return covar_df


def plot_continuous(emb: np.ndarray, values: np.ndarray, title: str, out_base: str):
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=values, s=15)
    cb = plt.colorbar(sc)
    cb.set_label(title)
    plt.title(title)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(out_base + ".png", dpi=300)
    plt.savefig(out_base + ".pdf")
    plt.close()


def plot_categorical(emb: np.ndarray, values: pd.Series, title: str, out_base: str):
    cat = pd.Categorical(values.astype(str))
    codes = cat.codes  # -1 for NaN in pandas < 2.1; here NaN converted to 'nan' so safe
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=codes, cmap="tab10", s=15)
    cb = plt.colorbar(sc, ticks=range(len(cat.categories)))
    cb.ax.set_yticklabels(list(cat.categories))
    cb.set_label(title)
    plt.title(title)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(out_base + ".png", dpi=300)
    plt.savefig(out_base + ".pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Merge UMAP coords with covariates (meanSUVR, age, sex, site) and plot.")
    parser.add_argument("--out_root", required=True, help="Root output directory used by umap_WMSUVR_fig.py")
    parser.add_argument("--date", required=True, help="Date folder name under out_root (e.g., 20250101)")
    parser.add_argument("--emb_dirname", default="wmsuvr_label_by_random/2d", help="Subdir under date containing reduction_with_labels.csv")
    parser.add_argument("--umap_csv", default="reduction_with_labels.csv", help="UMAP CSV filename inside emb_dirname")
    parser.add_argument("--covars_csv", required=True, help="Covariates CSV path (last column is meanSUVR)")

    args = parser.parse_args()

    emb_dir = os.path.join(args.out_root, args.date, args.emb_dirname)
    emb_csv_path = os.path.join(emb_dir, args.umap_csv)
    out_dir = emb_dir  # save alongside UMAP outputs
    ensure_dir(out_dir)

    # Load data
    df_umap = load_umap_csv(emb_csv_path)
    df_cov = load_covars_csv(args.covars_csv)

    # Merge
    if "MID" in df_umap.columns and "MID" in df_cov.columns:
        merged = df_umap.merge(df_cov, on="MID", how="left")
    elif "id" in df_umap.columns:
        # Align by 1-based row index: WMSUVR_dig.py generated 'id' as 1..N matching the original dataset order.
        df_cov = df_cov.copy()
        df_cov["_row_id"] = range(1, len(df_cov) + 1)
        merged = df_umap.merge(df_cov, left_on="id", right_on="_row_id", how="left").drop(columns=["_row_id"])
        # If UMAP lacked MID, ensure it exists after merge
        if "MID" not in merged.columns and "MID" in df_cov.columns:
            # MID now comes from covariates part of the merge
            pass
    else:
        raise ValueError("Cannot merge: neither 'MID' nor 'id' present in UMAP CSV.")

    # Save merged CSV
    out_csv = os.path.join(out_dir, "reduction_no_label_with_covars.csv")
    merged.to_csv(out_csv, index=False, encoding="utf-8")
    print("Saved merged CSV:", out_csv)

    # Plots (only if columns exist)
    emb = merged[["umap1", "umap2"]].to_numpy()

    if "meanSUVR" in merged.columns and merged["meanSUVR"].notna().any():
        plot_continuous(emb, merged["meanSUVR"].values, "meanSUVR", os.path.join(out_dir, "umap_colored_by_meanSUVR"))
    if "age" in merged.columns and merged["age"].notna().any():
        plot_continuous(emb, merged["age"].values, "age", os.path.join(out_dir, "umap_colored_by_age"))
    if "sex" in merged.columns and merged["sex"].notna().any():
        plot_categorical(emb, merged["sex"], "sex", os.path.join(out_dir, "umap_colored_by_sex"))
    if "site" in merged.columns and merged["site"].notna().any():
        plot_categorical(emb, merged["site"], "site", os.path.join(out_dir, "umap_colored_by_site"))

    print("Figures saved to:", out_dir)


if __name__ == "__main__":
    main()
