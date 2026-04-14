#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Permutation tests for UMAP embeddings using:
  - Calinski–Harabasz (CH) [larger = better]
  - Davies–Bouldin (DB)   [smaller = better]
  - Silhouette (SIL)      [larger = better]

Input CSV (wide, long-form):
  reduction_with_labels.csv with columns: id, umap1, umap2, label, perm_id
    * perm_id = 0   -> true labels
    * perm_id = 1..N -> permutation labels

Outputs (in /path/to/perm_test_outputs_YYYYMMDD):
  - permutation_summary.csv   : p-values and true values for CH/DB/SIL
  - metrics_true_vs_random_mean.csv : true values + random mean/std for CH/DB/SIL
  - ch_null.npy, db_null.npy, sil_null.npy
  - figure_CH_null_vs_true.(png|pdf)
  - figure_DB_null_vs_true.(png|pdf)
  - figure_SIL_null_vs_true.(png|pdf)

Usage:
  python perm_test_umap_metrics_with_silhouette.py \
      --input /path/to/reduction_with_labels.csv \
      --outdir /path/to/output_root   # optional
"""

import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def compute_metrics(X: np.ndarray, y: np.ndarray):
    """
    Compute CH, DB, SIL safely.
    Returns (ch, db, sil) as floats or np.nan if not computable.
    """
    # Need >= 2 clusters and not all same label
    unique = np.unique(y)
    if len(unique) < 2:
        return np.nan, np.nan, np.nan

    ch = np.nan
    db = np.nan
    sil = np.nan

    # CH
    try:
        ch = calinski_harabasz_score(X, y)
    except Exception:
        ch = np.nan

    # DB
    try:
        db = davies_bouldin_score(X, y)
    except Exception:
        db = np.nan

    # Silhouette (euclidean)
    try:
        # Must have at least 2 labels and fewer labels than samples
        if len(unique) >= 2 and len(unique) < len(y):
            sil = silhouette_score(X, y, metric="euclidean")
        else:
            sil = np.nan
    except Exception:
        sil = np.nan

    return ch, db, sil


def save_hist_with_true(null_values: np.ndarray, true_value: float,
                        title: str, xlabel: str, out_png: str, out_pdf: str):
    """
    Save histogram of null distribution with a vertical line at the true value.
    One chart per figure, no custom colors or styles.
    """
    valid = ~np.isnan(null_values)
    plt.figure(figsize=(6, 5))
    plt.hist(null_values[valid], bins=40)
    plt.axvline(true_value, color='red')
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./reduction_with_labels.csv")
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    # Output directory with date stamp
    date_str = datetime.now().strftime("%Y%m%d")
    out_dir = os.path.join(args.outdir, f"perm_test_outputs_{date_str}")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input)

    # True set
    df_true = df[df["perm_id"] == 0].copy()
    X_true = df_true[["umap1", "umap2"]].values
    y_true = df_true["label"].values

    # True metrics
    ch_true, db_true, sil_true = compute_metrics(X_true, y_true)

    # Permutations
    Nperm = int(df["perm_id"].max())
    ch_null, db_null, sil_null = [], [], []

    for pid in range(1, Nperm + 1):
        sub = df[df["perm_id"] == pid]
        Xp = sub[["umap1", "umap2"]].values
        yp = sub["label"].values
        ch, db, sil = compute_metrics(Xp, yp)
        ch_null.append(ch)
        db_null.append(db)
        sil_null.append(sil)

    ch_null = np.array(ch_null, dtype=float)
    db_null = np.array(db_null, dtype=float)
    sil_null = np.array(sil_null, dtype=float)

    # Remove NaNs for p-values & summary stats
    v_ch = ~np.isnan(ch_null)
    v_db = ~np.isnan(db_null)
    v_sil = ~np.isnan(sil_null)

    
    # Optional: compute metrics on no-label (unsupervised) embedding if available
    no_label_path = os.path.join(os.path.dirname(args.input), "reduction_no_label.csv")
    no_label_values = {"Calinski-Harabasz": np.nan, "Davies-Bouldin": np.nan, "Silhouette": np.nan}
    if os.path.exists(no_label_path):
        df_nl = pd.read_csv(no_label_path)
        # Use the same true labels to evaluate the unsupervised embedding
        X_nl = df_nl[["umap1", "umap2"]].values
        y_true_for_nl = df_nl["label"].values if "label" in df_nl.columns else df_true["label"].values
        ch_nl, db_nl, sil_nl = compute_metrics(X_nl, y_true_for_nl)
        no_label_values = {
            "Calinski-Harabasz": ch_nl,
            "Davies-Bouldin": db_nl,
            "Silhouette": sil_nl,
        }
    # Permutation p-values (+1 correction)
    # CH and SIL: larger is better -> p = P(null >= true)
    p_ch = (np.sum(ch_null[v_ch] >= ch_true) + 1) / (np.sum(v_ch) + 1)
    p_sil = (np.sum(sil_null[v_sil] >= sil_true) + 1) / (np.sum(v_sil) + 1)
    # DB: smaller is better -> p = P(null <= true)
    p_db = (np.sum(db_null[v_db] <= db_true) + 1) / (np.sum(v_db) + 1)

    # Save numerical results (p-values and true values)
    summary = pd.DataFrame({
        "metric": ["Calinski-Harabasz", "Davies-Bouldin", "Silhouette"],
        "true_value": [ch_true, db_true, sil_true],
        "no_label_value": [no_label_values["Calinski-Harabasz"], no_label_values["Davies-Bouldin"], no_label_values["Silhouette"]],
        "p_value": [p_ch, p_db, p_sil],
        "no_label_value": [no_label_values["Calinski-Harabasz"], no_label_values["Davies-Bouldin"], no_label_values["Silhouette"]],
        "n_perms_used": [int(np.sum(v_ch)), int(np.sum(v_db)), int(np.sum(v_sil))]
    })
    summary_path = os.path.join(out_dir, "permutation_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Save random means (for the paper)
    random_mean = [
        float(np.nanmean(ch_null)),
        float(np.nanmean(db_null)),
        float(np.nanmean(sil_null)),
    ]
    random_std = [
        float(np.nanstd(ch_null, ddof=1)),
        float(np.nanstd(db_null, ddof=1)),
        float(np.nanstd(sil_null, ddof=1)),
    ]
    paper_table = pd.DataFrame({
        "metric": ["Calinski-Harabasz", "Davies-Bouldin", "Silhouette"],
        "true_value": [ch_true, db_true, sil_true],
        "no_label_value": [no_label_values["Calinski-Harabasz"], no_label_values["Davies-Bouldin"], no_label_values["Silhouette"]],
        "random_mean": random_mean,
        "random_std": random_std,
        "n_perms_used": [int(np.sum(v_ch)), int(np.sum(v_db)), int(np.sum(v_sil))]
    })
    paper_csv_path = os.path.join(out_dir, "metrics_true_vs_random_mean.csv")
    paper_table.to_csv(paper_csv_path, index=False)

    # Save raw null arrays for reproducibility
    np.save(os.path.join(out_dir, "ch_null.npy"), ch_null)
    np.save(os.path.join(out_dir, "db_null.npy"), db_null)
    np.save(os.path.join(out_dir, "sil_null.npy"), sil_null)

    # Figures (one chart/figure)
    save_hist_with_true(
        ch_null, ch_true,
        "Permutation null (CH) and true value",
        "Calinski–Harabasz index",
        os.path.join(out_dir, "figure_CH_null_vs_true.png"),
        os.path.join(out_dir, "figure_CH_null_vs_true.pdf"),
    )
    save_hist_with_true(
        db_null, db_true,
        "Permutation null (DB) and true value",
        "Davies–Bouldin index",
        os.path.join(out_dir, "figure_DB_null_vs_true.png"),
        os.path.join(out_dir, "figure_DB_null_vs_true.pdf"),
    )
    save_hist_with_true(
        sil_null, sil_true,
        "Permutation null (Silhouette) and true value",
        "Silhouette score",
        os.path.join(out_dir, "figure_SIL_null_vs_true.png"),
        os.path.join(out_dir, "figure_SIL_null_vs_true.pdf"),
    )

    print("Saved:", summary_path)
    print("Saved:", paper_csv_path)
    print("Saved figures to:", out_dir)


if __name__ == "__main__":
    main()