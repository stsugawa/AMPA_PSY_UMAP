#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducible UMAP pipeline for the AMPAR-PET manuscript.

The pipeline implements the analyses described in the manuscript:
  1. SUVR-WB: unsupervised UMAP, weakly supervised true-label UMAP,
     and 1,000 weakly supervised label-permuted UMAP runs.
  2. SUVR-WM (raw): the same three conditions and permutation testing.
  3. SUVR-WM (age/sex residualized): the same sensitivity analysis.
  4. Figure 3 covariate overlays: age, sex, MaskMeanSUVR_output01, and site
     are plotted on the *raw SUVR-WM unsupervised embedding only*
     (perm_id=-1; embedding_type='unsupervised').

Important design choices
------------------------
- Exactly 219 participants and 75 regional features are required.
- Missing/non-finite analysis values are rejected; this script does not impute.
- SUVR-WM feature residualization is performed before feature z-standardization.
- UMAP uses Euclidean distance, n_neighbors=50, min_dist=1, spread=10,
  n_components=2, target_weight=0.005, and a fixed random seed.
- Label permutations are without replacement and preserve class sizes.
- CHI/Silhouette use the upper tail; DBI uses the lower tail.
- Empirical p-values use the +1 correction: (b + 1)/(N + 1).

Run all manuscript analyses:
  python ampa_umap_pipeline.py all \
      --wb-data /path/to/HCPSY219_WBSUVR_Dataset.csv \
      --wm-data /path/to/HCPSY219_WMSUVR_Dataset_withMaskMean.csv \
      --out-root ./result \
      --nperm 1000 \
      --seed 0

Replot Figure 3 from an existing unsupervised embedding:
  python ampa_umap_pipeline.py plot-covariates \
      --embedding-csv /path/to/wmsuvr_raw_label_by_random/2d/reduction_no_label.csv \
      --covars-csv /path/to/HCPSY219_WMSUVR_Dataset_withMaskMean.csv \
      --out-dir /path/to/wmsuvr_raw_label_by_random/2d
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


EXPECTED_N = 219
EXPECTED_D = 75

# The numeric coding preserves the order used by the existing Python scripts.
GROUP_ORDER = ("ASD", "BD", "MDD", "HC", "SCH")
GROUP_TO_CODE = {group: i + 1 for i, group in enumerate(GROUP_ORDER)}
CODE_TO_GROUP = {value: key for key, value in GROUP_TO_CODE.items()}
EXPECTED_GROUP_COUNTS = {
    "ASD": 35,
    "BD": 37,
    "MDD": 35,
    "HC": 70,
    "SCH": 42,
}

GROUP_ALIASES = {
    "ASD": "ASD",
    "AUTISM": "ASD",
    "AUTISMSPECTRUMDISORDER": "ASD",
    "BIP": "BD",
    "BD": "BD",
    "BIPOLAR": "BD",
    "BIPOLARDISORDER": "BD",
    "DEP": "MDD",
    "MDD": "MDD",
    "DEPRESSION": "MDD",
    "MAJORDEPRESSIVEDISORDER": "MDD",
    "HC": "HC",
    "CONTROL": "HC",
    "HEALTHYCONTROL": "HC",
    "HEALTHYCONTROLS": "HC",
    "SCH": "SCH",
    "SCZ": "SCH",
    "SCHIZOPHRENIA": "SCH",
}

GROUP_PLOT_ORDER = ("HC", "SCH", "BD", "MDD", "ASD")

DEFAULT_UMAP_PARAMETERS: dict[str, Any] = {
    "n_neighbors": 50,
    "min_dist": 1.0,
    "spread": 10.0,
    "n_components": 2,
    "metric": "euclidean",
    "target_metric": "categorical",
    "target_weight": 0.005,
}

DEFAULT_MEAN_SUVR_COLUMN = "MaskMeanSUVR_output01"


@dataclass(frozen=True)
class PreparedData:
    X: np.ndarray
    mids: np.ndarray
    groups: np.ndarray
    labels: np.ndarray
    feature_columns: tuple[str, ...]
    source_rows: np.ndarray
    residualized: bool


@dataclass(frozen=True)
class AnalysisResult:
    analysis_name: str
    output_dir: Path
    summary_csv: Path
    reduction_with_labels_csv: Path
    reduction_no_label_csv: Path
    reduction_true_labels_csv: Path


def _normalise_token(value: Any) -> str:
    text = str(value).strip().upper()
    return "".join(ch for ch in text if ch.isalnum())


def canonicalize_group(value: Any) -> str:
    token = _normalise_token(value)
    if token in GROUP_ALIASES:
        return GROUP_ALIASES[token]
    raise ValueError(f"Unrecognized diagnostic group value: {value!r}")


def resolve_column(
    df: pd.DataFrame,
    preferred: str,
    aliases: Sequence[str] = (),
    required: bool = True,
) -> str | None:
    """Resolve a column name case-insensitively."""
    candidates = (preferred, *aliases)
    exact = set(df.columns)
    for candidate in candidates:
        if candidate in exact:
            return candidate

    lower_map: dict[str, str] = {}
    for col in df.columns:
        lower_map.setdefault(str(col).casefold(), str(col))
    for candidate in candidates:
        found = lower_map.get(candidate.casefold())
        if found is not None:
            return found

    if required:
        raise ValueError(
            f"Required column {preferred!r} was not found. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require_expected_rows(df: pd.DataFrame, name: str) -> None:
    if len(df) != EXPECTED_N:
        raise ValueError(
            f"{name} contains {len(df)} rows; the manuscript analysis requires "
            f"exactly {EXPECTED_N} participants."
        )


def require_unique_nonmissing(series: pd.Series, name: str) -> None:
    if series.isna().any():
        rows = series.index[series.isna()].tolist()[:10]
        raise ValueError(f"{name} contains missing values at rows {rows}.")
    if series.astype(str).str.strip().eq("").any():
        rows = series.index[series.astype(str).str.strip().eq("")].tolist()[:10]
        raise ValueError(f"{name} contains blank values at rows {rows}.")
    if series.duplicated().any():
        dupes = series[series.duplicated(keep=False)].astype(str).unique().tolist()[:10]
        raise ValueError(f"{name} must be unique; duplicated values include {dupes}.")


def require_complete_numeric(
    frame: pd.DataFrame,
    name: str,
    expected_shape: tuple[int, int] | None = None,
) -> pd.DataFrame:
    if expected_shape is not None and frame.shape != expected_shape:
        raise ValueError(
            f"{name} has shape {frame.shape}; expected {expected_shape}."
        )

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    bad = ~np.isfinite(values)
    if bad.any():
        positions = np.argwhere(bad)
        examples: list[str] = []
        for row_i, col_i in positions[:10]:
            original = frame.iloc[row_i, col_i]
            examples.append(
                f"row={frame.index[row_i]!r}, column={frame.columns[col_i]!r}, "
                f"value={original!r}"
            )
        raise ValueError(
            f"{name} contains {int(bad.sum())} missing, non-numeric, or non-finite "
            f"values. The manuscript does not specify imputation, so the pipeline stops. "
            f"Examples: {'; '.join(examples)}"
        )
    return numeric.astype(float)


def extract_mids(df: pd.DataFrame) -> np.ndarray:
    mid_col = resolve_column(
        df,
        "MID",
        aliases=("participant_id", "participant", "subject_id", "subject", "ID"),
        required=False,
    )
    if mid_col is None:
        warnings.warn(
            "No participant-ID column was found. Stable row IDs will be generated. "
            "For future runs, include MID in the input CSV.",
            RuntimeWarning,
        )
        return np.asarray([f"ROW-{i:03d}" for i in range(1, len(df) + 1)], dtype=object)

    mids = df[mid_col].astype("string")
    require_unique_nonmissing(mids, f"Participant ID column {mid_col!r}")
    return mids.astype(str).to_numpy(dtype=object)


def extract_groups(df: pd.DataFrame, allow_row_fallback: bool) -> np.ndarray:
    group_col = resolve_column(
        df,
        "group",
        aliases=("diagnosis", "diagnostic_group", "disease", "Dx", "dx"),
        required=False,
    )

    if group_col is not None:
        if df[group_col].isna().any():
            raise ValueError(f"Diagnostic group column {group_col!r} contains missing values.")
        try:
            groups = np.asarray(
                [canonicalize_group(value) for value in df[group_col].tolist()],
                dtype=object,
            )
        except ValueError as exc:
            raise ValueError(
                f"Could not standardize values in diagnostic group column {group_col!r}."
            ) from exc
    elif allow_row_fallback:
        warnings.warn(
            "No diagnostic group column was found in the SUVR-WB file. Falling back "
            "to the historical row blocks: ASD 1-35, BD 36-72, MDD 73-107, "
            "HC 108-177, SCH 178-219. A group column is safer for future runs.",
            RuntimeWarning,
        )
        groups = np.asarray(
            ["ASD"] * 35
            + ["BD"] * 37
            + ["MDD"] * 35
            + ["HC"] * 70
            + ["SCH"] * 42,
            dtype=object,
        )
    else:
        raise ValueError(
            "No diagnostic group column was found. Add a 'group' column containing "
            "ASD, BIP/BD, DEP/MDD, HC, or SCH."
        )

    counts = pd.Series(groups).value_counts().to_dict()
    if counts != EXPECTED_GROUP_COUNTS:
        raise ValueError(
            f"Diagnostic group counts are {counts}; expected {EXPECTED_GROUP_COUNTS}."
        )
    return groups


def encode_groups(groups: np.ndarray) -> np.ndarray:
    try:
        return np.asarray([GROUP_TO_CODE[str(group)] for group in groups], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Unexpected canonical group: {exc.args[0]!r}") from exc


def select_wb_features(
    df: pd.DataFrame,
    start_1based: int = 8,
    end_1based: int = 82,
    prefix: str | None = None,
) -> list[str]:
    if prefix:
        selected = [col for col in df.columns if str(col).startswith(prefix)]
    else:
        if start_1based < 1 or end_1based < start_1based:
            raise ValueError("Invalid SUVR-WB feature-column range.")
        selected = list(df.columns[start_1based - 1 : end_1based])

    if len(selected) != EXPECTED_D:
        mode = f"prefix {prefix!r}" if prefix else f"columns {start_1based}:{end_1based}"
        raise ValueError(
            f"SUVR-WB feature selection by {mode} yielded {len(selected)} columns; "
            f"expected exactly {EXPECTED_D}. Selected columns: {selected}"
        )
    return [str(col) for col in selected]


def select_wm_features(df: pd.DataFrame, prefix: str = "ManualWM_") -> list[str]:
    selected = [col for col in df.columns if str(col).startswith(prefix)]
    if len(selected) != EXPECTED_D:
        raise ValueError(
            f"SUVR-WM feature prefix {prefix!r} yielded {len(selected)} columns; "
            f"expected exactly {EXPECTED_D}. Selected columns: {selected}"
        )
    return [str(col) for col in selected]


def encode_sex_for_regression(series: pd.Series) -> np.ndarray:
    """Convert a binary sex column to a finite numeric predictor."""
    if series.isna().any():
        rows = series.index[series.isna()].tolist()[:10]
        raise ValueError(f"Sex contains missing values at rows {rows}.")

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        values = numeric.to_numpy(dtype=float)
    else:
        mapping = {
            "M": 0.0,
            "MALE": 0.0,
            "MAN": 0.0,
            "F": 1.0,
            "FEMALE": 1.0,
            "WOMAN": 1.0,
        }
        converted: list[float] = []
        for value in series.tolist():
            token = _normalise_token(value)
            if token not in mapping:
                raise ValueError(
                    f"Sex value {value!r} is neither numeric nor a recognized binary label."
                )
            converted.append(mapping[token])
        values = np.asarray(converted, dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Sex contains non-finite numeric values.")
    if np.unique(values).size != 2:
        raise ValueError(
            f"Sex must contain exactly two categories for this analysis; found {np.unique(values)}."
        )
    return values


def residualize_features(
    X_raw: pd.DataFrame,
    age: np.ndarray,
    sex_numeric: np.ndarray,
) -> pd.DataFrame:
    design = np.column_stack([age.astype(float), sex_numeric.astype(float)])
    model = LinearRegression(fit_intercept=True)
    residuals = np.empty_like(X_raw.to_numpy(dtype=float), dtype=float)
    for j, column in enumerate(X_raw.columns):
        values = X_raw[column].to_numpy(dtype=float)
        model.fit(design, values)
        residuals[:, j] = values - model.predict(design)
    return pd.DataFrame(residuals, index=X_raw.index, columns=X_raw.columns)


def prepare_analysis_data(
    df: pd.DataFrame,
    analysis_kind: str,
    residualize: bool,
    wb_feature_start: int = 8,
    wb_feature_end: int = 82,
    wb_feature_prefix: str | None = None,
    wm_feature_prefix: str = "ManualWM_",
) -> PreparedData:
    require_expected_rows(df, "Input dataset")
    mids = extract_mids(df)

    if analysis_kind == "wb":
        groups = extract_groups(df, allow_row_fallback=True)
        feature_columns = select_wb_features(
            df,
            start_1based=wb_feature_start,
            end_1based=wb_feature_end,
            prefix=wb_feature_prefix,
        )
        if residualize:
            raise ValueError("Age/sex residualization is only defined here for SUVR-WM.")
    elif analysis_kind == "wm":
        groups = extract_groups(df, allow_row_fallback=False)
        feature_columns = select_wm_features(df, prefix=wm_feature_prefix)
    else:
        raise ValueError(f"Unknown analysis kind: {analysis_kind!r}")

    X_raw = require_complete_numeric(
        df[feature_columns].copy(),
        name=f"{analysis_kind.upper()} regional feature matrix",
        expected_shape=(EXPECTED_N, EXPECTED_D),
    )

    if residualize:
        age_col = resolve_column(df, "age", aliases=("Age",))
        sex_col = resolve_column(df, "sex", aliases=("Sex", "gender", "Gender"))
        age_frame = require_complete_numeric(
            df[[age_col]].copy(),
            name="Age covariate",
            expected_shape=(EXPECTED_N, 1),
        )
        age = age_frame[age_col].to_numpy(dtype=float)
        sex_numeric = encode_sex_for_regression(df[sex_col])
        X_used = residualize_features(X_raw, age, sex_numeric)
    else:
        X_used = X_raw

    feature_sd = X_used.std(axis=0, ddof=0)
    zero_sd = feature_sd.index[np.isclose(feature_sd.to_numpy(dtype=float), 0.0)].tolist()
    if zero_sd:
        raise ValueError(
            f"The following analysis features have zero variance and cannot be "
            f"z-standardized: {zero_sd}"
        )

    X = StandardScaler(with_mean=True, with_std=True).fit_transform(
        X_used.to_numpy(dtype=float)
    )
    if X.shape != (EXPECTED_N, EXPECTED_D) or not np.isfinite(X).all():
        raise RuntimeError("Unexpected non-finite values after feature standardization.")

    labels = encode_groups(groups)
    return PreparedData(
        X=X,
        mids=mids,
        groups=groups,
        labels=labels,
        feature_columns=tuple(feature_columns),
        source_rows=np.arange(1, EXPECTED_N + 1, dtype=int),
        residualized=residualize,
    )


def import_umap() -> Any:
    try:
        return importlib.import_module("umap")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The package 'umap-learn' is required for UMAP fitting. Install the "
            "dependencies with: pip install -r requirements.txt"
        ) from exc


def make_umap(seed: int, supervised: bool) -> Any:
    umap_module = import_umap()
    kwargs: dict[str, Any] = {
        "n_neighbors": DEFAULT_UMAP_PARAMETERS["n_neighbors"],
        "min_dist": DEFAULT_UMAP_PARAMETERS["min_dist"],
        "spread": DEFAULT_UMAP_PARAMETERS["spread"],
        "n_components": DEFAULT_UMAP_PARAMETERS["n_components"],
        "metric": DEFAULT_UMAP_PARAMETERS["metric"],
        "random_state": seed,
        "n_jobs": 1,
        "verbose": False,
    }
    if supervised:
        kwargs.update(
            target_metric=DEFAULT_UMAP_PARAMETERS["target_metric"],
            target_weight=DEFAULT_UMAP_PARAMETERS["target_weight"],
        )
    return umap_module.UMAP(**kwargs)


def fit_umap(X: np.ndarray, seed: int, y: np.ndarray | None = None) -> np.ndarray:
    reducer = make_umap(seed=seed, supervised=y is not None)
    if y is None:
        embedding = reducer.fit_transform(X)
    else:
        embedding = reducer.fit_transform(X, y=y)
    embedding = np.asarray(embedding, dtype=float)
    if embedding.shape != (EXPECTED_N, 2) or not np.isfinite(embedding).all():
        raise RuntimeError(f"Unexpected UMAP output shape or values: {embedding.shape}")
    return embedding


def compute_metrics(embedding: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unique = np.unique(labels)
    if unique.size < 2 or unique.size >= len(labels):
        raise ValueError("Cluster metrics require at least two non-singleton label groups.")
    return {
        "Calinski-Harabasz": float(calinski_harabasz_score(embedding, labels)),
        "Davies-Bouldin": float(davies_bouldin_score(embedding, labels)),
        "Silhouette": float(silhouette_score(embedding, labels, metric="euclidean")),
    }


def make_embedding_frame(
    prepared: PreparedData,
    embedding: np.ndarray,
    target_labels: np.ndarray,
    perm_id: int,
    embedding_type: str,
) -> pd.DataFrame:
    target_labels = np.asarray(target_labels, dtype=int)
    if target_labels.shape != (EXPECTED_N,):
        raise ValueError("Target labels have an unexpected shape.")
    target_groups = np.asarray([CODE_TO_GROUP[int(x)] for x in target_labels], dtype=object)
    return pd.DataFrame(
        {
            "id": prepared.source_rows,
            "MID": prepared.mids,
            "true_group": prepared.groups,
            "target_group": target_groups,
            "label": target_labels,
            "perm_id": int(perm_id),
            "embedding_type": embedding_type,
            "umap1": embedding[:, 0],
            "umap2": embedding[:, 1],
        }
    )


def _group_colors() -> dict[str, Any]:
    # Match the manuscript figures: HC gray, SCH gold, BD pink, MDD blue, ASD green.
    return {
        "HC": "#8C8C8C",
        "SCH": "#E69F00",
        "BD": "#CC79A7",
        "MDD": "#0072B2",
        "ASD": "#009E73",
    }


def save_group_plot(
    embedding: np.ndarray,
    groups: Sequence[str],
    title: str,
    output_base: Path,
) -> None:
    colors = _group_colors()
    fig, ax = plt.subplots(figsize=(6, 5))
    group_array = np.asarray(groups, dtype=object)
    for group in GROUP_PLOT_ORDER:
        mask = group_array == group
        if mask.any():
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=18,
                label=group,
                color=colors[group],
                linewidths=0,
                alpha=0.9,
            )
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(title="Group", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_metric_histogram(
    null_values: np.ndarray,
    true_value: float,
    metric: str,
    output_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(null_values, bins=40)
    ax.axvline(true_value, color="red", linewidth=2)
    ax.set_xlabel(metric)
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation null: {metric}")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def save_metric_panel(
    metric_table: pd.DataFrame,
    true_metrics: Mapping[str, float],
    output_base: Path,
) -> None:
    order = ("Calinski-Harabasz", "Silhouette", "Davies-Bouldin")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric in zip(axes, order):
        values = metric_table.loc[metric_table["perm_id"] > 0, metric].to_numpy(dtype=float)
        ax.hist(values, bins=40)
        ax.axvline(float(true_metrics[metric]), color="red", linewidth=2)
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def summarize_metric_table(
    metric_table: pd.DataFrame,
    n_planned_tests: int,
) -> pd.DataFrame:
    true_rows = metric_table.loc[metric_table["perm_id"] == 0]
    if len(true_rows) != 1:
        raise ValueError("Metric table must contain exactly one true-label row (perm_id=0).")
    null = metric_table.loc[metric_table["perm_id"] > 0]
    if null.empty:
        raise ValueError("Metric table contains no permutation rows.")

    records: list[dict[str, Any]] = []
    for metric, tail in (
        ("Calinski-Harabasz", "upper"),
        ("Silhouette", "upper"),
        ("Davies-Bouldin", "lower"),
    ):
        true_value = float(true_rows.iloc[0][metric])
        null_values = null[metric].to_numpy(dtype=float)
        valid = np.isfinite(null_values)
        valid_values = null_values[valid]
        if valid_values.size == 0:
            raise ValueError(f"No valid permutation values for {metric}.")
        if tail == "upper":
            b = int(np.sum(valid_values >= true_value))
        else:
            b = int(np.sum(valid_values <= true_value))
        p_value = (b + 1) / (valid_values.size + 1)
        threshold = 0.05 / n_planned_tests
        records.append(
            {
                "metric": metric,
                "tail": tail,
                "true_value": true_value,
                "null_mean": float(np.mean(valid_values)),
                "null_sd": float(np.std(valid_values, ddof=1)),
                "extreme_permutations_b": b,
                "n_valid_permutations": int(valid_values.size),
                "p_value": float(p_value),
                "n_planned_tests": int(n_planned_tests),
                "bonferroni_threshold": float(threshold),
                "significant_after_bonferroni": bool(p_value < threshold),
            }
        )
    return pd.DataFrame.from_records(records)


def save_metrics_outputs(
    output_dir: Path,
    metric_table: pd.DataFrame,
    n_planned_tests: int,
) -> Path:
    metrics_dir = output_dir / "permutation_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metric_long_path = metrics_dir / "permutation_metrics_long.csv"
    metric_table.to_csv(metric_long_path, index=False)

    summary = summarize_metric_table(metric_table, n_planned_tests=n_planned_tests)
    summary_path = metrics_dir / "permutation_summary.csv"
    summary.to_csv(summary_path, index=False)

    true_row = metric_table.loc[metric_table["perm_id"] == 0].iloc[0]
    true_metrics = {
        metric: float(true_row[metric])
        for metric in ("Calinski-Harabasz", "Davies-Bouldin", "Silhouette")
    }

    for metric, stem in (
        ("Calinski-Harabasz", "figure_CH_null_vs_true"),
        ("Davies-Bouldin", "figure_DB_null_vs_true"),
        ("Silhouette", "figure_SIL_null_vs_true"),
    ):
        null_values = metric_table.loc[metric_table["perm_id"] > 0, metric].to_numpy(dtype=float)
        save_metric_histogram(
            null_values=null_values,
            true_value=true_metrics[metric],
            metric=metric,
            output_base=metrics_dir / stem,
        )

    save_metric_panel(
        metric_table=metric_table,
        true_metrics=true_metrics,
        output_base=metrics_dir / "figure_permutation_metrics_panel",
    )

    for metric, filename in (
        ("Calinski-Harabasz", "ch_null.npy"),
        ("Davies-Bouldin", "db_null.npy"),
        ("Silhouette", "sil_null.npy"),
    ):
        values = metric_table.loc[metric_table["perm_id"] > 0, metric].to_numpy(dtype=float)
        np.save(metrics_dir / filename, values)

    return summary_path


def _categorical_order(values: pd.Series, preferred: Sequence[str] | None = None) -> list[str]:
    strings = values.astype("string").fillna("Missing").astype(str)
    observed = list(dict.fromkeys(strings.tolist()))
    if preferred is None:
        return sorted(observed)
    ordered = [value for value in preferred if value in observed]
    ordered.extend(value for value in observed if value not in ordered)
    return ordered


def _format_sex_values(series: pd.Series) -> pd.Series:
    """Use readable labels when the encoding can be inferred safely."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and numeric.nunique(dropna=True) == 2:
        unique = sorted(numeric.unique().tolist())
        # Common codings. If unknown, retain the original numbers.
        # The study CSV uses 0=Female and 1=Male. A 1/2 coding is also
        # supported as 1=Male and 2=Female. Other numeric codings are
        # retained verbatim rather than guessed.
        if unique == [0, 1]:
            return numeric.map({0: "Female", 1: "Male"}).astype("string")
        if unique == [1, 2]:
            return numeric.map({1: "Male", 2: "Female"}).astype("string")
    return series.astype("string")


def plot_continuous_on_axis(
    ax: plt.Axes,
    embedding: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
) -> None:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError(f"No finite values are available for the {title} overlay.")
    scatter = ax.scatter(
        embedding[valid, 0],
        embedding[valid, 1],
        c=values[valid],
        s=18,
        linewidths=0,
        cmap="viridis",
    )
    if (~valid).any():
        ax.scatter(
            embedding[~valid, 0],
            embedding[~valid, 1],
            s=18,
            linewidths=0,
            color="lightgray",
            label="Missing",
        )
        ax.legend(frameon=False, loc="best")
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    colorbar = ax.figure.colorbar(scatter, ax=ax)
    colorbar.set_label(colorbar_label)


def plot_categorical_on_axis(
    ax: plt.Axes,
    embedding: np.ndarray,
    values: pd.Series,
    title: str,
    preferred_order: Sequence[str] | None = None,
    palette: Mapping[str, Any] | None = None,
) -> None:
    strings = values.astype("string").fillna("Missing").astype(str)
    categories = _categorical_order(strings, preferred=preferred_order)
    cmap = plt.get_cmap("tab10")
    for i, category in enumerate(categories):
        mask = strings.to_numpy() == category
        color = palette.get(category, cmap(i % 10)) if palette is not None else cmap(i % 10)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=18,
            label=category,
            color=color,
            linewidths=0,
            alpha=0.9,
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(title="Category", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")


def save_covariate_plots(
    merged: pd.DataFrame,
    output_dir: Path,
    mean_suvr_column: str,
) -> None:
    embedding = merged[["umap1", "umap2"]].to_numpy(dtype=float)
    age = merged["age"].to_numpy(dtype=float)
    mean_suvr = merged[mean_suvr_column].to_numpy(dtype=float)
    sex_display = _format_sex_values(merged["sex"])
    site_display = (
        merged["site"]
        .astype("string")
        .replace(
            {
                "FUKUI": "Fukui",
                "Fukui": "Fukui",
                "KEIO": "Keio",
                "Keio": "Keio",
                "KYUSYU": "Kyushu",
                "KYUSHU": "Kyushu",
                "Kyusyu": "Kyushu",
                "Kyushu": "Kyushu",
                "YCU": "YCU",
            }
        )
    )
    sex_palette = {"Male": "#0072B2", "Female": "#E69F00", "Missing": "#BDBDBD"}
    site_palette = {
        "Fukui": "#E69F00",
        "Keio": "#009E73",
        "Kyushu": "#D55E00",
        "YCU": "#56B4E9",
        "Missing": "#BDBDBD",
    }

    # Individual panels.
    individual_specs = (
        ("age", "Age", "Age (years)"),
        (mean_suvr_column, "Mean SUVR-WM", "Mean SUVR-WM"),
    )
    for column, title, label in individual_specs:
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_continuous_on_axis(
            ax,
            embedding,
            merged[column].to_numpy(dtype=float),
            title=title,
            colorbar_label=label,
        )
        fig.tight_layout()
        stem = "age" if column == "age" else "mean_SUVR_WM"
        fig.savefig(output_dir / f"umap_unsupervised_colored_by_{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / f"umap_unsupervised_colored_by_{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    for values, title, stem, order, palette in (
        (sex_display, "Sex", "sex", ("Male", "Female"), sex_palette),
        (site_display, "Site", "site", ("Fukui", "Keio", "Kyushu", "YCU"), site_palette),
    ):
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_categorical_on_axis(
            ax,
            embedding,
            values,
            title=title,
            preferred_order=order,
            palette=palette,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"umap_unsupervised_colored_by_{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / f"umap_unsupervised_colored_by_{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Four-panel Figure 3 layout: age, sex, mean SUVR-WM, site.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_continuous_on_axis(axes[0, 0], embedding, age, "Age", "Age (years)")
    plot_categorical_on_axis(
        axes[0, 1],
        embedding,
        sex_display,
        "Sex",
        preferred_order=("Male", "Female"),
        palette=sex_palette,
    )
    plot_continuous_on_axis(
        axes[1, 0], embedding, mean_suvr, "Mean SUVR-WM", "Mean SUVR-WM"
    )
    plot_categorical_on_axis(
        axes[1, 1],
        embedding,
        site_display,
        "Site",
        preferred_order=("Fukui", "Keio", "Kyushu", "YCU"),
        palette=site_palette,
    )

    x_min, x_max = float(np.min(embedding[:, 0])), float(np.max(embedding[:, 0]))
    y_min, y_max = float(np.min(embedding[:, 1])), float(np.max(embedding[:, 1]))
    x_pad = max((x_max - x_min) * 0.05, 1e-6)
    y_pad = max((y_max - y_min) * 0.05, 1e-6)
    for ax in axes.flat:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    fig.tight_layout()
    fig.savefig(output_dir / "figure3_covariates_unsupervised.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "figure3_covariates_unsupervised.pdf", bbox_inches="tight")
    plt.close(fig)


def load_covariates(
    covars_csv: Path,
    mean_suvr_column: str,
) -> pd.DataFrame:
    if not covars_csv.exists():
        raise FileNotFoundError(f"Covariate CSV not found: {covars_csv}")
    covars = pd.read_csv(covars_csv)
    require_expected_rows(covars, "Covariate dataset")

    mid_col = resolve_column(
        covars,
        "MID",
        aliases=("participant_id", "participant", "subject_id", "subject", "ID"),
    )
    age_col = resolve_column(covars, "age", aliases=("Age",))
    sex_col = resolve_column(covars, "sex", aliases=("Sex", "gender", "Gender"))
    site_col = resolve_column(covars, "site", aliases=("Site", "acquisition_site"))
    mean_col = resolve_column(covars, mean_suvr_column)

    mids = covars[mid_col].astype("string")
    require_unique_nonmissing(mids, f"Covariate participant ID column {mid_col!r}")

    age_numeric = require_complete_numeric(
        covars[[age_col]].copy(),
        name="Figure 3 age covariate",
        expected_shape=(EXPECTED_N, 1),
    )[age_col].to_numpy(dtype=float)

    # Missing mean-SUVR values are not imputed. They remain in the output table and
    # are shown as gray points so that all participant locations remain visible.
    mean_numeric = pd.to_numeric(covars[mean_col], errors="coerce").to_numpy(dtype=float)
    mean_numeric[~np.isfinite(mean_numeric)] = np.nan

    result = pd.DataFrame(
        {
            "MID": mids.astype(str),
            "age": age_numeric,
            "sex": covars[sex_col].astype("string").fillna("Missing").to_numpy(),
            "site": covars[site_col].astype("string").fillna("Missing").to_numpy(),
            mean_suvr_column: mean_numeric,
        }
    )
    return result


def select_unsupervised_embedding(embedding_df: pd.DataFrame) -> pd.DataFrame:
    """Select the single raw SUVR-WM unsupervised embedding used for Figure 3."""
    required = {"umap1", "umap2"}
    missing = required.difference(embedding_df.columns)
    if missing:
        raise ValueError(f"Embedding CSV is missing columns: {sorted(missing)}")

    selected = embedding_df.copy()
    selector_found = False
    mask = pd.Series(True, index=selected.index, dtype=bool)

    if "embedding_type" in selected.columns:
        selector_found = True
        embedding_type = (
            selected["embedding_type"]
            .astype("string")
            .str.strip()
            .str.casefold()
        )
        mask &= embedding_type.eq("unsupervised").fillna(False)

    if "perm_id" in selected.columns:
        selector_found = True
        perm_id = pd.to_numeric(selected["perm_id"], errors="coerce")
        mask &= perm_id.eq(-1).fillna(False)

    if selector_found:
        selected = selected.loc[mask].copy()
    else:
        warnings.warn(
            "Embedding CSV has neither perm_id nor embedding_type. Assuming all rows "
            "are the unsupervised embedding.",
            RuntimeWarning,
        )

    if len(selected) != EXPECTED_N:
        raise ValueError(
            f"Unsupervised embedding selection yielded {len(selected)} rows; "
            f"expected {EXPECTED_N}. Use reduction_no_label.csv "
            "(perm_id=-1 and embedding_type='unsupervised'), not "
            "reduction_with_labels.csv."
        )
    if "id" in selected.columns:
        if selected["id"].duplicated().any():
            raise ValueError("Unsupervised embedding contains duplicated row IDs.")
        selected = selected.sort_values("id").reset_index(drop=True)
    else:
        selected = selected.reset_index(drop=True)
    return selected


def merge_embedding_with_covariates(
    embedding: pd.DataFrame,
    covariates: pd.DataFrame,
) -> pd.DataFrame:
    if "MID" in embedding.columns:
        emb_mid = embedding["MID"].astype("string")
        require_unique_nonmissing(emb_mid, "Embedding MID")
        left = embedding.copy()
        left["MID"] = emb_mid.astype(str)
        merged = left.merge(
            covariates,
            on="MID",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        unmatched = merged.loc[merged["_merge"] != "both", "MID"].tolist()
        if unmatched:
            raise ValueError(f"Covariates were not found for embedding MIDs: {unmatched[:10]}")
        merged = merged.drop(columns="_merge")
    elif "id" in embedding.columns:
        # Backward-compatible route for existing archived CSVs that did not save MID.
        warnings.warn(
            "The embedding CSV has no MID. Covariates are being aligned by the historical "
            "1-based row ID. This is supported for old outputs only; newly generated outputs "
            "save MID and merge one-to-one by participant ID.",
            RuntimeWarning,
        )
        ids = pd.to_numeric(embedding["id"], errors="coerce")
        expected_ids = np.arange(1, EXPECTED_N + 1, dtype=int)
        if ids.isna().any() or not np.array_equal(ids.to_numpy(dtype=int), expected_ids):
            raise ValueError(
                f"Row-ID fallback requires id to be exactly 1..{EXPECTED_N} "
                "in the original dataset order."
            )
        merged = pd.concat(
            [
                embedding.reset_index(drop=True),
                covariates.reset_index(drop=True),
            ],
            axis=1,
        )
    else:
        raise ValueError(
            f"Embedding CSV must contain MID, or historical row id 1..{EXPECTED_N}."
        )

    if len(merged) != EXPECTED_N:
        raise RuntimeError("Covariate merge changed the number of participants.")
    return merged


def generate_covariate_outputs(
    embedding_df: pd.DataFrame,
    covars_csv: Path,
    output_dir: Path,
    mean_suvr_column: str = DEFAULT_MEAN_SUVR_COLUMN,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_unsupervised_embedding(embedding_df)
    covariates = load_covariates(covars_csv, mean_suvr_column=mean_suvr_column)
    merged = merge_embedding_with_covariates(selected, covariates)

    # Make the Figure 3 provenance unambiguous in the saved table.
    merged["figure3_embedding"] = "raw_SUVR_WM_unsupervised"
    if "embedding_type" not in merged.columns:
        merged["embedding_type"] = "unsupervised"
    elif not (
        merged["embedding_type"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq("unsupervised")
        .all()
    ):
        raise RuntimeError(
            "Figure 3 table unexpectedly contains a non-unsupervised embedding_type."
        )
    merged["embedding_type"] = "unsupervised"

    if "perm_id" not in merged.columns:
        merged["perm_id"] = -1
    else:
        perm_id = pd.to_numeric(merged["perm_id"], errors="coerce")
        if not perm_id.eq(-1).all():
            raise RuntimeError(
                "Figure 3 table unexpectedly contains rows other than perm_id=-1."
            )
        merged["perm_id"] = perm_id.astype(int)

    output_csv = output_dir / "reduction_unsupervised_with_covariates.csv"
    merged.to_csv(output_csv, index=False)
    missingness = {
        "age_missing": int(pd.to_numeric(merged["age"], errors="coerce").isna().sum()),
        "sex_missing": int(merged["sex"].astype("string").eq("Missing").sum()),
        "site_missing": int(merged["site"].astype("string").eq("Missing").sum()),
        f"{mean_suvr_column}_missing": int(
            pd.to_numeric(merged[mean_suvr_column], errors="coerce").isna().sum()
        ),
        "figure3_embedding": "raw_SUVR_WM_unsupervised",
        "missing_value_handling": "not imputed; missing continuous values are plotted in gray",
    }
    with (output_dir / "figure3_covariate_missingness.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(missingness, stream, ensure_ascii=False, indent=2)
    save_covariate_plots(
        merged=merged,
        output_dir=output_dir,
        mean_suvr_column=mean_suvr_column,
    )
    return output_csv


def write_analysis_metadata(
    output_dir: Path,
    input_csv: Path,
    prepared: PreparedData,
    analysis_name: str,
    seed: int,
    nperm: int,
    n_planned_tests: int,
) -> None:
    try:
        umap_version = getattr(import_umap(), "__version__", "unknown")
    except ModuleNotFoundError:
        umap_version = "not installed"

    metadata = {
        "analysis_name": analysis_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": str(input_csv.resolve()),
        "input_sha256": sha256_file(input_csv),
        "n_participants": int(prepared.X.shape[0]),
        "n_features": int(prepared.X.shape[1]),
        "feature_columns": list(prepared.feature_columns),
        "group_counts": pd.Series(prepared.groups).value_counts().to_dict(),
        "age_sex_residualized_before_standardization": prepared.residualized,
        "feature_standardization": "z-score across participants using sklearn StandardScaler",
        "missing_value_policy": "fail; no imputation",
        "umap_parameters": {
            **DEFAULT_UMAP_PARAMETERS,
            "random_state": int(seed),
            "n_jobs": 1,
        },
        "n_permutations": int(nperm),
        "label_permutation": "numpy.random.Generator.permutation without replacement",
        "permutation_p_value": "(b + 1) / (N + 1)",
        "n_planned_tests_for_bonferroni": int(n_planned_tests),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": importlib.import_module("sklearn").__version__,
            "matplotlib": matplotlib.__version__,
            "umap_learn": umap_version,
        },
    }
    with (output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)


def run_umap_analysis(
    *,
    data_csv: Path,
    output_dir: Path,
    analysis_name: str,
    analysis_kind: str,
    residualize: bool,
    seed: int,
    nperm: int,
    representative_perm: int,
    save_perm_every: int,
    n_planned_tests: int,
    wb_feature_start: int = 8,
    wb_feature_end: int = 82,
    wb_feature_prefix: str | None = None,
    wm_feature_prefix: str = "ManualWM_",
    covars_csv_for_figure3: Path | None = None,
    mean_suvr_column: str = DEFAULT_MEAN_SUVR_COLUMN,
) -> AnalysisResult:
    if not data_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {data_csv}")
    if nperm < 1:
        raise ValueError("nperm must be at least 1.")
    if representative_perm < 1:
        raise ValueError("representative_perm must be at least 1.")
    if representative_perm > nperm:
        warnings.warn(
            f"representative_perm={representative_perm} exceeds nperm={nperm}; "
            f"using permutation {nperm} instead.",
            RuntimeWarning,
        )
        representative_perm = nperm
    if save_perm_every < 0:
        raise ValueError("save_perm_every must be 0 or a positive integer.")

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_csv)
    prepared = prepare_analysis_data(
        df=df,
        analysis_kind=analysis_kind,
        residualize=residualize,
        wb_feature_start=wb_feature_start,
        wb_feature_end=wb_feature_end,
        wb_feature_prefix=wb_feature_prefix,
        wm_feature_prefix=wm_feature_prefix,
    )

    print(f"[{analysis_name}] fitting true-label UMAP", flush=True)
    true_embedding = fit_umap(prepared.X, seed=seed, y=prepared.labels)
    true_frame = make_embedding_frame(
        prepared,
        true_embedding,
        target_labels=prepared.labels,
        perm_id=0,
        embedding_type="true_label",
    )
    true_csv = output_dir / "reduction_true_labels.csv"
    true_frame.to_csv(true_csv, index=False)
    save_group_plot(
        true_embedding,
        prepared.groups,
        title="UMAP (true labels)",
        output_base=output_dir / "umap_true_labels",
    )

    print(f"[{analysis_name}] fitting unsupervised UMAP", flush=True)
    no_label_embedding = fit_umap(prepared.X, seed=seed, y=None)
    no_label_frame = make_embedding_frame(
        prepared,
        no_label_embedding,
        target_labels=prepared.labels,
        perm_id=-1,
        embedding_type="unsupervised",
    )
    no_label_frame["target_group"] = "not_used"
    no_label_csv = output_dir / "reduction_no_label.csv"
    no_label_frame.to_csv(no_label_csv, index=False)
    save_group_plot(
        no_label_embedding,
        prepared.groups,
        title="UMAP (unsupervised; colored by true group)",
        output_base=output_dir / "umap_no_label",
    )

    true_metrics = compute_metrics(true_embedding, prepared.labels)
    metric_records: list[dict[str, Any]] = [
        {"perm_id": 0, "embedding_type": "true_label", **true_metrics}
    ]
    all_frames: list[pd.DataFrame] = [true_frame]

    rng = np.random.default_rng(seed)
    for perm_id in range(1, nperm + 1):
        target_labels = rng.permutation(prepared.labels)
        if not np.array_equal(
            np.sort(target_labels),
            np.sort(prepared.labels),
        ):
            raise RuntimeError("Permutation failed to preserve class sizes.")

        perm_embedding = fit_umap(prepared.X, seed=seed, y=target_labels)
        perm_frame = make_embedding_frame(
            prepared,
            perm_embedding,
            target_labels=target_labels,
            perm_id=perm_id,
            embedding_type="permuted_label",
        )
        all_frames.append(perm_frame)
        metric_records.append(
            {
                "perm_id": perm_id,
                "embedding_type": "permuted_label",
                **compute_metrics(perm_embedding, target_labels),
            }
        )

        should_save = perm_id == representative_perm or (
            save_perm_every > 0 and perm_id % save_perm_every == 0
        )
        if should_save:
            save_group_plot(
                perm_embedding,
                perm_frame["target_group"].to_numpy(dtype=object),
                title=f"UMAP (permuted labels; permutation {perm_id})",
                output_base=output_dir / f"umap_perm_{perm_id:04d}",
            )
        if perm_id == representative_perm:
            perm_frame.to_csv(
                output_dir / f"reduction_representative_permutation_{perm_id:04d}.csv",
                index=False,
            )

        if perm_id == 1 or perm_id % max(1, min(25, nperm)) == 0 or perm_id == nperm:
            print(f"[{analysis_name}] completed permutation {perm_id}/{nperm}", flush=True)

    all_results = pd.concat(all_frames, axis=0, ignore_index=True)
    expected_rows = EXPECTED_N * (nperm + 1)
    if len(all_results) != expected_rows:
        raise RuntimeError(
            f"Unexpected combined result length {len(all_results)}; expected {expected_rows}."
        )
    all_csv = output_dir / "reduction_with_labels.csv"
    all_results.to_csv(all_csv, index=False)

    metric_table = pd.DataFrame.from_records(metric_records)
    summary_csv = save_metrics_outputs(
        output_dir=output_dir,
        metric_table=metric_table,
        n_planned_tests=n_planned_tests,
    )

    write_analysis_metadata(
        output_dir=output_dir,
        input_csv=data_csv,
        prepared=prepared,
        analysis_name=analysis_name,
        seed=seed,
        nperm=nperm,
        n_planned_tests=n_planned_tests,
    )

    if covars_csv_for_figure3 is not None:
        if analysis_kind != "wm" or residualize:
            raise ValueError(
                "Figure 3 covariates may only be plotted for the raw SUVR-WM analysis."
            )
        print(
            f"[{analysis_name}] plotting Figure 3 covariates on the unsupervised embedding only",
            flush=True,
        )
        generate_covariate_outputs(
            embedding_df=no_label_frame,
            covars_csv=covars_csv_for_figure3,
            output_dir=output_dir,
            mean_suvr_column=mean_suvr_column,
        )

    return AnalysisResult(
        analysis_name=analysis_name,
        output_dir=output_dir,
        summary_csv=summary_csv,
        reduction_with_labels_csv=all_csv,
        reduction_no_label_csv=no_label_csv,
        reduction_true_labels_csv=true_csv,
    )


def summarize_existing_reduction(
    input_csv: Path,
    output_dir: Path,
    n_planned_tests: int,
) -> Path:
    if not input_csv.exists():
        raise FileNotFoundError(f"Embedding CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    required = {"umap1", "umap2", "label", "perm_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Embedding CSV is missing required columns: {sorted(missing)}")

    perm_ids = sorted(pd.to_numeric(df["perm_id"], errors="raise").astype(int).unique().tolist())
    positive = [value for value in perm_ids if value > 0]
    if 0 not in perm_ids or not positive:
        raise ValueError("Embedding CSV must contain perm_id=0 and positive permutation IDs.")
    if positive != list(range(1, max(positive) + 1)):
        raise ValueError("Permutation IDs must be consecutive from 1 to N.")

    records: list[dict[str, Any]] = []
    for perm_id in [0, *positive]:
        subset = df.loc[df["perm_id"] == perm_id].copy()
        if len(subset) != EXPECTED_N:
            raise ValueError(
                f"perm_id={perm_id} contains {len(subset)} rows; expected {EXPECTED_N}."
            )
        if "id" in subset.columns and subset["id"].duplicated().any():
            raise ValueError(f"perm_id={perm_id} contains duplicated participant IDs.")
        embedding = require_complete_numeric(
            subset[["umap1", "umap2"]],
            name=f"Embedding coordinates for perm_id={perm_id}",
            expected_shape=(EXPECTED_N, 2),
        ).to_numpy(dtype=float)
        labels = pd.to_numeric(subset["label"], errors="raise").to_numpy(dtype=int)
        if perm_id > 0:
            true_counts = (
                df.loc[df["perm_id"] == 0, "label"].value_counts().sort_index().to_dict()
            )
            perm_counts = subset["label"].value_counts().sort_index().to_dict()
            if perm_counts != true_counts:
                raise ValueError(
                    f"perm_id={perm_id} does not preserve class sizes: {perm_counts} vs {true_counts}."
                )
        records.append(
            {
                "perm_id": perm_id,
                "embedding_type": "true_label" if perm_id == 0 else "permuted_label",
                **compute_metrics(embedding, labels),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_table = pd.DataFrame.from_records(records)
    return save_metrics_outputs(
        output_dir=output_dir,
        metric_table=metric_table,
        n_planned_tests=n_planned_tests,
    )


def combine_summaries(results: Sequence[AnalysisResult], output_path: Path) -> None:
    frames: list[pd.DataFrame] = []
    for result in results:
        frame = pd.read_csv(result.summary_csv)
        frame.insert(0, "analysis", result.analysis_name)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_path, index=False)


def _date_string(value: str | None) -> str:
    if value is None:
        return datetime.now().strftime("%Y%m%d")
    if len(value) != 8 or not value.isdigit():
        raise ValueError("--run-date must use YYYYMMDD format.")
    return value


def add_common_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-root", "--out_root", dest="out_root", type=Path, default=Path("./result"))
    parser.add_argument("--run-date", "--run_date", "--date", dest="run_date", type=str, default=None, help="YYYYMMDD; default is today")
    parser.add_argument("--nperm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--representative-perm", "--representative_perm",
        dest="representative_perm",
        type=int,
        default=1000,
        help="Permutation ID used for the representative random-label panel.",
    )
    parser.add_argument(
        "--save-perm-every", "--save_perm_every",
        dest="save_perm_every",
        type=int,
        default=10,
        help="Save every nth permutation plot; use 0 to save only the representative run.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AMPAR-PET SUVR UMAP and permutation-analysis pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    all_parser = subparsers.add_parser("all", help="Run all three manuscript analyses.")
    all_parser.add_argument("--wb-data", "--wb_data", dest="wb_data", type=Path, required=True)
    all_parser.add_argument("--wm-data", "--wm_data", dest="wm_data", type=Path, required=True)
    all_parser.add_argument(
        "--wm-covars-data", "--wm_covars_data",
        dest="wm_covars_data",
        type=Path,
        default=None,
        help="CSV containing MID, age, sex, site, and MaskMeanSUVR_output01. "
        "Defaults to --wm-data.",
    )
    all_parser.add_argument("--wb-feature-start", "--wb_feature_start", dest="wb_feature_start", type=int, default=8)
    all_parser.add_argument("--wb-feature-end", "--wb_feature_end", dest="wb_feature_end", type=int, default=82)
    all_parser.add_argument("--wb-feature-prefix", "--wb_feature_prefix", dest="wb_feature_prefix", type=str, default=None)
    all_parser.add_argument("--wm-feature-prefix", "--wm_feature_prefix", dest="wm_feature_prefix", type=str, default="ManualWM_")
    all_parser.add_argument(
        "--mean-suvr-column", "--mean_suvr_column", dest="mean_suvr_column",
        type=str, default=DEFAULT_MEAN_SUVR_COLUMN
    )
    add_common_fit_arguments(all_parser)

    wb_parser = subparsers.add_parser("wb", help="Run the SUVR-WB analysis.")
    wb_parser.add_argument("--data", type=Path, required=True)
    wb_parser.add_argument("--wb-feature-start", "--wb_feature_start", dest="wb_feature_start", type=int, default=8)
    wb_parser.add_argument("--wb-feature-end", "--wb_feature_end", dest="wb_feature_end", type=int, default=82)
    wb_parser.add_argument("--wb-feature-prefix", "--wb_feature_prefix", dest="wb_feature_prefix", type=str, default=None)
    add_common_fit_arguments(wb_parser)

    wm_parser = subparsers.add_parser("wm", help="Run a raw or residualized SUVR-WM analysis.")
    wm_parser.add_argument("--data", type=Path, required=True)
    wm_parser.add_argument("--residualize", type=int, choices=(0, 1), default=0)
    wm_parser.add_argument("--wm-feature-prefix", "--wm_feature_prefix", dest="wm_feature_prefix", type=str, default="ManualWM_")
    wm_parser.add_argument(
        "--covars-csv", "--covars_csv",
        dest="covars_csv",
        type=Path,
        default=None,
        help="For raw SUVR-WM only: plot Figure 3 covariates on the unsupervised embedding using this CSV.",
    )
    wm_parser.add_argument(
        "--mean-suvr-column", "--mean_suvr_column", dest="mean_suvr_column",
        type=str, default=DEFAULT_MEAN_SUVR_COLUMN
    )
    add_common_fit_arguments(wm_parser)

    summary_parser = subparsers.add_parser(
        "summarize", help="Recompute CHI/DBI/Silhouette permutation statistics."
    )
    summary_parser.add_argument("--input", type=Path, required=True)
    summary_parser.add_argument("--out-dir", "--outdir", dest="out_dir", type=Path, required=True)
    summary_parser.add_argument(
        "--n-planned-tests", "--n_planned_tests",
        dest="n_planned_tests",
        type=int,
        required=True,
        help="6 for the primary SUVR-WB/SUVR-WM analyses; 3 for residualized SUVR-WM.",
    )

    covar_parser = subparsers.add_parser(
        "plot-covariates",
        help="Plot age, sex, MaskMeanSUVR_output01, and site on the unsupervised raw SUVR-WM embedding only.",
    )
    covar_parser.add_argument(
        "--embedding-csv", "--embedding_csv",
        dest="embedding_csv",
        type=Path,
        required=True,
        help="Unsupervised embedding CSV, normally reduction_no_label.csv.",
    )
    covar_parser.add_argument("--covars-csv", "--covars_csv", dest="covars_csv", type=Path, required=True)
    covar_parser.add_argument("--out-dir", "--out_dir", dest="out_dir", type=Path, required=True)
    covar_parser.add_argument(
        "--mean-suvr-column", "--mean_suvr_column", dest="mean_suvr_column",
        type=str, default=DEFAULT_MEAN_SUVR_COLUMN
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "all":
            run_date = _date_string(args.run_date)
            root = args.out_root / run_date
            covars_csv = args.wm_covars_data or args.wm_data

            results = [
                run_umap_analysis(
                    data_csv=args.wb_data,
                    output_dir=root / "normalizedWB_label_by_random" / "2d",
                    analysis_name="SUVR-WB",
                    analysis_kind="wb",
                    residualize=False,
                    seed=args.seed,
                    nperm=args.nperm,
                    representative_perm=args.representative_perm,
                    save_perm_every=args.save_perm_every,
                    n_planned_tests=6,
                    wb_feature_start=args.wb_feature_start,
                    wb_feature_end=args.wb_feature_end,
                    wb_feature_prefix=args.wb_feature_prefix,
                ),
                run_umap_analysis(
                    data_csv=args.wm_data,
                    output_dir=root / "wmsuvr_raw_label_by_random" / "2d",
                    analysis_name="SUVR-WM raw",
                    analysis_kind="wm",
                    residualize=False,
                    seed=args.seed,
                    nperm=args.nperm,
                    representative_perm=args.representative_perm,
                    save_perm_every=args.save_perm_every,
                    n_planned_tests=6,
                    wm_feature_prefix=args.wm_feature_prefix,
                    covars_csv_for_figure3=covars_csv,
                    mean_suvr_column=args.mean_suvr_column,
                ),
                run_umap_analysis(
                    data_csv=args.wm_data,
                    output_dir=root / "wmsuvr_resid_label_by_random" / "2d",
                    analysis_name="SUVR-WM age/sex residualized",
                    analysis_kind="wm",
                    residualize=True,
                    seed=args.seed,
                    nperm=args.nperm,
                    representative_perm=args.representative_perm,
                    save_perm_every=args.save_perm_every,
                    n_planned_tests=3,
                    wm_feature_prefix=args.wm_feature_prefix,
                ),
            ]
            combine_summaries(results, root / "manuscript_permutation_summary.csv")
            print(f"All analyses completed. Results: {root}")

        elif args.command == "wb":
            run_date = _date_string(args.run_date)
            output_dir = args.out_root / run_date / "normalizedWB_label_by_random" / "2d"
            run_umap_analysis(
                data_csv=args.data,
                output_dir=output_dir,
                analysis_name="SUVR-WB",
                analysis_kind="wb",
                residualize=False,
                seed=args.seed,
                nperm=args.nperm,
                representative_perm=args.representative_perm,
                save_perm_every=args.save_perm_every,
                n_planned_tests=6,
                wb_feature_start=args.wb_feature_start,
                wb_feature_end=args.wb_feature_end,
                wb_feature_prefix=args.wb_feature_prefix,
            )
            print(f"SUVR-WB analysis completed. Results: {output_dir}")

        elif args.command == "wm":
            run_date = _date_string(args.run_date)
            residualize = bool(args.residualize)
            mode = "resid" if residualize else "raw"
            output_dir = args.out_root / run_date / f"wmsuvr_{mode}_label_by_random" / "2d"
            if residualize and args.covars_csv is not None:
                raise ValueError(
                    "--covars-csv is only valid for raw SUVR-WM because Figure 3 uses "
                    "the raw unsupervised embedding."
                )
            run_umap_analysis(
                data_csv=args.data,
                output_dir=output_dir,
                analysis_name=(
                    "SUVR-WM age/sex residualized" if residualize else "SUVR-WM raw"
                ),
                analysis_kind="wm",
                residualize=residualize,
                seed=args.seed,
                nperm=args.nperm,
                representative_perm=args.representative_perm,
                save_perm_every=args.save_perm_every,
                n_planned_tests=3 if residualize else 6,
                wm_feature_prefix=args.wm_feature_prefix,
                covars_csv_for_figure3=args.covars_csv,
                mean_suvr_column=args.mean_suvr_column,
            )
            print(f"SUVR-WM analysis completed. Results: {output_dir}")

        elif args.command == "summarize":
            summary_path = summarize_existing_reduction(
                input_csv=args.input,
                output_dir=args.out_dir,
                n_planned_tests=args.n_planned_tests,
            )
            print(f"Saved permutation summary: {summary_path}")

        elif args.command == "plot-covariates":
            embedding_df = pd.read_csv(args.embedding_csv)
            output_csv = generate_covariate_outputs(
                embedding_df=embedding_df,
                covars_csv=args.covars_csv,
                output_dir=args.out_dir,
                mean_suvr_column=args.mean_suvr_column,
            )
            print(f"Saved unsupervised covariate table: {output_csv}")

        else:
            parser.error(f"Unsupported command: {args.command}")

    except (ValueError, FileNotFoundError, RuntimeError, ModuleNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
