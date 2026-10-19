# -*- coding: utf-8 -*-
"""
32H_threshold_sensitivity_and_anti_cherry_picking_audit.py

Purpose
-------
Step 32H for the fluoride coagulation ML project.

This script audits whether the high-R2 selective applicability-domain (AD)
retained-subdomain result confirmed in Step 32F2 and chemically characterized
in Step 32G is robust to nearby rule-threshold changes, or whether it is mainly
caused by a single cherry-picked threshold / literature group.

Core audited policy carried from Step 32F2 / 32G
-----------------------------------------------
Policy ID: P08
Strategy: RepeatedBalanced_Extended
Method: XGBoost
Rule: Reliability_Zone == Outside AND NN_Risk <= 0.814815

Strictness
----------
1. No model is retrained here.
2. No target value is used to define the rule; target values are used only for
   post-hoc evaluation of retained predictions.
3. The exact P08 rule is reconstructed directly from
   32b_predictions_with_reliability_zone.csv.
4. Threshold sensitivity, split-level stability, literature dominance, and
   transfer to other strategies/models are all reported.
5. Figures use a consistent EST-style white-background publication format.

Expected input
--------------
outputs/tables/32b_predictions_with_reliability_zone.csv
Optional but helpful:
outputs/tables/32g_retained_subdomain_decision_report.txt
outputs/tables/32f2_reproducible_selective_ad_policy_audit_report.txt

Outputs
-------
Tables:
- 32h_threshold_sensitivity_all.csv
- 32h_exact_policy_metrics.csv
- 32h_exact_policy_split_metrics.csv
- 32h_exact_policy_paper_dominance.csv
- 32h_threshold_plateau_summary.csv
- 32h_strategy_method_transfer_summary.csv
- 32h_bootstrap_uncertainty_exact_policy.csv
- 32h_threshold_sensitivity_decision_report.txt

Figures:
- 188_32h_threshold_sensitivity_core.png/pdf
- 189_32h_threshold_paper_dominance.png/pdf
- 190_32h_strategy_method_transfer_heatmap.png/pdf
- 191_32h_exact_policy_prediction_and_bootstrap.png/pdf

Run location
------------
Place this file directly under the project src folder and run it in PyCharm.
It auto-detects the project root by searching upward for outputs/tables.
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Global configuration
# =============================================================================

SCRIPT_NAME = "32h_threshold_sensitivity_and_anti_cherry_picking_audit.py"

BASE_STRATEGY = "RepeatedBalanced_Extended"
BASE_METHOD = "XGBoost"
BASE_ZONE = "Outside"
BASE_SCORE_COL = "NN_Risk"
BASE_THRESHOLD = 0.814815

OBS_COL_CANDIDATES = ["Observed", "y_true", "Actual", "氟去除率 (%)", "氟去除率"]
PRED_COL_CANDIDATES = ["Predicted", "y_pred", "Prediction"]
GROUP_COL_CANDIDATES = ["文献编号", "Paper", "Literature", "Group", "Source_Group"]
DATA_INDEX_CANDIDATES = ["Data_Index", "_Data_Index", "原始行号", "Local_Test_Index"]

MIN_N_FOR_MAIN = 30
MIN_COVERAGE_FOR_MAIN = 0.005
R2_TARGET = 0.50
BOOTSTRAP_N = 1000
RANDOM_STATE = 20260608

EST_BLUE = "#4C78A8"
EST_GREEN = "#6EA77A"
EST_ORANGE = "#D99A4E"
EST_RED = "#C44E52"
EST_GRAY = "#8F8F8F"
EST_LIGHT_GRAY = "#D9D9D9"
EST_DARK = "#2B2B2B"


# =============================================================================
# Basic utilities
# =============================================================================


def log(msg: str) -> None:
    print(msg, flush=True)


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find project root robustly for PyCharm/local testing."""
    candidates: List[Path] = []
    if start is not None:
        candidates.append(start.resolve())
    try:
        candidates.append(Path(__file__).resolve())
    except NameError:
        pass
    candidates.append(Path.cwd().resolve())

    checked: List[Path] = []
    for c in candidates:
        p = c if c.is_dir() else c.parent
        for parent in [p] + list(p.parents):
            checked.append(parent)
            if (parent / "outputs" / "tables").exists():
                return parent
            if (parent / "src").exists() and (parent / "outputs").exists():
                return parent
    # Fallback for this assistant-side test environment where files can be flat
    for p in [Path.cwd().resolve(), Path("/mnt/data")]:
        if (p / "32b_predictions_with_reliability_zone.csv").exists():
            return p
    return Path.cwd().resolve()


def ensure_dirs(root: Path) -> Tuple[Path, Path]:
    if (root / "outputs").exists() or root.name != "data":
        fig_dir = root / "outputs" / "figures"
        table_dir = root / "outputs" / "tables"
    else:
        fig_dir = root / "figures"
        table_dir = root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, table_dir


def find_existing_file(root: Path, names: Sequence[str]) -> Optional[Path]:
    search_dirs = [
        root / "outputs" / "tables",
        root / "outputs",
        root,
        Path.cwd(),
        Path("/mnt/data"),
    ]
    for d in search_dirs:
        for name in names:
            p = d / name
            if p.exists():
                return p
    # More flexible recursive fallback under project root, but not too broad.
    for name in names:
        try:
            hits = list(root.rglob(name))
            if hits:
                return hits[0]
        except Exception:
            pass
    return None


def read_csv_safely(path: Path) -> pd.DataFrame:
    log(f"[LOAD] {path}")
    return pd.read_csv(path, low_memory=False)


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Cannot find any of required columns: {candidates}")
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def safe_div(num: float, den: float) -> float:
    if den is None or den == 0 or not np.isfinite(den):
        return np.nan
    return num / den


def compute_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    n = int(len(y))
    if n == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "N": 0}
    err = y - p
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if n < 2 or denom <= 1e-12:
        r2 = np.nan
    else:
        r2 = float(1.0 - np.sum(err ** 2) / denom)
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "N": n}


def unique_row_count(df: pd.DataFrame, data_index_col: Optional[str]) -> int:
    if data_index_col is not None and data_index_col in df.columns:
        return int(df[data_index_col].nunique(dropna=True))
    return int(len(df))


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {path} | shape={df.shape}")


def setup_est_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
        "axes.edgecolor": EST_DARK,
        "axes.linewidth": 1.1,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 15,
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def polish_axis(ax: plt.Axes, grid: bool = True) -> None:
    if grid:
        ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
        ax.grid(True, axis="x", color="#EEEEEE", linewidth=0.4, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color(EST_DARK)
    ax.tick_params(direction="out", length=4, width=1.0, colors="black")


def save_fig(fig: plt.Figure, fig_dir: Path, name: str) -> None:
    png = fig_dir / name
    pdf = fig_dir / name.replace(".png", ".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] {png}")
    log(f"[FIG] {pdf}")


def annotate_bars(ax: plt.Axes, fmt: str = "{:.2f}", fontsize: int = 8) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    for patch in ax.patches:
        h = patch.get_height()
        if not np.isfinite(h):
            continue
        x = patch.get_x() + patch.get_width() / 2
        if h >= 0:
            y = h + 0.015 * span
            va = "bottom"
        else:
            y = h - 0.015 * span
            va = "top"
        ax.text(x, y, fmt.format(h), ha="center", va=va, fontsize=fontsize)


# =============================================================================
# Core analysis functions
# =============================================================================


def prepare_prediction_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    obs_col = first_existing_col(df, OBS_COL_CANDIDATES)
    pred_col = first_existing_col(df, PRED_COL_CANDIDATES)
    group_col = first_existing_col(df, GROUP_COL_CANDIDATES, required=False)
    data_index_col = first_existing_col(df, DATA_INDEX_CANDIDATES, required=False)

    required = ["Strategy", "Method", "Reliability_Zone", BASE_SCORE_COL, obs_col, pred_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in prediction table: {missing}")

    out = df.copy()
    out[obs_col] = safe_numeric(out[obs_col])
    out[pred_col] = safe_numeric(out[pred_col])
    out[BASE_SCORE_COL] = safe_numeric(out[BASE_SCORE_COL])
    out["Strategy_norm"] = out["Strategy"].map(norm_text)
    out["Method_norm"] = out["Method"].map(norm_text)
    out["Zone_norm"] = out["Reliability_Zone"].map(norm_text)
    if group_col:
        out[group_col] = out[group_col].map(norm_text)
    if data_index_col:
        out[data_index_col] = out[data_index_col]

    colinfo = {
        "obs_col": obs_col,
        "pred_col": pred_col,
        "group_col": group_col,
        "data_index_col": data_index_col,
    }
    return out, colinfo


def base_subset(df: pd.DataFrame, strategy: str, method: str, zone: Optional[str] = None) -> pd.DataFrame:
    mask = (df["Strategy_norm"] == norm_text(strategy)) & (df["Method_norm"] == norm_text(method))
    if zone is not None:
        mask &= (df["Zone_norm"].str.lower() == norm_text(zone).lower())
    return df.loc[mask].copy()


def evaluate_retained_subset(
    retained: pd.DataFrame,
    denominator_n: int,
    threshold: float,
    strategy: str,
    method: str,
    zone: str,
    colinfo: Dict[str, Optional[str]],
) -> Dict[str, object]:
    obs_col = colinfo["obs_col"]
    pred_col = colinfo["pred_col"]
    group_col = colinfo["group_col"]
    data_index_col = colinfo["data_index_col"]

    m = compute_metrics(retained[obs_col], retained[pred_col])
    n = int(m["N"])
    unique_n = unique_row_count(retained, data_index_col)
    coverage = safe_div(n, denominator_n)
    unique_coverage = safe_div(unique_n, denominator_n)

    if group_col and n > 0:
        vc = retained[group_col].value_counts(dropna=False)
        dominant_group = str(vc.index[0]) if len(vc) else ""
        dominant_n = int(vc.iloc[0]) if len(vc) else 0
        dominant_fraction = safe_div(dominant_n, n)
        n_groups = int(vc.shape[0])
    else:
        dominant_group = ""
        dominant_n = 0
        dominant_fraction = np.nan
        n_groups = 0

    return {
        "Strategy": strategy,
        "Method": method,
        "Zone": zone,
        "Score_Column": BASE_SCORE_COL,
        "Threshold": float(threshold),
        "R2": m["R2"],
        "RMSE": m["RMSE"],
        "MAE": m["MAE"],
        "N_Retained": n,
        "Unique_Row_N": unique_n,
        "Coverage_within_strategy_method": coverage,
        "Unique_Row_Coverage_within_strategy_method": unique_coverage,
        "N_Literature_Groups": n_groups,
        "Dominant_Group": dominant_group,
        "Dominant_Group_N": dominant_n,
        "Dominant_Group_Fraction": dominant_fraction,
        "Pass_R2_0p5": bool(np.isfinite(m["R2"]) and m["R2"] >= R2_TARGET),
        "Pass_Min_N": bool(n >= MIN_N_FOR_MAIN),
        "Pass_Min_Coverage": bool(np.isfinite(coverage) and coverage >= MIN_COVERAGE_FOR_MAIN),
    }


def evaluate_threshold_sensitivity(
    df: pd.DataFrame,
    colinfo: Dict[str, Optional[str]],
    strategy: str = BASE_STRATEGY,
    method: str = BASE_METHOD,
    zone: str = BASE_ZONE,
    base_threshold: float = BASE_THRESHOLD,
) -> pd.DataFrame:
    subset_all = base_subset(df, strategy, method, zone=None)
    subset_zone = base_subset(df, strategy, method, zone=zone)
    denominator_n = int(len(subset_all))
    if denominator_n == 0:
        raise ValueError(f"No rows found for strategy={strategy}, method={method}")
    if len(subset_zone) == 0:
        raise ValueError(f"No rows found for strategy={strategy}, method={method}, zone={zone}")

    score = subset_zone[BASE_SCORE_COL].dropna().to_numpy(dtype=float)
    if len(score) == 0:
        raise ValueError(f"No valid {BASE_SCORE_COL} values in selected zone subset")

    coarse = np.round(np.arange(0.50, 1.0001, 0.025), 6)
    local = np.round(np.arange(base_threshold - 0.080, base_threshold + 0.0801, 0.005), 6)
    quant = np.round(np.quantile(score, np.linspace(0.02, 0.98, 25)), 6)
    thresholds = sorted(set([float(x) for x in np.concatenate([coarse, local, quant, [base_threshold]]) if np.isfinite(x)]))
    thresholds = [t for t in thresholds if t >= 0 and t <= 1.5]

    rows: List[Dict[str, object]] = []
    for t in thresholds:
        retained = subset_zone.loc[subset_zone[BASE_SCORE_COL] <= t].copy()
        rows.append(evaluate_retained_subset(retained, denominator_n, t, strategy, method, zone, colinfo))
    res = pd.DataFrame(rows).sort_values("Threshold").reset_index(drop=True)
    res["Is_Exact_Base_Threshold"] = np.isclose(res["Threshold"].astype(float), base_threshold, atol=1e-9)
    return res


def split_level_metrics(
    df: pd.DataFrame,
    colinfo: Dict[str, Optional[str]],
    strategy: str,
    method: str,
    zone: str,
    threshold: float,
) -> pd.DataFrame:
    subset_all = base_subset(df, strategy, method, zone=None)
    subset_zone = base_subset(df, strategy, method, zone=zone)
    retained = subset_zone.loc[subset_zone[BASE_SCORE_COL] <= threshold].copy()
    obs_col = colinfo["obs_col"]
    pred_col = colinfo["pred_col"]
    group_col = colinfo["group_col"]
    data_index_col = colinfo["data_index_col"]

    if "Split_ID" not in df.columns:
        retained["Split_ID"] = "all"
        subset_all["Split_ID"] = "all"

    denom_by_split = subset_all.groupby("Split_ID", dropna=False).size().to_dict()
    rows: List[Dict[str, object]] = []
    for split_id, g in retained.groupby("Split_ID", dropna=False):
        m = compute_metrics(g[obs_col], g[pred_col])
        denom = int(denom_by_split.get(split_id, len(g)))
        row = {
            "Strategy": strategy,
            "Method": method,
            "Zone": zone,
            "Threshold": float(threshold),
            "Split_ID": split_id,
            "R2": m["R2"],
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
            "N_Retained": int(m["N"]),
            "Unique_Row_N": unique_row_count(g, data_index_col),
            "Coverage_within_split": safe_div(int(m["N"]), denom),
            "Pass_R2_0p5": bool(np.isfinite(m["R2"]) and m["R2"] >= R2_TARGET),
        }
        if group_col:
            vc = g[group_col].value_counts(dropna=False)
            row["Dominant_Group"] = str(vc.index[0]) if len(vc) else ""
            row["Dominant_Group_N"] = int(vc.iloc[0]) if len(vc) else 0
            row["Dominant_Group_Fraction"] = safe_div(row["Dominant_Group_N"], int(m["N"]))
            row["N_Literature_Groups"] = int(vc.shape[0])
        rows.append(row)
    return pd.DataFrame(rows)


def paper_dominance_by_threshold(
    df: pd.DataFrame,
    colinfo: Dict[str, Optional[str]],
    thresholds: Sequence[float],
    strategy: str = BASE_STRATEGY,
    method: str = BASE_METHOD,
    zone: str = BASE_ZONE,
) -> pd.DataFrame:
    group_col = colinfo["group_col"]
    if not group_col:
        return pd.DataFrame()
    subset_all = base_subset(df, strategy, method, zone=None)
    subset_zone = base_subset(df, strategy, method, zone=zone)
    denominator_n = int(len(subset_all))
    rows: List[Dict[str, object]] = []
    for t in thresholds:
        ret = subset_zone.loc[subset_zone[BASE_SCORE_COL] <= t]
        if len(ret) == 0:
            continue
        vc = ret[group_col].value_counts(dropna=False)
        for rank, (group, n) in enumerate(vc.head(10).items(), start=1):
            rows.append({
                "Threshold": float(t),
                "Rank": rank,
                "Literature_Group": group,
                "N": int(n),
                "Fraction_within_retained": safe_div(int(n), len(ret)),
                "Coverage_within_strategy_method": safe_div(len(ret), denominator_n),
                "Total_Retained_N": int(len(ret)),
            })
    return pd.DataFrame(rows)


def find_threshold_plateaus(threshold_df: pd.DataFrame) -> pd.DataFrame:
    if threshold_df.empty:
        return pd.DataFrame()
    df = threshold_df.sort_values("Threshold").copy().reset_index(drop=True)
    ok = (
        (df["R2"] >= R2_TARGET) &
        (df["N_Retained"] >= MIN_N_FOR_MAIN) &
        (df["Coverage_within_strategy_method"] >= MIN_COVERAGE_FOR_MAIN)
    ).fillna(False).to_numpy()
    rows: List[Dict[str, object]] = []
    start = None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = i
        if (not flag or i == len(ok) - 1) and start is not None:
            end = i - 1 if not flag else i
            part = df.iloc[start:end + 1]
            rows.append({
                "Plateau_ID": f"P{len(rows)+1:02d}",
                "Threshold_Min": float(part["Threshold"].min()),
                "Threshold_Max": float(part["Threshold"].max()),
                "Threshold_Count": int(len(part)),
                "R2_Min": float(part["R2"].min()),
                "R2_Median": float(part["R2"].median()),
                "R2_Max": float(part["R2"].max()),
                "RMSE_Median": float(part["RMSE"].median()),
                "MAE_Median": float(part["MAE"].median()),
                "N_Min": int(part["N_Retained"].min()),
                "N_Max": int(part["N_Retained"].max()),
                "Coverage_Min": float(part["Coverage_within_strategy_method"].min()),
                "Coverage_Max": float(part["Coverage_within_strategy_method"].max()),
                "Dominant_Group_Mode": part["Dominant_Group"].mode().iloc[0] if not part["Dominant_Group"].mode().empty else "",
                "Dominant_Group_Fraction_Median": float(part["Dominant_Group_Fraction"].median()),
            })
            start = None
    return pd.DataFrame(rows)


def strategy_method_transfer(
    df: pd.DataFrame,
    colinfo: Dict[str, Optional[str]],
    threshold: float = BASE_THRESHOLD,
    zone: str = BASE_ZONE,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    strategies = sorted(df["Strategy_norm"].dropna().unique().tolist())
    methods = sorted(df["Method_norm"].dropna().unique().tolist())
    for strategy in strategies:
        for method in methods:
            subset_all = base_subset(df, strategy, method, zone=None)
            subset_zone = base_subset(df, strategy, method, zone=zone)
            if len(subset_all) == 0 or len(subset_zone) == 0:
                continue
            retained = subset_zone.loc[subset_zone[BASE_SCORE_COL] <= threshold]
            if len(retained) == 0:
                row = {
                    "Strategy": strategy,
                    "Method": method,
                    "Zone": zone,
                    "Threshold": float(threshold),
                    "R2": np.nan,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "N_Retained": 0,
                    "Coverage_within_strategy_method": 0.0,
                    "Dominant_Group": "",
                    "Dominant_Group_Fraction": np.nan,
                }
            else:
                row = evaluate_retained_subset(retained, len(subset_all), threshold, strategy, method, zone, colinfo)
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_exact_policy(
    retained: pd.DataFrame,
    obs_col: str,
    pred_col: str,
    n_boot: int = BOOTSTRAP_N,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    y = safe_numeric(retained[obs_col]).to_numpy(dtype=float)
    p = safe_numeric(retained[pred_col]).to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    n = len(y)
    rows: List[Dict[str, float]] = []
    if n < 5:
        return pd.DataFrame()
    rng = np.random.default_rng(random_state)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(y[idx], p[idx])
        rows.append({"Bootstrap_ID": i + 1, **m})
    return pd.DataFrame(rows)


def summarize_bootstrap(boot: pd.DataFrame) -> pd.DataFrame:
    if boot.empty:
        return pd.DataFrame()
    rows = []
    for metric in ["R2", "RMSE", "MAE"]:
        vals = pd.to_numeric(boot[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        rows.append({
            "Metric": metric,
            "Mean": float(np.mean(vals)),
            "Median": float(np.median(vals)),
            "CI_2p5": float(np.quantile(vals, 0.025)),
            "CI_97p5": float(np.quantile(vals, 0.975)),
            "N_Bootstrap": int(len(vals)),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Plotting functions
# =============================================================================


def plot_threshold_sensitivity_core(thr: pd.DataFrame, fig_dir: Path) -> None:
    setup_est_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    fig.suptitle("32H threshold sensitivity of the selected target-free AD rule", y=0.985, fontweight="bold")

    x = thr["Threshold"].to_numpy(dtype=float)

    ax = axes[0, 0]
    ax.plot(x, thr["R2"], marker="o", markersize=4, linewidth=1.6, color=EST_BLUE)
    ax.axhline(R2_TARGET, color=EST_RED, linestyle="--", linewidth=1.3, label=r"$R^2=0.5$")
    ax.axvline(BASE_THRESHOLD, color=EST_DARK, linestyle=":", linewidth=1.2, label="P08 threshold")
    ax.set_xlabel("NN_Risk threshold")
    ax.set_ylabel(r"Retained $R^2$")
    ax.set_title(r"$R^2$ vs threshold")
    ax.legend(frameon=True, loc="best")
    polish_axis(ax)

    ax = axes[0, 1]
    ax.plot(x, thr["RMSE"], marker="o", markersize=4, linewidth=1.6, color=EST_ORANGE)
    ax.axvline(BASE_THRESHOLD, color=EST_DARK, linestyle=":", linewidth=1.2)
    ax.set_xlabel("NN_Risk threshold")
    ax.set_ylabel("Retained RMSE")
    ax.set_title("RMSE vs threshold")
    polish_axis(ax)

    ax = axes[1, 0]
    ax.plot(x, thr["N_Retained"], marker="o", markersize=4, linewidth=1.6, color=EST_GREEN, label="N retained")
    ax2 = ax.twinx()
    ax2.plot(x, thr["Coverage_within_strategy_method"], marker="s", markersize=3.5, linewidth=1.4, color=EST_BLUE, label="Coverage")
    ax.axvline(BASE_THRESHOLD, color=EST_DARK, linestyle=":", linewidth=1.2)
    ax.set_xlabel("NN_Risk threshold")
    ax.set_ylabel("N retained")
    ax2.set_ylabel("Coverage")
    ax.set_title("Coverage and retained sample size")
    polish_axis(ax)
    ax2.grid(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=True, loc="upper left")

    ax = axes[1, 1]
    ax.plot(x, thr["Dominant_Group_Fraction"], marker="o", markersize=4, linewidth=1.6, color=EST_RED, label="Dominant group fraction")
    ax.plot(x, thr["N_Literature_Groups"], marker="s", markersize=3.5, linewidth=1.3, color=EST_GRAY, label="N literature groups")
    ax.axvline(BASE_THRESHOLD, color=EST_DARK, linestyle=":", linewidth=1.2)
    ax.set_xlabel("NN_Risk threshold")
    ax.set_ylabel("Dominance / group count")
    ax.set_title("Literature dominance vs threshold")
    ax.legend(frameon=True, loc="best")
    polish_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, fig_dir, "188_32h_threshold_sensitivity_core.png")


def plot_paper_dominance(thr_dom: pd.DataFrame, fig_dir: Path) -> None:
    if thr_dom.empty:
        return
    setup_est_style()
    top = thr_dom[thr_dom["Rank"] == 1].copy().sort_values("Threshold")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.plot(top["Threshold"], top["Fraction_within_retained"], marker="o", color=EST_RED, linewidth=1.8)
    ax.axvline(BASE_THRESHOLD, color=EST_DARK, linestyle=":", linewidth=1.2)
    ax.axhline(0.8, color=EST_GRAY, linestyle="--", linewidth=1.0)
    ax.set_xlabel("NN_Risk threshold")
    ax.set_ylabel("Dominant literature-group fraction")
    ax.set_title("32H retained-subdomain literature dominance")
    for _, r in top.iloc[::max(1, len(top)//8)].iterrows():
        ax.text(r["Threshold"], r["Fraction_within_retained"] + 0.025, str(r["Literature_Group"]), fontsize=8, ha="center")
    polish_axis(ax)
    fig.tight_layout()
    save_fig(fig, fig_dir, "189_32h_threshold_paper_dominance.png")


def plot_strategy_method_transfer_heatmap(transfer: pd.DataFrame, fig_dir: Path) -> None:
    if transfer.empty:
        return
    setup_est_style()
    pivot = transfer.pivot_table(index="Method", columns="Strategy", values="R2", aggfunc="mean")
    preferred_methods = ["XGBoost", "CB-XGB", "CatBoost", "KRR_Poly", "KRR_CatBoost_Weighted_2Model", "KRR_XGBoost_Weighted_2Model", "GPR_Matern", "PLS"]
    row_order = [m for m in preferred_methods if m in pivot.index] + [m for m in pivot.index if m not in preferred_methods]
    col_order = [c for c in ["Balanced_5Seed_Current31F", "RepeatedBalanced_Extended", "RepeatedOrdinary_GroupedSplit", "GroupKFold_FullyGrouped", "LOLO_LeaveOneLiteratureOut"] if c in pivot.columns]
    col_order += [c for c in pivot.columns if c not in col_order]
    pivot = pivot.loc[row_order, col_order]

    fig_w = max(8.5, 1.4 * len(pivot.columns) + 4)
    fig_h = max(4.8, 0.45 * len(pivot.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdYlBu_r", vmin=-0.5, vmax=0.8, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_title("Transfer of the P08 rule threshold across strategies and models")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(r"Retained $R^2$")
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    fig.tight_layout()
    save_fig(fig, fig_dir, "190_32h_strategy_method_transfer_heatmap.png")


def plot_exact_prediction_and_bootstrap(retained: pd.DataFrame, boot_summary: pd.DataFrame, colinfo: Dict[str, Optional[str]], fig_dir: Path) -> None:
    setup_est_style()
    obs_col = colinfo["obs_col"]
    pred_col = colinfo["pred_col"]
    metrics = compute_metrics(retained[obs_col], retained[pred_col])

    fig = plt.figure(figsize=(12.5, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    y = safe_numeric(retained[obs_col])
    p = safe_numeric(retained[pred_col])
    ax.scatter(y, p, s=38, alpha=0.78, color=EST_BLUE, edgecolor="white", linewidth=0.35)
    ax.plot([0, 100], [0, 100], linestyle="--", color=EST_RED, linewidth=1.4)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Predicted F removal (%)")
    ax.set_title("Exact P08 retained prediction panel")
    text = (
        rf"$R^2$ = {metrics['R2']:.3f}\n"
        f"RMSE = {metrics['RMSE']:.2f}\n"
        f"MAE = {metrics['MAE']:.2f}\n"
        f"n = {metrics['N']}\n"
        f"rule: Outside & NN_Risk <= {BASE_THRESHOLD:.6f}"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=EST_GRAY, alpha=0.92))
    polish_axis(ax)

    ax = fig.add_subplot(gs[0, 1])
    if not boot_summary.empty:
        metrics_order = ["R2", "RMSE", "MAE"]
        sub = boot_summary.set_index("Metric").reindex(metrics_order).dropna(how="all")
        x = np.arange(len(sub))
        yvals = sub["Median"].to_numpy(dtype=float)
        lower = yvals - sub["CI_2p5"].to_numpy(dtype=float)
        upper = sub["CI_97p5"].to_numpy(dtype=float) - yvals
        ax.bar(x, yvals, color=[EST_GREEN, EST_BLUE, EST_ORANGE][:len(x)], edgecolor=EST_DARK, linewidth=0.8)
        ax.errorbar(x, yvals, yerr=[lower, upper], fmt="none", ecolor=EST_DARK, elinewidth=1.1, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index.tolist())
        ax.set_ylabel("Bootstrap median with 95% CI")
        ax.set_title("Bootstrap uncertainty")
        for i, val in enumerate(yvals):
            ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Bootstrap skipped\ninsufficient retained samples", ha="center", va="center")
        ax.set_axis_off()
    polish_axis(ax)
    fig.suptitle("32H exact-policy audit: prediction and uncertainty", y=1.02, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, fig_dir, "191_32h_exact_policy_prediction_and_bootstrap.png")


# =============================================================================
# Report
# =============================================================================


def write_report(
    path: Path,
    exact_metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    paper_dom: pd.DataFrame,
    plateaus: pd.DataFrame,
    transfer: pd.DataFrame,
    boot_summary: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("32H threshold sensitivity and anti-cherry-picking audit")
    lines.append("=" * 92)
    lines.append("")
    lines.append("Audited target-free policy")
    lines.append("-" * 92)
    lines.append(f"Strategy: {BASE_STRATEGY}")
    lines.append(f"Method: {BASE_METHOD}")
    lines.append(f"Rule: Reliability_Zone == {BASE_ZONE} AND {BASE_SCORE_COL} <= {BASE_THRESHOLD:.6f}")
    lines.append(f"Minimum N for main interpretation: {MIN_N_FOR_MAIN}")
    lines.append(f"Minimum coverage for main interpretation: {MIN_COVERAGE_FOR_MAIN}")
    lines.append("")

    if not exact_metrics.empty:
        r = exact_metrics.iloc[0]
        lines.append("Exact P08 reconstruction")
        lines.append("-" * 92)
        lines.append(f"R2: {r['R2']:.6f}")
        lines.append(f"RMSE: {r['RMSE']:.6f}")
        lines.append(f"MAE: {r['MAE']:.6f}")
        lines.append(f"N retained: {int(r['N_Retained'])}")
        lines.append(f"Unique row N: {int(r['Unique_Row_N'])}")
        lines.append(f"Coverage within strategy-method subset: {r['Coverage_within_strategy_method']:.6f}")
        lines.append(f"Dominant group: {r['Dominant_Group']}")
        lines.append(f"Dominant group fraction: {r['Dominant_Group_Fraction']:.6f}")
        lines.append(f"Pass R2 >= 0.5: {bool(r['Pass_R2_0p5'])}")
        lines.append(f"Pass minimum N: {bool(r['Pass_Min_N'])}")
        lines.append(f"Pass minimum coverage: {bool(r['Pass_Min_Coverage'])}")
        lines.append("")

    if not split_metrics.empty:
        lines.append("Split-level stability at exact P08 threshold")
        lines.append("-" * 92)
        n_splits = int(split_metrics.shape[0])
        n_pass = int(split_metrics["Pass_R2_0p5"].sum()) if "Pass_R2_0p5" in split_metrics else 0
        lines.append(f"Number of retained splits: {n_splits}")
        lines.append(f"Splits with R2 >= 0.5: {n_pass}/{n_splits}")
        lines.append(f"Median split R2: {split_metrics['R2'].median():.6f}")
        lines.append(f"Minimum split R2: {split_metrics['R2'].min():.6f}")
        lines.append(f"Minimum split retained N: {int(split_metrics['N_Retained'].min())}")
        lines.append("Split metrics:")
        for _, r in split_metrics.iterrows():
            lines.append(
                f"  {r['Split_ID']}: R2={r['R2']:.6f}, RMSE={r['RMSE']:.6f}, "
                f"MAE={r['MAE']:.6f}, N={int(r['N_Retained'])}"
            )
        lines.append("")

    if not plateaus.empty:
        lines.append("Threshold plateau summary")
        lines.append("-" * 92)
        lines.append(f"Number of threshold plateaus satisfying R2 >= 0.5, N >= {MIN_N_FOR_MAIN}, coverage >= {MIN_COVERAGE_FOR_MAIN}: {len(plateaus)}")
        for _, r in plateaus.iterrows():
            lines.append(
                f"  {r['Plateau_ID']}: threshold {r['Threshold_Min']:.6f}-{r['Threshold_Max']:.6f}, "
                f"R2 median={r['R2_Median']:.6f}, N={int(r['N_Min'])}-{int(r['N_Max'])}, "
                f"dominant={r['Dominant_Group_Mode']} (median fraction={r['Dominant_Group_Fraction_Median']:.3f})"
            )
        lines.append("")
    else:
        lines.append("Threshold plateau summary")
        lines.append("-" * 92)
        lines.append("No threshold plateau satisfies the predefined R2/N/coverage criteria.")
        lines.append("")

    if not paper_dom.empty:
        lines.append("Literature dominance at exact P08 threshold")
        lines.append("-" * 92)
        exact_dom = paper_dom[np.isclose(paper_dom["Threshold"].astype(float), BASE_THRESHOLD, atol=1e-9)]
        if not exact_dom.empty:
            for _, r in exact_dom.head(10).iterrows():
                lines.append(
                    f"  rank {int(r['Rank'])}: {r['Literature_Group']}, "
                    f"N={int(r['N'])}, fraction={r['Fraction_within_retained']:.6f}"
                )
        lines.append("")

    if not transfer.empty:
        lines.append("Transfer of exact P08 threshold to other strategies/models")
        lines.append("-" * 92)
        ranked = transfer.copy()
        ranked = ranked[(ranked["N_Retained"] >= MIN_N_FOR_MAIN) & ranked["R2"].notna()]
        ranked = ranked.sort_values(["R2", "N_Retained"], ascending=[False, False]).head(12)
        if ranked.empty:
            lines.append("No transferred strategy/model subset satisfies the minimum retained sample criterion.")
        else:
            for _, r in ranked.iterrows():
                lines.append(
                    f"  {r['Strategy']} | {r['Method']}: R2={r['R2']:.6f}, "
                    f"RMSE={r['RMSE']:.6f}, MAE={r['MAE']:.6f}, "
                    f"N={int(r['N_Retained'])}, coverage={r['Coverage_within_strategy_method']:.6f}, "
                    f"dominant={r.get('Dominant_Group','')} ({r.get('Dominant_Group_Fraction', np.nan):.3f})"
                )
        lines.append("")

    if not boot_summary.empty:
        lines.append("Bootstrap uncertainty for exact P08 retained subset")
        lines.append("-" * 92)
        for _, r in boot_summary.iterrows():
            lines.append(
                f"  {r['Metric']}: median={r['Median']:.6f}, "
                f"95% CI=[{r['CI_2p5']:.6f}, {r['CI_97p5']:.6f}]"
            )
        lines.append("")

    lines.append("Interpretation decision")
    lines.append("-" * 92)
    exact_ok = False
    split_ok = False
    dominance_warning = True
    plateau_ok = not plateaus.empty
    if not exact_metrics.empty:
        rr = exact_metrics.iloc[0]
        exact_ok = bool(rr["Pass_R2_0p5"] and rr["Pass_Min_N"] and rr["Pass_Min_Coverage"])
        dominance_warning = bool(pd.notna(rr["Dominant_Group_Fraction"]) and rr["Dominant_Group_Fraction"] >= 0.80)
    if not split_metrics.empty:
        split_ok = bool(split_metrics["Pass_R2_0p5"].mean() >= 0.50 and split_metrics["N_Retained"].min() >= 5)

    if exact_ok and plateau_ok and not dominance_warning and split_ok:
        decision = "Strong retained-subdomain evidence with acceptable threshold and split stability."
    elif exact_ok and plateau_ok and dominance_warning:
        decision = (
            "Chemistry/literature-specific retained-subdomain evidence. The R2 target is met, "
            "but literature dominance is high, so this must not be described as universal cross-literature generalization."
        )
    elif exact_ok:
        decision = (
            "Exploratory retained-subdomain evidence. The exact threshold meets the numeric R2 criterion, "
            "but threshold/split/dominance limitations require cautious wording."
        )
    else:
        decision = "The exact audited policy does not satisfy the predefined retained-subdomain interpretation criteria."
    lines.append(decision)
    lines.append("")
    lines.append("Recommended manuscript wording")
    lines.append("-" * 92)
    lines.append(
        "This result should be reported as a target-free selective AD retained-subdomain result. "
        "It should not be reported as all-domain grouped-validation performance. If literature dominance remains high, "
        "the subdomain should be described as a chemistry/literature-specific high-confidence operating window."
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[REPORT] {path}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    t0 = time.time()
    log("=" * 92)
    log(f"Running {SCRIPT_NAME}")
    root = find_project_root()
    fig_dir, table_dir = ensure_dirs(root)
    log(f"Project root: {root}")
    log(f"Figure dir: {fig_dir}")
    log(f"Table dir: {table_dir}")
    log("=" * 92)

    pred_path = find_existing_file(root, ["32b_predictions_with_reliability_zone.csv"])
    if pred_path is None:
        raise FileNotFoundError(
            "Cannot find required file: 32b_predictions_with_reliability_zone.csv. "
            "Please make sure Step 32B outputs are in outputs/tables."
        )

    raw = read_csv_safely(pred_path)
    df, colinfo = prepare_prediction_table(raw)
    log(f"[INFO] prediction table shape: {df.shape}")
    log(f"[INFO] observed column: {colinfo['obs_col']}")
    log(f"[INFO] predicted column: {colinfo['pred_col']}")
    log(f"[INFO] group column: {colinfo['group_col']}")
    log(f"[INFO] data-index column: {colinfo['data_index_col']}")

    log("\n[STEP] Threshold sensitivity around exact P08 rule...")
    thr = evaluate_threshold_sensitivity(df, colinfo)
    save_table(thr, table_dir / "32h_threshold_sensitivity_all.csv")

    exact = thr.loc[np.isclose(thr["Threshold"].astype(float), BASE_THRESHOLD, atol=1e-9)].copy()
    if exact.empty:
        # Use nearest threshold if exact is absent, although exact is intentionally included above.
        idx = (thr["Threshold"].astype(float) - BASE_THRESHOLD).abs().idxmin()
        exact = thr.loc[[idx]].copy()
    save_table(exact, table_dir / "32h_exact_policy_metrics.csv")

    log("\n[STEP] Split-level stability of exact P08 rule...")
    split = split_level_metrics(df, colinfo, BASE_STRATEGY, BASE_METHOD, BASE_ZONE, BASE_THRESHOLD)
    save_table(split, table_dir / "32h_exact_policy_split_metrics.csv")

    log("\n[STEP] Literature dominance by threshold...")
    dom_thresholds = sorted(set(np.concatenate([
        thr["Threshold"].to_numpy(dtype=float),
        np.array([BASE_THRESHOLD], dtype=float),
    ])))
    paper_dom = paper_dominance_by_threshold(df, colinfo, dom_thresholds)
    save_table(paper_dom, table_dir / "32h_exact_policy_paper_dominance.csv")

    log("\n[STEP] Threshold plateau detection...")
    plateaus = find_threshold_plateaus(thr)
    save_table(plateaus, table_dir / "32h_threshold_plateau_summary.csv")

    log("\n[STEP] Transfer exact threshold to other strategies/models...")
    transfer = strategy_method_transfer(df, colinfo, BASE_THRESHOLD, BASE_ZONE)
    save_table(transfer, table_dir / "32h_strategy_method_transfer_summary.csv")

    log("\n[STEP] Bootstrap uncertainty of exact P08 retained subset...")
    exact_subset = base_subset(df, BASE_STRATEGY, BASE_METHOD, zone=BASE_ZONE)
    exact_retained = exact_subset.loc[exact_subset[BASE_SCORE_COL] <= BASE_THRESHOLD].copy()
    boot = bootstrap_exact_policy(exact_retained, colinfo["obs_col"], colinfo["pred_col"], n_boot=BOOTSTRAP_N)
    boot_summary = summarize_bootstrap(boot)
    if not boot.empty:
        save_table(boot, table_dir / "32h_bootstrap_uncertainty_exact_policy_raw.csv")
    save_table(boot_summary, table_dir / "32h_bootstrap_uncertainty_exact_policy.csv")

    log("\n[STEP] Creating EST-style figures...")
    plot_threshold_sensitivity_core(thr, fig_dir)
    plot_paper_dominance(paper_dom, fig_dir)
    plot_strategy_method_transfer_heatmap(transfer, fig_dir)
    plot_exact_prediction_and_bootstrap(exact_retained, boot_summary, colinfo, fig_dir)

    report_path = table_dir / "32h_threshold_sensitivity_decision_report.txt"
    write_report(report_path, exact, split, paper_dom, plateaus, transfer, boot_summary)

    elapsed = time.time() - t0
    log("=" * 92)
    log("32H threshold sensitivity and anti-cherry-picking audit finished.")
    log(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    log("Please send me the report and the main sensitivity table for interpretation:")
    log(f"  {table_dir / '32h_threshold_sensitivity_decision_report.txt'}")
    log(f"  {table_dir / '32h_threshold_sensitivity_all.csv'}")
    log("=" * 92)


if __name__ == "__main__":
    main()
