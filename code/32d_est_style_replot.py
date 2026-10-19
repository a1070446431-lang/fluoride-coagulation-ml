# -*- coding: utf-8 -*-
"""
32d_est_style_replot.py

Purpose
-------
Re-plot Step 32D outputs in a cleaner EST-like journal style.

This script DOES NOT retrain any model and DOES NOT change numerical results.
It only reads existing Step 32D CSV outputs and regenerates publication-style
figures with cleaner typography, spacing, label handling, and low-saturation
colors.

Expected input files under outputs/tables:
- 32d_high_r2_policy_summary.csv
- 32d_best_policy_retained_predictions.csv
- 32d_best_policy_split_stability.csv

Output figures under outputs/figures:
- 165_32d_est_policy_metrics.png/pdf
- 166_32d_est_r2_coverage_frontier.png/pdf
- 167_32d_est_best_policy_prediction_panel.png/pdf
- 168_32d_est_best_policy_split_stability.png/pdf
"""

from __future__ import annotations

import time
import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


T0 = time.time()


def find_project_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path(__file__).resolve().parent
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "outputs").exists() and (p / "data").exists():
            return p
    for p in [start] + list(start.parents):
        if (p / "outputs").exists():
            return p
    if start.name.lower() == "src":
        return start.parent
    return start


ROOT = find_project_root()
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


log("=" * 90)
log("Running 32d_est_style_replot.py")
log(f"Project root: {ROOT}")
log("=" * 90)


def first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def read_csv_smart(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    low_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in low_map:
            return low_map[cand.lower()]
    if required:
        raise KeyError(
            "Cannot find required column. Candidates="
            + str(candidates)
            + "\nAvailable columns="
            + str(list(df.columns))
        )
    return None


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_label(x: object, width: int = 28) -> str:
    s = str(x).replace("_", " ").replace("  ", " ")
    return textwrap.shorten(s, width=width, placeholder="...")


def add_bar_labels(ax, fmt: str = "{:.2f}", fontsize: int = 7, dy: float = 0.01) -> None:
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 1.0
    for patch in ax.patches:
        h = patch.get_height()
        if not np.isfinite(h):
            continue
        x = patch.get_x() + patch.get_width() / 2
        if h >= 0:
            y = h + dy * yr
            va = "bottom"
        else:
            y = h - dy * yr
            va = "top"
        ax.text(x, y, fmt.format(h), ha="center", va=va, fontsize=fontsize)


def metric_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def metric_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def set_est_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.55,
    })


set_est_style()

COLORS = {
    "blue": "#4C78A8",
    "light_blue": "#86A9C7",
    "green": "#6FA57B",
    "orange": "#D99B52",
    "purple": "#9A7BB8",
    "gray": "#9A9A9A",
    "light_gray": "#D0D0D0",
    "dark_gray": "#4A4A4A",
    "red": "#C44E52",
}


def finish_axes(ax) -> None:
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(direction="out", length=3.2, width=0.7, colors="#222222")


def save_fig(fig: plt.Figure, basename: str) -> None:
    png = FIG_DIR / f"{basename}.png"
    pdf = FIG_DIR / f"{basename}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    log(f"[FIG] {png}")
    log(f"[FIG] {pdf}")


summary_path = first_existing([
    TABLE_DIR / "32d_high_r2_policy_summary.csv",
    TABLE_DIR / "32d_candidate_policy_summary.csv",
    TABLE_DIR / "32d_policy_summary.csv",
])
pred_path = first_existing([
    TABLE_DIR / "32d_best_policy_retained_predictions.csv",
    TABLE_DIR / "32d_best_retained_predictions.csv",
])
split_path = first_existing([
    TABLE_DIR / "32d_best_policy_split_stability.csv",
    TABLE_DIR / "32d_split_stability.csv",
])

if summary_path is None:
    raise FileNotFoundError(
        "Cannot find 32D policy summary CSV under outputs/tables. "
        "Expected 32d_high_r2_policy_summary.csv."
    )

summary = read_csv_smart(summary_path)
log(f"[TABLE] loaded {summary_path} | shape={summary.shape}")

if pred_path is not None:
    pred_df = read_csv_smart(pred_path)
    log(f"[TABLE] loaded {pred_path} | shape={pred_df.shape}")
else:
    pred_df = None
    log("[WARN] Best-policy retained-prediction table not found; prediction panel will be skipped.")

if split_path is not None:
    split_df = read_csv_smart(split_path)
    log(f"[TABLE] loaded {split_path} | shape={split_df.shape}")
else:
    split_df = None
    log("[WARN] Split-level stability table not found; stability figure will be skipped.")

r2_col = find_col(summary, ["R2", "Retained_R2", "R_squared"])
rmse_col = find_col(summary, ["RMSE", "Retained_RMSE"])
mae_col = find_col(summary, ["MAE", "Retained_MAE"])
n_col = find_col(summary, ["N_Retained", "n", "N", "Retained_N"], required=False)
cov_col = find_col(summary, ["Coverage_Within_Strategy_Method", "Coverage", "coverage"], required=False)
method_col = find_col(summary, ["Method", "Model", "method"], required=False)
strategy_col = find_col(summary, ["Strategy", "Validation_Strategy", "strategy"], required=False)
rule_col = find_col(summary, ["Rule_Name", "Rule", "Policy", "Rule_Description"], required=False)
main_col = find_col(summary, ["Publication_Candidate", "MainCandidate", "Main_Candidate"], required=False)

for c in [r2_col, rmse_col, mae_col, n_col, cov_col]:
    if c is not None:
        summary[c] = to_num(summary[c])

summary = summary.replace([np.inf, -np.inf], np.nan)
summary = summary.dropna(subset=[r2_col, rmse_col, mae_col]).copy()
summary = summary.sort_values(r2_col, ascending=False).reset_index(drop=True)

n_show = min(12, len(summary))
plot_df = summary.head(n_show).copy()
plot_df["Policy_ID"] = [f"P{i:02d}" for i in range(1, len(plot_df) + 1)]


def compose_policy_desc(row: pd.Series) -> str:
    parts = []
    if strategy_col is not None:
        parts.append(f"Strategy={row.get(strategy_col, '')}")
    if method_col is not None:
        parts.append(f"Method={row.get(method_col, '')}")
    if rule_col is not None:
        parts.append(f"Rule={row.get(rule_col, '')}")
    if n_col is not None:
        parts.append(f"n={row.get(n_col, np.nan):.0f}")
    if cov_col is not None:
        parts.append(f"coverage={row.get(cov_col, np.nan):.3f}")
    parts.append(f"R2={row.get(r2_col, np.nan):.4f}")
    parts.append(f"RMSE={row.get(rmse_col, np.nan):.3f}")
    parts.append(f"MAE={row.get(mae_col, np.nan):.3f}")
    return " | ".join(parts)


mapping = plot_df[["Policy_ID"]].copy()
mapping["Policy_Description"] = plot_df.apply(compose_policy_desc, axis=1)
mapping_out = TABLE_DIR / "32d_est_policy_label_mapping.csv"
mapping.to_csv(mapping_out, index=False, encoding="utf-8-sig")
log(f"[TABLE] {mapping_out} | shape={mapping.shape}")


def plot_policy_metrics() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), constrained_layout=True)
    axes = axes.ravel()

    x = np.arange(len(plot_df))
    labels = plot_df["Policy_ID"].tolist()

    metric_specs = [
        (r2_col, "Retained $R^2$", "$R^2$", "{:.3f}", True),
        (rmse_col, "Retained RMSE", "RMSE", "{:.2f}", False),
        (mae_col, "Retained MAE", "MAE", "{:.2f}", False),
        (cov_col, "Coverage", "Coverage", "{:.2f}", False),
    ]

    for ax, (col, title, ylabel, fmt, is_r2) in zip(axes, metric_specs):
        if col is None:
            ax.axis("off")
            continue

        vals = plot_df[col].to_numpy(dtype=float)
        colors = [COLORS["blue"]] * len(vals)
        if len(vals) > 0:
            best_idx = int(np.nanargmax(vals)) if is_r2 else int(np.nanargmin(vals))
            colors[best_idx] = COLORS["green"]

        ax.bar(x, vals, color=colors, edgecolor="#333333", linewidth=0.55, width=0.78)
        ax.set_title(title, pad=8)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Policy ID")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        if is_r2:
            ax.axhline(0.5, color=COLORS["red"], linestyle="--", linewidth=1.0, label="$R^2=0.5$")
            ax.legend(frameon=True, loc="upper right")
            ymin = min(-0.05, np.nanmin(vals) - 0.05)
            ymax = max(0.55, np.nanmax(vals) + 0.08)
            ax.set_ylim(ymin, ymax)
        else:
            ymin = 0
            ymax = np.nanmax(vals) * 1.18 if np.nanmax(vals) > 0 else 1
            ax.set_ylim(ymin, ymax)

        add_bar_labels(ax, fmt=fmt, fontsize=7, dy=0.008)
        finish_axes(ax)

    fig.suptitle("32D target-free selective AD subdomain policy metrics", y=1.03, fontweight="bold")
    save_fig(fig, "165_32d_est_policy_metrics")


def plot_frontier() -> None:
    if cov_col is None:
        log("[WARN] Coverage column not found; frontier figure skipped.")
        return

    df = summary.copy()
    df = df[np.isfinite(df[r2_col]) & np.isfinite(df[cov_col])].copy()
    if df.empty:
        log("[WARN] No valid summary rows for frontier figure.")
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)

    if main_col is not None:
        mc = df[main_col].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    else:
        mc = pd.Series(True, index=df.index)

    ax.scatter(
        df.loc[~mc, cov_col],
        df.loc[~mc, r2_col],
        s=18,
        color=COLORS["light_gray"],
        edgecolor="none",
        alpha=0.65,
        label="Other candidates",
        zorder=1,
    )
    ax.scatter(
        df.loc[mc, cov_col],
        df.loc[mc, r2_col],
        s=34,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.45,
        alpha=0.9,
        label="Main candidates",
        zorder=2,
    )

    anno = df.sort_values(r2_col, ascending=False).head(8).copy()
    for i, (_, row) in enumerate(anno.iterrows(), start=1):
        method = str(row.get(method_col, "")) if method_col else ""
        label = f"P{i:02d}"
        if method:
            label += f"\n{safe_label(method, 14)}"
        ax.annotate(
            label,
            xy=(row[cov_col], row[r2_col]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=7,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.72),
        )

    ax.axhline(0.5, color=COLORS["red"], linestyle="--", linewidth=1.0, label="$R^2=0.5$")
    ax.axvline(0.05, color=COLORS["dark_gray"], linestyle=":", linewidth=1.0, label="Minimum coverage")
    ax.set_title("$R^2$–coverage frontier for target-free selective AD policies", pad=9)
    ax.set_xlabel("Coverage within strategy–method subset")
    ax.set_ylabel("Retained-domain $R^2$")
    ax.set_xlim(left=max(0, float(np.nanmin(df[cov_col])) - 0.02))
    ax.set_ylim(
        bottom=min(0.25, float(np.nanmin(df[r2_col])) - 0.04),
        top=max(0.55, float(np.nanmax(df[r2_col])) + 0.06),
    )
    ax.legend(frameon=True, loc="best")
    finish_axes(ax)
    save_fig(fig, "166_32d_est_r2_coverage_frontier")


def plot_best_prediction() -> None:
    if pred_df is None or pred_df.empty:
        return

    obs_col = find_col(pred_df, ["Observed", "Observed_F_removal", "Actual", "y_true", "Target"], required=True)
    pred_col = find_col(pred_df, ["Predicted", "Prediction", "y_pred", "Fitted"], required=True)

    y_true = to_num(pred_df[obs_col]).to_numpy(dtype=float, copy=True)
    y_pred = to_num(pred_df[pred_col]).to_numpy(dtype=float, copy=True)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    r2 = metric_r2(y_true, y_pred)
    rmse = metric_rmse(y_true, y_pred)
    mae = metric_mae(y_true, y_pred)
    n = int(len(y_true))

    best = summary.iloc[0]
    method = str(best.get(method_col, "Best policy")) if method_col else "Best policy"
    rule = str(best.get(rule_col, "")) if rule_col else ""

    fig, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)
    ax.scatter(
        y_true,
        y_pred,
        s=24,
        color=COLORS["blue"],
        alpha=0.72,
        edgecolor="white",
        linewidth=0.25,
    )

    lim_low = 0
    lim_high = max(100, float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)])))
    ax.plot([lim_low, lim_high], [lim_low, lim_high],
            linestyle="--", color=COLORS["red"], linewidth=1.1)

    ax.set_xlim(lim_low, lim_high)
    ax.set_ylim(lim_low, lim_high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Predicted F removal (%)")
    ax.set_title(f"Best retained subdomain: {safe_label(method, 24)}", pad=9, fontweight="bold")

    if cov_col is not None:
        coverage = float(best.get(cov_col, np.nan))
        text = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}\nn = {n}\ncoverage = {coverage:.3f}"
    else:
        text = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}\nn = {n}"
    if rule:
        text += f"\nrule = {safe_label(rule, 34)}"

    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#888888", alpha=0.9),
    )
    finish_axes(ax)
    save_fig(fig, "167_32d_est_best_policy_prediction_panel")


def plot_split_stability() -> None:
    if split_df is None or split_df.empty:
        return

    split_col = find_col(split_df, ["Split", "Seed_or_Fold", "Seed", "Fold", "Split_ID"], required=False)
    sr2_col = find_col(split_df, ["R2", "Retained_R2"], required=True)
    srmse_col = find_col(split_df, ["RMSE", "Retained_RMSE"], required=True)
    sn_col = find_col(split_df, ["N_Retained", "n", "N", "Retained_N"], required=False)

    d = split_df.copy()
    for c in [sr2_col, srmse_col, sn_col]:
        if c is not None:
            d[c] = to_num(d[c])
    if split_col is None:
        d["Split_Label"] = [f"S{i+1}" for i in range(len(d))]
        split_col = "Split_Label"

    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[sr2_col, srmse_col]).copy()
    d = d.reset_index(drop=True)
    x = np.arange(len(d))
    labels = [safe_label(v, 16) for v in d[split_col].tolist()]

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0), constrained_layout=True)

    vals = d[sr2_col].to_numpy(dtype=float)
    colors = [COLORS["green"] if v >= 0.5 else COLORS["light_blue"] for v in vals]
    axes[0].bar(x, vals, color=colors, edgecolor="#333333", linewidth=0.55)
    axes[0].axhline(0.5, color=COLORS["red"], linestyle="--", linewidth=1.0)
    axes[0].set_title("$R^2$", pad=7)
    axes[0].set_ylabel("$R^2$")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    add_bar_labels(axes[0], fmt="{:.2f}", fontsize=7, dy=0.012)
    finish_axes(axes[0])

    vals = d[srmse_col].to_numpy(dtype=float)
    axes[1].bar(x, vals, color=COLORS["blue"], edgecolor="#333333", linewidth=0.55)
    axes[1].set_title("RMSE", pad=7)
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylim(0, max(1, float(np.nanmax(vals)) * 1.2))
    add_bar_labels(axes[1], fmt="{:.2f}", fontsize=7, dy=0.008)
    finish_axes(axes[1])

    if sn_col is not None:
        vals = d[sn_col].to_numpy(dtype=float)
        ncolors = [COLORS["orange"] if v < 10 else COLORS["blue"] for v in vals]
        axes[2].bar(x, vals, color=ncolors, edgecolor="#333333", linewidth=0.55)
        axes[2].set_title("Retained sample size", pad=7)
        axes[2].set_ylabel("N retained")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, rotation=35, ha="right")
        axes[2].set_ylim(0, max(1, float(np.nanmax(vals)) * 1.2))
        add_bar_labels(axes[2], fmt="{:.0f}", fontsize=7, dy=0.008)
        finish_axes(axes[2])
    else:
        axes[2].axis("off")

    fig.suptitle("Split-level stability of the selected 32D policy", y=1.06, fontweight="bold")
    save_fig(fig, "168_32d_est_best_policy_split_stability")


plot_policy_metrics()
plot_frontier()
plot_best_prediction()
plot_split_stability()

elapsed = time.time() - T0
log("=" * 90)
log("32D EST-style replot finished.")
log(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
log("=" * 90)
