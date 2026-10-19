# -*- coding: utf-8 -*-
"""
34a_est_figure4_final_8panels_v10.py

EST-style Figure 4 for the fluoride-coagulation machine-learning manuscript.
Panels a-b add descriptive Spearman correlations using audited fraction data.
The original v9 six panels are retained verbatim as c-h, except for the panel
letters changed during outer assembly. Their physical sizes remain unchanged.
Dependencies: Python >= 3.9; numpy, pandas, matplotlib, scipy.
No training, no input-file modification, no artificial predictions.
All 8 panels, input validation and audit exports are contained in this file.

Place in:
    fluoride_coagulation_ml_project/src/

Outputs:
    outputs/jece_final_package/figures/
    outputs/jece_final_package/tables/
    outputs/jece_final_package/reports/
    outputs/jece_final_package/logs/
"""
from __future__ import annotations

import math
import platform
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd

SCRIPT_NAME = "34a_est_figure4_final_8panels_v10"
FIGURE_STEM = "Figure_4_random_split_benchmark_and_model_selection_v10_8panels"
DPI = 600
FIGURE_SIZE = (12.0, 8.15)

COLORS = {
    "navy": "#4F718B",
    "blue": "#6F93AC",
    "teal": "#2F7F86",
    "sage": "#7E9B8B",
    "gold": "#C79655",
    "slate": "#66717B",
    "gray": "#A9B1B8",
    "light_gray": "#EEF1F3",
    "pale_blue": "#DDE9F0",
    "pale_teal": "#DCEBEC",
    "pale_gold": "#F1E4D2",
    "rose": "#B86F6A",
    "black": "#222222",
    "white": "#FFFFFF",
}

METHOD_LABELS = {
    "Selected_Weighted_4Model": "Weighted 4-model",
    "Selected_HistGradientBoosting": "HGB",
    "Selected_CatBoost": "CatBoost",
    "Selected_RF_CatBoost_Weighted": "RF + CatBoost",
    "RawCompact_HistGradientBoosting": "Raw HGB",
    "RawCompact_ExtraTrees": "Raw Extra Trees",
    "RawCompact_RandomForest": "Raw random forest",
    "Selected_RandomForest": "Selected RF",
    "Selected_ExtraTrees": "Selected extra trees",
}

METHOD_COLORS = {
    "Selected_Weighted_4Model": COLORS["gold"],
    "Selected_HistGradientBoosting": COLORS["teal"],
    "Selected_CatBoost": COLORS["navy"],
    "Selected_RF_CatBoost_Weighted": COLORS["sage"],
    "RawCompact_HistGradientBoosting": COLORS["blue"],
    "RawCompact_ExtraTrees": "#9AA7B2",
    "RawCompact_RandomForest": "#7F8D98",
    "Selected_RandomForest": "#8996A1",
    "Selected_ExtraTrees": "#AFB8C0",
}

RANKING_METHODS = [
    "Selected_Weighted_4Model",
    "Selected_HistGradientBoosting",
    "Selected_CatBoost",
    "Selected_RF_CatBoost_Weighted",
    "RawCompact_HistGradientBoosting",
    "RawCompact_ExtraTrees",
]

TOP_SELECTED_METHODS = [
    "Selected_Weighted_4Model",
    "Selected_HistGradientBoosting",
    "Selected_CatBoost",
    "Selected_RF_CatBoost_Weighted",
]

MARKERS = ["o", "s", "^", "D", "P", "X"]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    figures: Path
    tables: Path
    reports: Path
    logs: Path


# -----------------------------------------------------------------------------
# I/O and general style
# -----------------------------------------------------------------------------
def locate_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "outputs").exists() and (parent / "data").exists():
            return parent
    return current.parent.parent


def build_paths() -> ProjectPaths:
    root = locate_project_root()
    out = root / "outputs" / "jece_final_package"
    figures = out / "figures"
    tables = out / "tables"
    reports = out / "reports"
    logs = out / "logs"
    for folder in (figures, tables, reports, logs):
        folder.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(root, figures, tables, reports, logs)


def read_project_csv(root: Path, relative_path: str) -> pd.DataFrame:
    direct_path = root / relative_path
    if not direct_path.is_file():
        raise FileNotFoundError(
            f"Required existing v9 input table not found: {direct_path}\n"
            f"Project root: {root}\n"
            "Keep the original result files in outputs/tables; no substitute data are used."
        )
    return pd.read_csv(direct_path)


def load_tables(root: Path):
    summary = read_project_csv(root, "outputs/tables/19_selected_fusion_summary.csv")
    test_results = read_project_csv(root, "outputs/tables/19_selected_fusion_test_results.csv")
    predictions = read_project_csv(root, "outputs/tables/19_selected_fusion_predictions.csv")
    candidates = read_project_csv(root, "outputs/tables/19_selected_fusion_candidate_results.csv")
    return summary, test_results, predictions, candidates


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.8,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.1,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.55, alpha=0.34, color=COLORS["gray"])
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.095, y: float = 1.045) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=12.2,
            fontweight="bold", color=COLORS["black"], clip_on=False)


def compute_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[mask]
    predicted = predicted[mask]
    residual = predicted - observed
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    corr = float(np.corrcoef(observed, predicted)[0, 1])
    std_obs = float(np.std(observed, ddof=1))
    std_pred = float(np.std(predicted, ddof=1))
    centered_rmse = float(np.sqrt(np.mean(((predicted - np.mean(predicted)) - (observed - np.mean(observed))) ** 2)))
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "std_obs": std_obs,
        "std_pred": std_pred,
        "std_ratio": std_pred / std_obs,
        "crmse_norm": centered_rmse / std_obs,
        "n": len(observed),
    }


# -----------------------------------------------------------------------------
# Panels a-c
# -----------------------------------------------------------------------------
def draw_panel_a(ax: plt.Axes, summary: pd.DataFrame) -> None:
    df = summary[summary["Method"].isin(RANKING_METHODS)].copy()
    df["order"] = df["Method"].map({m: i for i, m in enumerate(RANKING_METHODS)})
    df = df.sort_values("order")
    y = np.arange(len(df))[::-1]

    for yi, (_, row) in zip(y, df.iterrows()):
        method = row["Method"]
        color = METHOD_COLORS[method]
        lower = row["R2_mean"] - row["R2_std"]
        upper = row["R2_mean"] + row["R2_std"]
        ax.hlines(yi, lower, upper, color=color, linewidth=1.45)
        ax.plot([lower, lower], [yi - 0.075, yi + 0.075], color=color, lw=0.9)
        ax.plot([upper, upper], [yi - 0.075, yi + 0.075], color=color, lw=0.9)
        ax.scatter(row["R2_mean"], yi, s=46, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        label_x = max(upper + 0.004, 0.898)
        ax.text(label_x, yi, f"R²={row['R2_mean']:.3f}±{row['R2_std']:.3f}; RMSE={row['RMSE_mean']:.2f}",
                ha="left", va="center", fontsize=6.15, color=COLORS["slate"], clip_on=False)

    ax.set_yticks(y, [METHOD_LABELS[m] for m in df["Method"]])
    ax.set_xlim(0.80, 0.947)
    ax.set_xlabel("Mean test R² across five random seeds")
    ax.set_title("Random-split performance ranking", loc="left", pad=7)
    clean_axis(ax, "x")
    panel_label(ax, "a", x=-0.09, y=1.035)


def draw_panel_b(ax: plt.Axes, test_results: pd.DataFrame) -> None:
    df = test_results[test_results["Method"].isin(TOP_SELECTED_METHODS)].copy()
    seeds = sorted(df["Seed"].unique())
    end_offsets = {
        "Selected_Weighted_4Model": 0.0008,
        "Selected_HistGradientBoosting": -0.0020,
        "Selected_CatBoost": 0.0020,
        "Selected_RF_CatBoost_Weighted": -0.0012,
    }
    short = {
        "Selected_Weighted_4Model": "W4",
        "Selected_HistGradientBoosting": "HGB",
        "Selected_CatBoost": "CAT",
        "Selected_RF_CatBoost_Weighted": "RF+CAT",
    }

    for method in TOP_SELECTED_METHODS:
        sub = df[df["Method"] == method].sort_values("Seed")
        ax.plot(sub["Seed"], sub["R2"], marker="o", markersize=4.5, linewidth=1.45,
                color=METHOD_COLORS[method])
        last = sub.iloc[-1]
        ax.text(float(last["Seed"]) + 2.8, float(last["R2"]) + end_offsets[method], short[method],
                ha="left", va="center", fontsize=6.4, fontweight="bold",
                color=METHOD_COLORS[method], clip_on=False)

    ax.set_xticks(seeds)
    ax.set_xlim(min(seeds) - 4, max(seeds) + 15)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Test R²")
    ax.set_ylim(0.855, 0.902)
    ax.set_title("Seed-level stability of the leading models", loc="left", pad=7)
    ax.text(0.98, 0.97, "Identical train/test splits for all models", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.1, color=COLORS["slate"])
    clean_axis(ax, "both")
    panel_label(ax, "b", x=-0.09, y=1.035)


def draw_panel_c(ax: plt.Axes, candidates: pd.DataFrame) -> None:
    df = candidates[candidates["Kind"].isin(TOP_SELECTED_METHODS)].copy()
    order = ["12", "18", "25", "35", "45", "all"]
    x = np.arange(len(order))
    short = {
        "Selected_Weighted_4Model": "W4",
        "Selected_HistGradientBoosting": "HGB",
        "Selected_CatBoost": "CAT",
        "Selected_RF_CatBoost_Weighted": "RF+CAT",
    }
    label_offsets = {
        "Selected_Weighted_4Model": -0.22,
        "Selected_HistGradientBoosting": -0.05,
        "Selected_CatBoost": 0.33,
        "Selected_RF_CatBoost_Weighted": 0.13,
    }

    grouped = df.groupby(["Kind", "Selected_K"])["Validation_RMSE"].agg(["mean", "std"]).reset_index()
    for method in TOP_SELECTED_METHODS:
        sub = grouped[grouped["Kind"] == method].copy()
        sub["x"] = sub["Selected_K"].astype(str).map({k: i for i, k in enumerate(order)})
        sub = sub.dropna(subset=["x"]).sort_values("x")
        ax.errorbar(sub["x"], sub["mean"], yerr=sub["std"], marker="o", markersize=4.0,
                    linewidth=1.35, elinewidth=0.70, capsize=2.0, color=METHOD_COLORS[method],
                    alpha=0.96)
        last = sub.iloc[-1]
        ax.text(float(last["x"]) + 0.12, float(last["mean"]) + label_offsets[method], short[method],
                fontsize=6.1, fontweight="bold", color=METHOD_COLORS[method],
                ha="left", va="center", clip_on=False)

    best = grouped.loc[grouped["mean"].idxmin()]
    best_x = order.index(str(best["Selected_K"]))
    ax.scatter(best_x, best["mean"], s=74, marker="*", facecolor=COLORS["gold"],
               edgecolor="white", linewidth=0.7, zorder=5)
    ax.text(2.35, 10.78, f"Best: W4 with the full feature set (RMSE={best['mean']:.2f})",
            fontsize=6.25, color=COLORS["slate"], ha="left", va="bottom")

    ax.set_xticks(x, ["12", "18", "25", "35", "45", "Full"])
    ax.set_xlim(-0.25, 5.88)
    ax.set_ylim(10.45, 19.60)
    ax.set_xlabel("Number of retained features")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Feature-selection and fusion screening", loc="left", pad=7)
    ax.text(0.02, 0.97, "Mean ± SD across five random seeds", transform=ax.transAxes,
            fontsize=6.05, color=COLORS["slate"], ha="left", va="top")
    clean_axis(ax, "both")
    panel_label(ax, "c", x=-0.09, y=1.035)


# -----------------------------------------------------------------------------
# Panel d: joint test-prediction plot
# -----------------------------------------------------------------------------
def draw_panel_d(subplot_spec, predictions: pd.DataFrame) -> None:
    nested = GridSpecFromSubplotSpec(
        6, 6, subplot_spec=subplot_spec,
        width_ratios=[1, 1, 1, 1, 0.10, 0.58],
        height_ratios=[0.23, 0.52, 1, 1, 1, 1],
        wspace=0.04, hspace=0.04,
    )
    ax_title = plt.subplot(nested[0, :])
    ax_top = plt.subplot(nested[1, :4])
    ax_scatter = plt.subplot(nested[2:, :4])
    ax_right = plt.subplot(nested[2:, 5])

    ax_title.axis("off")
    ax_title.text(0.0, 0.72, "d", fontsize=12.2, fontweight="bold", ha="left", va="center")
    ax_title.text(0.055, 0.72, "Five-seed pooled test predictions", fontsize=9.4, ha="left", va="center")

    df = predictions[predictions["Method"] == "Selected_Weighted_4Model"].copy()
    observed = df["Observed"].to_numpy(float)
    predicted = df["Predicted"].to_numpy(float)
    metrics = compute_metrics(observed, predicted)

    lower = min(0.0, float(np.nanmin([observed.min(), predicted.min()])))
    upper = max(100.0, float(np.nanmax([observed.max(), predicted.max()])))
    plot_min = math.floor(lower / 10.0) * 10.0
    plot_max = math.ceil(upper / 10.0) * 10.0

    ax_scatter.scatter(observed, predicted, s=14, color=COLORS["blue"], alpha=0.58,
                       edgecolor="white", linewidth=0.25, rasterized=True)
    ax_scatter.plot([plot_min, plot_max], [plot_min, plot_max], linestyle="--", color=COLORS["gold"], lw=1.05, label="1:1 line")
    slope, intercept = np.polyfit(observed, predicted, 1)
    xx = np.linspace(plot_min, plot_max, 200)
    ax_scatter.plot(xx, slope * xx + intercept, color=COLORS["teal"], lw=1.3, label="Linear fit")
    ax_scatter.set_xlim(plot_min, plot_max)
    ax_scatter.set_ylim(plot_min, plot_max)
    ax_scatter.set_xlabel("Observed fluoride removal efficiency (%)")
    ax_scatter.set_ylabel("Predicted fluoride removal efficiency (%)")
    ax_scatter.set_aspect("equal", adjustable="box")
    clean_axis(ax_scatter, "both")

    per_seed_n = int(round(metrics["n"] / df["Seed"].nunique()))
    text = (
        f"Pooled R² = {metrics['r2']:.3f}\n"
        f"RMSE = {metrics['rmse']:.2f} percentage points\n"
        f"MAE = {metrics['mae']:.2f} percentage points\n"
        f"{df['Seed'].nunique()} seeds × {per_seed_n} test predictions"
    )
    ax_scatter.text(
        0.035, 0.965, text, transform=ax_scatter.transAxes, ha="left", va="top", fontsize=6.35,
        bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=COLORS["gray"], lw=0.65, alpha=0.96),
    )

    bins = np.linspace(plot_min, plot_max, 19)
    ax_top.hist(observed, bins=bins, color=COLORS["slate"], alpha=0.30, edgecolor="white", linewidth=0.4, label="Observed")
    ax_top.hist(predicted, bins=bins, color=COLORS["pale_blue"], alpha=0.95, edgecolor="white", linewidth=0.4, label="Predicted")
    ax_top.set_xlim(plot_min, plot_max)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    ax_top.legend(frameon=False, loc="upper left", fontsize=6.1, ncol=2, handlelength=1.2, columnspacing=0.8)

    ax_right.hist(predicted, bins=bins, orientation="horizontal", color=COLORS["pale_blue"], alpha=0.95, edgecolor="white", linewidth=0.4)
    ax_right.hist(observed, bins=bins, orientation="horizontal", color=COLORS["slate"], alpha=0.30, edgecolor="white", linewidth=0.4)
    ax_right.set_ylim(plot_min, plot_max)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    for spine in ax_right.spines.values():
        spine.set_visible(False)


# -----------------------------------------------------------------------------
# Panel e: residual diagnostics
# -----------------------------------------------------------------------------
def draw_panel_e(ax: plt.Axes, predictions: pd.DataFrame) -> None:
    df = predictions[predictions["Method"] == "Selected_Weighted_4Model"].copy()
    df["Residual"] = df["Predicted"] - df["Observed"]

    ax.scatter(df["Predicted"], df["Residual"], s=12, color=COLORS["blue"], alpha=0.52,
               edgecolor="white", linewidth=0.20, rasterized=True)
    ax.axhline(0.0, linestyle="--", color=COLORS["gold"], lw=1.0)

    quantile_edges = np.unique(np.quantile(df["Predicted"], np.linspace(0, 1, 11)))
    centers, median_values, q10_values, q90_values = [], [], [], []
    for low, high in zip(quantile_edges[:-1], quantile_edges[1:]):
        subset = df[(df["Predicted"] >= low) & (df["Predicted"] <= high)]
        if len(subset) < 10:
            continue
        centers.append(float((low + high) / 2.0))
        median_values.append(float(subset["Residual"].median()))
        q10_values.append(float(subset["Residual"].quantile(0.10)))
        q90_values.append(float(subset["Residual"].quantile(0.90)))

    centers = np.asarray(centers)
    median_values = np.asarray(median_values)
    q10_values = np.asarray(q10_values)
    q90_values = np.asarray(q90_values)
    ax.fill_between(centers, q10_values, q90_values, color=COLORS["pale_teal"], alpha=0.88, linewidth=0.0,
                    label="10th-90th percentile")
    ax.plot(centers, median_values, color=COLORS["teal"], lw=1.55, label="Binned median")

    q_abs = float(np.quantile(np.abs(df["Residual"]), 0.995))
    limit = math.ceil(q_abs / 10.0) * 10.0
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("Predicted fluoride removal efficiency (%)")
    ax.set_ylabel("Residual (percentage points)")
    ax.set_title("Residual structure of the selected fusion model", loc="left", pad=7, fontsize=9.4)
    ax.legend(frameon=False, loc="upper right", fontsize=6.15, handlelength=2.1)
    ax.text(0.98, 0.04, "Residual = predicted - observed", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.2, color=COLORS["slate"])
    clean_axis(ax, "both")
    panel_label(ax, "e", x=-0.09, y=1.035)


# -----------------------------------------------------------------------------
# Panel f: enriched normalized Taylor diagram
# -----------------------------------------------------------------------------
def prepare_taylor_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in RANKING_METHODS:
        subset = predictions[predictions["Method"] == method]
        if subset.empty:
            continue
        metrics = compute_metrics(subset["Observed"], subset["Predicted"])
        metrics["Method"] = method
        metrics["x"] = metrics["std_ratio"] * metrics["corr"]
        metrics["y"] = metrics["std_ratio"] * math.sqrt(max(0.0, 1.0 - metrics["corr"] ** 2))
        rows.append(metrics)
    return pd.DataFrame(rows)


def _draw_taylor_background(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, 1.23)
    ax.set_ylim(0.0, 0.78)
    ax.axis("off")

    theta_limit = math.acos(0.80)
    theta = np.linspace(0.0, theta_limit, 320)

    # Normalized standard-deviation arcs.
    for radius in [0.8, 0.9, 1.0, 1.1, 1.2]:
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), color=COLORS["gray"], lw=0.55, alpha=0.55)
        ax.text(radius, -0.025, f"{radius:.1f}", ha="center", va="top", fontsize=5.9, color=COLORS["slate"])

    # Correlation rays and labels.
    for corr in [0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 1.00]:
        angle = math.acos(corr)
        ax.plot([0, 1.2 * math.cos(angle)], [0, 1.2 * math.sin(angle)], color=COLORS["gray"], lw=0.55, alpha=0.55)
        label_r = 1.245
        ax.text(label_r * math.cos(angle), label_r * math.sin(angle), f"{corr:.2f}",
                ha="center", va="center", fontsize=5.8, color=COLORS["slate"])

    # Outer boundary and lower axis.
    ax.plot(1.2 * np.cos(theta), 1.2 * np.sin(theta), color=COLORS["black"], lw=0.9)
    ax.plot([0.0, 1.2], [0.0, 0.0], color=COLORS["black"], lw=0.9)
    ax.plot([0.0, 1.2 * math.cos(theta_limit)], [0.0, 1.2 * math.sin(theta_limit)], color=COLORS["black"], lw=0.9)

    # Normalized centered-RMSE contours around the reference point (1, 0).
    phi = np.linspace(0.0, 2.0 * math.pi, 500)
    for level in [0.10, 0.20, 0.30, 0.40, 0.50]:
        xx = 1.0 + level * np.cos(phi)
        yy = level * np.sin(phi)
        mask = (xx >= 0.0) & (xx <= 1.23) & (yy >= 0.0) & (yy <= 0.78)
        ax.plot(xx[mask], yy[mask], color=COLORS["rose"], lw=0.65, ls="--", alpha=0.72)
        valid = np.where(mask)[0]
        if len(valid) > 0:
            idx = valid[len(valid) // 3]
            ax.text(xx[idx], yy[idx] + 0.012, f"{level:.1f}", fontsize=5.4, color=COLORS["rose"], ha="center")

    ax.scatter([1.0], [0.0], s=60, marker="*", color=COLORS["rose"], edgecolor="white", linewidth=0.5, zorder=7)
    ax.text(1.02, 0.026, "Reference", fontsize=5.9, color=COLORS["rose"], ha="left", va="bottom")
    ax.text(0.62, -0.083, "Normalized standard deviation", fontsize=6.3, ha="center", va="top")
    ax.text(1.06, 0.69, "Correlation coefficient", fontsize=6.25, color=COLORS["teal"], ha="center", rotation=-30)
    ax.text(0.03, 0.015, "Dashed contours: normalized centered RMSE", fontsize=5.4, color=COLORS["rose"], ha="left", va="bottom")


def draw_panel_f(subplot_spec, predictions: pd.DataFrame) -> pd.DataFrame:
    subgrid = GridSpecFromSubplotSpec(6, 1, subplot_spec=subplot_spec,
                                     height_ratios=[0.24, 1, 1, 1, 1, 0.04], hspace=0.00)
    ax_title = plt.subplot(subgrid[0, 0])
    ax_title.axis("off")
    ax_title.text(-0.02, 0.62, "f", fontsize=12.2, fontweight="bold", ha="left", va="center")
    ax_title.text(0.07, 0.62, "Normalized Taylor diagram", fontsize=9.4, ha="left", va="center")

    ax = plt.subplot(subgrid[1:5, 0])
    taylor = prepare_taylor_metrics(predictions)
    _draw_taylor_background(ax)

    compact_names = {
        "Selected_Weighted_4Model": "W4",
        "Selected_HistGradientBoosting": "HGB",
        "Selected_CatBoost": "CAT",
        "Selected_RF_CatBoost_Weighted": "RF+CAT",
        "RawCompact_HistGradientBoosting": "Raw HGB",
        "RawCompact_ExtraTrees": "Raw ET",
    }
    marker_map = {
        "Selected_Weighted_4Model": "o",
        "Selected_HistGradientBoosting": "s",
        "Selected_CatBoost": "^",
        "Selected_RF_CatBoost_Weighted": "D",
        "RawCompact_HistGradientBoosting": "P",
        "RawCompact_ExtraTrees": "X",
    }

    # Main panel points.
    for _, row in taylor.iterrows():
        method = row["Method"]
        ax.scatter(row["x"], row["y"], s=44, marker=marker_map[method], color=METHOD_COLORS[method],
                   edgecolor="white", linewidth=0.55, zorder=8)

    # Magnified inset resolves the tightly clustered models.
    inset = ax.inset_axes([0.05, 0.47, 0.47, 0.42])
    inset.set_facecolor("white")
    inset.set_xlim(0.792, 0.868)
    inset.set_ylim(0.302, 0.365)
    inset.grid(True, linestyle="--", linewidth=0.45, alpha=0.35, color=COLORS["gray"])
    inset.tick_params(labelsize=5.2, length=2.0, pad=1.0)
    inset.set_title("Magnified model cluster", fontsize=6.0, pad=2.0, loc="left")
    for spine in inset.spines.values():
        spine.set_linewidth(0.65)
        spine.set_edgecolor(COLORS["gray"])

    label_positions = {
        "Selected_Weighted_4Model": (0.836, 0.306),
        "Selected_HistGradientBoosting": (0.861, 0.326),
        "Selected_CatBoost": (0.840, 0.337),
        "Selected_RF_CatBoost_Weighted": (0.820, 0.325),
        "RawCompact_HistGradientBoosting": (0.861, 0.343),
        "RawCompact_ExtraTrees": (0.806, 0.362),
    }
    for _, row in taylor.iterrows():
        method = row["Method"]
        inset.scatter(row["x"], row["y"], s=35, marker=marker_map[method], color=METHOD_COLORS[method],
                      edgecolor="white", linewidth=0.45, zorder=5)
        tx, ty = label_positions[method]
        inset.annotate(compact_names[method], xy=(row["x"], row["y"]), xytext=(tx, ty), textcoords="data",
                       fontsize=5.05, color=COLORS["slate"], ha="center", va="center",
                       arrowprops=dict(arrowstyle="-", color=COLORS["gray"], lw=0.45),
                       bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.88))


    ax_note = plt.subplot(subgrid[5, 0])
    ax_note.axis("off")
    return taylor


# -----------------------------------------------------------------------------
# Figure assembly and audit output
# -----------------------------------------------------------------------------
def create_figure(summary, test_results, predictions, candidates):
    fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=False)
    grid = GridSpec(
        2, 3, figure=fig,
        left=0.060, right=0.985, bottom=0.075, top=0.965,
        wspace=0.28, hspace=0.30, width_ratios=[1.07, 1.00, 1.16],
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    draw_panel_a(ax_a, summary)
    draw_panel_b(ax_b, test_results)
    draw_panel_c(ax_c, candidates)

    draw_panel_d(grid[1, 0], predictions)
    ax_e = fig.add_subplot(grid[1, 1])
    draw_panel_e(ax_e, predictions)
    taylor_metrics = draw_panel_f(grid[1, 2], predictions)
    return fig, taylor_metrics


def export_audit(paths: ProjectPaths, summary, test_results, predictions, candidates, taylor_metrics, runtime_seconds: float) -> None:
    summary[summary["Method"].isin(RANKING_METHODS)].to_csv(
        paths.tables / f"{FIGURE_STEM}_ranking_data.csv", index=False, encoding="utf-8-sig"
    )
    taylor_export = taylor_metrics[["Method", "corr", "std_ratio", "crmse_norm", "r2", "rmse", "mae", "n"]].copy()
    taylor_export["Label"] = taylor_export["Method"].map(METHOD_LABELS)
    taylor_export.to_csv(paths.tables / f"{FIGURE_STEM}_taylor_metrics.csv", index=False, encoding="utf-8-sig")

    selected_predictions = predictions[predictions["Method"] == "Selected_Weighted_4Model"]
    pooled_metrics = compute_metrics(selected_predictions["Observed"], selected_predictions["Predicted"])
    report = f"""{SCRIPT_NAME}

Input tables:
- 19_selected_fusion_summary.csv: {len(summary)} rows
- 19_selected_fusion_test_results.csv: {len(test_results)} rows
- 19_selected_fusion_predictions.csv: {len(predictions)} rows
- 19_selected_fusion_candidate_results.csv: {len(candidates)} rows

Selected fusion model pooled test metrics:
- R2: {pooled_metrics['r2']:.6f}
- RMSE: {pooled_metrics['rmse']:.6f} percentage points
- MAE: {pooled_metrics['mae']:.6f} percentage points
- pooled predictions: {pooled_metrics['n']}
- seeds: {selected_predictions['Seed'].nunique()}

Important interpretation:
The actual-predicted panel contains pooled out-of-sample test predictions from five random splits.
It does not display training-set predictions because those predictions are not stored in the project tables.
No training predictions were fabricated.

Runtime: {runtime_seconds:.2f} s
Python: {sys.version.split()[0]}
Platform: {platform.platform()}
"""
    (paths.reports / f"{SCRIPT_NAME}_audit.txt").write_text(report, encoding="utf-8")
    (paths.logs / f"{SCRIPT_NAME}.log").write_text(report, encoding="utf-8")


# -----------------------------------------------------------------------------
# 6.0 handover: descriptive panels a-b, with the v9 panels preserved as c-h.
# All original draw_panel_* functions above are retained verbatim.
# -----------------------------------------------------------------------------
INTEGRATED_HEIGHT = 13.30
CORRELATION_COLUMNS = {
    "游离氟占比": "Free F fraction",
    "初始氟_mg_L": "Initial F",
    "干扰离子浓度_mmol_L": "Interfering ions",
    "水温_C": "Temperature",
    "剂量_mM": "Coagulant dose",
    "进水pH": "Influent pH",
    "停留时间_min": "Retention time",
    "氟去除率_pct": "Removal",
    "出水氟_mg_L": "Effluent F*",
}
BASE_INPUT_COLUMNS = list(CORRELATION_COLUMNS)[:7]
ANNOTATION_THRESHOLD = 0.50
DISPLAY_ASSOCIATIONS = 10


def checked_csv(path: Path, required_columns=()) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file is missing: {path}")
    frame = pd.read_csv(path)
    missing = [c for c in required_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {missing}")
    return frame


def load_correlation_data(root: Path):
    """Use real corrected data only; never correct or overwrite source files here."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr

    main_path = root / "data" / "processed" / "model_ready_v1.csv"
    source_path = root / "data" / "processed" / "model_ready_v1_with_source.csv"
    fixed_path = root / "data" / "processed" / "model_ready_v1_free_fraction_fixed_candidate.csv"
    base = checked_csv(main_path, [*CORRELATION_COLUMNS, "原始行号"])
    source = checked_csv(source_path, ["文献编号", "原始行号"])
    if base.shape != (1011, 26) or source.shape != (1011, 27):
        raise ValueError(f"Expected strict/source matrices (1011,26)/(1011,27); got {base.shape}/{source.shape}.")
    if not base["原始行号"].is_unique or not base["原始行号"].equals(source["原始行号"]):
        raise ValueError("Strict/source matrices do not have identical, unique row identifiers.")
    if source["文献编号"].isna().any() or source["文献编号"].nunique() != 27:
        raise ValueError("The locked dataset requires 27 nonmissing source groups.")

    original_fraction = pd.to_numeric(base["游离氟占比"], errors="raise")
    if original_fraction.between(0, 1).all():
        data, data_path = base.copy(), main_path
    else:
        if not fixed_path.is_file():
            raise FileNotFoundError(
                "The stored free-fluoride fraction is not fully in [0,1]. "
                "An existing audited corrected matrix is required; no automatic correction is made.\n"
                f"Checked official matrix: {main_path}\nRequired corrected candidate: {fixed_path}"
            )
        data = checked_csv(fixed_path, [*CORRELATION_COLUMNS, "原始行号"])
        data_path = fixed_path
        if data.shape != base.shape or not data["原始行号"].equals(base["原始行号"]):
            raise ValueError("Corrected matrix dimensions/row order differ from the locked dataset.")
        # Recalculated QC columns are irrelevant here. Every other plotted variable
        # must match the stored matrix; only the audited fraction may change.
        other_columns = [c for c in CORRELATION_COLUMNS if c != "游离氟占比"]
        if not np.allclose(data[other_columns], base[other_columns], rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError("The corrected candidate changes plotted variables other than free-F fraction.")
    if not pd.to_numeric(data["游离氟占比"], errors="raise").between(0, 1).all():
        raise ValueError("The selected free-fluoride fraction must be finite and in [0,1].")

    values = data[list(CORRELATION_COLUMNS)].apply(pd.to_numeric, errors="raise")
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Infinite values in descriptive analysis inputs.")
    bad = [c for c in values if values[c].nunique() < 2 or values[c].notna().sum() < 30]
    if bad:
        raise ValueError(f"Insufficient observations or variation for correlation: {bad}")
    corr = values.corr(method="spearman", min_periods=30)
    if not np.isfinite(corr.to_numpy()).all():
        raise ValueError("The Spearman matrix contains undefined correlations.")
    # Signed correlation distance: positive associations cluster together.
    # Average linkage requires only a symmetric dissimilarity, not Ward geometry.
    distance = np.clip(1.0 - corr.to_numpy(), 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=True), method="average", optimal_ordering=True)
    order = leaves_list(tree).astype(int)

    # Whitelist raw numerical inputs and the response. No derived transforms,
    # products, identifiers, category codes or effluent variables enter panel b.
    allowed = BASE_INPUT_COLUMNS + ["氟去除率_pct"]
    pairs = []
    for i, first in enumerate(allowed):
        for second in allowed[i + 1:]:
            paired = values[[first, second]].dropna()
            rho = float(spearmanr(paired[first], paired[second]).statistic)
            pairs.append({"Variable_1": first, "Variable_2": second,
                          "Label_1": CORRELATION_COLUMNS[first],
                          "Label_2": CORRELATION_COLUMNS[second],
                          "Spearman_rho": rho, "Absolute_rho": abs(rho), "N_pairs": len(paired),
                          "Relation": "Input-response" if "氟去除率_pct" in (first, second) else "Input-input"})
    pairs = pd.DataFrame(pairs).sort_values(
        ["Absolute_rho", "Label_1", "Label_2"], ascending=[False, True, True], kind="stable"
    ).reset_index(drop=True)
    pairs["Displayed_in_b"] = np.arange(len(pairs)) < DISPLAY_ASSOCIATIONS
    finite = values.notna().astype(int)
    pair_n = finite.T.dot(finite)
    return {"values": values, "corr": corr, "order": order, "pairs": pairs,
            "pair_n": pair_n, "data_path": data_path, "source_path": source_path,
            "source_groups": int(source["文献编号"].nunique()),
            "fraction_changed_rows": int((~np.isclose(original_fraction, data["游离氟占比"], equal_nan=True)).sum())}


def verify_locked_results(summary, test_results, predictions, candidates):
    """Reject incompatible/missing result tables instead of silently changing v9."""
    checks = [(summary, ["Method", "R2_mean", "R2_std", "RMSE_mean"], "summary"),
              (test_results, ["Seed", "Method", "R2", "RMSE", "MAE"], "test results"),
              (predictions, ["Seed", "Method", "Observed", "Predicted"], "predictions"),
              (candidates, ["Kind", "Selected_K", "Validation_RMSE"], "candidates")]
    for frame, required, name in checks:
        missing = set(required) - set(frame)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
    if summary["Method"].duplicated().any() or test_results.duplicated(["Method", "Seed"]).any():
        raise ValueError("Duplicate summary rows or model-seed results.")
    expected_seeds = {7, 21, 42, 66, 88}
    for method in RANKING_METHODS:
        sub = predictions[predictions["Method"] == method]
        tests = test_results[test_results["Method"] == method]
        if set(sub["Seed"]) != expected_seeds or set(tests["Seed"]) != expected_seeds:
            raise ValueError(f"Missing locked seeds for {method}.")
        if not np.isfinite(sub[["Observed", "Predicted"]].to_numpy()).all():
            raise ValueError(f"Nonfinite locked predictions for {method}.")
        for seed, group in sub.groupby("Seed"):
            if len(group) != 203:
                raise ValueError(f"Expected 203 test predictions for {method}, seed {seed}.")
            calculated = compute_metrics(group["Observed"], group["Predicted"])
            row = tests[tests["Seed"] == seed].iloc[0]
            for lower, upper in [("r2", "R2"), ("rmse", "RMSE"), ("mae", "MAE")]:
                if not np.isclose(calculated[lower], row[upper], rtol=0, atol=1e-7):
                    raise ValueError(f"Prediction/result mismatch: {method}, seed {seed}, {upper}.")
        ranked = summary[summary["Method"] == method]
        if len(ranked) != 1:
            raise ValueError(f"Missing unique summary for {method}.")
        for metric in ["R2", "RMSE", "MAE"]:
            for suffix, calculated in [("mean", tests[metric].mean()), ("std", tests[metric].std())]:
                if not np.isclose(ranked.iloc[0][f"{metric}_{suffix}"], calculated, rtol=0, atol=1e-7):
                    raise ValueError(f"Summary/result mismatch: {method}, {metric}_{suffix}.")
    selected = predictions[predictions["Method"] == "Selected_Weighted_4Model"]
    metrics = compute_metrics(selected["Observed"], selected["Predicted"])
    for key, reference in {"r2": 0.881701, "rmse": 10.267423, "mae": 6.403502}.items():
        if not np.isclose(metrics[key], reference, rtol=0, atol=1e-6):
            raise ValueError(f"Locked v9 pooled {key} differs: {metrics[key]} versus {reference}.")
    return metrics


def draw_correlation_panels(fig, bundle):
    """Add the two top panels without touching any of the six original axes."""
    from matplotlib.colors import LinearSegmentedColormap

    height = INTEGRATED_HEIGHT
    # Layout coordinates are specified in physical inches. This isolates the
    # added panels from the original 12 x 8.15 inch canvas below them.
    def axes_at(x, y, width, panel_height):
        return fig.add_axes([x / 12.0, y / height, width / 12.0, panel_height / height])

    ax_a = axes_at(1.15, 9.30, 3.42, 3.42)
    ax_b = axes_at(8.43, 9.30, 3.00, 3.42)
    for x, label, title in [(0.40, "a", "Spearman correlation structure"),
                            (6.08, "b", "Non-redundant rank associations")]:
        fig.text(x / 12.0, 12.98 / height, label, fontsize=12.2, fontweight="bold", ha="left", va="center")
        fig.text((x + 0.25) / 12.0, 12.98 / height, title, fontsize=9.4, ha="left", va="center")

    ordered = bundle["corr"].iloc[bundle["order"], bundle["order"]]
    labels = [CORRELATION_COLUMNS[c] for c in ordered]
    cmap = LinearSegmentedColormap.from_list("rank_diverging", [COLORS["navy"], "#FAFAF8", COLORS["rose"]])
    img = ax_a.imshow(ordered.to_numpy(), cmap=cmap, vmin=-1, vmax=1, interpolation="nearest", aspect="equal")
    ax_a.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=7.1)
    ax_a.set_yticks(np.arange(len(labels)), labels, fontsize=7.1)
    ax_a.tick_params(length=0, pad=4)
    ax_a.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax_a.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax_a.grid(which="minor", color="white", linewidth=0.65)
    ax_a.tick_params(which="minor", bottom=False, left=False)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = float(ordered.iloc[row, col])
            if row != col and abs(value) >= ANNOTATION_THRESHOLD:
                ax_a.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7.3,
                          color="white" if abs(value) >= 0.75 else COLORS["black"])
    for tick in [*ax_a.get_xticklabels(), *ax_a.get_yticklabels()]:
        if tick.get_text() == "Effluent F*":
            tick.set_color(COLORS["rose"])
        elif tick.get_text() == "Removal":
            tick.set_fontweight("bold")
    cax = axes_at(4.77, 9.30, 0.13, 3.42)
    colorbar = fig.colorbar(img, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    colorbar.set_label("Spearman ρ", fontsize=8.1, labelpad=5)
    colorbar.ax.tick_params(labelsize=7.0, width=0.6, length=2.5)
    colorbar.outline.set_visible(False)

    displayed = bundle["pairs"].loc[bundle["pairs"]["Displayed_in_b"]].copy()
    yy = np.arange(len(displayed))[::-1]
    for y, row in zip(yy, displayed.itertuples()):
        value = row.Spearman_rho
        color = COLORS["rose"] if value >= 0 else COLORS["navy"]
        marker = "D" if row.Relation == "Input-response" else "o"
        ax_b.hlines(y, 0, value, color=color, linewidth=1.6, alpha=0.8)
        ax_b.scatter(value, y, s=39, marker=marker, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax_b.text(value + (0.025 if value >= 0 else -0.025), y, f"{value:+.3f}",
                  ha="left" if value >= 0 else "right", va="center", fontsize=7.0, color=COLORS["slate"])
    pair_labels = [f"{r.Label_1} / {r.Label_2}" for r in displayed.itertuples()]
    ax_b.set_yticks(yy, pair_labels, fontsize=7.0)
    ax_b.tick_params(axis="y", length=0, pad=5)
    ax_b.set_ylim(-0.65, len(displayed) - 0.20)
    ax_b.set_xlim(-0.6, 0.6)
    ax_b.set_xticks([-0.6, -0.3, 0, 0.3, 0.6])
    ax_b.set_xlabel("Spearman ρ", fontsize=8.1)
    ax_b.axvline(0, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    clean_axis(ax_b, "x")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS["slate"], markersize=4.5, label="Input-input"),
               Line2D([], [], marker="D", linestyle="none", color=COLORS["slate"], markersize=4.5, label="Input-response")]
    ax_b.legend(handles=handles, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.01),
                borderaxespad=0, ncol=2, fontsize=6.4, handletextpad=0.3, columnspacing=1.0)

    n_min = int(bundle["pair_n"].to_numpy().min())
    n_max = int(bundle["pair_n"].to_numpy().max())
    n_label = str(n_min) if n_min == n_max else f"{n_min}-{n_max}"
    # Footnotes occupy their own strip above the unchanged original top row.
    fig.text(0.055, 8.33 / height,
             f"Average-linkage order (1 - ρ); labels: |ρ| ≥ {ANNOTATION_THRESHOLD:.2f}. Pairwise n = {n_label}.\n"
             "*Effluent F: descriptive only; excluded from prediction inputs. Removal is the response.",
             ha="left", va="bottom", fontsize=6.35, color=COLORS["slate"], linespacing=1.5)
    fig.text(0.525, 8.33 / height,
             "Ten largest |ρ| among unique raw input-input and input-response pairs.\n"
             "No derived transforms/products or effluent variables. Pooled descriptive associations.",
             ha="left", va="bottom", fontsize=6.35, color=COLORS["slate"], linespacing=1.5)
    return ax_a, ax_b


def create_integrated_figure(summary, test_results, predictions, candidates, bundle):
    # Only the outer GridSpec and canvas height differ from create_figure(v9).
    # Multiplying both old vertical margins by this ratio preserves every old
    # panel's position, width and height in physical inches, including insets.
    fig = plt.figure(figsize=(FIGURE_SIZE[0], INTEGRATED_HEIGHT), constrained_layout=False)
    scale = FIGURE_SIZE[1] / INTEGRATED_HEIGHT
    grid = GridSpec(2, 3, figure=fig,
                    left=0.060, right=0.985, bottom=0.075 * scale, top=0.965 * scale,
                    wspace=0.28, hspace=0.30, width_ratios=[1.07, 1.00, 1.16])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    draw_panel_a(ax_a, summary)
    draw_panel_b(ax_b, test_results)
    draw_panel_c(ax_c, candidates)
    draw_panel_d(grid[1, 0], predictions)
    ax_e = fig.add_subplot(grid[1, 1])
    draw_panel_e(ax_e, predictions)
    taylor_metrics = draw_panel_f(grid[1, 2], predictions)
    mapping = dict(zip("abcdef", "cdefgh"))
    renamed = []
    for ax in fig.axes:
        for text_artist in ax.texts:
            label = text_artist.get_text()
            if label in mapping and np.isclose(text_artist.get_fontsize(), 12.2):
                text_artist.set_text(mapping[label])
                renamed.append(label)
    if sorted(renamed) != list("abcdef"):
        raise RuntimeError(f"Could not uniquely relabel the six original panels: {renamed}")
    draw_correlation_panels(fig, bundle)
    return fig, taylor_metrics


def export_correlation_audit(paths, bundle, metrics, runtime):
    import hashlib
    import json

    stem = FIGURE_STEM
    rename = CORRELATION_COLUMNS
    order = bundle["order"]
    bundle["corr"].iloc[order, order].rename(index=rename, columns=rename).to_csv(
        paths.tables / f"{stem}_spearman_matrix.csv", encoding="utf-8-sig")
    bundle["pair_n"].rename(index=rename, columns=rename).to_csv(
        paths.tables / f"{stem}_spearman_pair_counts.csv", encoding="utf-8-sig")
    bundle["pairs"].to_csv(paths.tables / f"{stem}_nonredundant_associations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"Column": k, "Label": v,
                   "Role": "Descriptive outcome only" if k == "出水氟_mg_L" else "Response" if k == "氟去除率_pct" else "Available numerical input",
                   "Included_in_b": k != "出水氟_mg_L"} for k, v in rename.items()]).to_csv(
        paths.tables / f"{stem}_correlation_variable_roles.csv", index=False, encoding="utf-8-sig")
    inputs = [bundle["data_path"], bundle["source_path"],
              *[paths.root / "outputs" / "tables" / f"19_selected_fusion_{s}.csv"
                for s in ["summary", "test_results", "predictions", "candidate_results"]]]
    provenance = [{"path": str(p.relative_to(paths.root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]
    config_path = paths.root / "src" / "config.py"
    if config_path.is_file():
        provenance.append({"path": "src/config.py", "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()})
    report = {
        "script": SCRIPT_NAME, "figure": stem, "n_rows": len(bundle["values"]),
        "n_source_groups": bundle["source_groups"],
        "fraction_changed_rows_vs_stored_matrix": bundle["fraction_changed_rows"],
        "new_panels": "a: clustered Spearman matrix; b: top 10 nonredundant raw-variable associations",
        "old_panels": "Original v9 a-f retained as c-h; only panel letters changed.",
        "layout": "Original physical axes dimensions retained; extra space added only above original canvas.",
        "correlation": "Pairwise-complete Spearman; average linkage on 1-rho; no dendrogram.",
        "inference": "Pooled descriptive associations, not causal or source-independent effects. No inferential stars or CIs.",
        "b_selection": "All 28 unordered pairs among 7 base numerical inputs and response; show 10 largest absolute rho. All pairs exported.",
        "excluded_from_b": "Effluent F, effluent pH, calculated removal/difference, identifiers, raw duplicates, transforms and products.",
        "prediction_inputs": "Correlation panels do not define or change predictor sets. Effluent F and removal are outcomes.",
        "legacy_metrics": metrics,
        "metric_convention": "Original model ranking uses mean/SD over 5 seeds; original prediction panel uses pooled predictions (1015 events, not independent unique samples).",
        "training": "No models trained. Existing v9 results are historical locked results, not claimed to be recomputed after fraction correction.",
        "paths": "Source tables: outputs/tables; deliverables: outputs/jece_final_package/{figures,tables,reports,logs}, following v9.",
        "source_files_modified": False, "provenance": provenance,
        "total_runtime_seconds": round(runtime, 3),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    (paths.reports / f"{SCRIPT_NAME}_integration_audit.json").write_text(text, encoding="utf-8")
    (paths.logs / f"{SCRIPT_NAME}_integration.log").write_text(text, encoding="utf-8")


def main() -> None:
    start_time = time.perf_counter()
    configure_style()
    root = locate_project_root()
    # Load/check everything before creating output directories or drawing.
    summary, test_results, predictions, candidates = load_tables(root)
    metrics = verify_locked_results(summary, test_results, predictions, candidates)
    bundle = load_correlation_data(root)
    paths = build_paths()
    figure, taylor_metrics = create_integrated_figure(summary, test_results, predictions, candidates, bundle)
    png_path = paths.figures / f"{FIGURE_STEM}.png"
    pdf_path = paths.figures / f"{FIGURE_STEM}.pdf"
    figure.savefig(png_path, dpi=DPI, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    export_audit(paths, summary, test_results, predictions, candidates, taylor_metrics, time.perf_counter() - start_time)
    export_correlation_audit(paths, bundle, metrics, time.perf_counter() - start_time)
    print(f"[{SCRIPT_NAME}] Correlation input: {bundle['data_path']}")
    print(f"[{SCRIPT_NAME}] Original v9 six panels retained as c-h; locked metric checks passed.")
    print(f"[{SCRIPT_NAME}] PNG: {png_path}")
    print(f"[{SCRIPT_NAME}] PDF: {pdf_path}")
    print(f"Total runtime: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
