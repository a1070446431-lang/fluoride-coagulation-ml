# -*- coding: utf-8 -*-
"""
34a_est_figure4_random_split_benchmark_v2.py

EST-style Figure 4 for the fluoride-coagulation machine-learning manuscript.
The figure summarizes random-split model ranking, seed stability, feature/fusion
screening, pooled out-of-sample predictions, residual structure, and a normalized
Taylor diagram.

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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd

SCRIPT_NAME = "34a_est_figure4_random_split_benchmark_v9"
FIGURE_STEM = "Figure_4_random_split_benchmark_and_model_selection_v9"
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
    if direct_path.exists():
        return pd.read_csv(direct_path)

    zip_candidates = [root.parent / "fluoride_coagulation_ml_project.zip", Path("/mnt/data/fluoride_coagulation_ml_project.zip")]
    internal_path = f"fluoride_coagulation_ml_project/{relative_path.replace('\\', '/')}"
    for zip_path in zip_candidates:
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as archive:
                if internal_path in archive.namelist():
                    with archive.open(internal_path) as file_obj:
                        return pd.read_csv(file_obj)
    raise FileNotFoundError(f"Required table not found: {relative_path}")


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


def main() -> None:
    start_time = time.time()
    configure_style()
    paths = build_paths()
    summary, test_results, predictions, candidates = load_tables(paths.root)
    figure, taylor_metrics = create_figure(summary, test_results, predictions, candidates)

    png_path = paths.figures / f"{FIGURE_STEM}.png"
    pdf_path = paths.figures / f"{FIGURE_STEM}.pdf"
    figure.savefig(png_path, dpi=DPI, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    runtime_seconds = time.time() - start_time
    export_audit(paths, summary, test_results, predictions, candidates, taylor_metrics, runtime_seconds)
    print(f"[{SCRIPT_NAME}] PNG: {png_path}")
    print(f"[{SCRIPT_NAME}] PDF: {pdf_path}")
    print(f"[{SCRIPT_NAME}] Total runtime: {runtime_seconds:.2f} s")


if __name__ == "__main__":
    main()
