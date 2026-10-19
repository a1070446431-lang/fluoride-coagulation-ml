# -*- coding: utf-8 -*-
"""
25_est_style_figure3_final_2model.py

Expanded final EST-style Figure 3 for fluoride removal prediction.

Purpose
-------
Generate an expanded workflow-oriented Figure 3 after exact ablation.

The final model is updated from:
    Selected Weighted 4Model
to:
    Selected CatBoost-HGB Weighted 2Model

This figure is designed as a model-development evidence figure, not as a
simple pile-up of all models.

Panels
------
A: RF baseline
B: RF-CatBoost weighted fusion
C: Ridge stacking fusion
D: Tuned CatBoost
E: Selected CatBoost under final selected-feature workflow
F: Selected HGB under final selected-feature workflow
G: Selected Weighted 4Model before exact ablation
H: Final Selected CatBoost-HGB Weighted 2Model
I: Residual diagnosis of final 2Model
J: Pooled R2 comparison across representative stages
K: Pooled RMSE and MAE comparison across representative stages
L: Final CatBoost-HGB fusion weight stability

Required inputs
---------------
outputs/tables/12_multiseed_rf_catboost_predictions.csv
outputs/tables/16_tunedcat_multiseed_predictions.csv
outputs/tables/19_stacking_multiseed_predictions.csv
outputs/tables/24_exact_ablation_predictions.csv
outputs/tables/24_exact_ablation_weights.csv

Optional fallback
-----------------
outputs/tables/19_selected_fusion_predictions.csv

Outputs
-------
outputs/figures/Figure3_final_EST_workflow_2Model_expanded.png
outputs/figures/Figure3_final_EST_workflow_2Model_expanded.pdf

outputs/tables/25_est_figure3_expanded_metrics_used.csv
outputs/tables/25_est_figure3_expanded_weight_summary.csv
outputs/tables/25_est_figure3_expanded_stage_sources.csv
"""

from pathlib import Path
import warnings
import textwrap
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse

warnings.filterwarnings("ignore")


# ============================================================
# 1. Paths
# ============================================================

def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    candidates = [
        current,
        current.parent,
        current.parent.parent,
    ]

    for p in candidates:
        if (p / "outputs" / "tables").exists() and (p / "outputs" / "figures").exists():
            return p

    if current.name.lower() == "src":
        return current.parent

    raise FileNotFoundError(
        "Cannot locate project root. "
        "Please make sure this script is placed under the project src folder."
    )


PROJECT_ROOT = find_project_root()
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8.4,
    "axes.linewidth": 0.85,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.6,
    "xtick.labelsize": 6.9,
    "ytick.labelsize": 6.9,
    "legend.fontsize": 6.7,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

PURPLE = "#7E62A8"
LIGHT_PURPLE = "#9B72B0"
BLUE = "#4C78A8"
GREEN = "#63A97A"
TEAL = "#63B7B2"
ORANGE = "#E28E68"
RED = "#FF0000"
DARK = "#3B3B3B"
MAGENTA = "#EB4AA9"
GRAY = "#666666"
LIGHT_GRID = "#E7E7E7"
BLACK = "#222222"


STAGE_ORDER = [
    "RF baseline",
    "RF-CatBoost weighted fusion",
    "Ridge stacking fusion",
    "Tuned CatBoost",
    "Selected CatBoost",
    "Selected HGB",
    "Selected Weighted 4Model",
    "Selected CatBoost-HGB Weighted 2Model",
]

STAGE_COLORS = {
    "RF baseline": PURPLE,
    "RF-CatBoost weighted fusion": BLUE,
    "Ridge stacking fusion": GREEN,
    "Tuned CatBoost": LIGHT_PURPLE,
    "Selected CatBoost": LIGHT_PURPLE,
    "Selected HGB": GREEN,
    "Selected Weighted 4Model": ORANGE,
    "Selected CatBoost-HGB Weighted 2Model": TEAL,
}

STAGE_SHORT_LABELS = {
    "RF baseline": "RF",
    "RF-CatBoost weighted fusion": "RF-Cat",
    "Ridge stacking fusion": "Ridge stack",
    "Tuned CatBoost": "Tuned Cat",
    "Selected CatBoost": "Sel. Cat",
    "Selected HGB": "Sel. HGB",
    "Selected Weighted 4Model": "Sel. 4Model",
    "Selected CatBoost-HGB Weighted 2Model": "Final 2Model",
}

PRED_PANEL_TITLES = {
    "RF baseline": "RF baseline",
    "RF-CatBoost weighted fusion": "RF-CatBoost weighted fusion",
    "Ridge stacking fusion": "Ridge stacking fusion",
    "Tuned CatBoost": "Tuned CatBoost",
    "Selected CatBoost": "Selected CatBoost",
    "Selected HGB": "Selected HGB",
    "Selected Weighted 4Model": "Selected Weighted 4Model",
    "Selected CatBoost-HGB Weighted 2Model": "Final CatBoost-HGB Weighted 2Model",
}


# ============================================================
# 3. Basic utilities
# ============================================================

def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue

    raise ValueError(f"Cannot read CSV file: {path}")


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def safe_numeric(x) -> pd.Series:
    return pd.to_numeric(pd.Series(x), errors="coerce")


def to_percent_if_fraction(x) -> pd.Series:
    s = safe_numeric(x)
    valid = s.dropna()

    if len(valid) == 0:
        return s

    # If values are mainly 0-1, treat them as fractions.
    if valid.quantile(0.99) <= 1.2:
        return s * 100.0

    return s


def metric_values(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return np.nan, np.nan, np.nan, np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)

    return r2, rmse, mae, mse


def sample_for_plot(df: pd.DataFrame, n=650, random_state=42) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()

    return df.sample(n=n, random_state=random_state)


def add_panel_label(ax, label: str):
    ax.text(
        0.03,
        0.965,
        f"({label})",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="top",
        color=BLACK,
    )


def clean_stage_name(stage: str) -> str:
    return str(stage).strip()


# ============================================================
# 4. Load prediction data
# ============================================================

def load_wide_prediction_file(
    file_name: str,
    y_col: str,
    pred_col: str,
    stage: str,
    source_label: str,
    seed_col: str = "随机种子",
) -> pd.DataFrame:
    path = require_file(TABLE_DIR / file_name)
    df = read_csv_smart(path)

    if y_col not in df.columns:
        raise KeyError(f"Column '{y_col}' not found in {file_name}. Available columns: {list(df.columns)}")

    if pred_col not in df.columns:
        raise KeyError(f"Column '{pred_col}' not found in {file_name}. Available columns: {list(df.columns)}")

    out = pd.DataFrame({
        "Stage": stage,
        "Source": source_label,
        "Seed": df[seed_col] if seed_col in df.columns else np.nan,
        "Observed": to_percent_if_fraction(df[y_col]),
        "Predicted": to_percent_if_fraction(df[pred_col]),
    })

    out = out.dropna(subset=["Observed", "Predicted"])

    if len(out) < 10:
        raise ValueError(f"Too few valid prediction rows for stage: {stage}")

    return out


def load_long_method_prediction(
    file_name: str,
    method: str,
    stage: str,
    source_label: str,
    required: bool = True,
) -> pd.DataFrame:
    path = TABLE_DIR / file_name

    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.DataFrame()

    df = read_csv_smart(path)

    required_cols = ["Method", "Observed", "Predicted"]
    for c in required_cols:
        if c not in df.columns:
            if required:
                raise KeyError(f"Column '{c}' not found in {file_name}. Available columns: {list(df.columns)}")
            return pd.DataFrame()

    mask = df["Method"].astype(str).eq(method)

    if mask.sum() == 0:
        if required:
            print(f"[ERROR] Method '{method}' not found in {file_name}. Available methods:")
            for m in sorted(df["Method"].astype(str).unique()):
                print(f"    - {m}")
            raise ValueError(f"Method '{method}' not found in {file_name}.")
        return pd.DataFrame()

    sub = df.loc[mask].copy()

    out = pd.DataFrame({
        "Stage": stage,
        "Source": source_label,
        "Seed": sub["Seed"] if "Seed" in sub.columns else np.nan,
        "Observed": to_percent_if_fraction(sub["Observed"]),
        "Predicted": to_percent_if_fraction(sub["Predicted"]),
    })

    out = out.dropna(subset=["Observed", "Predicted"])

    if len(out) < 10 and required:
        raise ValueError(f"Too few valid rows for method '{method}' from {file_name}")

    return out


def load_selected_method_with_fallback(method: str, stage: str) -> pd.DataFrame:
    """
    Prefer exact ablation output from script 24.
    If it is absent, fall back to script 19 for selected single models and 4Model.

    For the final 2Model, script 24 is mandatory.
    """
    if method == "Selected_CatBoost_HGB_Weighted":
        return load_long_method_prediction(
            file_name="24_exact_ablation_predictions.csv",
            method=method,
            stage=stage,
            source_label="24_exact_ablation_predictions.csv",
            required=True,
        )

    out = load_long_method_prediction(
        file_name="24_exact_ablation_predictions.csv",
        method=method,
        stage=stage,
        source_label="24_exact_ablation_predictions.csv",
        required=False,
    )

    if len(out) >= 10:
        return out

    out = load_long_method_prediction(
        file_name="19_selected_fusion_predictions.csv",
        method=method,
        stage=stage,
        source_label="19_selected_fusion_predictions.csv",
        required=True,
    )

    return out


def collect_stage_predictions() -> pd.DataFrame:
    stage_frames = []

    # A. RF baseline
    stage_frames.append(
        load_wide_prediction_file(
            file_name="12_multiseed_rf_catboost_predictions.csv",
            y_col="真实氟去除率_pct",
            pred_col="RandomForest预测",
            stage="RF baseline",
            source_label="12_multiseed_rf_catboost_predictions.csv",
        )
    )

    # B. RF-CatBoost weighted fusion
    stage_frames.append(
        load_wide_prediction_file(
            file_name="12_multiseed_rf_catboost_predictions.csv",
            y_col="真实氟去除率_pct",
            pred_col="RF_CatBoost加权预测",
            stage="RF-CatBoost weighted fusion",
            source_label="12_multiseed_rf_catboost_predictions.csv",
        )
    )

    # C. Ridge stacking fusion
    stage_frames.append(
        load_wide_prediction_file(
            file_name="19_stacking_multiseed_predictions.csv",
            y_col="真实氟去除率_pct",
            pred_col="RidgeStacking预测",
            stage="Ridge stacking fusion",
            source_label="19_stacking_multiseed_predictions.csv",
        )
    )

    # D. Tuned CatBoost, preferably multi-seed version.
    stage_frames.append(
        load_wide_prediction_file(
            file_name="16_tunedcat_multiseed_predictions.csv",
            y_col="真实氟去除率_pct",
            pred_col="CatBoost_tuned预测",
            stage="Tuned CatBoost",
            source_label="16_tunedcat_multiseed_predictions.csv",
        )
    )

    # E-H. Final selected-feature workflow and exact ablation outputs.
    stage_frames.append(
        load_selected_method_with_fallback(
            method="Selected_CatBoost",
            stage="Selected CatBoost",
        )
    )

    stage_frames.append(
        load_selected_method_with_fallback(
            method="Selected_HistGradientBoosting",
            stage="Selected HGB",
        )
    )

    stage_frames.append(
        load_selected_method_with_fallback(
            method="Selected_Weighted_4Model",
            stage="Selected Weighted 4Model",
        )
    )

    stage_frames.append(
        load_selected_method_with_fallback(
            method="Selected_CatBoost_HGB_Weighted",
            stage="Selected CatBoost-HGB Weighted 2Model",
        )
    )

    pred_all = pd.concat(stage_frames, ignore_index=True)

    pred_all["Stage"] = pred_all["Stage"].astype(str).map(clean_stage_name)
    pred_all["Stage"] = pd.Categorical(pred_all["Stage"], categories=STAGE_ORDER, ordered=True)

    pred_all = pred_all.sort_values(["Stage", "Seed"]).reset_index(drop=True)

    return pred_all


def build_metrics(pred_all: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for stage in STAGE_ORDER:
        g = pred_all[pred_all["Stage"].astype(str) == stage].copy()

        if g.empty:
            continue

        r2, rmse, mae, mse = metric_values(g["Observed"], g["Predicted"])

        rows.append({
            "Stage": stage,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MSE": mse,
            "n_points": len(g),
            "source": "; ".join(sorted(g["Source"].astype(str).unique())),
        })

    metrics = pd.DataFrame(rows)
    metrics["Stage"] = pd.Categorical(metrics["Stage"], categories=STAGE_ORDER, ordered=True)
    metrics = metrics.sort_values("Stage").reset_index(drop=True)

    return metrics


# ============================================================
# 5. Load final 2Model weights
# ============================================================

def load_final_2model_weight_summary() -> pd.DataFrame:
    path = TABLE_DIR / "24_exact_ablation_weights.csv"

    if not path.exists():
        raise FileNotFoundError(
            "24_exact_ablation_weights.csv not found. "
            "Run 24_exact_ablation_selected_fusion.py first."
        )

    df = read_csv_smart(path)

    required_cols = ["Seed", "Method", "Model", "Weight"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in 24_exact_ablation_weights.csv")

    sub = df[df["Method"].astype(str).eq("Selected_CatBoost_HGB_Weighted")].copy()

    if sub.empty:
        print("[ERROR] Available methods in 24_exact_ablation_weights.csv:")
        for m in sorted(df["Method"].astype(str).unique()):
            print(f"    - {m}")
        raise ValueError("Selected_CatBoost_HGB_Weighted not found in 24_exact_ablation_weights.csv")

    sub["Weight"] = pd.to_numeric(sub["Weight"], errors="coerce")
    sub = sub.dropna(subset=["Weight"])

    def rename_base_model(x):
        x = str(x)

        if "CatBoost" in x:
            return "CatBoost"

        if "HistGradientBoosting" in x or "HGB" in x:
            return "HGB"

        return x

    sub["Base model"] = sub["Model"].map(rename_base_model)

    # Normalize within each seed as a safety step.
    sub["Weight_norm"] = np.nan

    for seed, g in sub.groupby("Seed"):
        total = g["Weight"].sum()
        if total > 0:
            sub.loc[g.index, "Weight_norm"] = g["Weight"] / total

    sub = sub.dropna(subset=["Weight_norm"])

    if sub.empty:
        raise ValueError("No valid normalized final 2Model weights.")

    summary = (
        sub
        .groupby("Base model", as_index=False)
        .agg(
            mean_weight=("Weight_norm", "mean"),
            std_weight=("Weight_norm", "std"),
            min_weight=("Weight_norm", "min"),
            max_weight=("Weight_norm", "max"),
            n_seed=("Seed", "nunique"),
        )
    )

    summary["std_weight"] = summary["std_weight"].fillna(0)

    # Normalize average weights again for display.
    total_mean = summary["mean_weight"].sum()
    if total_mean > 0:
        summary["mean_weight"] = summary["mean_weight"] / total_mean

    order_map = {"CatBoost": 0, "HGB": 1}
    summary["order"] = summary["Base model"].map(order_map).fillna(99)
    summary = summary.sort_values("order").drop(columns=["order"]).reset_index(drop=True)

    return summary


def load_final_2model_weight_by_seed() -> pd.DataFrame:
    path = require_file(TABLE_DIR / "24_exact_ablation_weights.csv")
    df = read_csv_smart(path)

    sub = df[df["Method"].astype(str).eq("Selected_CatBoost_HGB_Weighted")].copy()

    if sub.empty:
        return pd.DataFrame()

    sub["Weight"] = pd.to_numeric(sub["Weight"], errors="coerce")
    sub = sub.dropna(subset=["Weight"])

    def rename_base_model(x):
        x = str(x)

        if "CatBoost" in x:
            return "CatBoost"

        if "HistGradientBoosting" in x or "HGB" in x:
            return "HGB"

        return x

    sub["Base model"] = sub["Model"].map(rename_base_model)

    # Normalize within seed.
    out_frames = []

    for seed, g in sub.groupby("Seed"):
        g = g.copy()
        total = g["Weight"].sum()

        if total > 0:
            g["Weight_norm"] = g["Weight"] / total
            out_frames.append(g)

    if not out_frames:
        return pd.DataFrame()

    return pd.concat(out_frames, ignore_index=True)


# ============================================================
# 6. Plot helpers
# ============================================================

def add_cov_ellipse(ax, x, y, color, n_std=1.10):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 20:
        return

    # Remove extreme points for visual ellipse only.
    try:
        qx1, qx2 = np.percentile(x, [3, 97])
        qy1, qy2 = np.percentile(y, [3, 97])
        keep = (x >= qx1) & (x <= qx2) & (y >= qy1) & (y <= qy2)

        x = x[keep]
        y = y[keep]
    except Exception:
        return

    if len(x) < 20:
        return

    cov = np.cov(x, y)

    if not np.all(np.isfinite(cov)):
        return

    try:
        vals, vecs = np.linalg.eigh(cov)
    except Exception:
        return

    if np.any(vals <= 0):
        return

    order = vals.argsort()[::-1]

    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linestyle=(0, (4, 3)),
        linewidth=1.05,
        alpha=0.82,
    )

    ax.add_patch(ellipse)


def plot_pred_obs(ax, pred_all: pd.DataFrame, stage: str, panel_label: str):
    g = pred_all[pred_all["Stage"].astype(str) == stage].copy()

    if g.empty:
        ax.axis("off")
        add_panel_label(ax, panel_label)
        ax.text(0.5, 0.5, f"No data\n{stage}", ha="center", va="center")
        return

    g_s = sample_for_plot(
        g,
        n=650,
        random_state=101 + ord(panel_label[0])
    )

    color = STAGE_COLORS.get(stage, TEAL)
    r2, rmse, mae, mse = metric_values(g["Observed"], g["Predicted"])

    ax.scatter(
        g_s["Observed"],
        g_s["Predicted"],
        s=10.5,
        color=color,
        alpha=0.66,
        edgecolor="none",
    )

    ax.plot(
        [0, 100],
        [0, 100],
        color=RED,
        linestyle=(0, (4, 3)),
        linewidth=0.95,
    )

    add_cov_ellipse(
        ax,
        g_s["Observed"],
        g_s["Predicted"],
        color=color,
        n_std=1.08,
    )

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)

    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Predicted F removal (%)")

    ax.set_title(
        PRED_PANEL_TITLES.get(stage, stage),
        pad=4,
        fontsize=8.5,
        fontweight="bold" if stage == "Selected CatBoost-HGB Weighted 2Model" else "normal",
    )

    ax.grid(False)

    # Metric box: use lower-right to avoid overlapping panel label.
    ax.text(
        0.58,
        0.075,
        f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="white",
            edgecolor="none",
            alpha=0.78,
        )
    )

    add_panel_label(ax, panel_label)


def plot_residual_panel(ax, pred_all: pd.DataFrame, panel_label: str):
    final_stage = "Selected CatBoost-HGB Weighted 2Model"

    g = pred_all[pred_all["Stage"].astype(str) == final_stage].copy()

    if g.empty:
        ax.axis("off")
        add_panel_label(ax, panel_label)
        ax.text(0.5, 0.5, "No final 2Model data", ha="center", va="center")
        return

    g_s = sample_for_plot(g, n=850, random_state=202)

    residual_all = g["Observed"] - g["Predicted"]
    residual_s = g_s["Observed"] - g_s["Predicted"]

    rmse_band = math.sqrt(np.mean(np.asarray(residual_all, dtype=float) ** 2))
    bias = np.mean(residual_all)

    ax.scatter(
        g_s["Predicted"],
        residual_s,
        s=10.5,
        color=TEAL,
        alpha=0.63,
        edgecolor="none",
    )

    ax.axhline(
        0,
        color=RED,
        linestyle=(0, (4, 3)),
        linewidth=0.95,
    )

    ax.axhspan(
        -rmse_band,
        rmse_band,
        color=TEAL,
        alpha=0.10,
        linewidth=0,
    )

    ax.set_xlim(0, 105)

    y_abs = np.nanpercentile(np.abs(residual_s), 98)
    y_lim = max(35, min(90, y_abs * 1.20))
    ax.set_ylim(-y_lim, y_lim)

    ax.set_xlabel("Predicted F removal (%)")
    ax.set_ylabel("Residual (%)")
    ax.set_title("Residual diagnosis of final 2Model", pad=4, fontsize=8.5)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.55)

    ax.text(
        0.05,
        0.08,
        f"RMSE band: +/-{rmse_band:.2f}%\nMean bias: {bias:+.2f}%",
        transform=ax.transAxes,
        fontsize=6.7,
        color=GRAY,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="white",
            edgecolor="none",
            alpha=0.72,
        )
    )

    add_panel_label(ax, panel_label)


def plot_r2_comparison(ax, metrics: pd.DataFrame, panel_label: str):
    data = metrics.copy()
    data["Stage"] = pd.Categorical(data["Stage"], categories=STAGE_ORDER, ordered=True)
    data = data.sort_values("Stage")

    x = np.arange(len(data))
    vals = data["R2"].values
    colors = [STAGE_COLORS.get(str(s), TEAL) for s in data["Stage"]]

    bars = ax.bar(
        x,
        vals,
        color=colors,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.90,
    )

    # Emphasize final model with thicker edge.
    for bar, stage in zip(bars, data["Stage"].astype(str)):
        if stage == "Selected CatBoost-HGB Weighted 2Model":
            bar.set_edgecolor(RED)
            bar.set_linewidth(1.1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [STAGE_SHORT_LABELS.get(str(s), str(s)) for s in data["Stage"]],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel("Pooled $R^2$")
    ax.set_title("Pooled $R^2$ comparison", pad=4, fontsize=8.5)
    ax.set_ylim(0, min(1.0, max(0.90, np.nanmax(vals) * 1.12)))
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.55)

    y_top = ax.get_ylim()[1]

    for xi, v in zip(x, vals):
        ax.text(
            xi,
            v + y_top * 0.017,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=5.9,
        )



    add_panel_label(ax, panel_label)


def plot_error_comparison(ax, metrics: pd.DataFrame, panel_label: str):
    data = metrics.copy()
    data["Stage"] = pd.Categorical(data["Stage"], categories=STAGE_ORDER, ordered=True)
    data = data.sort_values("Stage")

    x = np.arange(len(data))
    width = 0.35

    rmse_vals = data["RMSE"].values
    mae_vals = data["MAE"].values

    ax.bar(
        x - width / 2,
        rmse_vals,
        width,
        color=DARK,
        edgecolor="none",
        label="RMSE",
        alpha=0.92,
    )

    ax.bar(
        x + width / 2,
        mae_vals,
        width,
        color=MAGENTA,
        edgecolor="none",
        label="MAE",
        alpha=0.92,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [STAGE_SHORT_LABELS.get(str(s), str(s)) for s in data["Stage"]],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel("Metric value")
    ax.set_title("Prediction error comparison", pad=4, fontsize=8.5)

    ymax = max(np.nanmax(rmse_vals), np.nanmax(mae_vals)) * 1.24
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.55)

    for xi, rmse, mae in zip(x, rmse_vals, mae_vals):
        ax.text(
            xi - width / 2,
            rmse + ymax * 0.014,
            f"{rmse:.2f}",
            ha="center",
            va="bottom",
            fontsize=5.5,
        )

        ax.text(
            xi + width / 2,
            mae + ymax * 0.014,
            f"{mae:.2f}",
            ha="center",
            va="bottom",
            fontsize=5.5,
        )

    ax.legend(frameon=False, loc="upper right", fontsize=6.4)



    add_panel_label(ax, panel_label)


def plot_weight_stability_panel(
    ax,
    weight_summary: pd.DataFrame,
    weight_by_seed: pd.DataFrame,
    panel_label: str,
):
    if weight_summary.empty:
        ax.axis("off")
        add_panel_label(ax, panel_label)
        ax.text(0.5, 0.5, "No final 2Model weight data", ha="center", va="center")
        return

    data = weight_summary.copy()

    data = data.sort_values("mean_weight", ascending=True)

    y = np.arange(len(data))

    color_map = {
        "CatBoost": LIGHT_PURPLE,
        "HGB": GREEN,
    }

    bar_colors = [color_map.get(m, TEAL) for m in data["Base model"]]

    ax.barh(
        y,
        data["mean_weight"],
        xerr=data["std_weight"].fillna(0),
        color=bar_colors,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.88,
        capsize=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(data["Base model"])

    ax.set_xlabel("Normalized mean selected weight")
    ax.set_title("Final 2Model weight stability", pad=4, fontsize=8.5)

    xmax = min(1.0, max(0.75, float(data["mean_weight"].max()) * 1.35))
    ax.set_xlim(0, xmax)

    ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.55)

    for yi, v, sd in zip(y, data["mean_weight"], data["std_weight"]):
        ax.text(
            min(v + xmax * 0.035, xmax * 0.96),
            yi,
            f"{v:.2f} +/- {sd:.2f}",
            va="center",
            ha="left",
            fontsize=6.2,
        )

    # Add a compact seed-level line inset-like textual summary.
    if not weight_by_seed.empty:
        seeds = sorted(weight_by_seed["Seed"].dropna().unique())
        seed_text = f"n seeds = {len(seeds)}"
    else:
        seed_text = f"n seeds = {int(data['n_seed'].max())}"

    ax.text(
        0.04,
        0.08,
        seed_text,
        transform=ax.transAxes,
        fontsize=6.4,
        color=GRAY,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="white",
            edgecolor="none",
            alpha=0.70,
        )
    )

    add_panel_label(ax, panel_label)


# ============================================================
# 7. Main
# ============================================================

def main():
    print("=" * 100)
    print("Generating expanded final EST-style Figure 3 with 8 prediction panels ...")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Table dir    : {TABLE_DIR}")
    print(f"Figure dir   : {FIG_DIR}")
    print("=" * 100)

    pred_all = collect_stage_predictions()
    metrics = build_metrics(pred_all)

    weight_summary = load_final_2model_weight_summary()
    weight_by_seed = load_final_2model_weight_by_seed()

    # Save data used for figure.
    metrics_out = TABLE_DIR / "25_est_figure3_expanded_metrics_used.csv"
    weights_out = TABLE_DIR / "25_est_figure3_expanded_weight_summary.csv"
    source_out = TABLE_DIR / "25_est_figure3_expanded_stage_sources.csv"

    metrics.to_csv(metrics_out, index=False, encoding="utf-8-sig")
    weight_summary.to_csv(weights_out, index=False, encoding="utf-8-sig")

    source_df = (
        pred_all
        .groupby(["Stage", "Source"], observed=False)
        .size()
        .reset_index(name="n_rows")
    )

    source_df.to_csv(source_out, index=False, encoding="utf-8-sig")

    print("[OK] Stage predictions loaded:")
    for stage in STAGE_ORDER:
        n = int((pred_all["Stage"].astype(str) == stage).sum())
        print(f"    - {stage}: {n} rows")

    print("\n[OK] Metrics used in Figure 3:")
    print(metrics[["Stage", "R2", "RMSE", "MAE", "n_points"]].to_string(index=False))

    print("\n[OK] Final 2Model weight summary:")
    print(weight_summary.to_string(index=False))

    # --------------------------------------------------------
    # Figure layout: 3 rows x 4 columns
    # A-H: prediction panels
    # I: residual
    # J: R2
    # K: RMSE/MAE
    # L: weights
    # --------------------------------------------------------
    fig = plt.figure(figsize=(16.2, 12.6), constrained_layout=False)

    gs = GridSpec(
        3,
        4,
        figure=fig,
        left=0.055,
        right=0.985,
        top=0.905,
        bottom=0.185,
        wspace=0.32,
        hspace=0.52,
    )

    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(4)]

    # A-H
    for i, stage in enumerate(STAGE_ORDER):
        plot_pred_obs(
            ax=axes[i],
            pred_all=pred_all,
            stage=stage,
            panel_label=chr(ord("A") + i),
        )

    # I
    plot_residual_panel(
        ax=axes[8],
        pred_all=pred_all,
        panel_label="I",
    )

    # J
    plot_r2_comparison(
        ax=axes[9],
        metrics=metrics,
        panel_label="J",
    )

    # K
    plot_error_comparison(
        ax=axes[10],
        metrics=metrics,
        panel_label="K",
    )

    # L
    plot_weight_stability_panel(
        ax=axes[11],
        weight_summary=weight_summary,
        weight_by_seed=weight_by_seed,
        panel_label="L",
    )

    fig.suptitle(
        "Workflow-oriented validation of the final CatBoost-HGB weighted model",
        fontsize=14.0,
        fontweight="bold",
        y=0.975,
    )

    caption = (
        "Figure 3. Workflow-oriented validation of machine-learning models for fluoride removal prediction. "
        "Panels A-H show representative predicted-vs-observed comparisons along the model-development pathway, "
        "including the RF baseline, RF-CatBoost weighted fusion, Ridge stacking, tuned CatBoost, selected CatBoost, "
        "selected HGB, selected weighted 4Model, and the final selected CatBoost-HGB weighted 2Model. "
        "Panel I presents residual diagnosis of the final 2Model. "
        "Panels J and K compare pooled R2, RMSE, and MAE across representative stages. "
        "Panel L shows the normalized validation-optimized fusion weights of the final CatBoost-HGB weighted model. "
        "The final 2Model was retained based on exact ablation under the same feature-engineering, feature-selection, "
        "splitting, model-parameter, and validation-RMSE weight-search workflow as the selected-feature fusion model."
    )

    wrapped_caption = textwrap.fill(caption, width=190)

    fig.text(
        0.055,
        0.055,
        wrapped_caption,
        ha="left",
        va="bottom",
        fontsize=8.1,
        color=GRAY,
        bbox=dict(
            boxstyle="round,pad=0.38",
            facecolor="white",
            edgecolor="lightgray",
            linewidth=0.55,
            alpha=0.96,
        )
    )

    png_path = FIG_DIR / "Figure3_final_EST_workflow_2Model_expanded.png"
    pdf_path = FIG_DIR / "Figure3_final_EST_workflow_2Model_expanded.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("=" * 100)
    print("[DONE] Expanded final Figure 3 generated successfully.")
    print(f"PNG     : {png_path}")
    print(f"PDF     : {pdf_path}")
    print(f"METRICS : {metrics_out}")
    print(f"WEIGHTS : {weights_out}")
    print(f"SOURCES : {source_out}")
    print("=" * 100)


if __name__ == "__main__":
    main()