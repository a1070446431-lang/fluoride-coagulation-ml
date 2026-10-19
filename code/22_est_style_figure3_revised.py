# -*- coding: utf-8 -*-
"""
22_est_style_figure3_revised.py

Revised EST-style Figure 3 for the fluoride coagulation ML project.

This version explicitly uses the staged model outputs generated in the project:

A: RF baseline
   Source: 12_multiseed_rf_catboost_predictions.csv
   Column: RandomForest预测

B: Early RF-CatBoost weighted fusion
   Source: 12_multiseed_rf_catboost_predictions.csv
   Column: RF_CatBoost加权预测

C: Ridge stacking fusion
   Source: 19_stacking_multiseed_predictions.csv
   Column: RidgeStacking预测

D: Final selected weighted 4Model
   Source: 19_selected_fusion_predictions.csv
   Method: Selected Weighted 4Model

E: Residual diagnosis of final model
F: Stage-wise R2 comparison
G: Stage-wise RMSE / MAE comparison
H: Final fusion weight stability

Outputs:
    outputs/figures/Figure3_revised_EST_workflow_comparison.png
    outputs/figures/Figure3_revised_EST_workflow_comparison.pdf
    outputs/tables/22_est_figure3_revised_metrics_used.csv
    outputs/tables/22_est_figure3_revised_weight_summary.csv
"""

from pathlib import Path
import re
import warnings
import textwrap

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
    candidates = [current, current.parent, current.parent.parent]

    for p in candidates:
        if (p / "outputs" / "tables").exists() and (p / "outputs" / "figures").exists():
            return p

    if current.name.lower() == "src":
        return current.parent

    raise FileNotFoundError("Cannot locate project root.")


PROJECT_ROOT = find_project_root()
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Style
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9.0,
    "axes.linewidth": 0.9,
    "axes.labelsize": 8.6,
    "axes.titlesize": 9.2,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 7.0,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

PURPLE = "#7E62A8"
BLUE = "#4C78A8"
GREEN = "#63A97A"
TEAL = "#63B7B2"
RED = "#FF0000"
DARK = "#3B3B3B"
MAGENTA = "#EB4AA9"
GRAY = "#666666"
LIGHT_GRID = "#E7E7E7"

STAGE_COLORS = {
    "RF baseline": PURPLE,
    "RF-CatBoost weighted fusion": BLUE,
    "Ridge stacking fusion": GREEN,
    "Selected Weighted 4Model": TEAL,
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
    raise ValueError(f"Cannot read csv file: {path}")


def safe_numeric(x) -> pd.Series:
    return pd.to_numeric(pd.Series(x), errors="coerce")


def to_percent_if_fraction(x) -> pd.Series:
    """
    If the values look like 0-1 fractions, convert them to 0-100 percentages.
    """
    s = safe_numeric(x)
    valid = s.dropna()

    if len(valid) == 0:
        return s

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
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)

    return r2, rmse, mae, mse


def short_stage_label(stage: str) -> str:
    mapping = {
        "RF baseline": "RF baseline",
        "RF-CatBoost weighted fusion": "RF-CatBoost",
        "Ridge stacking fusion": "Ridge stacking",
        "Selected Weighted 4Model": "Selected 4Model",
    }
    return mapping.get(stage, stage)


def clean_model_name_for_weight(x: str) -> str:
    s = str(x).strip()
    low = s.lower()

    if "rf" in low or "random" in low:
        return "RF"
    if "cat" in low:
        return "CatBoost"
    if "hgb" in low or "hist" in low:
        return "HGB"
    if "extra" in low:
        return "ExtraTrees"

    return s.replace("_", " ")


# ============================================================
# 4. Load staged prediction data explicitly
# ============================================================

def load_wide_prediction_file(file_name: str, y_col: str, pred_col: str,
                              stage: str, source_label: str) -> pd.DataFrame:
    path = TABLE_DIR / file_name

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    df = read_csv_smart(path)

    if y_col not in df.columns:
        raise KeyError(f"Column '{y_col}' not found in {file_name}")

    if pred_col not in df.columns:
        raise KeyError(f"Column '{pred_col}' not found in {file_name}")

    out = pd.DataFrame({
        "Stage": stage,
        "Source": source_label,
        "Seed": df["随机种子"] if "随机种子" in df.columns else np.nan,
        "Observed": to_percent_if_fraction(df[y_col]),
        "Predicted": to_percent_if_fraction(df[pred_col]),
    })

    out = out.dropna(subset=["Observed", "Predicted"])

    if len(out) < 10:
        raise ValueError(f"Too few valid prediction rows for {stage} from {file_name}")

    return out


def load_selected_weighted_predictions() -> pd.DataFrame:
    file_name = "19_selected_fusion_predictions.csv"
    path = TABLE_DIR / file_name

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    df = read_csv_smart(path)

    required_cols = ["Seed", "Method", "Observed", "Predicted"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in {file_name}")

    method_text = df["Method"].astype(str)

    mask = (
        method_text.str.contains("Selected", case=False, na=False)
        & method_text.str.contains("Weighted", case=False, na=False)
        & method_text.str.contains("4", case=False, na=False)
    )

    if mask.sum() == 0:
        print("[WARN] Could not find exact Selected Weighted 4Model. Available methods:")
        print(sorted(df["Method"].astype(str).unique()))
        raise ValueError("Selected Weighted 4Model not found in 19_selected_fusion_predictions.csv")

    sub = df.loc[mask].copy()

    out = pd.DataFrame({
        "Stage": "Selected Weighted 4Model",
        "Source": "19_selected_fusion_predictions.csv",
        "Seed": sub["Seed"],
        "Observed": to_percent_if_fraction(sub["Observed"]),
        "Predicted": to_percent_if_fraction(sub["Predicted"]),
    })

    out = out.dropna(subset=["Observed", "Predicted"])

    if len(out) < 10:
        raise ValueError("Too few valid rows for Selected Weighted 4Model")

    return out


def collect_stage_predictions() -> pd.DataFrame:
    """
    Fixed staged model sources based on actual project output files.
    """
    stages = []

    stages.append(
        load_wide_prediction_file(
            file_name="12_multiseed_rf_catboost_predictions.csv",
            y_col="真实氟去除率_pct",
            pred_col="RandomForest预测",
            stage="RF baseline",
            source_label="12_multiseed_rf_catboost_predictions.csv"
        )
    )

    stages.append(
        load_wide_prediction_file(
            file_name="12_multiseed_rf_catboost_predictions.csv",
            y_col="真实氟去除率_pct",
            pred_col="RF_CatBoost加权预测",
            stage="RF-CatBoost weighted fusion",
            source_label="12_multiseed_rf_catboost_predictions.csv"
        )
    )

    stages.append(
        load_wide_prediction_file(
            file_name="19_stacking_multiseed_predictions.csv",
            y_col="真实氟去除率_pct",
            pred_col="RidgeStacking预测",
            stage="Ridge stacking fusion",
            source_label="19_stacking_multiseed_predictions.csv"
        )
    )

    stages.append(load_selected_weighted_predictions())

    pred_all = pd.concat(stages, ignore_index=True)

    stage_order = [
        "RF baseline",
        "RF-CatBoost weighted fusion",
        "Ridge stacking fusion",
        "Selected Weighted 4Model",
    ]

    pred_all["Stage"] = pd.Categorical(pred_all["Stage"], categories=stage_order, ordered=True)
    pred_all = pred_all.sort_values(["Stage", "Seed"]).reset_index(drop=True)

    return pred_all


def build_metrics(pred_all: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for stage, g in pred_all.groupby("Stage", observed=True):
        r2, rmse, mae, mse = metric_values(g["Observed"], g["Predicted"])

        rows.append({
            "Stage": str(stage),
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MSE": mse,
            "n_points": len(g),
            "source": "; ".join(sorted(g["Source"].astype(str).unique())),
        })

    return pd.DataFrame(rows)


# ============================================================
# 5. Load final fusion weights
# ============================================================

def load_final_weights() -> pd.DataFrame:
    file_name = "19_selected_fusion_weights.csv"
    path = TABLE_DIR / file_name

    if not path.exists():
        print(f"[WARN] Weight file not found: {path}")
        return pd.DataFrame()

    df = read_csv_smart(path)

    required = ["Seed", "Method", "Model", "Weight"]
    for c in required:
        if c not in df.columns:
            print(f"[WARN] Column {c} not found in {file_name}")
            return pd.DataFrame()

    method_text = df["Method"].astype(str)

    mask = (
        method_text.str.contains("Selected", case=False, na=False)
        & method_text.str.contains("Weighted", case=False, na=False)
    )

    sub = df.loc[mask].copy()

    if sub.empty:
        print("[WARN] No Selected Weighted rows found in weight file.")
        return pd.DataFrame()

    sub["Weight"] = pd.to_numeric(sub["Weight"], errors="coerce")
    sub = sub.dropna(subset=["Weight"])
    sub = sub[sub["Weight"] >= 0]

    if sub.empty:
        return pd.DataFrame()

    sub["Base model"] = sub["Model"].map(clean_model_name_for_weight)

    # normalize within each seed
    sub["Weight_norm"] = np.nan
    for seed, g in sub.groupby("Seed"):
        total = g["Weight"].sum()
        if total > 0:
            sub.loc[g.index, "Weight_norm"] = g["Weight"] / total

    sub = sub.dropna(subset=["Weight_norm"])

    if sub.empty:
        return pd.DataFrame()

    out = (
        sub
        .groupby("Base model", as_index=False)
        .agg(
            mean_weight=("Weight_norm", "mean"),
            std_weight=("Weight_norm", "std"),
            n=("Weight_norm", "count")
        )
    )

    out["std_weight"] = out["std_weight"].fillna(0)

    # final safety normalization
    total = out["mean_weight"].sum()
    if total > 0:
        out["mean_weight"] = out["mean_weight"] / total
        out["std_weight"] = out["std_weight"] / total

    out = out.sort_values("mean_weight", ascending=False).reset_index(drop=True)

    return out


# ============================================================
# 6. Plot helpers
# ============================================================

def sample_for_plot(df: pd.DataFrame, n=700, random_state=42):
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=random_state)


def add_panel_label(ax, label):
    ax.text(
        0.03, 0.96,
        f"({label})",
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        ha="left",
        va="top"
    )


def add_cov_ellipse(ax, x, y, color, n_std=1.10):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 20:
        return

    qx1, qx2 = np.percentile(x, [3, 97])
    qy1, qy2 = np.percentile(y, [3, 97])

    keep = (x >= qx1) & (x <= qx2) & (y >= qy1) & (y <= qy2)
    x = x[keep]
    y = y[keep]

    if len(x) < 20:
        return

    cov = np.cov(x, y)

    if not np.all(np.isfinite(cov)):
        return

    vals, vecs = np.linalg.eigh(cov)

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
        linewidth=1.2,
        alpha=0.85
    )

    ax.add_patch(ellipse)


def plot_pred_obs(ax, pred_all, stage, label, display_title):
    g = pred_all[pred_all["Stage"].astype(str) == stage].copy()
    g_s = sample_for_plot(g, n=700, random_state=100 + ord(label))

    color = STAGE_COLORS.get(stage, TEAL)
    r2, rmse, mae, mse = metric_values(g["Observed"], g["Predicted"])

    ax.scatter(
        g_s["Observed"],
        g_s["Predicted"],
        s=12,
        color=color,
        alpha=0.72,
        edgecolor="none"
    )

    ax.plot(
        [0, 100],
        [0, 100],
        color=RED,
        linestyle=(0, (4, 3)),
        linewidth=1.0
    )

    add_cov_ellipse(ax, g_s["Observed"], g_s["Predicted"], color=color, n_std=1.10)

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)

    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Predicted F removal (%)")
    ax.set_title(display_title, pad=4)

    ax.grid(False)

    ax.text(
        0.58,
        0.08,
        f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.78
        )
    )

    add_panel_label(ax, label)


def plot_residual_panel(ax, pred_all, label):
    final_stage = "Selected Weighted 4Model"
    g = pred_all[pred_all["Stage"].astype(str) == final_stage].copy()
    g_s = sample_for_plot(g, n=800, random_state=88)

    residual = g_s["Observed"] - g_s["Predicted"]
    residual_all = g["Observed"] - g["Predicted"]
    rmse_res = np.sqrt(np.mean(residual_all ** 2))

    ax.scatter(
        g_s["Predicted"],
        residual,
        s=12,
        color=TEAL,
        alpha=0.70,
        edgecolor="none"
    )

    ax.axhline(
        0,
        color=RED,
        linestyle=(0, (4, 3)),
        linewidth=1.0
    )

    ax.axhspan(-rmse_res, rmse_res, color=TEAL, alpha=0.10, linewidth=0)

    ax.set_xlim(0, 105)
    ax.set_xlabel("Predicted F removal (%)")
    ax.set_ylabel("Residual (%)")
    ax.set_title("Residual diagnosis of final model", pad=4)
    ax.grid(False)

    ax.text(
        0.05,
        0.08,
        f"RMSE band: +/-{rmse_res:.2f}%",
        transform=ax.transAxes,
        fontsize=7.2,
        color=GRAY,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65)
    )

    add_panel_label(ax, label)


def plot_r2_bar(ax, metrics, stage_order, label):
    data = metrics.copy()
    data["Stage"] = pd.Categorical(data["Stage"], categories=stage_order, ordered=True)
    data = data.sort_values("Stage")

    x = np.arange(len(data))
    vals = data["R2"].values
    colors = [STAGE_COLORS.get(str(s), TEAL) for s in data["Stage"]]

    ax.bar(
        x,
        vals,
        color=colors,
        edgecolor="none",
        alpha=0.88
    )

    ax.set_xticks(x)
    ax.set_xticklabels([short_stage_label(str(s)) for s in data["Stage"]], rotation=22, ha="right")

    ax.set_ylabel("$R^2$")
    ax.set_title("$R^2$ comparison", pad=4)
    ax.set_ylim(0, min(1.0, max(0.9, np.nanmax(vals) * 1.12)))
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.6)

    ymax = ax.get_ylim()[1]
    for xi, v in zip(x, vals):
        ax.text(
            xi,
            v + ymax * 0.020,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.8
        )

    add_panel_label(ax, label)


def plot_metric_bars(ax, metrics, stage_order, label):
    data = metrics.copy()
    data["Stage"] = pd.Categorical(data["Stage"], categories=stage_order, ordered=True)
    data = data.sort_values("Stage")

    x = np.arange(len(data))
    width = 0.34

    ax.bar(
        x - width / 2,
        data["RMSE"],
        width,
        color=DARK,
        edgecolor="none",
        label="RMSE"
    )

    ax.bar(
        x + width / 2,
        data["MAE"],
        width,
        color=MAGENTA,
        edgecolor="none",
        label="MAE"
    )

    ax.set_xticks(x)
    ax.set_xticklabels([short_stage_label(str(s)) for s in data["Stage"]], rotation=22, ha="right")

    ax.set_ylabel("Metric value")
    ax.set_xlabel("Model stage")
    ax.set_title("Prediction error comparison", pad=4)

    ymax = max(data["RMSE"].max(), data["MAE"].max()) * 1.24
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.6)

    for xi, rmse, mae in zip(x, data["RMSE"], data["MAE"]):
        ax.text(
            xi - width / 2,
            rmse + ymax * 0.020,
            f"{rmse:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5
        )
        ax.text(
            xi + width / 2,
            mae + ymax * 0.020,
            f"{mae:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5
        )

    ax.legend(frameon=False, loc="upper right")
    add_panel_label(ax, label)


def plot_weight_panel(ax, weights_df, label):
    if weights_df is None or weights_df.empty:
        ax.axis("off")
        add_panel_label(ax, label)
        ax.text(0.5, 0.5, "No valid weight data", ha="center", va="center")
        return

    data = weights_df.copy()
    data = data[data["mean_weight"] > 1e-6]
    data = data.sort_values("mean_weight", ascending=True)

    y = np.arange(len(data))

    ax.barh(
        y,
        data["mean_weight"],
        xerr=data["std_weight"].fillna(0),
        color=TEAL,
        alpha=0.86,
        edgecolor="none",
        capsize=3
    )

    ax.set_yticks(y)
    ax.set_yticklabels(data["Base model"])

    ax.set_xlabel("Normalized mean selected weight")
    ax.set_title("Final fusion weight stability", pad=4)
    ax.set_xlim(0, min(1.0, max(0.55, data["mean_weight"].max() * 1.35)))
    ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.6)

    for yi, v in zip(y, data["mean_weight"]):
        ax.text(
            v + 0.015,
            yi,
            f"{v:.2f}",
            va="center",
            fontsize=6.8
        )

    add_panel_label(ax, label)


# ============================================================
# 7. Main workflow
# ============================================================

def main():
    print("=" * 95)
    print("Generating corrected revised Figure 3 using fixed staged outputs ...")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Table dir    : {TABLE_DIR}")
    print(f"Figure dir   : {FIG_DIR}")
    print("=" * 95)

    pred_all = collect_stage_predictions()
    metrics = build_metrics(pred_all)
    weights_df = load_final_weights()

    stage_order = [
        "RF baseline",
        "RF-CatBoost weighted fusion",
        "Ridge stacking fusion",
        "Selected Weighted 4Model",
    ]

    print("[OK] Stage predictions loaded:")
    for s in stage_order:
        n = (pred_all["Stage"].astype(str) == s).sum()
        print(f"    - {s}: {n} rows")

    print("[OK] Stage metrics:")
    print(metrics[["Stage", "R2", "RMSE", "MAE", "n_points"]].to_string(index=False))

    metrics_out = TABLE_DIR / "22_est_figure3_revised_metrics_used.csv"
    metrics.to_csv(metrics_out, index=False, encoding="utf-8-sig")

    weights_out = TABLE_DIR / "22_est_figure3_revised_weight_summary.csv"
    if weights_df is not None and not weights_df.empty:
        weights_df.to_csv(weights_out, index=False, encoding="utf-8-sig")
        print("[OK] Final normalized fusion weights:")
        print(weights_df.to_string(index=False))
    else:
        print("[WARN] No valid final fusion weight data.")

    # --------------------------------------------------------
    # Figure layout
    # --------------------------------------------------------
    fig = plt.figure(figsize=(15.8, 9.2), constrained_layout=False)

    gs = GridSpec(
        2,
        4,
        figure=fig,
        left=0.055,
        right=0.985,
        top=0.885,
        bottom=0.225,
        wspace=0.34,
        hspace=0.42
    )

    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(4)]

    display_titles = {
        "RF baseline": "Baseline model\n(RF)",
        "RF-CatBoost weighted fusion": "Early weighted fusion\n(RF-CatBoost)",
        "Ridge stacking fusion": "Intermediate optimized model\n(Ridge stacking)",
        "Selected Weighted 4Model": "Final selected weighted 4Model",
    }

    # A-D
    for i, stage in enumerate(stage_order):
        plot_pred_obs(
            axes[i],
            pred_all=pred_all,
            stage=stage,
            label=chr(ord("A") + i),
            display_title=display_titles[stage]
        )

    # E
    plot_residual_panel(
        axes[4],
        pred_all=pred_all,
        label="E"
    )

    # F
    plot_r2_bar(
        axes[5],
        metrics=metrics,
        stage_order=stage_order,
        label="F"
    )

    # G
    plot_metric_bars(
        axes[6],
        metrics=metrics,
        stage_order=stage_order,
        label="G"
    )

    # H
    plot_weight_panel(
        axes[7],
        weights_df=weights_df,
        label="H"
    )

    fig.suptitle(
        "Workflow-oriented validation of machine-learning models for fluoride removal prediction",
        fontsize=14.0,
        fontweight="bold",
        y=0.965
    )

    caption = (
        "Figure 3. Workflow-oriented comparison of representative models for fluoride removal prediction. "
        "Panels A-D show the development pathway from the RF baseline, RF-CatBoost weighted fusion, "
        "Ridge stacking fusion, to the final selected weighted 4Model. "
        "Panel E presents residual diagnosis of the final model. "
        "Panels F and G compare R2, RMSE, and MAE across the selected development stages. "
        "Panel H shows normalized fusion weight stability of the final selected weighted-fusion model. "
        "Fusion weights were selected by minimizing validation-set RMSE; the test set was used only for final evaluation."
    )

    wrapped_caption = textwrap.fill(caption, width=176)

    fig.text(
        0.055,
        0.060,
        wrapped_caption,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color=GRAY,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="lightgray",
            linewidth=0.6,
            alpha=0.96
        )
    )

    png_path = FIG_DIR / "Figure3_revised_EST_workflow_comparison.png"
    pdf_path = FIG_DIR / "Figure3_revised_EST_workflow_comparison.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("=" * 95)
    print("[DONE] Corrected revised Figure 3 generated successfully.")
    print(f"PNG     : {png_path}")
    print(f"PDF     : {pdf_path}")
    print(f"METRICS : {metrics_out}")
    print(f"WEIGHTS : {weights_out}")
    print("=" * 95)


if __name__ == "__main__":
    main()