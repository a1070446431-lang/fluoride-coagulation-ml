# -*- coding: utf-8 -*-
"""
20_est_style_figure3.py

Final publication-quality EST-like Figure 3 for the fluoride coagulation ML project.

Final layout:
    A-D: Predicted vs observed plots for representative models
         RF Baseline, CatBoost, HGB, Selected Weighted 4Model

    E: Residual diagnosis of Selected Weighted 4Model
    F: R2 comparison
    G: RMSE / MAE comparison
    H: Normalized fusion weight stability

Key improvements:
    1. Avoid over-crowded overlay scatter panels.
    2. Use one model per predicted-vs-observed panel for clearer comparison.
    3. Normalize fusion weights so the mean contribution sums to 1.
    4. Use ACS/EST-like colors, red dashed 1:1 lines, compact panel labels.
    5. Use Figure 3-style caption at the bottom.

Outputs:
    outputs/figures/Figure3_EST_ACS_like_model_performance.png
    outputs/figures/Figure3_EST_ACS_like_model_performance.pdf
    outputs/tables/20_est_figure3_metrics_used.csv
    outputs/tables/20_est_figure3_weight_summary.csv
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
# 2. Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9.0,
    "axes.linewidth": 0.9,
    "axes.labelsize": 8.8,
    "axes.titlesize": 9.2,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.4,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

PURPLE = "#75539B"
TEAL = "#55AAA4"
BLUE = "#4C78A8"
GREEN = "#54A24B"
ORANGE = "#F58518"
RED = "#FF0000"
DARK = "#3D3D3D"
MAGENTA = "#F24BAA"
GRAY = "#666666"
LIGHT_GRID = "#E8E8E8"

MODEL_COLORS = {
    "RF Baseline": PURPLE,
    "CatBoost": "#8E6AB3",
    "HGB": "#4F9C74",
    "ExtraTrees": "#6E9FDC",
    "Selected Weighted 4Model": TEAL,
}


# ============================================================
# 3. Utilities
# ============================================================

def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Cannot read csv: {path}")


def clean_col_name(x) -> str:
    s = str(x).strip().lower()
    s = s.replace("²", "2")
    s = s.replace("％", "%")
    s = s.replace("−", "-")
    s = re.sub(r"[^a-z0-9_%]+", "_", s)
    return s.strip("_")


def pretty_model_name(name: str) -> str:
    s = str(name).strip()
    low = s.lower()
    norm = re.sub(r"[^a-z0-9]+", "_", low).strip("_")

    if "selected" in norm and "weighted" in norm:
        return "Selected Weighted 4Model"
    if "4model" in norm or "4_model" in norm:
        return "Selected Weighted 4Model"

    if norm in ["rf", "rf_baseline", "random_forest", "randomforest"]:
        return "RF Baseline"
    if "random" in norm and "forest" in norm:
        return "RF Baseline"
    if norm.startswith("rf_") or norm.endswith("_rf"):
        return "RF Baseline"

    if "catboost" in norm and ("tuned" in norm or "manual" in norm):
        return "CatBoost"
    if "catboost" in norm or norm == "cat":
        return "CatBoost"

    if "hgb" in norm or "hist_gradient" in norm or "histgradient" in norm:
        return "HGB"

    if "extra_trees" in norm or "extratrees" in norm or norm == "extra":
        return "ExtraTrees"

    if "rf" in norm and "cat" in norm and "weighted" in norm:
        return "RF-CatBoost Weighted"

    if "ridge" in norm and "stack" in norm:
        return "Ridge Stacking"
    if "linear" in norm and "stack" in norm:
        return "Linear Stacking"
    if "stack" in norm:
        return "Stacking"

    s = s.replace("_pred", "")
    s = s.replace("_prediction", "")
    s = s.replace("prediction_", "")
    s = s.replace("pred_", "")
    s = s.replace("_weight", "")
    s = s.replace("weight_", "")
    s = s.replace("_", " ")

    return s.strip().title()


def short_label(name: str) -> str:
    name = pretty_model_name(name)
    mapping = {
        "RF Baseline": "RF",
        "CatBoost": "CatBoost",
        "HGB": "HGB",
        "ExtraTrees": "ExtraTrees",
        "Selected Weighted 4Model": "Selected 4Model",
        "RF-CatBoost Weighted": "RF-CatBoost",
    }
    return mapping.get(name, name)


def detect_actual_column(df: pd.DataFrame):
    cleaned = {c: clean_col_name(c) for c in df.columns}

    exact_keys = [
        "y_true", "true", "actual", "observed", "experimental",
        "target", "y_test", "f_removal_true", "removal_true"
    ]

    for key in exact_keys:
        for c, cc in cleaned.items():
            if cc == key and pd.api.types.is_numeric_dtype(df[c]):
                return c

    contains_keys = ["y_true", "actual", "observed", "experimental", "target", "removal_true", "true"]

    for key in contains_keys:
        for c, cc in cleaned.items():
            if key in cc and pd.api.types.is_numeric_dtype(df[c]):
                return c

    return None


def detect_pred_columns(df: pd.DataFrame):
    pred_cols = []

    for c in df.columns:
        cc = clean_col_name(c)

        if any(k in cc for k in ["pred", "prediction", "predicted", "y_hat"]):
            if pd.api.types.is_numeric_dtype(df[c]):
                pred_cols.append(c)

    return pred_cols


def detect_model_column(df: pd.DataFrame):
    for c in df.columns:
        cc = clean_col_name(c)
        if cc in ["model", "model_name", "method", "algorithm", "learner", "base_model"]:
            return c
    return None


def detect_seed_column(df: pd.DataFrame):
    for c in df.columns:
        cc = clean_col_name(c)
        if cc in ["seed", "random_seed", "split_seed"]:
            return c
    return None


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


# ============================================================
# 4. Prediction data
# ============================================================

def parse_prediction_file(path: Path):
    try:
        df = read_csv_smart(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return []

    if df.empty:
        return []

    actual_col = detect_actual_column(df)
    pred_cols = detect_pred_columns(df)
    model_col = detect_model_column(df)
    seed_col = detect_seed_column(df)

    if actual_col is None or len(pred_cols) == 0:
        return []

    rows = []

    for pred_col in pred_cols:
        tmp = pd.DataFrame()
        tmp["y_true"] = pd.to_numeric(df[actual_col], errors="coerce")
        tmp["y_pred"] = pd.to_numeric(df[pred_col], errors="coerce")

        if model_col is not None and len(pred_cols) == 1:
            tmp["model"] = df[model_col].astype(str).map(pretty_model_name)
        else:
            tmp["model"] = pretty_model_name(pred_col)

        if seed_col is not None:
            tmp["seed"] = df[seed_col]
        else:
            tmp["seed"] = np.nan

        tmp["source_file"] = path.name
        tmp = tmp.dropna(subset=["y_true", "y_pred"])

        if len(tmp) >= 10:
            rows.append(tmp)

    return rows


def collect_predictions():
    csv_files = sorted(TABLE_DIR.glob("*.csv"))

    pred_files = []
    for p in csv_files:
        low = p.name.lower()
        if any(k in low for k in ["pred", "prediction", "predicted"]):
            pred_files.append(p)

    if not pred_files:
        raise FileNotFoundError("No prediction csv files found in outputs/tables.")

    preferred = [
        p for p in pred_files
        if "19_selected" in p.name.lower() and any(k in p.name.lower() for k in ["pred", "prediction"])
    ]

    files_to_read = preferred if preferred else pred_files

    print("[INFO] Prediction files used:")
    for p in files_to_read:
        print(f"    - {p.name}")

    all_rows = []
    for p in files_to_read:
        all_rows.extend(parse_prediction_file(p))

    if not all_rows:
        raise FileNotFoundError("No usable prediction table was found.")

    pred_all = pd.concat(all_rows, ignore_index=True)
    pred_all["model"] = pred_all["model"].map(pretty_model_name)

    pred_all = pred_all.drop_duplicates(
        subset=["model", "seed", "y_true", "y_pred", "source_file"]
    )

    return pred_all


def build_metrics(pred_all: pd.DataFrame):
    rows = []

    for m, g in pred_all.groupby("model"):
        r2, rmse, mae, mse = metric_values(g["y_true"], g["y_pred"])
        rows.append({
            "model": m,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MSE": mse,
            "n_points": len(g),
        })

    return pd.DataFrame(rows).sort_values("R2", ascending=False)


# ============================================================
# 5. Fusion weights
# ============================================================

def clean_weight_model_name(col_name: str) -> str:
    s = str(col_name)
    s = re.sub(r"(?i)^weight[_\s-]*", "", s)
    s = re.sub(r"(?i)[_\s-]*weight$", "", s)
    s = re.sub(r"(?i)^w[_\s-]*", "", s)
    s = s.strip("_ -")
    return pretty_model_name(s)


def normalize_weight_summary(out: pd.DataFrame):
    """
    Final safety step:
    normalize mean weights so that their sum is 1.
    """
    out = out.copy()
    out = out.dropna(subset=["mean_weight"])

    out["mean_weight"] = pd.to_numeric(out["mean_weight"], errors="coerce")
    out["std_weight"] = pd.to_numeric(out["std_weight"], errors="coerce").fillna(0)

    out = out.dropna(subset=["mean_weight"])
    out = out[out["mean_weight"] >= 0]

    total = out["mean_weight"].sum()

    if total > 0:
        out["mean_weight"] = out["mean_weight"] / total
        out["std_weight"] = out["std_weight"] / total

    out = out.sort_values("mean_weight", ascending=False)

    return out


def collect_weights():
    weight_files = sorted(TABLE_DIR.glob("*weight*.csv"))

    if not weight_files:
        print("[WARN] No weight file found.")
        return None

    preferred = None
    for p in weight_files:
        if "19_selected" in p.name.lower():
            preferred = p
            break

    path = preferred if preferred is not None else weight_files[0]

    print(f"[INFO] Reading weight file: {path.name}")

    try:
        df = read_csv_smart(path)
    except Exception as e:
        print(f"[WARN] Failed to read weight file: {e}")
        return None

    if df.empty:
        return None

    cols_clean = {c: clean_col_name(c) for c in df.columns}

    model_col = None
    weight_col = None
    seed_col = None

    for c, cc in cols_clean.items():
        if cc in ["model", "base_model", "learner", "algorithm", "method"]:
            model_col = c
        if cc in ["weight", "mean_weight", "selected_weight", "fusion_weight"]:
            weight_col = c
        if cc in ["seed", "random_seed", "split_seed"]:
            seed_col = c

    # Long format: seed | base_model | weight
    if model_col is not None and weight_col is not None:
        tmp = pd.DataFrame({
            "base_model": df[model_col].astype(str).map(pretty_model_name),
            "weight": pd.to_numeric(df[weight_col], errors="coerce")
        })

        if seed_col is not None:
            tmp["seed"] = df[seed_col].values
        else:
            tmp["seed"] = 0

        tmp = tmp.dropna(subset=["weight"])
        tmp = tmp[tmp["weight"] >= 0]

        # normalize within each seed
        tmp["weight_norm"] = np.nan
        for seed, g in tmp.groupby("seed"):
            s = g["weight"].sum()
            if s > 0:
                tmp.loc[g.index, "weight_norm"] = g["weight"] / s

        tmp = tmp.dropna(subset=["weight_norm"])

        if len(tmp) > 0:
            out = (
                tmp
                .groupby("base_model", as_index=False)
                .agg(
                    mean_weight=("weight_norm", "mean"),
                    std_weight=("weight_norm", "std"),
                    n=("weight_norm", "count")
                )
            )

            out["std_weight"] = out["std_weight"].fillna(0)
            out = normalize_weight_summary(out)

            print("[OK] Weight table parsed as long format and normalized.")
            return out

    # Wide format: seed | RF | CatBoost | HGB | ExtraTrees
    skip_keywords = [
        "seed", "random_seed", "split_seed",
        "r2", "rmse", "mae", "mse",
        "k", "feature", "rank",
        "n_feature", "num_feature",
        "best", "score", "metric",
        "fold", "repeat", "trial", "error", "loss"
    ]

    allowed_models = {"RF Baseline", "CatBoost", "HGB", "ExtraTrees"}

    numeric_cols = []

    for c in df.columns:
        cc = clean_col_name(c)

        if any(k in cc for k in skip_keywords):
            continue

        model_name = clean_weight_model_name(c)
        if model_name not in allowed_models:
            continue

        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() == 0:
            continue

        vmin = vals.min()
        vmax = vals.max()

        if vmin >= -0.05 and vmax <= 1.05:
            numeric_cols.append(c)

    if not numeric_cols:
        print("[WARN] Could not identify weight columns.")
        return None

    wide = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    wide[wide < 0] = 0

    row_sum = wide.sum(axis=1)
    valid_rows = row_sum > 0

    wide_norm = wide.loc[valid_rows].div(row_sum.loc[valid_rows], axis=0)

    rows = []
    for c in numeric_cols:
        vals = wide_norm[c].dropna()
        if len(vals) == 0:
            continue

        rows.append({
            "base_model": clean_weight_model_name(c),
            "mean_weight": vals.mean(),
            "std_weight": vals.std(ddof=1) if len(vals) > 1 else 0,
            "n": len(vals),
        })

    if not rows:
        return None

    out = pd.DataFrame(rows)
    out = normalize_weight_summary(out)

    print("[OK] Weight table parsed as wide format and normalized.")
    return out


# ============================================================
# 6. Plot helpers
# ============================================================

def sample_for_plot(df: pd.DataFrame, n=700, random_state=42):
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=random_state).copy()


def add_panel_label(ax, label):
    ax.text(
        0.03,
        0.96,
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
        alpha=0.9
    )

    ax.add_patch(ellipse)


def get_model_data(pred_all: pd.DataFrame, model: str):
    return pred_all[pred_all["model"] == model].copy()


def plot_pred_obs(ax, pred_all, model, label):
    g = get_model_data(pred_all, model)
    g_s = sample_for_plot(g, n=700, random_state=100 + ord(label))

    color = MODEL_COLORS.get(model, TEAL)

    r2, rmse, mae, mse = metric_values(g["y_true"], g["y_pred"])

    ax.scatter(
        g_s["y_true"],
        g_s["y_pred"],
        s=14,
        color=color,
        alpha=0.70,
        edgecolor="none"
    )

    ax.plot(
        [0, 100],
        [0, 100],
        color=RED,
        linestyle=(0, (4, 3)),
        linewidth=1.0
    )

    add_cov_ellipse(
        ax,
        g_s["y_true"],
        g_s["y_pred"],
        color=color,
        n_std=1.10
    )

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)

    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Predicted F removal (%)")
    ax.set_title(model, pad=5)

    ax.grid(False)

    ax.text(
        0.58,
        0.08,
        f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.75
        )
    )

    add_panel_label(ax, label)


def plot_residual_panel(ax, pred_all, final_model, label):
    g = get_model_data(pred_all, final_model)
    g_s = sample_for_plot(g, n=800, random_state=88)

    residual = g_s["y_true"] - g_s["y_pred"]
    residual_all = g["y_true"] - g["y_pred"]

    rmse_res = np.sqrt(np.mean(residual_all ** 2))

    ax.scatter(
        g_s["y_pred"],
        residual,
        s=14,
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

    ax.axhspan(-rmse_res, rmse_res, color=TEAL, alpha=0.11, linewidth=0)

    ax.set_xlim(0, 105)
    ax.set_xlabel("Predicted F removal (%)")
    ax.set_ylabel("Residual (%)")
    ax.set_title("Residual diagnosis", pad=5)
    ax.grid(False)

    ax.text(
        0.05,
        0.08,
        f"RMSE band: ±{rmse_res:.2f}%",
        transform=ax.transAxes,
        fontsize=7.4,
        color=GRAY
    )

    add_panel_label(ax, label)


def plot_r2_bar(ax, metrics, selected_models, label):
    data = metrics[metrics["model"].isin(selected_models)].copy()
    data["model"] = pd.Categorical(data["model"], categories=selected_models, ordered=True)
    data = data.sort_values("model")

    x = np.arange(len(data))
    vals = data["R2"].values

    colors = [MODEL_COLORS.get(m, BLUE) for m in data["model"]]

    ax.bar(
        x,
        vals,
        color=colors,
        edgecolor="none",
        alpha=0.88
    )

    ax.set_xticks(x)
    ax.set_xticklabels([short_label(m) for m in data["model"]], rotation=25, ha="right")
    ax.set_ylabel("$R^2$")
    ax.set_title("$R^2$ comparison", pad=5)
    ax.set_ylim(0, min(1.0, max(0.9, np.nanmax(vals) * 1.12)))
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.6)

    ymax = ax.get_ylim()[1]
    for xi, v in zip(x, vals):
        ax.text(
            xi,
            v + ymax * 0.025,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.0
        )

    add_panel_label(ax, label)


def plot_metric_bars(ax, metrics, selected_models, label):
    data = metrics[metrics["model"].isin(selected_models)].copy()
    data["model"] = pd.Categorical(data["model"], categories=selected_models, ordered=True)
    data = data.sort_values("model")

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
    ax.set_xticklabels([short_label(m) for m in data["model"]], rotation=25, ha="right")

    ax.set_ylabel("Metric value")
    ax.set_xlabel("Model")
    ax.set_title("Prediction error comparison", pad=5)

    ymax = max(data["RMSE"].max(), data["MAE"].max()) * 1.25
    ax.set_ylim(0, ymax)

    for xi, rmse, mae in zip(x, data["RMSE"], data["MAE"]):
        ax.text(
            xi - width / 2,
            rmse + ymax * 0.025,
            f"{rmse:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.8
        )

        ax.text(
            xi + width / 2,
            mae + ymax * 0.025,
            f"{mae:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.8
        )

    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.6)

    add_panel_label(ax, label)


def plot_weight_panel(ax, weights_df, label):
    data = weights_df.copy()
    data = data[data["mean_weight"] > 1e-6]
    data = data.sort_values("mean_weight", ascending=True)

    if len(data) == 0:
        ax.axis("off")
        add_panel_label(ax, label)
        ax.text(0.5, 0.5, "No valid weight data", ha="center", va="center")
        return

    y = np.arange(len(data))

    ax.barh(
        y,
        data["mean_weight"],
        xerr=data["std_weight"].fillna(0),
        color=TEAL,
        alpha=0.85,
        edgecolor="none",
        capsize=3
    )

    ax.set_yticks(y)
    ax.set_yticklabels([short_label(m) for m in data["base_model"]])

    ax.set_xlabel("Normalized mean selected weight")
    ax.set_title("Fusion weight stability", pad=5)
    ax.set_xlim(0, min(1.0, max(0.55, data["mean_weight"].max() * 1.35)))

    for yi, v in zip(y, data["mean_weight"]):
        ax.text(
            v + 0.015,
            yi,
            f"{v:.2f}",
            va="center",
            fontsize=7.2
        )

    ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.6)

    add_panel_label(ax, label)


# ============================================================
# 7. Main
# ============================================================

def main():
    print("=" * 90)
    print("Generating FINAL EST-like Figure 3 ...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Table dir   : {TABLE_DIR}")
    print(f"Figure dir  : {FIG_DIR}")
    print("=" * 90)

    pred_all = collect_predictions()
    metrics = build_metrics(pred_all)

    print("[OK] Models found:")
    for m in sorted(pred_all["model"].unique()):
        print(f"    - {m}")

    print("[OK] Metrics:")
    print(metrics[["model", "R2", "RMSE", "MAE", "n_points"]].to_string(index=False))

    final_model = "Selected Weighted 4Model"
    if final_model not in pred_all["model"].unique():
        final_model = metrics.sort_values("R2", ascending=False).iloc[0]["model"]

    preferred_panels = ["RF Baseline", "CatBoost", "HGB", final_model]
    available = list(pred_all["model"].unique())

    panel_models = []
    for m in preferred_panels:
        if m in available and m not in panel_models:
            panel_models.append(m)

    # If one of the preferred models is absent, fill by best R2
    for m in metrics.sort_values("R2", ascending=False)["model"]:
        if m in available and m not in panel_models:
            panel_models.append(m)
        if len(panel_models) >= 4:
            break

    panel_models = panel_models[:4]

    # Ensure final model appears
    if final_model not in panel_models:
        panel_models[-1] = final_model

    selected_models_for_metric = panel_models.copy()

    metrics_out = TABLE_DIR / "20_est_figure3_metrics_used.csv"
    metrics[metrics["model"].isin(selected_models_for_metric)].to_csv(
        metrics_out,
        index=False,
        encoding="utf-8-sig"
    )

    weights_df = collect_weights()
    if weights_df is not None:
        weights_out = TABLE_DIR / "20_est_figure3_weight_summary.csv"
        weights_df.to_csv(weights_out, index=False, encoding="utf-8-sig")
        print(f"[OK] Normalized weight summary saved to: {weights_out}")
        print(weights_df.to_string(index=False))
    else:
        print("[WARN] No valid weight summary. Panel H will show empty placeholder.")

    print("[OK] Panel models:")
    for m in panel_models:
        print(f"    - {m}")

    # --------------------------------------------------------
    # Figure layout
    # --------------------------------------------------------
    fig = plt.figure(figsize=(15.8, 8.8), constrained_layout=False)

    gs = GridSpec(
        2,
        4,
        figure=fig,
        left=0.055,
        right=0.985,
        top=0.890,
        bottom=0.130,
        wspace=0.34,
        hspace=0.42
    )

    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(4)]

    # A-D: predicted vs observed
    for i, m in enumerate(panel_models):
        plot_pred_obs(
            axes[i],
            pred_all=pred_all,
            model=m,
            label=chr(ord("A") + i)
        )

    # E: residual
    plot_residual_panel(
        axes[4],
        pred_all=pred_all,
        final_model=final_model,
        label="E"
    )

    # F: R2
    plot_r2_bar(
        axes[5],
        metrics=metrics,
        selected_models=selected_models_for_metric,
        label="F"
    )

    # G: RMSE / MAE
    plot_metric_bars(
        axes[6],
        metrics=metrics,
        selected_models=selected_models_for_metric,
        label="G"
    )

    # H: normalized weights
    if weights_df is not None and not weights_df.empty:
        plot_weight_panel(
            axes[7],
            weights_df=weights_df,
            label="H"
        )
    else:
        axes[7].axis("off")
        add_panel_label(axes[7], "H")
        axes[7].text(0.5, 0.5, "No valid weight data", ha="center", va="center")

    fig.suptitle(
        "Integrated validation of machine-learning models for fluoride removal prediction",
        fontsize=14.2,
        fontweight="bold",
        y=0.965
    )

    caption = (
        "Figure 3. Integrated validation of representative machine-learning models for fluoride removal prediction. "
        "Panels A-D show predicted versus observed fluoride removal for RF, CatBoost, HGB, and the selected weighted fusion model. "
        "Panel E shows residual diagnosis of the selected model. Panels F and G compare R2, RMSE, and MAE. "
        "Panel H shows normalized mean fusion weights selected by validation-set RMSE minimization; the test set was used only for final evaluation."
    )

    wrapped_caption = textwrap.fill(caption, width=180)

    fig.text(
        0.055,
        0.050,
        wrapped_caption,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color=GRAY
    )

    png_path = FIG_DIR / "Figure3_EST_ACS_like_model_performance.png"
    pdf_path = FIG_DIR / "Figure3_EST_ACS_like_model_performance.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("=" * 90)
    print("[DONE] FINAL EST-like Figure 3 generated successfully.")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"METRICS: {metrics_out}")
    print("=" * 90)


if __name__ == "__main__":
    main()