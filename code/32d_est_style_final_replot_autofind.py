# -*- coding: utf-8 -*-
"""
32d_est_style_final_replot_autofind.py

Purpose
-------
Replot Step 32D figures in an EST-like publication style.

Key fixes
---------
1. Does NOT require outputs/tables/32d_policy_summary.csv.
2. Automatically scans outputs/tables for 32D/32C2/32B CSV files.
3. Automatically detects policy metric tables containing R2/RMSE/MAE/Coverage.
4. Automatically detects prediction tables containing Observed/Predicted columns.
5. Does NOT retrain models and does NOT change results.
6. Saves PNG and PDF figures.
7. Prints total runtime / elapsed time at the end.

Usage
-----
Put this file into your project src folder and run it directly in PyCharm.
"""

from __future__ import annotations

import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, Path.cwd(), Path.cwd().parent]
    for c in candidates:
        if (c / "outputs" / "tables").exists() and (c / "outputs" / "figures").exists():
            return c
    return here.parent.parent


ROOT = find_project_root()
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# EST-like style
# =========================================================
def set_est_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.95,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "grid.color": "#D6D6D6",
        "grid.linestyle": "-",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.62,
        "figure.dpi": 180,
        "savefig.dpi": 600,
    })


set_est_style()

C_BLUE = "#4E79A7"
C_BLUE_LIGHT = "#88AFCB"
C_GREEN = "#6FA77A"
C_ORANGE = "#D9A057"
C_RED = "#C94F4F"
C_GRAY = "#BFBFBF"
C_DARK = "#333333"


def log(msg: str) -> None:
    print(msg, flush=True)


def clean_name(x: object) -> str:
    return str(x).strip().replace("\n", " ").replace("\r", " ")


def norm_col(c: object) -> str:
    s = str(c).strip().lower()
    s = re.sub(r"[\s\-\(\)\[\]{}:;]+", "_", s)
    s = s.replace("__", "_")
    return s


def read_csv_robust(path: Path) -> pd.DataFrame | None:
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    return None


def numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def find_col(df: pd.DataFrame, patterns: list[str], exclude: list[str] | None = None) -> str | None:
    exclude = exclude or []
    cols = list(df.columns)
    ncols = {c: norm_col(c) for c in cols}

    for pat in patterns:
        pat_l = pat.lower()
        for c in cols:
            if ncols[c] == pat_l:
                return c

    for pat in patterns:
        regex = re.compile(pat, flags=re.I)
        for c in cols:
            cn = ncols[c]
            if any(re.search(ex, cn, flags=re.I) for ex in exclude):
                continue
            if regex.search(str(c)) or regex.search(cn):
                return c
    return None


def find_metric_cols(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "policy": find_col(df, [r"^policy_id$", r"policy.*id", r"rule.*id", r"candidate.*id", r"^id$"]),
        "method": find_col(df, [r"^method$", r"model", r"algorithm", r"learner"]),
        "strategy": find_col(df, [r"strategy", r"validation", r"split_strategy"]),
        "rule": find_col(df, [r"rule", r"policy_rule", r"filter", r"condition"]),
        "r2": find_col(df, [r"retained.*r2", r"within.*r2", r"domain.*r2", r"^r2$", r"r_?2", r"R²"]),
        "rmse": find_col(df, [r"retained.*rmse", r"within.*rmse", r"domain.*rmse", r"^rmse$", r"pooled.*rmse"]),
        "mae": find_col(df, [r"retained.*mae", r"within.*mae", r"domain.*mae", r"^mae$", r"pooled.*mae"]),
        "coverage": find_col(df, [r"coverage", r"retained.*fraction", r"coverage.*retained", r"frac.*retained"]),
        "n": find_col(df, [r"n_retained", r"retained.*n", r"n.*retained", r"sample.*size", r"^n$"]),
    }


def file_priority(path: Path) -> int:
    name = path.name.lower()
    score = 0
    if name.startswith("32d"):
        score += 1000
    if "32d" in name:
        score += 500
    if "32c2" in name:
        score += 250
    if "summary" in name:
        score += 90
    if "policy" in name:
        score += 90
    if "metric" in name:
        score += 70
    if "candidate" in name:
        score += 30
    if "prediction" in name or "pred" in name:
        score -= 50
    return score


def detect_policy_summary() -> tuple[pd.DataFrame, dict[str, str | None], Path]:
    csvs = sorted(TABLE_DIR.glob("*.csv"), key=file_priority, reverse=True)
    tried = []
    best = None
    best_cols = None
    best_path = None
    best_score = -10**9

    for p in csvs:
        name = p.name.lower()
        if not any(k in name for k in ["32d", "32c2", "32c", "32b"]):
            continue
        df = read_csv_robust(p)
        if df is None or df.empty:
            continue
        cols = find_metric_cols(df)
        if not all([cols["r2"], cols["rmse"], cols["mae"], cols["coverage"]]):
            tried.append((p.name, "missing R2/RMSE/MAE/Coverage columns"))
            continue
        tmp = df.copy()
        for key in ["r2", "rmse", "mae", "coverage"]:
            tmp[cols[key]] = numeric_series(tmp[cols[key]])
        valid_rows = tmp[[cols["r2"], cols["rmse"], cols["mae"], cols["coverage"]]].notna().all(axis=1).sum()
        if valid_rows == 0:
            tried.append((p.name, "no valid numeric metric rows"))
            continue
        score = file_priority(p) + valid_rows
        if cols["policy"]:
            score += 50
        if cols["method"]:
            score += 30
        if cols["rule"]:
            score += 20
        if score > best_score:
            best = tmp
            best_cols = cols
            best_path = p
            best_score = score

    if best is None:
        msg = [
            "Could not auto-detect a 32D policy/summary CSV.",
            f"Scanned folder: {TABLE_DIR}",
            "Please check whether Step 32D generated CSV tables containing R2, RMSE, MAE and Coverage.",
            "Files scanned / reasons:",
        ]
        msg.extend([f"- {a}: {b}" for a, b in tried[:120]])
        raise FileNotFoundError("\n".join(msg))

    log(f"[INFO] Detected policy summary table: {best_path.name} | shape={best.shape}")
    return best, best_cols, best_path


def detect_prediction_table() -> tuple[pd.DataFrame | None, dict[str, str | None], Path | None]:
    csvs = sorted(TABLE_DIR.glob("*.csv"), key=file_priority, reverse=True)
    best_df = None
    best_path = None
    best_score = -10**9
    best_cols = {}

    for p in csvs:
        name = p.name.lower()
        if not any(k in name for k in ["32d", "32c2", "32c", "32b"]):
            continue
        if not any(k in name for k in ["pred", "prediction", "retained", "zone", "subdomain"]):
            continue
        df = read_csv_robust(p)
        if df is None or df.empty:
            continue
        cols = {
            "obs": find_col(df, [r"^observed$", r"observed", r"actual", r"y_true", r"experimental"]),
            "pred": find_col(df, [r"^predicted$", r"predicted", r"prediction", r"y_pred"]),
            "policy": find_col(df, [r"policy.*id", r"rule.*id", r"candidate.*id", r"^policy$"]),
            "method": find_col(df, [r"^method$", r"model", r"algorithm", r"learner"]),
            "rule": find_col(df, [r"rule", r"policy_rule", r"filter", r"condition"]),
            "strategy": find_col(df, [r"strategy", r"validation", r"split_strategy"]),
            "split": find_col(df, [r"split", r"seed", r"fold", r"seed_or_fold"]),
            "zone": find_col(df, [r"reliability.*zone", r"zone", r"domain"]),
        }
        if cols["obs"] is None or cols["pred"] is None:
            continue
        tmp = df.copy()
        tmp[cols["obs"]] = numeric_series(tmp[cols["obs"]])
        tmp[cols["pred"]] = numeric_series(tmp[cols["pred"]])
        valid_n = tmp[[cols["obs"], cols["pred"]]].notna().all(axis=1).sum()
        if valid_n < 2:
            continue
        score = file_priority(p) + valid_n
        if name.startswith("32d"):
            score += 300
        if "retained" in name or "best" in name:
            score += 200
        if score > best_score:
            best_score = score
            best_df = tmp
            best_path = p
            best_cols = cols

    if best_df is None:
        log("[WARN] No prediction table detected. Prediction panel will be skipped.")
        return None, {}, None
    log(f"[INFO] Detected prediction table: {best_path.name} | shape={best_df.shape}")
    return best_df, best_cols, best_path


def filter_best_predictions(pred_df: pd.DataFrame, pred_cols: dict[str, str | None], best_row: pd.Series, pc: dict[str, str | None]) -> pd.DataFrame:
    df = pred_df.copy()
    if pred_cols.get("policy") and pc.get("policy"):
        target = clean_name(best_row[pc["policy"]])
        tmp = df[df[pred_cols["policy"]].astype(str).map(clean_name) == target]
        if len(tmp) >= 2:
            return tmp.copy()
    if pred_cols.get("method") and pc.get("method"):
        target_method = clean_name(best_row[pc["method"]])
        tmp = df[df[pred_cols["method"]].astype(str).map(clean_name) == target_method]
        if pred_cols.get("rule") and pc.get("rule"):
            target_rule = clean_name(best_row[pc["rule"]])
            tmp2 = tmp[tmp[pred_cols["rule"]].astype(str).map(clean_name) == target_rule]
            if len(tmp2) >= 2:
                return tmp2.copy()
        if len(tmp) >= 2:
            return tmp.copy()
    log("[WARN] Could not exactly filter prediction table to best policy. Using whole detected prediction table as fallback.")
    return df.copy()


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).copy()
    p = np.asarray(y_pred, dtype=float).copy()
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    n = len(y)
    if n == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "N": 0}
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if n < 2 or sst <= 1e-12 else float(1.0 - np.sum((y - p) ** 2) / sst)
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "N": n}


def savefig(fig: plt.Figure, name: str) -> None:
    png = FIG_DIR / name
    pdf = FIG_DIR / name.replace(".png", ".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] {png}")
    log(f"[FIG] {pdf}")


def style_axis(ax, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y")
    if xgrid:
        ax.grid(axis="x")
    ax.tick_params(axis="both", length=4, width=0.85, color=C_DARK)
    for sp in ax.spines.values():
        sp.set_linewidth(0.95)
        sp.set_color(C_DARK)


def add_bar_labels(ax, bars, fmt: str = "{:.2f}", pad_frac: float = 0.015, fs: float = 8.5) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    for b in bars:
        h = float(b.get_height())
        if not np.isfinite(h):
            continue
        va = "bottom"
        y = h + pad_frac * span
        if h < 0:
            va = "top"
            y = h - pad_frac * span
        ax.text(b.get_x() + b.get_width() / 2, y, fmt.format(h), ha="center", va=va, fontsize=fs, color="black", clip_on=False)


def short_method(s: object) -> str:
    s = clean_name(s)
    mapping = {
        "KRR_CatBoost_Weighted_2Model": "KRR-CB",
        "KRR_XGBoost_Weighted_2Model": "KRR-XGB",
        "KRR_Poly": "KRR",
        "XGBoost": "XGB",
        "CatBoost": "CB",
        "CB-XGB": "CB-XGB",
        "GPR_Matern": "GPR",
        "PLS": "PLS",
    }
    return mapping.get(s, s[:12])


def short_strategy(s: object) -> str:
    s = clean_name(s)
    mapping = {
        "Balanced_5Seed_Current31F": "balanced5",
        "RepeatedBalanced_Extended": "rep-balanced",
        "RepeatedOrdinary_GroupedSplit": "rep-grouped",
        "GroupKFold_FullyGrouped": "GroupKFold",
        "LOLO_LeaveOneLiteratureOut": "LOLO",
        "seed_21": "seed21",
        "seed_42": "seed42",
        "seed_66": "seed66",
        "seed_7": "seed7",
        "seed_88": "seed88",
    }
    return mapping.get(s, s[:14])


# =========================================================
# Load policy summary
# =========================================================
policy_df, pc, policy_path = detect_policy_summary()
for key in ["r2", "rmse", "mae", "coverage", "n"]:
    if pc.get(key):
        policy_df[pc[key]] = numeric_series(policy_df[pc[key]])
policy_df = policy_df.dropna(subset=[pc["r2"], pc["rmse"], pc["mae"], pc["coverage"]]).copy()
policy_df = policy_df.sort_values(pc["r2"], ascending=False).reset_index(drop=True)
if pc["policy"] is None:
    policy_df["Policy_ID_Auto"] = [f"P{i+1:02d}" for i in range(len(policy_df))]
    pc["policy"] = "Policy_ID_Auto"

top_n = min(12, len(policy_df))
top_df = policy_df.head(top_n).copy()
top_df["Display_Policy_ID"] = [f"P{i+1:02d}" for i in range(len(top_df))]
best_row = top_df.iloc[0].copy()
best_policy_display = best_row["Display_Policy_ID"]
best_method = best_row[pc["method"]] if pc.get("method") else "Best model"
best_rule = best_row[pc["rule"]] if pc.get("rule") else "N/A"

log(f"[INFO] Best detected policy: {best_policy_display} | method={best_method} | R2={best_row[pc['r2']]:.3f} | coverage={best_row[pc['coverage']]:.3f}")

mapping_cols = ["Display_Policy_ID", pc["policy"], pc["r2"], pc["rmse"], pc["mae"], pc["coverage"]]
for c in [pc.get("method"), pc.get("strategy"), pc.get("rule")]:
    if c and c not in mapping_cols:
        mapping_cols.append(c)
mapping_cols = [c for c in mapping_cols if c is not None and c in top_df.columns]
top_df[mapping_cols].to_csv(TABLE_DIR / "32d_est_replot_policy_id_mapping.csv", index=False, encoding="utf-8-sig")
log(f"[TABLE] {TABLE_DIR / '32d_est_replot_policy_id_mapping.csv'}")


# =========================================================
# Figure 165: policy metric panel
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.7))
axes = axes.ravel()
x = np.arange(len(top_df))
ids = top_df["Display_Policy_ID"].tolist()
colors = [C_GREEN if i == 0 else C_BLUE for i in range(len(top_df))]

panel_specs = [
    ("r2", "Retained $R^2$", "$R^2$", "{:.3f}"),
    ("rmse", "Retained RMSE", "RMSE", "{:.2f}"),
    ("mae", "Retained MAE", "MAE", "{:.2f}"),
    ("coverage", "Coverage", "Coverage", "{:.3f}"),
]

for ax, (key, title, ylabel, fmt) in zip(axes, panel_specs):
    col = pc[key]
    vals = top_df[col].to_numpy(dtype=float)
    bars = ax.bar(x, vals, color=colors, edgecolor="#3F3F3F", linewidth=0.75, zorder=3)
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Policy ID")
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    if key == "r2":
        y_min = min(-0.05, float(np.nanmin(vals)) - 0.06)
        y_max = max(0.8, float(np.nanmax(vals)) + 0.08)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0.5, color=C_RED, linestyle="--", linewidth=1.2, label="$R^2=0.5$")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#B0B0B0")
    elif key == "coverage":
        ax.set_ylim(0, max(0.15, float(np.nanmax(vals)) * 1.32))
    else:
        ax.set_ylim(0, max(float(np.nanmax(vals)) * 1.23, 1.0))
    style_axis(ax)
    add_bar_labels(ax, bars, fmt=fmt, fs=8.5)

fig.suptitle("32D target-free selective AD subdomain policy metrics", fontsize=15.5, fontweight="bold", y=0.985)
fig.subplots_adjust(wspace=0.22, hspace=0.34, top=0.90, bottom=0.10)
savefig(fig, "165_32d_est_replot_policy_metrics.png")


# =========================================================
# Figure 166: R2-coverage frontier
# =========================================================
fig, ax = plt.subplots(figsize=(9.8, 6.9))
front_df = policy_df.copy().dropna(subset=[pc["r2"], pc["coverage"]])
front_df[pc["r2"]] = numeric_series(front_df[pc["r2"]])
front_df[pc["coverage"]] = numeric_series(front_df[pc["coverage"]])
front_df = front_df.dropna(subset=[pc["r2"], pc["coverage"]]).copy()

y_lower = max(-0.30, float(np.nanpercentile(front_df[pc["r2"]], 2)) - 0.08)
y_upper = max(0.78, min(0.95, float(np.nanmax(front_df[pc["r2"]]) + 0.08)))
display_y = front_df[pc["r2"]].clip(lower=y_lower, upper=y_upper)

ax.scatter(front_df[pc["coverage"]], display_y, s=28, color="#C7C7C7", alpha=0.45, edgecolor="none", label="Other candidates", zorder=1)
main_df = front_df[front_df[pc["r2"]] >= 0.5].sort_values(pc["r2"], ascending=False).copy()
if not main_df.empty:
    ax.scatter(main_df[pc["coverage"]], main_df[pc["r2"]].clip(lower=y_lower, upper=y_upper), s=64, color=C_BLUE, alpha=0.93, edgecolor="white", linewidth=0.8, label="Main candidates", zorder=4)
    label_df = main_df.head(5).copy()
    offsets = [(9, 10), (9, -17), (11, 16), (12, -13), (14, 7)]
    for i, (_, row) in enumerate(label_df.iterrows()):
        dx, dy = offsets[i % len(offsets)]
        method_txt = short_method(row[pc["method"]]) if pc.get("method") else "Model"
        label_txt = f"{method_txt}\n$R^2$={row[pc['r2']]:.3f}"
        ax.annotate(label_txt, xy=(row[pc["coverage"]], min(max(row[pc["r2"]], y_lower), y_upper)), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, ha="left", va="center", bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.76), arrowprops=dict(arrowstyle="-", lw=0.6, color="#8A8A8A"), zorder=5)

ax.axhline(0.5, color=C_RED, linestyle="--", linewidth=1.2, label="$R^2=0.5$")
ax.axvline(0.05, color="#666666", linestyle=":", linewidth=1.2, label="Minimum coverage")
x_max = float(np.nanmax(front_df[pc["coverage"]]))
ax.set_xlim(0, max(0.18, min(1.02, x_max + 0.06)))
ax.set_ylim(y_lower, y_upper)
ax.set_xlabel("Coverage retained within strategy-method subset")
ax.set_ylabel("Retained-domain $R^2$")
ax.set_title("$R^2$-coverage frontier for target-free selective AD policies", pad=12)
style_axis(ax)
ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#B0B0B0")
savefig(fig, "166_32d_est_replot_r2_coverage_frontier.png")


# =========================================================
# Prediction panel
# =========================================================
pred_df, pred_cols, pred_path = detect_prediction_table()
best_pred = None
if pred_df is not None:
    best_pred = filter_best_predictions(pred_df, pred_cols, best_row, pc)
    best_pred[pred_cols["obs"]] = numeric_series(best_pred[pred_cols["obs"]])
    best_pred[pred_cols["pred"]] = numeric_series(best_pred[pred_cols["pred"]])
    best_pred = best_pred.dropna(subset=[pred_cols["obs"], pred_cols["pred"]]).copy()
    if len(best_pred) > 3000 and pred_cols.get("method") and pc.get("method"):
        tmp = best_pred[best_pred[pred_cols["method"]].astype(str).map(clean_name) == clean_name(best_method)].copy()
        if len(tmp) >= 2:
            best_pred = tmp

    y = best_pred[pred_cols["obs"]].to_numpy(float)
    p = best_pred[pred_cols["pred"]].to_numpy(float)
    m = metric_values(y, p)

    fig, ax = plt.subplots(figsize=(7.5, 7.1))
    ax.scatter(y, p, s=42, color=C_BLUE, alpha=0.78, edgecolor="white", linewidth=0.42, zorder=3)
    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1.45, color=C_RED, zorder=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Predicted F removal (%)")
    ax.set_title(f"Best retained subdomain: {short_method(best_method)}", fontsize=15, fontweight="bold", pad=12)
    info = f"$R^2$ = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nMAE = {m['MAE']:.2f}\nn = {int(m['N'])}\ncoverage = {float(best_row[pc['coverage']]):.3f}\npolicy = {best_policy_display}"
    ax.text(0.045, 0.955, info, transform=ax.transAxes, ha="left", va="top", fontsize=10.5, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#8F8F8F", lw=0.9, alpha=0.94))
    style_axis(ax)
    savefig(fig, "167_32d_est_replot_best_prediction_panel.png")
    best_pred.to_csv(TABLE_DIR / "32d_est_replot_best_policy_points.csv", index=False, encoding="utf-8-sig")
    log(f"[TABLE] {TABLE_DIR / '32d_est_replot_best_policy_points.csv'}")


# =========================================================
# Split-level stability
# =========================================================
split_df = None
split_cols = None
for pth in sorted(TABLE_DIR.glob("*.csv"), key=file_priority, reverse=True):
    name = pth.name.lower()
    if not any(k in name for k in ["32d", "32c2", "32c"]):
        continue
    if not any(k in name for k in ["split", "stability", "seed", "fold"]):
        continue
    df = read_csv_robust(pth)
    if df is None or df.empty:
        continue
    sc = {
        "split": find_col(df, [r"split", r"seed", r"fold", r"seed_or_fold"]),
        "r2": find_col(df, [r"^r2$", r"retained.*r2", r"within.*r2", r"r_?2"]),
        "rmse": find_col(df, [r"^rmse$", r"retained.*rmse", r"within.*rmse"]),
        "n": find_col(df, [r"n_retained", r"retained.*n", r"sample.*size", r"^n$"]),
    }
    if all(sc.values()):
        split_df = df.copy()
        split_cols = sc
        log(f"[INFO] Detected split-stability table: {pth.name} | shape={split_df.shape}")
        break

if split_df is None and best_pred is not None and pred_cols.get("split"):
    split_col = pred_cols["split"]
    rows = []
    for split, g in best_pred.groupby(split_col):
        mm = metric_values(g[pred_cols["obs"]].values, g[pred_cols["pred"]].values)
        rows.append({"Split": split, "R2": mm["R2"], "RMSE": mm["RMSE"], "N_Retained": mm["N"]})
    split_df = pd.DataFrame(rows)
    split_cols = {"split": "Split", "r2": "R2", "rmse": "RMSE", "n": "N_Retained"}
    log("[INFO] Computed split-stability metrics from detected prediction table.")

if split_df is not None and len(split_df) > 0:
    for key in ["r2", "rmse", "n"]:
        split_df[split_cols[key]] = numeric_series(split_df[split_cols[key]])
    split_df = split_df.dropna(subset=[split_cols["r2"], split_cols["rmse"], split_cols["n"]]).copy()
    if len(split_df) > 20:
        if "Method" in split_df.columns and pc.get("method"):
            tmp = split_df[split_df["Method"].astype(str).map(clean_name) == clean_name(best_method)].copy()
            if 2 <= len(tmp) <= 20:
                split_df = tmp
        elif "Policy_ID" in split_df.columns:
            tmp = split_df[split_df["Policy_ID"].astype(str).map(clean_name) == clean_name(best_row[pc["policy"]])].copy()
            if 2 <= len(tmp) <= 20:
                split_df = tmp
    if len(split_df) > 12:
        split_df = split_df.sort_values(split_cols["r2"], ascending=False).head(12).copy()

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.3))
    xx = np.arange(len(split_df))
    labels = [short_strategy(v) for v in split_df[split_cols["split"]].astype(str)]
    r2_vals = split_df[split_cols["r2"]].to_numpy(float)
    rmse_vals = split_df[split_cols["rmse"]].to_numpy(float)
    n_vals = split_df[split_cols["n"]].to_numpy(float)

    colors_r2 = [C_GREEN if r >= 0.5 and n >= 10 else (C_ORANGE if n < 10 else C_BLUE_LIGHT) for r, n in zip(r2_vals, n_vals)]
    bars = axes[0].bar(xx, r2_vals, color=colors_r2, edgecolor="#3F3F3F", linewidth=0.75, zorder=3)
    axes[0].axhline(0.5, color=C_RED, linestyle="--", linewidth=1.1)
    axes[0].set_title("$R^2$", pad=8)
    axes[0].set_ylabel("$R^2$")
    axes[0].set_xticks(xx)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylim(min(-0.3, np.nanmin(r2_vals) - 0.15), max(0.65, np.nanmax(r2_vals) + 0.12))
    style_axis(axes[0])
    add_bar_labels(axes[0], bars, "{:.2f}")

    colors_n = [C_ORANGE if n < 10 else C_BLUE for n in n_vals]
    bars = axes[1].bar(xx, rmse_vals, color=colors_n, edgecolor="#3F3F3F", linewidth=0.75, zorder=3)
    axes[1].set_title("RMSE", pad=8)
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticks(xx)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylim(0, max(1.0, np.nanmax(rmse_vals) * 1.25))
    style_axis(axes[1])
    add_bar_labels(axes[1], bars, "{:.2f}")

    bars = axes[2].bar(xx, n_vals, color=colors_n, edgecolor="#3F3F3F", linewidth=0.75, zorder=3)
    axes[2].set_title("Retained sample size", pad=8)
    axes[2].set_ylabel("N retained")
    axes[2].set_xticks(xx)
    axes[2].set_xticklabels(labels, rotation=35, ha="right")
    axes[2].set_ylim(0, max(5, np.nanmax(n_vals) * 1.25))
    style_axis(axes[2])
    add_bar_labels(axes[2], bars, "{:.0f}")

    fig.suptitle("Split-level stability of the selected 32D policy", fontsize=15.5, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.33, bottom=0.24, top=0.80)
    savefig(fig, "168_32d_est_replot_split_stability.png")


# =========================================================
# Report
# =========================================================
report_lines = [
    "32D EST-style final replot report",
    "=" * 80,
    f"Project root: {ROOT}",
    f"Policy summary table used: {policy_path.name}",
    f"Prediction table used: {pred_path.name if pred_path else 'Not detected'}",
    "",
    "Best policy detected from policy summary:",
    f"- Display Policy ID: {best_policy_display}",
    f"- Method: {best_method}",
    f"- Rule: {best_rule}",
    f"- Retained R2: {float(best_row[pc['r2']]):.6f}",
    f"- Retained RMSE: {float(best_row[pc['rmse']]):.6f}",
    f"- Retained MAE: {float(best_row[pc['mae']]):.6f}",
    f"- Coverage: {float(best_row[pc['coverage']]):.6f}",
    "",
    "Generated figures:",
    "- 165_32d_est_replot_policy_metrics.png/pdf",
    "- 166_32d_est_replot_r2_coverage_frontier.png/pdf",
    "- 167_32d_est_replot_best_prediction_panel.png/pdf, if prediction table was detected",
    "- 168_32d_est_replot_split_stability.png/pdf, if split information was detected",
]
report_path = TABLE_DIR / "32d_est_replot_report.txt"
report_path.write_text("\n".join(report_lines), encoding="utf-8")
log(f"[REPORT] {report_path}")

elapsed = time.time() - T0
log("=" * 90)
log("32D EST-style final replot finished.")
log(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
log("=" * 90)
