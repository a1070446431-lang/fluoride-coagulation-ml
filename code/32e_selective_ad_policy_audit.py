# -*- coding: utf-8 -*-
"""
32e_selective_ad_policy_audit.py

Purpose
-------
Strict audit for Step 32D selective applicability-domain (AD) subdomain results.

Why this script is needed
-------------------------
Step 32D found target-free selective subdomain policies with retained R2 > 0.5.
However, before using the result in a manuscript, we must verify:

1. The selected policy is truly target-free.
2. The policy-summary metrics can be reproduced from the prediction table.
3. The retained prediction panel uses exactly the same retained rows as the
   policy-summary calculation.
4. The result is not only caused by one split or a tiny retained subset.
5. Literature-group dominance is diagnosed.

Important
---------
- This script DOES NOT retrain any model.
- This script DOES NOT use test residuals or observed errors to select a policy.
- This script only audits existing Step 32D / 32C / 32B outputs.
- All figures are exported in a cleaner EST-like style.
- Total runtime is printed at the end.

Put this file under:
    fluoride_coagulation_ml_project/src/32e_selective_ad_policy_audit.py

Run in PyCharm:
    right click -> Run '32e_selective_ad_policy_audit'
"""

from __future__ import annotations

import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0. Global settings
# =============================================================================

SCRIPT_NAME = "32e_selective_ad_policy_audit.py"
START_TIME = time.time()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.unicode_minus": False,
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "axes.linewidth": 1.05,
    "axes.edgecolor": "#222222",
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
})

COLOR_MAIN = "#4E79A7"
COLOR_BEST = "#6FA77B"
COLOR_WARN = "#D9A051"
COLOR_BAD = "#C44E52"
COLOR_GRAY = "#BDBDBD"
COLOR_LIGHT = "#D9D9D9"
GRID_COLOR = "#E6E6E6"


# =============================================================================
# 1. Path utilities
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Locate project root by walking upward until data/outputs/src are found.
    """
    if start is None:
        start = Path(__file__).resolve().parent
    start = Path(start).resolve()

    candidates = [start] + list(start.parents)
    for p in candidates:
        if (p / "outputs").exists() and ((p / "src").exists() or (p / "data").exists()):
            return p

    # Fallback: if script is inside src, parent is project root.
    if start.name.lower() == "src":
        return start.parent
    return start


PROJECT_ROOT = find_project_root()
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def find_existing_file(candidates: List[Path], required: bool = True) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    if required:
        msg = "Missing required file. Tried:\n" + "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(msg)
    return None


# =============================================================================
# 2. Generic table utilities
# =============================================================================

def norm_text(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    """
    Find a column by exact / normalized / partial matching.
    """
    cols = list(df.columns)
    norm_map = {norm_text(c): c for c in cols}

    for cand in candidates:
        if cand in cols:
            return cand

    for cand in candidates:
        nc = norm_text(cand)
        if nc in norm_map:
            return norm_map[nc]

    for cand in candidates:
        nc = norm_text(cand)
        for col in cols:
            if nc and nc in norm_text(col):
                return col

    if required:
        raise KeyError(f"Cannot find required column from candidates={candidates}\nAvailable columns={cols}")
    return None


def safe_float_array(s: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    return arr


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=bool)
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = finite_mask(y_true, y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return np.nan
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 1e-12:
        return np.nan
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = finite_mask(y_true, y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = finite_mask(y_true, y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": r2_score_np(y_true, y_pred),
        "RMSE": rmse_np(y_true, y_pred),
        "MAE": mae_np(y_true, y_pred),
        "N": int(finite_mask(y_true, y_pred).sum()),
    }


def sanitize_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a copy and convert infinity-like strings to NaN where possible.
    """
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def read_csv_safely(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = sanitize_numeric_df(df)
    log(f"[TABLE] Loaded {path} | shape={df.shape}")
    return df


def to_csv_safely(df: pd.DataFrame, filename: str) -> Path:
    path = TABLE_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {path} | shape={df.shape}")
    return path


# =============================================================================
# 3. Plot utilities
# =============================================================================

def est_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.5, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.05)


def annotate_bars(ax: plt.Axes, fmt: str = "{:.2f}", fontsize: float = 9.5) -> None:
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 1.0
    for patch in ax.patches:
        h = patch.get_height()
        if not np.isfinite(h):
            continue
        x = patch.get_x() + patch.get_width() / 2
        if h >= 0:
            y = h + 0.015 * yr
            va = "bottom"
        else:
            y = h - 0.025 * yr
            va = "top"
        try:
            label = fmt.format(h)
        except Exception:
            label = str(h)
        ax.text(x, y, label, ha="center", va=va, fontsize=fontsize, color="black")


def save_fig(fig: plt.Figure, name: str) -> None:
    png = FIG_DIR / name
    pdf = FIG_DIR / name.replace(".png", ".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] {png}")
    log(f"[FIG] {pdf}")


# =============================================================================
# 4. Rule parsing and masks
# =============================================================================

def normalize_zone_value(x: object) -> str:
    s = str(x).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def match_zone(series: pd.Series, target_zone: str) -> pd.Series:
    z = normalize_zone_value(target_zone)
    vals = series.astype(str).map(normalize_zone_value)

    # Support common spellings.
    if z in {"reliable", "reliable domain", "rd"}:
        keys = ["reliable"]
    elif z in {"warning", "warning domain", "warn"}:
        keys = ["warning"]
    elif z in {"outside", "outside domain", "out", "ood"}:
        keys = ["outside"]
    else:
        keys = [z]

    mask = pd.Series(False, index=series.index)
    for k in keys:
        mask |= vals.str.contains(re.escape(k), na=False)
    return mask


def parse_condition_token(token: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse condition token such as:
      NN_Risk_le_0.814815
      Boundary_AD_Risk_ge_0.2
      Corrected_AD_Risk_lt_0.5
    Returns (column_name_text, operator, value_text).
    """
    token = str(token).strip()
    patterns = [
        r"^(?P<col>.+)_(?P<op>le|lt|ge|gt|eq|ne)_(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
        r"^(?P<col>.+)\s*(?P<op><=|<|>=|>|==|!=)\s*(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
    ]
    op_map = {"<=": "le", "<": "lt", ">=": "ge", ">": "gt", "==": "eq", "!=": "ne"}
    for pat in patterns:
        m = re.match(pat, token)
        if m:
            col = m.group("col")
            op = op_map.get(m.group("op"), m.group("op"))
            val = m.group("val")
            return col, op, val
    return None


def condition_mask(df: pd.DataFrame, col_text: str, op: str, val_text: str) -> Tuple[pd.Series, str]:
    col = find_col(df, [col_text], required=True)
    x = pd.to_numeric(df[col], errors="coerce")
    val = float(val_text)

    if op == "le":
        mask = x <= val
    elif op == "lt":
        mask = x < val
    elif op == "ge":
        mask = x >= val
    elif op == "gt":
        mask = x > val
    elif op == "eq":
        mask = np.isclose(x, val, equal_nan=False)
    elif op == "ne":
        mask = ~np.isclose(x, val, equal_nan=False)
    else:
        raise ValueError(f"Unsupported operator: {op}")

    return pd.Series(mask, index=df.index).fillna(False), col


def rule_mask(df: pd.DataFrame, rule: str) -> Tuple[pd.Series, List[str], List[str]]:
    """
    Build a boolean mask from a target-free policy rule string.

    This parser is intentionally conservative. If a token cannot be parsed,
    it reports it in unresolved_tokens rather than silently ignoring it.
    """
    if not isinstance(rule, str) or not rule.strip():
        return pd.Series(True, index=df.index), [], ["EMPTY_RULE"]

    used_cols: List[str] = []
    unresolved: List[str] = []
    mask = pd.Series(True, index=df.index)

    tokens = [t.strip() for t in re.split(r"__|&&|\bAND\b", rule) if t.strip()]

    reliability_col = find_col(
        df,
        ["Reliability_Zone", "Reliability Zone", "AD_Zone", "Domain_Zone", "Zone", "RD_Zone"],
        required=False,
    )

    for token in tokens:
        t = token.strip()

        # Zone token: Zone_Outside / Zone_Warning / Zone_Reliable
        if norm_text(t).startswith("zone"):
            zone_value = re.sub(r"^Zone[_\s=]*", "", t, flags=re.IGNORECASE).strip()
            if zone_value.lower().startswith("eq_"):
                zone_value = zone_value[3:]
            if reliability_col is None:
                unresolved.append(f"{token}: no reliability-zone column found")
                continue
            zmask = match_zone(df[reliability_col], zone_value)
            mask &= zmask
            used_cols.append(reliability_col)
            continue

        parsed = parse_condition_token(t)
        if parsed is None:
            unresolved.append(token)
            continue

        col_text, op, val_text = parsed
        try:
            cmask, actual_col = condition_mask(df, col_text, op, val_text)
            mask &= cmask
            used_cols.append(actual_col)
        except Exception as exc:
            unresolved.append(f"{token}: {exc}")

    used_cols = list(dict.fromkeys(used_cols))
    return mask.fillna(False), used_cols, unresolved


# =============================================================================
# 5. Main audit
# =============================================================================

def infer_best_row(policy_df: pd.DataFrame) -> pd.Series:
    r2_col = find_col(policy_df, ["Retained_R2", "R2", "retained_r2", "Retained R2"])
    cov_col = find_col(policy_df, ["Coverage", "Retained_Coverage", "coverage"], required=False)

    tmp = policy_df.copy()
    tmp["_audit_r2"] = pd.to_numeric(tmp[r2_col], errors="coerce")

    # Prefer rows over 0.5. If several exist, choose highest R2.
    valid = tmp[np.isfinite(tmp["_audit_r2"])]
    if valid.empty:
        raise ValueError("No finite retained R2 values found in policy summary.")
    best_idx = valid["_audit_r2"].idxmax()
    return policy_df.loc[best_idx].copy()


def get_row_value(row: pd.Series, candidates: List[str], default: Optional[object] = None) -> Optional[object]:
    for c in candidates:
        if c in row.index:
            return row[c]
    nmap = {norm_text(k): k for k in row.index}
    for c in candidates:
        nc = norm_text(c)
        if nc in nmap:
            return row[nmap[nc]]
    return default


def filter_by_method_strategy(pred_df: pd.DataFrame, best_row: pd.Series) -> Tuple[pd.DataFrame, Dict[str, object]]:
    method_col = find_col(pred_df, ["Method", "Model", "model_name", "method"], required=False)
    strategy_col = find_col(pred_df, ["Strategy", "Validation_Strategy", "strategy"], required=False)

    best_method = get_row_value(best_row, ["Method", "Model", "Model_Name", "method"], None)
    best_strategy = get_row_value(best_row, ["Strategy", "Validation_Strategy", "strategy"], None)

    out = pred_df.copy()
    info: Dict[str, object] = {
        "method_col": method_col,
        "strategy_col": strategy_col,
        "best_method": best_method,
        "best_strategy": best_strategy,
        "method_filter_applied": False,
        "strategy_filter_applied": False,
    }

    if method_col is not None and best_method is not None and str(best_method).strip():
        m = out[method_col].astype(str).map(norm_text) == norm_text(best_method)
        if m.sum() > 0:
            out = out.loc[m].copy()
            info["method_filter_applied"] = True

    if strategy_col is not None and best_strategy is not None and str(best_strategy).strip():
        s = out[strategy_col].astype(str).map(norm_text) == norm_text(best_strategy)
        if s.sum() > 0:
            out = out.loc[s].copy()
            info["strategy_filter_applied"] = True

    return out, info


def build_audit_tables() -> Dict[str, object]:
    policy_path = find_existing_file([
        TABLE_DIR / "32d_high_r2_policy_summary.csv",
        TABLE_DIR / "32d_est_replot_policy_id_mapping.csv",
        TABLE_DIR / "32d_policy_summary.csv",
        TABLE_DIR / "32c_fast_policy_summary.csv",
    ])

    pred_path = find_existing_file([
        TABLE_DIR / "32c_fast_selected_holdout_predictions.csv",
        TABLE_DIR / "32b_predictions_with_reliability_zone.csv",
        TABLE_DIR / "32a2_ad_predictions_with_scores.csv",
        TABLE_DIR / "32a_ad_predictions_with_scores.csv",
    ])

    policy_df = read_csv_safely(policy_path)
    pred_df = read_csv_safely(pred_path)

    best_row = infer_best_row(policy_df)

    policy_id = get_row_value(best_row, ["Display_Policy_ID", "Policy_ID", "Policy ID", "ID"], "P_best")
    method = get_row_value(best_row, ["Method", "Model", "Model_Name"], "")
    rule = get_row_value(best_row, ["Rule", "Policy_Rule", "Policy", "Rule_Text"], "")

    r2_summary = get_row_value(best_row, ["Retained_R2", "R2"], np.nan)
    rmse_summary = get_row_value(best_row, ["Retained_RMSE", "RMSE"], np.nan)
    mae_summary = get_row_value(best_row, ["Retained_MAE", "MAE"], np.nan)
    cov_summary = get_row_value(best_row, ["Coverage", "Retained_Coverage"], np.nan)

    filtered_df, filter_info = filter_by_method_strategy(pred_df, best_row)
    mask, used_cols, unresolved = rule_mask(filtered_df, str(rule))
    retained = filtered_df.loc[mask].copy()

    obs_col = find_col(pred_df, ["Observed", "Observed_F_removal", "Observed F removal (%)", "y_true", "Actual"])
    pred_col = find_col(pred_df, ["Predicted", "Predicted_F_removal", "Predicted F removal (%)", "y_pred", "Prediction"])

    y = safe_float_array(retained[obs_col]) if len(retained) else np.array([])
    yhat = safe_float_array(retained[pred_col]) if len(retained) else np.array([])
    m = metric_dict(y, yhat) if len(retained) else {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "N": 0}
    m["Coverage_vs_filtered"] = float(len(retained) / len(filtered_df)) if len(filtered_df) else np.nan
    m["Coverage_vs_all_prediction_rows"] = float(len(retained) / len(pred_df)) if len(pred_df) else np.nan

    summary_numbers = {
        "R2": pd.to_numeric(pd.Series([r2_summary]), errors="coerce").iloc[0],
        "RMSE": pd.to_numeric(pd.Series([rmse_summary]), errors="coerce").iloc[0],
        "MAE": pd.to_numeric(pd.Series([mae_summary]), errors="coerce").iloc[0],
        "Coverage": pd.to_numeric(pd.Series([cov_summary]), errors="coerce").iloc[0],
    }

    # Consistency flags.
    consistency_rows = []
    checks = [
        ("R2", summary_numbers["R2"], m["R2"], 0.005),
        ("RMSE", summary_numbers["RMSE"], m["RMSE"], 0.05),
        ("MAE", summary_numbers["MAE"], m["MAE"], 0.05),
        ("Coverage", summary_numbers["Coverage"], m["Coverage_vs_filtered"], 0.002),
    ]
    for name, a, b, tol in checks:
        diff = abs(float(a) - float(b)) if np.isfinite(a) and np.isfinite(b) else np.nan
        consistency_rows.append({
            "Metric": name,
            "Summary_Value": a,
            "Recomputed_Value": b,
            "Abs_Difference": diff,
            "Tolerance": tol,
            "Pass": bool(np.isfinite(diff) and diff <= tol),
        })
    consistency_df = pd.DataFrame(consistency_rows)

    forbidden = [
        "observed", "actual", "target", "removal", "predicted", "prediction",
        "residual", "error", "absolute", "abs", "rmse", "mae", "r2", "heldout",
        "testlabel", "ytrue", "ypred",
    ]
    leakage_rows = []
    for c in used_cols:
        nc = norm_text(c)
        hits = [f for f in forbidden if f in nc]
        leakage_rows.append({
            "Used_Rule_Column": c,
            "Forbidden_Substring_Hits": ";".join(hits),
            "Target_Free_Pass": len(hits) == 0,
        })
    if not leakage_rows:
        leakage_rows.append({
            "Used_Rule_Column": "(none parsed)",
            "Forbidden_Substring_Hits": "UNRESOLVED_RULE",
            "Target_Free_Pass": False,
        })
    leakage_df = pd.DataFrame(leakage_rows)

    # Split-level stability.
    split_col = find_col(
        retained,
        ["Split_ID", "Seed_or_Fold", "Seed", "Fold", "split", "Random seed", "Random_Seed"],
        required=False,
    )
    split_rows = []
    if split_col is not None and len(retained) > 0:
        for sid, g in retained.groupby(split_col, dropna=False):
            yy = safe_float_array(g[obs_col])
            pp = safe_float_array(g[pred_col])
            md = metric_dict(yy, pp)
            split_rows.append({
                "Split_ID": sid,
                "R2": md["R2"],
                "RMSE": md["RMSE"],
                "MAE": md["MAE"],
                "N_Retained": md["N"],
            })
    split_df = pd.DataFrame(split_rows)

    # Literature-group dominance.
    group_col = find_col(
        retained,
        ["文献编号", "Literature_ID", "Paper_ID", "Source_Group", "Group", "Literature", "Paper"],
        required=False,
    )
    group_df = pd.DataFrame()
    if group_col is not None and len(retained) > 0:
        group_df = (
            retained[group_col]
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("Literature_Group")
            .reset_index(name="N_Retained")
        )
        group_df["Fraction_of_Retained"] = group_df["N_Retained"] / group_df["N_Retained"].sum()

    audit_meta = {
        "policy_path": policy_path,
        "prediction_path": pred_path,
        "obs_col": obs_col,
        "pred_col": pred_col,
        "policy_id": policy_id,
        "method": method,
        "rule": rule,
        "used_cols": used_cols,
        "unresolved_rule_tokens": unresolved,
        "filter_info": filter_info,
        "summary_numbers": summary_numbers,
        "recomputed_metrics": m,
        "filtered_n": len(filtered_df),
        "prediction_n": len(pred_df),
        "retained_n": len(retained),
        "split_col": split_col,
        "group_col": group_col,
    }

    return {
        "policy_df": policy_df,
        "pred_df": pred_df,
        "filtered_df": filtered_df,
        "retained": retained,
        "consistency_df": consistency_df,
        "leakage_df": leakage_df,
        "split_df": split_df,
        "group_df": group_df,
        "audit_meta": audit_meta,
    }


# =============================================================================
# 6. Figures
# =============================================================================

def plot_consistency(consistency_df: pd.DataFrame, audit_meta: Dict[str, object]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))
    axes = axes.ravel()

    metrics = ["R2", "RMSE", "MAE", "Coverage"]
    for ax, metric in zip(axes, metrics):
        row = consistency_df[consistency_df["Metric"] == metric]
        if row.empty:
            continue
        vals = [float(row["Summary_Value"].iloc[0]), float(row["Recomputed_Value"].iloc[0])]
        labels = ["Policy summary", "Recomputed"]
        colors = [COLOR_MAIN, COLOR_BEST if bool(row["Pass"].iloc[0]) else COLOR_BAD]

        ax.bar(labels, vals, color=colors, edgecolor="#333333", linewidth=0.8, width=0.62)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        est_ax(ax)
        annotate_bars(ax, "{:.3f}" if metric in ["R2", "Coverage"] else "{:.2f}", fontsize=10)

        if metric == "R2":
            ax.axhline(0.5, color=COLOR_BAD, linestyle="--", linewidth=1.0)
            ax.text(0.02, 0.52, r"$R^2=0.5$", transform=ax.get_yaxis_transform(),
                    fontsize=9, color=COLOR_BAD, va="bottom")

    title = f"32E consistency audit for selected policy: {audit_meta.get('policy_id')} / {audit_meta.get('method')}"
    fig.suptitle(title, y=0.98, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "169_32e_policy_consistency_audit.png")


def plot_corrected_prediction_panel(retained: pd.DataFrame, audit_meta: Dict[str, object]) -> None:
    obs_col = str(audit_meta["obs_col"])
    pred_col = str(audit_meta["pred_col"])

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    est_ax(ax)

    y = safe_float_array(retained[obs_col]) if len(retained) else np.array([])
    p = safe_float_array(retained[pred_col]) if len(retained) else np.array([])
    mask = finite_mask(y, p) if len(y) else np.array([], dtype=bool)
    y = y[mask]
    p = p[mask]
    md = metric_dict(y, p) if len(y) else {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "N": 0}

    ax.scatter(
        y, p,
        s=42,
        alpha=0.82,
        color=COLOR_MAIN,
        edgecolor="white",
        linewidth=0.45,
    )
    ax.plot([0, 100], [0, 100], "--", color=COLOR_BAD, linewidth=1.4)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Predicted F removal (%)")
    ax.set_title(f"Corrected retained prediction panel: {audit_meta.get('method')}")

    box = (
        rf"$R^2$ = {md['R2']:.3f}" + "\n"
        f"RMSE = {md['RMSE']:.2f}\n"
        f"MAE = {md['MAE']:.2f}\n"
        f"n = {md['N']}\n"
        f"policy = {audit_meta.get('policy_id')}"
    )
    ax.text(
        0.06, 0.94, box,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#999999", alpha=0.92),
    )

    fig.tight_layout()
    save_fig(fig, "170_32e_corrected_retained_prediction_panel.png")


def plot_split_stability(split_df: pd.DataFrame) -> None:
    if split_df.empty:
        log("[WARN] No split column found; skip split stability plot.")
        return

    df = split_df.copy()
    df["Split_Label"] = df["Split_ID"].astype(str)
    df = df.sort_values("Split_Label").reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5))
    panels = [
        ("R2", r"$R^2$", "{:.2f}"),
        ("RMSE", "RMSE", "{:.2f}"),
        ("N_Retained", "N retained", "{:.0f}"),
    ]

    for ax, (col, ylabel, fmt) in zip(axes, panels):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        if col == "R2":
            colors = [COLOR_BEST if np.isfinite(v) and v >= 0.5 else COLOR_WARN for v in vals]
        else:
            colors = [COLOR_MAIN for _ in vals]

        ax.bar(df["Split_Label"], vals, color=colors, edgecolor="#333333", linewidth=0.8)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Split")
        est_ax(ax)
        ax.tick_params(axis="x", rotation=35)
        annotate_bars(ax, fmt, fontsize=9.5)

        if col == "R2":
            ax.axhline(0.5, color=COLOR_BAD, linestyle="--", linewidth=1.1)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(min(ymin, np.nanmin(vals) - 0.1 if np.isfinite(vals).any() else ymin), max(ymax, 0.6))

    fig.suptitle("32E split-level stability audit of retained subdomain", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "171_32e_split_level_stability_audit.png")


def plot_group_dominance(group_df: pd.DataFrame) -> None:
    if group_df.empty:
        log("[WARN] No literature group column found; skip group dominance plot.")
        return

    df = group_df.head(15).copy()
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.barh(df["Literature_Group"], df["N_Retained"], color=COLOR_MAIN, edgecolor="#333333", linewidth=0.8)
    ax.set_xlabel("Retained sample count")
    ax.set_ylabel("Literature group")
    ax.set_title("Literature-group dominance in retained subdomain")
    est_ax(ax)

    xmax = max(df["N_Retained"].max(), 1)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["N_Retained"] + 0.02 * xmax,
            i,
            f"{int(row['N_Retained'])} ({row['Fraction_of_Retained']:.1%})",
            va="center", ha="left", fontsize=9.5,
        )

    fig.tight_layout()
    save_fig(fig, "172_32e_retained_literature_group_dominance.png")


def plot_rule_columns(leakage_df: pd.DataFrame) -> None:
    df = leakage_df.copy()
    df["Status"] = np.where(df["Target_Free_Pass"], "Target-free", "Check needed")
    df["Plot_Value"] = 1

    fig, ax = plt.subplots(figsize=(9.2, max(3.8, 0.42 * len(df) + 1.4)))
    y = np.arange(len(df))
    colors = [COLOR_BEST if ok else COLOR_BAD for ok in df["Target_Free_Pass"]]
    ax.barh(y, df["Plot_Value"], color=colors, edgecolor="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Used_Rule_Column"])
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Policy-rule column audit")
    ax.set_title("Target-free leakage guard for selected policy rule")
    ax.set_xticks([])
    est_ax(ax)

    for i, row in df.iterrows():
        txt = "pass" if row["Target_Free_Pass"] else f"check: {row['Forbidden_Substring_Hits']}"
        ax.text(1.02, i, txt, va="center", ha="left", fontsize=9.5)

    fig.tight_layout()
    save_fig(fig, "173_32e_target_free_rule_column_audit.png")


# =============================================================================
# 7. Decision report
# =============================================================================

def make_decision_report(
    audit_meta: Dict[str, object],
    consistency_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    split_df: pd.DataFrame,
    group_df: pd.DataFrame,
) -> str:
    m = audit_meta["recomputed_metrics"]
    summary = audit_meta["summary_numbers"]

    consistency_pass = bool(consistency_df["Pass"].all()) if not consistency_df.empty else False
    leakage_pass = bool(leakage_df["Target_Free_Pass"].all()) if not leakage_df.empty else False

    r2_ok = np.isfinite(m["R2"]) and m["R2"] >= 0.5
    n_ok = int(m["N"]) >= 50
    coverage_ok = np.isfinite(m["Coverage_vs_filtered"]) and m["Coverage_vs_filtered"] >= 0.01

    if not split_df.empty and "R2" in split_df:
        finite_split = split_df[np.isfinite(pd.to_numeric(split_df["R2"], errors="coerce"))].copy()
        n_splits = len(finite_split)
        n_good = int((pd.to_numeric(finite_split["R2"], errors="coerce") >= 0.5).sum())
        frac_good = n_good / n_splits if n_splits else np.nan
        min_n = int(pd.to_numeric(split_df["N_Retained"], errors="coerce").min()) if len(split_df) else 0
        split_stability_pass = bool(n_splits >= 3 and frac_good >= 0.6 and min_n >= 5)
    else:
        n_splits = 0
        n_good = 0
        frac_good = np.nan
        min_n = 0
        split_stability_pass = False

    if not group_df.empty:
        max_group_frac = float(group_df["Fraction_of_Retained"].max())
        group_dominance_pass = max_group_frac <= 0.35
    else:
        max_group_frac = np.nan
        group_dominance_pass = False

    strict_accept = bool(
        consistency_pass and leakage_pass and r2_ok and n_ok and coverage_ok and split_stability_pass
    )
    exploratory_accept = bool(
        leakage_pass and r2_ok and n_ok and coverage_ok
    )

    lines = []
    lines.append("32E selective AD policy audit report")
    lines.append("=" * 88)
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Policy summary table: {audit_meta['policy_path']}")
    lines.append(f"Prediction table: {audit_meta['prediction_path']}")
    lines.append("")
    lines.append("Selected policy audited")
    lines.append("-" * 88)
    lines.append(f"Policy ID: {audit_meta.get('policy_id')}")
    lines.append(f"Method: {audit_meta.get('method')}")
    lines.append(f"Rule: {audit_meta.get('rule')}")
    lines.append(f"Used rule columns: {audit_meta.get('used_cols')}")
    lines.append(f"Unresolved rule tokens: {audit_meta.get('unresolved_rule_tokens')}")
    lines.append(f"Filter info: {audit_meta.get('filter_info')}")
    lines.append("")
    lines.append("Summary metrics vs recomputed metrics")
    lines.append("-" * 88)
    lines.append(f"Summary R2/RMSE/MAE/Coverage: {summary['R2']:.6f}, {summary['RMSE']:.6f}, {summary['MAE']:.6f}, {summary['Coverage']:.6f}")
    lines.append(f"Recomputed R2/RMSE/MAE/N: {m['R2']:.6f}, {m['RMSE']:.6f}, {m['MAE']:.6f}, {m['N']}")
    lines.append(f"Coverage vs filtered method-strategy rows: {m['Coverage_vs_filtered']:.6f}")
    lines.append(f"Coverage vs all prediction rows: {m['Coverage_vs_all_prediction_rows']:.6f}")
    lines.append("")
    lines.append("Audit flags")
    lines.append("-" * 88)
    lines.append(f"Consistency pass: {consistency_pass}")
    lines.append(f"Target-free leakage pass: {leakage_pass}")
    lines.append(f"R2 >= 0.5 pass: {r2_ok}")
    lines.append(f"N retained >= 50 pass: {n_ok}")
    lines.append(f"Coverage >= 0.01 pass: {coverage_ok}")
    lines.append(f"Split stability pass: {split_stability_pass} | good splits = {n_good}/{n_splits}, fraction = {frac_good:.3f}, min split n = {min_n}")
    lines.append(f"Literature dominance pass: {group_dominance_pass} | max retained group fraction = {max_group_frac:.3f}" if np.isfinite(max_group_frac) else "Literature dominance pass: not evaluated")
    lines.append("")
    lines.append("Decision")
    lines.append("-" * 88)
    if strict_accept:
        lines.append("STRICT_ACCEPT: The selected high-R2 target-free AD subdomain is internally consistent and sufficiently stable for main-text reporting.")
    elif exploratory_accept:
        lines.append("EXPLORATORY_ACCEPT_ONLY: The selected high-R2 target-free AD subdomain reaches R2 >= 0.5 without obvious target leakage, but one or more robustness checks failed.")
        lines.append("Recommended wording: report as a highly restricted reliable subdomain / selective-prediction diagnostic, not as full-domain model generalization.")
    else:
        lines.append("DO_NOT_ACCEPT_YET: The selected policy does not pass the minimum audit required for manuscript-level use.")
        lines.append("Recommended action: inspect consistency, filtering, rule parsing, and prediction-panel row selection before using this result.")
    lines.append("")
    lines.append("Generated outputs")
    lines.append("-" * 88)
    lines.append("Tables:")
    lines.append("  - 32e_policy_consistency_audit.csv")
    lines.append("  - 32e_target_free_rule_column_audit.csv")
    lines.append("  - 32e_split_level_stability_audit.csv")
    lines.append("  - 32e_retained_literature_group_dominance.csv")
    lines.append("  - 32e_corrected_retained_predictions.csv")
    lines.append("Figures:")
    lines.append("  - 169_32e_policy_consistency_audit.png/pdf")
    lines.append("  - 170_32e_corrected_retained_prediction_panel.png/pdf")
    lines.append("  - 171_32e_split_level_stability_audit.png/pdf")
    lines.append("  - 172_32e_retained_literature_group_dominance.png/pdf")
    lines.append("  - 173_32e_target_free_rule_column_audit.png/pdf")
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# 8. Run
# =============================================================================

def main() -> None:
    log("=" * 88)
    log(f"Running {SCRIPT_NAME}")
    log(f"Project root: {PROJECT_ROOT}")
    log("=" * 88)

    result = build_audit_tables()

    policy_df = result["policy_df"]
    retained = result["retained"]
    consistency_df = result["consistency_df"]
    leakage_df = result["leakage_df"]
    split_df = result["split_df"]
    group_df = result["group_df"]
    audit_meta = result["audit_meta"]

    # Save audit tables.
    to_csv_safely(consistency_df, "32e_policy_consistency_audit.csv")
    to_csv_safely(leakage_df, "32e_target_free_rule_column_audit.csv")
    to_csv_safely(split_df, "32e_split_level_stability_audit.csv")
    to_csv_safely(group_df, "32e_retained_literature_group_dominance.csv")
    to_csv_safely(retained, "32e_corrected_retained_predictions.csv")

    # Figures.
    plot_consistency(consistency_df, audit_meta)
    plot_corrected_prediction_panel(retained, audit_meta)
    plot_split_stability(split_df)
    plot_group_dominance(group_df)
    plot_rule_columns(leakage_df)

    # Decision report.
    report = make_decision_report(audit_meta, consistency_df, leakage_df, split_df, group_df)
    report_path = TABLE_DIR / "32e_selective_ad_policy_audit_report.txt"
    report_path.write_text(report, encoding="utf-8")
    log(f"[REPORT] {report_path}")
    log("")
    log(report)

    elapsed = time.time() - START_TIME
    log("=" * 88)
    log("32E selective AD policy audit finished.")
    log(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    log("=" * 88)


if __name__ == "__main__":
    main()
