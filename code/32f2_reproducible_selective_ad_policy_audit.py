# -*- coding: utf-8 -*-
"""
32f2_reproducible_selective_ad_policy_audit.py

Purpose
-------
Step 32F2 for the fluoride coagulation ML project.

This script performs a targeted, reproducible audit of the Step 32D selective
applicability-domain (AD) policies. It intentionally avoids the very large and
slow all-candidate text/string scan used in 32F.

Core questions answered
-----------------------
1. Can the Step 32D high-R2 policy table be exactly reproduced from the correct
   prediction table?
2. Which prediction table reproduces Step 32D: 32b_predictions_with_reliability_zone.csv
   or 32c_fast_selected_holdout_predictions.csv?
3. Is the selected high-R2 retained subdomain target-free, parseable, and stable
   enough to report as a selective AD result?
4. Which literature groups dominate the retained subdomain?

Strictness
----------
- No model is retrained.
- No test residual is used to build or tune a new model.
- Policy rules are taken from 32d_policy_label_mapping.csv / 32d_high_r2_policy_summary.csv.
- Metrics are recomputed directly from prediction rows.
- The final decision distinguishes:
    full-domain generalization vs selective retained-subdomain performance.

Put this file under:
    fluoride_coagulation_ml_project/src/32f2_reproducible_selective_ad_policy_audit.py

Run in PyCharm:
    Right click -> Run '32f2_reproducible_selective_ad_policy_audit'
"""

from __future__ import annotations

import math
import re
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0. Global settings
# =============================================================================

SCRIPT_NAME = "32f2_reproducible_selective_ad_policy_audit.py"
START_TIME = time.time()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# EST-like plotting style: white background, clean serif font, low-saturation colors.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.unicode_minus": False,
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "axes.linewidth": 1.05,
    "axes.edgecolor": "#222222",
    "axes.titlesize": 14,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
})

COLOR_BLUE = "#4E79A7"
COLOR_GREEN = "#6FA77B"
COLOR_ORANGE = "#D9A051"
COLOR_RED = "#C44E52"
COLOR_GRAY = "#8E8E8E"
COLOR_LIGHT = "#D8D8D8"
GRID_COLOR = "#E6E6E6"

# Acceptance thresholds for the manuscript-readiness audit.
R2_TARGET = 0.50
MIN_N_RETAINED = 50
MIN_COVERAGE_WITHIN_METHOD_STRATEGY = 0.01
MAX_SUMMARY_R2_DIFF = 1e-8
MAX_SUMMARY_RMSE_DIFF = 1e-8
MAX_SUMMARY_MAE_DIFF = 1e-8


# =============================================================================
# 1. Path utilities
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find project root when the script is placed under src/."""
    if start is None:
        start = Path(__file__).resolve().parent
    start = Path(start).resolve()

    for p in [start] + list(start.parents):
        if (p / "outputs").exists() and ((p / "src").exists() or (p / "data").exists()):
            return p

    # Fallback for testing outside the user's Windows project.
    return start.parent if start.name.lower() == "src" else start


PROJECT_ROOT = find_project_root()
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Additional search folders make the script testable if CSV files are placed next to the script.
SEARCH_DIRS = [
    TABLE_DIR,
    PROJECT_ROOT,
    Path(__file__).resolve().parent,
    Path.cwd(),
    Path("/mnt/data"),  # harmless fallback; useful only in ChatGPT-side testing
]


def find_input_file(filename: str, required: bool = True) -> Optional[Path]:
    """Find an input file from outputs/tables first, then safe fallbacks."""
    seen = set()
    for d in SEARCH_DIRS:
        if not d.exists():
            continue
        p = (d / filename).resolve()
        if str(p) in seen:
            continue
        seen.add(str(p))
        if p.exists():
            return p
    if required:
        tried = "\n".join(str((d / filename).resolve()) for d in SEARCH_DIRS if d.exists())
        raise FileNotFoundError(f"Missing required file: {filename}\nTried:\n{tried}")
    return None


def read_csv_required(filename: str) -> Tuple[pd.DataFrame, Path]:
    path = find_input_file(filename, required=True)
    assert path is not None
    df = pd.read_csv(path, low_memory=False)
    df = df.replace([np.inf, -np.inf], np.nan)
    log(f"[TABLE] Loaded {path} | shape={df.shape}")
    return df, path


def read_csv_optional(filename: str) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    path = find_input_file(filename, required=False)
    if path is None:
        log(f"[INFO] Optional file not found: {filename}")
        return None, None
    df = pd.read_csv(path, low_memory=False)
    df = df.replace([np.inf, -np.inf], np.nan)
    log(f"[TABLE] Loaded {path} | shape={df.shape}")
    return df, path


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    out = TABLE_DIR / filename
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log(f"[TABLE] Saved {out} | shape={df.shape}")
    return out


# =============================================================================
# 2. Data and metric utilities
# =============================================================================

def norm_text(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def find_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    cols = list(df.columns)
    norm_map = {norm_text(c): c for c in cols}

    for cand in candidates:
        if cand in cols:
            return cand

    for cand in candidates:
        key = norm_text(cand)
        if key in norm_map:
            return norm_map[key]

    for cand in candidates:
        key = norm_text(cand)
        if not key:
            continue
        for c in cols:
            if key in norm_text(c):
                return c

    if required:
        raise KeyError(f"Cannot find required column from {list(candidates)}. Available: {cols}")
    return None


def numeric_array(s: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    return arr


def regression_metrics_from_arrays(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(y) & np.isfinite(p)
    n = int(m.sum())
    if n == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "N": 0}

    yy = y[m]
    pp = p[m]
    rmse = float(np.sqrt(np.mean((yy - pp) ** 2)))
    mae = float(np.mean(np.abs(yy - pp)))

    if n < 2:
        r2 = np.nan
    else:
        denom = float(np.sum((yy - yy.mean()) ** 2))
        r2 = np.nan if denom <= 1e-12 else float(1.0 - np.sum((yy - pp) ** 2) / denom)

    return {"R2": r2, "RMSE": rmse, "MAE": mae, "N": n}


def regression_metrics(df: pd.DataFrame, obs_col: str = "Observed", pred_col: str = "Predicted") -> Dict[str, float]:
    if df.empty:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "N": 0}
    return regression_metrics_from_arrays(numeric_array(df[obs_col]), numeric_array(df[pred_col]))


def almost_equal(a: object, b: object, tol: float) -> bool:
    try:
        aa = float(a)
        bb = float(b)
    except Exception:
        return False
    if not np.isfinite(aa) or not np.isfinite(bb):
        return False
    return abs(aa - bb) <= tol


# =============================================================================
# 3. Rule parser
# =============================================================================

def normalize_zone(x: object) -> str:
    s = str(x).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def apply_zone_mask(series: pd.Series, zone_token: str) -> pd.Series:
    z = zone_token.strip()
    if z.lower().startswith("eq_"):
        z = z[3:]
    z_norm = normalize_zone(z)
    vals = series.astype(str).map(normalize_zone)

    if z_norm in {"reliable", "reliable domain"}:
        key = "reliable"
    elif z_norm in {"warning", "warning domain"}:
        key = "warning"
    elif z_norm in {"outside", "outside domain", "out"}:
        key = "outside"
    else:
        key = z_norm
    return vals.str.contains(re.escape(key), na=False)


def parse_numeric_condition(token: str) -> Optional[Tuple[str, str, float]]:
    """Parse conditions like Boundary_AD_Risk_le_0.266667 or NN_Risk <= 0.9."""
    t = str(token).strip().replace("≤", "<=").replace("≥", ">=")

    m = re.match(r"^(?P<col>.+)_(?P<op>le|lt|ge|gt|eq|ne)_(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", t)
    if m:
        return m.group("col"), m.group("op"), float(m.group("val"))

    m = re.match(r"^(?P<col>.+?)\s*(?P<op><=|<|>=|>|==|!=)\s*(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", t)
    if m:
        op_map = {"<=": "le", "<": "lt", ">=": "ge", ">": "gt", "==": "eq", "!=": "ne"}
        return m.group("col"), op_map[m.group("op")], float(m.group("val"))

    return None


def apply_numeric_mask(df: pd.DataFrame, col_hint: str, op: str, value: float) -> Tuple[pd.Series, str]:
    col = find_col(df, [col_hint], required=True)
    x = pd.to_numeric(df[col], errors="coerce")

    if op == "le":
        mask = x <= value
    elif op == "lt":
        mask = x < value
    elif op == "ge":
        mask = x >= value
    elif op == "gt":
        mask = x > value
    elif op == "eq":
        mask = pd.Series(np.isclose(x.to_numpy(dtype=float), value, equal_nan=False), index=df.index)
    elif op == "ne":
        mask = pd.Series(~np.isclose(x.to_numpy(dtype=float), value, equal_nan=False), index=df.index)
    else:
        raise ValueError(f"Unsupported operator: {op}")

    return pd.Series(mask, index=df.index).fillna(False), col


def parse_rule_tokens(rule_name: str) -> List[str]:
    rule = str(rule_name).strip()
    if not rule or rule.lower() in {"nan", "none", "null"}:
        return []
    return [t.strip() for t in re.split(r"__|&&|\bAND\b|\band\b|;", rule) if t.strip()]


def apply_policy_rule(df: pd.DataFrame, rule_name: str) -> Tuple[pd.Series, List[str], List[str]]:
    tokens = parse_rule_tokens(rule_name)
    if not tokens:
        return pd.Series(False, index=df.index), [], ["EMPTY_RULE"]

    zone_col = find_col(df, ["Reliability_Zone", "AD_Zone", "Domain_Zone", "Zone"], required=False)
    mask = pd.Series(True, index=df.index)
    used_cols: List[str] = []
    unresolved: List[str] = []

    for token in tokens:
        if token.lower().startswith("zone_"):
            if zone_col is None:
                unresolved.append(f"{token}: no Reliability_Zone column")
                continue
            zone_token = token[5:]
            mask &= apply_zone_mask(df[zone_col], zone_token)
            used_cols.append(zone_col)
            continue

        parsed = parse_numeric_condition(token)
        if parsed is None:
            unresolved.append(f"{token}: cannot parse")
            continue

        col_hint, op, value = parsed
        try:
            m, actual_col = apply_numeric_mask(df, col_hint, op, value)
            mask &= m
            used_cols.append(actual_col)
        except Exception as exc:
            unresolved.append(f"{token}: {exc}")

    used_cols = list(dict.fromkeys(used_cols))
    return mask.fillna(False), used_cols, unresolved


def target_free_audit(rule_name: str, used_cols: List[str]) -> Tuple[bool, str]:
    text = " ".join([str(rule_name)] + [str(c) for c in used_cols]).lower()
    leakage_terms = [
        "observed", "actual", "target", "氟去除率", "f removal", "removal_rate",
        "predicted", "prediction", "residual", "error", "abs_error", "absolute_error",
        "rmse", "mae", "mse", "r2", "test_error", "heldout_error",
    ]
    hits = [t for t in leakage_terms if t in text]
    return len(hits) == 0, ";".join(hits)


# =============================================================================
# 4. Candidate evaluation
# =============================================================================

def evaluate_policy_on_prediction_table(
    pred_df: pd.DataFrame,
    policy_row: pd.Series,
    source_label: str,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    obs_col = find_col(pred_df, ["Observed", "Actual", "y_true"], required=True)
    pred_col = find_col(pred_df, ["Predicted", "Prediction", "y_pred"], required=True)
    method_col = find_col(pred_df, ["Method", "Model"], required=True)
    strategy_col = find_col(pred_df, ["Strategy", "Validation_Strategy"], required=True)

    method = str(policy_row["Method"])
    strategy = str(policy_row["Strategy"])
    rule_name = str(policy_row["Rule_Name"])

    base = pred_df[
        (pred_df[method_col].astype(str) == method)
        & (pred_df[strategy_col].astype(str) == strategy)
    ].copy()

    mask, used_cols, unresolved = apply_policy_rule(base, rule_name)
    retained = base.loc[mask].copy()

    md = regression_metrics(retained, obs_col=obs_col, pred_col=pred_col)
    coverage = float(len(retained) / len(base)) if len(base) else np.nan
    target_free, leakage_hits = target_free_audit(rule_name, used_cols)

    r2_diff = abs(float(policy_row["R2"]) - md["R2"]) if np.isfinite(md["R2"]) else np.nan
    rmse_diff = abs(float(policy_row["RMSE"]) - md["RMSE"]) if np.isfinite(md["RMSE"]) else np.nan
    mae_diff = abs(float(policy_row["MAE"]) - md["MAE"]) if np.isfinite(md["MAE"]) else np.nan
    n_diff = abs(int(policy_row["N_Retained"]) - int(md["N"])) if np.isfinite(md["N"]) else np.nan

    exact_reproduction = (
        almost_equal(policy_row["R2"], md["R2"], MAX_SUMMARY_R2_DIFF)
        and almost_equal(policy_row["RMSE"], md["RMSE"], MAX_SUMMARY_RMSE_DIFF)
        and almost_equal(policy_row["MAE"], md["MAE"], MAX_SUMMARY_MAE_DIFF)
        and int(policy_row["N_Retained"]) == int(md["N"])
    )

    manuscript_ready = (
        bool(exact_reproduction)
        and bool(target_free)
        and len(unresolved) == 0
        and np.isfinite(md["R2"])
        and md["R2"] >= R2_TARGET
        and int(md["N"]) >= MIN_N_RETAINED
        and np.isfinite(coverage)
        and coverage >= MIN_COVERAGE_WITHIN_METHOD_STRATEGY
    )

    result = {
        "Prediction_Table": source_label,
        "Policy_ID": policy_row.get("Policy_ID", ""),
        "Rank": policy_row.get("Rank", np.nan),
        "Strategy": strategy,
        "Method": method,
        "Rule_Name": rule_name,
        "Rule_Description": policy_row.get("Rule_Description", ""),
        "N_Base": int(len(base)),
        "N_Retained_Recomputed": int(md["N"]),
        "Coverage_Within_Strategy_Method_Recomputed": coverage,
        "R2_Recomputed": md["R2"],
        "RMSE_Recomputed": md["RMSE"],
        "MAE_Recomputed": md["MAE"],
        "Summary_N_Retained": policy_row.get("N_Retained", np.nan),
        "Summary_Coverage": policy_row.get("Coverage_Within_Strategy_Method", np.nan),
        "Summary_R2": policy_row.get("R2", np.nan),
        "Summary_RMSE": policy_row.get("RMSE", np.nan),
        "Summary_MAE": policy_row.get("MAE", np.nan),
        "R2_Abs_Diff": r2_diff,
        "RMSE_Abs_Diff": rmse_diff,
        "MAE_Abs_Diff": mae_diff,
        "N_Abs_Diff": n_diff,
        "Exact_Reproduction_Pass": bool(exact_reproduction),
        "Target_Free_Pass": bool(target_free),
        "Leakage_Hits": leakage_hits,
        "Rule_Parse_Pass": len(unresolved) == 0,
        "Unresolved_Rule_Tokens": ";".join(unresolved),
        "Used_Rule_Columns": ";".join(used_cols),
        "R2_Target_Pass": bool(np.isfinite(md["R2"]) and md["R2"] >= R2_TARGET),
        "Min_N_Pass": bool(int(md["N"]) >= MIN_N_RETAINED),
        "Min_Coverage_Pass": bool(np.isfinite(coverage) and coverage >= MIN_COVERAGE_WITHIN_METHOD_STRATEGY),
        "Manuscript_Ready_Selective_AD": bool(manuscript_ready),
    }

    retained["_32F2_Prediction_Table"] = source_label
    retained["_32F2_Policy_ID"] = policy_row.get("Policy_ID", "")
    retained["_32F2_Method"] = method
    retained["_32F2_Strategy"] = strategy
    retained["_32F2_Rule_Name"] = rule_name

    return result, retained


def evaluate_all_policies(policy_df: pd.DataFrame, primary_pred: pd.DataFrame, optional_pred: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows: List[Dict[str, object]] = []
    retained_by_key: Dict[str, pd.DataFrame] = {}

    for _, row in policy_df.iterrows():
        result, retained = evaluate_policy_on_prediction_table(primary_pred, row, "32b_predictions_with_reliability_zone")
        rows.append(result)
        retained_by_key[f"32b::{result['Policy_ID']}::{result['Method']}::{result['Rule_Name']}"] = retained

        if optional_pred is not None:
            try:
                result2, retained2 = evaluate_policy_on_prediction_table(optional_pred, row, "32c_fast_selected_holdout_predictions")
                rows.append(result2)
                retained_by_key[f"32c::{result2['Policy_ID']}::{result2['Method']}::{result2['Rule_Name']}"] = retained2
            except Exception as exc:
                failed = dict(result)
                failed.update({
                    "Prediction_Table": "32c_fast_selected_holdout_predictions",
                    "N_Base": 0,
                    "N_Retained_Recomputed": 0,
                    "Coverage_Within_Strategy_Method_Recomputed": np.nan,
                    "R2_Recomputed": np.nan,
                    "RMSE_Recomputed": np.nan,
                    "MAE_Recomputed": np.nan,
                    "Exact_Reproduction_Pass": False,
                    "Manuscript_Ready_Selective_AD": False,
                    "Unresolved_Rule_Tokens": str(exc),
                })
                rows.append(failed)

    out = pd.DataFrame(rows)
    return out, retained_by_key


def select_best_reproduced_policy(audit_df: pd.DataFrame) -> Optional[pd.Series]:
    if audit_df.empty:
        return None
    df = audit_df.copy()
    for col in ["R2_Recomputed", "RMSE_Recomputed", "MAE_Recomputed", "N_Retained_Recomputed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    strict = df[
        (df["Prediction_Table"] == "32b_predictions_with_reliability_zone")
        & (df["Exact_Reproduction_Pass"] == True)
        & (df["Target_Free_Pass"] == True)
        & (df["Rule_Parse_Pass"] == True)
        & (df["R2_Target_Pass"] == True)
        & (df["Min_N_Pass"] == True)
        & (df["Min_Coverage_Pass"] == True)
    ].copy()

    if strict.empty:
        return None
    return strict.sort_values(["R2_Recomputed", "N_Retained_Recomputed"], ascending=[False, False]).iloc[0]


def retained_for_selected(retained_by_key: Dict[str, pd.DataFrame], selected: pd.Series) -> pd.DataFrame:
    key = f"32b::{selected['Policy_ID']}::{selected['Method']}::{selected['Rule_Name']}"
    return retained_by_key.get(key, pd.DataFrame()).copy()


# =============================================================================
# 5. Stability and dominance audits
# =============================================================================

def split_level_metrics(retained: pd.DataFrame) -> pd.DataFrame:
    if retained.empty:
        return pd.DataFrame()
    split_col = find_col(retained, ["Split_ID", "Seed_or_Fold", "Seed", "Fold"], required=False)
    obs_col = find_col(retained, ["Observed", "Actual", "y_true"], required=True)
    pred_col = find_col(retained, ["Predicted", "Prediction", "y_pred"], required=True)
    if split_col is None:
        return pd.DataFrame()

    rows = []
    for split, g in retained.groupby(split_col, dropna=False):
        md = regression_metrics(g, obs_col=obs_col, pred_col=pred_col)
        rows.append({
            "Split_ID": split,
            "R2": md["R2"],
            "RMSE": md["RMSE"],
            "MAE": md["MAE"],
            "N_Retained": md["N"],
            "R2_Target_Pass": bool(np.isfinite(md["R2"]) and md["R2"] >= R2_TARGET),
        })
    return pd.DataFrame(rows).sort_values("Split_ID").reset_index(drop=True)


def literature_group_dominance(retained: pd.DataFrame) -> pd.DataFrame:
    if retained.empty:
        return pd.DataFrame()
    group_col = find_col(retained, ["文献编号", "Literature_ID", "Paper_ID", "Source_Group", "Group"], required=False)
    if group_col is None:
        return pd.DataFrame()

    counts = retained[group_col].astype(str).value_counts(dropna=False)
    out = counts.rename_axis("Literature_Group").reset_index(name="N_Retained")
    out["Fraction_of_Retained"] = out["N_Retained"] / max(out["N_Retained"].sum(), 1)
    return out


def source_table_summary(audit_df: pd.DataFrame) -> pd.DataFrame:
    if audit_df.empty:
        return pd.DataFrame()
    rows = []
    for table_name, g in audit_df.groupby("Prediction_Table"):
        rows.append({
            "Prediction_Table": table_name,
            "N_Policies_Evaluated": len(g),
            "N_Exact_Reproduced": int(g["Exact_Reproduction_Pass"].sum()),
            "N_R2_Target_Pass": int(g["R2_Target_Pass"].sum()),
            "N_Manuscript_Ready": int(g["Manuscript_Ready_Selective_AD"].sum()),
            "Best_R2": pd.to_numeric(g["R2_Recomputed"], errors="coerce").max(),
            "Best_Exact_R2": pd.to_numeric(g.loc[g["Exact_Reproduction_Pass"] == True, "R2_Recomputed"], errors="coerce").max(),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 6. Plotting functions
# =============================================================================

def est_ax(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.set_facecolor("white")
    if grid_axis in {"both", "y"}:
        ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    if grid_axis in {"both", "x"}:
        ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.55, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.05)


def annotate_bars(ax: plt.Axes, fmt: str = "{:.2f}", fontsize: float = 8.8) -> None:
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 1.0
    for p in ax.patches:
        h = p.get_height()
        if not np.isfinite(h):
            continue
        x = p.get_x() + p.get_width() / 2
        if h >= 0:
            y = h + 0.012 * yr
            va = "bottom"
        else:
            y = h - 0.018 * yr
            va = "top"
        ax.text(x, y, fmt.format(h), ha="center", va=va, fontsize=fontsize)


def save_fig(fig: plt.Figure, filename: str) -> None:
    png = FIG_DIR / filename
    pdf = FIG_DIR / filename.replace(".png", ".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] Saved {png}")
    log(f"[FIG] Saved {pdf}")


def plot_policy_metrics(audit_32b: pd.DataFrame) -> None:
    df = audit_32b.copy()
    df = df.sort_values("R2_Recomputed", ascending=False).head(12).reset_index(drop=True)
    if df.empty:
        return

    labels = df["Policy_ID"].astype(str).tolist()
    colors = [COLOR_GREEN if bool(v) else COLOR_BLUE for v in df["Manuscript_Ready_Selective_AD"].tolist()]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    axes = axes.ravel()
    panels = [
        ("R2_Recomputed", r"Retained $R^2$", "{:.3f}"),
        ("RMSE_Recomputed", "Retained RMSE", "{:.2f}"),
        ("MAE_Recomputed", "Retained MAE", "{:.2f}"),
        ("Coverage_Within_Strategy_Method_Recomputed", "Coverage", "{:.3f}"),
    ]

    for ax, (col, title, fmt) in zip(axes, panels):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        ax.bar(labels, vals, color=colors, edgecolor="#333333", linewidth=0.8, width=0.72)
        ax.set_title(title)
        ax.set_xlabel("Policy ID")
        ax.set_ylabel(title.replace("Retained ", ""))
        est_ax(ax, grid_axis="y")
        annotate_bars(ax, fmt=fmt, fontsize=8.5)
        if col == "R2_Recomputed":
            ax.axhline(R2_TARGET, color=COLOR_RED, linestyle="--", linewidth=1.1, label=rf"$R^2={R2_TARGET:.1f}$")
            ax.legend(frameon=True, loc="best")

    fig.suptitle("32F2 reproducible selective AD policy audit", fontsize=17, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "179_32f2_reproducible_policy_metrics.png")


def plot_source_comparison(source_df: pd.DataFrame) -> None:
    if source_df.empty:
        return
    df = source_df.copy()
    labels = [s.replace("_", "\n") for s in df["Prediction_Table"].astype(str)]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    panels = [
        ("N_Exact_Reproduced", "Policies exactly reproduced", "{:.0f}"),
        ("N_Manuscript_Ready", "Manuscript-ready policies", "{:.0f}"),
        ("Best_Exact_R2", r"Best exactly reproduced $R^2$", "{:.3f}"),
    ]

    for ax, (col, title, fmt) in zip(axes, panels):
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(float)
        ax.bar(labels, vals, color=[COLOR_GREEN if i == 0 else COLOR_ORANGE for i in range(len(vals))], edgecolor="#333333", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(title)
        est_ax(ax, grid_axis="y")
        annotate_bars(ax, fmt=fmt, fontsize=8.5)
        ax.tick_params(axis="x", labelsize=8.5)

    fig.suptitle("Which prediction table reproduces Step 32D?", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "180_32f2_prediction_table_reproduction_check.png")


def plot_best_prediction_panel(retained: pd.DataFrame, selected: pd.Series) -> None:
    if retained.empty:
        return
    obs_col = find_col(retained, ["Observed", "Actual", "y_true"], required=True)
    pred_col = find_col(retained, ["Predicted", "Prediction", "y_pred"], required=True)

    y = numeric_array(retained[obs_col])
    p = numeric_array(retained[pred_col])
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    md = regression_metrics_from_arrays(y, p)

    fig, ax = plt.subplots(figsize=(7.3, 6.4))
    est_ax(ax, grid_axis="both")

    ax.scatter(y, p, s=46, alpha=0.84, color=COLOR_BLUE, edgecolor="white", linewidth=0.45)
    ax.plot([0, 100], [0, 100], linestyle="--", color=COLOR_RED, linewidth=1.35)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Predicted F removal (%)")
    ax.set_title(f"Audited retained subdomain: {selected['Method']}", fontweight="bold")

    txt = (
        rf"$R^2$ = {md['R2']:.3f}" + "\n"
        f"RMSE = {md['RMSE']:.2f}\n"
        f"MAE = {md['MAE']:.2f}\n"
        f"n = {md['N']}\n"
        f"coverage = {float(selected['Coverage_Within_Strategy_Method_Recomputed']):.3f}\n"
        f"policy = {selected['Policy_ID']}"
    )
    ax.text(0.06, 0.94, txt, transform=ax.transAxes, ha="left", va="top", fontsize=12.2,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#999999", alpha=0.92))

    fig.tight_layout()
    save_fig(fig, "181_32f2_best_reproduced_prediction_panel.png")


def plot_split_stability(split_df: pd.DataFrame) -> None:
    if split_df.empty:
        return
    df = split_df.copy().sort_values("Split_ID").reset_index(drop=True)
    df["Short_Split"] = df["Split_ID"].astype(str).str.replace("balanced5_", "", regex=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3))
    panels = [
        ("R2", r"$R^2$", "{:.2f}"),
        ("RMSE", "RMSE", "{:.2f}"),
        ("N_Retained", "N retained", "{:.0f}"),
    ]
    for ax, (col, ylabel, fmt) in zip(axes, panels):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        if col == "R2":
            colors = [COLOR_GREEN if np.isfinite(v) and v >= R2_TARGET else COLOR_ORANGE for v in vals]
        else:
            colors = [COLOR_BLUE for _ in vals]
        ax.bar(df["Short_Split"], vals, color=colors, edgecolor="#333333", linewidth=0.8, width=0.68)
        ax.set_title(ylabel)
        ax.set_xlabel("Grouped split")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=32)
        est_ax(ax, grid_axis="y")
        annotate_bars(ax, fmt=fmt, fontsize=8.6)
        if col == "R2":
            ax.axhline(R2_TARGET, color=COLOR_RED, linestyle="--", linewidth=1.1)

    fig.suptitle("Split-level stability of the audited retained subdomain", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "182_32f2_best_policy_split_stability.png")


def plot_literature_dominance(group_df: pd.DataFrame) -> None:
    if group_df.empty:
        return
    df = group_df.head(15).iloc[::-1].copy()

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.barh(df["Literature_Group"], df["N_Retained"], color=COLOR_BLUE, edgecolor="#333333", linewidth=0.8)
    ax.set_xlabel("Retained sample count")
    ax.set_ylabel("Literature group")
    ax.set_title("Literature-group dominance in retained subdomain", fontweight="bold")
    est_ax(ax, grid_axis="x")

    xmax = max(float(df["N_Retained"].max()), 1.0)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["N_Retained"] + 0.018 * xmax, i,
                f"{int(row['N_Retained'])} ({row['Fraction_of_Retained']:.1%})",
                va="center", ha="left", fontsize=9.0)
    ax.set_xlim(0, xmax * 1.18)
    fig.tight_layout()
    save_fig(fig, "183_32f2_retained_literature_group_dominance.png")


# =============================================================================
# 7. Report
# =============================================================================

def write_report(
    selected: Optional[pd.Series],
    audit_df: pd.DataFrame,
    source_df: pd.DataFrame,
    split_df: pd.DataFrame,
    group_df: pd.DataFrame,
    primary_path: Path,
    optional_path: Optional[Path],
    policy_path: Path,
) -> Path:
    lines: List[str] = []
    lines.append("32F2 reproducible selective AD policy audit report")
    lines.append("=" * 90)
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Policy table: {policy_path}")
    lines.append(f"Primary prediction table: {primary_path}")
    if optional_path is not None:
        lines.append(f"Optional comparison table: {optional_path}")
    lines.append("")

    lines.append("Source-table reproduction check")
    lines.append("-" * 90)
    if source_df.empty:
        lines.append("No source-table summary generated.")
    else:
        for _, r in source_df.iterrows():
            lines.append(
                f"{r['Prediction_Table']}: evaluated={int(r['N_Policies_Evaluated'])}, "
                f"exact_reproduced={int(r['N_Exact_Reproduced'])}, "
                f"R2_target_pass={int(r['N_R2_Target_Pass'])}, "
                f"manuscript_ready={int(r['N_Manuscript_Ready'])}, "
                f"best_exact_R2={r['Best_Exact_R2']:.6f}" if np.isfinite(r['Best_Exact_R2']) else
                f"{r['Prediction_Table']}: evaluated={int(r['N_Policies_Evaluated'])}, exact_reproduced={int(r['N_Exact_Reproduced'])}"
            )
    lines.append("")

    if selected is None:
        lines.append("Final decision")
        lines.append("-" * 90)
        lines.append("DO_NOT_ACCEPT: No target-free, exactly reproduced, R2>=0.5 selective AD policy was found.")
    else:
        lines.append("Selected reproducible policy")
        lines.append("-" * 90)
        lines.append(f"Policy ID: {selected['Policy_ID']}")
        lines.append(f"Strategy: {selected['Strategy']}")
        lines.append(f"Method: {selected['Method']}")
        lines.append(f"Rule name: {selected['Rule_Name']}")
        lines.append(f"Rule description: {selected.get('Rule_Description', '')}")
        lines.append(f"Used rule columns: {selected['Used_Rule_Columns']}")
        lines.append(f"Target-free pass: {bool(selected['Target_Free_Pass'])}")
        lines.append(f"Exact reproduction pass: {bool(selected['Exact_Reproduction_Pass'])}")
        lines.append(f"Rule parse pass: {bool(selected['Rule_Parse_Pass'])}")
        lines.append("")
        lines.append("Recomputed retained-subdomain metrics")
        lines.append("-" * 90)
        lines.append(f"R2: {selected['R2_Recomputed']:.6f}")
        lines.append(f"RMSE: {selected['RMSE_Recomputed']:.6f}")
        lines.append(f"MAE: {selected['MAE_Recomputed']:.6f}")
        lines.append(f"N retained: {int(selected['N_Retained_Recomputed'])}")
        lines.append(f"Coverage within strategy-method subset: {selected['Coverage_Within_Strategy_Method_Recomputed']:.6f}")
        lines.append("")
        lines.append("Split-level stability")
        lines.append("-" * 90)
        if split_df.empty:
            lines.append("No split-level stability table generated.")
        else:
            good = int((pd.to_numeric(split_df["R2"], errors="coerce") >= R2_TARGET).sum())
            total = int(split_df["R2"].notna().sum())
            min_n = int(pd.to_numeric(split_df["N_Retained"], errors="coerce").min())
            lines.append(f"Splits with R2 >= {R2_TARGET}: {good}/{total}")
            lines.append(f"Minimum retained N across splits: {min_n}")
            lines.append("Important: retained-subdomain R2 is pooled; split-level instability must be disclosed.")
        lines.append("")
        lines.append("Literature-group dominance")
        lines.append("-" * 90)
        if group_df.empty:
            lines.append("No literature-group dominance table generated.")
        else:
            top = group_df.iloc[0]
            lines.append(f"Largest retained literature group: {top['Literature_Group']} | n={int(top['N_Retained'])} | fraction={top['Fraction_of_Retained']:.3f}")
            lines.append("If one or two papers dominate, this result should be described as a high-confidence subdomain, not universal generalization.")
        lines.append("")
        lines.append("Final decision")
        lines.append("-" * 90)
        lines.append("ACCEPT_AS_SELECTIVE_AD_RESULT: The policy is target-free and exactly reproducible from 32b_predictions_with_reliability_zone.csv.")
        lines.append("Do NOT describe it as all-domain grouped-validation performance. Describe it as retained-subdomain performance under selective AD filtering.")

    lines.append("")
    lines.append("Generated outputs")
    lines.append("-" * 90)
    lines.append("Tables:")
    lines.append("  32f2_policy_reproduction_audit.csv")
    lines.append("  32f2_source_table_reproduction_summary.csv")
    lines.append("  32f2_best_reproduced_retained_predictions.csv")
    lines.append("  32f2_best_policy_split_stability.csv")
    lines.append("  32f2_best_policy_literature_group_dominance.csv")
    lines.append("  32f2_reproducible_selective_ad_policy_audit_report.txt")
    lines.append("Figures:")
    lines.append("  179_32f2_reproducible_policy_metrics.png/pdf")
    lines.append("  180_32f2_prediction_table_reproduction_check.png/pdf")
    lines.append("  181_32f2_best_reproduced_prediction_panel.png/pdf")
    lines.append("  182_32f2_best_policy_split_stability.png/pdf")
    lines.append("  183_32f2_retained_literature_group_dominance.png/pdf")

    out = TABLE_DIR / "32f2_reproducible_selective_ad_policy_audit_report.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"[REPORT] Saved {out}")
    return out


# =============================================================================
# 8. Main
# =============================================================================

def main() -> None:
    log("=" * 90)
    log(f"Running {SCRIPT_NAME}")
    log(f"Project root: {PROJECT_ROOT}")
    log("=" * 90)

    policy_df, policy_path = read_csv_required("32d_policy_label_mapping.csv")
    primary_pred, primary_path = read_csv_required("32b_predictions_with_reliability_zone.csv")
    optional_pred, optional_path = read_csv_optional("32c_fast_selected_holdout_predictions.csv")

    required_policy_cols = [
        "Policy_ID", "Rank", "Strategy", "Method", "Rule_Name", "N_Retained",
        "Coverage_Within_Strategy_Method", "R2", "RMSE", "MAE",
    ]
    missing = [c for c in required_policy_cols if c not in policy_df.columns]
    if missing:
        raise KeyError(f"32d_policy_label_mapping.csv is missing required columns: {missing}")

    log("[STEP] Recomputing 32D policy metrics from prediction tables...")
    audit_df, retained_by_key = evaluate_all_policies(policy_df, primary_pred, optional_pred)
    save_csv(audit_df, "32f2_policy_reproduction_audit.csv")

    source_df = source_table_summary(audit_df)
    save_csv(source_df, "32f2_source_table_reproduction_summary.csv")

    selected = select_best_reproduced_policy(audit_df)
    if selected is None:
        log("[DECISION] No strict reproduced high-R2 selective AD policy found.")
        retained = pd.DataFrame()
        split_df = pd.DataFrame()
        group_df = pd.DataFrame()
    else:
        log("[DECISION] Best strict reproduced selective AD policy found:")
        log(f"  Policy_ID = {selected['Policy_ID']}")
        log(f"  Method    = {selected['Method']}")
        log(f"  Strategy  = {selected['Strategy']}")
        log(f"  Rule      = {selected['Rule_Name']}")
        log(f"  R2/RMSE/MAE/N = {selected['R2_Recomputed']:.6f} / {selected['RMSE_Recomputed']:.6f} / {selected['MAE_Recomputed']:.6f} / {int(selected['N_Retained_Recomputed'])}")

        retained = retained_for_selected(retained_by_key, selected)
        split_df = split_level_metrics(retained)
        group_df = literature_group_dominance(retained)

    save_csv(retained, "32f2_best_reproduced_retained_predictions.csv")
    save_csv(split_df, "32f2_best_policy_split_stability.csv")
    save_csv(group_df, "32f2_best_policy_literature_group_dominance.csv")

    # Figures: all EST-like, uncluttered, and value-annotated.
    audit_32b = audit_df[audit_df["Prediction_Table"] == "32b_predictions_with_reliability_zone"].copy()
    plot_policy_metrics(audit_32b)
    plot_source_comparison(source_df)
    if selected is not None and not retained.empty:
        plot_best_prediction_panel(retained, selected)
    if not split_df.empty:
        plot_split_stability(split_df)
    if not group_df.empty:
        plot_literature_dominance(group_df)

    report_path = write_report(
        selected=selected,
        audit_df=audit_df,
        source_df=source_df,
        split_df=split_df,
        group_df=group_df,
        primary_path=primary_path,
        optional_path=optional_path,
        policy_path=policy_path,
    )

    log("")
    log("[KEY OUTPUTS]")
    log(f"  Audit table : {TABLE_DIR / '32f2_policy_reproduction_audit.csv'}")
    log(f"  Report      : {report_path}")
    log(f"  Figures dir : {FIG_DIR}")

    elapsed = time.time() - START_TIME
    log("=" * 90)
    log("32F2 reproducible selective AD policy audit finished.")
    log(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    log("=" * 90)


if __name__ == "__main__":
    main()
