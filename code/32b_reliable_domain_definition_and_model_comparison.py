# -*- coding: utf-8 -*-
"""
32B_reliable_domain_definition_and_model_comparison.py

Purpose
-------
Formalize a publication-ready reliable-domain (RD) definition after Step 32A2
corrected applicability-domain (AD) analysis.

What this script does
---------------------
1. Reuses Step 32A2 outputs directly; it does NOT retrain models and does NOT
   remove samples from all-domain reporting.
2. Defines a target-free reliable-domain policy from corrected AD components.
3. Reports all-domain, reliable-domain, warning-domain and outside-domain metrics
   side by side.
4. Compares model behavior across validation strategies, methods, RD rules and
   literature-level error-source types.
5. Produces manuscript/supplementary-ready figures and tables.

Strictness
----------
- No observed target values are used to decide whether a sample is in reliable,
  warning or outside domain.
- Test labels and test residuals are used only for downstream evaluation after
  the target-free domain labels have been assigned.
- 文献编号 is used only for literature-level diagnostics, not as a model input.
- All-domain metrics are always reported beside reliable-domain metrics.

Required input files from prior steps
-------------------------------------
outputs/tables/32a2_ad_reliable_rule_performance.csv
outputs/tables/32a2_ad_error_source_taxonomy.csv
outputs/tables/32a2_ad_score_error_correlation.csv
outputs/tables/32a2_ad_feature_space_audit.csv

Optional but strongly recommended
---------------------------------
outputs/tables/32a2_ad_predictions_with_scores.csv
outputs/tables/32a2_ad_sample_scores.csv
outputs/tables/31p_rmse_policy_summary.csv
outputs/tables/31p_rmse_policy_predictions.csv
outputs/tables/31n_grouped_validation_summary.csv
outputs/tables/31n_grouped_validation_paper_macro_summary.csv

Outputs
-------
Figures:
152_32b_rd_rule_selection_overview.png
153_32b_all_vs_reliable_domain_metrics.png
154_32b_reliability_zone_metrics.png
155_32b_strategy_sensitivity_heatmap.png
156_32b_paper_error_source_taxonomy.png
157_32b_reliable_domain_prediction_panels.png

Note: SAVE_PDF is False by default because CJK-font PDF rendering can be slow in PyCharm. Set SAVE_PDF=True near the top only when vector PDF export is needed.

Tables:
32b_rd_rule_catalog.csv
32b_rule_selection_by_strategy_method.csv
32b_all_vs_reliable_domain_summary.csv
32b_reliability_zone_metrics.csv
32b_strategy_model_recommendation.csv
32b_paper_error_source_summary.csv
32b_final_rd_decision_report.txt
"""

from __future__ import annotations

import os
import re
import sys
import time
import math
import textwrap
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for metric calculation.") from exc


# =============================================================================
# Basic configuration
# =============================================================================

SCRIPT_NAME = "32b_reliable_domain_definition_and_model_comparison.py"
START_TIME = time.time()

FIG_DPI = 300
SAVE_PDF = False  # PNG is used by default to avoid slow CJK-font PDF rendering in PyCharm.
RANDOM_NOTE = "No random model training is performed in Step 32B."

# Publication-style, low-saturation colors.
COLORS = {
    "red": "#D55E5E",
    "orange": "#DFA35D",
    "green": "#86B88F",
    "teal": "#72B7B2",
    "blue": "#78A6C8",
    "purple": "#B28ABF",
    "gray": "#8E8E8E",
    "darkgray": "#4A4A4A",
    "lightgray": "#E8E8E8",
    "grid": "#DDDDDD",
    "reliable": "#86B88F",
    "warning": "#DFA35D",
    "outside": "#D55E5E",
}

METHOD_COLORS = {
    "CB-XGB": "#8E8E8E",
    "CatBoost": "#B28ABF",
    "XGBoost": "#DFA35D",
    "KRR_Poly": "#86B88F",
    "KRR_CatBoost_Weighted_2Model": "#78A6C8",
    "KRR_XGBoost_Weighted_2Model": "#9CC3D5",
    "ValidationRMSE_Selected_Model": "#D55E5E",
    "StrategyBestFixedByValidation": "#72B7B2",
    "PLS": "#A6A6A6",
    "GPR_Matern": "#C08CA0",
}

METHOD_ORDER = [
    "CB-XGB",
    "CatBoost",
    "XGBoost",
    "KRR_Poly",
    "KRR_CatBoost_Weighted_2Model",
    "KRR_XGBoost_Weighted_2Model",
    "ValidationRMSE_Selected_Model",
    "StrategyBestFixedByValidation",
    "PLS",
    "GPR_Matern",
]

STRATEGY_ORDER = [
    "Balanced_5Seed_Current31F",
    "RepeatedBalanced_Extended",
    "RepeatedOrdinary_GroupedSplit",
    "GroupKFold_FullyGrouped",
    "LOLO_LeaveOneLiteratureOut",
]

ZONE_ORDER = ["Reliable", "Warning", "Outside"]
ZONE_LABELS = {
    "Reliable": "Reliable domain",
    "Warning": "Warning domain",
    "Outside": "Outside domain",
}

RULE_ORDER = [
    "AllDomain_NoFiltering",
    "Rule_A_strict_component",
    "Rule_B_balanced_component",
    "Rule_C_distance_focused",
    "Rule_D_boundary_focused",
    "Rule_E_loose_screen",
]

RULE_LABELS = {
    "AllDomain_NoFiltering": "All domain",
    "Rule_A_strict_component": "A: strict component",
    "Rule_B_balanced_component": "B: balanced component",
    "Rule_C_distance_focused": "C: distance focused",
    "Rule_D_boundary_focused": "D: boundary focused",
    "Rule_E_loose_screen": "E: loose screen",
}

# 32B publication policy. These are not learned from the test labels. They are
# set as transparent, target-free reporting rules after 32A2 showed that one
# scalar corrected AD risk alone should not be used as a hard error predictor.
RD_POLICY = {
    "Reliable": {
        "Corrected_AD_Risk_max": 0.60,
        "Boundary_AD_Risk_max": 0.35,
        "Categorical_Unseen_Fraction_max": 0.20,
        "Numeric_Strict_Range_Violation_Fraction_max": 0.05,
    },
    "Warning": {
        "Corrected_AD_Risk_max": 0.75,
        "Boundary_AD_Risk_max": 0.55,
        "Categorical_Unseen_Fraction_max": 0.35,
        "Numeric_Strict_Range_Violation_Fraction_max": 0.10,
    },
}


# =============================================================================
# Paths and logging
# =============================================================================

def find_project_root() -> Path:
    """Find project root robustly from src or project-root execution."""
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, Path.cwd(), Path.cwd().parent]
    for cand in candidates:
        if (cand / "outputs" / "tables").exists() and (cand / "outputs" / "figures").exists():
            return cand
        if (cand / "data").exists() and (cand / "src").exists():
            return cand
    # Fallback: if script is in src, parent is project root.
    if here.parent.name.lower() == "src":
        return here.parent.parent
    return Path.cwd()

ROOT = find_project_root()
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "reports"
for d in [TABLE_DIR, FIG_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def find_table(filename: str, required: bool = True) -> Optional[Path]:
    """Find a table first in outputs/tables, then project tree, then cwd."""
    search_dirs = [TABLE_DIR, ROOT, ROOT / "src", Path.cwd(), Path.cwd() / "outputs" / "tables"]
    for d in search_dirs:
        p = d / filename
        if p.exists():
            return p
    matches = list(ROOT.rglob(filename)) if ROOT.exists() else []
    if matches:
        # Prefer outputs/tables if there are duplicate backup files.
        matches = sorted(matches, key=lambda x: ("outputs" not in str(x), len(str(x))))
        return matches[0]
    if required:
        raise FileNotFoundError(
            f"Cannot find required table: {filename}\n"
            f"Checked root: {ROOT}\n"
            f"Expected location: {TABLE_DIR / filename}"
        )
    return None


def read_csv_safely(filename: str, required: bool = True) -> Optional[pd.DataFrame]:
    p = find_table(filename, required=required)
    if p is None:
        log(f"[SKIP] Optional table not found: {filename}")
        return None
    df = pd.read_csv(p, low_memory=False)
    log(f"[TABLE] {p} | shape={df.shape}")
    return df


# =============================================================================
# Font, text and figure helpers
# =============================================================================

def configure_matplotlib() -> None:
    font_candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS",
        "Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in font_candidates if f in installed), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": chosen,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.8,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "axes.labelcolor": "#222222",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    log(f"[INFO] Matplotlib font set to: {chosen}")


SUP_SUB_MAP = str.maketrans({
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "−": "-", "·": ".", "Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III",
})


def sanitize_text(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).translate(SUP_SUB_MAP)
    s = s.replace("SO4^2-", "SO4-2").replace("CO3^2-", "CO3-2")
    s = s.replace("PO4^3-", "PO4-3").replace("Ca^2+", "Ca2+")
    return s


def short_label(x, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(sanitize_text(x), width=width, break_long_words=False))


def add_bar_labels(ax, bars, fmt: str = "{:.2f}", fontsize: int = 7, rotation: int = 0) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    for b in bars:
        h = b.get_height()
        if not np.isfinite(h):
            continue
        if h >= 0:
            y = h + span * 0.015
            va = "bottom"
        else:
            y = h - span * 0.015
            va = "top"
        ax.text(
            b.get_x() + b.get_width() / 2,
            y,
            fmt.format(h),
            ha="center",
            va=va,
            fontsize=fontsize,
            rotation=rotation,
        )


def apply_axis_style(ax) -> None:
    ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(axis="both", labelsize=8)


def save_fig(fig: plt.Figure, filename: str) -> None:
    png = FIG_DIR / filename
    pdf = FIG_DIR / filename.replace(".png", ".pdf")
    # Avoid repeated tight_layout optimization on dense CJK/long-label figures,
    # which can be very slow in some PyCharm/Matplotlib builds. Individual plot
    # functions already allocate readable figure sizes.
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.13, top=0.88, wspace=0.35, hspace=0.55)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Glyph .* missing from font.*", category=UserWarning)
        fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
        if SAVE_PDF:
            fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] {png}")
    if SAVE_PDF:
        log(f"[FIG] {pdf}")


def save_table(df: pd.DataFrame, filename: str) -> Path:
    path = TABLE_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {path} | shape={df.shape}")
    return path


# =============================================================================
# Metrics and robust column utilities
# =============================================================================

def metric_values(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "Bias": np.nan, "n": 0}
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    if len(y_true) < 2 or np.nanstd(y_true) < 1e-12:
        r2 = np.nan
    else:
        r2 = r2_score(y_true, y_pred)
    return {"R2": float(r2), "RMSE": float(rmse), "MAE": float(mae), "MSE": float(mse), "Bias": bias, "n": int(len(y_true))}


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def require_cols(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}\nAvailable columns: {list(df.columns)}")


def method_sort_key(method: str) -> Tuple[int, str]:
    return (METHOD_ORDER.index(method) if method in METHOD_ORDER else 999, method)


def strategy_sort_key(strategy: str) -> Tuple[int, str]:
    return (STRATEGY_ORDER.index(strategy) if strategy in STRATEGY_ORDER else 999, strategy)


def ordered_unique(values: Iterable[str], preferred: Sequence[str]) -> List[str]:
    vals = [v for v in preferred if v in set(values)]
    vals += sorted([v for v in set(values) if v not in set(vals)])
    return vals


# =============================================================================
# Reliable-domain rule definition
# =============================================================================

def build_rule_catalog() -> pd.DataFrame:
    rows = [
        {
            "Rule_ID": "AllDomain_NoFiltering",
            "Publication_Name": "All-domain report",
            "Domain_Label": "AllDomain",
            "Corrected_AD_Risk_max": np.nan,
            "Boundary_AD_Risk_max": np.nan,
            "Categorical_Unseen_Fraction_max": np.nan,
            "Numeric_Strict_Range_Violation_Fraction_max": np.nan,
            "Use_in_32B": "mandatory_reference",
            "Interpretation": "All grouped-test samples are retained. This is the primary generalization benchmark.",
        },
        {
            "Rule_ID": "Rule_A_strict_component",
            "Publication_Name": "Strict reliable domain",
            "Domain_Label": "Reliable",
            "Corrected_AD_Risk_max": 0.50,
            "Boundary_AD_Risk_max": 0.25,
            "Categorical_Unseen_Fraction_max": 0.10,
            "Numeric_Strict_Range_Violation_Fraction_max": 0.02,
            "Use_in_32B": "sensitivity_only",
            "Interpretation": "Very conservative target-free domain; often too low in coverage for main reporting.",
        },
        {
            "Rule_ID": "Rule_B_balanced_component",
            "Publication_Name": "Balanced reliable domain",
            "Domain_Label": "Reliable",
            "Corrected_AD_Risk_max": 0.60,
            "Boundary_AD_Risk_max": 0.35,
            "Categorical_Unseen_Fraction_max": 0.20,
            "Numeric_Strict_Range_Violation_Fraction_max": 0.05,
            "Use_in_32B": "main_predefined_rule",
            "Interpretation": "Main pre-specified target-free RD rule: moderate coverage with multiple component constraints.",
        },
        {
            "Rule_ID": "Rule_C_distance_focused",
            "Publication_Name": "Distance-focused reliable domain",
            "Domain_Label": "Reliable",
            "Corrected_AD_Risk_max": 0.55,
            "Boundary_AD_Risk_max": np.nan,
            "Categorical_Unseen_Fraction_max": 0.20,
            "Numeric_Strict_Range_Violation_Fraction_max": 0.05,
            "Use_in_32B": "sensitivity_only",
            "Interpretation": "Focused on distance/leverage components; can reduce RMSE but tends to retain too few samples.",
        },
        {
            "Rule_ID": "Rule_D_boundary_focused",
            "Publication_Name": "Boundary-warning domain",
            "Domain_Label": "Warning",
            "Corrected_AD_Risk_max": 0.75,
            "Boundary_AD_Risk_max": 0.55,
            "Categorical_Unseen_Fraction_max": 0.35,
            "Numeric_Strict_Range_Violation_Fraction_max": 0.10,
            "Use_in_32B": "warning_domain_rule",
            "Interpretation": "Intermediate warning domain; predictions are allowed but should be flagged as less certain.",
        },
        {
            "Rule_ID": "Rule_E_loose_screen",
            "Publication_Name": "Loose screening domain",
            "Domain_Label": "Warning",
            "Corrected_AD_Risk_max": 0.80,
            "Boundary_AD_Risk_max": 0.65,
            "Categorical_Unseen_Fraction_max": 0.50,
            "Numeric_Strict_Range_Violation_Fraction_max": 0.15,
            "Use_in_32B": "sensitivity_only",
            "Interpretation": "Loose screen; useful for sensitivity analysis but not strict enough as the main RD.",
        },
    ]
    return pd.DataFrame(rows)


def choose_primary_rule(rule_perf: pd.DataFrame) -> str:
    # We explicitly use the predefined balanced rule as the main RD rule when available.
    if "Rule_B_balanced_component" in set(rule_perf["Rule"].astype(str)):
        return "Rule_B_balanced_component"
    # Fallback: choose rule nearest 0.4-0.6 average coverage, not from target labels.
    non_all = rule_perf[rule_perf["Rule"].astype(str) != "AllDomain_NoFiltering"].copy()
    coverage_by_rule = non_all.groupby("Rule")["Coverage_Fraction"].mean().reset_index()
    coverage_by_rule["Coverage_Distance_To_0p5"] = (coverage_by_rule["Coverage_Fraction"] - 0.5).abs()
    return str(coverage_by_rule.sort_values("Coverage_Distance_To_0p5").iloc[0]["Rule"])


def choose_warning_rule(rule_perf: pd.DataFrame) -> str:
    if "Rule_D_boundary_focused" in set(rule_perf["Rule"].astype(str)):
        return "Rule_D_boundary_focused"
    non_all = rule_perf[rule_perf["Rule"].astype(str) != "AllDomain_NoFiltering"].copy()
    coverage_by_rule = non_all.groupby("Rule")["Coverage_Fraction"].mean().reset_index()
    coverage_by_rule["Coverage_Distance_To_0p6"] = (coverage_by_rule["Coverage_Fraction"] - 0.6).abs()
    return str(coverage_by_rule.sort_values("Coverage_Distance_To_0p6").iloc[0]["Rule"])


def prepare_rule_selection(rule_perf: pd.DataFrame, main_rule: str, warning_rule: str) -> pd.DataFrame:
    require_cols(rule_perf, ["Strategy", "Method", "Rule", "Coverage_Fraction", "R2", "RMSE", "MAE", "MSE", "Bias"], "32a2_ad_reliable_rule_performance")
    rows = []
    for (strategy, method), sub in rule_perf.groupby(["Strategy", "Method"], dropna=False):
        sub = sub.copy()
        all_row = sub[sub["Rule"] == "AllDomain_NoFiltering"]
        main_row = sub[sub["Rule"] == main_rule]
        warning_row = sub[sub["Rule"] == warning_rule]
        if all_row.empty:
            continue
        a = all_row.iloc[0]
        if not main_row.empty:
            m = main_row.iloc[0]
            rows.append({
                "Strategy": strategy,
                "Method": method,
                "Selected_Rule_Role": "ReliableDomain_MainRule",
                "Selected_Rule": main_rule,
                "AllDomain_R2": a["R2"],
                "AllDomain_RMSE": a["RMSE"],
                "AllDomain_MAE": a["MAE"],
                "Reliable_R2": m["R2"],
                "Reliable_RMSE": m["RMSE"],
                "Reliable_MAE": m["MAE"],
                "Reliable_MSE": m["MSE"],
                "Reliable_Bias": m["Bias"],
                "Coverage_Fraction": m["Coverage_Fraction"],
                "n_in_domain": m.get("n_in_domain", np.nan),
                "RMSE_Change_vs_AllDomain": m["RMSE"] - a["RMSE"],
                "MAE_Change_vs_AllDomain": m["MAE"] - a["MAE"],
                "R2_Change_vs_AllDomain": m["R2"] - a["R2"],
            })
        if not warning_row.empty:
            w = warning_row.iloc[0]
            rows.append({
                "Strategy": strategy,
                "Method": method,
                "Selected_Rule_Role": "WarningDomain_SecondaryRule",
                "Selected_Rule": warning_rule,
                "AllDomain_R2": a["R2"],
                "AllDomain_RMSE": a["RMSE"],
                "AllDomain_MAE": a["MAE"],
                "Reliable_R2": w["R2"],
                "Reliable_RMSE": w["RMSE"],
                "Reliable_MAE": w["MAE"],
                "Reliable_MSE": w["MSE"],
                "Reliable_Bias": w["Bias"],
                "Coverage_Fraction": w["Coverage_Fraction"],
                "n_in_domain": w.get("n_in_domain", np.nan),
                "RMSE_Change_vs_AllDomain": w["RMSE"] - a["RMSE"],
                "MAE_Change_vs_AllDomain": w["MAE"] - a["MAE"],
                "R2_Change_vs_AllDomain": w["R2"] - a["R2"],
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Strategy_Order"] = out["Strategy"].map(lambda x: strategy_sort_key(str(x))[0])
        out["Method_Order"] = out["Method"].map(lambda x: method_sort_key(str(x))[0])
        out = out.sort_values(["Selected_Rule_Role", "Strategy_Order", "Method_Order"]).drop(columns=["Strategy_Order", "Method_Order"])
    return out


# =============================================================================
# Optional sample-level domain-zone assignment
# =============================================================================

def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    y_col = first_existing_col(df, ["Observed", "y_true", "True", "Actual", "F_removal", "氟去除率"])
    p_col = first_existing_col(df, ["Predicted", "y_pred", "Prediction", "Pred"])
    if y_col and y_col != "Observed":
        rename_map[y_col] = "Observed"
    if p_col and p_col != "Predicted":
        rename_map[p_col] = "Predicted"
    df = df.rename(columns=rename_map)
    if "Residual" not in df.columns and {"Observed", "Predicted"}.issubset(df.columns):
        df["Residual"] = df["Observed"] - df["Predicted"]
    return df


def load_prediction_with_scores() -> Optional[pd.DataFrame]:
    candidates = [
        "32a2_ad_predictions_with_scores.csv",
        "32a_ad_predictions_with_scores.csv",
        "31p_rmse_policy_predictions.csv",
        "31n_grouped_validation_predictions.csv",
    ]
    for name in candidates:
        df = read_csv_safely(name, required=False)
        if df is None:
            continue
        df = normalize_prediction_columns(df)
        if {"Strategy", "Method", "Observed", "Predicted"}.issubset(df.columns):
            if "Corrected_AD_Risk" in df.columns or "AD_Risk_Score" in df.columns:
                log(f"[INFO] Using prediction-with-score table: {name}")
                return df
            if name.startswith("31"):
                log(f"[INFO] Prediction table {name} found but no AD score columns; zone-specific metrics will be limited.")
                return df
    log("[INFO] No sample-level prediction-with-score table found. 32B will use rule-performance tables only.")
    return None


def classify_domain_zone(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Harmonize component column names.
    aliases = {
        "Corrected_AD_Risk": ["Corrected_AD_Risk", "AD_Risk_Score", "Mean_Corrected_AD_Risk"],
        "Boundary_AD_Risk": ["Boundary_AD_Risk", "Mean_Boundary_AD_Risk"],
        "Categorical_Unseen_Fraction": ["Categorical_Unseen_Fraction", "Mean_Categorical_Unseen_Fraction"],
        "Numeric_Strict_Range_Violation_Fraction": [
            "Numeric_Strict_Range_Violation_Fraction",
            "Mean_Numeric_Strict_Range_Violation_Fraction",
            "Range_Violation_Fraction",
            "Numeric_Range_Violation_Fraction",
        ],
    }
    for target, opts in aliases.items():
        col = first_existing_col(df, opts)
        if col and col != target:
            df[target] = df[col]
        elif col is None:
            df[target] = 0.0 if target in ["Categorical_Unseen_Fraction", "Numeric_Strict_Range_Violation_Fraction"] else np.nan
    if df["Corrected_AD_Risk"].isna().all():
        df["Reliability_Zone"] = "Unknown"
        return df
    # Missing component scores are treated conservatively when possible.
    for c in ["Boundary_AD_Risk", "Categorical_Unseen_Fraction", "Numeric_Strict_Range_Violation_Fraction"]:
        if df[c].isna().all():
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(df[c].median())

    rel = RD_POLICY["Reliable"]
    warn = RD_POLICY["Warning"]
    reliable_mask = (
        (df["Corrected_AD_Risk"] <= rel["Corrected_AD_Risk_max"]) &
        (df["Boundary_AD_Risk"] <= rel["Boundary_AD_Risk_max"]) &
        (df["Categorical_Unseen_Fraction"] <= rel["Categorical_Unseen_Fraction_max"]) &
        (df["Numeric_Strict_Range_Violation_Fraction"] <= rel["Numeric_Strict_Range_Violation_Fraction_max"])
    )
    warning_mask = (
        (df["Corrected_AD_Risk"] <= warn["Corrected_AD_Risk_max"]) &
        (df["Boundary_AD_Risk"] <= warn["Boundary_AD_Risk_max"]) &
        (df["Categorical_Unseen_Fraction"] <= warn["Categorical_Unseen_Fraction_max"]) &
        (df["Numeric_Strict_Range_Violation_Fraction"] <= warn["Numeric_Strict_Range_Violation_Fraction_max"])
    )
    df["Reliability_Zone"] = np.where(reliable_mask, "Reliable", np.where(warning_mask, "Warning", "Outside"))
    return df


def compute_zone_metrics(pred_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if pred_df is None or not {"Observed", "Predicted", "Reliability_Zone"}.issubset(pred_df.columns):
        return pd.DataFrame()
    rows = []
    total_by_sm = pred_df.groupby(["Strategy", "Method"]).size().rename("n_total").reset_index()
    for (strategy, method, zone), sub in pred_df.groupby(["Strategy", "Method", "Reliability_Zone"], dropna=False):
        m = metric_values(sub["Observed"], sub["Predicted"])
        n_total = int(total_by_sm[(total_by_sm["Strategy"] == strategy) & (total_by_sm["Method"] == method)]["n_total"].iloc[0])
        rows.append({
            "Strategy": strategy,
            "Method": method,
            "Reliability_Zone": zone,
            "Coverage_Fraction": m["n"] / max(n_total, 1),
            "n_zone": m["n"],
            "R2": m["R2"],
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
            "MSE": m["MSE"],
            "Bias": m["Bias"],
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Zone_Order"] = out["Reliability_Zone"].map(lambda z: ZONE_ORDER.index(z) if z in ZONE_ORDER else 99)
        out["Strategy_Order"] = out["Strategy"].map(lambda x: strategy_sort_key(str(x))[0])
        out["Method_Order"] = out["Method"].map(lambda x: method_sort_key(str(x))[0])
        out = out.sort_values(["Strategy_Order", "Method_Order", "Zone_Order"]).drop(columns=["Strategy_Order", "Method_Order", "Zone_Order"])
    return out


# =============================================================================
# Decision logic and paper-level summaries
# =============================================================================

def summarize_strategy_recommendations(rule_perf: pd.DataFrame, rule_selection: pd.DataFrame, zone_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strategies = ordered_unique(rule_perf["Strategy"].astype(str).unique(), STRATEGY_ORDER)
    for strategy in strategies:
        sub_all = rule_perf[(rule_perf["Strategy"] == strategy) & (rule_perf["Rule"] == "AllDomain_NoFiltering")].copy()
        if sub_all.empty:
            continue
        sub_all = sub_all.sort_values("RMSE")
        best_all = sub_all.iloc[0]
        sub_rel = rule_selection[(rule_selection["Strategy"] == strategy) & (rule_selection["Selected_Rule_Role"] == "ReliableDomain_MainRule")].copy()
        if not sub_rel.empty:
            sub_rel = sub_rel.sort_values("Reliable_RMSE")
            best_rel = sub_rel.iloc[0]
        else:
            best_rel = None
        # Conservative rule: main recommendation remains all-domain best unless reliable-domain rule is stable and coverage is usable.
        if best_rel is not None:
            rmse_gain = float(best_all["RMSE"] - best_rel["Reliable_RMSE"])
            coverage = float(best_rel["Coverage_Fraction"])
            if rmse_gain >= 0.5 and coverage >= 0.35:
                decision = "Report reliable-domain improvement beside all-domain; do not replace all-domain main metric."
            elif coverage < 0.35:
                decision = "Reliable-domain rule retains too few samples; use as sensitivity only."
            else:
                decision = "Reliable-domain rule is diagnostic, not an improvement claim."
            rel_method = best_rel["Method"]
            rel_rmse = best_rel["Reliable_RMSE"]
            rel_mae = best_rel["Reliable_MAE"]
            rel_r2 = best_rel["Reliable_R2"]
        else:
            rmse_gain = np.nan
            coverage = np.nan
            rel_method = np.nan
            rel_rmse = np.nan
            rel_mae = np.nan
            rel_r2 = np.nan
            decision = "No reliable-domain rule available."
        rows.append({
            "Strategy": strategy,
            "Best_AllDomain_Method": best_all["Method"],
            "Best_AllDomain_R2": best_all["R2"],
            "Best_AllDomain_RMSE": best_all["RMSE"],
            "Best_AllDomain_MAE": best_all["MAE"],
            "Best_ReliableDomain_Method": rel_method,
            "Best_ReliableDomain_R2": rel_r2,
            "Best_ReliableDomain_RMSE": rel_rmse,
            "Best_ReliableDomain_MAE": rel_mae,
            "ReliableDomain_Coverage": coverage,
            "ReliableDomain_RMSE_Gain_vs_BestAll": rmse_gain,
            "32B_Recommendation": decision,
        })
    return pd.DataFrame(rows)


def summarize_paper_error_sources(error_tax: pd.DataFrame) -> pd.DataFrame:
    require_cols(error_tax, ["Strategy", "Method", "文献编号", "Paper_RMSE", "Paper_MAE", "n_predictions", "Error_Source_Tags_Diagnostic"], "32a2_ad_error_source_taxonomy")
    df = error_tax.copy()
    tag_rows = []
    for _, row in df.iterrows():
        tags = str(row.get("Error_Source_Tags_Diagnostic", "")).split(";")
        tags = [t.strip() for t in tags if t.strip() and t.strip().lower() != "nan"]
        if not tags:
            tags = ["unclassified"]
        for tag in tags:
            r = row.to_dict()
            r["Error_Source_Tag"] = tag
            tag_rows.append(r)
    tag_df = pd.DataFrame(tag_rows)
    summary = tag_df.groupby(["Strategy", "Method", "Error_Source_Tag"], dropna=False).agg(
        n_paper_entries=("文献编号", "count"),
        n_unique_papers=("文献编号", "nunique"),
        Mean_Paper_RMSE=("Paper_RMSE", "mean"),
        Median_Paper_RMSE=("Paper_RMSE", "median"),
        Mean_Paper_MAE=("Paper_MAE", "mean"),
        Total_Predictions=("n_predictions", "sum"),
    ).reset_index()
    summary = summary.sort_values(["Strategy", "Method", "Mean_Paper_RMSE"], ascending=[True, True, False])
    return summary


# =============================================================================
# Plot functions
# =============================================================================

def plot_rule_selection_overview(rule_perf: pd.DataFrame, main_rule: str, warning_rule: str) -> None:
    df = rule_perf.copy()
    # Restrict to main methods to keep figure readable.
    methods = [m for m in METHOD_ORDER if m in set(df["Method"])]
    if not methods:
        methods = sorted(df["Method"].unique())[:6]
    df = df[df["Method"].isin(methods)].copy()
    agg = df.groupby("Rule", dropna=False).agg(
        Mean_Coverage=("Coverage_Fraction", "mean"),
        Mean_RMSE=("RMSE", "mean"),
        Mean_MAE=("MAE", "mean"),
        Mean_R2=("R2", "mean"),
    ).reset_index()
    agg["Rule_Order"] = agg["Rule"].map(lambda r: RULE_ORDER.index(r) if r in RULE_ORDER else 99)
    agg = agg.sort_values("Rule_Order")
    x = np.arange(len(agg))
    labels = [short_label(RULE_LABELS.get(r, r), 14) for r in agg["Rule"]]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    fig.suptitle("32B reliable-domain rule selection overview", fontsize=15, fontweight="bold")

    specs = [
        ("Mean_Coverage", "Mean coverage", "{:.2f}", "max"),
        ("Mean_RMSE", "Mean RMSE", "{:.2f}", "min"),
        ("Mean_MAE", "Mean MAE", "{:.2f}", "min"),
    ]
    for ax, (col, ylabel, fmt, _) in zip(axes, specs):
        vals = agg[col].astype(float).to_numpy()
        colors = []
        for r in agg["Rule"]:
            if r == main_rule:
                colors.append(COLORS["green"])
            elif r == warning_rule:
                colors.append(COLORS["orange"])
            elif r == "AllDomain_NoFiltering":
                colors.append(COLORS["gray"])
            else:
                colors.append(COLORS["blue"])
        bars = ax.bar(x, vals, color=colors, edgecolor="#333333", linewidth=0.5, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        apply_axis_style(ax)
        if col == "Mean_Coverage":
            ax.set_ylim(0, 1.08)
            ax.axhline(0.35, color=COLORS["darkgray"], linestyle="--", linewidth=0.8, alpha=0.7)
            ax.text(0.02, 0.37, "minimum useful coverage", transform=ax.get_yaxis_transform(), fontsize=7, color=COLORS["darkgray"])
        else:
            ax.set_ylim(0, max(vals) * 1.18 if len(vals) else 1)
        add_bar_labels(ax, bars, fmt=fmt, fontsize=7)
    fig.text(0.5, 0.005, "Green = predefined main reliable-domain rule; orange = warning-domain rule; gray = all-domain reference.", ha="center", fontsize=8)
    save_fig(fig, "152_32b_rd_rule_selection_overview.png")


def plot_all_vs_reliable(rule_selection: pd.DataFrame, main_rule: str) -> None:
    df = rule_selection[rule_selection["Selected_Rule_Role"] == "ReliableDomain_MainRule"].copy()
    if df.empty:
        log("[SKIP] plot_all_vs_reliable: no main reliable rule rows.")
        return
    # Use key methods; select available.
    methods = [m for m in ["CB-XGB", "CatBoost", "XGBoost", "KRR_Poly", "KRR_CatBoost_Weighted_2Model"] if m in set(df["Method"])]
    df = df[df["Method"].isin(methods)]
    strategies = ordered_unique(df["Strategy"].astype(str).unique(), STRATEGY_ORDER)

    fig, axes = plt.subplots(len(strategies), 3, figsize=(14.8, max(3.2 * len(strategies), 5.5)), squeeze=False)
    fig.suptitle("32B all-domain vs predefined reliable-domain model comparison", fontsize=15, fontweight="bold")
    metric_specs = [
        ("R2", "R2", "{:.2f}", "max"),
        ("RMSE", "RMSE", "{:.2f}", "min"),
        ("MAE", "MAE", "{:.2f}", "min"),
    ]
    for i, strategy in enumerate(strategies):
        sub = df[df["Strategy"] == strategy].copy()
        sub["Method_Order"] = sub["Method"].map(lambda m: method_sort_key(str(m))[0])
        sub = sub.sort_values("Method_Order")
        x = np.arange(len(sub))
        width = 0.36
        labels = [short_label(m, 12) for m in sub["Method"]]
        for j, (metric, ylabel, fmt, rule) in enumerate(metric_specs):
            ax = axes[i, j]
            all_vals = sub[f"AllDomain_{metric}"].astype(float).to_numpy()
            rel_vals = sub[f"Reliable_{metric}"].astype(float).to_numpy()
            bars1 = ax.bar(x - width/2, all_vals, width, color=COLORS["gray"], edgecolor="#333333", linewidth=0.4, label="All domain")
            bars2 = ax.bar(x + width/2, rel_vals, width, color=COLORS["green"], edgecolor="#333333", linewidth=0.4, label="Reliable domain")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{short_label(strategy, 28)}\n{ylabel}", fontsize=10)
            apply_axis_style(ax)
            vals = np.concatenate([all_vals[np.isfinite(all_vals)], rel_vals[np.isfinite(rel_vals)]])
            if len(vals):
                if metric == "R2":
                    lo = min(-0.4, float(np.nanmin(vals)) - 0.08)
                    hi = max(0.25, float(np.nanmax(vals)) + 0.08)
                    ax.set_ylim(lo, hi)
                    ax.axhline(0, color="#333333", linewidth=0.8)
                else:
                    ax.set_ylim(0, float(np.nanmax(vals)) * 1.20)
            add_bar_labels(ax, bars1, fmt=fmt, fontsize=6, rotation=90 if metric == "R2" else 0)
            add_bar_labels(ax, bars2, fmt=fmt, fontsize=6, rotation=90 if metric == "R2" else 0)
            if i == 0 and j == 2:
                ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(0.5, 0.005, f"Reliable domain uses predefined {RULE_LABELS.get(main_rule, main_rule)}; all-domain metrics remain the mandatory benchmark.", ha="center", fontsize=8)
    save_fig(fig, "153_32b_all_vs_reliable_domain_metrics.png")


def plot_zone_metrics(zone_metrics: pd.DataFrame) -> None:
    if zone_metrics.empty:
        log("[SKIP] plot_zone_metrics: no sample-level zone metrics.")
        return
    # Select a compact set: best available methods under LOLO/GroupKFold if present.
    methods = [m for m in ["CB-XGB", "CatBoost", "XGBoost", "KRR_Poly", "KRR_CatBoost_Weighted_2Model"] if m in set(zone_metrics["Method"])]
    strategies = [s for s in ["LOLO_LeaveOneLiteratureOut", "GroupKFold_FullyGrouped", "Balanced_5Seed_Current31F"] if s in set(zone_metrics["Strategy"])]
    if not strategies:
        strategies = ordered_unique(zone_metrics["Strategy"].astype(str).unique(), STRATEGY_ORDER)[:3]
    df = zone_metrics[zone_metrics["Method"].isin(methods) & zone_metrics["Strategy"].isin(strategies)].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(len(strategies), 3, figsize=(14.8, max(3.2 * len(strategies), 5.2)), squeeze=False)
    fig.suptitle("32B performance stratified by target-free reliability zone", fontsize=15, fontweight="bold")
    for i, strategy in enumerate(strategies):
        for j, metric in enumerate(["Coverage_Fraction", "RMSE", "MAE"]):
            ax = axes[i, j]
            sub = df[df["Strategy"] == strategy].copy()
            # Plot best method averages by zone for readability.
            pivot = sub.pivot_table(index="Reliability_Zone", columns="Method", values=metric, aggfunc="mean")
            pivot = pivot.reindex([z for z in ZONE_ORDER if z in pivot.index])
            pivot = pivot[[m for m in methods if m in pivot.columns]]
            if pivot.empty:
                continue
            x = np.arange(len(pivot.index))
            width = min(0.78 / max(len(pivot.columns), 1), 0.18)
            for k, method in enumerate(pivot.columns):
                vals = pivot[method].astype(float).to_numpy()
                bars = ax.bar(x + (k - (len(pivot.columns)-1)/2) * width, vals, width,
                              label=method if i == 0 and j == 2 else None,
                              color=METHOD_COLORS.get(method, COLORS["gray"]), edgecolor="#333333", linewidth=0.35, alpha=0.9)
                add_bar_labels(ax, bars, fmt="{:.2f}" if metric != "Coverage_Fraction" else "{:.2f}", fontsize=5, rotation=90 if len(pivot.columns) > 3 else 0)
            ax.set_xticks(x)
            ax.set_xticklabels([ZONE_LABELS.get(z, z) for z in pivot.index], rotation=0, fontsize=8)
            ylabel = "Coverage" if metric == "Coverage_Fraction" else metric
            ax.set_ylabel(ylabel)
            ax.set_title(f"{short_label(strategy, 28)}\n{ylabel}", fontsize=10)
            apply_axis_style(ax)
            if metric == "Coverage_Fraction":
                ax.set_ylim(0, 1.05)
            else:
                vmax = np.nanmax(pivot.to_numpy()) if np.isfinite(pivot.to_numpy()).any() else 1
                ax.set_ylim(0, vmax * 1.22)
            if i == 0 and j == 2:
                ax.legend(frameon=False, fontsize=7, ncol=1, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.text(0.5, 0.005, "Domain zones are assigned only from target-free AD components; observed values are used only for evaluation.", ha="center", fontsize=8)
    save_fig(fig, "154_32b_reliability_zone_metrics.png")


def plot_strategy_sensitivity(rule_perf: pd.DataFrame) -> None:
    all_df = rule_perf[rule_perf["Rule"] == "AllDomain_NoFiltering"].copy()
    methods = [m for m in ["CB-XGB", "CatBoost", "XGBoost", "KRR_Poly", "KRR_CatBoost_Weighted_2Model"] if m in set(all_df["Method"])]
    strategies = ordered_unique(all_df["Strategy"].astype(str).unique(), STRATEGY_ORDER)
    pivot = all_df[all_df["Method"].isin(methods)].pivot_table(index="Method", columns="Strategy", values="RMSE", aggfunc="mean")
    pivot = pivot.reindex(index=methods, columns=strategies)
    if pivot.empty:
        log("[SKIP] plot_strategy_sensitivity: empty pivot.")
        return
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    arr = pivot.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="YlGnBu_r")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([short_label(c, 18) for c in pivot.columns], rotation=25, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([short_label(i, 22) for i in pivot.index], fontsize=8)
    ax.set_title("All-domain RMSE sensitivity across fully grouped validation strategies", fontsize=13, fontweight="bold", pad=16)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Pooled RMSE", fontsize=9)
    fig.text(0.5, 0.005, "Lower RMSE is better; strategy sensitivity must be reported because no model dominates every fully grouped validation form.", ha="center", fontsize=8)
    save_fig(fig, "155_32b_strategy_sensitivity_heatmap.png")


def plot_paper_error_taxonomy(error_tax: pd.DataFrame, paper_summary: pd.DataFrame) -> None:
    # Focus on LOLO and key methods if possible, otherwise use all.
    df = error_tax.copy()
    if "LOLO_LeaveOneLiteratureOut" in set(df["Strategy"]):
        df = df[df["Strategy"] == "LOLO_LeaveOneLiteratureOut"].copy()
    methods = [m for m in ["CB-XGB", "CatBoost", "XGBoost", "KRR_Poly"] if m in set(df["Method"])]
    if methods:
        df = df[df["Method"].isin(methods)]
    # Count tags.
    tag_counts = []
    for _, row in df.iterrows():
        tags = str(row.get("Error_Source_Tags_Diagnostic", "unclassified")).split(";")
        tags = [t.strip() for t in tags if t.strip()]
        if not tags:
            tags = ["unclassified"]
        for tag in tags:
            tag_counts.append({"Method": row["Method"], "Tag": tag, "Paper_RMSE": row["Paper_RMSE"], "n_predictions": row["n_predictions"]})
    tag_df = pd.DataFrame(tag_counts)
    if tag_df.empty:
        log("[SKIP] plot_paper_error_taxonomy: no tag data.")
        return
    count_pivot = tag_df.pivot_table(index="Tag", columns="Method", values="n_predictions", aggfunc="count", fill_value=0)
    # Sort by total count and keep top tags.
    count_pivot["__total"] = count_pivot.sum(axis=1)
    count_pivot = count_pivot.sort_values("__total", ascending=False).head(12).drop(columns="__total")
    count_pivot = count_pivot[[m for m in methods if m in count_pivot.columns]] if methods else count_pivot

    high_df = df.sort_values("Paper_RMSE", ascending=False).head(18)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.2), gridspec_kw={"width_ratios": [1.05, 1.25]})
    fig.suptitle("32B literature-level error-source taxonomy", fontsize=15, fontweight="bold")

    # Left stacked bars.
    ax = axes[0]
    y = np.arange(len(count_pivot.index))
    left = np.zeros(len(count_pivot.index))
    for method in count_pivot.columns:
        vals = count_pivot[method].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, color=METHOD_COLORS.get(method, COLORS["gray"]), edgecolor="#333333", linewidth=0.3, label=method)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([short_label(t, 28) for t in count_pivot.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Paper-count entries")
    ax.set_title("Diagnostic error-source tag frequency")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    apply_axis_style(ax)

    # Right high-error paper scatter.
    ax = axes[1]
    x = np.arange(len(high_df))
    colors = [METHOD_COLORS.get(m, COLORS["gray"]) for m in high_df["Method"]]
    sizes = np.clip(high_df["n_predictions"].astype(float).to_numpy(), 10, 200)
    sc = ax.scatter(x, high_df["Paper_RMSE"], s=sizes, c=colors, alpha=0.82, edgecolor="#333333", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(p, 10) for p in high_df["文献编号"]], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Paper-level RMSE")
    ax.set_title("High-error literature groups")
    apply_axis_style(ax)
    for xi, rmse in zip(x, high_df["Paper_RMSE"]):
        ax.text(xi, rmse + 1.0, f"{rmse:.1f}", ha="center", va="bottom", fontsize=6, rotation=90)
    fig.text(0.5, 0.005, "Tags are diagnostic only; they use paper-level outcomes for interpretation, not for model deployment filtering.", ha="center", fontsize=8)
    save_fig(fig, "156_32b_paper_error_source_taxonomy.png")


def plot_prediction_panels(pred_df: Optional[pd.DataFrame], main_methods: Optional[List[str]] = None) -> None:
    if pred_df is None or not {"Observed", "Predicted", "Reliability_Zone"}.issubset(pred_df.columns):
        log("[SKIP] plot_prediction_panels: no scored prediction table.")
        return
    df = pred_df.copy()
    # Choose LOLO if available, otherwise all data from first strategy.
    if "LOLO_LeaveOneLiteratureOut" in set(df["Strategy"]):
        df = df[df["Strategy"] == "LOLO_LeaveOneLiteratureOut"].copy()
    else:
        first_strategy = ordered_unique(df["Strategy"].astype(str).unique(), STRATEGY_ORDER)[0]
        df = df[df["Strategy"] == first_strategy].copy()
    if main_methods is None:
        main_methods = [m for m in ["CB-XGB", "CatBoost", "XGBoost", "KRR_Poly"] if m in set(df["Method"])]
    if not main_methods:
        main_methods = ordered_unique(df["Method"].astype(str).unique(), METHOD_ORDER)[:4]
    main_methods = main_methods[:4]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), squeeze=False)
    fig.suptitle("32B predicted-vs-observed plots by reliability zone", fontsize=15, fontweight="bold")
    zone_color = {"Reliable": COLORS["reliable"], "Warning": COLORS["warning"], "Outside": COLORS["outside"], "Unknown": COLORS["gray"]}
    for ax, method in zip(axes.ravel(), main_methods):
        sub = df[df["Method"] == method].copy()
        if sub.empty:
            ax.axis("off")
            continue
        for zone in ZONE_ORDER:
            zdf = sub[sub["Reliability_Zone"] == zone]
            if zdf.empty:
                continue
            ax.scatter(zdf["Observed"], zdf["Predicted"], s=18, alpha=0.58, color=zone_color[zone], edgecolors="none", label=ZONE_LABELS[zone])
        ax.plot([0, 100], [0, 100], color="red", linestyle="--", linewidth=1.0)
        m_all = metric_values(sub["Observed"], sub["Predicted"])
        rel_sub = sub[sub["Reliability_Zone"] == "Reliable"]
        m_rel = metric_values(rel_sub["Observed"], rel_sub["Predicted"]) if not rel_sub.empty else {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "n": 0}
        txt = (
            f"All: R2={m_all['R2']:.2f}, RMSE={m_all['RMSE']:.2f}, MAE={m_all['MAE']:.2f}\n"
            f"Reliable: n={m_rel['n']}, RMSE={m_rel['RMSE']:.2f}, MAE={m_rel['MAE']:.2f}"
        )
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="#CCCCCC", alpha=0.85, boxstyle="round,pad=0.25"))
        ax.set_title(method)
        ax.set_xlabel("Observed F removal (%)")
        ax.set_ylabel("Predicted F removal (%)")
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.grid(True, color=COLORS["grid"], alpha=0.7, linewidth=0.7)
        if method == main_methods[0]:
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    save_fig(fig, "157_32b_reliable_domain_prediction_panels.png")


# =============================================================================
# Report writing
# =============================================================================

def write_decision_report(
    rule_catalog: pd.DataFrame,
    rule_selection: pd.DataFrame,
    recommendations: pd.DataFrame,
    score_corr: pd.DataFrame,
    paper_source_summary: pd.DataFrame,
    main_rule: str,
    warning_rule: str,
) -> Path:
    path = TABLE_DIR / "32b_final_rd_decision_report.txt"
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("Step 32B reliable-domain definition and model-comparison report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Define a target-free reliable-domain / warning-domain reporting framework after 32A2 corrected AD diagnosis.")
    lines.append("")
    lines.append("Strictness:")
    lines.append("- No samples or literature groups are removed from all-domain results.")
    lines.append("- Domain labels are assigned only by target-free AD components.")
    lines.append("- Observed F removal, residuals and test labels are used only for evaluation after domain assignment.")
    lines.append("- 文献编号 is used only for literature-level diagnostics, never as a model input.")
    lines.append("- All-domain metrics are always reported beside reliable-domain metrics.")
    lines.append("")
    lines.append("Main 32B domain rules:")
    lines.append(f"- Main reliable-domain rule: {main_rule} / {RULE_LABELS.get(main_rule, main_rule)}")
    lines.append(f"- Secondary warning-domain rule: {warning_rule} / {RULE_LABELS.get(warning_rule, warning_rule)}")
    lines.append("")
    lines.append("Rule catalog:")
    lines.append(rule_catalog.to_string(index=False))
    lines.append("")

    if not recommendations.empty:
        lines.append("Strategy-level recommendation summary:")
        keep = [
            "Strategy", "Best_AllDomain_Method", "Best_AllDomain_RMSE", "Best_AllDomain_MAE", "Best_AllDomain_R2",
            "Best_ReliableDomain_Method", "Best_ReliableDomain_RMSE", "Best_ReliableDomain_MAE", "Best_ReliableDomain_R2",
            "ReliableDomain_Coverage", "ReliableDomain_RMSE_Gain_vs_BestAll", "32B_Recommendation",
        ]
        lines.append(recommendations[keep].to_string(index=False))
        lines.append("")

    if not rule_selection.empty:
        lines.append("Main reliable-domain rows by Strategy and Method:")
        main_rows = rule_selection[rule_selection["Selected_Rule_Role"] == "ReliableDomain_MainRule"].copy()
        main_rows = main_rows.sort_values(["Strategy", "Reliable_RMSE"])
        keep = [
            "Strategy", "Method", "Selected_Rule", "Coverage_Fraction", "AllDomain_RMSE", "Reliable_RMSE",
            "RMSE_Change_vs_AllDomain", "AllDomain_MAE", "Reliable_MAE", "MAE_Change_vs_AllDomain",
            "AllDomain_R2", "Reliable_R2",
        ]
        lines.append(main_rows[keep].to_string(index=False))
        lines.append("")

    if not score_corr.empty:
        lines.append("AD-error relationship audit:")
        corr_sub = score_corr[score_corr["Score_Component"].astype(str).isin([
            "Corrected_AD_Risk", "Distance_AD_Risk", "Boundary_AD_Risk", "Categorical_Unseen_Fraction", "Numeric_Strict_Range_Violation_Fraction"
        ])].copy()
        if not corr_sub.empty:
            mean_corr = corr_sub.groupby(["Score_Component", "Method"], dropna=False)["Spearman_with_AbsError"].mean().reset_index()
            lines.append(mean_corr.pivot_table(index="Score_Component", columns="Method", values="Spearman_with_AbsError").round(3).to_string())
        lines.append("Interpretation: corrected AD risk should not be presented as a monotonic error predictor; use it as reliability flag and error-source diagnosis.")
        lines.append("")

    if not paper_source_summary.empty:
        lines.append("Paper-level error-source tag summary, top rows by mean paper RMSE:")
        top = paper_source_summary.sort_values("Mean_Paper_RMSE", ascending=False).head(20)
        lines.append(top.to_string(index=False))
        lines.append("")

    lines.append("32B decision:")
    lines.append("1. Keep all-domain fully grouped metrics as the main cross-literature generalization benchmark.")
    lines.append("2. Use the predefined reliable-domain rule for side-by-side reliable-region reporting, not for deleting data from the main result.")
    lines.append("3. Use the warning-domain / outside-domain labels as prediction reliability flags and manuscript discussion of deployment boundaries.")
    lines.append("4. Use paper-level error-source taxonomy to explain why some low-AD-risk papers still have high error: target-distribution novelty, small-n papers, and literature-specific effects.")
    lines.append("5. In the manuscript, avoid claiming that AD risk monotonically predicts error unless a later analysis demonstrates that relationship.")
    lines.append("")
    lines.append("Recommended next step:")
    lines.append("32C uncertainty/rejection-policy visualization or manuscript figure assembly: convert 32B outputs into a final AD figure and Supplementary Tables.")
    lines.append("=" * 100)
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[REPORT] {path}")
    return path


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    configure_matplotlib()
    log("=" * 100)
    log("Step 32B reliable-domain definition and model comparison started.")
    log(f"Project root: {ROOT}")
    log(RANDOM_NOTE)
    log("=" * 100)

    # Required Step 32A2 outputs.
    rule_perf = read_csv_safely("32a2_ad_reliable_rule_performance.csv", required=True)
    error_tax = read_csv_safely("32a2_ad_error_source_taxonomy.csv", required=True)
    score_corr = read_csv_safely("32a2_ad_score_error_correlation.csv", required=True)
    feature_audit = read_csv_safely("32a2_ad_feature_space_audit.csv", required=True)

    # Optional all-domain context tables.
    rmse_policy_summary = read_csv_safely("31p_rmse_policy_summary.csv", required=False)
    n_summary = read_csv_safely("31n_grouped_validation_summary.csv", required=False)
    paper_macro = read_csv_safely("31n_grouped_validation_paper_macro_summary.csv", required=False)

    # Step 32B rule catalog and selection.
    rule_catalog = build_rule_catalog()
    main_rule = choose_primary_rule(rule_perf)
    warning_rule = choose_warning_rule(rule_perf)
    rule_catalog["Selected_As_32B_Main"] = rule_catalog["Rule_ID"].eq(main_rule)
    rule_catalog["Selected_As_32B_Warning"] = rule_catalog["Rule_ID"].eq(warning_rule)
    save_table(rule_catalog, "32b_rd_rule_catalog.csv")

    rule_selection = prepare_rule_selection(rule_perf, main_rule=main_rule, warning_rule=warning_rule)
    save_table(rule_selection, "32b_rule_selection_by_strategy_method.csv")

    # Convert 32A2 rule performance to clean all-vs-domain summary.
    all_vs_domain = rule_perf.copy()
    all_vs_domain["Rule_Label"] = all_vs_domain["Rule"].map(lambda r: RULE_LABELS.get(str(r), str(r)))
    all_vs_domain["Is_32B_Main_Reliable_Rule"] = all_vs_domain["Rule"].eq(main_rule)
    all_vs_domain["Is_32B_Warning_Rule"] = all_vs_domain["Rule"].eq(warning_rule)
    save_table(all_vs_domain, "32b_all_vs_reliable_domain_summary.csv")

    # Optional sample-level zone assignment and metrics.
    pred_df = load_prediction_with_scores()
    if pred_df is not None and (("Corrected_AD_Risk" in pred_df.columns) or ("AD_Risk_Score" in pred_df.columns)) :
        pred_df = classify_domain_zone(pred_df)
        save_table(pred_df, "32b_predictions_with_reliability_zone.csv")
    elif pred_df is not None:
        log("[WARN] Sample-level predictions lack corrected AD components; no reliability-zone assignment produced.")
    zone_metrics = compute_zone_metrics(pred_df)
    if not zone_metrics.empty:
        save_table(zone_metrics, "32b_reliability_zone_metrics.csv")
    else:
        save_table(pd.DataFrame(), "32b_reliability_zone_metrics.csv")

    # Strategy recommendations and paper error-source summaries.
    recommendations = summarize_strategy_recommendations(rule_perf, rule_selection, zone_metrics)
    save_table(recommendations, "32b_strategy_model_recommendation.csv")

    paper_source_summary = summarize_paper_error_sources(error_tax)
    save_table(paper_source_summary, "32b_paper_error_source_summary.csv")

    # Additional audit summaries.
    feature_audit_summary = feature_audit.groupby(["Included_in_AD", "Reason", "AD_Type"], dropna=False).agg(
        n_features=("Feature_Table_Column", "count"),
        mean_missing_fraction=("Missing_Fraction", "mean"),
        mean_unique_count=("Unique_Count", "mean"),
    ).reset_index()
    save_table(feature_audit_summary, "32b_feature_space_audit_summary.csv")

    # Figures.
    plot_rule_selection_overview(rule_perf, main_rule=main_rule, warning_rule=warning_rule)
    plot_all_vs_reliable(rule_selection, main_rule=main_rule)
    plot_zone_metrics(zone_metrics)
    plot_strategy_sensitivity(rule_perf)
    plot_paper_error_taxonomy(error_tax, paper_source_summary)
    plot_prediction_panels(pred_df)

    # Decision report.
    write_decision_report(
        rule_catalog=rule_catalog,
        rule_selection=rule_selection,
        recommendations=recommendations,
        score_corr=score_corr,
        paper_source_summary=paper_source_summary,
        main_rule=main_rule,
        warning_rule=warning_rule,
    )

    elapsed = time.time() - START_TIME
    log("=" * 100)
    log("32B reliable-domain definition and model comparison finished.")
    log(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    log("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        elapsed = time.time() - START_TIME
        log("=" * 100)
        log(f"[ERROR] {SCRIPT_NAME} failed after {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
        log("Please send the last traceback lines if this appears in PyCharm.")
        log("=" * 100)
        raise
