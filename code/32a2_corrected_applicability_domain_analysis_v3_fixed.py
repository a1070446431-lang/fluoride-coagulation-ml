# -*- coding: utf-8 -*-
"""
32a2_corrected_applicability_domain_analysis.py

Purpose
-------
Corrected and expanded applicability-domain (AD; 适用域) analysis after Step 32A.

Why 32A2 is needed
------------------
Step 32A showed that the first target-free AD risk score did not have a stable
monotonic relationship with grouped prediction error. It also found that
non-chemical index variables such as "_Data_Index" and "原始行号" entered the
numeric range-violation drivers. This script rebuilds the AD feature space and
adds several diagnostics required for a paper-level AD section.

Strictness
----------
1. No samples or papers are removed from all-domain results.
2. 文献编号 is used only for grouped-domain construction and diagnostics, never
   as a model input feature.
3. Main deployment AD scores are target-free: they do NOT use observed F removal,
   test residuals, or test labels.
4. Target distribution novelty is computed ONLY as diagnostic evidence; it must
   not be used as a deployable reliable-domain rule.
5. Reliable-domain metrics are reported side by side with all-domain metrics,
   not as replacement results.
6. Total runtime is printed at the end.

Main corrections vs 32A
-----------------------
- Remove index/id/source/order columns from AD feature space.
- Remove duplicated raw-copy numeric variables such as "*_原始" when the cleaned
  counterpart exists.
- Report multiple AD components instead of relying on one opaque score:
  KNN distance, Mahalanobis distance, PCA reconstruction error, leverage,
  numeric range/quantile extrapolation, categorical novelty.
- Add component-wise reliable-domain rules and threshold sweeps.
- Add model-specific reliable-domain metrics.
- Add target-distribution novelty and paper-size diagnostics for explanation
  only.
- Add error-source taxonomy for high-error literature groups.

Expected input files
--------------------
Feature table, preferred:
    data/processed/model_ready_v1_with_source_mechanism31b.csv
Fallback:
    data/processed/model_ready_v1_with_source.csv

Prediction table, preferred:
    outputs/tables/31n_grouped_validation_predictions.csv
Fallback:
    outputs/tables/31p_rmse_policy_predictions.csv
Fallback:
    outputs/tables/32a_ad_predictions_with_scores.csv

Split table, preferred:
    outputs/tables/31n_grouped_validation_split_summary.csv
Fallback:
    outputs/tables/31p_rmse_policy_selected_by_split.csv
If unavailable, this script reconstructs the reference domain as non-test groups
and writes a warning. That fallback is less strict than using original
train/valid/test split summaries.

Outputs
-------
Tables:
    32a2_ad_feature_space_audit.csv
    32a2_ad_sample_scores.csv
    32a2_ad_predictions_with_scores.csv
    32a2_ad_score_error_correlation.csv
    32a2_ad_performance_by_risk_quantile.csv
    32a2_ad_reliable_rule_performance.csv
    32a2_ad_component_rule_sweep.csv
    32a2_ad_paper_summary.csv
    32a2_ad_error_source_taxonomy.csv
    32a2_ad_feature_range_violation_summary.csv
    32a2_ad_category_unseen_summary.csv
    32a2_ad_decision_report.txt

Figures:
    145_32a2_score_error_relationship.png/pdf
    146_32a2_performance_by_risk_quantile.png/pdf
    147_32a2_component_correlation_heatmap.png/pdf
    148_32a2_reliable_rule_tradeoff.png/pdf
    149_32a2_paper_error_source_map.png/pdf
    150_32a2_violation_drivers_cleaned.png/pdf
    151_32a2_model_specific_reliable_domain.png/pdf
"""

from __future__ import annotations

import os
import re
import math
import time
import warnings
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text

from sklearn.compose import ColumnTransformer
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# ============================================================
# 0. Global config
# ============================================================

SCRIPT_START = time.time()

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name.lower() == "src" else THIS_FILE.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 20260605

# Deployment score weights. These are still diagnostic and will be evaluated
# empirically; they are not assumed valid before checking the error relationship.
DEPLOYABLE_SCORE_WEIGHTS = {
    "NN_Risk": 0.25,
    "Mahalanobis_Risk": 0.20,
    "PCA_Reconstruction_Risk": 0.15,
    "Leverage_Risk": 0.10,
    "Numeric_Boundary_Risk": 0.15,
    "Categorical_Unseen_Risk": 0.15,
}

DISTANCE_SCORE_WEIGHTS = {
    "NN_Risk": 0.35,
    "Mahalanobis_Risk": 0.25,
    "PCA_Reconstruction_Risk": 0.20,
    "Leverage_Risk": 0.20,
}

BOUNDARY_SCORE_WEIGHTS = {
    "Numeric_Boundary_Risk": 0.60,
    "Categorical_Unseen_Risk": 0.40,
}

# Predefined target-free reliable-domain rules to be evaluated, not blindly accepted.
RELIABLE_RULES = {
    "Rule_A_strict_component": {
        "Corrected_AD_Risk": 0.55,
        "Distance_AD_Risk": 0.60,
        "Numeric_Strict_Range_Violation_Fraction": 0.10,
        "Categorical_Unseen_Fraction": 0.10,
    },
    "Rule_B_balanced_component": {
        "Corrected_AD_Risk": 0.65,
        "Distance_AD_Risk": 0.75,
        "Numeric_Strict_Range_Violation_Fraction": 0.20,
        "Categorical_Unseen_Fraction": 0.20,
    },
    "Rule_C_distance_focused": {
        "Distance_AD_Risk": 0.65,
        "NN_Risk": 0.80,
        "Mahalanobis_Risk": 0.85,
        "PCA_Reconstruction_Risk": 0.85,
    },
    "Rule_D_boundary_focused": {
        "Boundary_AD_Risk": 0.35,
        "Numeric_Strict_Range_Violation_Fraction": 0.10,
        "Categorical_Unseen_Fraction": 0.20,
    },
    "Rule_E_loose_screen": {
        "Corrected_AD_Risk": 0.75,
        "Numeric_Strict_Range_Violation_Fraction": 0.25,
        "Categorical_Unseen_Fraction": 0.30,
    },
}

PRIMARY_METHOD_CANDIDATES = ["CB-XGB", "CatBoost", "XGBoost", "KRR_Poly", "KRR_CatBoost_Weighted_2Model"]

LOW_SAT_COLORS = [
    "#8ab6a0", "#d7a46f", "#8aa9c7", "#b58db6", "#a0a0a0",
    "#c99a9a", "#a4c3a2", "#c4b08a", "#9eb7d0", "#b0a3c2"
]


# ============================================================
# 1. Utility
# ============================================================

def log(msg: str) -> None:
    print(msg, flush=True)


# ---- Figure text/font/layout helpers --------------------------------------
# These helpers are intentionally kept inside this script so that 32A2 can be
# copied as a complete replacement file without editing config.py or utils.py.

SUP_SUB_TRANSLATION = str.maketrans({
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "₊": "+", "₋": "-",
})


def normalize_plot_text_for_matplotlib(text):
    """Make figure labels safe for Windows Matplotlib fonts.

    The chemistry labels in this project may contain Unicode subscript/superscript
    characters such as SO₄²⁻ and PO₄³⁻. Some fonts bundled with Matplotlib or
    Windows do not contain these glyphs, so saving PNG/PDF repeatedly prints long
    UserWarning messages. We convert them to ASCII chemistry notation before
    saving figures: SO4^2-, PO4^3-, Ca^2+, etc.
    """
    if text is None:
        return text
    if not isinstance(text, str):
        text = str(text)

    specific_map = {
        "SO₄²⁻": "SO4^2-", "CO₃²⁻": "CO3^2-", "PO₄³⁻": "PO4^3-",
        "HCO₃⁻": "HCO3-", "NO₃⁻": "NO3-", "Cl⁻": "Cl-", "F⁻": "F-", "OH⁻": "OH-",
        "Ca²⁺": "Ca^2+", "Mg²⁺": "Mg^2+", "Al³⁺": "Al^3+", "Fe³⁺": "Fe^3+", "Fe²⁺": "Fe^2+",
        "SO₄": "SO4", "CO₃": "CO3", "PO₄": "PO4", "NH₄": "NH4",
        "SO₄⁻²": "SO4^2-", "PO₄⁻³": "PO4^3-", "CO₃⁻²": "CO3^2-",
    }
    for old, new in specific_map.items():
        text = text.replace(old, new)
    return text.translate(SUP_SUB_TRANSLATION)


def wrap_plot_text(text, width=24):
    """Wrap long labels to prevent overlap while keeping short labels unchanged."""
    text = normalize_plot_text_for_matplotlib(text)
    if text is None:
        return text
    if "\n" in text or len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False, break_on_hyphens=False))


def set_est_style_fonts():
    """Set robust, publication-style fonts and spacing defaults."""
    preferred = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    font_list = [f for f in preferred if f in available] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": font_list,
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "savefig.dpi": 600,
    })


def sanitize_figure_text(fig, wrap_width=26):
    """Sanitize every text object in a figure before saving."""
    for text_obj in fig.findobj(match=Text):
        try:
            old = text_obj.get_text()
            # Do not over-wrap numeric annotations.
            width = wrap_width
            if isinstance(old, str) and re.fullmatch(r"[-+]?\d+(\.\d+)?", old.strip()):
                width = 999
            text_obj.set_text(wrap_plot_text(old, width=width))
        except Exception:
            pass

    for ax in fig.axes:
        try:
            ax.set_title(wrap_plot_text(ax.get_title(), width=34), pad=14)
            ax.set_xlabel(wrap_plot_text(ax.get_xlabel(), width=34), labelpad=10)
            ax.set_ylabel(wrap_plot_text(ax.get_ylabel(), width=34), labelpad=10)
        except Exception:
            pass
        try:
            for lab in ax.get_xticklabels():
                lab.set_text(wrap_plot_text(lab.get_text(), width=18))
        except Exception:
            pass
        try:
            for lab in ax.get_yticklabels():
                lab.set_text(wrap_plot_text(lab.get_text(), width=22))
        except Exception:
            pass
        try:
            leg = ax.get_legend()
            if leg is not None:
                for lab in leg.get_texts():
                    lab.set_text(wrap_plot_text(lab.get_text(), width=24))
        except Exception:
            pass

    try:
        if getattr(fig, "_suptitle", None) is not None:
            fig._suptitle.set_text(wrap_plot_text(fig._suptitle.get_text(), width=55))
            fig._suptitle.set_y(0.985)
    except Exception:
        pass


def prettify_axis(ax, rotate_x=30, ha="right", wrap_x=18, wrap_y=22, grid_axis="y"):
    """Apply consistent spacing to one axis."""
    try:
        ax.grid(True, axis=grid_axis, alpha=0.25)
    except Exception:
        pass
    # Matplotlib Text objects do not consistently support set_pad across versions.
    # Use Axes.set_title(..., pad=...) instead, which is stable.
    try:
        current_title = ax.get_title()
        if current_title:
            ax.set_title(current_title, pad=14)
    except Exception:
        pass
    try:
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
    except Exception:
        pass
    try:
        ticks = ax.get_xticks()
        labels = [wrap_plot_text(t.get_text(), width=wrap_x) for t in ax.get_xticklabels()]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=rotate_x, ha=ha, va="top")
    except Exception:
        pass
    try:
        ticks = ax.get_yticks()
        labels = [wrap_plot_text(t.get_text(), width=wrap_y) for t in ax.get_yticklabels()]
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    except Exception:
        pass


def expand_y_for_text(ax, top_frac=0.12, bottom_frac=0.05):
    """Leave headroom for value labels above bars."""
    try:
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        if span <= 0:
            return
        ax.set_ylim(ymin - span * bottom_frac, ymax + span * top_frac)
    except Exception:
        pass


def move_legend_outside(ax, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1):
    leg = ax.get_legend()
    if leg is not None:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, [wrap_plot_text(x, 24) for x in labels], loc=loc,
                      bbox_to_anchor=bbox_to_anchor, frameon=False, ncol=ncol)


def finalize_figure_layout(fig, rect=(0.03, 0.05, 0.98, 0.94), h_pad=1.4, w_pad=1.4):
    """Final anti-overlap layout pass."""
    try:
        fig.tight_layout(rect=rect, h_pad=h_pad, w_pad=w_pad)
    except Exception:
        pass


def save_fig(fig, name: str) -> None:
    png = FIG_DIR / name
    pdf = FIG_DIR / name.replace(".png", ".pdf")

    sanitize_figure_text(fig, wrap_width=26)
    finalize_figure_layout(fig, rect=(0.035, 0.055, 0.985, 0.94), h_pad=1.6, w_pad=1.6)

    # Only hide harmless missing-glyph warnings. Real exceptions still stop the script.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Glyph .* missing from font.*", category=UserWarning)
        fig.savefig(png, dpi=600, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] {png}")
    log(f"[FIG] {pdf}")


def safe_to_csv(df: pd.DataFrame, filename: str) -> Path:
    path = TABLE_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {path} | shape={df.shape}")
    return path


def find_existing_file(candidates: List[Path], required: bool = True) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    if required:
        raise FileNotFoundError("Cannot find any required file:\n" + "\n".join(map(str, candidates)))
    return None


def normalize_col_name(c: str) -> str:
    return str(c).strip()


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "氟去除率", "氟去除率_%", "F removal", "F removal (%)", "F_removal", "F_removal_percent",
        "Removal", "Removal_%", "target", "Target"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if ("removal" in lc or "去除" in str(c)) and ("预测" not in str(c)) and ("pred" not in lc):
            return c
    return None


def detect_source_column(df: pd.DataFrame) -> str:
    for c in ["文献编号", "source", "Source", "Paper_ID", "paper_id", "Literature_ID", "literature_id"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "文献" in str(c) or "paper" in str(c).lower() or "literature" in str(c).lower():
            return c
    raise ValueError("Cannot find literature/source group column, such as 文献编号.")


def sanitize_method_name(m: str) -> str:
    return str(m).replace("/", "_").replace("\\", "_").replace(" ", "_")


def metrics_dict(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "Bias": np.nan, "n": 0}
    yt = y_true[mask]
    yp = y_pred[mask]
    mse = mean_squared_error(yt, yp)
    if len(np.unique(yt)) < 2:
        r2 = np.nan
    else:
        r2 = r2_score(yt, yp)
    return {
        "R2": float(r2),
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(mean_absolute_error(yt, yp)),
        "MSE": float(mse),
        "Bias": float(np.mean(yp - yt)),
        "n": int(mask.sum()),
    }


def empirical_risk(values_train: np.ndarray, values_test: np.ndarray) -> np.ndarray:
    """Map larger test values to percentile risk using train empirical distribution."""
    train = np.asarray(values_train, dtype=float)
    test = np.asarray(values_test, dtype=float)
    train = train[np.isfinite(train)]
    if len(train) == 0:
        return np.full_like(test, np.nan, dtype=float)
    train_sorted = np.sort(train)
    ranks = np.searchsorted(train_sorted, test, side="right") / len(train_sorted)
    return np.clip(ranks, 0.0, 1.0)


def weighted_sum(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    out = np.zeros(len(df), dtype=float)
    total = 0.0
    for k, w in weights.items():
        if k in df.columns:
            vals = pd.to_numeric(df[k], errors="coerce").fillna(0.0).values
            out += w * vals
            total += w
    if total <= 0:
        return np.full(len(df), np.nan)
    return np.clip(out / total, 0.0, 1.0)


def split_semicolon_groups(x) -> set:
    if pd.isna(x):
        return set()
    return {str(v).strip() for v in str(x).split(";") if str(v).strip()}


# ============================================================
# 2. Load data
# ============================================================

def load_feature_table() -> Tuple[pd.DataFrame, Path, str, Optional[str]]:
    feature_path = find_existing_file([
        PROCESSED_DIR / "model_ready_v1_with_source_mechanism31b.csv",
        PROCESSED_DIR / "model_ready_v1_with_source.csv",
        PROCESSED_DIR / "model_ready_v1.csv",
    ], required=True)
    df = pd.read_csv(feature_path, low_memory=False)
    df.columns = [normalize_col_name(c) for c in df.columns]
    if "Data_Index" not in df.columns:
        df.insert(0, "Data_Index", np.arange(len(df), dtype=int))
    source_col = detect_source_column(df)
    target_col = detect_target_column(df)
    log(f"[DATA] Feature table: {feature_path} | shape={df.shape} | source={source_col} | target={target_col}")
    return df.reset_index(drop=True), feature_path, source_col, target_col


def load_prediction_table() -> Tuple[pd.DataFrame, Path]:
    pred_path = find_existing_file([
        TABLE_DIR / "31n_grouped_validation_predictions.csv",
        TABLE_DIR / "31p_rmse_policy_predictions.csv",
        TABLE_DIR / "32a_ad_predictions_with_scores.csv",
    ], required=True)
    pred = pd.read_csv(pred_path, low_memory=False)
    pred.columns = [normalize_col_name(c) for c in pred.columns]

    # If fallback is 32A enriched table, keep raw prediction columns and avoid duplicated AD columns.
    required_cols = ["Strategy", "Split_ID", "Method", "Data_Index", "文献编号", "Observed", "Predicted"]
    missing = [c for c in required_cols if c not in pred.columns]
    if missing:
        raise ValueError(f"Prediction table lacks required columns: {missing}. File={pred_path}")

    pred["Data_Index"] = pd.to_numeric(pred["Data_Index"], errors="coerce").astype("Int64")
    pred = pred.dropna(subset=["Data_Index", "Observed", "Predicted"]).copy()
    pred["Data_Index"] = pred["Data_Index"].astype(int)
    pred["Observed"] = pd.to_numeric(pred["Observed"], errors="coerce")
    pred["Predicted"] = pd.to_numeric(pred["Predicted"], errors="coerce")
    pred["Residual"] = pred["Observed"] - pred["Predicted"]
    pred["Abs_Error"] = np.abs(pred["Residual"])
    log(f"[DATA] Prediction table: {pred_path} | shape={pred.shape} | methods={pred['Method'].nunique()} | strategies={pred['Strategy'].nunique()}")
    return pred, pred_path


def load_split_table() -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    split_path = find_existing_file([
        TABLE_DIR / "31n_grouped_validation_split_summary.csv",
        TABLE_DIR / "31p_rmse_policy_selected_by_split.csv",
        TABLE_DIR / "31f_balanced_grouped_screen_group_distribution.csv",
    ], required=False)
    if split_path is None:
        log("[WARN] No split-summary table found. Reference domain will be reconstructed as non-test groups.")
        return None, None
    try:
        split = pd.read_csv(split_path, low_memory=False)
        split.columns = [normalize_col_name(c) for c in split.columns]
        log(f"[DATA] Split table: {split_path} | shape={split.shape}")
        return split, split_path
    except Exception as e:
        log(f"[WARN] Failed to read split table {split_path}: {e}")
        return None, None


# ============================================================
# 3. Corrected AD feature-space audit
# ============================================================

def should_exclude_feature(col: str, source_col: str, target_col: Optional[str], all_cols: Iterable[str]) -> Tuple[bool, str]:
    c = str(col)
    lc = c.lower()
    all_cols_set = set(map(str, all_cols))

    # Source / target / prediction / output / leakage.
    if c == source_col:
        return True, "source_group_column"
    if target_col is not None and c == target_col:
        return True, "target_column"
    if c in {"Data_Index", "_Data_Index", "Local_Test_Index", "index", "Index"}:
        return True, "index_or_order_column"
    if c in {"原始行号", "行号", "序号", "编号"}:
        return True, "row_order_or_id_column"

    leak_patterns = [
        "文献编号", "paper", "literature", "source", "doi", "reference", "ref_",
        "predicted", "prediction", "residual", "observed", "actual", "error",
        "去除率", "出水", "effluent", "target", "label", "split", "fold", "seed",
        "data_index", "local_test_index"
    ]
    for p in leak_patterns:
        if p in lc or p in c:
            return True, f"leakage_or_identifier_pattern:{p}"

    # Duplicate raw numeric/text copies.
    raw_suffixes = ["_原始", "_raw", "_Raw", "_RAW"]
    for suf in raw_suffixes:
        if c.endswith(suf):
            base = c[: -len(suf)]
            if base in all_cols_set:
                return True, "duplicated_raw_copy_with_clean_counterpart"

    # High-level administrative variables.
    admin_patterns = ["文件", "sheet", "备注", "说明", "单位", "flag", "是否删除", "缺失标记"]
    for p in admin_patterns:
        if p in lc or p in c:
            return True, f"administrative_or_flag_column:{p}"

    return False, "included"


def corrected_feature_space_audit(df: pd.DataFrame, source_col: str, target_col: Optional[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    rows = []
    included = []
    for c in df.columns:
        exclude, reason = should_exclude_feature(c, source_col, target_col, df.columns)
        dtype = str(df[c].dtype)
        miss = float(df[c].isna().mean())
        nunique = int(df[c].nunique(dropna=True))
        if not exclude:
            # Drop all-missing or constant columns.
            if miss >= 0.98:
                exclude, reason = True, "nearly_all_missing"
            elif nunique <= 1:
                exclude, reason = True, "constant_or_single_value"

        if not exclude:
            included.append(c)

        rows.append({
            "Feature_Table_Column": c,
            "Included_in_AD": bool(not exclude),
            "Reason": reason if exclude else "included_corrected_AD_feature",
            "DType": dtype,
            "Missing_Fraction": miss,
            "Unique_Count": nunique,
        })

    audit = pd.DataFrame(rows)

    # Split numeric and categorical.
    numeric_features = []
    categorical_features = []
    for c in included:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_features.append(c)
        else:
            categorical_features.append(c)

    # Avoid extremely high-cardinality categorical one-hot explosion unless the
    # column is one of the chemically meaningful categories.
    chem_cat_keywords = ["混凝剂", "干扰离子", "氟形态", "类别", "大类", "form", "ion", "coagulant"]
    final_cats = []
    for c in categorical_features:
        nunique = int(df[c].nunique(dropna=True))
        if nunique <= 80 or any(k in c for k in chem_cat_keywords):
            final_cats.append(c)
        else:
            audit.loc[audit["Feature_Table_Column"] == c, "Included_in_AD"] = False
            audit.loc[audit["Feature_Table_Column"] == c, "Reason"] = "excluded_high_cardinality_categorical"
    categorical_features = final_cats

    return audit, numeric_features, categorical_features


def build_core_chemical_feature_flags(df: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]) -> pd.DataFrame:
    """Mark whether each included feature is chemical/core/mechanism for interpretation."""
    rows = []
    core_keywords = [
        "剂量", "dose", "ph", "pH", "初始氟", "initial", "水温", "temperature",
        "停留", "retention", "干扰", "ion", "混凝", "coagulant", "氟形态",
        "Mg", "Al", "Fe", "Ca", "Zr", "Si", "La", "SO", "CO", "PO", "Cl",
        "机理", "log1p", "自由", "游离"
    ]
    for c in numeric_features + categorical_features:
        rows.append({
            "Feature": c,
            "AD_Type": "numeric" if c in numeric_features else "categorical",
            "Core_Chemical_Or_Mechanism": any(k in str(c) for k in core_keywords),
        })
    return pd.DataFrame(rows)


# ============================================================
# 4. Split reconstruction
# ============================================================

def build_split_reference_map(
    feature_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    split_df: Optional[pd.DataFrame],
    source_col: str
) -> Tuple[Dict[Tuple[str, str], dict], str]:
    """Return mapping (Strategy, Split_ID) -> train/valid/test indices/groups.

    Preferred: use split summary with train_groups/valid_groups/test_groups.
    Fallback: infer test groups from prediction table; reference groups = all non-test groups.
    """
    groups = feature_df[source_col].astype(str).reset_index(drop=True)
    all_groups = set(groups.unique())
    mapping = {}
    mode = "fallback_non_test_reference"

    if split_df is not None:
        # Best case: exact group lists are available.
        required = {"Strategy", "Split_ID", "train_groups", "valid_groups", "test_groups"}
        if required.issubset(set(split_df.columns)):
            for _, row in split_df.iterrows():
                strategy = str(row["Strategy"])
                split_id = str(row["Split_ID"])
                train_groups = split_semicolon_groups(row["train_groups"])
                valid_groups = split_semicolon_groups(row["valid_groups"])
                test_groups = split_semicolon_groups(row["test_groups"])
                train_idx = np.where(groups.isin(train_groups).values)[0]
                valid_idx = np.where(groups.isin(valid_groups).values)[0]
                test_idx = np.where(groups.isin(test_groups).values)[0]
                mapping[(strategy, split_id)] = {
                    "train_groups": train_groups,
                    "valid_groups": valid_groups,
                    "test_groups": test_groups,
                    "train_idx": train_idx,
                    "valid_idx": valid_idx,
                    "test_idx": test_idx,
                    "Reference_Mode": "exact_train_groups_from_split_summary",
                }
            mode = "exact_train_groups_from_split_summary"
        else:
            log("[WARN] Split table found but does not contain train_groups/valid_groups/test_groups. Using fallback.")

    # Fill missing split keys from prediction table.
    for (strategy, split_id), part in pred_df.groupby(["Strategy", "Split_ID"], dropna=False):
        key = (str(strategy), str(split_id))
        if key in mapping:
            continue
        test_groups = set(part["文献编号"].astype(str).unique())
        ref_groups = all_groups - test_groups
        train_idx = np.where(groups.isin(ref_groups).values)[0]
        test_idx = np.where(groups.isin(test_groups).values)[0]
        mapping[key] = {
            "train_groups": ref_groups,
            "valid_groups": set(),
            "test_groups": test_groups,
            "train_idx": train_idx,
            "valid_idx": np.array([], dtype=int),
            "test_idx": test_idx,
            "Reference_Mode": "fallback_all_non_test_groups_as_reference",
        }
    return mapping, mode


# ============================================================
# 5. AD component calculation
# ============================================================

def fit_numeric_pipeline(X_train_num: pd.DataFrame) -> Pipeline:
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(quantile_range=(25, 75))),
    ])
    pipe.fit(X_train_num)
    return pipe


def build_combined_transformer(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(25, 75))),
        ]), num_cols))
    if cat_cols:
        # Compatibility with different sklearn versions.
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=None)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", enc),
        ]), cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def safe_pca_components(n_samples: int, n_features: int) -> int:
    if n_samples <= 3 or n_features <= 1:
        return 1
    return int(max(1, min(20, n_features, n_samples - 1)))


def calculate_ad_for_split(
    feature_df: pd.DataFrame,
    source_col: str,
    target_col: Optional[str],
    numeric_features: List[str],
    categorical_features: List[str],
    strategy: str,
    split_id: str,
    train_idx: np.ndarray,
    test_indices: np.ndarray,
    reference_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate corrected target-free AD components for one split."""
    if len(train_idx) < 5:
        raise RuntimeError(f"Too few reference rows for {strategy} / {split_id}.")

    train = feature_df.iloc[train_idx].copy()
    test = feature_df.iloc[test_indices].copy()

    # Ensure feature subsets exist.
    num_cols = [c for c in numeric_features if c in feature_df.columns]
    cat_cols = [c for c in categorical_features if c in feature_df.columns]

    # Numeric data.
    X_train_num = train[num_cols].copy() if num_cols else pd.DataFrame(index=train.index)
    X_test_num = test[num_cols].copy() if num_cols else pd.DataFrame(index=test.index)

    # Combined numeric + categorical space for KNN.
    transformer = build_combined_transformer(num_cols, cat_cols)
    X_train_mix = transformer.fit_transform(train[num_cols + cat_cols])
    X_test_mix = transformer.transform(test[num_cols + cat_cols])
    if hasattr(X_train_mix, "toarray"):
        X_train_mix = X_train_mix.toarray()
        X_test_mix = X_test_mix.toarray()
    X_train_mix = np.asarray(X_train_mix, dtype=float)
    X_test_mix = np.asarray(X_test_mix, dtype=float)

    # KNN distance risk.
    k = int(max(1, min(5, len(train_idx) - 1)))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_train_mix)
    test_dists, _ = nn.kneighbors(X_test_mix, n_neighbors=k)
    test_knn_mean = test_dists.mean(axis=1)

    # Train leave-one-out or train k+1 distances to build reference distribution.
    k_ref = int(max(2, min(k + 1, len(train_idx))))
    nn_ref = NearestNeighbors(n_neighbors=k_ref)
    nn_ref.fit(X_train_mix)
    train_dists, _ = nn_ref.kneighbors(X_train_mix, n_neighbors=k_ref)
    if train_dists.shape[1] > 1:
        train_knn_ref = train_dists[:, 1:].mean(axis=1)
    else:
        train_knn_ref = train_dists[:, 0]
    nn_risk = empirical_risk(train_knn_ref, test_knn_mean)

    # Numeric scaled matrix for Mahalanobis, PCA, leverage.
    if num_cols:
        num_pipe = fit_numeric_pipeline(X_train_num)
        Xtr_num = np.asarray(num_pipe.transform(X_train_num), dtype=float)
        Xte_num = np.asarray(num_pipe.transform(X_test_num), dtype=float)
    else:
        Xtr_num = np.zeros((len(train_idx), 1))
        Xte_num = np.zeros((len(test_indices), 1))

    # Remove zero-variance numeric dimensions after scaling.
    stds = np.nanstd(Xtr_num, axis=0)
    keep = stds > 1e-12
    if keep.sum() == 0:
        Xtr_num2 = np.zeros((len(train_idx), 1))
        Xte_num2 = np.zeros((len(test_indices), 1))
    else:
        Xtr_num2 = Xtr_num[:, keep]
        Xte_num2 = Xte_num[:, keep]

    # Mahalanobis risk with shrinkage covariance.
    try:
        lw = LedoitWolf().fit(Xtr_num2)
        mu = lw.location_
        precision = lw.precision_
        train_md = np.sqrt(np.sum((Xtr_num2 - mu) @ precision * (Xtr_num2 - mu), axis=1))
        test_md = np.sqrt(np.sum((Xte_num2 - mu) @ precision * (Xte_num2 - mu), axis=1))
    except Exception:
        mu = np.nanmean(Xtr_num2, axis=0)
        train_md = np.sqrt(np.sum((Xtr_num2 - mu) ** 2, axis=1))
        test_md = np.sqrt(np.sum((Xte_num2 - mu) ** 2, axis=1))
    md_risk = empirical_risk(train_md, test_md)

    # PCA reconstruction risk.
    n_comp = safe_pca_components(Xtr_num2.shape[0], Xtr_num2.shape[1])
    try:
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        pca.fit(Xtr_num2)
        train_rec = pca.inverse_transform(pca.transform(Xtr_num2))
        test_rec = pca.inverse_transform(pca.transform(Xte_num2))
        train_re = np.sqrt(np.mean((Xtr_num2 - train_rec) ** 2, axis=1))
        test_re = np.sqrt(np.mean((Xte_num2 - test_rec) ** 2, axis=1))
    except Exception:
        train_re = np.zeros(len(train_idx))
        test_re = np.zeros(len(test_indices))
        n_comp = 0
    rec_risk = empirical_risk(train_re, test_re)

    # Leverage risk in PCA score space.
    try:
        if n_comp > 0:
            Ztr = pca.transform(Xtr_num2)
            Zte = pca.transform(Xte_num2)
            # Add intercept for leverage calculation.
            Ztr_i = np.column_stack([np.ones(len(Ztr)), Ztr])
            Zte_i = np.column_stack([np.ones(len(Zte)), Zte])
            xtx_inv = np.linalg.pinv(Ztr_i.T @ Ztr_i)
            train_lev = np.sum(Ztr_i @ xtx_inv * Ztr_i, axis=1)
            test_lev = np.sum(Zte_i @ xtx_inv * Zte_i, axis=1)
        else:
            train_lev = np.zeros(len(train_idx))
            test_lev = np.zeros(len(test_indices))
    except Exception:
        train_lev = np.zeros(len(train_idx))
        test_lev = np.zeros(len(test_indices))
    lev_risk = empirical_risk(train_lev, test_lev)

    # Numeric range and quantile boundary risk.
    range_records = []
    strict_violation_count = np.zeros(len(test_indices), dtype=float)
    quantile_violation_count = np.zeros(len(test_indices), dtype=float)
    strict_severity_sum = np.zeros(len(test_indices), dtype=float)
    quantile_severity_sum = np.zeros(len(test_indices), dtype=float)

    if num_cols:
        train_num_raw = train[num_cols].apply(pd.to_numeric, errors="coerce")
        test_num_raw = test[num_cols].apply(pd.to_numeric, errors="coerce")
        for c in num_cols:
            tr = train_num_raw[c].values.astype(float)
            te = test_num_raw[c].values.astype(float)
            tr = tr[np.isfinite(tr)]
            if len(tr) < 5:
                continue
            mn, mx = np.nanmin(tr), np.nanmax(tr)
            q01, q99 = np.nanquantile(tr, 0.01), np.nanquantile(tr, 0.99)
            iqr = np.nanquantile(tr, 0.75) - np.nanquantile(tr, 0.25)
            scale = max(float(iqr), 1e-9)

            below = te < mn
            above = te > mx
            strict = below | above
            strict_sev = np.zeros_like(te, dtype=float)
            strict_sev[below] = (mn - te[below]) / scale
            strict_sev[above] = (te[above] - mx) / scale
            strict_sev = np.clip(strict_sev, 0, 5)

            qbelow = te < q01
            qabove = te > q99
            qviol = qbelow | qabove
            qsev = np.zeros_like(te, dtype=float)
            qsev[qbelow] = (q01 - te[qbelow]) / scale
            qsev[qabove] = (te[qabove] - q99) / scale
            qsev = np.clip(qsev, 0, 5)

            finite_mask = np.isfinite(te)
            strict_violation_count += (strict & finite_mask).astype(float)
            quantile_violation_count += (qviol & finite_mask).astype(float)
            strict_severity_sum += np.where(finite_mask, strict_sev, 0.0)
            quantile_severity_sum += np.where(finite_mask, qsev, 0.0)

            if int(np.sum(strict & finite_mask)) > 0 or int(np.sum(qviol & finite_mask)) > 0:
                range_records.append({
                    "Strategy": strategy,
                    "Split_ID": split_id,
                    "Feature": c,
                    "Reference_Mode": reference_mode,
                    "n_test": int(len(te)),
                    "Strict_Violations": int(np.sum(strict & finite_mask)),
                    "Quantile_Violations": int(np.sum(qviol & finite_mask)),
                    "Strict_Violation_Fraction": float(np.mean(strict[finite_mask])) if finite_mask.sum() else 0.0,
                    "Quantile_Violation_Fraction": float(np.mean(qviol[finite_mask])) if finite_mask.sum() else 0.0,
                    "Mean_Strict_Severity": float(np.mean(strict_sev[finite_mask])) if finite_mask.sum() else 0.0,
                    "Mean_Quantile_Severity": float(np.mean(qsev[finite_mask])) if finite_mask.sum() else 0.0,
                    "Max_Strict_Severity": float(np.max(strict_sev[finite_mask])) if finite_mask.sum() else 0.0,
                })

    denom = max(len(num_cols), 1)
    strict_fraction = strict_violation_count / denom
    quantile_fraction = quantile_violation_count / denom
    strict_sev_mean = strict_severity_sum / denom
    quantile_sev_mean = quantile_severity_sum / denom

    # Normalize numeric boundary risk to [0, 1]. Quantile boundary is deliberately
    # included because min-max range can be too permissive for small heterogeneous data.
    numeric_boundary_risk = np.clip(
        0.45 * strict_fraction / 0.20 +
        0.35 * quantile_fraction / 0.35 +
        0.20 * np.tanh((strict_sev_mean + 0.5 * quantile_sev_mean) / 1.5),
        0, 1
    )

    # Categorical unseen risk.
    cat_records = []
    unseen_count = np.zeros(len(test_indices), dtype=float)
    if cat_cols:
        for c in cat_cols:
            train_vals = set(train[c].astype(str).fillna("__MISSING__").unique())
            te_vals = test[c].astype(str).fillna("__MISSING__").values
            unseen = np.array([v not in train_vals for v in te_vals], dtype=bool)
            unseen_count += unseen.astype(float)
            vals, counts = np.unique(te_vals[unseen], return_counts=True)
            for v, cnt in zip(vals, counts):
                cat_records.append({
                    "Strategy": strategy,
                    "Split_ID": split_id,
                    "Feature": c,
                    "Unseen_Value": v,
                    "Total_Count": int(cnt),
                    "Fraction_in_Test": float(cnt / max(len(te_vals), 1)),
                    "Reference_Mode": reference_mode,
                })
    cat_fraction = unseen_count / max(len(cat_cols), 1)
    cat_risk = np.clip(cat_fraction / 0.30, 0, 1)

    score = pd.DataFrame({
        "Strategy": strategy,
        "Split_ID": split_id,
        "Data_Index": test_indices.astype(int),
        "文献编号": test[source_col].astype(str).values,
        "Reference_Mode": reference_mode,
        "n_reference_rows": int(len(train_idx)),
        "n_reference_groups": int(train[source_col].nunique()),
        "n_test_rows": int(len(test_indices)),
        "n_test_groups": int(test[source_col].nunique()),
        "AD_Numeric_Features_Used": int(len(num_cols)),
        "AD_Categorical_Features_Used": int(len(cat_cols)),
        "AD_PCA_Components": int(n_comp),
        "KNN_Mean_Distance": test_knn_mean,
        "NN_Risk": nn_risk,
        "Mahalanobis_Distance": test_md,
        "Mahalanobis_Risk": md_risk,
        "PCA_Reconstruction_Error": test_re,
        "PCA_Reconstruction_Risk": rec_risk,
        "Leverage": test_lev,
        "Leverage_Risk": lev_risk,
        "Numeric_Strict_Range_Violation_Fraction": strict_fraction,
        "Numeric_Quantile_Range_Violation_Fraction": quantile_fraction,
        "Numeric_Strict_Range_Mean_Severity": strict_sev_mean,
        "Numeric_Quantile_Range_Mean_Severity": quantile_sev_mean,
        "Numeric_Boundary_Risk": numeric_boundary_risk,
        "Categorical_Unseen_Fraction": cat_fraction,
        "Categorical_Unseen_Risk": cat_risk,
    })
    score["Distance_AD_Risk"] = weighted_sum(score, DISTANCE_SCORE_WEIGHTS)
    score["Boundary_AD_Risk"] = weighted_sum(score, BOUNDARY_SCORE_WEIGHTS)
    score["Corrected_AD_Risk"] = weighted_sum(score, DEPLOYABLE_SCORE_WEIGHTS)
    score["Max_Component_Risk"] = score[[
        "NN_Risk", "Mahalanobis_Risk", "PCA_Reconstruction_Risk", "Leverage_Risk",
        "Numeric_Boundary_Risk", "Categorical_Unseen_Risk"
    ]].max(axis=1)

    return score, pd.DataFrame(range_records), pd.DataFrame(cat_records)


# ============================================================
# 6. Target-distribution diagnostics only
# ============================================================

def calculate_target_diagnostics(
    feature_df: pd.DataFrame,
    source_col: str,
    target_col: Optional[str],
    pred_df: pd.DataFrame,
    split_map: Dict[Tuple[str, str], dict],
) -> pd.DataFrame:
    if target_col is None or target_col not in feature_df.columns:
        return pd.DataFrame()

    y_all = pd.to_numeric(feature_df[target_col], errors="coerce").reset_index(drop=True)
    rows = []
    for (strategy, split_id), part in pred_df.groupby(["Strategy", "Split_ID"], dropna=False):
        key = (str(strategy), str(split_id))
        if key not in split_map:
            continue
        info = split_map[key]
        train_idx = info["train_idx"]
        y_train = y_all.iloc[train_idx].dropna().values.astype(float)
        if len(y_train) < 5:
            continue
        train_mean = float(np.mean(y_train))
        train_std = float(np.std(y_train) + 1e-9)
        train_min, train_max = float(np.min(y_train)), float(np.max(y_train))
        q05, q95 = float(np.quantile(y_train, 0.05)), float(np.quantile(y_train, 0.95))

        for paper, pp in part.groupby("文献编号"):
            obs = pd.to_numeric(pp["Observed"], errors="coerce").dropna().values.astype(float)
            if len(obs) == 0:
                continue
            obs_mean = float(np.mean(obs))
            obs_std = float(np.std(obs))
            mean_z = abs(obs_mean - train_mean) / train_std
            outside_minmax = float(np.mean((obs < train_min) | (obs > train_max)))
            outside_q = float(np.mean((obs < q05) | (obs > q95)))
            extreme_low = float(obs_mean <= q05)
            extreme_high = float(obs_mean >= q95)
            rows.append({
                "Strategy": strategy,
                "Split_ID": split_id,
                "文献编号": str(paper),
                "TargetDiagnosticOnly_n": int(len(obs)),
                "Observed_mean": obs_mean,
                "Observed_std": obs_std,
                "Train_target_mean": train_mean,
                "Train_target_std": train_std,
                "Target_Mean_Z_DiagnosticOnly": float(mean_z),
                "Target_Outside_Train_MinMax_Fraction_DiagnosticOnly": outside_minmax,
                "Target_Outside_Train_Q05Q95_Fraction_DiagnosticOnly": outside_q,
                "Target_Extreme_Low_DiagnosticOnly": extreme_low,
                "Target_Extreme_High_DiagnosticOnly": extreme_high,
                "Small_Paper_Flag_DiagnosticOnly": int(len(obs) < 5),
                "Zero_Or_NearZero_Target_Mean_DiagnosticOnly": int(obs_mean <= 5.0),
                "High_Target_Variance_DiagnosticOnly": int(obs_std >= 25.0),
            })
    return pd.DataFrame(rows)


# ============================================================
# 7. Performance and diagnostics
# ============================================================

def merge_scores_to_predictions(pred_df: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["Strategy", "Split_ID", "Data_Index", "文献编号"]
    score_cols = [c for c in score_df.columns if c not in merge_keys]
    merged = pred_df.merge(score_df[merge_keys + score_cols], on=merge_keys, how="left")
    merged["Abs_Error"] = np.abs(pd.to_numeric(merged["Observed"], errors="coerce") - pd.to_numeric(merged["Predicted"], errors="coerce"))
    return merged


def assign_risk_quantiles(df: pd.DataFrame, score_col: str = "Corrected_AD_Risk", q: int = 5) -> pd.Series:
    vals = pd.to_numeric(df[score_col], errors="coerce")
    try:
        return pd.qcut(vals, q=q, labels=[f"Q{i}" for i in range(1, q + 1)], duplicates="drop")
    except Exception:
        return pd.Series(["Unknown"] * len(df), index=df.index)


def performance_by_risk_quantile(pred_score: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, method), part0 in pred_score.groupby(["Strategy", "Method"], dropna=False):
        part = part0.dropna(subset=["Corrected_AD_Risk", "Observed", "Predicted"]).copy()
        if part.empty:
            continue
        part["Corrected_AD_Risk_Quantile"] = assign_risk_quantiles(part, "Corrected_AD_Risk", q=5)
        for qbin, pp in part.groupby("Corrected_AD_Risk_Quantile", dropna=False):
            m = metrics_dict(pp["Observed"], pp["Predicted"])
            rows.append({
                "Strategy": strategy,
                "Method": method,
                "Risk_Quantile": str(qbin),
                "n": m["n"],
                "Coverage_in_Method_Strategy": float(len(pp) / len(part)),
                "Mean_Corrected_AD_Risk": float(pp["Corrected_AD_Risk"].mean()),
                "Mean_Distance_AD_Risk": float(pp["Distance_AD_Risk"].mean()),
                "Mean_Boundary_AD_Risk": float(pp["Boundary_AD_Risk"].mean()),
                **{k: v for k, v in m.items() if k != "n"},
            })
    return pd.DataFrame(rows)


def score_error_correlations(pred_score: pd.DataFrame) -> pd.DataFrame:
    score_cols = [
        "Corrected_AD_Risk", "Distance_AD_Risk", "Boundary_AD_Risk", "Max_Component_Risk",
        "NN_Risk", "Mahalanobis_Risk", "PCA_Reconstruction_Risk", "Leverage_Risk",
        "Numeric_Boundary_Risk", "Categorical_Unseen_Risk",
        "Numeric_Strict_Range_Violation_Fraction", "Numeric_Quantile_Range_Violation_Fraction",
        "Categorical_Unseen_Fraction",
    ]
    rows = []
    for (strategy, method), part in pred_score.groupby(["Strategy", "Method"], dropna=False):
        part = part.dropna(subset=["Abs_Error"]).copy()
        if len(part) < 10:
            continue
        for sc in score_cols:
            if sc not in part.columns:
                continue
            x = pd.to_numeric(part[sc], errors="coerce")
            y = pd.to_numeric(part["Abs_Error"], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10 or x[mask].nunique() < 2:
                pearson = np.nan
                spearman = np.nan
                auc75 = np.nan
                auc90 = np.nan
            else:
                pearson = float(pd.Series(x[mask]).corr(pd.Series(y[mask]), method="pearson"))
                spearman = float(pd.Series(x[mask]).corr(pd.Series(y[mask]), method="spearman"))
                # High-error detection diagnostic.
                y75 = (y[mask] >= np.nanquantile(y[mask], 0.75)).astype(int)
                y90 = (y[mask] >= np.nanquantile(y[mask], 0.90)).astype(int)
                try:
                    auc75 = float(roc_auc_score(y75, x[mask])) if len(np.unique(y75)) > 1 else np.nan
                    auc90 = float(roc_auc_score(y90, x[mask])) if len(np.unique(y90)) > 1 else np.nan
                except Exception:
                    auc75, auc90 = np.nan, np.nan
            rows.append({
                "Strategy": strategy,
                "Method": method,
                "Score_Component": sc,
                "n": int(mask.sum()),
                "Pearson_with_AbsError": pearson,
                "Spearman_with_AbsError": spearman,
                "AUROC_for_Top25pct_Error_Diagnostic": auc75,
                "AUROC_for_Top10pct_Error_Diagnostic": auc90,
            })
    return pd.DataFrame(rows)


def apply_reliable_rules(pred_score: pd.DataFrame) -> pd.DataFrame:
    out = pred_score.copy()
    for rule_name, thresholds in RELIABLE_RULES.items():
        mask = pd.Series(True, index=out.index)
        for col, thr in thresholds.items():
            if col not in out.columns:
                mask &= False
            else:
                mask &= pd.to_numeric(out[col], errors="coerce").fillna(np.inf) <= float(thr)
        out[rule_name] = mask.astype(int)
    return out


def reliable_rule_performance(pred_score: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rule_cols = list(RELIABLE_RULES.keys())
    for (strategy, method), part in pred_score.groupby(["Strategy", "Method"], dropna=False):
        total = len(part)
        # All-domain baseline.
        m_all = metrics_dict(part["Observed"], part["Predicted"])
        rows.append({
            "Strategy": strategy,
            "Method": method,
            "Rule": "AllDomain_NoFiltering",
            "Coverage_Fraction": 1.0,
            "n_in_domain": total,
            **{k: v for k, v in m_all.items() if k != "n"},
        })
        for rule in rule_cols:
            pp = part[part[rule] == 1]
            m = metrics_dict(pp["Observed"], pp["Predicted"]) if len(pp) else {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "Bias": np.nan, "n": 0}
            rows.append({
                "Strategy": strategy,
                "Method": method,
                "Rule": rule,
                "Coverage_Fraction": float(len(pp) / total) if total else np.nan,
                "n_in_domain": int(len(pp)),
                **{k: v for k, v in m.items() if k != "n"},
            })
    return pd.DataFrame(rows)


def component_rule_sweep(pred_score: pd.DataFrame, primary_methods: List[str]) -> pd.DataFrame:
    rows = []
    thresholds = np.round(np.arange(0.40, 0.91, 0.05), 2)
    for (strategy, method), part in pred_score[pred_score["Method"].isin(primary_methods)].groupby(["Strategy", "Method"], dropna=False):
        total = len(part)
        if total == 0:
            continue
        for score_col in ["Corrected_AD_Risk", "Distance_AD_Risk", "Boundary_AD_Risk", "Max_Component_Risk"]:
            for thr in thresholds:
                pp = part[pd.to_numeric(part[score_col], errors="coerce") <= thr]
                if len(pp) < 10:
                    m = {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "Bias": np.nan, "n": len(pp)}
                else:
                    m = metrics_dict(pp["Observed"], pp["Predicted"])
                rows.append({
                    "Strategy": strategy,
                    "Method": method,
                    "Score_Column": score_col,
                    "Threshold": float(thr),
                    "Coverage_Fraction": float(len(pp) / total),
                    "n_in_domain": int(len(pp)),
                    **{k: v for k, v in m.items() if k != "n"},
                })
    return pd.DataFrame(rows)


def paper_summary_and_taxonomy(pred_score: pd.DataFrame, target_diag: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    component_cols = [
        "Corrected_AD_Risk", "Distance_AD_Risk", "Boundary_AD_Risk", "Max_Component_Risk",
        "NN_Risk", "Mahalanobis_Risk", "PCA_Reconstruction_Risk", "Leverage_Risk",
        "Numeric_Boundary_Risk", "Categorical_Unseen_Risk",
        "Numeric_Strict_Range_Violation_Fraction", "Numeric_Quantile_Range_Violation_Fraction",
        "Categorical_Unseen_Fraction",
    ]
    for (strategy, method, paper), part in pred_score.groupby(["Strategy", "Method", "文献编号"], dropna=False):
        m = metrics_dict(part["Observed"], part["Predicted"])
        row = {
            "Strategy": strategy,
            "Method": method,
            "文献编号": str(paper),
            "n_predictions": int(len(part)),
            "Observed_mean": float(pd.to_numeric(part["Observed"], errors="coerce").mean()),
            "Observed_std": float(pd.to_numeric(part["Observed"], errors="coerce").std(ddof=0)),
            **{f"Paper_{k}": v for k, v in m.items() if k != "n"},
        }
        for c in component_cols:
            if c in part.columns:
                row[f"Mean_{c}"] = float(pd.to_numeric(part[c], errors="coerce").mean())
                row[f"P75_{c}"] = float(pd.to_numeric(part[c], errors="coerce").quantile(0.75))
        for rule in RELIABLE_RULES:
            if rule in part.columns:
                row[f"{rule}_Fraction"] = float(part[rule].mean())
        rows.append(row)
    paper = pd.DataFrame(rows)

    # Merge target diagnostic by Strategy/Split/Paper is too granular; aggregate diagnostic by Strategy/Paper.
    if not target_diag.empty:
        agg_cols = [c for c in target_diag.columns if c.endswith("_DiagnosticOnly") or c in ["TargetDiagnosticOnly_n", "Observed_mean", "Observed_std"]]
        diag_agg = target_diag.groupby(["Strategy", "文献编号"], as_index=False)[agg_cols].mean(numeric_only=True)
        paper = paper.merge(diag_agg, on=["Strategy", "文献编号"], how="left", suffixes=("", "_TargetDiag"))

    # Error-source taxonomy: diagnostic, not deployable.
    tax_rows = []
    for _, r in paper.iterrows():
        reasons = []
        if r.get("Mean_Corrected_AD_Risk", 0) >= 0.65 or r.get("Mean_Distance_AD_Risk", 0) >= 0.75:
            reasons.append("feature_space_out_of_domain")
        if r.get("Mean_Numeric_Strict_Range_Violation_Fraction", 0) >= 0.10 or r.get("Mean_Numeric_Boundary_Risk", 0) >= 0.50:
            reasons.append("numeric_extrapolation")
        if r.get("Mean_Categorical_Unseen_Fraction", 0) >= 0.20 or r.get("Mean_Categorical_Unseen_Risk", 0) >= 0.50:
            reasons.append("categorical_novelty")
        if r.get("TargetDiagnosticOnly_n", 999) < 5 or r.get("n_predictions", 999) < 5:
            reasons.append("small_literature_group")
        if r.get("Zero_Or_NearZero_Target_Mean_DiagnosticOnly", 0) >= 0.5:
            reasons.append("extreme_low_target_diagnostic_only")
        if r.get("Target_Outside_Train_Q05Q95_Fraction_DiagnosticOnly", 0) >= 0.50:
            reasons.append("target_distribution_novelty_diagnostic_only")
        if r.get("High_Target_Variance_DiagnosticOnly", 0) >= 0.5 or r.get("Observed_std", 0) >= 25:
            reasons.append("high_within_paper_target_variance")
        if not reasons:
            reasons.append("unexplained_or_model_bias")

        tax_rows.append({
            "Strategy": r["Strategy"],
            "Method": r["Method"],
            "文献编号": r["文献编号"],
            "Paper_RMSE": r.get("Paper_RMSE", np.nan),
            "Paper_MAE": r.get("Paper_MAE", np.nan),
            "Paper_Bias": r.get("Paper_Bias", np.nan),
            "n_predictions": r.get("n_predictions", np.nan),
            "Mean_Corrected_AD_Risk": r.get("Mean_Corrected_AD_Risk", np.nan),
            "Mean_Distance_AD_Risk": r.get("Mean_Distance_AD_Risk", np.nan),
            "Mean_Boundary_AD_Risk": r.get("Mean_Boundary_AD_Risk", np.nan),
            "Mean_Categorical_Unseen_Fraction": r.get("Mean_Categorical_Unseen_Fraction", np.nan),
            "Mean_Numeric_Strict_Range_Violation_Fraction": r.get("Mean_Numeric_Strict_Range_Violation_Fraction", np.nan),
            "Error_Source_Tags_Diagnostic": ";".join(reasons),
        })
    taxonomy = pd.DataFrame(tax_rows)
    return paper, taxonomy


# ============================================================
# 8. Plotting
# ============================================================

def plot_score_error_relationship(pred_score: pd.DataFrame) -> None:
    primary_methods = [m for m in PRIMARY_METHOD_CANDIDATES if m in set(pred_score["Method"])]
    if not primary_methods:
        primary_methods = list(pred_score["Method"].dropna().unique())[:4]
    plot_df = pred_score[pred_score["Method"].isin(primary_methods)].copy()
    if len(plot_df) > 12000:
        plot_df = plot_df.sample(12000, random_state=RANDOM_STATE)

    fig, axes = plt.subplots(1, 2, figsize=(17, 5.8))
    for i, method in enumerate(primary_methods[:5]):
        pp = plot_df[plot_df["Method"] == method]
        axes[0].scatter(pp["Corrected_AD_Risk"], pp["Abs_Error"], s=12, alpha=0.35, label=method, color=LOW_SAT_COLORS[i % len(LOW_SAT_COLORS)])
    axes[0].set_xlabel("Corrected target-free AD risk score")
    axes[0].set_ylabel("Absolute error in F removal (%)")
    axes[0].set_title("Sample-level error vs corrected AD risk")
    axes[0].grid(True, alpha=0.25)
    move_legend_outside(axes[0], bbox_to_anchor=(1.02, 1.0))

    # Binned mean abs error.
    rows = []
    for method, part in plot_df.groupby("Method"):
        part = part.dropna(subset=["Corrected_AD_Risk", "Abs_Error"]).copy()
        if len(part) < 30:
            continue
        part["RiskQ"] = assign_risk_quantiles(part, "Corrected_AD_Risk", q=5)
        for q, pp in part.groupby("RiskQ"):
            rows.append({"Method": method, "RiskQ": str(q), "MeanAbsError": pp["Abs_Error"].mean(), "n": len(pp)})
    bdf = pd.DataFrame(rows)
    if not bdf.empty:
        q_order = sorted(bdf["RiskQ"].unique())
        x = np.arange(len(q_order))
        width = 0.8 / max(1, len(primary_methods[:4]))
        for i, method in enumerate(primary_methods[:4]):
            vals = []
            for q in q_order:
                tmp = bdf[(bdf["Method"] == method) & (bdf["RiskQ"] == q)]
                vals.append(tmp["MeanAbsError"].iloc[0] if len(tmp) else np.nan)
            axes[1].bar(x + (i - len(primary_methods[:4])/2)*width + width/2, vals, width=width,
                        label=method, color=LOW_SAT_COLORS[i % len(LOW_SAT_COLORS)], alpha=0.9)
            for xi, yi in zip(x + (i - len(primary_methods[:4])/2)*width + width/2, vals):
                if np.isfinite(yi):
                    axes[1].text(xi, yi, f"{yi:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(q_order)
        axes[1].set_xlabel("Corrected AD risk quantile")
        axes[1].set_ylabel("Mean absolute error (%)")
        axes[1].set_title("Mean error by risk quantile")
        axes[1].grid(True, axis="y", alpha=0.25)
        move_legend_outside(axes[1], bbox_to_anchor=(1.02, 1.0))
    for ax in axes:
        prettify_axis(ax, rotate_x=25, ha="right", wrap_x=16, wrap_y=22)
        expand_y_for_text(ax, top_frac=0.16, bottom_frac=0.04)
    fig.suptitle("32A2 corrected applicability-domain risk and prediction error", fontsize=14, fontweight="bold")
    save_fig(fig, "145_32a2_score_error_relationship.png")


def plot_performance_by_risk_quantile(perf_q: pd.DataFrame) -> None:
    if perf_q.empty:
        return
    # Use primary method if present.
    method = "CB-XGB" if "CB-XGB" in set(perf_q["Method"]) else perf_q["Method"].iloc[0]
    plot_df = perf_q[perf_q["Method"] == method].copy()
    strategies = list(plot_df["Strategy"].dropna().unique())[:5]
    plot_df = plot_df[plot_df["Strategy"].isin(strategies)]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))
    metrics = ["R2", "RMSE", "MAE"]
    for ax, met in zip(axes, metrics):
        q_order = sorted(plot_df["Risk_Quantile"].unique())
        x = np.arange(len(q_order))
        width = 0.8 / max(1, len(strategies))
        for i, strategy in enumerate(strategies):
            vals = []
            for q in q_order:
                tmp = plot_df[(plot_df["Strategy"] == strategy) & (plot_df["Risk_Quantile"] == q)]
                vals.append(tmp[met].iloc[0] if len(tmp) else np.nan)
            xpos = x + (i - len(strategies)/2)*width + width/2
            ax.bar(xpos, vals, width=width, label=strategy if ax is axes[0] else None,
                   color=LOW_SAT_COLORS[i % len(LOW_SAT_COLORS)], alpha=0.9)
            for xi, yi in zip(xpos, vals):
                if np.isfinite(yi):
                    ax.text(xi, yi, f"{yi:.2f}" if met == "R2" else f"{yi:.1f}",
                            ha="center", va="bottom" if yi >= 0 else "top", fontsize=7, rotation=90)
        ax.set_title(f"{method}: {met} by corrected AD risk quantile")
        ax.set_xticks(x)
        ax.set_xticklabels(q_order)
        ax.set_xlabel("Risk quantile")
        ax.set_ylabel(met)
        ax.grid(True, axis="y", alpha=0.25)
    move_legend_outside(axes[0], bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("32A2 model performance stratified by corrected AD risk", fontsize=14, fontweight="bold")
    save_fig(fig, "146_32a2_performance_by_risk_quantile.png")


def plot_component_correlation_heatmap(corr_df: pd.DataFrame) -> None:
    if corr_df.empty:
        return
    primary = corr_df[corr_df["Method"].isin(PRIMARY_METHOD_CANDIDATES)].copy()
    if primary.empty:
        primary = corr_df.copy()
    # Mean Spearman across strategies/methods.
    piv = primary.pivot_table(index="Score_Component", columns="Method", values="Spearman_with_AbsError", aggfunc="mean")
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(6, 0.42 * len(piv))))
    im = ax.imshow(piv.values, aspect="auto", vmin=-0.5, vmax=0.5, cmap="coolwarm")
    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Mean Spearman correlation between AD components and absolute error")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spearman correlation", labelpad=12)
    prettify_axis(ax, rotate_x=35, ha="right", wrap_x=18, wrap_y=26, grid_axis="both")
    save_fig(fig, "147_32a2_component_correlation_heatmap.png")


def plot_rule_tradeoff(rule_perf: pd.DataFrame) -> None:
    if rule_perf.empty:
        return
    # Focus on stable tree model if present.
    method = "CB-XGB" if "CB-XGB" in set(rule_perf["Method"]) else rule_perf["Method"].iloc[0]
    plot_df = rule_perf[(rule_perf["Method"] == method) & (rule_perf["Rule"] != "AllDomain_NoFiltering")].copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (rule, part) in enumerate(plot_df.groupby("Rule")):
        ax.scatter(part["Coverage_Fraction"], part["RMSE"], s=80, alpha=0.85,
                   label=rule, color=LOW_SAT_COLORS[i % len(LOW_SAT_COLORS)])
        for _, r in part.iterrows():
            ax.text(r["Coverage_Fraction"], r["RMSE"], str(r["Strategy"])[:12],
                    fontsize=7, alpha=0.8)
    ax.set_xlabel("Coverage fraction retained")
    ax.set_ylabel("Reliable-domain RMSE")
    ax.set_title(f"{method}: component-rule reliable-domain trade-off")
    ax.grid(True, alpha=0.25)
    move_legend_outside(ax, bbox_to_anchor=(1.02, 1.0))
    save_fig(fig, "148_32a2_reliable_rule_tradeoff.png")


def plot_paper_error_source_map(paper_summary: pd.DataFrame, taxonomy: pd.DataFrame) -> None:
    if paper_summary.empty:
        return
    method = "CB-XGB" if "CB-XGB" in set(paper_summary["Method"]) else paper_summary["Method"].iloc[0]
    strategy = "LOLO_LeaveOneLiteratureOut" if "LOLO_LeaveOneLiteratureOut" in set(paper_summary["Strategy"]) else paper_summary["Strategy"].iloc[0]
    df = paper_summary[(paper_summary["Method"] == method) & (paper_summary["Strategy"] == strategy)].copy()
    if df.empty:
        df = paper_summary[paper_summary["Method"] == method].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 7.5))
    size = 40 + 4 * pd.to_numeric(df["n_predictions"], errors="coerce").fillna(1).clip(1, 100)
    sc = ax.scatter(df["Mean_Corrected_AD_Risk"], df["Paper_RMSE"], s=size,
                    c=df["Mean_Boundary_AD_Risk"], cmap="YlOrRd", alpha=0.75, edgecolor="black", linewidth=0.4)
    # Label highest RMSE papers.
    for _, r in df.sort_values("Paper_RMSE", ascending=False).head(12).iterrows():
        ax.text(r["Mean_Corrected_AD_Risk"], r["Paper_RMSE"], str(r["文献编号"]), fontsize=8)
    ax.set_xlabel("Mean corrected target-free AD risk by paper")
    ax.set_ylabel("Paper-level RMSE")
    ax.set_title(f"{strategy} / {method}: paper-level RMSE vs corrected AD risk")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Mean boundary AD risk", labelpad=12)
    prettify_axis(ax, rotate_x=0, ha="center", wrap_x=24, wrap_y=22, grid_axis="both")
    save_fig(fig, "149_32a2_paper_error_source_map.png")


def plot_violation_drivers(range_summary: pd.DataFrame, cat_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(21, 9))
    if not range_summary.empty:
        rtop = range_summary.sort_values("Total_Strict_Violations", ascending=False).head(15)
        axes[0].barh(np.arange(len(rtop)), rtop["Total_Strict_Violations"], color="#9bb7cf", edgecolor="black", linewidth=0.4)
        axes[0].set_yticks(np.arange(len(rtop)))
        axes[0].set_yticklabels([wrap_plot_text(x, width=18) for x in rtop["Feature"]])
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Total strict range violations")
        axes[0].set_title("Cleaned numeric range-violation drivers")
        for i, v in enumerate(rtop["Total_Strict_Violations"]):
            axes[0].text(v, i, f"{int(v)}", va="center", fontsize=8)
    else:
        axes[0].axis("off")
    if not cat_summary.empty:
        ctop = cat_summary.sort_values("Total_Count", ascending=False).head(15)
        axes[1].barh(np.arange(len(ctop)), ctop["Total_Count"], color="#d1b277", edgecolor="black", linewidth=0.4)
        axes[1].set_yticks(np.arange(len(ctop)))
        axes[1].set_yticklabels([wrap_plot_text(x, width=22) for x in (ctop["Feature"].astype(str) + " = " + ctop["Unseen_Value"].astype(str))])
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Total unseen-category count")
        axes[1].set_title("Categorical novelty drivers")
        for i, v in enumerate(ctop["Total_Count"]):
            axes[1].text(v, i, f"{int(v)}", va="center", fontsize=8)
    else:
        axes[1].axis("off")
    for ax in axes:
        prettify_axis(ax, rotate_x=0, ha="center", wrap_x=20, wrap_y=18)
    fig.suptitle("32A2 corrected applicability-domain violation drivers", fontsize=14, fontweight="bold")
    save_fig(fig, "150_32a2_violation_drivers_cleaned.png")


def plot_model_specific_reliable_domain(rule_perf: pd.DataFrame) -> None:
    if rule_perf.empty:
        return
    # Compare all domain vs balanced rule across methods for LOLO if available.
    strategy = "LOLO_LeaveOneLiteratureOut" if "LOLO_LeaveOneLiteratureOut" in set(rule_perf["Strategy"]) else rule_perf["Strategy"].iloc[0]
    methods = [m for m in PRIMARY_METHOD_CANDIDATES if m in set(rule_perf["Method"])]
    if not methods:
        methods = list(rule_perf["Method"].dropna().unique())[:5]
    rules = ["AllDomain_NoFiltering", "Rule_B_balanced_component", "Rule_C_distance_focused", "Rule_D_boundary_focused"]
    df = rule_perf[(rule_perf["Strategy"] == strategy) & (rule_perf["Method"].isin(methods)) & (rule_perf["Rule"].isin(rules))].copy()
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.0))
    metrics = ["R2", "RMSE", "MAE"]
    for ax, met in zip(axes, metrics):
        x = np.arange(len(methods))
        width = 0.8 / len(rules)
        for i, rule in enumerate(rules):
            vals = []
            for m in methods:
                tmp = df[(df["Method"] == m) & (df["Rule"] == rule)]
                vals.append(tmp[met].iloc[0] if len(tmp) else np.nan)
            xpos = x + (i - len(rules)/2) * width + width/2
            ax.bar(xpos, vals, width=width, color=LOW_SAT_COLORS[i % len(LOW_SAT_COLORS)], label=rule if ax is axes[0] else None)
            for xi, yi in zip(xpos, vals):
                if np.isfinite(yi):
                    ax.text(xi, yi, f"{yi:.2f}" if met == "R2" else f"{yi:.1f}",
                            ha="center", va="bottom" if yi >= 0 else "top", fontsize=7, rotation=90)
        ax.set_title(f"{strategy}: {met}")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    move_legend_outside(axes[0], bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("32A2 model-specific reliable-domain comparison", fontsize=14, fontweight="bold")
    save_fig(fig, "151_32a2_model_specific_reliable_domain.png")


# ============================================================
# 9. Decision report
# ============================================================

def write_decision_report(
    feature_path: Path,
    pred_path: Path,
    split_path: Optional[Path],
    reference_mode: str,
    audit: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    corr_df: pd.DataFrame,
    perf_q: pd.DataFrame,
    rule_perf: pd.DataFrame,
    paper_summary: pd.DataFrame,
    taxonomy: pd.DataFrame,
    range_summary: pd.DataFrame,
    cat_summary: pd.DataFrame,
) -> None:
    lines = []
    lines.append("=" * 100)
    lines.append("Step 32A2 corrected applicability-domain analysis report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Rebuild the target-free applicability-domain analysis after Step 32A showed a weak/non-monotonic AD-error relationship.")
    lines.append("")
    lines.append("Strictness:")
    lines.append("- No samples or papers are removed from all-domain results.")
    lines.append("- Main AD scores are target-free and do not use observed F removal or test residuals.")
    lines.append("- Target-distribution novelty is diagnostic only and must not be used for deployable reliable-domain screening.")
    lines.append("- 文献编号 is used only for grouped-domain construction and diagnostics, not as an input feature.")
    lines.append("")
    lines.append(f"Feature table used: {feature_path}")
    lines.append(f"Prediction table used: {pred_path}")
    lines.append(f"Split table used: {split_path if split_path else 'None'}")
    lines.append(f"Reference-domain reconstruction mode: {reference_mode}")
    lines.append("")
    lines.append("Corrected AD feature space:")
    lines.append(f"- Included numeric features: {len(numeric_features)}")
    lines.append(f"- Included categorical features: {len(categorical_features)}")
    excluded_index = audit[(audit["Included_in_AD"] == False) & (audit["Reason"].astype(str).str.contains("index|row_order|identifier|duplicated_raw", case=False, regex=True))]
    lines.append(f"- Removed index/id/raw-copy/leakage columns: {len(excluded_index)}")
    if len(excluded_index):
        lines.append("  Examples: " + ", ".join(excluded_index["Feature_Table_Column"].head(12).astype(str).tolist()))
    lines.append("")
    lines.append("Deployment AD components:")
    lines.append(str(DEPLOYABLE_SCORE_WEIGHTS))
    lines.append("")
    lines.append("Reliable-domain rules evaluated:")
    for rule, thr in RELIABLE_RULES.items():
        lines.append(f"- {rule}: {thr}")
    lines.append("")

    # Correlation summary.
    if not corr_df.empty:
        primary = corr_df[corr_df["Method"].isin(PRIMARY_METHOD_CANDIDATES)].copy()
        comp = primary.groupby("Score_Component", as_index=False).agg(
            Mean_Spearman=("Spearman_with_AbsError", "mean"),
            Mean_AUROC_Top25=("AUROC_for_Top25pct_Error_Diagnostic", "mean"),
            n=("Spearman_with_AbsError", "count"),
        ).sort_values("Mean_Spearman", ascending=False)
        lines.append("Mean association between AD components and absolute error:")
        lines.append(comp.to_string(index=False))
        lines.append("")

    # Rule performance summary.
    if not rule_perf.empty:
        for strat in ["LOLO_LeaveOneLiteratureOut", "GroupKFold_FullyGrouped", "RepeatedOrdinary_GroupedSplit", "RepeatedBalanced_Extended", "Balanced_5Seed_Current31F"]:
            tmp = rule_perf[(rule_perf["Strategy"] == strat) & (rule_perf["Rule"].isin(["AllDomain_NoFiltering", "Rule_B_balanced_component", "Rule_C_distance_focused"]))]
            if tmp.empty:
                continue
            tmp = tmp[tmp["Method"].isin(PRIMARY_METHOD_CANDIDATES)]
            if tmp.empty:
                continue
            best = tmp.sort_values(["Rule", "RMSE"]).groupby("Rule").head(1)
            lines.append(f"Best model by RMSE under strategy: {strat}")
            lines.append(best[["Rule", "Method", "Coverage_Fraction", "R2", "RMSE", "MAE", "Bias", "n_in_domain"]].to_string(index=False))
            lines.append("")

    if not taxonomy.empty:
        lines.append("High-error taxonomy examples for LOLO/CB-XGB or available primary method:")
        t = taxonomy.copy()
        if "LOLO_LeaveOneLiteratureOut" in set(t["Strategy"]):
            t = t[t["Strategy"] == "LOLO_LeaveOneLiteratureOut"]
        if "CB-XGB" in set(t["Method"]):
            t = t[t["Method"] == "CB-XGB"]
        lines.append(t.sort_values("Paper_RMSE", ascending=False).head(20).to_string(index=False))
        lines.append("")

    if not range_summary.empty:
        lines.append("Top cleaned numeric range-violation drivers:")
        lines.append(range_summary.sort_values("Total_Strict_Violations", ascending=False).head(20).to_string(index=False))
        lines.append("")
    if not cat_summary.empty:
        lines.append("Top categorical novelty drivers:")
        lines.append(cat_summary.sort_values("Total_Count", ascending=False).head(20).to_string(index=False))
        lines.append("")

    lines.append("Interpretation guide:")
    lines.append("- If corrected AD risk/components still do not correlate positively with errors, do not claim a simple monotonic AD-error relationship.")
    lines.append("- If a component-wise rule reduces RMSE at reasonable coverage, it may be used as a candidate reliable-domain rule in 32B.")
    lines.append("- If high-error papers are dominated by target-distribution or small-paper flags, discuss them as data/regime limitations rather than target-free AD failures.")
    lines.append("")
    lines.append("Recommended next step:")
    lines.append("If 32A2 identifies a defensible rule, proceed to 32B reliable-domain definition. If not, proceed to 32B as a cautious domain-stratified reliability analysis, not as a hard filtering claim.")
    lines.append("=" * 100)

    path = TABLE_DIR / "32a2_ad_decision_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[REPORT] {path}")


# ============================================================
# 10. Main workflow
# ============================================================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    set_est_style_fonts()

    feature_df, feature_path, source_col, target_col = load_feature_table()
    pred_df, pred_path = load_prediction_table()
    split_df, split_path = load_split_table()

    # If prediction fallback is 32A enriched table, keep only raw prediction rows
    # and deduplicate exact repeated enriched rows.
    pred_base_cols = ["Strategy", "Split_ID", "Seed_or_Fold", "Method", "Data_Index", "Local_Test_Index", "文献编号", "Observed", "Predicted", "Residual", "Abs_Error"]
    pred_cols = [c for c in pred_base_cols if c in pred_df.columns]
    pred_df = pred_df[pred_cols].drop_duplicates().copy()

    audit, numeric_features, categorical_features = corrected_feature_space_audit(feature_df, source_col, target_col)
    chem_flags = build_core_chemical_feature_flags(feature_df, numeric_features, categorical_features)
    audit = audit.merge(chem_flags, left_on="Feature_Table_Column", right_on="Feature", how="left")
    audit.drop(columns=[c for c in ["Feature"] if c in audit.columns], inplace=True)
    safe_to_csv(audit, "32a2_ad_feature_space_audit.csv")

    log(f"[AD] Corrected AD features: numeric={len(numeric_features)}, categorical={len(categorical_features)}")
    if len(numeric_features) == 0 and len(categorical_features) == 0:
        raise RuntimeError("No AD features available after corrected audit.")

    split_map, reference_mode = build_split_reference_map(feature_df, pred_df, split_df, source_col)

    # Compute target-free AD scores for each unique split/sample pair.
    score_frames = []
    range_frames = []
    cat_frames = []

    unique_split_parts = list(pred_df.groupby(["Strategy", "Split_ID"], dropna=False))
    log(f"[AD] Calculating corrected AD scores for {len(unique_split_parts)} unique strategy/split pairs...")

    for n_done, ((strategy, split_id), part) in enumerate(unique_split_parts, start=1):
        key = (str(strategy), str(split_id))
        info = split_map[key]
        test_indices = np.array(sorted(part["Data_Index"].dropna().astype(int).unique()), dtype=int)
        train_idx = np.asarray(info["train_idx"], dtype=int)

        # Ensure test indices are valid.
        test_indices = test_indices[(test_indices >= 0) & (test_indices < len(feature_df))]
        train_idx = train_idx[(train_idx >= 0) & (train_idx < len(feature_df))]

        if len(test_indices) == 0:
            continue
        try:
            score, range_rec, cat_rec = calculate_ad_for_split(
                feature_df=feature_df,
                source_col=source_col,
                target_col=target_col,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                strategy=str(strategy),
                split_id=str(split_id),
                train_idx=train_idx,
                test_indices=test_indices,
                reference_mode=info["Reference_Mode"],
            )
            score_frames.append(score)
            if not range_rec.empty:
                range_frames.append(range_rec)
            if not cat_rec.empty:
                cat_frames.append(cat_rec)
        except Exception as e:
            log(f"[WARN] Failed AD calculation for {strategy} / {split_id}: {e}")
            continue

        if n_done % 5 == 0 or n_done == len(unique_split_parts):
            log(f"[AD] Completed {n_done}/{len(unique_split_parts)} split pairs.")

    if not score_frames:
        raise RuntimeError("No AD score frames were produced.")
    score_df = pd.concat(score_frames, ignore_index=True)
    range_records = pd.concat(range_frames, ignore_index=True) if range_frames else pd.DataFrame()
    cat_records = pd.concat(cat_frames, ignore_index=True) if cat_frames else pd.DataFrame()

    # Save sample scores.
    safe_to_csv(score_df, "32a2_ad_sample_scores.csv")

    # Summarize violations.
    if not range_records.empty:
        range_summary = range_records.groupby("Feature", as_index=False).agg(
            Total_Test_Folds=("Split_ID", "count"),
            Total_Strict_Violations=("Strict_Violations", "sum"),
            Total_Quantile_Violations=("Quantile_Violations", "sum"),
            Mean_Strict_Violation_Fraction=("Strict_Violation_Fraction", "mean"),
            Mean_Quantile_Violation_Fraction=("Quantile_Violation_Fraction", "mean"),
            Mean_Strict_Severity=("Mean_Strict_Severity", "mean"),
            Mean_Quantile_Severity=("Mean_Quantile_Severity", "mean"),
            Max_Strict_Severity=("Max_Strict_Severity", "max"),
        ).sort_values("Total_Strict_Violations", ascending=False)
    else:
        range_summary = pd.DataFrame()
    if not cat_records.empty:
        cat_summary = cat_records.groupby(["Feature", "Unseen_Value"], as_index=False).agg(
            Total_Count=("Total_Count", "sum"),
            Mean_Fraction_in_Test=("Fraction_in_Test", "mean"),
            Appeared_in_Splits=("Split_ID", "nunique"),
        ).sort_values("Total_Count", ascending=False)
    else:
        cat_summary = pd.DataFrame()

    safe_to_csv(range_summary, "32a2_ad_feature_range_violation_summary.csv")
    safe_to_csv(cat_summary, "32a2_ad_category_unseen_summary.csv")

    # Merge to predictions.
    pred_score = merge_scores_to_predictions(pred_df, score_df)
    pred_score = apply_reliable_rules(pred_score)

    # Target novelty diagnostic only.
    target_diag = calculate_target_diagnostics(feature_df, source_col, target_col, pred_df, split_map)
    if not target_diag.empty:
        safe_to_csv(target_diag, "32a2_target_distribution_diagnostic_only.csv")

    safe_to_csv(pred_score, "32a2_ad_predictions_with_scores.csv")

    # Main diagnostics.
    corr_df = score_error_correlations(pred_score)
    perf_q = performance_by_risk_quantile(pred_score)
    rule_perf = reliable_rule_performance(pred_score)
    sweep = component_rule_sweep(pred_score, PRIMARY_METHOD_CANDIDATES)
    paper_summary, taxonomy = paper_summary_and_taxonomy(pred_score, target_diag)

    safe_to_csv(corr_df, "32a2_ad_score_error_correlation.csv")
    safe_to_csv(perf_q, "32a2_ad_performance_by_risk_quantile.csv")
    safe_to_csv(rule_perf, "32a2_ad_reliable_rule_performance.csv")
    safe_to_csv(sweep, "32a2_ad_component_rule_sweep.csv")
    safe_to_csv(paper_summary, "32a2_ad_paper_summary.csv")
    safe_to_csv(taxonomy, "32a2_ad_error_source_taxonomy.csv")

    # Figures.
    plot_score_error_relationship(pred_score)
    plot_performance_by_risk_quantile(perf_q)
    plot_component_correlation_heatmap(corr_df)
    plot_rule_tradeoff(rule_perf)
    plot_paper_error_source_map(paper_summary, taxonomy)
    plot_violation_drivers(range_summary, cat_summary)
    plot_model_specific_reliable_domain(rule_perf)

    # Decision report.
    write_decision_report(
        feature_path=feature_path,
        pred_path=pred_path,
        split_path=split_path,
        reference_mode=reference_mode,
        audit=audit,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        corr_df=corr_df,
        perf_q=perf_q,
        rule_perf=rule_perf,
        paper_summary=paper_summary,
        taxonomy=taxonomy,
        range_summary=range_summary,
        cat_summary=cat_summary,
    )

    elapsed = time.time() - SCRIPT_START
    log("=" * 100)
    log(f"32A2 corrected applicability-domain analysis finished.")
    log(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    log("=" * 100)


if __name__ == "__main__":
    main()
