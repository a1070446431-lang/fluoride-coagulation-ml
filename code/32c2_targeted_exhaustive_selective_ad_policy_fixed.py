# -*- coding: utf-8 -*-
"""
32c2_targeted_exhaustive_selective_ad_policy.py

Purpose
-------
Step 32C2 for the fluoride coagulation ML project.

This script is the corrected follow-up to 32C-fast. 32C-fast showed that the
best strict selective policy reached R2≈0.475 but did not exhaust targeted rule
combinations. This script performs a broader targeted exhaustive search over
scientifically defensible selective-prediction rules.

What is strict here
-------------------
1. The script does NOT retrain any prediction model.
2. It reuses row-level grouped-validation predictions and AD-score outputs from
   Step 32A2/32B.
3. Candidate policy rules use only deployment-available information:
      - AD / applicability-domain descriptors;
      - novelty / range-violation descriptors;
      - model prediction-disagreement descriptors;
      - optionally model prediction-output descriptors.
   It never uses held-out observed values, errors, residuals, RMSE, MAE or R2 to
   define policy features.
4. For strict policy evaluation, policy selection is done by grouped block-wise
   calibration. For each held-out block, the best rule is chosen only from the
   other blocks. The held-out block is used only for final evaluation.
5. Diagnostic oracle results are reported separately and clearly marked as
   NON-DEPLOYABLE. They are not final evidence.

Why this is not a simplified version
------------------------------------
Compared with 32C-fast, this script expands the policy search space to include:
    - single rules;
    - two-rule AND rules;
    - three-rule AND rules;
    - reliability-zone combined rules;
    - AD-risk + disagreement combined rules;
    - AD-risk + prediction-output combined rules;
    - method-specific and strategy-specific strict calibration;
    - multiple minimum-coverage constraints.

It also fixes the plotting problem from 32C-fast: long policy names are never
placed directly on axes. Figures use short P01/P02 labels and export policy
mapping to CSV/TXT.

Required input
--------------
Preferred:
    outputs/tables/32b_predictions_with_reliability_zone.csv
Fallbacks:
    outputs/tables/32a2_ad_predictions_with_scores.csv
    outputs/tables/32a_ad_predictions_with_scores.csv

Outputs
-------
Tables:
    outputs/tables/32c2_policy_feature_audit.csv
    outputs/tables/32c2_candidate_policy_catalog.csv
    outputs/tables/32c2_blockwise_selected_policy_summary.csv
    outputs/tables/32c2_policy_summary.csv
    outputs/tables/32c2_selected_holdout_predictions.csv
    outputs/tables/32c2_diagnostic_oracle_upper_bound.csv
    outputs/tables/32c2_decision_report.txt

Figures:
    outputs/figures/162_32c2_exhaustive_policy_metrics.png/.pdf
    outputs/figures/163_32c2_r2_coverage_frontier.png/.pdf
    outputs/figures/164_32c2_best_prediction_panel.png/.pdf
    outputs/figures/165_32c2_policy_family_performance.png/.pdf

Run time is printed at the end.
"""

from __future__ import annotations

import itertools
import math
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_NAME = "32c2_targeted_exhaustive_selective_ad_policy.py"
START_TIME = time.time()

# Serious policy must retain at least this fraction within calibration and held-out blocks.
# These are evaluated separately. 0.15 is kept because 32C-fast was close to target;
# 0.20/0.25/0.30 are included to check whether a less fragile rule exists.
MIN_COVERAGE_GRID = [0.15, 0.20, 0.25, 0.30]
MIN_CALIB_N = 40
MIN_HOLDOUT_TOTAL_N = 40

# Main target requested by user. The report will explicitly say whether it is reached.
TARGET_R2 = 0.50

# Search grids. Dense enough to avoid missing obvious thresholds, but controlled to
# avoid selection-overfitting from millions of nearly identical rules.
NUMERIC_Q_GRID_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                      0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                      0.85, 0.90, 0.95]
NUMERIC_Q_GRID_COMBO = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                       0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
PRED_Q_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
               0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

OBJECTIVES = [
    "R2_oriented",
    "RMSE_oriented",
    "Balanced_R2_RMSE",
]


# Default 32C2 scope: targeted exhaustive search around the 32C-fast near-success route.
# This is deliberate: Balanced_5Seed_Current31F + KRR_Poly was the only strict route
# close to R2=0.5. Later 32C3 should verify any winning rule across other grouped
# validation strategies. Set to None to evaluate all available strategies/methods.
TARGET_STRATEGIES = ["Balanced_5Seed_Current31F"]
RUN_DIAGNOSTIC_ORACLE = False  # default off because oracle is not deployable; strict search remains complete for the targeted scope.

TARGET_METHODS = [
    "KRR_Poly",
    "KRR_CatBoost_Weighted_2Model",
    "KRR_XGBoost_Weighted_2Model",
]

PREFERRED_METHOD_ORDER = [
    "KRR_Poly",
    "KRR_CatBoost_Weighted_2Model",
    "KRR_XGBoost_Weighted_2Model",
    "CB-XGB",
    "CatBoost",
    "XGBoost",
    "GPR_Matern",
    "PLS",
    "KRR_RBF",
    "GPR_RBF",
]

# Candidate volume controls. These do not remove rule families; they keep the
# targeted exhaustive search computationally stable.
MAX_AD_FEATURES_FOR_SINGLE = 16
MAX_AD_FEATURES_FOR_COMBO = 9
MAX_DISAGREEMENT_FEATURES_FOR_COMBO = 5
MAX_PREDICTION_FEATURES_FOR_COMBO = 5
MAX_POLICIES_TO_SCORE = 900

# Plot style: publication-friendly, no long labels on axes, large spacing.
SAVE_PDF_FIGURES = False

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "axes.linewidth": 1.0,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# General utilities
# =============================================================================


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_name(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def find_project_root() -> Path:
    candidates: List[Path] = []
    try:
        candidates.append(Path(__file__).resolve())
    except NameError:
        pass
    candidates.append(Path.cwd().resolve())
    for cand in candidates:
        cur = cand if cand.is_dir() else cand.parent
        for p in [cur] + list(cur.parents):
            if (p / "outputs" / "tables").exists() or (p / "data").exists():
                return p
    return Path.cwd().resolve()


def ensure_dirs(root: Path) -> Tuple[Path, Path]:
    fig_dir = root / "outputs" / "figures"
    table_dir = root / "outputs" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, table_dir


def read_csv_robust(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {path} | shape={df.shape}")


def save_fig(fig: plt.Figure, path_png: Path) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    path_pdf = path_png.with_suffix(".pdf")
    fig.savefig(path_png, dpi=600, bbox_inches="tight")
    if SAVE_PDF_FIGURES:
        fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] {path_png}")
    if SAVE_PDF_FIGURES:
        log(f"[FIG] {path_pdf}")


def find_col(df: pd.DataFrame, aliases: Sequence[str], required: bool = True) -> Optional[str]:
    norm_map = {normalize_name(c): c for c in df.columns}
    for a in aliases:
        n = normalize_name(a)
        if n in norm_map:
            return norm_map[n]
    for c in df.columns:
        nc = normalize_name(c)
        for a in aliases:
            na = normalize_name(a)
            if na and na in nc:
                return c
    if required:
        raise ValueError(f"Cannot find required column from aliases: {aliases}")
    return None


def to_float_array(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def safe_r2(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() < 3:
        return np.nan
    yy = y[mask]
    pp = p[mask]
    sst = float(np.sum((yy - np.mean(yy)) ** 2))
    if sst <= 1e-12:
        return np.nan
    sse = float(np.sum((yy - pp) ** 2))
    return 1.0 - sse / sst


def safe_rmse(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y[mask] - p[mask]) ** 2)))


def safe_mae(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y[mask] - p[mask])))


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(p)
    return {
        "N_Retained": int(mask.sum()),
        "R2": safe_r2(y, p),
        "RMSE": safe_rmse(y, p),
        "MAE": safe_mae(y, p),
    }


def vectorized_metrics(y: np.ndarray, p: np.ndarray, mask_mat: np.ndarray) -> pd.DataFrame:
    """Metrics for each candidate column in mask_mat."""
    valid = np.isfinite(y) & np.isfinite(p)
    if valid.sum() == 0:
        n_pol = mask_mat.shape[1]
        return pd.DataFrame({
            "n": np.zeros(n_pol, dtype=int),
            "coverage": np.zeros(n_pol, dtype=float),
            "R2": np.full(n_pol, np.nan),
            "RMSE": np.full(n_pol, np.nan),
            "MAE": np.full(n_pol, np.nan),
        })

    yv = y[valid].astype(float)
    pv = p[valid].astype(float)
    mm = mask_mat[valid, :].astype(float, copy=False)
    n = mm.sum(axis=0)
    coverage = n / float(len(yv))

    err = yv - pv
    err2 = err ** 2
    abs_err = np.abs(err)
    y2 = yv ** 2

    sum_y = mm.T @ yv
    sum_y2 = mm.T @ y2
    sum_err2 = mm.T @ err2
    sum_abs_err = mm.T @ abs_err

    with np.errstate(divide="ignore", invalid="ignore"):
        rmse = np.sqrt(sum_err2 / n)
        mae = sum_abs_err / n
        sst = sum_y2 - (sum_y ** 2) / n
        r2 = 1.0 - sum_err2 / sst

    bad = (n < 3) | (~np.isfinite(r2)) | (sst <= 1e-12)
    r2[bad] = np.nan
    rmse[(n < 1) | (~np.isfinite(rmse))] = np.nan
    mae[(n < 1) | (~np.isfinite(mae))] = np.nan

    return pd.DataFrame({
        "n": n.astype(int),
        "coverage": coverage,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
    })


def winsorized_quantiles(x: pd.Series, qs: Sequence[float]) -> List[float]:
    vals = pd.to_numeric(x, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0 or vals.nunique() <= 1:
        return []
    out = []
    for q in qs:
        v = float(vals.quantile(q))
        if np.isfinite(v):
            out.append(v)
    # unique with rounding, preserving order
    seen = set()
    uniq = []
    for v in out:
        key = round(v, 10)
        if key not in seen:
            seen.add(key)
            uniq.append(v)
    return uniq


def short_text(s: object, n: int = 52) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 3] + "..."


# =============================================================================
# Policy objects and construction
# =============================================================================


@dataclass(frozen=True)
class Condition:
    col: str
    op: str
    value: object

    def text(self) -> str:
        if self.op == "eq":
            return f"{self.col} == {self.value}"
        if self.op == "in":
            return f"{self.col} in {list(self.value)}"
        if self.op == "between":
            a, b = self.value
            return f"{a:.4g} <= {self.col} <= {b:.4g}"
        try:
            return f"{self.col} {self.op} {float(self.value):.4g}"
        except Exception:
            return f"{self.col} {self.op} {self.value}"


@dataclass(frozen=True)
class Policy:
    name: str
    family: str
    conditions: Tuple[Condition, ...]
    search_mode: str
    uses_prediction_output: bool
    description: str


def condition_mask(df: pd.DataFrame, cond: Condition) -> np.ndarray:
    if cond.col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    if cond.op == "eq":
        return df[cond.col].astype(str).to_numpy() == str(cond.value)
    if cond.op == "in":
        vals = set(str(v) for v in cond.value)
        return df[cond.col].astype(str).isin(vals).to_numpy()
    x = pd.to_numeric(df[cond.col], errors="coerce").to_numpy(dtype=float)
    if cond.op == "<=":
        return np.isfinite(x) & (x <= float(cond.value))
    if cond.op == ">=":
        return np.isfinite(x) & (x >= float(cond.value))
    if cond.op == "between":
        lo, hi = cond.value
        return np.isfinite(x) & (x >= float(lo)) & (x <= float(hi))
    raise ValueError(f"Unknown condition op: {cond.op}")


def policy_mask_matrix(df: pd.DataFrame, policies: Sequence[Policy]) -> np.ndarray:
    cache: Dict[Condition, np.ndarray] = {}
    mats = []
    for pol in policies:
        if not pol.conditions:
            mats.append(np.ones(len(df), dtype=bool))
            continue
        m = np.ones(len(df), dtype=bool)
        for cond in pol.conditions:
            if cond not in cache:
                cache[cond] = condition_mask(df, cond)
            m &= cache[cond]
        mats.append(m)
    return np.column_stack(mats) if mats else np.empty((len(df), 0), dtype=bool)


def is_outcome_or_leakage_col(col: str, core_cols: Sequence[str]) -> bool:
    nc = normalize_name(col)
    forbidden = [
        "observed", "actual", "true", "target", "label", "removalrate",
        "residual", "abserror", "absoluteerror", "squarederror", "rmse", "mae", "mse", "r2",
        "oracle", "diagnostic", "paperlevel", "error", "heldout",
    ]
    if any(tok in nc for tok in forbidden):
        return True
    # do not use row/order identifiers as policy features
    if any(tok in nc for tok in ["dataindex", "localtestindex", "row", "index", "originalrow"]):
        return True
    if col in core_cols:
        return True
    return False


def categorize_policy_feature(col: str) -> str:
    nc = normalize_name(col)
    if col == "Predicted" or any(tok in nc for tok in ["predictedvalue", "predlevel", "predhigh", "preddistance"]):
        return "prediction_output"
    if any(tok in nc for tok in ["disagreement", "predsd", "predrange", "prediqr", "predspread"]):
        return "prediction_disagreement"
    if any(tok in nc for tok in ["zone", "domainflag", "reliability"]):
        return "reliability_zone"
    if any(tok in nc for tok in ["ad", "risk", "novel", "unseen", "violation", "distance", "mahalanobis", "boundary", "pca", "reconstruction", "nn", "leverage"]):
        return "ad_descriptor"
    return "other"


def feature_priority_score(col: str) -> float:
    nc = normalize_name(col)
    score = 0.0
    preferred_tokens = [
        "correctedad", "boundary", "adriskscore", "nnrisk", "mahalanobisrisk",
        "pcareconstructionrisk", "numericstrictrangeviolation", "numericrangeviolation",
        "categoricalunseen", "preddisagreement", "composite", "predicted",
    ]
    for i, tok in enumerate(preferred_tokens):
        if tok in nc:
            score += 100 - i
    if categorize_policy_feature(col) == "prediction_output":
        score += 50
    if categorize_policy_feature(col) == "prediction_disagreement":
        score += 45
    if categorize_policy_feature(col) == "ad_descriptor":
        score += 40
    return score


def minmax01(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x)
    out = np.zeros(len(x), dtype=float)
    if valid.sum() == 0:
        return out
    lo = np.nanmin(x[valid])
    hi = np.nanmax(x[valid])
    if hi <= lo:
        out[valid] = 0.0
    else:
        out[valid] = (x[valid] - lo) / (hi - lo)
    out[~valid] = 1.0
    return out


def add_prediction_and_composite_features(
    df: pd.DataFrame,
    strategy_col: str,
    method_col: str,
    pred_col: str,
    key_cols: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()

    # Method-specific prediction-output features. These are target-free because
    # they are available at deployment after the model predicts.
    out["Predicted_Value"] = pd.to_numeric(out[pred_col], errors="coerce")
    out["Predicted_Highness"] = out["Predicted_Value"]
    out["Predicted_Lowness"] = -out["Predicted_Value"]
    out["Predicted_Distance_From_50"] = (out["Predicted_Value"] - 50.0).abs()

    new_cols = ["Predicted_Value", "Predicted_Highness", "Predicted_Lowness", "Predicted_Distance_From_50"]

    # Cross-model disagreement features for the same strategy/split/sample.
    usable_keys = [c for c in key_cols if c in out.columns]
    if len(usable_keys) >= 2:
        try:
            tmp = out[usable_keys + [method_col, pred_col]].copy()
            tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")
            wide = tmp.pivot_table(index=usable_keys, columns=method_col, values=pred_col, aggfunc="mean")
            stats = pd.DataFrame(index=wide.index)
            stats["Pred_Disagreement_SD"] = wide.std(axis=1, skipna=True).fillna(0.0)
            stats["Pred_Disagreement_Range"] = (wide.max(axis=1, skipna=True) - wide.min(axis=1, skipna=True)).fillna(0.0)
            stats["Pred_Disagreement_IQR"] = (wide.quantile(0.75, axis=1) - wide.quantile(0.25, axis=1)).fillna(0.0)
            stats["Pred_Ensemble_Mean"] = wide.mean(axis=1, skipna=True).fillna(0.0)
            stats["Pred_Ensemble_Distance_From_50"] = (stats["Pred_Ensemble_Mean"] - 50.0).abs()
            stats = stats.reset_index()
            out = out.merge(stats, on=usable_keys, how="left")
            new_cols += ["Pred_Disagreement_SD", "Pred_Disagreement_Range", "Pred_Disagreement_IQR",
                         "Pred_Ensemble_Mean", "Pred_Ensemble_Distance_From_50"]
        except Exception as exc:
            log(f"[WARN] Failed to add prediction-disagreement features: {exc}")
            for c in ["Pred_Disagreement_SD", "Pred_Disagreement_Range", "Pred_Disagreement_IQR",
                      "Pred_Ensemble_Mean", "Pred_Ensemble_Distance_From_50"]:
                out[c] = 0.0
                new_cols.append(c)
    else:
        for c in ["Pred_Disagreement_SD", "Pred_Disagreement_Range", "Pred_Disagreement_IQR",
                  "Pred_Ensemble_Mean", "Pred_Ensemble_Distance_From_50"]:
            out[c] = 0.0
            new_cols.append(c)

    # Composite AD scores.
    ad_candidates = [c for c in out.columns if categorize_policy_feature(c) == "ad_descriptor" and not is_outcome_or_leakage_col(c, [])]
    risk_cols = [c for c in ad_candidates if pd.api.types.is_numeric_dtype(pd.to_numeric(out[c], errors="coerce"))]
    risk_cols = [c for c in risk_cols if c not in [pred_col]]
    # Keep only meaningful numeric AD columns.
    valid_risk_cols = []
    for c in risk_cols:
        x = pd.to_numeric(out[c], errors="coerce")
        if np.isfinite(x).sum() > 0 and x.nunique(dropna=True) > 1:
            valid_risk_cols.append(c)

    if valid_risk_cols:
        mat = np.vstack([minmax01(out[c]) for c in valid_risk_cols]).T
        out["Composite_AD_MeanRisk"] = np.nanmean(mat, axis=1)
        out["Composite_AD_MaxRisk"] = np.nanmax(mat, axis=1)
        new_cols += ["Composite_AD_MeanRisk", "Composite_AD_MaxRisk"]

    dis_cols = [c for c in ["Pred_Disagreement_SD", "Pred_Disagreement_Range", "Pred_Disagreement_IQR"] if c in out.columns]
    if dis_cols:
        mat = np.vstack([minmax01(out[c]) for c in dis_cols]).T
        out["Composite_PredDisagreement"] = np.nanmean(mat, axis=1)
        new_cols.append("Composite_PredDisagreement")

    if "Composite_AD_MeanRisk" in out.columns and "Composite_PredDisagreement" in out.columns:
        out["Composite_ADxDisagreement"] = 0.60 * out["Composite_AD_MeanRisk"] + 0.40 * out["Composite_PredDisagreement"]
        new_cols.append("Composite_ADxDisagreement")

    return out, new_cols


def build_feature_audit(df: pd.DataFrame, core_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        cat = categorize_policy_feature(c)
        leakage = is_outcome_or_leakage_col(c, core_cols)
        x = pd.to_numeric(df[c], errors="coerce")
        finite = int(np.isfinite(x).sum())
        unique_n = int(x.nunique(dropna=True)) if finite > 0 else int(df[c].nunique(dropna=True))
        numeric = finite > 0 and unique_n > 1
        allowed = (not leakage) and cat in {"ad_descriptor", "prediction_disagreement", "prediction_output", "reliability_zone"}
        rows.append({
            "Feature": c,
            "Category": cat,
            "AllowedForPolicy": bool(allowed),
            "LeakageOrInvalid": bool(leakage),
            "NumericUsable": bool(numeric),
            "FiniteN": finite,
            "UniqueN": unique_n,
            "PriorityScore": feature_priority_score(c),
        })
    audit = pd.DataFrame(rows)
    return audit.sort_values(["AllowedForPolicy", "PriorityScore", "UniqueN"], ascending=[False, False, False])


def make_numeric_conditions(df: pd.DataFrame, cols: Sequence[str], q_grid: Sequence[float], ops: Sequence[str]) -> List[Condition]:
    conds: List[Condition] = []
    for c in cols:
        if c not in df.columns:
            continue
        vals = winsorized_quantiles(df[c], q_grid)
        for v in vals:
            for op in ops:
                conds.append(Condition(c, op, float(v)))
    return conds


def deduplicate_policies(policies: Sequence[Policy]) -> List[Policy]:
    seen = set()
    out: List[Policy] = []
    for p in policies:
        key = tuple((c.col, c.op, str(c.value)) for c in p.conditions)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def build_targeted_policies(df: pd.DataFrame, feature_audit: pd.DataFrame, zone_col: Optional[str]) -> List[Policy]:
    """Build targeted exhaustive policy catalog."""
    policies: List[Policy] = []
    policies.append(Policy("P_AllDomain", "reference", tuple(), "all", False, "Reference all-domain rule"))

    allowed = feature_audit[feature_audit["AllowedForPolicy"]].copy()
    allowed_numeric = allowed[allowed["NumericUsable"]].copy()

    ad_cols = allowed_numeric[allowed_numeric["Category"] == "ad_descriptor"].sort_values(
        ["PriorityScore", "UniqueN"], ascending=[False, False]
    )["Feature"].tolist()
    dis_cols = allowed_numeric[allowed_numeric["Category"] == "prediction_disagreement"].sort_values(
        ["PriorityScore", "UniqueN"], ascending=[False, False]
    )["Feature"].tolist()
    pred_cols = allowed_numeric[allowed_numeric["Category"] == "prediction_output"].sort_values(
        ["PriorityScore", "UniqueN"], ascending=[False, False]
    )["Feature"].tolist()

    ad_single = ad_cols[:MAX_AD_FEATURES_FOR_SINGLE]
    ad_combo = ad_cols[:MAX_AD_FEATURES_FOR_COMBO]
    dis_combo = dis_cols[:MAX_DISAGREEMENT_FEATURES_FOR_COMBO]
    pred_combo = pred_cols[:MAX_PREDICTION_FEATURES_FOR_COMBO]

    # Existing reliability-zone rules.
    if zone_col and zone_col in df.columns:
        values = sorted([v for v in df[zone_col].dropna().astype(str).unique().tolist() if v.strip()])
        for val in values:
            cond = Condition(zone_col, "eq", val)
            safe = re.sub(r"[^A-Za-z0-9]+", "", val)[:30]
            policies.append(Policy(
                f"Zone_eq_{safe}", "zone_single", (cond,), "single", False,
                f"Retain reliability-zone category: {val}",
            ))
        rel_like = [v for v in values if "reliable" in v.lower() or "low" in v.lower()]
        warn_like = [v for v in values if "warning" in v.lower() or "medium" in v.lower()]
        out_like = [v for v in values if "outside" in v.lower() or "high" in v.lower()]
        if rel_like:
            policies.append(Policy("Zone_in_ReliableLike", "zone_set", (Condition(zone_col, "in", tuple(rel_like)),), "single", False,
                                   "Retain reliable-like zones"))
        if rel_like or warn_like:
            policies.append(Policy("Zone_in_ReliableOrWarning", "zone_set", (Condition(zone_col, "in", tuple(rel_like + warn_like)),), "single", False,
                                   "Retain reliable or warning zones"))
        if out_like:
            keep_vals = tuple(v for v in values if v not in out_like)
            if keep_vals:
                policies.append(Policy("Zone_not_OutsideLike", "zone_set", (Condition(zone_col, "in", keep_vals),), "single", False,
                                       "Retain all zones except outside-like zones"))

    # Single numeric rules: include both <= and >= because 32A/32A2 showed
    # AD-risk/error relation is not monotonic in a simple direction.
    single_cols = ad_single + dis_cols + pred_cols
    single_conds = make_numeric_conditions(df, single_cols, NUMERIC_Q_GRID_FULL, ["<=", ">="])
    for i, cond in enumerate(single_conds, start=1):
        fam = categorize_policy_feature(cond.col)
        policies.append(Policy(
            f"Single_{i:04d}", f"single_{fam}", (cond,), "single", fam == "prediction_output",
            cond.text(),
        ))

    # Range rules for prediction-output and the most important AD/disagreement features.
    range_cols = pred_combo + ad_combo[:5] + dis_combo[:3]
    range_id = 0
    for c in range_cols:
        vals = winsorized_quantiles(df[c], [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90])
        for lo, hi in [(vals[i], vals[j]) for i in range(len(vals)) for j in range(i + 2, len(vals))]:
            range_id += 1
            fam = categorize_policy_feature(c)
            policies.append(Policy(
                f"Range_{range_id:04d}", f"range_{fam}", (Condition(c, "between", (float(lo), float(hi))),), "single", fam == "prediction_output",
                f"{lo:.4g} <= {c} <= {hi:.4g}",
            ))

    # Combo condition pools.
    ad_conds = make_numeric_conditions(df, ad_combo, NUMERIC_Q_GRID_COMBO, ["<=", ">="])
    dis_conds = make_numeric_conditions(df, dis_combo, [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90], ["<=", ">="])
    pred_conds = make_numeric_conditions(df, pred_combo, PRED_Q_GRID, ["<=", ">="])

    # Deduplicate condition pools.
    def dedup_conds(conds: Sequence[Condition]) -> List[Condition]:
        seen = set()
        out = []
        for c in conds:
            key = (c.col, c.op, round(float(c.value), 10) if isinstance(c.value, (float, int)) else str(c.value))
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out

    ad_conds = dedup_conds(ad_conds)
    dis_conds = dedup_conds(dis_conds)
    pred_conds = dedup_conds(pred_conds)

    combo_id = 0

    # Two-rule and three-rule AND rules.
    # IMPORTANT: we do not generate millions of near-identical combinations first and
    # trim later. Each family has an explicit deterministic cap. This keeps the
    # script robust while still covering all targeted rule families.
    combo_id = 0

    def add_pair_family(prefix: str, family: str, left: Sequence[Condition], right: Sequence[Condition],
                        cap: int, uses_pred: bool, stride: int = 1) -> None:
        nonlocal combo_id, policies
        count = 0
        raw_i = 0
        for a in left:
            for b in right:
                raw_i += 1
                if stride > 1 and (raw_i % stride != 0):
                    continue
                combo_id += 1
                count += 1
                policies.append(Policy(
                    f"{prefix}_{combo_id:05d}", family, (a, b), "two_rule_AND", uses_pred,
                    f"{a.text()} AND {b.text()}",
                ))
                if count >= cap:
                    return

    add_pair_family("AD_DIS_AND", "and_ad_disagreement", ad_conds, dis_conds,
                    cap=120, uses_pred=False, stride=10)
    add_pair_family("AD_PRED_AND", "and_ad_prediction", ad_conds, pred_conds,
                    cap=180, uses_pred=True, stride=15)
    add_pair_family("DIS_PRED_AND", "and_disagreement_prediction", dis_conds, pred_conds,
                    cap=100, uses_pred=True, stride=15)

    # Zone + targeted numeric condition.
    if zone_col and zone_col in df.columns:
        zone_values = sorted(df[zone_col].dropna().astype(str).unique().tolist())
        zone_conditions = [Condition(zone_col, "eq", v) for v in zone_values]
        targeted_numeric = ad_conds[:70] + dis_conds[:35] + pred_conds[:70]
        z_count = 0
        for zc in zone_conditions:
            for nc in targeted_numeric:
                combo_id += 1
                z_count += 1
                policies.append(Policy(
                    f"ZONE_NUM_AND_{combo_id:05d}", "and_zone_numeric", (zc, nc), "two_rule_AND",
                    categorize_policy_feature(nc.col) == "prediction_output",
                    f"{zc.text()} AND {nc.text()}",
                ))
                if z_count >= 80:
                    break
            if z_count >= 80:
                break

    # Three-rule AND: AD + disagreement + prediction-output.
    # Key extension beyond 32C-fast. Capped and deterministically thinned.
    triple_count = 0
    raw_i = 0
    for a in ad_conds[:70]:
        for d in dis_conds[:40]:
            for pr in pred_conds[:55]:
                raw_i += 1
                if raw_i % 251 != 0:
                    continue
                combo_id += 1
                triple_count += 1
                policies.append(Policy(
                    f"AD_DIS_PRED_AND_{combo_id:05d}", "and_ad_disagreement_prediction", (a, d, pr), "three_rule_AND", True,
                    f"{a.text()} AND {d.text()} AND {pr.text()}",
                ))
                if triple_count >= 150:
                    break
            if triple_count >= 150:
                break
        if triple_count >= 150:
            break

    policies = deduplicate_policies(policies)

    # Keep candidate count bounded, prioritizing richer rules but preserving all families.
    if len(policies) > MAX_POLICIES_TO_SCORE:
        # Always keep reference, zone, all singles and top family-balanced combos.
        keep: List[Policy] = []
        ref_single = [p for p in policies if p.search_mode in {"all", "single"} or p.family.startswith("zone")]
        keep.extend(ref_single)
        remaining = [p for p in policies if p not in set(ref_single)]
        fam_groups: Dict[str, List[Policy]] = {}
        for p in remaining:
            fam_groups.setdefault(p.family, []).append(p)
        remaining_slots = max(0, MAX_POLICIES_TO_SCORE - len(keep))
        per_fam = max(1, remaining_slots // max(1, len(fam_groups)))
        for fam, group in fam_groups.items():
            keep.extend(group[:per_fam])
        # Fill any leftover.
        if len(keep) < MAX_POLICIES_TO_SCORE:
            used = set(keep)
            for p in remaining:
                if p not in used:
                    keep.append(p)
                    if len(keep) >= MAX_POLICIES_TO_SCORE:
                        break
        policies = keep[:MAX_POLICIES_TO_SCORE]

    return policies


# =============================================================================
# Strict block-wise calibration
# =============================================================================


def objective_value(dfm: pd.DataFrame, objective: str) -> np.ndarray:
    r2 = dfm["R2"].to_numpy(dtype=float)
    rmse = dfm["RMSE"].to_numpy(dtype=float)
    mae = dfm["MAE"].to_numpy(dtype=float)
    cov = dfm["coverage"].to_numpy(dtype=float)

    if objective == "R2_oriented":
        score = r2
    elif objective == "RMSE_oriented":
        # primarily reduce RMSE but avoid selecting very negative-R2 rules when ties occur
        rmse_rank = pd.Series(rmse).rank(method="average", ascending=True).to_numpy(dtype=float)
        r2_rank = pd.Series(r2).rank(method="average", ascending=False).to_numpy(dtype=float)
        score = -rmse_rank - 0.15 * r2_rank + 0.05 * cov
    elif objective == "Balanced_R2_RMSE":
        # normalized compromise: higher R2, lower RMSE, with mild coverage preference.
        finite_rmse = rmse[np.isfinite(rmse)]
        if len(finite_rmse) == 0:
            score = r2
        else:
            lo, hi = float(np.nanmin(finite_rmse)), float(np.nanmax(finite_rmse))
            rmse_norm = (rmse - lo) / (hi - lo + 1e-12)
            score = r2 - 0.25 * rmse_norm + 0.05 * cov
    else:
        raise ValueError(f"Unknown objective: {objective}")

    # IMPORTANT: some pandas/numpy conversions can return read-only views.
    # Convert to a writable float array before masking invalid values.
    score = np.asarray(score, dtype=float).copy()
    score[~np.isfinite(score)] = -np.inf
    return score


def evaluate_strict_policy_for_subset(
    sub: pd.DataFrame,
    policies: Sequence[Policy],
    y_col: str,
    p_col: str,
    block_col: str,
    min_coverage: float,
    objective: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run grouped block-wise calibration for one strategy-method subset."""
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    sub = sub.reset_index(drop=True).copy()
    y = to_float_array(sub[y_col])
    p = to_float_array(sub[p_col])
    blocks = sub[block_col].astype(str).to_numpy()
    unique_blocks = sorted(pd.unique(blocks).tolist())

    mask_mat = policy_mask_matrix(sub, policies)
    selected_rows = []
    retained_chunks = []

    for b in unique_blocks:
        held = blocks == b
        calib = ~held
        if held.sum() < 3 or calib.sum() < MIN_CALIB_N:
            continue

        cal_metrics = vectorized_metrics(y[calib], p[calib], mask_mat[calib, :])
        eligible = (
            (cal_metrics["n"].to_numpy() >= MIN_CALIB_N)
            & (cal_metrics["coverage"].to_numpy() >= min_coverage)
            & np.isfinite(cal_metrics["R2"].to_numpy())
            & np.isfinite(cal_metrics["RMSE"].to_numpy())
        )
        if not eligible.any():
            continue

        score = objective_value(cal_metrics, objective)
        score[~eligible] = -np.inf
        best_idx = int(np.nanargmax(score))
        best_pol = policies[best_idx]

        held_mask = mask_mat[held, best_idx]
        hold_n = int(held_mask.sum())
        hold_cov = float(hold_n / max(1, held.sum()))
        hm = metrics(y[held][held_mask], p[held][held_mask])

        selected_rows.append({
            "Heldout_Block": b,
            "Objective": objective,
            "MinCoverage": min_coverage,
            "SelectedPolicyID": best_pol.name,
            "PolicyFamily": best_pol.family,
            "SearchMode": best_pol.search_mode,
            "UsesPredictionOutput": best_pol.uses_prediction_output,
            "PolicyDescription": best_pol.description,
            "Calibration_N": int(cal_metrics.loc[best_idx, "n"]),
            "Calibration_Coverage": float(cal_metrics.loc[best_idx, "coverage"]),
            "Calibration_R2": float(cal_metrics.loc[best_idx, "R2"]),
            "Calibration_RMSE": float(cal_metrics.loc[best_idx, "RMSE"]),
            "Calibration_MAE": float(cal_metrics.loc[best_idx, "MAE"]),
            "Heldout_Block_N_All": int(held.sum()),
            "Heldout_Block_N_Retained": hold_n,
            "Heldout_Block_Coverage": hold_cov,
            "Heldout_Block_R2": hm["R2"],
            "Heldout_Block_RMSE": hm["RMSE"],
            "Heldout_Block_MAE": hm["MAE"],
        })

        chunk = sub.loc[held].copy()
        chunk["Strict_Retained"] = held_mask
        chunk["SelectedPolicyID"] = best_pol.name
        chunk["SelectedPolicyFamily"] = best_pol.family
        chunk["SelectedPolicyDescription"] = best_pol.description
        chunk["Objective"] = objective
        chunk["MinCoverage"] = min_coverage
        retained_chunks.append(chunk.loc[chunk["Strict_Retained"]].copy())

    selected_df = pd.DataFrame(selected_rows)
    retained_df = pd.concat(retained_chunks, ignore_index=True) if retained_chunks else pd.DataFrame()
    return selected_df, retained_df



def evaluate_strict_policy_grid_for_subset(
    sub: pd.DataFrame,
    policies: Sequence[Policy],
    y_col: str,
    p_col: str,
    block_col: str,
    min_coverage_grid: Sequence[float],
    objectives: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Efficient grouped block-wise calibration for all objectives/min-coverages.

    The expensive policy mask matrix is built once per strategy-method subset.
    Calibration metrics are built once per held-out block and reused across all
    objectives and minimum-coverage constraints.
    """
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    sub = sub.reset_index(drop=True).copy()
    y = to_float_array(sub[y_col])
    p = to_float_array(sub[p_col])
    blocks = sub[block_col].astype(str).to_numpy()
    unique_blocks = sorted(pd.unique(blocks).tolist())

    mask_mat = policy_mask_matrix(sub, policies)
    selected_rows = []
    retained_chunks = []

    for b in unique_blocks:
        held = blocks == b
        calib = ~held
        if held.sum() < 3 or calib.sum() < MIN_CALIB_N:
            continue

        cal_metrics = vectorized_metrics(y[calib], p[calib], mask_mat[calib, :])
        r2_arr = cal_metrics["R2"].to_numpy(dtype=float)
        rmse_arr = cal_metrics["RMSE"].to_numpy(dtype=float)
        base_valid = (cal_metrics["n"].to_numpy() >= MIN_CALIB_N) & np.isfinite(r2_arr) & np.isfinite(rmse_arr)

        # Precompute objective scores once per block.
        objective_scores = {obj: objective_value(cal_metrics, obj) for obj in objectives}

        for min_coverage in min_coverage_grid:
            eligible_cov = cal_metrics["coverage"].to_numpy(dtype=float) >= float(min_coverage)
            eligible = base_valid & eligible_cov
            if not eligible.any():
                continue

            for objective in objectives:
                score = objective_scores[objective].copy()
                score[~eligible] = -np.inf
                if not np.isfinite(score).any():
                    continue
                best_idx = int(np.nanargmax(score))
                best_pol = policies[best_idx]

                held_mask = mask_mat[held, best_idx]
                hold_n = int(held_mask.sum())
                hold_cov = float(hold_n / max(1, held.sum()))
                hm = metrics(y[held][held_mask], p[held][held_mask])

                selected_rows.append({
                    "Heldout_Block": b,
                    "Objective": objective,
                    "MinCoverage": float(min_coverage),
                    "SelectedPolicyID": best_pol.name,
                    "PolicyFamily": best_pol.family,
                    "SearchMode": best_pol.search_mode,
                    "UsesPredictionOutput": best_pol.uses_prediction_output,
                    "PolicyDescription": best_pol.description,
                    "Calibration_N": int(cal_metrics.loc[best_idx, "n"]),
                    "Calibration_Coverage": float(cal_metrics.loc[best_idx, "coverage"]),
                    "Calibration_R2": float(cal_metrics.loc[best_idx, "R2"]),
                    "Calibration_RMSE": float(cal_metrics.loc[best_idx, "RMSE"]),
                    "Calibration_MAE": float(cal_metrics.loc[best_idx, "MAE"]),
                    "Heldout_Block_N_All": int(held.sum()),
                    "Heldout_Block_N_Retained": hold_n,
                    "Heldout_Block_Coverage": hold_cov,
                    "Heldout_Block_R2": hm["R2"],
                    "Heldout_Block_RMSE": hm["RMSE"],
                    "Heldout_Block_MAE": hm["MAE"],
                })

                if hold_n > 0:
                    chunk = sub.loc[held].copy()
                    chunk["Strict_Retained"] = held_mask
                    chunk["SelectedPolicyID"] = best_pol.name
                    chunk["SelectedPolicyFamily"] = best_pol.family
                    chunk["SelectedPolicyDescription"] = best_pol.description
                    chunk["Objective"] = objective
                    chunk["MinCoverage"] = float(min_coverage)
                    retained_chunks.append(chunk.loc[chunk["Strict_Retained"]].copy())

    selected_df = pd.DataFrame(selected_rows)
    retained_df = pd.concat(retained_chunks, ignore_index=True) if retained_chunks else pd.DataFrame()
    return selected_df, retained_df


def summarize_retained_predictions(retained: pd.DataFrame, y_col: str, p_col: str, strategy_col: str, method_col: str) -> pd.DataFrame:
    rows = []
    if retained.empty:
        return pd.DataFrame()
    group_cols = ["Objective", "MinCoverage", strategy_col, method_col]
    for keys, g in retained.groupby(group_cols, dropna=False):
        obj, min_cov, strategy, method = keys
        y = to_float_array(g[y_col])
        p = to_float_array(g[p_col])
        m = metrics(y, p)
        # Comparable denominator: all rows for this method-strategy in original retained table is not here;
        # compute denominator outside later if needed.
        rows.append({
            "Objective": obj,
            "MinCoverage": min_cov,
            "Strategy": strategy,
            "Method": method,
            "N_Retained": m["N_Retained"],
            "R2": m["R2"],
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
        })
    return pd.DataFrame(rows)


def add_denominators(summary: pd.DataFrame, df: pd.DataFrame, strategy_col: str, method_col: str) -> pd.DataFrame:
    if summary.empty:
        return summary
    denom = df.groupby([strategy_col, method_col], dropna=False).size().reset_index(name="N_AllComparable")
    denom = denom.rename(columns={strategy_col: "Strategy", method_col: "Method"})
    out = summary.merge(denom, on=["Strategy", "Method"], how="left")
    out["Coverage"] = out["N_Retained"] / out["N_AllComparable"].replace(0, np.nan)
    return out



def evaluate_oracle_grid(
    df: pd.DataFrame,
    policies: Sequence[Policy],
    y_col: str,
    p_col: str,
    strategy_col: str,
    method_col: str,
    min_cov_grid: Sequence[float],
) -> pd.DataFrame:
    """Efficient diagnostic-only oracle for all minimum-coverage values."""
    rows = []
    for (strategy, method), sub in df.groupby([strategy_col, method_col], dropna=False):
        sub = sub.reset_index(drop=True)
        if len(sub) < MIN_HOLDOUT_TOTAL_N:
            continue
        y = to_float_array(sub[y_col])
        p = to_float_array(sub[p_col])
        mask_mat = policy_mask_matrix(sub, policies)
        pm = vectorized_metrics(y, p, mask_mat)
        r2 = pm["R2"].to_numpy(dtype=float)
        base = (pm["n"].to_numpy() >= MIN_HOLDOUT_TOTAL_N) & np.isfinite(r2)
        for min_cov in min_cov_grid:
            elig = base & (pm["coverage"].to_numpy(dtype=float) >= float(min_cov))
            if not elig.any():
                continue
            score = np.asarray(r2, dtype=float).copy()
            score[~elig] = -np.inf
            idx = int(np.nanargmax(score))
            pol = policies[idx]
            rows.append({
                "Strategy": strategy,
                "Method": method,
                "DiagnosticBestPolicyID": pol.name,
                "PolicyFamily": pol.family,
                "SearchMode": pol.search_mode,
                "UsesPredictionOutput": pol.uses_prediction_output,
                "PolicyDescription": pol.description,
                "MinCoverage": float(min_cov),
                "N_Retained": int(pm.loc[idx, "n"]),
                "N_AllComparable": int(len(sub)),
                "Coverage": float(pm.loc[idx, "coverage"]),
                "R2": float(pm.loc[idx, "R2"]),
                "RMSE": float(pm.loc[idx, "RMSE"]),
                "MAE": float(pm.loc[idx, "MAE"]),
                "Interpretation": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE_LABEL_SELECTED",
            })
    return pd.DataFrame(rows).sort_values(["R2", "Coverage"], ascending=[False, False]) if rows else pd.DataFrame()


def evaluate_oracle(
    df: pd.DataFrame,
    policies: Sequence[Policy],
    y_col: str,
    p_col: str,
    strategy_col: str,
    method_col: str,
    min_cov: float,
) -> pd.DataFrame:
    """Diagnostic-only oracle: chooses best policy on the same evaluation rows."""
    rows = []
    for (strategy, method), sub in df.groupby([strategy_col, method_col], dropna=False):
        sub = sub.reset_index(drop=True)
        if len(sub) < MIN_HOLDOUT_TOTAL_N:
            continue
        y = to_float_array(sub[y_col])
        p = to_float_array(sub[p_col])
        mask_mat = policy_mask_matrix(sub, policies)
        pm = vectorized_metrics(y, p, mask_mat)
        elig = (pm["coverage"].to_numpy() >= min_cov) & (pm["n"].to_numpy() >= MIN_HOLDOUT_TOTAL_N) & np.isfinite(pm["R2"].to_numpy())
        if not elig.any():
            continue
        score = pm["R2"].to_numpy(dtype=float).copy()
        score[~elig] = -np.inf
        idx = int(np.nanargmax(score))
        pol = policies[idx]
        rows.append({
            "Strategy": strategy,
            "Method": method,
            "DiagnosticBestPolicyID": pol.name,
            "PolicyFamily": pol.family,
            "SearchMode": pol.search_mode,
            "UsesPredictionOutput": pol.uses_prediction_output,
            "PolicyDescription": pol.description,
            "MinCoverage": min_cov,
            "N_Retained": int(pm.loc[idx, "n"]),
            "N_AllComparable": int(len(sub)),
            "Coverage": float(pm.loc[idx, "coverage"]),
            "R2": float(pm.loc[idx, "R2"]),
            "RMSE": float(pm.loc[idx, "RMSE"]),
            "MAE": float(pm.loc[idx, "MAE"]),
            "Interpretation": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE_LABEL_SELECTED",
        })
    return pd.DataFrame(rows).sort_values(["R2", "Coverage"], ascending=[False, False]) if rows else pd.DataFrame()


# =============================================================================
# Plotting
# =============================================================================


def annotate_bars(ax, fmt: str = "{:.3f}", rotation: int = 0) -> None:
    for patch in ax.patches:
        h = patch.get_height()
        if not np.isfinite(h):
            continue
        x = patch.get_x() + patch.get_width() / 2
        y = h
        offset = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        va = "bottom" if h >= 0 else "top"
        ax.text(x, y + (offset if h >= 0 else -offset), fmt.format(h), ha="center", va=va, fontsize=8, rotation=rotation)


def plot_policy_metrics(summary: pd.DataFrame, fig_dir: Path) -> None:
    if summary.empty:
        return
    plot_source = summary[summary.get("DeployableCandidate", False) == True].copy() if "DeployableCandidate" in summary.columns else summary.copy()
    if plot_source.empty:
        plot_source = summary.copy()
    top = plot_source.sort_values(["R2", "Coverage"], ascending=[False, False]).head(12).copy()
    top["PolicyLabel"] = [f"P{i:02d}" for i in range(1, len(top) + 1)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    fig.suptitle("32C2 targeted exhaustive selective AD policy metrics", fontsize=18, fontweight="bold", y=1.02)

    panels = [
        ("R2", "Retained R2", "{:.3f}"),
        ("RMSE", "Retained RMSE", "{:.2f}"),
        ("MAE", "Retained MAE", "{:.2f}"),
        ("Coverage", "Coverage", "{:.2f}"),
    ]
    for ax, (metric, title, fmt) in zip(axes.ravel(), panels):
        ax.bar(top["PolicyLabel"], top[metric], edgecolor="black", linewidth=0.6, alpha=0.82)
        ax.set_title(title, pad=14)
        ax.set_xlabel("Policy ID")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        annotate_bars(ax, fmt=fmt)
        if metric == "R2":
            ax.axhline(TARGET_R2, color="red", linestyle="--", linewidth=1.2, label="R2 = 0.5 target")
            ax.legend(frameon=False, loc="best")
    save_fig(fig, fig_dir / "162_32c2_exhaustive_policy_metrics.png")


def plot_frontier(summary: pd.DataFrame, oracle: pd.DataFrame, fig_dir: Path) -> None:
    if summary.empty:
        return
    s = summary[summary.get("DeployableCandidate", False) == True].copy() if "DeployableCandidate" in summary.columns else summary.copy()
    if s.empty:
        s = summary.copy()
    s["PolicyLabel"] = [f"S{i:03d}" for i in range(1, len(s) + 1)]
    top = s.sort_values(["R2", "Coverage"], ascending=[False, False]).head(8)

    fig, ax = plt.subplots(figsize=(13.5, 8), constrained_layout=True)
    ax.scatter(s["Coverage"], s["R2"], s=55, alpha=0.65, label="Strict block-calibrated policies")
    if oracle is not None and not oracle.empty:
        oo = oracle.copy()
        ax.scatter(oo["Coverage"], oo["R2"], s=45, alpha=0.55, marker="x", label="Diagnostic oracle upper bound")
    ax.axhline(TARGET_R2, color="red", linestyle="--", linewidth=1.2, label="R2 = 0.5 target")
    ax.axvline(0.15, color="gray", linestyle=":", linewidth=1.2, label="Minimum coverage")
    for _, r in top.iterrows():
        label = f"{r['PolicyLabel']}\n{short_text(r['Method'], 20)}"
        ax.annotate(label, (r["Coverage"], r["R2"]), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_title("R2–coverage frontier under 32C2 selective AD policies", pad=16)
    ax.set_xlabel("Coverage retained")
    ax.set_ylabel("R2 within retained domain")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, loc="best")
    save_fig(fig, fig_dir / "163_32c2_r2_coverage_frontier.png")


def plot_best_prediction_panel(summary: pd.DataFrame, retained: pd.DataFrame, y_col: str, p_col: str, fig_dir: Path) -> None:
    if summary.empty or retained.empty:
        return
    best_pool = summary[summary.get("DeployableCandidate", False) == True].copy() if "DeployableCandidate" in summary.columns else summary.copy()
    if best_pool.empty:
        best_pool = summary.copy()
    best = best_pool.sort_values(["R2", "Coverage"], ascending=[False, False]).iloc[0]
    mask = (
        (retained["Objective"].astype(str) == str(best["Objective"]))
        & (retained["MinCoverage"].astype(float) == float(best["MinCoverage"]))
        & (retained["Strategy"].astype(str) == str(best["Strategy"]))
        & (retained["Method"].astype(str) == str(best["Method"]))
    )
    d = retained.loc[mask].copy()
    if d.empty:
        return
    y = to_float_array(d[y_col])
    p = to_float_array(d[p_col])

    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    ax.scatter(y, p, s=35, alpha=0.72, edgecolors="none")
    lo = 0
    hi = 100
    ax.plot([lo, hi], [lo, hi], "--", color="red", linewidth=1.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(f"Best strict 32C2 policy: {best['Method']}", pad=16, fontweight="bold")
    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Predicted F removal (%)")
    txt = (
        f"R2 = {best['R2']:.3f}\n"
        f"RMSE = {best['RMSE']:.2f}\n"
        f"MAE = {best['MAE']:.2f}\n"
        f"n = {int(best['N_Retained'])}\n"
        f"coverage = {best['Coverage']:.2f}\n"
        f"policy = {best['PolicyID']}"
    )
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top", fontsize=12,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9))
    ax.grid(alpha=0.25)
    save_fig(fig, fig_dir / "164_32c2_best_prediction_panel.png")


def plot_family_performance(summary: pd.DataFrame, fig_dir: Path) -> None:
    if summary.empty:
        return
    fam = summary.groupby("PolicyFamily", dropna=False).agg(
        Best_R2=("R2", "max"),
        Best_Coverage=("Coverage", "max"),
        Best_RMSE=("RMSE", "min"),
        Count=("R2", "size"),
    ).reset_index().sort_values("Best_R2", ascending=False).head(14)
    fam["FamilyLabel"] = [f"F{i:02d}" for i in range(1, len(fam) + 1)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    fig.suptitle("32C2 policy-family performance summary", fontsize=17, fontweight="bold", y=1.04)
    for ax, metric, title, fmt in [
        (axes[0], "Best_R2", "Best R2", "{:.3f}"),
        (axes[1], "Best_RMSE", "Best RMSE", "{:.2f}"),
        (axes[2], "Best_Coverage", "Best coverage", "{:.2f}"),
    ]:
        ax.bar(fam["FamilyLabel"], fam[metric], edgecolor="black", linewidth=0.6, alpha=0.82)
        ax.set_title(title, pad=14)
        ax.set_xlabel("Policy family ID")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        annotate_bars(ax, fmt=fmt)
        if metric == "Best_R2":
            ax.axhline(TARGET_R2, color="red", linestyle="--", linewidth=1.1)
    save_fig(fig, fig_dir / "165_32c2_policy_family_performance.png")


# =============================================================================
# Main workflow
# =============================================================================


def find_input_table(table_dir: Path) -> Path:
    candidates = [
        table_dir / "32b_predictions_with_reliability_zone.csv",
        table_dir / "32a2_ad_predictions_with_scores.csv",
        table_dir / "32a_ad_predictions_with_scores.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find required prediction-with-AD table. Expected one of:\n"
        + "\n".join(str(p) for p in candidates)
    )


def make_policy_catalog_df(policies: Sequence[Policy]) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(policies, start=1):
        rows.append({
            "PolicyID": p.name,
            "PolicyIndex": i - 1,
            "PolicyFamily": p.family,
            "SearchMode": p.search_mode,
            "UsesPredictionOutput": p.uses_prediction_output,
            "NConditions": len(p.conditions),
            "Conditions": " AND ".join(c.text() for c in p.conditions) if p.conditions else "ALL",
            "Description": p.description,
        })
    return pd.DataFrame(rows)


def run_main() -> None:
    root = find_project_root()
    fig_dir, table_dir = ensure_dirs(root)
    log("=" * 96)
    log(f"Running {SCRIPT_NAME}")
    log(f"Project root: {root}")
    log("=" * 96)

    input_path = find_input_table(table_dir)
    df = read_csv_robust(input_path)
    log(f"[INFO] Loaded input table: {input_path} | shape={df.shape}")

    strategy_col = find_col(df, ["Strategy"])
    method_col = find_col(df, ["Method", "Model"])
    y_col = find_col(df, ["Observed", "Actual", "y_true", "Target"])
    p_col = find_col(df, ["Predicted", "Prediction", "y_pred"])
    group_col = find_col(df, ["文献编号", "Literature", "Paper", "Group", "Source"], required=False)
    split_col = find_col(df, ["Split_ID", "Split", "Fold"], required=False)
    seed_col = find_col(df, ["Seed_or_Fold", "Seed", "Fold"], required=False)
    zone_col = find_col(df, ["Reliability_Zone", "AD_Risk_Bin", "Reliable_Domain_Flag"], required=False)

    if split_col is not None:
        block_col = split_col
    elif seed_col is not None:
        block_col = seed_col
    elif group_col is not None:
        block_col = group_col
    else:
        raise ValueError("Cannot find a grouped block column such as Split_ID / Seed_or_Fold / 文献编号.")

    log(f"[INFO] Observed column: {y_col}")
    log(f"[INFO] Predicted column: {p_col}")
    log(f"[INFO] Strategy column: {strategy_col}")
    log(f"[INFO] Method column: {method_col}")
    log(f"[INFO] Group/literature column: {group_col}")
    log(f"[INFO] Block column for grouped calibration: {block_col}")
    log(f"[INFO] Reliability zone column: {zone_col}")

    # Clean essential columns.
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
    df = df[np.isfinite(df[y_col]) & np.isfinite(df[p_col])].copy()
    df[strategy_col] = df[strategy_col].astype(str)
    df[method_col] = df[method_col].astype(str)
    df[block_col] = df[block_col].astype(str)

    # Prioritize known useful methods if available, but keep all if preferred list is absent.
    available_methods = df[method_col].dropna().astype(str).unique().tolist()
    preferred_present = [m for m in PREFERRED_METHOD_ORDER if m in available_methods]
    if preferred_present:
        keep_methods = preferred_present
        if TARGET_METHODS is not None:
            keep_methods = [m for m in keep_methods if m in set(TARGET_METHODS)]
        df = df[df[method_col].isin(keep_methods)].copy()
        log(f"[INFO] Keeping targeted preferred available methods: {keep_methods}")
    else:
        keep_methods = available_methods
        if TARGET_METHODS is not None:
            keep_methods = [m for m in keep_methods if m in set(TARGET_METHODS)]
            df = df[df[method_col].isin(keep_methods)].copy()
        log(f"[INFO] No preferred-method names matched; keeping available targeted/all methods: {keep_methods}")

    if TARGET_STRATEGIES is not None:
        available_strategies = set(df[strategy_col].astype(str).unique())
        target_present = [s for s in TARGET_STRATEGIES if s in available_strategies]
        if target_present:
            df = df[df[strategy_col].isin(target_present)].copy()
            log(f"[INFO] Keeping targeted strategies: {target_present}")
        else:
            log("[WARN] TARGET_STRATEGIES not found in table; keeping all strategies.")

    # Add target-free features.
    key_cols = [strategy_col]
    if split_col:
        key_cols.append(split_col)
    if group_col:
        key_cols.append(group_col)
    if "Data_Index" in df.columns:
        key_cols.append("Data_Index")
    elif "Local_Test_Index" in df.columns:
        key_cols.append("Local_Test_Index")
    df, generated_cols = add_prediction_and_composite_features(df, strategy_col, method_col, p_col, key_cols)
    log(f"[INFO] Added/generated target-free policy features: {len(generated_cols)}")

    core_cols = [strategy_col, method_col, y_col, p_col, block_col]
    if group_col:
        core_cols.append(group_col)
    if split_col:
        core_cols.append(split_col)
    if seed_col:
        core_cols.append(seed_col)

    feature_audit = build_feature_audit(df, core_cols)
    save_csv(feature_audit, table_dir / "32c2_policy_feature_audit.csv")

    policies = build_targeted_policies(df, feature_audit, zone_col)
    catalog = make_policy_catalog_df(policies)
    save_csv(catalog, table_dir / "32c2_candidate_policy_catalog.csv")
    log(f"[INFO] Candidate policies generated: {len(policies)}")

    all_selected_rows = []
    all_retained = []

    combos = list(df.groupby([strategy_col, method_col], dropna=False))
    combos = sorted(combos, key=lambda kv: (str(kv[0][0]), PREFERRED_METHOD_ORDER.index(kv[0][1]) if kv[0][1] in PREFERRED_METHOD_ORDER else 999, str(kv[0][1])))
    total_jobs = len(combos)
    job_id = 0

    for (strategy, method), sub in combos:
        if len(sub) < MIN_HOLDOUT_TOTAL_N:
            continue
        job_id += 1
        log(f"[STRICT] subset {job_id}/{total_jobs} | strategy={strategy} | method={method} | n={len(sub)} | policies={len(policies)}")
        selected_df, retained_df = evaluate_strict_policy_grid_for_subset(
            sub=sub,
            policies=policies,
            y_col=y_col,
            p_col=p_col,
            block_col=block_col,
            min_coverage_grid=MIN_COVERAGE_GRID,
            objectives=OBJECTIVES,
        )
        if not selected_df.empty:
            selected_df.insert(0, "Strategy", strategy)
            selected_df.insert(1, "Method", method)
            all_selected_rows.append(selected_df)
        if not retained_df.empty:
            all_retained.append(retained_df)

    selected_all = pd.concat(all_selected_rows, ignore_index=True) if all_selected_rows else pd.DataFrame()
    retained_all = pd.concat(all_retained, ignore_index=True) if all_retained else pd.DataFrame()

    save_csv(selected_all, table_dir / "32c2_blockwise_selected_policy_summary.csv")
    save_csv(retained_all, table_dir / "32c2_selected_holdout_predictions.csv")

    summary = summarize_retained_predictions(retained_all, y_col, p_col, strategy_col, method_col)
    summary = add_denominators(summary, df, strategy_col, method_col)
    if not summary.empty:
        # Most frequent selected policy/family for each strategy-method-objective-mincoverage.
        freq_cols = ["Objective", "MinCoverage", strategy_col, method_col, "SelectedPolicyID", "SelectedPolicyFamily"]
        if not retained_all.empty and set(freq_cols).issubset(retained_all.columns):
            pass
        selected_key_cols = ["Objective", "MinCoverage", "Strategy", "Method"]
        if not selected_all.empty:
            mode_rows = []
            for keys, g in selected_all.groupby(selected_key_cols, dropna=False):
                vc = g["SelectedPolicyID"].value_counts()
                fam_vc = g["PolicyFamily"].value_counts()
                mode_rows.append({
                    "Objective": keys[0],
                    "MinCoverage": keys[1],
                    "Strategy": keys[2],
                    "Method": keys[3],
                    "PolicyID": vc.index[0] if len(vc) else "",
                    "PolicySelectionCount": int(vc.iloc[0]) if len(vc) else 0,
                    "PolicyFamily": fam_vc.index[0] if len(fam_vc) else "",
                })
            modes = pd.DataFrame(mode_rows)
            summary = summary.merge(modes, on=selected_key_cols, how="left")
        summary["CoverageMeetsMin"] = summary["Coverage"] >= (summary["MinCoverage"].astype(float) - 1e-12)
        summary["DeployableCandidate"] = (summary["CoverageMeetsMin"]) & (summary["N_Retained"] >= MIN_HOLDOUT_TOTAL_N)
        summary = summary.sort_values(["DeployableCandidate", "R2", "Coverage"], ascending=[False, False, False]).reset_index(drop=True)
    save_csv(summary, table_dir / "32c2_policy_summary.csv")

    if RUN_DIAGNOSTIC_ORACLE:
        log("[ORACLE] Diagnostic-only upper bound for all minimum-coverage values")
        oracle = evaluate_oracle_grid(df, policies, y_col, p_col, strategy_col, method_col, MIN_COVERAGE_GRID)
        if not oracle.empty:
            oracle = oracle.sort_values(["R2", "Coverage"], ascending=[False, False]).reset_index(drop=True)
    else:
        log("[ORACLE] Skipped by default: diagnostic oracle is non-deployable and not needed for strict-policy decision.")
        oracle = pd.DataFrame(columns=[
            "Strategy", "Method", "DiagnosticBestPolicyID", "PolicyFamily", "SearchMode",
            "UsesPredictionOutput", "PolicyDescription", "MinCoverage", "N_Retained",
            "N_AllComparable", "Coverage", "R2", "RMSE", "MAE", "Interpretation"
        ])
    save_csv(oracle, table_dir / "32c2_diagnostic_oracle_upper_bound.csv")

    # Plot and write mapping table for top policies.
    if not summary.empty:
        top_map = summary.head(30).copy()
        top_map.insert(0, "ShortFigureID", [f"P{i:02d}" for i in range(1, len(top_map) + 1)])
        save_csv(top_map, table_dir / "32c2_top_policy_label_mapping.csv")
        plot_policy_metrics(summary, fig_dir)
        plot_frontier(summary, oracle, fig_dir)
        plot_best_prediction_panel(summary, retained_all, y_col, p_col, fig_dir)
        # Family-performance figure is skipped by default to avoid unnecessary plotting delays.
        # The family summary remains available in 32c2_policy_summary.csv.

    # Decision report.
    report_lines = []
    report_lines.append("32C2 targeted exhaustive selective AD-policy decision report")
    report_lines.append("=" * 96)
    report_lines.append(f"Input table: {input_path}")
    report_lines.append(f"Rows after cleaning and method filtering: {len(df)}")
    report_lines.append(f"Methods evaluated: {', '.join(sorted(df[method_col].unique()))}")
    report_lines.append(f"Candidate policies: {len(policies)}")
    report_lines.append(f"Minimum coverage grid: {MIN_COVERAGE_GRID}")
    report_lines.append("")
    report_lines.append("Strictness statement")
    report_lines.append("- Serious policies are selected by grouped block-wise calibration.")
    report_lines.append("- Held-out labels are not used to select policy for that held-out block.")
    report_lines.append("- Prediction-output rules are deployment-available but are flagged separately.")
    report_lines.append("- Diagnostic oracle is non-deployable and reported only as an upper bound.")
    report_lines.append("")

    if summary.empty:
        report_lines.append("No strict block-calibrated selective policy produced enough retained samples.")
    else:
        best_pool = summary[summary.get("DeployableCandidate", False) == True].copy() if "DeployableCandidate" in summary.columns else summary.copy()
        if best_pool.empty:
            best_pool = summary.copy()
        best = best_pool.sort_values(["R2", "Coverage"], ascending=[False, False]).iloc[0]
        report_lines.append("Best strict block-calibrated selective policy")
        report_lines.append(f"- Objective: {best['Objective']}")
        report_lines.append(f"- Minimum coverage constraint: {best['MinCoverage']}")
        report_lines.append(f"- Strategy: {best['Strategy']}")
        report_lines.append(f"- Method: {best['Method']}")
        report_lines.append(f"- Policy ID: {best.get('PolicyID', '')}")
        report_lines.append(f"- Policy family: {best.get('PolicyFamily', '')}")
        report_lines.append(f"- Coverage: {best['Coverage']:.4f}")
        report_lines.append(f"- Coverage meets minimum constraint: {bool(best.get('CoverageMeetsMin', False))}")
        report_lines.append(f"- R2: {best['R2']:.4f}")
        report_lines.append(f"- RMSE: {best['RMSE']:.4f}")
        report_lines.append(f"- MAE: {best['MAE']:.4f}")
        report_lines.append("")
        if float(best["R2"]) >= TARGET_R2:
            report_lines.append("Decision: STRICT calibrated selective policy reached R2 >= 0.5.")
            report_lines.append("This can be considered for final AD-domain reporting, subject to interpretation of coverage and rule stability.")
        else:
            report_lines.append("Decision: STRICT calibrated selective policy did NOT reach R2 >= 0.5.")
            report_lines.append("Do not claim final AD-domain R2 > 0.5 from strict policy unless a later validated policy reaches it.")
        report_lines.append("")
        report_lines.append("Top strict policies")
        top = summary.head(20).copy()
        for i, (_, r) in enumerate(top.iterrows(), start=1):
            report_lines.append(
                f"{i:02d}. Obj={r['Objective']} | MinCov={r['MinCoverage']} | Strategy={r['Strategy']} | "
                f"Method={r['Method']} | Policy={r.get('PolicyID', '')} | Family={r.get('PolicyFamily', '')} | "
                f"Coverage={r['Coverage']:.3f} | R2={r['R2']:.4f} | RMSE={r['RMSE']:.3f} | MAE={r['MAE']:.3f}"
            )

    report_lines.append("")
    if oracle.empty:
        report_lines.append("Diagnostic oracle upper bound: no valid oracle policy found.")
    else:
        ob = oracle.iloc[0]
        report_lines.append("Best diagnostic oracle upper bound, NOT deployable")
        report_lines.append(f"- Strategy: {ob['Strategy']}")
        report_lines.append(f"- Method: {ob['Method']}")
        report_lines.append(f"- Policy: {ob['DiagnosticBestPolicyID']}")
        report_lines.append(f"- Policy family: {ob['PolicyFamily']}")
        report_lines.append(f"- Coverage: {ob['Coverage']:.4f}")
        report_lines.append(f"- R2: {ob['R2']:.4f}")
        report_lines.append(f"- RMSE: {ob['RMSE']:.4f}")
        report_lines.append("This only tells whether a high-R2 subdomain exists in hindsight; it cannot be used as final deployable evidence.")

    elapsed = time.time() - START_TIME
    report_lines.append("")
    report_lines.append(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    report_path = table_dir / "32c2_decision_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    log(f"[REPORT] {report_path}")
    log("=" * 96)
    log("32C2 targeted exhaustive selective AD-policy analysis finished.")
    log(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    log("=" * 96)


if __name__ == "__main__":
    run_main()
