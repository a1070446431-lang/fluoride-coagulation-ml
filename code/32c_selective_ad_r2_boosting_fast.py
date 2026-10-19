# -*- coding: utf-8 -*-
"""
32c_selective_ad_r2_boosting_fast.py

Purpose
-------
Fast audited replacement for Step 32C in the fluoride coagulation ML project.

This script searches whether a high-R2 reliable subdomain can be obtained by
selective prediction / rejection policy under fully grouped validation outputs.
It is designed to replace the previous long-running 32C brute-force search.

Strictness
----------
1. It does NOT retrain the main prediction models.
2. It reuses row-level prediction + AD-score outputs from Step 32A2/32B.
3. Candidate reliable-domain policies use target-free columns only:
   AD risks, novelty/violation fractions, reliability-zone flags, and
   prediction-disagreement features derived from model predictions.
4. The serious deployable policy is selected by grouped block-wise calibration:
   for each held-out block, policy choice is made only on the other blocks
   within the same validation strategy and model/method.
5. Held-out observed labels are used only for final evaluation, not for policy
   selection for that held-out block.
6. Diagnostic-oracle results are also reported, but clearly marked as
   non-deployable upper bound because they select policies using evaluation labels.

Required input
--------------
Preferred:
    outputs/tables/32b_predictions_with_reliability_zone.csv
Fallback:
    outputs/tables/32a2_ad_predictions_with_scores.csv
    outputs/tables/32a_ad_predictions_with_scores.csv

Outputs
-------
Tables:
    outputs/tables/32c_fast_policy_feature_audit.csv
    outputs/tables/32c_fast_candidate_policy_catalog.csv
    outputs/tables/32c_fast_blockwise_selected_policy_summary.csv
    outputs/tables/32c_fast_policy_summary.csv
    outputs/tables/32c_fast_selected_holdout_predictions.csv
    outputs/tables/32c_fast_diagnostic_oracle_upper_bound.csv
    outputs/tables/32c_fast_decision_report.txt

Figures:
    outputs/figures/158_32c_fast_policy_metrics.png/.pdf
    outputs/figures/159_32c_fast_r2_coverage_frontier.png/.pdf
    outputs/figures/160_32c_fast_best_prediction_panel.png/.pdf
    outputs/figures/161_32c_fast_strategy_breakdown.png/.pdf

Note
----
This script is intentionally fast. It avoids exponential brute-force rule
combinations and uses vectorized candidate-mask scoring.
"""

from __future__ import annotations

import math
import os
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
# Global configuration
# =============================================================================

SCRIPT_NAME = "32c_selective_ad_r2_boosting_fast.py"
START_TIME = time.time()

# Minimum retained coverage for a deployable selective policy.
# If this is too small, a rule may look good but be scientifically weak.
MIN_HELDOUT_COVERAGE = 0.15
MIN_CALIB_COVERAGE = 0.15
MIN_CALIB_N = 40
MIN_HELDOUT_N_TOTAL = 40

# Candidate thresholds. More thresholds = slower. These are enough to test a
# dense frontier without repeating almost identical rules.
QUANTILE_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# Objective choices for the strict grouped calibration selection.
OBJECTIVES = ["R2_oriented", "RMSE_oriented", "Balanced_R2_RMSE"]

# To keep runtime stable, only the most relevant target-free feature columns are
# used for candidate policies. This is not a simplified scientific workflow; it
# is an audited search-space reduction to avoid the previous brute-force stall.
MAX_SINGLE_RULE_FEATURES = 24
MAX_COMBO_FEATURES = 8

# Methods to prioritize if available. Other methods are still kept if present.
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

# Plot settings. Use English labels to avoid CJK glyph warnings in Matplotlib.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Utilities
# =============================================================================


def log(msg: str) -> None:
    print(msg, flush=True)


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find project root by walking upward from this script file / cwd."""
    candidates: List[Path] = []
    if start is not None:
        candidates.append(start.resolve())
    try:
        candidates.append(Path(__file__).resolve())
    except NameError:
        pass
    candidates.append(Path.cwd().resolve())

    for cand in candidates:
        cur = cand if cand.is_dir() else cand.parent
        for parent in [cur] + list(cur.parents):
            if (parent / "outputs" / "tables").exists() or (parent / "data").exists():
                return parent
    return Path.cwd().resolve()


def ensure_dirs(root: Path) -> Tuple[Path, Path]:
    fig_dir = root / "outputs" / "figures"
    table_dir = root / "outputs" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, table_dir


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read CSV with mixed-type warning avoided."""
    return pd.read_csv(path, low_memory=False)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {path} | shape={df.shape}")


def save_fig(fig: plt.Figure, path_png: Path) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    path_pdf = path_png.with_suffix(".pdf")
    fig.tight_layout()
    fig.savefig(path_png, dpi=600, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] {path_png}")
    log(f"[FIG] {path_pdf}")


def normalize_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def find_col(df: pd.DataFrame, aliases: Sequence[str], required: bool = True) -> Optional[str]:
    norm_map = {normalize_name(c): c for c in df.columns}
    for a in aliases:
        n = normalize_name(a)
        if n in norm_map:
            return norm_map[n]
    # fuzzy contains
    for c in df.columns:
        nc = normalize_name(c)
        for a in aliases:
            na = normalize_name(a)
            if na and na in nc:
                return c
    if required:
        raise ValueError(f"Cannot find required column from aliases: {aliases}")
    return None


def to_numeric_array(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def safe_r2(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) < 3:
        return np.nan
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst <= 1e-12:
        return np.nan
    sse = float(np.sum((y - p) ** 2))
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


def metrics_from_arrays(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(np.isfinite(y).sum()),
        "R2": safe_r2(y, p),
        "RMSE": safe_rmse(y, p),
        "MAE": safe_mae(y, p),
    }


def vectorized_metrics_for_masks(
    y: np.ndarray,
    p: np.ndarray,
    mask_mat: np.ndarray,
) -> pd.DataFrame:
    """Compute n/R2/RMSE/MAE for each candidate mask over one subset."""
    if mask_mat.ndim != 2:
        raise ValueError("mask_mat must be 2D")
    valid = np.isfinite(y) & np.isfinite(p)
    if not np.all(valid):
        y = y[valid]
        p = p[valid]
        mask_mat = mask_mat[valid, :]

    if len(y) == 0:
        c = mask_mat.shape[1]
        return pd.DataFrame({
            "n": np.zeros(c, dtype=int),
            "coverage": np.zeros(c, dtype=float),
            "R2": np.full(c, np.nan),
            "RMSE": np.full(c, np.nan),
            "MAE": np.full(c, np.nan),
        })

    m = mask_mat.astype(float, copy=False)
    n = m.sum(axis=0)
    coverage = n / float(len(y))

    err = y - p
    abs_err = np.abs(err)
    err2 = err ** 2
    y2 = y ** 2

    sum_y = m.T @ y
    sum_y2 = m.T @ y2
    sum_err2 = m.T @ err2
    sum_abs_err = m.T @ abs_err

    with np.errstate(divide="ignore", invalid="ignore"):
        rmse = np.sqrt(sum_err2 / n)
        mae = sum_abs_err / n
        sst = sum_y2 - (sum_y ** 2) / n
        r2 = 1.0 - sum_err2 / sst

    r2[(n < 3) | (~np.isfinite(r2)) | (sst <= 1e-12)] = np.nan
    rmse[(n < 1) | (~np.isfinite(rmse))] = np.nan
    mae[(n < 1) | (~np.isfinite(mae))] = np.nan

    return pd.DataFrame({
        "n": n.astype(int),
        "coverage": coverage,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
    })


def short_label(s: str, max_len: int = 32) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


# =============================================================================
# Candidate policy construction
# =============================================================================


@dataclass(frozen=True)
class Condition:
    col: str
    op: str
    value: object


@dataclass(frozen=True)
class Policy:
    name: str
    family: str
    conditions: Tuple[Condition, ...]
    description: str


def is_target_free_policy_feature(col: str, core_cols: Sequence[str]) -> bool:
    c = str(col)
    cl = normalize_name(c)

    # Core bookkeeping and target/prediction outcome columns are not policy features.
    forbidden_tokens = [
        "observed", "actual", "true", "target", "label", "removalrate",
        "removalpercent", "effluent", "outlet", "predictionerror", "abserror",
        "absoluteerror", "residual", "rmse", "mae", "mse", "r2", "scoreerror",
        "paperlevelrmse", "paperlevelmae", "oracle", "diagnostic",
    ]
    if any(tok in cl for tok in forbidden_tokens):
        return False
    if col in core_cols:
        return False

    # Data index / row index should not be used for AD policy.
    index_tokens = ["dataindex", "row", "index", "originalrow", "rawrow"]
    if any(tok in cl for tok in index_tokens):
        return False

    # Allowed target-free information types.
    allowed_tokens = [
        "ad", "risk", "novel", "unseen", "violation", "fraction", "distance",
        "leverage", "mahalanobis", "boundary", "pca", "reconstruction", "nn",
        "domain", "reliability", "zone", "disagreement", "predsd", "predrange",
        "prediqr", "uncertainty", "variance", "std",
    ]
    return any(tok in cl for tok in allowed_tokens)


def add_prediction_disagreement_features(
    df: pd.DataFrame,
    strategy_col: str,
    method_col: str,
    pred_col: str,
    key_cols: Sequence[str],
) -> pd.DataFrame:
    """Add target-free prediction-disagreement features across methods."""
    out = df.copy()
    usable_keys = [c for c in key_cols if c in out.columns]
    if len(usable_keys) < 2 or method_col not in out.columns:
        out["Pred_Disagreement_SD"] = 0.0
        out["Pred_Disagreement_Range"] = 0.0
        out["Pred_Disagreement_IQR"] = 0.0
        return out

    try:
        tmp = out[usable_keys + [method_col, pred_col]].copy()
        tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")
        wide = tmp.pivot_table(index=usable_keys, columns=method_col, values=pred_col, aggfunc="mean")
        stats = pd.DataFrame(index=wide.index)
        stats["Pred_Disagreement_SD"] = wide.std(axis=1, skipna=True).fillna(0.0)
        stats["Pred_Disagreement_Range"] = (wide.max(axis=1, skipna=True) - wide.min(axis=1, skipna=True)).fillna(0.0)
        q75 = wide.quantile(0.75, axis=1, interpolation="linear")
        q25 = wide.quantile(0.25, axis=1, interpolation="linear")
        stats["Pred_Disagreement_IQR"] = (q75 - q25).fillna(0.0)
        stats = stats.reset_index()
        out = out.merge(stats, on=usable_keys, how="left")
        for c in ["Pred_Disagreement_SD", "Pred_Disagreement_Range", "Pred_Disagreement_IQR"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    except Exception as exc:
        log(f"[WARN] Failed to construct disagreement features: {exc}")
        out["Pred_Disagreement_SD"] = 0.0
        out["Pred_Disagreement_Range"] = 0.0
        out["Pred_Disagreement_IQR"] = 0.0
    return out


def minmax_rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if valid.sum() == 0:
        return np.zeros_like(x, dtype=float)
    vals = x[valid]
    lo = np.nanmin(vals)
    hi = np.nanmax(vals)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out[valid] = 0.0
    else:
        out[valid] = (vals - lo) / (hi - lo)
    out[~valid] = 1.0
    return out


def add_composite_policy_scores(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Add target-free composite AD/disagreement scores."""
    out = df.copy()
    cols = [c for c in feature_cols if c in out.columns]

    ad_like = [c for c in cols if any(tok in normalize_name(c) for tok in ["risk", "violation", "unseen", "novel", "distance", "leverage", "boundary", "mahalanobis", "reconstruction", "nn"])]
    disagreement = [c for c in ["Pred_Disagreement_SD", "Pred_Disagreement_Range", "Pred_Disagreement_IQR"] if c in out.columns]

    new_cols: List[str] = []
    if ad_like:
        arrs = [minmax_rank01(to_numeric_array(out[c])) for c in ad_like]
        mat = np.vstack(arrs).T
        out["Composite_AD_MeanRisk"] = np.nanmean(mat, axis=1)
        out["Composite_AD_MaxRisk"] = np.nanmax(mat, axis=1)
        new_cols += ["Composite_AD_MeanRisk", "Composite_AD_MaxRisk"]

    if disagreement:
        arrs = [minmax_rank01(to_numeric_array(out[c])) for c in disagreement]
        mat = np.vstack(arrs).T
        out["Composite_PredDisagreement"] = np.nanmean(mat, axis=1)
        new_cols.append("Composite_PredDisagreement")

    if ad_like and disagreement:
        out["Composite_ADxDisagreement"] = 0.60 * out["Composite_AD_MeanRisk"] + 0.40 * out["Composite_PredDisagreement"]
        new_cols.append("Composite_ADxDisagreement")

    return out, new_cols


def rank_policy_features(df: pd.DataFrame, candidate_feature_cols: Sequence[str]) -> List[str]:
    """Rank target-free features for policy search using availability and variability."""
    rows = []
    for c in candidate_feature_cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        finite = x[np.isfinite(x)]
        if len(finite) == 0:
            continue
        unique_n = finite.nunique(dropna=True)
        if unique_n <= 1:
            continue
        priority = 0
        nc = normalize_name(c)
        for token in ["correctedad", "boundary", "distance", "leverage", "mahalanobis", "nn", "reconstruction", "unseen", "violation", "disagreement", "composite"]:
            if token in nc:
                priority += 1
        rows.append({
            "Feature": c,
            "FiniteFraction": len(finite) / len(df),
            "UniqueN": unique_n,
            "Std": float(np.nanstd(finite)),
            "Priority": priority,
        })
    if not rows:
        return []
    audit = pd.DataFrame(rows)
    audit["Score"] = audit["Priority"] * 10 + audit["FiniteFraction"] + np.log1p(audit["UniqueN"]) / 10
    audit = audit.sort_values(["Score", "Std"], ascending=[False, False])
    return audit["Feature"].head(MAX_SINGLE_RULE_FEATURES).tolist()


def make_policies(df: pd.DataFrame, feature_cols: Sequence[str], zone_col: Optional[str]) -> List[Policy]:
    """Create a finite candidate policy set without exponential brute force."""
    policies: List[Policy] = []
    policies.append(Policy(
        name="All_domain",
        family="reference",
        conditions=tuple(),
        description="Reference: retain all samples; not a selective rule.",
    ))

    # Existing reliability-zone policies, if present.
    if zone_col is not None and zone_col in df.columns:
        z = df[zone_col].astype(str).str.lower()
        values = sorted(df[zone_col].dropna().astype(str).unique().tolist())
        for value in values:
            vl = value.lower()
            if any(k in vl for k in ["reliable", "warning", "outside", "domain"]):
                policies.append(Policy(
                    name=f"Zone_eq_{re.sub(r'[^A-Za-z0-9]+', '', value)[:24]}",
                    family="existing_zone",
                    conditions=(Condition(zone_col, "eq", value),),
                    description=f"Retain existing reliability-zone category: {value}",
                ))
        # Common useful rule: not outside.
        outside_values = [v for v in values if "outside" in v.lower()]
        if outside_values:
            policies.append(Policy(
                name="Zone_not_outside",
                family="existing_zone",
                conditions=tuple(Condition(zone_col, "ne", v) for v in outside_values),
                description="Retain samples not assigned to outside-domain zone.",
            ))

    # Single feature threshold policies.
    for c in feature_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        finite = x[np.isfinite(x)]
        if len(finite) < 20 or finite.nunique() <= 1:
            continue
        qs = np.nanquantile(finite.to_numpy(dtype=float), QUANTILE_GRID)
        seen = set()
        for q, thr in zip(QUANTILE_GRID, qs):
            if not np.isfinite(thr):
                continue
            key = round(float(thr), 10)
            if key in seen:
                continue
            seen.add(key)
            policies.append(Policy(
                name=f"{short_label(c, 26)}_le_q{int(round(q * 100)):02d}",
                family="single_threshold",
                conditions=(Condition(c, "<=", float(thr)),),
                description=f"Retain samples with {c} <= its global q={q:.2f} threshold ({thr:.4g}).",
            ))

    # Controlled two-condition policies using the strongest AD/composite features.
    combo_cols = [c for c in feature_cols if c in df.columns][:MAX_COMBO_FEATURES]
    combo_quantiles = [0.35, 0.50, 0.65, 0.80]
    for i, c1 in enumerate(combo_cols):
        for c2 in combo_cols[i + 1:]:
            x1 = pd.to_numeric(df[c1], errors="coerce")
            x2 = pd.to_numeric(df[c2], errors="coerce")
            f1 = x1[np.isfinite(x1)]
            f2 = x2[np.isfinite(x2)]
            if len(f1) < 20 or len(f2) < 20 or f1.nunique() <= 1 or f2.nunique() <= 1:
                continue
            for q in combo_quantiles:
                t1 = float(np.nanquantile(f1, q))
                t2 = float(np.nanquantile(f2, q))
                if np.isfinite(t1) and np.isfinite(t2):
                    policies.append(Policy(
                        name=f"Combo_{short_label(c1, 14)}_{short_label(c2, 14)}_q{int(q*100)}",
                        family="two_condition",
                        conditions=(Condition(c1, "<=", t1), Condition(c2, "<=", t2)),
                        description=f"Retain samples satisfying {c1} <= q{q:.2f} and {c2} <= q{q:.2f}.",
                    ))

    # Deduplicate by name.
    dedup: Dict[str, Policy] = {}
    for p in policies:
        base = p.name
        name = base
        k = 1
        while name in dedup:
            k += 1
            name = f"{base}_{k}"
        if name != p.name:
            p = Policy(name=name, family=p.family, conditions=p.conditions, description=p.description)
        dedup[name] = p
    return list(dedup.values())


def build_policy_mask_matrix(df: pd.DataFrame, policies: Sequence[Policy]) -> np.ndarray:
    """Return N x C boolean matrix of candidate policy masks."""
    n = len(df)
    mats = np.zeros((n, len(policies)), dtype=bool)
    numeric_cache: Dict[str, np.ndarray] = {}
    string_cache: Dict[str, np.ndarray] = {}

    for j, policy in enumerate(policies):
        mask = np.ones(n, dtype=bool)
        for cond in policy.conditions:
            col = cond.col
            if col not in df.columns:
                mask &= False
                continue
            if cond.op in ["<=", ">=", "<", ">"]:
                if col not in numeric_cache:
                    numeric_cache[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                arr = numeric_cache[col]
                val = float(cond.value)
                if cond.op == "<=":
                    mask &= np.isfinite(arr) & (arr <= val)
                elif cond.op == ">=":
                    mask &= np.isfinite(arr) & (arr >= val)
                elif cond.op == "<":
                    mask &= np.isfinite(arr) & (arr < val)
                elif cond.op == ">":
                    mask &= np.isfinite(arr) & (arr > val)
            elif cond.op in ["eq", "ne"]:
                if col not in string_cache:
                    string_cache[col] = df[col].astype(str).to_numpy()
                arrs = string_cache[col]
                if cond.op == "eq":
                    mask &= (arrs == str(cond.value))
                else:
                    mask &= (arrs != str(cond.value))
            elif cond.op == "contains_ci":
                if col not in string_cache:
                    string_cache[col] = df[col].astype(str).str.lower().to_numpy()
                arrs = string_cache[col]
                mask &= np.array([str(cond.value).lower() in x for x in arrs], dtype=bool)
            else:
                raise ValueError(f"Unknown condition op: {cond.op}")
        mats[:, j] = mask
    return mats


# =============================================================================
# Strict grouped calibration and diagnostic oracle
# =============================================================================


def objective_scores(metrics: pd.DataFrame, objective: str) -> np.ndarray:
    r2 = metrics["R2"].to_numpy(dtype=float)
    rmse = metrics["RMSE"].to_numpy(dtype=float)
    cov = metrics["coverage"].to_numpy(dtype=float)
    n = metrics["n"].to_numpy(dtype=float)

    valid = np.isfinite(r2) & np.isfinite(rmse) & (cov >= MIN_CALIB_COVERAGE) & (n >= MIN_CALIB_N)
    score = np.full(len(metrics), -np.inf, dtype=float)

    if objective == "R2_oriented":
        # Favor R2, but avoid microscopic coverage.
        score[valid] = r2[valid] + 0.15 * cov[valid] - 0.002 * rmse[valid]
    elif objective == "RMSE_oriented":
        # Favor low RMSE, but keep coverage meaningful.
        score[valid] = -rmse[valid] + 5.0 * cov[valid] + 3.0 * np.nan_to_num(r2[valid], nan=-1.0)
    elif objective == "Balanced_R2_RMSE":
        score[valid] = r2[valid] - 0.010 * rmse[valid] + 0.25 * cov[valid]
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return score


def run_grouped_calibration(
    df: pd.DataFrame,
    mask_mat: np.ndarray,
    policies: Sequence[Policy],
    obs_col: str,
    pred_col: str,
    strategy_col: str,
    method_col: str,
    block_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Grouped block-wise policy selection and held-out evaluation."""
    y_all = to_numeric_array(df[obs_col])
    p_all = to_numeric_array(df[pred_col])

    selected_blocks: List[Dict[str, object]] = []
    retained_frames: List[pd.DataFrame] = []

    strategies = [x for x in df[strategy_col].dropna().astype(str).unique().tolist()]
    methods_all = [x for x in df[method_col].dropna().astype(str).unique().tolist()]
    methods = [m for m in PREFERRED_METHOD_ORDER if m in methods_all] + [m for m in methods_all if m not in PREFERRED_METHOD_ORDER]

    total_jobs = sum(len(df[(df[strategy_col].astype(str) == str(s)) & (df[method_col].astype(str) == str(m))][block_col].dropna().astype(str).unique())
                     for s in strategies for m in methods)
    job_i = 0

    for strategy in strategies:
        for method in methods:
            sub_mask = (df[strategy_col].astype(str).to_numpy() == str(strategy)) & \
                       (df[method_col].astype(str).to_numpy() == str(method))
            sub_idx = np.where(sub_mask)[0]
            if len(sub_idx) < 20:
                continue
            blocks = sorted(pd.Series(df.iloc[sub_idx][block_col]).dropna().astype(str).unique().tolist())
            if len(blocks) < 2:
                # Cannot do block-wise calibration. Skip deployable policy.
                continue

            block_values = df[block_col].astype(str).to_numpy()
            for block in blocks:
                job_i += 1
                if job_i % 25 == 0:
                    log(f"  [progress] grouped calibration jobs: {job_i}/{total_jobs}")

                held_idx = sub_idx[block_values[sub_idx] == str(block)]
                calib_idx = sub_idx[block_values[sub_idx] != str(block)]
                if len(held_idx) == 0 or len(calib_idx) < MIN_CALIB_N:
                    continue

                calib_y = y_all[calib_idx]
                calib_p = p_all[calib_idx]
                calib_masks = mask_mat[calib_idx, :]
                calib_metrics = vectorized_metrics_for_masks(calib_y, calib_p, calib_masks)

                for objective in OBJECTIVES:
                    scores = objective_scores(calib_metrics, objective)
                    if not np.isfinite(scores).any():
                        # Fall back to All_domain if no selective candidate is valid.
                        best_j = 0
                    else:
                        best_j = int(np.nanargmax(scores))
                    best_policy = policies[best_j]

                    held_policy_mask = mask_mat[held_idx, best_j]
                    selected_held_idx = held_idx[held_policy_mask]

                    held_y = y_all[selected_held_idx]
                    held_p = p_all[selected_held_idx]
                    m = metrics_from_arrays(held_y, held_p) if len(selected_held_idx) > 0 else {"n": 0, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan}
                    held_coverage = len(selected_held_idx) / max(1, len(held_idx))

                    selected_blocks.append({
                        "Strategy": strategy,
                        "Method": method,
                        "Block": block,
                        "Objective": objective,
                        "SelectedPolicy": best_policy.name,
                        "PolicyFamily": best_policy.family,
                        "Calibration_R2": float(calib_metrics.iloc[best_j]["R2"]),
                        "Calibration_RMSE": float(calib_metrics.iloc[best_j]["RMSE"]),
                        "Calibration_MAE": float(calib_metrics.iloc[best_j]["MAE"]),
                        "Calibration_Coverage": float(calib_metrics.iloc[best_j]["coverage"]),
                        "Heldout_N_All": int(len(held_idx)),
                        "Heldout_N_Retained": int(len(selected_held_idx)),
                        "Heldout_Coverage": float(held_coverage),
                        "Heldout_R2": m["R2"],
                        "Heldout_RMSE": m["RMSE"],
                        "Heldout_MAE": m["MAE"],
                    })

                    if len(selected_held_idx) > 0:
                        part = df.iloc[selected_held_idx].copy()
                        part["32C_Objective"] = objective
                        part["32C_SelectedPolicy"] = best_policy.name
                        part["32C_PolicyFamily"] = best_policy.family
                        part["32C_CalibrationBlock"] = block
                        retained_frames.append(part)

    selected_block_df = pd.DataFrame(selected_blocks)
    retained_df = pd.concat(retained_frames, ignore_index=True) if retained_frames else pd.DataFrame()

    # Pooled summary for retained predictions.
    rows = []
    if not retained_df.empty:
        group_cols = ["32C_Objective", strategy_col, method_col]
        for keys, g in retained_df.groupby(group_cols, dropna=False):
            obj, strat, meth = keys
            base = df[(df[strategy_col].astype(str) == str(strat)) & (df[method_col].astype(str) == str(meth))]
            y = to_numeric_array(g[obs_col])
            p = to_numeric_array(g[pred_col])
            m = metrics_from_arrays(y, p)
            rows.append({
                "Objective": obj,
                "Strategy": strat,
                "Method": meth,
                "N_Retained": int(len(g)),
                "N_AllComparable": int(len(base)),
                "Coverage": float(len(g) / max(1, len(base))),
                "R2": m["R2"],
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
            })

        # Also pooled across strategies for each method/objective.
        for keys, g in retained_df.groupby(["32C_Objective", method_col], dropna=False):
            obj, meth = keys
            base = df[df[method_col].astype(str) == str(meth)]
            y = to_numeric_array(g[obs_col])
            p = to_numeric_array(g[pred_col])
            m = metrics_from_arrays(y, p)
            rows.append({
                "Objective": obj,
                "Strategy": "ALL_STRATEGIES_POOLED",
                "Method": meth,
                "N_Retained": int(len(g)),
                "N_AllComparable": int(len(base)),
                "Coverage": float(len(g) / max(1, len(base))),
                "R2": m["R2"],
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
            })

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["R2", "RMSE"], ascending=[False, True]).reset_index(drop=True)
    return selected_block_df, retained_df, summary_df


def run_diagnostic_oracle(
    df: pd.DataFrame,
    mask_mat: np.ndarray,
    policies: Sequence[Policy],
    obs_col: str,
    pred_col: str,
    strategy_col: str,
    method_col: str,
) -> pd.DataFrame:
    """Non-deployable upper bound: evaluates all candidate policies using labels."""
    y_all = to_numeric_array(df[obs_col])
    p_all = to_numeric_array(df[pred_col])
    rows = []

    for (strategy, method), g in df.groupby([strategy_col, method_col], dropna=False):
        idx = g.index.to_numpy()
        if len(idx) < MIN_HELDOUT_N_TOTAL:
            continue
        mdf = vectorized_metrics_for_masks(y_all[idx], p_all[idx], mask_mat[idx, :])
        # Enforce meaningful coverage for the diagnostic upper bound too.
        valid = (mdf["coverage"].to_numpy() >= MIN_HELDOUT_COVERAGE) & (mdf["n"].to_numpy() >= MIN_HELDOUT_N_TOTAL) & np.isfinite(mdf["R2"].to_numpy())
        if not valid.any():
            continue
        scores = mdf["R2"].to_numpy().copy()
        scores[~valid] = -np.inf
        best_j = int(np.nanargmax(scores))
        rows.append({
            "Strategy": strategy,
            "Method": method,
            "DiagnosticBestPolicy": policies[best_j].name,
            "PolicyFamily": policies[best_j].family,
            "N_Retained": int(mdf.iloc[best_j]["n"]),
            "N_AllComparable": int(len(idx)),
            "Coverage": float(mdf.iloc[best_j]["coverage"]),
            "R2": float(mdf.iloc[best_j]["R2"]),
            "RMSE": float(mdf.iloc[best_j]["RMSE"]),
            "MAE": float(mdf.iloc[best_j]["MAE"]),
            "Interpretation": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE_LABEL_SELECTED",
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["R2", "RMSE"], ascending=[False, True]).reset_index(drop=True)
    return out


# =============================================================================
# Plotting
# =============================================================================


def annotate_bars(ax: plt.Axes, fmt: str = "{:.2f}", rotation: int = 0) -> None:
    for patch in ax.patches:
        h = patch.get_height()
        if not np.isfinite(h):
            continue
        x = patch.get_x() + patch.get_width() / 2
        offset = 0.01 * max(1.0, abs(ax.get_ylim()[1] - ax.get_ylim()[0]))
        if h >= 0:
            y = h + offset
            va = "bottom"
        else:
            y = h - offset
            va = "top"
        ax.text(x, y, fmt.format(h), ha="center", va=va, fontsize=8, rotation=rotation)


def plot_policy_metrics(summary_df: pd.DataFrame, fig_dir: Path) -> None:
    if summary_df.empty:
        return
    top = summary_df.sort_values(["R2", "RMSE"], ascending=[False, True]).head(12).copy()
    labels = [short_label(f"{o}|{m}|{s}", 36) for o, m, s in zip(top["Objective"], top["Method"], top["Strategy"])]

    fig, axes = plt.subplots(1, 4, figsize=(21, 5.2))
    for ax, metric, title in zip(axes, ["R2", "RMSE", "MAE", "Coverage"], ["Retained R2", "Retained RMSE", "Retained MAE", "Coverage"]):
        values = top[metric].to_numpy(dtype=float)
        ax.bar(range(len(top)), values, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
        annotate_bars(ax, fmt="{:.2f}" if metric != "R2" else "{:.3f}", rotation=0)
        if metric == "R2":
            ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0, label="R2=0.5 target")
            ax.legend(frameon=False)
    fig.suptitle("32C fast selective AD policy metrics", fontsize=16, fontweight="bold")
    save_fig(fig, fig_dir / "158_32c_fast_policy_metrics.png")


def plot_frontier(summary_df: pd.DataFrame, oracle_df: pd.DataFrame, fig_dir: Path) -> None:
    if summary_df.empty and oracle_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    if not summary_df.empty:
        sc = ax.scatter(summary_df["Coverage"], summary_df["R2"], s=80, alpha=0.75, label="Strict block-calibrated policy")
        for _, r in summary_df.sort_values("R2", ascending=False).head(8).iterrows():
            ax.text(r["Coverage"] + 0.006, r["R2"], short_label(f"{r['Method']}|{r['Strategy']}", 28), fontsize=8)
    if not oracle_df.empty:
        ax.scatter(oracle_df["Coverage"], oracle_df["R2"], s=50, alpha=0.45, marker="x", label="Diagnostic oracle upper bound")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0, label="R2=0.5 target")
    ax.axvline(MIN_HELDOUT_COVERAGE, color="gray", linestyle=":", linewidth=1.0, label="min coverage")
    ax.set_xlabel("Coverage retained")
    ax.set_ylabel("R2 within retained domain")
    ax.set_title("R2-coverage frontier under selective AD policies")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="best")
    save_fig(fig, fig_dir / "159_32c_fast_r2_coverage_frontier.png")


def plot_best_prediction_panel(retained_df: pd.DataFrame, summary_df: pd.DataFrame, obs_col: str, pred_col: str, fig_dir: Path) -> None:
    if retained_df.empty or summary_df.empty:
        return
    best = summary_df.sort_values(["R2", "RMSE"], ascending=[False, True]).iloc[0]
    g = retained_df[
        (retained_df["32C_Objective"].astype(str) == str(best["Objective"])) &
        (retained_df["Strategy"].astype(str) == str(best["Strategy"])) &
        (retained_df["Method"].astype(str) == str(best["Method"]))
    ].copy()
    if g.empty and best["Strategy"] == "ALL_STRATEGIES_POOLED":
        g = retained_df[
            (retained_df["32C_Objective"].astype(str) == str(best["Objective"])) &
            (retained_df["Method"].astype(str) == str(best["Method"]))
        ].copy()
    if g.empty:
        return

    y = to_numeric_array(g[obs_col])
    p = to_numeric_array(g[pred_col])
    m = metrics_from_arrays(y, p)

    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    ax.scatter(y, p, s=24, alpha=0.65, edgecolor="none")
    ax.plot([0, 100], [0, 100], "r--", linewidth=1.2)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Predicted F removal (%)")
    ax.set_title(f"Best strict selective policy: {best['Method']}")
    ax.text(5, 97, f"R2 = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nMAE = {m['MAE']:.2f}\nn = {m['n']}\ncoverage = {best['Coverage']:.2f}",
            va="top", ha="left", fontsize=10, bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))
    ax.grid(alpha=0.25)
    save_fig(fig, fig_dir / "160_32c_fast_best_prediction_panel.png")


def plot_strategy_breakdown(summary_df: pd.DataFrame, fig_dir: Path) -> None:
    if summary_df.empty:
        return
    non_pooled = summary_df[summary_df["Strategy"].astype(str) != "ALL_STRATEGIES_POOLED"].copy()
    if non_pooled.empty:
        return
    # Keep best method per strategy/objective by R2.
    best_by_strategy = non_pooled.sort_values(["Strategy", "R2", "RMSE"], ascending=[True, False, True]).groupby("Strategy", as_index=False).head(3)
    labels = [short_label(f"{s}|{m}", 32) for s, m in zip(best_by_strategy["Strategy"], best_by_strategy["Method"])]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, metric in zip(axes, ["R2", "RMSE", "Coverage"]):
        vals = best_by_strategy[metric].to_numpy(dtype=float)
        ax.bar(range(len(vals)), vals, edgecolor="black", linewidth=0.5)
        ax.set_title(metric)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
        annotate_bars(ax, fmt="{:.2f}" if metric != "R2" else "{:.3f}")
        if metric == "R2":
            ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0)
    fig.suptitle("Best selective-policy candidates by validation strategy", fontsize=15, fontweight="bold")
    save_fig(fig, fig_dir / "161_32c_fast_strategy_breakdown.png")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    log(f"Running {SCRIPT_NAME}")
    root = find_project_root()
    fig_dir, table_dir = ensure_dirs(root)
    log(f"Project root: {root}")

    input_candidates = [
        table_dir / "32b_predictions_with_reliability_zone.csv",
        table_dir / "32a2_ad_predictions_with_scores.csv",
        table_dir / "32a_ad_predictions_with_scores.csv",
        table_dir / "32a_ad_predictions_with_scores.csv",
    ]
    input_path = next((p for p in input_candidates if p.exists()), None)
    if input_path is None:
        raise FileNotFoundError(
            "Cannot find Step 32 prediction-with-AD table. Expected one of:\n" +
            "\n".join(str(p) for p in input_candidates)
        )

    df = read_csv_robust(input_path)
    df = df.reset_index(drop=True)
    log(f"[INFO] Loaded: {input_path} | shape={df.shape}")

    obs_col = find_col(df, ["Observed", "Actual", "y_true", "Observed F removal (%)", "F removal", "氟去除率"])
    pred_col = find_col(df, ["Predicted", "Prediction", "y_pred", "Predicted F removal (%)"])
    method_col = find_col(df, ["Method", "Model", "Estimator"])
    strategy_col = find_col(df, ["Strategy", "ValidationStrategy", "Validation_Strategy"])
    group_col = find_col(df, ["文献编号", "Literature", "Paper", "Group", "Source_Group"], required=False)
    split_col = find_col(df, ["Split_ID", "SplitID", "Seed_or_Fold", "Seed", "Fold", "Block"], required=False)
    data_index_col = find_col(df, ["Data_Index", "_Data_Index", "Index", "Row_ID"], required=False)
    zone_col = find_col(df, ["Reliability_Zone", "Domain_Zone", "Reliable_Zone", "AD_Zone"], required=False)

    if split_col is None:
        # Fall back to group column as block if needed.
        if group_col is not None:
            split_col = group_col
        else:
            df["_Synthetic_Block"] = np.arange(len(df)) % 5
            split_col = "_Synthetic_Block"

    # Standardize key columns as strings where needed.
    df[strategy_col] = df[strategy_col].astype(str)
    df[method_col] = df[method_col].astype(str)
    df[split_col] = df[split_col].astype(str)
    if group_col is not None:
        df[group_col] = df[group_col].astype(str)

    log(f"[INFO] Observed column: {obs_col}")
    log(f"[INFO] Predicted column: {pred_col}")
    log(f"[INFO] Strategy column: {strategy_col}")
    log(f"[INFO] Method column: {method_col}")
    log(f"[INFO] Block column for grouped calibration: {split_col}")
    log(f"[INFO] Literature/group column: {group_col}")
    log(f"[INFO] Reliability-zone column: {zone_col}")

    # Clean numeric target/prediction.
    df[obs_col] = pd.to_numeric(df[obs_col], errors="coerce")
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    before = len(df)
    df = df[np.isfinite(df[obs_col]) & np.isfinite(df[pred_col])].copy().reset_index(drop=True)
    if len(df) < before:
        log(f"[WARN] Removed {before - len(df)} rows with invalid observed/predicted values.")

    key_cols = [strategy_col, split_col]
    if group_col is not None:
        key_cols.append(group_col)
    if data_index_col is not None:
        key_cols.append(data_index_col)

    df = add_prediction_disagreement_features(df, strategy_col, method_col, pred_col, key_cols)

    core_cols = [obs_col, pred_col, method_col, strategy_col, split_col]
    if group_col:
        core_cols.append(group_col)
    if data_index_col:
        core_cols.append(data_index_col)

    initial_feature_cols = [c for c in df.columns if is_target_free_policy_feature(c, core_cols)]
    # Add composite scores based on target-free features.
    df, composite_cols = add_composite_policy_scores(df, initial_feature_cols)
    all_feature_cols = list(dict.fromkeys(initial_feature_cols + composite_cols + [
        "Pred_Disagreement_SD", "Pred_Disagreement_Range", "Pred_Disagreement_IQR"
    ]))
    ranked_features = rank_policy_features(df, all_feature_cols)

    feature_audit_rows = []
    for c in ranked_features:
        x = pd.to_numeric(df[c], errors="coerce")
        feature_audit_rows.append({
            "PolicyFeature": c,
            "FiniteFraction": float(np.isfinite(x).mean()),
            "UniqueN": int(x.nunique(dropna=True)),
            "Min": float(np.nanmin(x)) if np.isfinite(x).any() else np.nan,
            "Median": float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan,
            "Max": float(np.nanmax(x)) if np.isfinite(x).any() else np.nan,
            "UsedForPolicy": True,
        })
    feature_audit = pd.DataFrame(feature_audit_rows)
    save_csv(feature_audit, table_dir / "32c_fast_policy_feature_audit.csv")
    log(f"[INFO] Policy feature columns used: {len(ranked_features)}")

    policies = make_policies(df, ranked_features, zone_col)
    policy_catalog = pd.DataFrame({
        "PolicyName": [p.name for p in policies],
        "Family": [p.family for p in policies],
        "NConditions": [len(p.conditions) for p in policies],
        "Description": [p.description for p in policies],
    })
    save_csv(policy_catalog, table_dir / "32c_fast_candidate_policy_catalog.csv")
    log(f"[INFO] Candidate policies: {len(policies)}")

    log("[STEP] Building vectorized candidate mask matrix...")
    mask_mat = build_policy_mask_matrix(df, policies)
    log(f"[INFO] Mask matrix shape: {mask_mat.shape}; memory approx {mask_mat.nbytes / 1024 / 1024:.1f} MB")

    log("[STEP] Strict grouped block-wise calibrated policy search...")
    selected_block_df, retained_df, summary_df = run_grouped_calibration(
        df=df,
        mask_mat=mask_mat,
        policies=policies,
        obs_col=obs_col,
        pred_col=pred_col,
        strategy_col=strategy_col,
        method_col=method_col,
        block_col=split_col,
    )
    save_csv(selected_block_df, table_dir / "32c_fast_blockwise_selected_policy_summary.csv")
    save_csv(summary_df, table_dir / "32c_fast_policy_summary.csv")
    if not retained_df.empty:
        save_csv(retained_df, table_dir / "32c_fast_selected_holdout_predictions.csv")
    else:
        save_csv(pd.DataFrame(), table_dir / "32c_fast_selected_holdout_predictions.csv")

    log("[STEP] Diagnostic oracle upper-bound screen...")
    oracle_df = run_diagnostic_oracle(
        df=df,
        mask_mat=mask_mat,
        policies=policies,
        obs_col=obs_col,
        pred_col=pred_col,
        strategy_col=strategy_col,
        method_col=method_col,
    )
    save_csv(oracle_df, table_dir / "32c_fast_diagnostic_oracle_upper_bound.csv")

    log("[STEP] Plotting...")
    plot_policy_metrics(summary_df, fig_dir)
    plot_frontier(summary_df, oracle_df, fig_dir)
    plot_best_prediction_panel(retained_df, summary_df, obs_col, pred_col, fig_dir)
    plot_strategy_breakdown(summary_df, fig_dir)

    # Decision report.
    report_lines: List[str] = []
    report_lines.append("32C fast selective AD R2 boosting decision report")
    report_lines.append("=" * 78)
    report_lines.append(f"Input table: {input_path}")
    report_lines.append(f"Rows after cleaning: {len(df)}")
    report_lines.append(f"Candidate policy features: {len(ranked_features)}")
    report_lines.append(f"Candidate policies: {len(policies)}")
    report_lines.append("")
    report_lines.append("Strictness statement")
    report_lines.append("- Serious policies are selected by grouped block-wise calibration.")
    report_lines.append("- Held-out labels are not used to select policy for that held-out block.")
    report_lines.append("- Diagnostic oracle is reported only as a non-deployable upper bound.")
    report_lines.append("")

    if summary_df.empty:
        report_lines.append("No valid strict selective policy was obtained. Check input columns and block definitions.")
    else:
        best = summary_df.sort_values(["R2", "RMSE"], ascending=[False, True]).iloc[0]
        report_lines.append("Best strict block-calibrated selective policy")
        report_lines.append(f"- Objective: {best['Objective']}")
        report_lines.append(f"- Strategy: {best['Strategy']}")
        report_lines.append(f"- Method: {best['Method']}")
        report_lines.append(f"- Coverage: {best['Coverage']:.3f}")
        report_lines.append(f"- R2: {best['R2']:.4f}")
        report_lines.append(f"- RMSE: {best['RMSE']:.4f}")
        report_lines.append(f"- MAE: {best['MAE']:.4f}")
        report_lines.append("")
        if np.isfinite(best["R2"]) and float(best["R2"]) >= 0.5:
            report_lines.append("Decision: STRICT calibrated selective policy reached R2 >= 0.5.")
            report_lines.append("This is potentially usable as a publication-grade reliable-domain result, subject to manual review of coverage and chemistry.")
        else:
            report_lines.append("Decision: STRICT calibrated selective policy did NOT reach R2 >= 0.5.")
            report_lines.append("Do not claim final AD-domain R2 > 0.5 from strict policy unless a later validated policy reaches it.")
        report_lines.append("")
        report_lines.append("Top strict policies")
        for i, (_, r) in enumerate(summary_df.sort_values(["R2", "RMSE"], ascending=[False, True]).head(12).iterrows(), 1):
            report_lines.append(
                f"{i:02d}. Obj={r['Objective']} | Strategy={r['Strategy']} | Method={r['Method']} | "
                f"Coverage={r['Coverage']:.3f} | R2={r['R2']:.4f} | RMSE={r['RMSE']:.3f} | MAE={r['MAE']:.3f}"
            )

    report_lines.append("")
    if not oracle_df.empty:
        ob = oracle_df.iloc[0]
        report_lines.append("Diagnostic oracle upper bound, NOT deployable")
        report_lines.append(f"- Strategy: {ob['Strategy']}")
        report_lines.append(f"- Method: {ob['Method']}")
        report_lines.append(f"- Policy: {ob['DiagnosticBestPolicy']}")
        report_lines.append(f"- Coverage: {ob['Coverage']:.3f}")
        report_lines.append(f"- R2: {ob['R2']:.4f}")
        report_lines.append(f"- RMSE: {ob['RMSE']:.4f}")
        report_lines.append("This only tells whether a high-R2 subdomain exists in hindsight; it cannot be used as final deployable evidence.")
    else:
        report_lines.append("No diagnostic oracle result available.")

    elapsed = time.time() - START_TIME
    report_lines.append("")
    report_lines.append(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    report_path = table_dir / "32c_fast_decision_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    log(f"[REPORT] {report_path}")

    log("=" * 90)
    log("32C fast selective AD R2 boosting finished.")
    log(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    log("=" * 90)


if __name__ == "__main__":
    main()
