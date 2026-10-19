# -*- coding: utf-8 -*-
"""
32c_selective_ad_r2_boosting.py

Purpose
-------
Step 32C for fluoride coagulation ML project.

This script searches whether a high-R2 reliable subdomain can be obtained by
selective prediction / rejection policy under grouped validation.

Important strictness
--------------------
1. It does NOT retrain the main prediction models.
2. It reuses row-level prediction + AD-score outputs from Step 32A2 / 32B.
3. Serious calibrated policies are selected on calibration/grouped blocks and
   evaluated on held-out blocks.
4. It does NOT use held-out/test residuals to train the policy for that same
   held-out block.
5. It also reports a diagnostic oracle upper bound. This is NOT deployable and
   is only used to judge whether the existing AD/prediction signals contain any
   potentially high-R2 subdomain.

Expected input files
--------------------
Preferred:
    outputs/tables/32b_predictions_with_reliability_zone.csv
Fallback:
    outputs/tables/32a2_ad_predictions_with_scores.csv
    outputs/tables/32a_ad_predictions_with_scores.csv

Outputs
-------
outputs/tables/
    32c_selective_policy_feature_audit.csv
    32c_selective_policy_summary.csv
    32c_selective_policy_predictions.csv
    32c_selected_policy_details.csv
    32c_diagnostic_oracle_upper_bound.csv
    32c_selective_ad_decision_report.txt

outputs/figures/
    158_32c_selective_policy_metrics.png/pdf
    159_32c_r2_coverage_frontier.png/pdf
    160_32c_best_selective_predicted_vs_observed.png/pdf
    161_32c_dynamic_method_selection_frequency.png/pdf
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_NAME = "32c_selective_ad_r2_boosting.py"
START_TIME = time.time()

RANDOM_STATE = 20260605
TARGET_R2 = 0.50

MIN_COVERAGE_GRID = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
MIN_N_ABSOLUTE = 30

PREFERRED_METHODS = [
    "KRR_Poly",
    "KRR_CatBoost_Weighted_2Model",
    "KRR_XGBoost_Weighted_2Model",
    "CB-XGB",
    "CatBoost",
    "XGBoost",
    "GPR_Matern",
    "PLS",
]

MAX_METHODS_FOR_DYNAMIC_SWITCH = 8

QUANTILE_GRID_FINE = [
    0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
]

WINDOW_Q_LOW = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
WINDOW_Q_HIGH = [0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 1.00]

RISK_RETAIN_FRACTIONS = [
    0.05, 0.08, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.50, 0.60, 0.70,
]


def log(msg: str) -> None:
    print(msg, flush=True)


def find_project_root() -> Path:
    starts: List[Path] = []
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    starts.append(Path.cwd().resolve())

    for start in starts:
        p = start
        for _ in range(8):
            if (p / "outputs").exists() and (p / "data").exists():
                return p
            if (p / "outputs" / "tables").exists():
                return p
            p = p.parent

    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df: pd.DataFrame, filename: str) -> Path:
    path = TABLE_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {path} | shape={df.shape}")
    return path


def save_fig(fig: plt.Figure, filename: str) -> None:
    png = FIG_DIR / filename
    pdf = FIG_DIR / filename.replace(".png", ".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] {png}")
    log(f"[FIG] {pdf}")


def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def finite_array(s: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def infer_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in low:
            return low[c.lower()]
    return None


def infer_true_pred_columns(df: pd.DataFrame) -> Tuple[str, str]:
    y_col = infer_column(df, [
        "Observed", "Observed_F_removal", "Observed_F_removal_pct",
        "Observed_F_removal(%)", "Observed F removal (%)", "Actual",
        "Actual_F_removal", "y_true", "Y_true", "target", "Target",
        "F_removal", "氟去除率", "氟去除率(%)",
    ])
    p_col = infer_column(df, [
        "Predicted", "Predicted_F_removal", "Predicted_F_removal_pct",
        "Predicted_F_removal(%)", "Predicted F removal (%)", "Prediction",
        "y_pred", "Y_pred", "pred",
    ])

    if y_col is None:
        for c in df.columns:
            lc = c.lower()
            if ("observ" in lc or "actual" in lc or "true" in lc or "target" in lc) and ("error" not in lc):
                y_col = c
                break

    if p_col is None:
        for c in df.columns:
            lc = c.lower()
            if ("pred" in lc or "prediction" in lc) and ("error" not in lc):
                p_col = c
                break

    if y_col is None or p_col is None:
        raise ValueError(
            "Cannot infer observed/predicted columns. Current columns:\n"
            + "\n".join(map(str, df.columns))
        )

    return y_col, p_col


def infer_strategy_method_columns(df: pd.DataFrame) -> Tuple[str, str]:
    strategy_col = infer_column(df, ["Strategy", "Validation_Strategy", "strategy"])
    method_col = infer_column(df, ["Method", "Model", "Estimator", "method", "model"])

    if strategy_col is None:
        df["Strategy"] = "Unknown_Strategy"
        strategy_col = "Strategy"
    if method_col is None:
        raise ValueError("Cannot infer Method/Model column.")

    return strategy_col, method_col


def infer_group_column(df: pd.DataFrame) -> Optional[str]:
    return infer_column(df, [
        "文献编号", "Literature_ID", "Paper_ID", "Paper",
        "Source_Group", "Group", "Source", "Literature",
    ])


def infer_role_column(df: pd.DataFrame) -> Optional[str]:
    return infer_column(df, [
        "Split_Role", "Set", "Subset", "Data_Split", "Role",
        "Train_Valid_Test", "Dataset_Role",
    ])


def normalize_role_value(x: Any) -> str:
    s = str(x).strip().lower()
    if any(t in s for t in ["valid", "val", "calib", "calibration"]):
        return "validation"
    if any(t in s for t in ["test", "held", "holdout"]):
        return "test"
    if "train" in s:
        return "train"
    return "unknown"


def infer_block_columns(df: pd.DataFrame, group_col: Optional[str]) -> List[str]:
    block_candidates = [
        "Seed", "Random_Seed", "seed", "Fold", "fold", "Split",
        "Split_ID", "split_id", "Outer_Fold", "OuterFold", "Repeat",
        "repeat", "Run", "run",
    ]
    found = [c for c in block_candidates if c in df.columns]
    if not found and group_col is not None:
        found = [group_col]
    if not found:
        df["_Single_Block"] = "block_0"
        found = ["_Single_Block"]
    return found


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    n = len(y_true)
    if n == 0:
        return {
            "n": 0, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
            "MSE": np.nan, "Bias": np.nan, "Target_SD": np.nan,
            "Target_Range": np.nan,
        }

    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    sse = float(np.sum((y_true - y_pred) ** 2))
    r2 = np.nan if sst <= 1e-12 else float(1.0 - sse / sst)

    return {
        "n": int(n),
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "Bias": bias,
        "Target_SD": float(np.std(y_true, ddof=1)) if n > 1 else 0.0,
        "Target_Range": float(np.max(y_true) - np.min(y_true)) if n > 0 else np.nan,
    }


def load_row_level_prediction_table() -> pd.DataFrame:
    candidates = [
        TABLE_DIR / "32b_predictions_with_reliability_zone.csv",
        TABLE_DIR / "32a2_ad_predictions_with_scores.csv",
        TABLE_DIR / "32a_ad_predictions_with_scores.csv",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(
            "Cannot find row-level AD prediction table. Please make sure one file exists:\n"
            "outputs/tables/32b_predictions_with_reliability_zone.csv\n"
            "outputs/tables/32a2_ad_predictions_with_scores.csv\n"
            "outputs/tables/32a_ad_predictions_with_scores.csv"
        )

    path = existing[0]
    df = pd.read_csv(path)
    log(f"[INFO] Loaded: {path} | shape={df.shape}")
    return df


def get_selected_methods(df: pd.DataFrame, method_col: str) -> List[str]:
    present = list(pd.Series(df[method_col].astype(str).unique()).dropna())
    selected = [m for m in PREFERRED_METHODS if m in present]

    if len(selected) < 3:
        counts = df[method_col].astype(str).value_counts()
        for m in counts.index:
            if m not in selected:
                selected.append(m)
            if len(selected) >= MAX_METHODS_FOR_DYNAMIC_SWITCH:
                break

    return selected[:MAX_METHODS_FOR_DYNAMIC_SWITCH]


def add_block_id(df: pd.DataFrame, strategy_col: str, block_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    block = out[block_cols[0]].astype(str)
    for c in block_cols[1:]:
        block = block + "__" + out[c].astype(str)
    out["_Block_ID"] = out[strategy_col].astype(str) + "__" + block
    return out


def infer_sample_key_columns(
    df: pd.DataFrame,
    strategy_col: str,
    method_col: str,
    group_col: Optional[str],
    block_cols: List[str],
) -> List[str]:
    sample_candidates = [
        "_Data_Index", "Data_Index", "Sample_Index", "Sample_ID", "Row_ID",
        "Original_Row_Index", "原始行号", "Original_Index", "Index", "idx",
    ]

    key_cols = [strategy_col]
    for c in block_cols:
        if c not in key_cols:
            key_cols.append(c)
    if group_col is not None and group_col not in key_cols:
        key_cols.append(group_col)

    found_sample = False
    for c in sample_candidates:
        if c in df.columns and c not in key_cols:
            key_cols.append(c)
            found_sample = True

    if not found_sample:
        df["_Row_Order_Sample_Key"] = np.arange(len(df))
        key_cols.append("_Row_Order_Sample_Key")

    return key_cols


def add_model_disagreement_features(
    df: pd.DataFrame,
    key_cols: List[str],
    method_col: str,
    pred_col: str,
) -> pd.DataFrame:
    out = df.copy()
    key_cols = [c for c in key_cols if c in out.columns and c != method_col]

    try:
        tmp = out[key_cols + [method_col, pred_col]].copy()
        tmp[pred_col] = clean_numeric(tmp[pred_col])
        tmp[method_col] = tmp[method_col].astype(str)

        grouped = tmp.groupby(key_cols + [method_col], dropna=False)[pred_col].mean().reset_index()
        wide = grouped.pivot_table(index=key_cols, columns=method_col, values=pred_col, aggfunc="mean")

        wide["_Pred_Disagreement_STD"] = wide.std(axis=1, skipna=True)
        wide["_Pred_Disagreement_Range"] = wide.max(axis=1, skipna=True) - wide.min(axis=1, skipna=True)
        wide["_Pred_Disagreement_IQR"] = wide.quantile(0.75, axis=1) - wide.quantile(0.25, axis=1)
        wide["_Pred_Ensemble_Mean"] = wide.mean(axis=1, skipna=True)
        wide["_Pred_Ensemble_Median"] = wide.median(axis=1, skipna=True)
        wide["_Pred_N_Methods"] = wide.notna().sum(axis=1)

        dis_cols = [
            "_Pred_Disagreement_STD", "_Pred_Disagreement_Range", "_Pred_Disagreement_IQR",
            "_Pred_Ensemble_Mean", "_Pred_Ensemble_Median", "_Pred_N_Methods",
        ]

        wide = wide[dis_cols].reset_index()
        out = out.merge(wide, on=key_cols, how="left")

        out["_Pred_Distance_To_Ensemble_Median"] = (
            clean_numeric(out[pred_col]) - clean_numeric(out["_Pred_Ensemble_Median"])
        ).abs()
        out["_Pred_Distance_To_50"] = (clean_numeric(out[pred_col]) - 50.0).abs()
        out["_Pred_Distance_To_Bounds"] = np.minimum(
            clean_numeric(out[pred_col]).abs(),
            (100.0 - clean_numeric(out[pred_col])).abs(),
        )
        log("[INFO] Added prediction disagreement features.")
    except Exception as e:
        log(f"[WARNING] Failed to add model-disagreement features: {e}")

    return out


def select_policy_feature_columns(df: pd.DataFrame, y_col: str, pred_col: str, strategy_col: str, method_col: str) -> List[str]:
    forbidden_exact = {
        y_col, "_Abs_Error", "_Signed_Error", "Abs_Error", "Absolute_Error",
        "Error", "Residual", "Squared_Error", "RMSE", "MAE", "R2", "MSE",
        "Observed", "Actual", "y_true", "target", "Target",
    }
    forbidden_tokens = [
        "abs_error", "absolute_error", "signed_error", "residual",
        "paper_rmse", "paper_mae", "paper_r2", "observed", "actual", "true",
    ]
    allowed_tokens = [
        "risk", "score", "fraction", "violation", "unseen", "boundary",
        "distance", "leverage", "mahalanobis", "reconstruction", "nn_",
        "pca", "pred_", "prediction", "disagreement", "novelty", "coverage",
    ]

    cols: List[str] = []
    for c in df.columns:
        if c in forbidden_exact or c in [strategy_col, method_col]:
            continue
        lc = c.lower()
        if any(t in lc for t in forbidden_tokens):
            continue
        if c == pred_col or any(t in lc for t in allowed_tokens):
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().mean() >= 0.30 and vals.nunique(dropna=True) > 1:
                cols.append(c)

    for c in [
        pred_col, "_Pred_Disagreement_STD", "_Pred_Disagreement_Range",
        "_Pred_Disagreement_IQR", "_Pred_Distance_To_Ensemble_Median",
        "_Pred_Distance_To_50", "_Pred_Distance_To_Bounds", "_Pred_N_Methods",
    ]:
        if c in df.columns and c not in cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().mean() >= 0.30 and vals.nunique(dropna=True) > 1:
                cols.append(c)

    return cols


def select_score_columns_for_rules(policy_features: List[str], pred_col: str) -> List[str]:
    cols = []
    for c in policy_features:
        lc = c.lower()
        if c == pred_col or any(t in lc for t in [
            "risk", "score", "fraction", "violation", "unseen",
            "boundary", "distance", "leverage", "mahalanobis",
            "reconstruction", "disagreement", "pred_distance",
        ]):
            cols.append(c)

    priority = []
    priority_tokens = [
        "corrected_ad_risk", "boundary_ad_risk", "distance_ad_risk",
        "categorical_unseen", "numeric_strict", "range_violation",
        "mahalanobis", "nn_risk", "pca_reconstruction", "pred_disagreement",
        "pred_distance",
    ]

    for tok in priority_tokens:
        for c in cols:
            if tok in c.lower() and c not in priority:
                priority.append(c)

    for c in cols:
        if c not in priority:
            priority.append(c)

    return priority[:18]


@dataclass
class RuleSpec:
    name: str
    kind: str
    col1: Optional[str] = None
    lo1: Optional[float] = None
    hi1: Optional[float] = None
    col2: Optional[str] = None
    lo2: Optional[float] = None
    hi2: Optional[float] = None


def mask_from_rule(df: pd.DataFrame, rule: RuleSpec) -> np.ndarray:
    if rule.kind == "all":
        return np.ones(len(df), dtype=bool)
    if rule.col1 is None or rule.col1 not in df.columns:
        return np.zeros(len(df), dtype=bool)

    x1 = pd.to_numeric(df[rule.col1], errors="coerce")
    mask = np.isfinite(x1.to_numpy(dtype=float))

    if rule.lo1 is not None:
        mask &= (x1 >= rule.lo1).to_numpy()
    if rule.hi1 is not None:
        mask &= (x1 <= rule.hi1).to_numpy()

    if rule.kind in ["two_col_le", "window_and_le"]:
        if rule.col2 is None or rule.col2 not in df.columns:
            return np.zeros(len(df), dtype=bool)
        x2 = pd.to_numeric(df[rule.col2], errors="coerce")
        mask &= np.isfinite(x2.to_numpy(dtype=float))
        if rule.lo2 is not None:
            mask &= (x2 >= rule.lo2).to_numpy()
        if rule.hi2 is not None:
            mask &= (x2 <= rule.hi2).to_numpy()

    return np.asarray(mask, dtype=bool)


def qvalue(s: pd.Series, q: float) -> Optional[float]:
    arr = finite_array(s)
    if len(arr) == 0:
        return None
    return float(np.quantile(arr, q))


def generate_rule_candidates(calib_df: pd.DataFrame, score_cols: List[str]) -> List[RuleSpec]:
    rules: List[RuleSpec] = [RuleSpec(name="AllDomain_NoFiltering", kind="all")]

    for col in score_cols:
        if col not in calib_df.columns:
            continue
        vals = pd.to_numeric(calib_df[col], errors="coerce")
        if vals.notna().sum() < MIN_N_ABSOLUTE or vals.nunique(dropna=True) <= 3:
            continue

        for q in QUANTILE_GRID_FINE:
            v = qvalue(vals, q)
            if v is None:
                continue
            rules.append(RuleSpec(name=f"{col} <= q{q:.2f}", kind="le", col1=col, hi1=v))
            rules.append(RuleSpec(name=f"{col} >= q{q:.2f}", kind="ge", col1=col, lo1=v))

        for ql in WINDOW_Q_LOW:
            lo = qvalue(vals, ql)
            if lo is None:
                continue
            for qh in WINDOW_Q_HIGH:
                if qh <= ql:
                    continue
                hi = qvalue(vals, qh)
                if hi is None or hi <= lo:
                    continue
                rules.append(RuleSpec(
                    name=f"{col} window q{ql:.2f}-q{qh:.2f}",
                    kind="window", col1=col, lo1=lo, hi1=hi,
                ))

    priority_cols: List[str] = []
    for token in [
        "Corrected_AD_Risk", "Boundary_AD_Risk", "Distance_AD_Risk",
        "Categorical_Unseen_Fraction", "Numeric_Strict_Range_Violation_Fraction",
        "_Pred_Disagreement_STD", "_Pred_Disagreement_Range",
        "_Pred_Distance_To_Ensemble_Median",
    ]:
        for c in score_cols:
            if token.lower() in c.lower() and c not in priority_cols:
                priority_cols.append(c)

    priority_cols = priority_cols[:8]

    for i, c1 in enumerate(priority_cols):
        for c2 in priority_cols[i + 1:]:
            s1 = pd.to_numeric(calib_df[c1], errors="coerce")
            s2 = pd.to_numeric(calib_df[c2], errors="coerce")
            if s1.notna().sum() < MIN_N_ABSOLUTE or s2.notna().sum() < MIN_N_ABSOLUTE:
                continue

            for q1 in [0.35, 0.50, 0.60, 0.70, 0.80]:
                v1 = qvalue(s1, q1)
                if v1 is None:
                    continue
                for q2 in [0.35, 0.50, 0.60, 0.70, 0.80]:
                    v2 = qvalue(s2, q2)
                    if v2 is None:
                        continue
                    rules.append(RuleSpec(
                        name=f"{c1}<=q{q1:.2f} AND {c2}<=q{q2:.2f}",
                        kind="two_col_le", col1=c1, hi1=v1, col2=c2, hi2=v2,
                    ))

    return rules


def evaluate_rule_candidates(
    df: pd.DataFrame,
    rules: List[RuleSpec],
    y_col: str,
    pred_col: str,
    min_coverage: float,
    min_n: int,
    objective: str = "R2_first",
) -> Tuple[Optional[RuleSpec], Dict[str, float], pd.DataFrame]:
    rows = []
    for rule in rules:
        mask = mask_from_rule(df, rule)
        coverage = float(mask.mean()) if len(mask) else 0.0
        n = int(mask.sum())
        if n < min_n or coverage < min_coverage:
            continue
        met = metric_dict(df.loc[mask, y_col].to_numpy(), df.loc[mask, pred_col].to_numpy())
        rows.append({
            "Rule_Name": rule.name,
            "Rule_Kind": rule.kind,
            "Coverage": coverage,
            **met,
        })

    cand = pd.DataFrame(rows)
    if cand.empty:
        return None, {}, cand

    if objective == "R2_first":
        cand = cand.sort_values(["R2", "RMSE", "Coverage"], ascending=[False, True, False])
    else:
        cand = cand.sort_values(["RMSE", "R2", "Coverage"], ascending=[True, False, False])

    best = cand.iloc[0].to_dict()
    best_rule = next((r for r in rules if r.name == best["Rule_Name"]), None)
    return best_rule, best, cand


def make_modeling_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    method_col: Optional[str] = None,
    fit_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    X = pd.DataFrame(index=df.index)

    for c in feature_cols:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce")

    if method_col is not None and method_col in df.columns:
        dummies = pd.get_dummies(df[method_col].astype(str), prefix="Method")
        X = pd.concat([X, dummies], axis=1)

    X = X.replace([np.inf, -np.inf], np.nan)

    if fit_columns is not None:
        for c in fit_columns:
            if c not in X.columns:
                X[c] = 0.0
        X = X[fit_columns]
        return X, fit_columns

    return X, list(X.columns)


def build_risk_model(model_type: str) -> Any:
    if model_type == "hgb":
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.045,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            )),
        ])

    if model_type == "rf":
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=10,
                max_features=0.70,
                bootstrap=True,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ])

    if model_type == "ridge":
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ])

    raise ValueError(f"Unknown risk model type: {model_type}")


def select_best_retain_fraction_on_calib(
    calib_df: pd.DataFrame,
    risk_col: str,
    y_col: str,
    pred_col: str,
    min_coverage: float,
    min_n: int,
) -> Tuple[Optional[float], Dict[str, float], pd.DataFrame]:
    rows = []
    vals = finite_array(calib_df[risk_col])
    if len(vals) < min_n:
        return None, {}, pd.DataFrame()

    for frac in RISK_RETAIN_FRACTIONS:
        if frac < min_coverage:
            continue
        cutoff = float(np.quantile(vals, frac))
        mask = pd.to_numeric(calib_df[risk_col], errors="coerce") <= cutoff
        coverage = float(mask.mean())
        n = int(mask.sum())
        if n < min_n or coverage < min_coverage:
            continue
        met = metric_dict(calib_df.loc[mask, y_col].to_numpy(), calib_df.loc[mask, pred_col].to_numpy())
        rows.append({"Retain_Fraction": frac, "Risk_Cutoff": cutoff, "Coverage": coverage, **met})

    cand = pd.DataFrame(rows)
    if cand.empty:
        return None, {}, cand

    cand = cand.sort_values(["R2", "RMSE", "Coverage"], ascending=[False, True, False])
    best = cand.iloc[0].to_dict()
    return float(best["Risk_Cutoff"]), best, cand


def choose_dynamic_predictions_by_risk(df: pd.DataFrame, sample_key_cols: List[str], risk_col: str) -> pd.DataFrame:
    key_cols = [c for c in sample_key_cols if c in df.columns]
    tmp = df.copy()
    tmp[risk_col] = pd.to_numeric(tmp[risk_col], errors="coerce")
    tmp = tmp.dropna(subset=[risk_col])
    if tmp.empty or not key_cols:
        return tmp
    idx = tmp.groupby(key_cols, dropna=False)[risk_col].idxmin()
    return tmp.loc[idx].copy()


def run_calibrated_rule_search(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    strategy_col: str,
    method_col: str,
    score_cols: List[str],
    selected_methods: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    detail_rows = []

    for strategy, sdf in df.groupby(strategy_col, dropna=False):
        sdf = sdf.copy()
        blocks = list(pd.Series(sdf["_Block_ID"].unique()).dropna())
        if len(blocks) < 2:
            log(f"[RULE] {strategy}: only one block; skip calibrated rule search.")
            continue

        for method in selected_methods:
            mdf = sdf[sdf[method_col].astype(str) == str(method)].copy()
            if len(mdf) < MIN_N_ABSOLUTE * 2:
                continue

            for min_cov in MIN_COVERAGE_GRID:
                for block in blocks:
                    eval_df = mdf[mdf["_Block_ID"] == block].copy()
                    calib_df = mdf[mdf["_Block_ID"] != block].copy()

                    if len(eval_df) < 5 or len(calib_df) < MIN_N_ABSOLUTE:
                        continue

                    rules = generate_rule_candidates(calib_df, score_cols)
                    best_rule, best_calib, _ = evaluate_rule_candidates(
                        calib_df, rules, y_col, pred_col,
                        min_coverage=min_cov, min_n=MIN_N_ABSOLUTE,
                    )
                    if best_rule is None:
                        continue

                    mask_eval = mask_from_rule(eval_df, best_rule)
                    if mask_eval.sum() == 0:
                        continue

                    kept = eval_df.loc[mask_eval].copy()
                    kept["_32C_Policy"] = "Calibrated_AD_Rule"
                    kept["_32C_MinCoverage"] = min_cov
                    kept["_32C_SelectedRule"] = best_rule.name
                    kept["_32C_Calibration_R2"] = best_calib.get("R2", np.nan)
                    kept["_32C_Calibration_RMSE"] = best_calib.get("RMSE", np.nan)
                    kept["_32C_Calibration_Coverage"] = best_calib.get("Coverage", np.nan)
                    prediction_rows.append(kept)

                    detail_rows.append({
                        "Strategy": strategy,
                        "Method": method,
                        "Block_ID": block,
                        "Policy": "Calibrated_AD_Rule",
                        "MinCoverage": min_cov,
                        "SelectedRule": best_rule.name,
                        "Calibration_R2": best_calib.get("R2", np.nan),
                        "Calibration_RMSE": best_calib.get("RMSE", np.nan),
                        "Calibration_Coverage": best_calib.get("Coverage", np.nan),
                        "Eval_n_kept": int(mask_eval.sum()),
                        "Eval_n_total": int(len(eval_df)),
                        "Eval_coverage": float(mask_eval.mean()),
                    })

    pred = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    details = pd.DataFrame(detail_rows)
    return pred, details


def run_calibrated_residual_risk_selection(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    strategy_col: str,
    method_col: str,
    feature_cols: List[str],
    selected_methods: List[str],
    sample_key_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    detail_rows = []
    risk_model_types = ["hgb", "rf", "ridge"]

    for strategy, sdf0 in df.groupby(strategy_col, dropna=False):
        sdf0 = sdf0.copy()
        sdf0 = sdf0[sdf0[method_col].astype(str).isin(selected_methods)].copy()
        blocks = list(pd.Series(sdf0["_Block_ID"].unique()).dropna())
        if len(blocks) < 2:
            log(f"[RISK] {strategy}: only one block; skip calibrated risk search.")
            continue

        log(f"[RISK] Strategy={strategy} | blocks={len(blocks)} | rows={len(sdf0)}")

        for block in blocks:
            eval_all = sdf0[sdf0["_Block_ID"] == block].copy()
            calib_all = sdf0[sdf0["_Block_ID"] != block].copy()

            if len(eval_all) < 10 or len(calib_all) < MIN_N_ABSOLUTE:
                continue

            y_abs = pd.to_numeric(calib_all["_Abs_Error"], errors="coerce")
            valid = y_abs.notna()
            calib_all = calib_all.loc[valid].copy()
            y_abs = y_abs.loc[valid]

            if len(calib_all) < MIN_N_ABSOLUTE:
                continue

            for risk_type in risk_model_types:
                try:
                    X_calib, fit_cols = make_modeling_matrix(calib_all, feature_cols, method_col)
                    X_eval, _ = make_modeling_matrix(eval_all, feature_cols, method_col, fit_columns=fit_cols)
                    if X_calib.shape[1] == 0:
                        continue

                    model = build_risk_model(risk_type)
                    model.fit(X_calib, y_abs.to_numpy())

                    calib_scored = calib_all.copy()
                    eval_scored = eval_all.copy()
                    calib_scored["_32C_Predicted_AbsError_Risk"] = model.predict(X_calib)
                    eval_scored["_32C_Predicted_AbsError_Risk"] = model.predict(X_eval)

                    # Fixed method reliability filter.
                    for method in selected_methods:
                        calib_m = calib_scored[calib_scored[method_col].astype(str) == str(method)].copy()
                        eval_m = eval_scored[eval_scored[method_col].astype(str) == str(method)].copy()

                        if len(calib_m) < MIN_N_ABSOLUTE or len(eval_m) < 5:
                            continue

                        for min_cov in MIN_COVERAGE_GRID:
                            cutoff, best_calib, _ = select_best_retain_fraction_on_calib(
                                calib_m, "_32C_Predicted_AbsError_Risk",
                                y_col, pred_col, min_coverage=min_cov, min_n=MIN_N_ABSOLUTE,
                            )
                            if cutoff is None:
                                continue

                            mask_eval = pd.to_numeric(eval_m["_32C_Predicted_AbsError_Risk"], errors="coerce") <= cutoff
                            if mask_eval.sum() == 0:
                                continue

                            kept = eval_m.loc[mask_eval].copy()
                            kept["_32C_Policy"] = f"ResidualRisk_{risk_type}_FixedMethod"
                            kept["_32C_MinCoverage"] = min_cov
                            kept["_32C_SelectedRule"] = f"PredictedAbsError <= {cutoff:.6g}"
                            kept["_32C_Calibration_R2"] = best_calib.get("R2", np.nan)
                            kept["_32C_Calibration_RMSE"] = best_calib.get("RMSE", np.nan)
                            kept["_32C_Calibration_Coverage"] = best_calib.get("Coverage", np.nan)
                            prediction_rows.append(kept)

                            detail_rows.append({
                                "Strategy": strategy,
                                "Method": method,
                                "Block_ID": block,
                                "Policy": f"ResidualRisk_{risk_type}_FixedMethod",
                                "MinCoverage": min_cov,
                                "SelectedRule": f"PredictedAbsError <= {cutoff:.6g}",
                                "Calibration_R2": best_calib.get("R2", np.nan),
                                "Calibration_RMSE": best_calib.get("RMSE", np.nan),
                                "Calibration_Coverage": best_calib.get("Coverage", np.nan),
                                "Eval_n_kept": int(mask_eval.sum()),
                                "Eval_n_total": int(len(eval_m)),
                                "Eval_coverage": float(mask_eval.mean()),
                            })

                    # Dynamic model choice by predicted residual risk.
                    calib_dyn = choose_dynamic_predictions_by_risk(
                        calib_scored, sample_key_cols, "_32C_Predicted_AbsError_Risk"
                    )
                    eval_dyn = choose_dynamic_predictions_by_risk(
                        eval_scored, sample_key_cols, "_32C_Predicted_AbsError_Risk"
                    )

                    if len(calib_dyn) >= MIN_N_ABSOLUTE and len(eval_dyn) >= 5:
                        dynamic_name = f"DynamicSwitch_{risk_type}"
                        for min_cov in MIN_COVERAGE_GRID:
                            cutoff, best_calib, _ = select_best_retain_fraction_on_calib(
                                calib_dyn, "_32C_Predicted_AbsError_Risk",
                                y_col, pred_col, min_coverage=min_cov, min_n=MIN_N_ABSOLUTE,
                            )
                            if cutoff is None:
                                continue

                            mask_eval = pd.to_numeric(eval_dyn["_32C_Predicted_AbsError_Risk"], errors="coerce") <= cutoff
                            if mask_eval.sum() == 0:
                                continue

                            kept = eval_dyn.loc[mask_eval].copy()
                            kept["_32C_Policy"] = f"ResidualRisk_{risk_type}_DynamicMethodSwitch"
                            kept["_32C_MinCoverage"] = min_cov
                            kept["_32C_SelectedRule"] = f"PredictedAbsError <= {cutoff:.6g}"
                            kept["_32C_Calibration_R2"] = best_calib.get("R2", np.nan)
                            kept["_32C_Calibration_RMSE"] = best_calib.get("RMSE", np.nan)
                            kept["_32C_Calibration_Coverage"] = best_calib.get("Coverage", np.nan)
                            kept["_32C_Dynamic_Selected_Method"] = kept[method_col].astype(str)
                            kept[method_col] = dynamic_name
                            prediction_rows.append(kept)

                            detail_rows.append({
                                "Strategy": strategy,
                                "Method": dynamic_name,
                                "Block_ID": block,
                                "Policy": f"ResidualRisk_{risk_type}_DynamicMethodSwitch",
                                "MinCoverage": min_cov,
                                "SelectedRule": f"PredictedAbsError <= {cutoff:.6g}",
                                "Calibration_R2": best_calib.get("R2", np.nan),
                                "Calibration_RMSE": best_calib.get("RMSE", np.nan),
                                "Calibration_Coverage": best_calib.get("Coverage", np.nan),
                                "Eval_n_kept": int(mask_eval.sum()),
                                "Eval_n_total": int(len(eval_dyn)),
                                "Eval_coverage": float(mask_eval.mean()),
                            })

                except Exception as e:
                    log(f"[WARNING] Risk model failed | strategy={strategy} | block={block} | type={risk_type}: {e}")

    pred = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    details = pd.DataFrame(detail_rows)
    return pred, details


def summarize_selective_predictions(
    pred_df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    strategy_col: str,
    method_col: str,
) -> pd.DataFrame:
    rows = []
    if pred_df.empty:
        return pd.DataFrame()

    group_cols = [strategy_col, method_col, "_32C_Policy", "_32C_MinCoverage"]
    for keys, g in pred_df.groupby(group_cols, dropna=False):
        strategy, method, policy, mincov = keys
        met = metric_dict(g[y_col].to_numpy(), g[pred_col].to_numpy())
        rows.append({
            "Strategy": strategy,
            "Method": method,
            "Policy": policy,
            "MinCoverage_Setting": mincov,
            "n_retained_rows": met["n"],
            "R2": met["R2"],
            "RMSE": met["RMSE"],
            "MAE": met["MAE"],
            "MSE": met["MSE"],
            "Bias": met["Bias"],
            "Target_SD": met["Target_SD"],
            "Target_Range": met["Target_Range"],
            "Hit_Target_R2_0p5": bool(np.isfinite(met["R2"]) and met["R2"] >= TARGET_R2),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["R2", "RMSE", "n_retained_rows"], ascending=[False, True, False])
    return out


def add_all_domain_baselines(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    strategy_col: str,
    method_col: str,
    selected_methods: List[str],
) -> pd.DataFrame:
    rows = []
    for (strategy, method), g in df[df[method_col].astype(str).isin(selected_methods)].groupby([strategy_col, method_col]):
        met = metric_dict(g[y_col].to_numpy(), g[pred_col].to_numpy())
        rows.append({
            "Strategy": strategy,
            "Method": method,
            "Policy": "AllDomain_NoFiltering",
            "MinCoverage_Setting": 1.0,
            "n_retained_rows": met["n"],
            "R2": met["R2"],
            "RMSE": met["RMSE"],
            "MAE": met["MAE"],
            "MSE": met["MSE"],
            "Bias": met["Bias"],
            "Target_SD": met["Target_SD"],
            "Target_Range": met["Target_Range"],
            "Hit_Target_R2_0p5": bool(np.isfinite(met["R2"]) and met["R2"] >= TARGET_R2),
        })
    return pd.DataFrame(rows)


def run_oracle_upper_bound(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    strategy_col: str,
    method_col: str,
    score_cols: List[str],
    selected_methods: List[str],
) -> pd.DataFrame:
    rows = []
    for (strategy, method), g in df[df[method_col].astype(str).isin(selected_methods)].groupby([strategy_col, method_col]):
        if len(g) < MIN_N_ABSOLUTE:
            continue

        rules = generate_rule_candidates(g, score_cols)
        for min_cov in MIN_COVERAGE_GRID:
            best_rule, best, _ = evaluate_rule_candidates(
                g, rules, y_col, pred_col,
                min_coverage=min_cov, min_n=MIN_N_ABSOLUTE,
            )
            if best_rule is None:
                continue

            rows.append({
                "Strategy": strategy,
                "Method": method,
                "MinCoverage": min_cov,
                "BestRule": best_rule.name,
                "DiagnosticOnly_R2": best.get("R2", np.nan),
                "DiagnosticOnly_RMSE": best.get("RMSE", np.nan),
                "DiagnosticOnly_MAE": best.get("MAE", np.nan),
                "DiagnosticOnly_Coverage": best.get("Coverage", np.nan),
                "DiagnosticOnly_n": best.get("n", np.nan),
                "Hit_Target_R2_0p5": bool(np.isfinite(best.get("R2", np.nan)) and best.get("R2", np.nan) >= TARGET_R2),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["DiagnosticOnly_R2", "DiagnosticOnly_RMSE"], ascending=[False, True])
    return out


def plot_top_metrics(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    top = summary[summary["Policy"] != "AllDomain_NoFiltering"].copy().head(15)
    if top.empty:
        top = summary.head(15)

    labels = [
        f"{r.Method}\n{r.Policy.replace('ResidualRisk_', 'Risk_')}\nminCov={r.MinCoverage_Setting}"
        for r in top.itertuples(index=False)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    for ax, col in zip(axes, ["R2", "RMSE", "MAE"]):
        vals = top[col].to_numpy(dtype=float)
        x = np.arange(len(vals))
        bars = ax.bar(x, vals, edgecolor="black", linewidth=0.6)
        ax.set_title(col, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)

        if col == "R2":
            ax.axhline(TARGET_R2, color="red", linestyle="--", linewidth=1.2)
            ax.text(0.02, TARGET_R2 + 0.01, "target R2=0.50", color="red", transform=ax.get_yaxis_transform())

        for b, v in zip(bars, vals):
            if np.isfinite(v):
                label = f"{v:.3f}" if col == "R2" else f"{v:.2f}"
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), label,
                        ha="center", va="bottom", fontsize=7, rotation=90)

    fig.suptitle("32C selective AD policy: top retained-domain metrics", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    save_fig(fig, "158_32c_selective_policy_metrics.png")


def plot_r2_coverage_frontier(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    dat = summary[summary["Policy"] != "AllDomain_NoFiltering"].copy()
    if dat.empty:
        return

    baseline = summary[summary["Policy"] == "AllDomain_NoFiltering"][
        ["Strategy", "Method", "n_retained_rows"]
    ].rename(columns={"n_retained_rows": "_AllDomain_n"})

    dat = dat.merge(baseline, on=["Strategy", "Method"], how="left")
    dat["Approx_Coverage"] = dat["n_retained_rows"] / dat["_AllDomain_n"]
    dat["Approx_Coverage"] = dat["Approx_Coverage"].clip(0, 1)

    fig, ax = plt.subplots(figsize=(9, 6.2))
    for policy, g in dat.groupby("Policy"):
        ax.scatter(g["Approx_Coverage"], g["R2"], s=50, alpha=0.75, label=policy)

    ax.axhline(TARGET_R2, color="red", linestyle="--", linewidth=1.2, label="target R2=0.50")
    ax.set_xlabel("Approximate retained coverage")
    ax.set_ylabel("Pooled R2 in retained domain")
    ax.set_title("R2 vs retained coverage under selective AD policies", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_fig(fig, "159_32c_r2_coverage_frontier.png")


def plot_best_predicted_vs_observed(
    pred_df: pd.DataFrame,
    summary: pd.DataFrame,
    y_col: str,
    pred_col: str,
    strategy_col: str,
    method_col: str,
) -> None:
    if pred_df.empty or summary.empty:
        return

    best_candidates = summary[summary["Policy"] != "AllDomain_NoFiltering"].copy()
    if best_candidates.empty:
        return

    best = best_candidates.sort_values(["R2", "RMSE", "n_retained_rows"], ascending=[False, True, False]).iloc[0]

    mask = (
        (pred_df[strategy_col].astype(str) == str(best["Strategy"])) &
        (pred_df[method_col].astype(str) == str(best["Method"])) &
        (pred_df["_32C_Policy"].astype(str) == str(best["Policy"])) &
        (pd.to_numeric(pred_df["_32C_MinCoverage"], errors="coerce") == float(best["MinCoverage_Setting"]))
    )
    g = pred_df.loc[mask].copy()
    if g.empty:
        return

    met = metric_dict(g[y_col].to_numpy(), g[pred_col].to_numpy())

    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    ax.scatter(g[y_col], g[pred_col], s=22, alpha=0.65, edgecolor="none")
    ax.plot([0, 100], [0, 100], color="red", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Predicted F removal (%)")
    ax.set_title(
        "Best 32C retained-domain predicted vs observed\n"
        f"{best['Strategy']} | {best['Method']} | {best['Policy']}",
        fontsize=12,
        fontweight="bold",
    )
    text = f"R2 = {met['R2']:.3f}\nRMSE = {met['RMSE']:.2f}\nMAE = {met['MAE']:.2f}\nn = {met['n']}"
    ax.text(
        0.04, 0.96, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.5"),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "160_32c_best_selective_predicted_vs_observed.png")


def plot_dynamic_method_frequency(pred_df: pd.DataFrame) -> None:
    if pred_df.empty:
        return

    if "_32C_Dynamic_Selected_Method" not in pred_df.columns:
        return

    dyn = pred_df[pred_df["_32C_Policy"].astype(str).str.contains("DynamicMethodSwitch", na=False)].copy()
    if dyn.empty:
        return

    counts = dyn["_32C_Dynamic_Selected_Method"].astype(str).value_counts().head(12)
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.barh(np.arange(len(counts)), counts.values, edgecolor="black", linewidth=0.6)
    ax.set_yticks(np.arange(len(counts)))
    ax.set_yticklabels(counts.index)
    ax.invert_yaxis()
    ax.set_xlabel("Selection count")
    ax.set_title("Dynamic method-switching frequency in retained samples", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)

    for b, v in zip(bars, counts.values):
        ax.text(v, b.get_y() + b.get_height() / 2, f"{int(v)}", va="center", ha="left", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "161_32c_dynamic_method_selection_frequency.png")


def write_report(
    summary: pd.DataFrame,
    oracle: pd.DataFrame,
    details: pd.DataFrame,
    selected_methods: List[str],
    feature_cols: List[str],
    score_cols: List[str],
) -> Path:
    path = TABLE_DIR / "32c_selective_ad_decision_report.txt"
    lines: List[str] = []

    lines.append("=" * 100)
    lines.append("Step 32C selective AD-based R2 boosting report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Search whether a high-R2 retained reliable subdomain can be obtained by selective prediction.")
    lines.append("")
    lines.append("Strictness:")
    lines.append("- Serious calibrated policies are selected on calibration/grouped blocks and evaluated on held-out blocks.")
    lines.append("- Held-out/test residuals are not used to train residual-risk models for the same held-out block.")
    lines.append("- Diagnostic oracle upper bound is diagnostic only, not deployable.")
    lines.append("")
    lines.append(f"Requested target: retained-domain R2 >= {TARGET_R2:.2f}")
    lines.append("")
    lines.append("Selected methods:")
    for m in selected_methods:
        lines.append(f"  - {m}")
    lines.append("")
    lines.append(f"Policy feature count: {len(feature_cols)}")
    for c in feature_cols[:40]:
        lines.append(f"  - {c}")
    lines.append("")
    lines.append(f"Direct rule score count: {len(score_cols)}")
    for c in score_cols:
        lines.append(f"  - {c}")
    lines.append("")

    if summary.empty:
        lines.append("No calibrated selective-policy result was generated.")
    else:
        non_baseline = summary[summary["Policy"] != "AllDomain_NoFiltering"].copy()
        lines.append("Top calibrated/selective results:")
        lines.append(non_baseline.head(20).to_string(index=False))
        lines.append("")

        hit = non_baseline[non_baseline["R2"] >= TARGET_R2]
        if not hit.empty:
            lines.append("TARGET HIT:")
            lines.append(hit.head(20).to_string(index=False))
        else:
            lines.append("TARGET NOT HIT:")
            lines.append("No calibrated selective policy reached R2 >= 0.50.")
        lines.append("")

    if not oracle.empty:
        lines.append("Diagnostic oracle upper bound:")
        lines.append("This part is diagnostic only because it selects rules using same-table labels.")
        lines.append(oracle.head(20).to_string(index=False))
        lines.append("")
        if (oracle["DiagnosticOnly_R2"] >= TARGET_R2).any():
            lines.append("Oracle upper bound reached R2 >= 0.50 in at least one score-defined subset.")
        else:
            lines.append("Oracle upper bound did not reach R2 >= 0.50 with the current score-defined subset search.")
        lines.append("")

    if not details.empty:
        lines.append("Calibration details, first 40 rows:")
        lines.append(details.head(40).to_string(index=False))
        lines.append("")

    elapsed = time.time() - START_TIME
    lines.append(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    lines.append("=" * 100)

    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[REPORT] {path}")
    return path


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    log("=" * 100)
    log(f"Running {SCRIPT_NAME}")
    log(f"Project root: {PROJECT_ROOT}")
    log("=" * 100)

    df0 = load_row_level_prediction_table()

    y_col, pred_col = infer_true_pred_columns(df0)
    strategy_col, method_col = infer_strategy_method_columns(df0)
    group_col = infer_group_column(df0)
    role_col = infer_role_column(df0)

    log(f"[INFO] Observed column: {y_col}")
    log(f"[INFO] Predicted column: {pred_col}")
    log(f"[INFO] Strategy column: {strategy_col}")
    log(f"[INFO] Method column: {method_col}")
    log(f"[INFO] Literature/group column: {group_col}")
    log(f"[INFO] Role column: {role_col}")

    df = df0.copy()
    df[y_col] = clean_numeric(df[y_col])
    df[pred_col] = clean_numeric(df[pred_col])
    df["_Abs_Error"] = (df[pred_col] - df[y_col]).abs()
    df["_Signed_Error"] = df[pred_col] - df[y_col]

    if role_col is not None:
        df["_Role_Normalized"] = df[role_col].map(normalize_role_value)
        if (df["_Role_Normalized"] == "train").any():
            df = df[df["_Role_Normalized"] != "train"].copy()

    block_cols = infer_block_columns(df, group_col)
    df = add_block_id(df, strategy_col, block_cols)
    sample_key_cols = infer_sample_key_columns(df, strategy_col, method_col, group_col, block_cols)

    log(f"[INFO] Block columns: {block_cols}")
    log(f"[INFO] Sample key columns: {sample_key_cols}")

    selected_methods = get_selected_methods(df, method_col)
    log(f"[INFO] Selected methods: {selected_methods}")

    df = df[df[method_col].astype(str).isin(selected_methods)].copy()
    df = add_model_disagreement_features(df, sample_key_cols, method_col, pred_col)

    policy_feature_cols = select_policy_feature_columns(df, y_col, pred_col, strategy_col, method_col)
    score_cols = select_score_columns_for_rules(policy_feature_cols, pred_col)

    log(f"[INFO] Policy feature columns: {len(policy_feature_cols)}")
    log(f"[INFO] Direct rule score columns: {len(score_cols)}")

    safe_to_csv(
        pd.DataFrame({
            "Feature": policy_feature_cols,
            "Used_For_Direct_Rule": [c in score_cols for c in policy_feature_cols],
        }),
        "32c_selective_policy_feature_audit.csv",
    )

    log("\n[STEP] Calibrated AD rule search...")
    rule_pred, rule_details = run_calibrated_rule_search(
        df, y_col, pred_col, strategy_col, method_col, score_cols, selected_methods
    )

    log("\n[STEP] Calibrated residual-risk selective prediction...")
    risk_pred, risk_details = run_calibrated_residual_risk_selection(
        df, y_col, pred_col, strategy_col, method_col,
        policy_feature_cols, selected_methods, sample_key_cols,
    )

    pred_parts = [x for x in [rule_pred, risk_pred] if not x.empty]
    detail_parts = [x for x in [rule_details, risk_details] if not x.empty]

    selective_pred = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    selected_details = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()

    selective_summary = summarize_selective_predictions(
        selective_pred, y_col, pred_col, strategy_col, method_col
    )
    baseline_summary = add_all_domain_baselines(
        df, y_col, pred_col, strategy_col, method_col, selected_methods
    )

    full_summary = pd.concat([baseline_summary, selective_summary], ignore_index=True)
    if not full_summary.empty:
        full_summary = full_summary.sort_values(["R2", "RMSE", "n_retained_rows"], ascending=[False, True, False])

    log("\n[STEP] Diagnostic oracle upper-bound search...")
    oracle = run_oracle_upper_bound(
        df, y_col, pred_col, strategy_col, method_col, score_cols, selected_methods
    )

    safe_to_csv(full_summary, "32c_selective_policy_summary.csv")
    if not selective_pred.empty:
        safe_to_csv(selective_pred, "32c_selective_policy_predictions.csv")
    if not selected_details.empty:
        safe_to_csv(selected_details, "32c_selected_policy_details.csv")
    if not oracle.empty:
        safe_to_csv(oracle, "32c_diagnostic_oracle_upper_bound.csv")

    plot_top_metrics(full_summary)
    plot_r2_coverage_frontier(full_summary)
    plot_best_predicted_vs_observed(selective_pred, full_summary, y_col, pred_col, strategy_col, method_col)
    plot_dynamic_method_frequency(selective_pred)

    write_report(full_summary, oracle, selected_details, selected_methods, policy_feature_cols, score_cols)

    log("\n" + "=" * 100)
    log("32C finished.")
    if not full_summary.empty:
        non_baseline = full_summary[full_summary["Policy"] != "AllDomain_NoFiltering"].copy()
        if not non_baseline.empty:
            best = non_baseline.iloc[0]
            log("\n[BEST CALIBRATED SELECTIVE RESULT]")
            log(best.to_string())
            if bool(best["Hit_Target_R2_0p5"]):
                log("\nSUCCESS: calibrated selective AD policy reached R2 >= 0.50.")
            else:
                log("\nNOT YET: calibrated selective AD policy did not reach R2 >= 0.50.")

    if not oracle.empty:
        log("\n[BEST DIAGNOSTIC ORACLE RESULT - NOT DEPLOYABLE]")
        log(oracle.iloc[0].to_string())

    elapsed = time.time() - START_TIME
    log("=" * 100)
    log(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    log("=" * 100)


if __name__ == "__main__":
    main()
