# -*- coding: utf-8 -*-
"""
32a_applicability_domain_analysis.py

Purpose
-------
Applicability-domain (AD; 适用域) analysis after all-domain grouped validation
and all-domain RMSE-oriented optimization.

This script is deliberately comprehensive rather than a simplified version.
It does NOT remove samples or literature groups from the all-domain report.
It quantifies where the current all-domain grouped models are reliable or
high-risk under literature-level grouped validation.

Background from previous steps
------------------------------
31P showed that forcing additional all-domain model optimization is not very
promising:
- model ranking changes across fully grouped validation forms;
- validation-RMSE model-switching does not reliably improve held-out test RMSE;
- residual correction and ordinary fusion do not solve the shared cross-literature
  error structure.

Therefore, this script starts the applicability-domain route.

Core questions
--------------
1. Can we quantify extrapolation risk without using test labels?
2. Do high-error papers/samples concentrate in high AD-risk regions?
3. What is the performance inside low-risk / medium-risk / high-risk domains?
4. Which features or categories cause out-of-domain behavior?
5. What reliable-domain performance can be reported beside all-domain performance?

Strict workflow
---------------
- 文献编号 is used only as the group label, never as a model input.
- AD scores are target-free: Observed F removal and prediction residuals are NOT
  used to construct AD scores.
- Test labels are used only after AD scoring for diagnostic evaluation.
- No samples are deleted from the all-domain results.
- Reliable-domain metrics are reported as stratified / thresholded diagnostics,
  not as a replacement for all-domain metrics.
- Total runtime is printed at the end.

Inputs expected
---------------
Required or strongly recommended:
    data/processed/model_ready_v1_with_source_mechanism31b.csv
        Preferred feature table if available; includes mechanism-aware features.
    or data/processed/model_ready_v1_with_source.csv
        Fallback feature table with 文献编号.

    outputs/tables/31n_grouped_validation_predictions.csv
        Preferred prediction table for all fixed models and fully grouped strategies.

Fallback prediction source:
    outputs/tables/31p_rmse_policy_predictions.csv
        Used only if 31N prediction table is unavailable; contains policy predictions.

Optional diagnostics:
    outputs/tables/31p_rmse_lolo_paper_model_metrics.csv
    outputs/tables/31p_rmse_high_error_paper_diagnosis.csv
    outputs/tables/31p_rmse_policy_summary.csv
    outputs/tables/31p_rmse_strategy_robust_ranking.csv
    outputs/tables/31p_rmse_global_decision.txt

Outputs
-------
Tables:
    outputs/tables/32a_ad_feature_space_audit.csv
    outputs/tables/32a_ad_sample_scores.csv
    outputs/tables/32a_ad_predictions_with_scores.csv
    outputs/tables/32a_ad_performance_by_risk_bin.csv
    outputs/tables/32a_ad_threshold_sweep.csv
    outputs/tables/32a_ad_paper_summary.csv
    outputs/tables/32a_ad_feature_range_violation_summary.csv
    outputs/tables/32a_ad_category_unseen_summary.csv
    outputs/tables/32a_ad_high_error_paper_merge.csv
    outputs/tables/32a_ad_decision_report.txt

Figures:
    outputs/figures/139_32a_ad_risk_error_relationship.png/pdf
    outputs/figures/140_32a_ad_performance_by_risk_bin.png/pdf
    outputs/figures/141_32a_paper_ad_risk_vs_rmse.png/pdf
    outputs/figures/142_32a_ad_violation_driver_summary.png/pdf
    outputs/figures/143_32a_reliable_domain_threshold_sweep.png/pdf
    outputs/figures/144_32a_high_error_paper_ad_component_heatmap.png/pdf
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 0. User-facing settings
# ============================================================

# The robust all-domain RMSE candidate from Step 31P. This is used as the
# primary AD diagnostic model, not as a hard final claim.
PRIMARY_METHOD = "CB-XGB"

# Other methods retained for comparison. Only methods present in the prediction
# table will be used.
METHOD_ORDER = [
    "CB-XGB",
    "CatBoost",
    "XGBoost",
    "KRR_Poly",
    "KRR_CatBoost_Weighted_2Model",
    "KRR_XGBoost_Weighted_2Model",
    "ValidationRMSE_Selected_Model",
    "StrategyBestFixedByValidation",
]

STRATEGY_ORDER = [
    "Balanced_5Seed_Current31F",
    "RepeatedBalanced_Extended",
    "RepeatedOrdinary_GroupedSplit",
    "GroupKFold_FullyGrouped",
    "LOLO_LeaveOneLiteratureOut",
]

# Risk score weights. All components are target-free.
AD_COMPONENT_WEIGHTS = {
    "NN_Risk": 0.25,
    "Mahalanobis_Risk": 0.25,
    "PCA_Reconstruction_Risk": 0.15,
    "Numeric_Range_Risk": 0.20,
    "Categorical_Unseen_Risk": 0.15,
}

# Fixed thresholds for reporting reliable-domain strata. These are not selected
# by test labels. A threshold sweep is also exported for transparency.
LOW_RISK_THRESHOLD = 0.50
MEDIUM_RISK_THRESHOLD = 0.70
RELIABLE_DOMAIN_THRESHOLD = 0.60
THRESHOLD_SWEEP = np.round(np.arange(0.40, 0.91, 0.05), 2)

# PCA settings for manifold-distance calculation.
PCA_MAX_COMPONENTS = 10
PCA_EXPLAINED_VARIANCE = 0.95
KNN_K = 5
EPS = 1e-12

# Keywords to identify target/post-treatment columns. These are excluded from
# target-free AD feature construction. The script writes an audit table so the
# user can check exactly what was included/excluded.
SOURCE_COL_CANDIDATES = ["文献编号", "source", "Source", "paper", "Paper", "literature", "Literature"]
TARGET_EXCLUDE_KEYWORDS = [
    "氟去除率", "去除率", "removal", "Removal", "target", "Target", "y_true", "Observed", "Predicted", "Residual",
]
POST_TREATMENT_EXCLUDE_KEYWORDS = [
    "出水F", "出水 F", "出水氟", "effluent f", "Effluent F", "Effluent_F",
    "残差", "Residual", "预测", "Predicted",
]
# 出水pH may or may not be available before deployment. We keep a separate audit
# flag; by default, we exclude it from target-free AD scoring to avoid relying on
# post-treatment information.
EXCLUDE_EFFLUENT_PH_FOR_AD = True
EFFLUENT_PH_KEYWORDS = ["出水pH", "出水 pH", "effluent ph", "Effluent pH", "Effluent_pH"]

COLORS = {
    "CB-XGB": "#808080",
    "CatBoost": "#9B72B0",
    "XGBoost": "#D28C3D",
    "KRR_Poly": "#76B083",
    "KRR_CatBoost_Weighted_2Model": "#67A9CF",
    "KRR_XGBoost_Weighted_2Model": "#B2ABD2",
    "ValidationRMSE_Selected_Model": "#E64B35",
    "StrategyBestFixedByValidation": "#4DBBD5",
    "Low": "#76B083",
    "Medium": "#E5C46B",
    "High": "#D95F5F",
    "VeryHigh": "#A23E3E",
    "grid": "#E6E6E6",
    "red": "#FF0000",
    "gray": "#666666",
}

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun", "Microsoft YaHei", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.unicode_minus": False,
})


# ============================================================
# 1. Basic utilities
# ============================================================

def log(msg: str):
    print(msg, flush=True)


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent] + list(here.parents)
    for root in candidates:
        if (root / "data").exists() and (root / "outputs").exists():
            return root
        if (root / "outputs" / "tables").exists() and (root / "outputs" / "figures").exists():
            return root
    raise FileNotFoundError("Cannot locate project root containing data/ and outputs/ folders.")


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def read_csv_required(path: Path, desc: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required {desc} not found: {path}")
    df = pd.read_csv(path)
    log(f"Read {desc}: {path} | shape={df.shape}")
    return df


def read_csv_optional(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        log(f"[Optional missing] {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    log(f"Read optional: {path} | shape={df.shape}")
    return df


def save_fig(fig, filename: str):
    png = FIG_DIR / filename
    pdf = FIG_DIR / filename.replace(".png", ".pdf")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved figure: {png}")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "Bias": np.nan, "n": 0}
    mse = mean_squared_error(yt, yp)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(yt, yp)
    bias = float(np.mean(yp - yt))
    r2 = float(r2_score(yt, yp)) if len(yt) >= 2 and np.nanstd(yt) > 0 else np.nan
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MSE": float(mse), "Bias": bias, "n": int(len(yt))}


def empirical_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return empirical percentile rank of each value against reference distribution."""
    ref = np.asarray(reference, dtype=float)
    vals = np.asarray(values, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        return np.full(len(vals), np.nan)
    ref_sorted = np.sort(ref)
    ranks = np.searchsorted(ref_sorted, vals, side="right") / len(ref_sorted)
    return np.clip(ranks, 0, 1)


def short_text(x, max_len=48):
    s = str(x)
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


# ============================================================
# 2. Data loading and standardization
# ============================================================

def locate_feature_table() -> Path:
    candidates = [
        PROCESSED_DIR / "model_ready_v1_with_source_mechanism31b.csv",
        PROCESSED_DIR / "model_ready_v1_with_source.csv",
        PROCESSED_DIR / "model_ready_v1.csv",
    ]
    p = first_existing(candidates)
    if p is None:
        raise FileNotFoundError(
            "Cannot find model-ready data. Expected one of:\n" + "\n".join(str(x) for x in candidates)
        )
    return p


def find_source_col(df: pd.DataFrame) -> Optional[str]:
    for c in SOURCE_COL_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = str(c).lower()
        if "文献" in str(c) or "paper" in cl or "source" in cl or "literature" in cl:
            return c
    return None


def standardize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aliases = {
        "Observed": ["Observed", "y_true", "Y_true", "Actual", "actual", "True", "target", "Target", "氟去除率"],
        "Predicted": ["Predicted", "y_pred", "Y_pred", "Prediction", "prediction", "Pred", "pred"],
        "Data_Index": ["Data_Index", "Original_Index", "Row_Index", "index", "Index", "data_index"],
        "文献编号": ["文献编号", "Paper", "paper", "Source", "source", "Literature", "literature"],
        "Method": ["Method", "Model", "model", "Candidate", "method"],
        "Strategy": ["Strategy", "Validation_Strategy", "strategy"],
        "Split_ID": ["Split_ID", "Split", "Fold_ID", "split_id"],
        "Seed_or_Fold": ["Seed_or_Fold", "Seed", "Fold", "seed", "fold"],
    }
    for target, cands in aliases.items():
        if target in df.columns:
            continue
        for c in cands:
            if c in df.columns:
                df = df.rename(columns={c: target})
                break
    if "Strategy" not in df.columns:
        df["Strategy"] = "UnknownStrategy"
    if "Method" not in df.columns:
        df["Method"] = "UnknownMethod"
    if "Split_ID" not in df.columns:
        if "Seed_or_Fold" in df.columns:
            df["Split_ID"] = df["Strategy"].astype(str) + "_" + df["Seed_or_Fold"].astype(str)
        else:
            df["Split_ID"] = df["Strategy"].astype(str) + "_split0"
    if "Seed_or_Fold" not in df.columns:
        df["Seed_or_Fold"] = df["Split_ID"].astype(str)
    required = ["Observed", "Predicted", "Data_Index", "文献编号", "Method", "Strategy", "Split_ID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction table is missing required standardized columns: {missing}\nColumns={list(df.columns)}")
    df["Data_Index"] = pd.to_numeric(df["Data_Index"], errors="coerce").astype("Int64")
    df["Observed"] = pd.to_numeric(df["Observed"], errors="coerce")
    df["Predicted"] = pd.to_numeric(df["Predicted"], errors="coerce")
    df["Residual"] = df["Observed"] - df["Predicted"]
    return df


def load_prediction_table() -> Tuple[pd.DataFrame, str]:
    # Prefer 31N because it contains fixed-model predictions for multiple fully
    # grouped validation strategies. Fall back to 31P policy predictions if needed.
    path_31n = TABLE_DIR / "31n_grouped_validation_predictions.csv"
    path_31p = TABLE_DIR / "31p_rmse_policy_predictions.csv"
    if path_31n.exists():
        raw = pd.read_csv(path_31n)
        source = "31n_grouped_validation_predictions.csv"
    elif path_31p.exists():
        raw = pd.read_csv(path_31p)
        source = "31p_rmse_policy_predictions.csv"
    else:
        raise FileNotFoundError(
            f"Cannot find prediction table. Expected {path_31n} or {path_31p}."
        )
    log(f"Read prediction table: {source} | shape={raw.shape}")
    pred = standardize_prediction_columns(raw)
    return pred, source


def is_excluded_col(col: str, source_col: str) -> Tuple[bool, str]:
    s = str(col)
    sl = s.lower()
    if s == source_col:
        return True, "source_group_label"
    if s in ["Data_Index", "Local_Test_Index", "Split_ID", "Seed_or_Fold"]:
        return True, "index_or_split_identifier"
    for kw in TARGET_EXCLUDE_KEYWORDS:
        if kw.lower() in sl or kw in s:
            return True, "target_or_prediction_related"
    for kw in POST_TREATMENT_EXCLUDE_KEYWORDS:
        if kw.lower() in sl or kw in s:
            return True, "post_treatment_output_or_target_derived"
    if EXCLUDE_EFFLUENT_PH_FOR_AD:
        for kw in EFFLUENT_PH_KEYWORDS:
            if kw.lower() in sl or kw in s:
                return True, "post_treatment_effluent_pH_excluded_by_AD_setting"
    return False, "included"


@dataclass
class FeatureSpace:
    name: str
    feature_cols: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]
    audit_df: pd.DataFrame


def build_feature_space(df: pd.DataFrame, source_col: str) -> FeatureSpace:
    audit_records = []
    feature_cols = []
    for c in df.columns:
        excluded, reason = is_excluded_col(c, source_col)
        if not excluded:
            feature_cols.append(c)
        audit_records.append({
            "Feature_Table_Column": c,
            "Included_in_AD": not excluded,
            "Reason": reason,
            "DType": str(df[c].dtype),
            "Missing_Fraction": float(df[c].isna().mean()),
            "Unique_Count": int(df[c].nunique(dropna=True)),
        })
    audit_df = pd.DataFrame(audit_records)

    numeric_cols = []
    categorical_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            # Exclude fully missing numeric columns.
            if df[c].notna().sum() > 0:
                numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    return FeatureSpace(
        name="TargetFree_InputMechanism_AD",
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        audit_df=audit_df,
    )


# ============================================================
# 3. Fold-wise target-free AD scoring
# ============================================================

@dataclass
class FoldADResult:
    sample_scores: pd.DataFrame
    range_violations: pd.DataFrame
    category_unseen: pd.DataFrame


def prepare_numeric_matrices(
    feature_df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    numeric_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if len(numeric_cols) == 0:
        return np.empty((len(train_idx), 0)), np.empty((len(test_idx), 0)), []

    train_num = feature_df.iloc[train_idx][numeric_cols].copy()
    test_num = feature_df.iloc[test_idx][numeric_cols].copy()

    # Keep columns with at least two non-missing values and nonzero training variance.
    keep_cols = []
    for c in numeric_cols:
        vals = pd.to_numeric(train_num[c], errors="coerce")
        if vals.notna().sum() >= 2 and float(vals.std(skipna=True)) > EPS:
            keep_cols.append(c)
    if len(keep_cols) == 0:
        return np.empty((len(train_idx), 0)), np.empty((len(test_idx), 0)), []

    train_num = train_num[keep_cols].apply(pd.to_numeric, errors="coerce")
    test_num = test_num[keep_cols].apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(train_num)
    X_test_imp = imputer.transform(test_num)
    X_train = scaler.fit_transform(X_train_imp)
    X_test = scaler.transform(X_test_imp)
    return X_train, X_test, keep_cols


def compute_knn_risk(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = X_train.shape[0]
    if X_train.shape[1] == 0 or n_train < 3:
        n_test = X_test.shape[0]
        return np.full(n_test, np.nan), np.full(n_test, np.nan), np.full(n_train, np.nan)
    k_train = min(KNN_K + 1, n_train)
    nn_train = NearestNeighbors(n_neighbors=k_train, metric="euclidean")
    nn_train.fit(X_train)
    train_dist, _ = nn_train.kneighbors(X_train)
    # Exclude self-neighbor when possible.
    if train_dist.shape[1] > 1:
        train_internal = np.mean(train_dist[:, 1:], axis=1)
    else:
        train_internal = train_dist[:, 0]

    k_test = min(KNN_K, n_train)
    nn_test = NearestNeighbors(n_neighbors=k_test, metric="euclidean")
    nn_test.fit(X_train)
    test_dist, _ = nn_test.kneighbors(X_test)
    test_mean_dist = np.mean(test_dist, axis=1)
    risk = empirical_percentile(test_mean_dist, train_internal)
    return test_mean_dist, risk, train_internal


def compute_pca_risks(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    n_train, n_features = X_train.shape
    n_test = X_test.shape[0]
    if n_features == 0 or n_train < 5:
        return (
            np.full(n_test, np.nan), np.full(n_test, np.nan),
            np.full(n_test, np.nan), np.full(n_test, np.nan), 0
        )
    max_possible = max(1, min(PCA_MAX_COMPONENTS, n_features, n_train - 1))
    pca_full = PCA(n_components=max_possible, random_state=42)
    pca_full.fit(X_train)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumsum, PCA_EXPLAINED_VARIANCE) + 1)
    n_comp = max(1, min(n_comp, max_possible))

    pca = PCA(n_components=n_comp, random_state=42)
    train_scores = pca.fit_transform(X_train)
    test_scores = pca.transform(X_test)
    train_var = np.var(train_scores, axis=0, ddof=1) + EPS
    train_md = np.sqrt(np.sum((train_scores ** 2) / train_var, axis=1))
    test_md = np.sqrt(np.sum((test_scores ** 2) / train_var, axis=1))
    md_risk = empirical_percentile(test_md, train_md)

    train_recon = pca.inverse_transform(train_scores)
    test_recon = pca.inverse_transform(test_scores)
    train_recon_err = np.sqrt(np.mean((X_train - train_recon) ** 2, axis=1))
    test_recon_err = np.sqrt(np.mean((X_test - test_recon) ** 2, axis=1))
    recon_risk = empirical_percentile(test_recon_err, train_recon_err)
    return test_md, md_risk, test_recon_err, recon_risk, n_comp


def compute_numeric_range_risk(
    feature_df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    numeric_cols: List[str],
    strategy: str,
    split_id: str,
    test_groups: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_test = len(test_idx)
    if len(numeric_cols) == 0:
        return (
            pd.DataFrame({
                "Numeric_Range_Violation_Fraction": np.zeros(n_test),
                "Numeric_Range_Mean_Severity": np.zeros(n_test),
                "Numeric_Range_Risk": np.zeros(n_test),
            }),
            pd.DataFrame(),
        )

    train = feature_df.iloc[train_idx]
    test = feature_df.iloc[test_idx]
    sample_violation_count = np.zeros(n_test)
    sample_severity_sum = np.zeros(n_test)
    feature_records = []

    usable_count = 0
    for c in numeric_cols:
        tr = pd.to_numeric(train[c], errors="coerce")
        te = pd.to_numeric(test[c], errors="coerce")
        if tr.notna().sum() < 2:
            continue
        tr_min = float(tr.min())
        tr_max = float(tr.max())
        q1 = float(tr.quantile(0.25))
        q3 = float(tr.quantile(0.75))
        iqr = max(q3 - q1, EPS)
        vals = te.to_numpy(dtype=float)
        below = vals < tr_min
        above = vals > tr_max
        violation = below | above
        severity = np.zeros(n_test, dtype=float)
        severity[below] = (tr_min - vals[below]) / iqr
        severity[above] = (vals[above] - tr_max) / iqr
        severity = np.where(np.isfinite(severity), severity, 0)
        sample_violation_count += violation.astype(float)
        sample_severity_sum += np.clip(severity, 0, 5)
        usable_count += 1
        if violation.sum() > 0:
            feature_records.append({
                "Strategy": strategy,
                "Split_ID": split_id,
                "Feature": c,
                "Test_Groups": ";".join(map(str, test_groups)),
                "n_test": n_test,
                "n_violations": int(violation.sum()),
                "Violation_Fraction": float(violation.mean()),
                "Mean_Severity_All_Test": float(np.mean(np.clip(severity, 0, 5))),
                "Mean_Severity_Among_Violations": float(np.mean(np.clip(severity[violation], 0, 5))) if violation.sum() > 0 else 0.0,
                "Max_Severity": float(np.max(np.clip(severity, 0, 5))) if n_test > 0 else 0.0,
                "Train_Min": tr_min,
                "Train_Max": tr_max,
                "Train_IQR": iqr,
            })

    denom = max(usable_count, 1)
    frac = sample_violation_count / denom
    mean_severity = sample_severity_sum / denom
    # Risk combines frequency and capped severity. Range risk remains target-free.
    range_risk = np.clip(0.70 * frac + 0.30 * np.minimum(mean_severity / 2.0, 1.0), 0, 1)
    sample_df = pd.DataFrame({
        "Numeric_Range_Violation_Fraction": frac,
        "Numeric_Range_Mean_Severity": mean_severity,
        "Numeric_Range_Risk": range_risk,
    })
    return sample_df, pd.DataFrame(feature_records)


def compute_categorical_unseen_risk(
    feature_df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    categorical_cols: List[str],
    strategy: str,
    split_id: str,
    test_groups: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_test = len(test_idx)
    if len(categorical_cols) == 0:
        return (
            pd.DataFrame({
                "Categorical_Unseen_Fraction": np.zeros(n_test),
                "Categorical_Unseen_Risk": np.zeros(n_test),
            }),
            pd.DataFrame(),
        )
    train = feature_df.iloc[train_idx]
    test = feature_df.iloc[test_idx]
    unseen_count = np.zeros(n_test)
    usable_count = 0
    records = []
    for c in categorical_cols:
        tr_vals = train[c].astype(str).fillna("<MISSING>")
        te_vals = test[c].astype(str).fillna("<MISSING>")
        train_set = set(tr_vals.unique())
        unseen = ~te_vals.isin(train_set)
        unseen_count += unseen.to_numpy(dtype=float)
        usable_count += 1
        if unseen.sum() > 0:
            vc = te_vals[unseen].value_counts()
            for val, cnt in vc.items():
                records.append({
                    "Strategy": strategy,
                    "Split_ID": split_id,
                    "Feature": c,
                    "Unseen_Value": val,
                    "Count": int(cnt),
                    "Fraction_in_Test": float(cnt / max(n_test, 1)),
                    "Test_Groups": ";".join(map(str, test_groups)),
                })
    denom = max(usable_count, 1)
    frac = unseen_count / denom
    risk = np.clip(frac, 0, 1)
    sample_df = pd.DataFrame({
        "Categorical_Unseen_Fraction": frac,
        "Categorical_Unseen_Risk": risk,
    })
    return sample_df, pd.DataFrame(records)


def compute_fold_ad(
    feature_df: pd.DataFrame,
    source_col: str,
    feature_space: FeatureSpace,
    strategy: str,
    split_id: str,
    test_data_indices: np.ndarray,
) -> FoldADResult:
    # Map Data_Index to actual row positions. In this project Data_Index is the
    # row number in the model-ready table. We guard against missing indices.
    available = feature_df["_Data_Index"].to_numpy()
    test_data_indices = np.array([int(x) for x in test_data_indices if pd.notna(x)], dtype=int)
    test_mask = feature_df["_Data_Index"].isin(test_data_indices).to_numpy()
    test_idx = np.where(test_mask)[0]
    if len(test_idx) == 0:
        raise ValueError(f"No test rows found for Strategy={strategy}, Split_ID={split_id}")
    test_groups = sorted(feature_df.iloc[test_idx][source_col].astype(str).unique().tolist())
    # Strict grouped AD reference: all literature groups in the test split are
    # excluded from the AD reference/training domain.
    train_mask = ~feature_df[source_col].astype(str).isin(test_groups)
    train_idx = np.where(train_mask.to_numpy())[0]
    if len(train_idx) < 5:
        raise ValueError(f"Too few training/reference rows for Strategy={strategy}, Split_ID={split_id}")

    X_train, X_test, used_num_cols = prepare_numeric_matrices(
        feature_df, train_idx, test_idx, feature_space.numeric_cols
    )
    knn_dist, knn_risk, _ = compute_knn_risk(X_train, X_test)
    md, md_risk, recon, recon_risk, n_pca = compute_pca_risks(X_train, X_test)
    range_df, range_records = compute_numeric_range_risk(
        feature_df, train_idx, test_idx, used_num_cols, strategy, split_id, test_groups
    )
    cat_df, cat_records = compute_categorical_unseen_risk(
        feature_df, train_idx, test_idx, feature_space.categorical_cols, strategy, split_id, test_groups
    )

    score_df = pd.DataFrame({
        "Strategy": strategy,
        "Split_ID": split_id,
        "Data_Index": feature_df.iloc[test_idx]["_Data_Index"].to_numpy(dtype=int),
        "文献编号": feature_df.iloc[test_idx][source_col].astype(str).to_numpy(),
        "n_reference_train_rows": len(train_idx),
        "n_reference_train_groups": feature_df.iloc[train_idx][source_col].nunique(),
        "n_test_groups_in_split": len(test_groups),
        "AD_Numeric_Features_Used": len(used_num_cols),
        "AD_Categorical_Features_Used": len(feature_space.categorical_cols),
        "AD_PCA_Components": n_pca,
        "KNN_Mean_Distance": knn_dist,
        "NN_Risk": knn_risk,
        "Mahalanobis_Distance": md,
        "Mahalanobis_Risk": md_risk,
        "PCA_Reconstruction_Error": recon,
        "PCA_Reconstruction_Risk": recon_risk,
    })
    score_df = pd.concat([score_df.reset_index(drop=True), range_df.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)

    # Weighted target-free AD risk score.
    risk_components = []
    for comp, w in AD_COMPONENT_WEIGHTS.items():
        if comp in score_df.columns:
            score_df[comp] = pd.to_numeric(score_df[comp], errors="coerce").fillna(0.5)
            risk_components.append(w * score_df[comp].to_numpy(dtype=float))
    if risk_components:
        score_df["AD_Risk_Score"] = np.clip(np.sum(risk_components, axis=0), 0, 1)
    else:
        score_df["AD_Risk_Score"] = np.nan

    score_df["AD_Risk_Bin"] = pd.cut(
        score_df["AD_Risk_Score"],
        bins=[-np.inf, LOW_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    # A stricter pre-defined reliable-domain flag. It is target-free and does not
    # use observed/predicted errors.
    score_df["Reliable_Domain_Flag"] = (
        (score_df["AD_Risk_Score"] <= RELIABLE_DOMAIN_THRESHOLD)
        & (score_df["Numeric_Range_Violation_Fraction"] <= 0.20)
        & (score_df["Categorical_Unseen_Fraction"] <= 0.20)
    )

    return FoldADResult(score_df, range_records, cat_records)


def compute_all_ad_scores(
    feature_df: pd.DataFrame,
    source_col: str,
    feature_space: FeatureSpace,
    pred_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_unique = pred_df[["Strategy", "Split_ID", "Data_Index"]].dropna().drop_duplicates()
    all_scores = []
    all_range = []
    all_cat = []
    grouped = pred_unique.groupby(["Strategy", "Split_ID"], sort=False)
    log(f"Computing AD scores for {len(grouped)} grouped split/fold units ...")
    for i, ((strategy, split_id), g) in enumerate(grouped, start=1):
        if i % 10 == 0 or i == 1:
            log(f"  AD fold {i}/{len(grouped)} | {strategy} | {split_id} | n_test_unique={g['Data_Index'].nunique()}")
        try:
            result = compute_fold_ad(
                feature_df=feature_df,
                source_col=source_col,
                feature_space=feature_space,
                strategy=str(strategy),
                split_id=str(split_id),
                test_data_indices=g["Data_Index"].dropna().astype(int).unique(),
            )
            all_scores.append(result.sample_scores)
            if len(result.range_violations) > 0:
                all_range.append(result.range_violations)
            if len(result.category_unseen) > 0:
                all_cat.append(result.category_unseen)
        except Exception as e:
            log(f"[Warning] AD score failed for {strategy} | {split_id}: {e}")
    sample_scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    range_df = pd.concat(all_range, ignore_index=True) if all_range else pd.DataFrame()
    cat_df = pd.concat(all_cat, ignore_index=True) if all_cat else pd.DataFrame()
    return sample_scores, range_df, cat_df


# ============================================================
# 4. Diagnostic summaries
# ============================================================

def merge_predictions_with_ad(pred_df: pd.DataFrame, ad_scores: pd.DataFrame) -> pd.DataFrame:
    cols = ["Strategy", "Split_ID", "Data_Index"]
    merged = pred_df.merge(ad_scores, on=cols, how="left", suffixes=("", "_AD"))
    if "文献编号_AD" in merged.columns:
        merged["文献编号"] = merged["文献编号"].fillna(merged["文献编号_AD"])
        merged = merged.drop(columns=["文献编号_AD"])
    merged["Abs_Error"] = np.abs(merged["Observed"] - merged["Predicted"])
    merged["Squared_Error"] = (merged["Observed"] - merged["Predicted"]) ** 2
    return merged


def summarize_performance_by_risk(pred_ad: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_cols = ["Strategy", "Method", "AD_Risk_Bin"]
    for keys, g in pred_ad.dropna(subset=["AD_Risk_Bin"]).groupby(group_cols, sort=False):
        strategy, method, risk_bin = keys
        m = regression_metrics(g["Observed"], g["Predicted"])
        records.append({
            "Strategy": strategy,
            "Method": method,
            "AD_Risk_Bin": risk_bin,
            "n": m["n"],
            "Coverage_in_Method_Strategy": float(len(g) / max(len(pred_ad[(pred_ad["Strategy"] == strategy) & (pred_ad["Method"] == method)]), 1)),
            "R2": m["R2"],
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
            "MSE": m["MSE"],
            "Bias": m["Bias"],
            "Mean_AD_Risk": float(g["AD_Risk_Score"].mean()),
            "Median_AD_Risk": float(g["AD_Risk_Score"].median()),
            "Mean_NN_Risk": float(g["NN_Risk"].mean()),
            "Mean_Mahalanobis_Risk": float(g["Mahalanobis_Risk"].mean()),
            "Mean_Range_Violation_Fraction": float(g["Numeric_Range_Violation_Fraction"].mean()),
            "Mean_Categorical_Unseen_Fraction": float(g["Categorical_Unseen_Fraction"].mean()),
        })
    out = pd.DataFrame(records)
    if len(out) > 0:
        out["AD_Risk_Bin"] = pd.Categorical(out["AD_Risk_Bin"], categories=["Low", "Medium", "High"], ordered=True)
        out = out.sort_values(["Strategy", "Method", "AD_Risk_Bin"])
    return out


def threshold_sweep(pred_ad: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    records = []
    base = pred_ad[pred_ad["Method"].isin(methods)].copy()
    for (strategy, method), g0 in base.groupby(["Strategy", "Method"], sort=False):
        total_n = len(g0)
        for thr in THRESHOLD_SWEEP:
            g = g0[g0["AD_Risk_Score"] <= thr]
            if len(g) == 0:
                continue
            m = regression_metrics(g["Observed"], g["Predicted"])
            records.append({
                "Strategy": strategy,
                "Method": method,
                "AD_Risk_Threshold": float(thr),
                "n_in_domain": int(len(g)),
                "coverage_fraction": float(len(g) / max(total_n, 1)),
                "R2": m["R2"],
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "Bias": m["Bias"],
                "Mean_AD_Risk": float(g["AD_Risk_Score"].mean()),
            })
    return pd.DataFrame(records)


def paper_level_summary(pred_ad: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_cols = ["Strategy", "Method", "文献编号"]
    for keys, g in pred_ad.groupby(group_cols, sort=False):
        strategy, method, paper = keys
        m = regression_metrics(g["Observed"], g["Predicted"])
        records.append({
            "Strategy": strategy,
            "Method": method,
            "文献编号": paper,
            "n_samples_or_predictions": len(g),
            "Observed_mean": float(g["Observed"].mean()),
            "Observed_std": float(g["Observed"].std(ddof=1)) if len(g) > 1 else 0.0,
            "Paper_R2": m["R2"],
            "Paper_RMSE": m["RMSE"],
            "Paper_MAE": m["MAE"],
            "Paper_Bias": m["Bias"],
            "Mean_AD_Risk": float(g["AD_Risk_Score"].mean()),
            "Median_AD_Risk": float(g["AD_Risk_Score"].median()),
            "P75_AD_Risk": float(g["AD_Risk_Score"].quantile(0.75)),
            "High_Risk_Fraction": float((g["AD_Risk_Bin"] == "High").mean()),
            "Reliable_Domain_Fraction": float(g["Reliable_Domain_Flag"].mean()),
            "Mean_NN_Risk": float(g["NN_Risk"].mean()),
            "Mean_Mahalanobis_Risk": float(g["Mahalanobis_Risk"].mean()),
            "Mean_Reconstruction_Risk": float(g["PCA_Reconstruction_Risk"].mean()),
            "Mean_Range_Violation_Fraction": float(g["Numeric_Range_Violation_Fraction"].mean()),
            "Mean_Range_Severity": float(g["Numeric_Range_Mean_Severity"].mean()),
            "Mean_Categorical_Unseen_Fraction": float(g["Categorical_Unseen_Fraction"].mean()),
        })
    return pd.DataFrame(records)


def summarize_feature_violations(range_df: pd.DataFrame) -> pd.DataFrame:
    if range_df.empty:
        return range_df
    out = range_df.groupby("Feature", as_index=False).agg(
        Total_Test_Folds=("Split_ID", "nunique"),
        Total_Violations=("n_violations", "sum"),
        Mean_Violation_Fraction=("Violation_Fraction", "mean"),
        Max_Violation_Fraction=("Violation_Fraction", "max"),
        Mean_Severity_All_Test=("Mean_Severity_All_Test", "mean"),
        Max_Severity=("Max_Severity", "max"),
    )
    out = out.sort_values(["Total_Violations", "Mean_Severity_All_Test"], ascending=False)
    return out


def summarize_category_unseen(cat_df: pd.DataFrame) -> pd.DataFrame:
    if cat_df.empty:
        return cat_df
    out = cat_df.groupby(["Feature", "Unseen_Value"], as_index=False).agg(
        Total_Count=("Count", "sum"),
        Mean_Fraction_in_Test=("Fraction_in_Test", "mean"),
        Appeared_in_Splits=("Split_ID", "nunique"),
    )
    out = out.sort_values(["Total_Count", "Mean_Fraction_in_Test"], ascending=False)
    return out


# ============================================================
# 5. Figures
# ============================================================

def plot_ad_risk_error_relationship(pred_ad: pd.DataFrame, methods: List[str]):
    data = pred_ad[pred_ad["Method"].isin(methods)].copy()
    # Prefer LOLO for paper-wise AD diagnosis; otherwise use all.
    if "LOLO_LeaveOneLiteratureOut" in data["Strategy"].unique():
        data = data[data["Strategy"] == "LOLO_LeaveOneLiteratureOut"].copy()
    if data.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    for method in methods:
        g = data[data["Method"] == method]
        if g.empty:
            continue
        ax.scatter(g["AD_Risk_Score"], g["Abs_Error"], s=18, alpha=0.45, label=method, color=COLORS.get(method, None), edgecolors="none")
    ax.set_xlabel("Target-free AD risk score")
    ax.set_ylabel("Absolute error in F removal (%)")
    ax.set_title("Sample-level error vs applicability-domain risk")
    ax.grid(True, color=COLORS["grid"], linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    # Binned RMSE for primary method.
    primary = PRIMARY_METHOD if PRIMARY_METHOD in data["Method"].unique() else data["Method"].iloc[0]
    g = data[data["Method"] == primary].dropna(subset=["AD_Risk_Bin"])
    bins = ["Low", "Medium", "High"]
    vals = []
    labels = []
    counts = []
    for b in bins:
        gb = g[g["AD_Risk_Bin"] == b]
        if len(gb) == 0:
            vals.append(np.nan); counts.append(0)
        else:
            vals.append(regression_metrics(gb["Observed"], gb["Predicted"])["RMSE"]); counts.append(len(gb))
        labels.append(f"{b}\n(n={counts[-1]})")
    x = np.arange(len(bins))
    bars = ax.bar(x, vals, color=[COLORS[b] for b in bins], edgecolor="#333333", linewidth=0.6)
    for bar, val in zip(bars, vals):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE")
    ax.set_title(f"{primary}: RMSE by AD risk bin")
    ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.6)
    fig.suptitle("32A applicability-domain risk and prediction error", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "139_32a_ad_risk_error_relationship.png")


def plot_performance_by_risk(perf: pd.DataFrame, methods: List[str]):
    if perf.empty:
        return
    data = perf[perf["Method"].isin(methods)].copy()
    if "LOLO_LeaveOneLiteratureOut" in data["Strategy"].unique():
        data = data[data["Strategy"] == "LOLO_LeaveOneLiteratureOut"].copy()
    if data.empty:
        return
    bins = ["Low", "Medium", "High"]
    methods_present = [m for m in methods if m in data["Method"].unique()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, metric in zip(axes, ["R2", "RMSE", "MAE"]):
        width = 0.8 / max(len(methods_present), 1)
        x = np.arange(len(bins))
        for i, method in enumerate(methods_present):
            vals = []
            for b in bins:
                sub = data[(data["Method"] == method) & (data["AD_Risk_Bin"].astype(str) == b)]
                vals.append(float(sub[metric].iloc[0]) if len(sub) > 0 else np.nan)
            offset = (i - (len(methods_present)-1)/2) * width
            bars = ax.bar(x + offset, vals, width=width, label=method, color=COLORS.get(method, None), edgecolor="#333333", linewidth=0.4, alpha=0.88)
            for bar, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.text(bar.get_x()+bar.get_width()/2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("32A model performance stratified by AD risk bin", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "140_32a_ad_performance_by_risk_bin.png")


def plot_paper_ad_risk_vs_rmse(paper_summary: pd.DataFrame, methods: List[str]):
    data = paper_summary[paper_summary["Method"].isin(methods)].copy()
    if "LOLO_LeaveOneLiteratureOut" in data["Strategy"].unique():
        data = data[data["Strategy"] == "LOLO_LeaveOneLiteratureOut"].copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for method in methods:
        g = data[data["Method"] == method]
        if g.empty:
            continue
        ax.scatter(g["Mean_AD_Risk"], g["Paper_RMSE"], s=np.clip(g["n_samples_or_predictions"].astype(float), 10, 160), alpha=0.65, label=method, color=COLORS.get(method, None), edgecolors="#333333", linewidth=0.3)
    # Label highest-error papers for primary method.
    primary = PRIMARY_METHOD if PRIMARY_METHOD in data["Method"].unique() else data["Method"].iloc[0]
    gp = data[data["Method"] == primary].sort_values("Paper_RMSE", ascending=False).head(12)
    for _, row in gp.iterrows():
        ax.text(row["Mean_AD_Risk"] + 0.005, row["Paper_RMSE"], str(row["文献编号"]), fontsize=7)
    ax.set_xlabel("Mean target-free AD risk score by paper")
    ax.set_ylabel("Paper-level RMSE")
    ax.set_title("Paper-level RMSE vs applicability-domain risk")
    ax.grid(True, color=COLORS["grid"], linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, "141_32a_paper_ad_risk_vs_rmse.png")


def plot_violation_driver_summary(range_summary: pd.DataFrame, cat_summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    if not range_summary.empty:
        top = range_summary.head(15).iloc[::-1]
        bars = ax.barh(top["Feature"].astype(str), top["Total_Violations"], color="#9BB8D3", edgecolor="#333333", linewidth=0.5)
        for bar, val in zip(bars, top["Total_Violations"]):
            ax.text(bar.get_width(), bar.get_y()+bar.get_height()/2, f"{int(val)}", va="center", ha="left", fontsize=7)
        ax.set_xlabel("Total range violations across grouped splits")
        ax.set_title("Numeric out-of-range drivers")
        ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.6)
    else:
        ax.text(0.5, 0.5, "No numeric range violations", ha="center", va="center")
        ax.axis("off")

    ax = axes[1]
    if not cat_summary.empty:
        top = cat_summary.head(15).copy()
        top["Label"] = top["Feature"].astype(str) + " = " + top["Unseen_Value"].astype(str).map(lambda x: short_text(x, 32))
        top = top.iloc[::-1]
        bars = ax.barh(top["Label"], top["Total_Count"], color="#D7B377", edgecolor="#333333", linewidth=0.5)
        for bar, val in zip(bars, top["Total_Count"]):
            ax.text(bar.get_width(), bar.get_y()+bar.get_height()/2, f"{int(val)}", va="center", ha="left", fontsize=7)
        ax.set_xlabel("Total unseen-category count")
        ax.set_title("Categorical novelty drivers")
        ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.6)
    else:
        ax.text(0.5, 0.5, "No categorical unseen values", ha="center", va="center")
        ax.axis("off")
    fig.suptitle("32A applicability-domain violation drivers", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "142_32a_ad_violation_driver_summary.png")


def plot_threshold_sweep(sweep: pd.DataFrame, methods: List[str]):
    if sweep.empty:
        return
    data = sweep[sweep["Method"].isin(methods)].copy()
    if "LOLO_LeaveOneLiteratureOut" in data["Strategy"].unique():
        data = data[data["Strategy"] == "LOLO_LeaveOneLiteratureOut"].copy()
    if data.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for method in methods:
        g = data[data["Method"] == method].sort_values("AD_Risk_Threshold")
        if g.empty:
            continue
        axes[0].plot(g["AD_Risk_Threshold"], g["coverage_fraction"], marker="o", label=method, color=COLORS.get(method, None))
        axes[1].plot(g["coverage_fraction"], g["RMSE"], marker="o", label=method, color=COLORS.get(method, None))
        for _, row in g.iterrows():
            if abs(row["AD_Risk_Threshold"] - RELIABLE_DOMAIN_THRESHOLD) < 1e-9:
                axes[1].scatter(row["coverage_fraction"], row["RMSE"], s=90, facecolors="none", edgecolors=COLORS["red"], linewidth=1.5)
    axes[0].axvline(RELIABLE_DOMAIN_THRESHOLD, color=COLORS["red"], linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("AD risk threshold")
    axes[0].set_ylabel("Coverage fraction")
    axes[0].set_title("Coverage retained by risk threshold")
    axes[1].set_xlabel("Coverage fraction")
    axes[1].set_ylabel("RMSE within threshold")
    axes[1].set_title("Reliable-domain RMSE trade-off")
    for ax in axes:
        ax.grid(True, color=COLORS["grid"], linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("32A reliable-domain threshold sweep", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "143_32a_reliable_domain_threshold_sweep.png")


def plot_high_error_component_heatmap(paper_summary: pd.DataFrame):
    if paper_summary.empty:
        return
    data = paper_summary.copy()
    if "LOLO_LeaveOneLiteratureOut" in data["Strategy"].unique():
        data = data[data["Strategy"] == "LOLO_LeaveOneLiteratureOut"]
    if PRIMARY_METHOD in data["Method"].unique():
        data = data[data["Method"] == PRIMARY_METHOD]
    elif len(data) > 0:
        data = data[data["Method"] == data["Method"].iloc[0]]
    if data.empty:
        return
    top = data.sort_values("Paper_RMSE", ascending=False).head(18).copy()
    comps = [
        "Mean_AD_Risk", "Mean_NN_Risk", "Mean_Mahalanobis_Risk", "Mean_Reconstruction_Risk",
        "Mean_Range_Violation_Fraction", "Mean_Categorical_Unseen_Fraction", "High_Risk_Fraction",
    ]
    comps = [c for c in comps if c in top.columns]
    mat = top[comps].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.35 * len(top) + 1.8)))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="YlOrRd")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["文献编号"].astype(str).tolist())
    ax.set_xticks(np.arange(len(comps)))
    ax.set_xticklabels(comps, rotation=35, ha="right")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Component value / fraction")
    ax.set_title(f"High-error papers: AD component heatmap ({PRIMARY_METHOD})")
    fig.tight_layout()
    save_fig(fig, "144_32a_high_error_paper_ad_component_heatmap.png")


# ============================================================
# 6. Decision report
# ============================================================

def write_decision_report(
    feature_path: Path,
    pred_source: str,
    feature_space: FeatureSpace,
    pred_ad: pd.DataFrame,
    perf: pd.DataFrame,
    sweep: pd.DataFrame,
    paper_summary: pd.DataFrame,
    range_summary: pd.DataFrame,
    cat_summary: pd.DataFrame,
    high_error_merge: pd.DataFrame,
):
    lines = []
    lines.append("=" * 100)
    lines.append("Step 32A applicability-domain analysis report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Quantify target-free applicability-domain risk after all-domain grouped optimization reached its practical limit.")
    lines.append("")
    lines.append("Strictness:")
    lines.append("- No samples or literature groups are removed from all-domain results.")
    lines.append("- AD scores are target-free and do not use observed F removal, test residuals, or test labels.")
    lines.append("- 文献编号 is used only to construct fully grouped reference domains, never as a model input.")
    lines.append("- Reliable-domain metrics are reported beside all-domain metrics, not as a replacement.")
    lines.append("")
    lines.append(f"Feature table used: {feature_path}")
    lines.append(f"Prediction table used: {pred_source}")
    lines.append(f"Feature space name: {feature_space.name}")
    lines.append(f"Included AD features: total={len(feature_space.feature_cols)}, numeric={len(feature_space.numeric_cols)}, categorical={len(feature_space.categorical_cols)}")
    lines.append(f"AD component weights: {AD_COMPONENT_WEIGHTS}")
    lines.append(f"Risk bins: Low <= {LOW_RISK_THRESHOLD}, Medium <= {MEDIUM_RISK_THRESHOLD}, High > {MEDIUM_RISK_THRESHOLD}")
    lines.append(f"Predefined reliable-domain threshold: AD_Risk_Score <= {RELIABLE_DOMAIN_THRESHOLD}, numeric range violation fraction <= 0.20, categorical unseen fraction <= 0.20")
    lines.append("")

    # Overall risk distribution.
    if not pred_ad.empty:
        unique_scores = pred_ad[["Strategy", "Split_ID", "Data_Index", "AD_Risk_Score", "AD_Risk_Bin", "Reliable_Domain_Flag"]].drop_duplicates()
        lines.append("AD risk distribution across unique split-sample pairs:")
        lines.append(unique_scores["AD_Risk_Bin"].value_counts(dropna=False).to_string())
        lines.append(f"Reliable-domain fraction: {unique_scores['Reliable_Domain_Flag'].mean():.3f}")
        lines.append("")

    # Performance by risk bin for primary method.
    if not perf.empty:
        lines.append(f"Performance by risk bin for primary method: {PRIMARY_METHOD}")
        p = perf[perf["Method"] == PRIMARY_METHOD].copy()
        cols = ["Strategy", "AD_Risk_Bin", "n", "Coverage_in_Method_Strategy", "R2", "RMSE", "MAE", "Bias", "Mean_AD_Risk"]
        if not p.empty:
            lines.append(p[cols].to_string(index=False))
        else:
            lines.append("Primary method not found in performance table.")
        lines.append("")

    # Threshold recommendation.
    if not sweep.empty:
        lines.append("Reliable-domain threshold sweep at predefined threshold:")
        s = sweep[np.isclose(sweep["AD_Risk_Threshold"], RELIABLE_DOMAIN_THRESHOLD)].copy()
        s = s[s["Method"].isin([PRIMARY_METHOD, "CatBoost", "XGBoost", "KRR_Poly", "KRR_CatBoost_Weighted_2Model"])]
        cols = ["Strategy", "Method", "coverage_fraction", "R2", "RMSE", "MAE", "Bias", "n_in_domain"]
        if not s.empty:
            lines.append(s[cols].sort_values(["Strategy", "RMSE"]).to_string(index=False))
        else:
            lines.append("No threshold-sweep rows at predefined threshold.")
        lines.append("")

    # Paper-risk correlations.
    if not paper_summary.empty:
        lines.append("Paper-level correlation between AD risk and paper RMSE:")
        corr_records = []
        for (strategy, method), g in paper_summary.groupby(["Strategy", "Method"]):
            if len(g) >= 3 and g["Mean_AD_Risk"].std() > 0 and g["Paper_RMSE"].std() > 0:
                corr = float(g["Mean_AD_Risk"].corr(g["Paper_RMSE"]))
            else:
                corr = np.nan
            corr_records.append({"Strategy": strategy, "Method": method, "Risk_RMSE_Correlation": corr, "n_papers": len(g)})
        corr_df = pd.DataFrame(corr_records)
        corr_df = corr_df[corr_df["Method"].isin([PRIMARY_METHOD, "CatBoost", "XGBoost", "KRR_Poly"])]
        lines.append(corr_df.sort_values(["Strategy", "Method"]).to_string(index=False))
        lines.append("")

    if not high_error_merge.empty:
        lines.append("High-error paper merge: top rows by primary-method paper RMSE or existing high-error diagnosis:")
        display_cols = [c for c in [
            "文献编号", "n_samples", "Observed_mean", "Observed_std", "KRR_Poly_Paper_RMSE", "XGBoost_Paper_RMSE", "CBXGB_Paper_RMSE",
            "Mean_AD_Risk", "High_Risk_Fraction", "Reliable_Domain_Fraction", "Mean_NN_Risk", "Mean_Mahalanobis_Risk",
            "Mean_Range_Violation_Fraction", "Mean_Categorical_Unseen_Fraction",
        ] if c in high_error_merge.columns]
        lines.append(high_error_merge[display_cols].head(25).to_string(index=False))
        lines.append("")

    if not range_summary.empty:
        lines.append("Top numeric range-violation drivers:")
        lines.append(range_summary.head(20).to_string(index=False))
        lines.append("")
    if not cat_summary.empty:
        lines.append("Top categorical unseen-value drivers:")
        lines.append(cat_summary.head(20).to_string(index=False))
        lines.append("")

    lines.append("Interpretation guide:")
    lines.append("- If AD risk increases with error and reliable-domain RMSE improves at reasonable coverage, reliable-domain reporting is justified.")
    lines.append("- If high-error papers also show high novelty/range-risk scores, the main limitation is cross-literature extrapolation rather than only model type.")
    lines.append("- If AD risk does not explain errors, additional data-quality and mechanism-specific diagnosis is required before using AD thresholds.")
    lines.append("")
    lines.append("Recommended next step:")
    lines.append("32B reliable-domain definition and model comparison: choose a pre-specified target-free AD rule, report all-domain and reliable-domain metrics side by side, and diagnose which chemical/literature regions remain unreliable.")
    lines.append("=" * 100)

    out_path = TABLE_DIR / "32a_ad_decision_report.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Saved decision report: {out_path}")


# ============================================================
# 7. Main
# ============================================================

def main():
    t0 = time.time()
    log("\n" + "=" * 100)
    log("32A applicability-domain analysis")
    log("=" * 100)
    log(f"Project root: {PROJECT_ROOT}")

    # Load feature table.
    feature_path = locate_feature_table()
    feature_df = read_csv_required(feature_path, "feature table")
    feature_df = feature_df.reset_index(drop=True).copy()
    feature_df["_Data_Index"] = np.arange(len(feature_df), dtype=int)
    source_col = find_source_col(feature_df)
    if source_col is None:
        raise ValueError(
            "Cannot find source/literature column in feature table. Expected 文献编号 or source-like column."
        )
    log(f"Source column: {source_col}")

    # Load predictions.
    pred_df, pred_source = load_prediction_table()
    pred_df = pred_df[pred_df["Data_Index"].notna()].copy()
    pred_df["Data_Index"] = pred_df["Data_Index"].astype(int)
    # Keep only methods and strategies of interest if possible, but do not drop
    # policy methods if they are the only available predictions.
    methods_present = pred_df["Method"].astype(str).unique().tolist()
    strategies_present = pred_df["Strategy"].astype(str).unique().tolist()
    log(f"Methods in prediction table: {methods_present}")
    log(f"Strategies in prediction table: {strategies_present}")

    preferred_methods = [m for m in METHOD_ORDER if m in methods_present]
    if preferred_methods:
        pred_df = pred_df[pred_df["Method"].isin(preferred_methods)].copy()
    log(f"Prediction rows used after method filter: {pred_df.shape}")

    # Align prediction source labels with feature table source labels when possible.
    feature_sources = feature_df[source_col].astype(str)
    # If pred source is missing or inconsistent, use feature table source by Data_Index.
    source_by_idx = feature_df.set_index("_Data_Index")[source_col].astype(str).to_dict()
    pred_df["文献编号"] = pred_df["Data_Index"].map(source_by_idx).fillna(pred_df["文献编号"].astype(str))

    # Build target-free feature space and save audit.
    feature_space = build_feature_space(feature_df, source_col)
    audit_path = TABLE_DIR / "32a_ad_feature_space_audit.csv"
    feature_space.audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")
    log(f"Saved feature-space audit: {audit_path}")
    log(f"AD feature counts: total={len(feature_space.feature_cols)}, numeric={len(feature_space.numeric_cols)}, categorical={len(feature_space.categorical_cols)}")

    # Compute target-free AD score for every split/sample used by predictions.
    ad_scores, range_df, cat_df = compute_all_ad_scores(feature_df, source_col, feature_space, pred_df)
    ad_scores_path = TABLE_DIR / "32a_ad_sample_scores.csv"
    ad_scores.to_csv(ad_scores_path, index=False, encoding="utf-8-sig")
    log(f"Saved sample AD scores: {ad_scores_path} | shape={ad_scores.shape}")

    # Merge with predictions and compute diagnostics.
    pred_ad = merge_predictions_with_ad(pred_df, ad_scores)
    pred_ad_path = TABLE_DIR / "32a_ad_predictions_with_scores.csv"
    pred_ad.to_csv(pred_ad_path, index=False, encoding="utf-8-sig")
    log(f"Saved predictions with AD scores: {pred_ad_path} | shape={pred_ad.shape}")

    perf = summarize_performance_by_risk(pred_ad)
    perf_path = TABLE_DIR / "32a_ad_performance_by_risk_bin.csv"
    perf.to_csv(perf_path, index=False, encoding="utf-8-sig")
    log(f"Saved performance by risk bin: {perf_path} | shape={perf.shape}")

    methods_for_sweep = [m for m in [PRIMARY_METHOD, "CatBoost", "XGBoost", "KRR_Poly", "KRR_CatBoost_Weighted_2Model", "KRR_XGBoost_Weighted_2Model"] if m in pred_ad["Method"].unique()]
    sweep = threshold_sweep(pred_ad, methods_for_sweep)
    sweep_path = TABLE_DIR / "32a_ad_threshold_sweep.csv"
    sweep.to_csv(sweep_path, index=False, encoding="utf-8-sig")
    log(f"Saved AD threshold sweep: {sweep_path} | shape={sweep.shape}")

    paper_sum = paper_level_summary(pred_ad)
    paper_path = TABLE_DIR / "32a_ad_paper_summary.csv"
    paper_sum.to_csv(paper_path, index=False, encoding="utf-8-sig")
    log(f"Saved AD paper summary: {paper_path} | shape={paper_sum.shape}")

    range_summary = summarize_feature_violations(range_df)
    range_path = TABLE_DIR / "32a_ad_feature_range_violation_summary.csv"
    range_summary.to_csv(range_path, index=False, encoding="utf-8-sig")
    log(f"Saved numeric range violation summary: {range_path} | shape={range_summary.shape}")

    cat_summary = summarize_category_unseen(cat_df)
    cat_path = TABLE_DIR / "32a_ad_category_unseen_summary.csv"
    cat_summary.to_csv(cat_path, index=False, encoding="utf-8-sig")
    log(f"Saved categorical unseen summary: {cat_path} | shape={cat_summary.shape}")

    # Merge with existing 31P high-error paper diagnosis, if present.
    high_error_31p = read_csv_optional("31p_rmse_high_error_paper_diagnosis.csv")
    high_error_merge = pd.DataFrame()
    if not high_error_31p.empty:
        # Use LOLO + primary method AD paper summary if available.
        ad_paper = paper_sum.copy()
        if "LOLO_LeaveOneLiteratureOut" in ad_paper["Strategy"].unique():
            ad_paper = ad_paper[ad_paper["Strategy"] == "LOLO_LeaveOneLiteratureOut"]
        if PRIMARY_METHOD in ad_paper["Method"].unique():
            ad_paper = ad_paper[ad_paper["Method"] == PRIMARY_METHOD]
        else:
            # Fall back to first available method.
            if len(ad_paper) > 0:
                ad_paper = ad_paper[ad_paper["Method"] == ad_paper["Method"].iloc[0]]
        ad_cols = [
            "文献编号", "Mean_AD_Risk", "Median_AD_Risk", "P75_AD_Risk", "High_Risk_Fraction",
            "Reliable_Domain_Fraction", "Mean_NN_Risk", "Mean_Mahalanobis_Risk", "Mean_Reconstruction_Risk",
            "Mean_Range_Violation_Fraction", "Mean_Range_Severity", "Mean_Categorical_Unseen_Fraction",
        ]
        ad_cols = [c for c in ad_cols if c in ad_paper.columns]
        high_error_merge = high_error_31p.merge(ad_paper[ad_cols].drop_duplicates("文献编号"), on="文献编号", how="left")
    else:
        high_error_merge = paper_sum.copy()
    high_error_path = TABLE_DIR / "32a_ad_high_error_paper_merge.csv"
    high_error_merge.to_csv(high_error_path, index=False, encoding="utf-8-sig")
    log(f"Saved high-error paper merge: {high_error_path} | shape={high_error_merge.shape}")

    # Figures.
    plot_methods = [m for m in [PRIMARY_METHOD, "CatBoost", "XGBoost", "KRR_Poly"] if m in pred_ad["Method"].unique()]
    if len(plot_methods) == 0:
        plot_methods = pred_ad["Method"].unique().tolist()[:4]
    plot_ad_risk_error_relationship(pred_ad, plot_methods)
    plot_performance_by_risk(perf, plot_methods)
    plot_paper_ad_risk_vs_rmse(paper_sum, plot_methods)
    plot_violation_driver_summary(range_summary, cat_summary)
    plot_threshold_sweep(sweep, plot_methods)
    plot_high_error_component_heatmap(paper_sum)

    # Decision report.
    write_decision_report(
        feature_path=feature_path,
        pred_source=pred_source,
        feature_space=feature_space,
        pred_ad=pred_ad,
        perf=perf,
        sweep=sweep,
        paper_summary=paper_sum,
        range_summary=range_summary,
        cat_summary=cat_summary,
        high_error_merge=high_error_merge,
    )

    elapsed = time.time() - t0
    log("\n" + "=" * 100)
    log("32A applicability-domain analysis finished.")
    log(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    log("=" * 100)


if __name__ == "__main__":
    main()
