# -*- coding: utf-8 -*-
"""
31g_expanded_model_complementarity_screening.py

Purpose
-------
Expanded machine-learning model complementarity screening under the exact
Step-31F balanced grouped-split framework.

This script intentionally reuses the real project scripts instead of rebuilding
a simplified workflow:
    - script 19: feature engineering / preprocessing / original model factories
    - script 31F: Step-31B mechanism-aware features, balanced grouped split,
      literature-group sample weights, plotting style, and metrics utilities

Main question
-------------
Which conventional ML model(s) genuinely complement the current CatBoost/HGB
main line under literature-level grouped validation?

Candidate base models
---------------------
1. CatBoost
2. HistGradientBoosting
3. ExtraTrees
4. RandomForest
5. XGBoost
6. LightGBM
7. Ridge
8. ElasticNet
9. SVR-RBF

Candidate fusion pools
----------------------
Validation-RMSE-optimized non-negative weighted fusion is tested for selected
pairs/triads/expanded pools, including CatBoost-HGB, CatBoost-ExtraTrees,
CatBoost-XGBoost, CatBoost-LightGBM, CatBoost-XGBoost-LightGBM,
CatBoost-HGB-ExtraTrees, CatBoost-HGB-XGBoost-LightGBM, and an expanded pool.

Strict rules
------------
1. 文献编号 is used only for grouped splitting; it is never an input feature.
2. Step-31B mechanism-aware feature representation is reused.
3. Step-31F balanced grouped split logic is reused.
4. Literature-group-weighted training is reused.
5. Test labels are never used for feature selection, weight search, tuning, or
   model selection.
6. This is still screening, not final tuning. Hyperparameter tuning is reserved
   for Step 31H after promising models are identified.

Outputs
-------
Tables:
    outputs/tables/31g_expanded_model_screen_split_summary.csv
    outputs/tables/31g_expanded_model_screen_validation_candidates.csv
    outputs/tables/31g_expanded_model_screen_test_results.csv
    outputs/tables/31g_expanded_model_screen_predictions.csv
    outputs/tables/31g_expanded_model_screen_summary.csv
    outputs/tables/31g_residual_correlation_matrix.csv
    outputs/tables/31g_paper_level_rmse_matrix.csv
    outputs/tables/31g_model_win_count_by_paper.csv
    outputs/tables/31g_pairwise_fusion_gain.csv
    outputs/tables/31g_expanded_model_complementarity_report.txt

Figures:
    outputs/figures/95_expanded_model_screen_metrics.png/pdf
    outputs/figures/96_expanded_model_residual_correlation.png/pdf
    outputs/figures/97_expanded_model_pairwise_fusion_gain.png/pdf
    outputs/figures/98_expanded_model_paper_level_rmse_heatmap.png/pdf
"""

from __future__ import annotations

import ast
import math
import warnings
import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# ============================================================
# 0. Basic logging and dynamic imports
# ============================================================

def log(msg: str):
    print(msg, flush=True)


def find_file_upwards(start_path: Path, filename: str) -> Path:
    start_path = start_path.resolve()
    search_roots = [start_path.parent] + list(start_path.parents)
    for root in search_roots:
        candidates = [root / filename, root / "src" / filename]
        for p in candidates:
            if p.exists():
                return p
    raise FileNotFoundError(f"Cannot find required file: {filename}")


def load_module_from_file(filename: str, module_name: str):
    module_path = find_file_upwards(Path(__file__), filename)
    log("=" * 100)
    log(f"Loading real {filename}")
    log(str(module_path))
    log("=" * 100)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s31f = load_module_from_file("31f_balanced_grouped_split_mechanism_validation.py", "s31f_real")
sf19 = s31f.sf19

PROJECT_ROOT = Path(s31f.PROJECT_ROOT)
FIG_DIR = Path(s31f.FIG_DIR)
TABLE_DIR = Path(s31f.TABLE_DIR)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Required optional libraries for expanded screening.
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception as e:
    HAS_XGBOOST = False
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = repr(e)

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except Exception as e:
    HAS_LIGHTGBM = False
    LGBMRegressor = None
    LIGHTGBM_IMPORT_ERROR = repr(e)

if not HAS_XGBOOST or not HAS_LIGHTGBM:
    msg = []
    msg.append("XGBoost/LightGBM are required for Step 31G expanded screening.")
    if not HAS_XGBOOST:
        msg.append(f"XGBoost import failed: {XGBOOST_IMPORT_ERROR}")
    if not HAS_LIGHTGBM:
        msg.append(f"LightGBM import failed: {LIGHTGBM_IMPORT_ERROR}")
    msg.append("Install in the project .venv: .\\.venv\\Scripts\\python.exe -m pip install xgboost lightgbm")
    raise ImportError("\n".join(msg))


# ============================================================
# 1. Settings inherited from script 19/31F
# ============================================================

SEEDS = list(getattr(sf19, "SEEDS", [7, 21, 42, 66, 88]))
K_OPTIONS = list(getattr(sf19, "K_OPTIONS", ["all"]))

BASE_MODEL_NAMES = [
    "CatBoost",
    "HistGradientBoosting",
    "ExtraTrees",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "Ridge",
    "ElasticNet",
    "SVR_RBF",
]

SINGLE_MODEL_TO_METHOD = {
    "CatBoost": "Selected_CatBoost",
    "HistGradientBoosting": "Selected_HistGradientBoosting",
    "ExtraTrees": "Selected_ExtraTrees",
    "RandomForest": "Selected_RandomForest",
    "XGBoost": "Selected_XGBoost",
    "LightGBM": "Selected_LightGBM",
    "Ridge": "Selected_Ridge",
    "ElasticNet": "Selected_ElasticNet",
    "SVR_RBF": "Selected_SVR_RBF",
}

# Predefined fusion pools. These are screened by validation-RMSE-optimized
# non-negative weights. Pools are intentionally interpretable, not arbitrary.
FUSION_POOLS = {
    "CatBoost_HGB_Weighted_2Model": ["CatBoost", "HistGradientBoosting"],
    "CatBoost_ExtraTrees_Weighted_2Model": ["CatBoost", "ExtraTrees"],
    "CatBoost_XGBoost_Weighted_2Model": ["CatBoost", "XGBoost"],
    "CatBoost_LightGBM_Weighted_2Model": ["CatBoost", "LightGBM"],
    "XGBoost_LightGBM_Weighted_2Model": ["XGBoost", "LightGBM"],
    "CatBoost_HGB_ExtraTrees_Weighted_3Model": ["CatBoost", "HistGradientBoosting", "ExtraTrees"],
    "CatBoost_XGBoost_LightGBM_Weighted_3Model": ["CatBoost", "XGBoost", "LightGBM"],
    "CatBoost_HGB_XGBoost_LightGBM_Weighted_4Model": ["CatBoost", "HistGradientBoosting", "XGBoost", "LightGBM"],
    "Expanded_Weighted_AllModel": [
        "CatBoost", "HistGradientBoosting", "ExtraTrees", "RandomForest",
        "XGBoost", "LightGBM", "Ridge", "ElasticNet", "SVR_RBF",
    ],
}

METHOD_ORDER = [
    # Existing / project-main methods first
    "CatBoost_HGB_Weighted_2Model",
    "Selected_CatBoost",
    "Selected_HistGradientBoosting",
    "Selected_ExtraTrees",
    "Selected_RandomForest",
    # Expanded base models
    "Selected_XGBoost",
    "Selected_LightGBM",
    "Selected_Ridge",
    "Selected_ElasticNet",
    "Selected_SVR_RBF",
    # Expanded fusion models
    "CatBoost_ExtraTrees_Weighted_2Model",
    "CatBoost_XGBoost_Weighted_2Model",
    "CatBoost_LightGBM_Weighted_2Model",
    "XGBoost_LightGBM_Weighted_2Model",
    "CatBoost_HGB_ExtraTrees_Weighted_3Model",
    "CatBoost_XGBoost_LightGBM_Weighted_3Model",
    "CatBoost_HGB_XGBoost_LightGBM_Weighted_4Model",
    "Expanded_Weighted_AllModel",
]

METHOD_LABELS = {
    "CatBoost_HGB_Weighted_2Model": "CatBoost-HGB 2Model",
    "Selected_CatBoost": "CatBoost",
    "Selected_HistGradientBoosting": "HGB",
    "Selected_ExtraTrees": "ExtraTrees",
    "Selected_RandomForest": "RF",
    "Selected_XGBoost": "XGBoost",
    "Selected_LightGBM": "LightGBM",
    "Selected_Ridge": "Ridge",
    "Selected_ElasticNet": "ElasticNet",
    "Selected_SVR_RBF": "SVR-RBF",
    "CatBoost_ExtraTrees_Weighted_2Model": "CatBoost-ET 2Model",
    "CatBoost_XGBoost_Weighted_2Model": "CatBoost-XGB 2Model",
    "CatBoost_LightGBM_Weighted_2Model": "CatBoost-LGBM 2Model",
    "XGBoost_LightGBM_Weighted_2Model": "XGB-LGBM 2Model",
    "CatBoost_HGB_ExtraTrees_Weighted_3Model": "CatBoost-HGB-ET 3Model",
    "CatBoost_XGBoost_LightGBM_Weighted_3Model": "CatBoost-XGB-LGBM 3Model",
    "CatBoost_HGB_XGBoost_LightGBM_Weighted_4Model": "CatBoost-HGB-XGB-LGBM 4Model",
    "Expanded_Weighted_AllModel": "Expanded AllModel",
}

COLORS = {
    "CatBoost_HGB_Weighted_2Model": "#63B7B2",
    "Selected_CatBoost": "#9B72B0",
    "Selected_HistGradientBoosting": "#63A97A",
    "Selected_ExtraTrees": "#A779A5",
    "Selected_RandomForest": "#4C78A8",
    "Selected_XGBoost": "#D28C3D",
    "Selected_LightGBM": "#6A9F58",
    "Selected_Ridge": "#7F7F7F",
    "Selected_ElasticNet": "#9E9E9E",
    "Selected_SVR_RBF": "#C05A63",
    "CatBoost_ExtraTrees_Weighted_2Model": "#B084B8",
    "CatBoost_XGBoost_Weighted_2Model": "#D09A6A",
    "CatBoost_LightGBM_Weighted_2Model": "#8FBF88",
    "XGBoost_LightGBM_Weighted_2Model": "#A6A15E",
    "CatBoost_HGB_ExtraTrees_Weighted_3Model": "#8C6BB1",
    "CatBoost_XGBoost_LightGBM_Weighted_3Model": "#B9825E",
    "CatBoost_HGB_XGBoost_LightGBM_Weighted_4Model": "#7B8EAD",
    "Expanded_Weighted_AllModel": "#8C8C8C",
    "grid": "#E6E6E6",
    "red": "#FF0000",
    "gray": "#666666",
}

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


# ============================================================
# 2. General utilities
# ============================================================

def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": math.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "Bias": float(np.mean(y_true - y_pred)),
    }


def weight_dict_to_text(weight_dict: dict) -> str:
    return str({str(k): float(v) for k, v in weight_dict.items()})


def parse_weight_text(x) -> dict:
    if pd.isna(x):
        return {}
    s = str(x).strip()
    if not s:
        return {}
    return ast.literal_eval(s)


def save_fig(fig, filename: str):
    png_path = FIG_DIR / filename
    pdf_path = FIG_DIR / filename.replace(".png", ".pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] {png_path}")
    log(f"[FIG] {pdf_path}")


def format_selected_k(values):
    vals = []
    for v in values:
        if pd.isna(v):
            continue
        if isinstance(v, float) and v.is_integer():
            vals.append(str(int(v)))
        else:
            vals.append(str(v))
    return ",".join(sorted(set(vals))) if vals else ""


# ============================================================
# 3. Model factories and prediction wrappers
# ============================================================

def make_model(model_name: str, seed: int):
    """Create a model with conservative screening parameters.

    For the four original project models, reuse script-19 factories exactly.
    For the expanded models, use reasonable conservative defaults. Step 31H will
    tune only the models that prove promising here.
    """
    if model_name in getattr(sf19, "MODEL_FACTORIES", {}):
        return sf19.MODEL_FACTORIES[model_name](seed)

    if model_name == "XGBoost":
        return XGBRegressor(
            n_estimators=600,
            learning_rate=0.035,
            max_depth=3,
            min_child_weight=3.0,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.05,
            reg_lambda=5.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )

    if model_name == "LightGBM":
        return LGBMRegressor(
            n_estimators=600,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=15,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.05,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    if model_name == "Ridge":
        return Ridge(alpha=5.0, random_state=seed)

    if model_name == "ElasticNet":
        return ElasticNet(alpha=0.001, l1_ratio=0.20, max_iter=20000, random_state=seed)

    if model_name == "SVR_RBF":
        return SVR(kernel="rbf", C=10.0, epsilon=2.0, gamma="scale")

    raise KeyError(f"Unknown model_name: {model_name}")


def needs_scaler(model_name: str) -> bool:
    return model_name in {"Ridge", "ElasticNet", "SVR_RBF"}


def fit_predict_model(model_name: str, X_train, y_train, X_target, seed: int, sample_weight=None):
    preprocessor, _, _ = sf19.build_preprocessor(X_train)
    model = make_model(model_name, seed)

    steps = [("preprocessor", preprocessor)]
    if needs_scaler(model_name):
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("model", model))

    pipe = Pipeline(steps)

    if sample_weight is None:
        pipe.fit(X_train, y_train)
    else:
        try:
            pipe.fit(X_train, y_train, model__sample_weight=np.asarray(sample_weight, dtype=float))
        except TypeError:
            log(f"[WARN] {model_name} does not accept sample_weight in this environment; fitting without sample_weight.")
            pipe.fit(X_train, y_train)

    pred = pipe.predict(X_target)
    return np.asarray(pred, dtype=float)


def select_columns_from_importance(imp_df: pd.DataFrame, X_mix: pd.DataFrame, selected_k):
    cols = sf19.select_top_features(imp_df, X_mix, selected_k)
    cols = [c for c in cols if c in X_mix.columns]
    return cols


# ============================================================
# 4. Validation evaluation and final testing
# ============================================================

def evaluate_validation_candidates(
    X_train_k,
    y_train,
    X_valid_k,
    y_valid,
    seed,
    selected_k,
    selected_cols,
    sample_weight_train=None,
):
    rows = []
    valid_pred_by_model = {}

    for model_name in BASE_MODEL_NAMES:
        log(f"    fitting validation model: {model_name}")
        pred_valid = fit_predict_model(
            model_name=model_name,
            X_train=X_train_k,
            y_train=y_train,
            X_target=X_valid_k,
            seed=seed,
            sample_weight=sample_weight_train,
        )
        valid_pred_by_model[model_name] = pred_valid
        method = SINGLE_MODEL_TO_METHOD[model_name]
        metrics = regression_metrics(y_valid, pred_valid)
        rows.append({
            "Seed": seed,
            "Method": method,
            "Candidate_Type": "single",
            "Base_Models": model_name,
            "Selected_K": selected_k,
            "Selected_Features": ";".join(selected_cols),
            "Validation_R2": metrics["R2"],
            "Validation_RMSE": metrics["RMSE"],
            "Validation_MAE": metrics["MAE"],
            "Validation_MSE": metrics["MSE"],
            "Validation_Bias": metrics["Bias"],
            "Weights": "",
        })

    # Predefined fusion pools
    for method, pool in FUSION_POOLS.items():
        pred_dict = {m: valid_pred_by_model[m] for m in pool if m in valid_pred_by_model}
        if len(pred_dict) < 2:
            continue
        weights, _ = sf19.search_best_weights(pred_dict, y_valid, seed)
        pred_fused = sf19.apply_weights(pred_dict, weights)
        metrics = regression_metrics(y_valid, pred_fused)
        rows.append({
            "Seed": seed,
            "Method": method,
            "Candidate_Type": "fusion",
            "Base_Models": ";".join(pool),
            "Selected_K": selected_k,
            "Selected_Features": ";".join(selected_cols),
            "Validation_R2": metrics["R2"],
            "Validation_RMSE": metrics["RMSE"],
            "Validation_MAE": metrics["MAE"],
            "Validation_MSE": metrics["MSE"],
            "Validation_Bias": metrics["Bias"],
            "Weights": weight_dict_to_text(weights),
        })

    return rows


def test_predict_method(method, weights_text, X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full=None):
    # Single model
    inv_single = {v: k for k, v in SINGLE_MODEL_TO_METHOD.items()}
    if method in inv_single:
        return fit_predict_model(
            inv_single[method],
            X_train_full_k,
            y_train_full,
            X_test_k,
            seed,
            sample_weight=sample_weight_train_full,
        )

    # Fusion model
    if method in FUSION_POOLS:
        weights = parse_weight_text(weights_text)
        pool = FUSION_POOLS[method]
        pred_dict = {}
        for m in pool:
            pred_dict[m] = fit_predict_model(
                m,
                X_train_full_k,
                y_train_full,
                X_test_k,
                seed,
                sample_weight=sample_weight_train_full,
            )
        return sf19.apply_weights(pred_dict, weights)

    raise ValueError(f"Unknown method: {method}")


def run_one_seed(X_mix: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int):
    log("\n" + "=" * 100)
    log(f"31G expanded model complementarity screening seed = {seed}")
    log("=" * 100)

    train_idx, valid_idx, test_idx, split_info = s31f.grouped_split_indices(groups, seed)

    log(
        f"Rows: train={split_info['n_train']}, valid={split_info['n_valid']}, test={split_info['n_test']}"
    )
    log(
        f"Groups: train={split_info['n_train_groups']}, valid={split_info['n_valid_groups']}, test={split_info['n_test_groups']}"
    )

    train_valid_idx = np.concatenate([train_idx, valid_idx])

    X_train = X_mix.iloc[train_idx].reset_index(drop=True)
    X_valid = X_mix.iloc[valid_idx].reset_index(drop=True)
    X_test = X_mix.iloc[test_idx].reset_index(drop=True)
    X_train_full = X_mix.iloc[train_valid_idx].reset_index(drop=True)

    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_valid = y.iloc[valid_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    y_train_full = y.iloc[train_valid_idx].reset_index(drop=True)

    train_groups_for_weight = groups.iloc[train_idx].reset_index(drop=True)
    train_full_groups_for_weight = groups.iloc[train_valid_idx].reset_index(drop=True)
    test_groups = groups.iloc[test_idx].reset_index(drop=True)

    sample_weight_train = s31f.make_literature_group_weights(train_groups_for_weight)
    sample_weight_train_full = s31f.make_literature_group_weights(train_full_groups_for_weight)

    log("Ranking features using weighted RF on grouped training set only...")
    imp_df = s31f.rank_features_by_rf_importance_weighted(
        X_train,
        y_train,
        seed,
        sample_weight=sample_weight_train,
    )

    candidate_rows = []

    for selected_k in K_OPTIONS:
        selected_cols = select_columns_from_importance(imp_df, X_mix, selected_k)
        if len(selected_cols) < 3:
            continue

        log("-" * 100)
        log(f"Seed={seed} | K={selected_k} | n_features={len(selected_cols)}")
        log("-" * 100)

        X_train_k = X_train[selected_cols].reset_index(drop=True)
        X_valid_k = X_valid[selected_cols].reset_index(drop=True)

        rows = evaluate_validation_candidates(
            X_train_k=X_train_k,
            y_train=y_train,
            X_valid_k=X_valid_k,
            y_valid=y_valid,
            seed=seed,
            selected_k=selected_k,
            selected_cols=selected_cols,
            sample_weight_train=sample_weight_train,
        )
        candidate_rows.extend(rows)

    candidate_df = pd.DataFrame(candidate_rows)

    test_rows = []
    pred_rows = []

    for method in METHOD_ORDER:
        sub = candidate_df[candidate_df["Method"] == method].copy()
        if sub.empty:
            continue

        best = sub.sort_values("Validation_RMSE", ascending=True).iloc[0]
        selected_k = best["Selected_K"]
        selected_cols = str(best["Selected_Features"]).split(";")
        selected_cols = [c for c in selected_cols if c in X_mix.columns]

        X_train_full_k = X_train_full[selected_cols].reset_index(drop=True)
        X_test_k = X_test[selected_cols].reset_index(drop=True)

        log("=" * 100)
        log(f"Final grouped test | seed={seed} | method={method} | K={selected_k} | n_features={len(selected_cols)}")
        log("=" * 100)

        pred_test = test_predict_method(
            method=method,
            weights_text=best["Weights"],
            X_train_full_k=X_train_full_k,
            y_train_full=y_train_full,
            X_test_k=X_test_k,
            seed=seed,
            sample_weight_train_full=sample_weight_train_full,
        )

        metrics = regression_metrics(y_test, pred_test)

        test_rows.append({
            "Seed": seed,
            "Method": method,
            "Candidate_Type": best.get("Candidate_Type", ""),
            "Base_Models": best.get("Base_Models", ""),
            "Selected_K": selected_k,
            "Selected_Features": ";".join(selected_cols),
            "Validation_R2": best["Validation_R2"],
            "Validation_RMSE": best["Validation_RMSE"],
            "Validation_MAE": best["Validation_MAE"],
            "Weights": best["Weights"],
            "Test_R2": metrics["R2"],
            "Test_RMSE": metrics["RMSE"],
            "Test_MAE": metrics["MAE"],
            "Test_MSE": metrics["MSE"],
            "Test_Bias": metrics["Bias"],
            "n_test": len(y_test),
            "n_test_groups": test_groups.nunique(),
        })

        for local_i, (obs, pred, gid, data_idx) in enumerate(zip(y_test, pred_test, test_groups, test_idx)):
            pred_rows.append({
                "Seed": seed,
                "Method": method,
                "Candidate_Type": best.get("Candidate_Type", ""),
                "Base_Models": best.get("Base_Models", ""),
                "Selected_K": selected_k,
                "Data_Index": int(data_idx),
                "Local_Test_Index": int(local_i),
                "文献编号": gid,
                "Observed": float(obs),
                "Predicted": float(pred),
                "Residual": float(obs - pred),
            })

    return {
        "split": pd.DataFrame([split_info]),
        "candidate": candidate_df,
        "test": pd.DataFrame(test_rows),
        "prediction": pd.DataFrame(pred_rows),
    }


# ============================================================
# 5. Summary and complementarity diagnostics
# ============================================================

def summarize_results(test_df: pd.DataFrame, pred_df: pd.DataFrame):
    summary = (
        test_df.groupby("Method", as_index=False)
        .agg(
            R2_mean=("Test_R2", "mean"),
            R2_std=("Test_R2", "std"),
            RMSE_mean=("Test_RMSE", "mean"),
            RMSE_std=("Test_RMSE", "std"),
            MAE_mean=("Test_MAE", "mean"),
            MAE_std=("Test_MAE", "std"),
            MSE_mean=("Test_MSE", "mean"),
            MSE_std=("Test_MSE", "std"),
            Bias_mean=("Test_Bias", "mean"),
            Bias_std=("Test_Bias", "std"),
            Selected_K=("Selected_K", format_selected_k),
            n_seed=("Seed", "nunique"),
            n_test_total=("n_test", "sum"),
        )
    )

    pooled_rows = []
    for method, sub in pred_df.groupby("Method"):
        metrics = regression_metrics(sub["Observed"], sub["Predicted"])
        pooled_rows.append({
            "Method": method,
            "Pooled_R2": metrics["R2"],
            "Pooled_RMSE": metrics["RMSE"],
            "Pooled_MAE": metrics["MAE"],
            "Pooled_MSE": metrics["MSE"],
            "Pooled_Bias": metrics["Bias"],
            "n_pooled": len(sub),
            "n_literature_groups_pooled": sub["文献编号"].nunique(),
        })

    pooled_df = pd.DataFrame(pooled_rows)
    summary = summary.merge(pooled_df, on="Method", how="left")
    summary["Method"] = pd.Categorical(summary["Method"], categories=METHOD_ORDER, ordered=True)
    summary = summary.sort_values("Method").reset_index(drop=True)
    return summary


def compute_residual_correlation(pred_df: pd.DataFrame):
    single_methods = [SINGLE_MODEL_TO_METHOD[m] for m in BASE_MODEL_NAMES]
    sub = pred_df[pred_df["Method"].isin(single_methods)].copy()
    pivot = sub.pivot_table(
        index=["Seed", "Data_Index"],
        columns="Method",
        values="Residual",
        aggfunc="mean",
    )
    corr = pivot.corr(method="pearson")
    corr = corr.reindex(index=[m for m in single_methods if m in corr.index], columns=[m for m in single_methods if m in corr.columns])
    return corr


def rmse_array(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else np.nan


def compute_paper_level_rmse(pred_df: pd.DataFrame):
    rows = []
    for (method, paper), sub in pred_df.groupby(["Method", "文献编号"]):
        rows.append({
            "Method": method,
            "文献编号": paper,
            "n": len(sub),
            "RMSE": rmse_array(sub["Residual"].values),
            "MAE": float(np.mean(np.abs(sub["Residual"].values))),
            "Bias_obs_minus_pred": float(np.mean(sub["Residual"].values)),
        })
    df = pd.DataFrame(rows)
    matrix = df.pivot_table(index="文献编号", columns="Method", values="RMSE", aggfunc="mean")
    # Sort papers by best-model RMSE to make heatmap interpretable.
    matrix["_min"] = matrix.min(axis=1)
    matrix = matrix.sort_values("_min", ascending=False).drop(columns=["_min"])
    return df, matrix


def compute_model_win_count(paper_rmse_df: pd.DataFrame):
    rows = []
    for paper, sub in paper_rmse_df.groupby("文献编号"):
        sub2 = sub.dropna(subset=["RMSE"]).copy()
        if sub2.empty:
            continue
        best = sub2.sort_values("RMSE", ascending=True).iloc[0]
        rows.append({
            "文献编号": paper,
            "Best_Method": best["Method"],
            "Best_RMSE": best["RMSE"],
            "n": int(best["n"]),
        })
    win_by_paper = pd.DataFrame(rows)
    win_count = (
        win_by_paper.groupby("Best_Method", as_index=False)
        .agg(n_best_papers=("文献编号", "nunique"), mean_best_rmse=("Best_RMSE", "mean"))
        .sort_values("n_best_papers", ascending=False)
    )
    return win_by_paper, win_count


def compute_pairwise_fusion_gain(summary_df: pd.DataFrame):
    # Compare each pairwise weighted model against the better of its single-model components.
    method_to_rmse = dict(zip(summary_df["Method"].astype(str), summary_df["Pooled_RMSE"].astype(float)))
    method_to_r2 = dict(zip(summary_df["Method"].astype(str), summary_df["Pooled_R2"].astype(float)))

    rows = []
    for method, pool in FUSION_POOLS.items():
        if len(pool) != 2 or method not in method_to_rmse:
            continue
        single_methods = [SINGLE_MODEL_TO_METHOD[p] for p in pool if SINGLE_MODEL_TO_METHOD[p] in method_to_rmse]
        if len(single_methods) != 2:
            continue
        best_single = min(single_methods, key=lambda m: method_to_rmse[m])
        rows.append({
            "Fusion_Method": method,
            "Base_Models": ";".join(pool),
            "Best_Single_Method": best_single,
            "Fusion_Pooled_RMSE": method_to_rmse[method],
            "Best_Single_Pooled_RMSE": method_to_rmse[best_single],
            "RMSE_Gain_vs_Best_Single": method_to_rmse[best_single] - method_to_rmse[method],
            "Fusion_Pooled_R2": method_to_r2.get(method, np.nan),
            "Best_Single_Pooled_R2": method_to_r2.get(best_single, np.nan),
            "R2_Gain_vs_Best_Single": method_to_r2.get(method, np.nan) - method_to_r2.get(best_single, np.nan),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("RMSE_Gain_vs_Best_Single", ascending=False).reset_index(drop=True)
    return out


# ============================================================
# 6. Figures and report
# ============================================================

def plot_metrics(summary_df: pd.DataFrame):
    data = summary_df.copy()
    # Show all methods but sort by RMSE for readability.
    data = data.sort_values("Pooled_RMSE", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.3))
    specs = [
        ("Pooled_R2", "Pooled $R^2$", "{:.3f}", "max"),
        ("Pooled_RMSE", "Pooled RMSE", "{:.2f}", "min"),
        ("Pooled_MAE", "Pooled MAE", "{:.2f}", "min"),
    ]

    for ax, (col, ylabel, fmt, best_rule) in zip(axes, specs):
        vals = data[col].astype(float).values
        methods = data["Method"].astype(str).tolist()
        x = np.arange(len(data))
        bar_colors = [COLORS.get(m, COLORS["gray"]) for m in methods]
        bars = ax.bar(x, vals, color=bar_colors, edgecolor="black", linewidth=0.45, alpha=0.9)
        best_idx = int(np.nanargmax(vals)) if best_rule == "max" else int(np.nanargmin(vals))
        bars[best_idx].set_edgecolor(COLORS["red"])
        bars[best_idx].set_linewidth(1.4)

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=38, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)

        ymin = np.nanmin(vals)
        ymax = np.nanmax(vals)
        if col == "Pooled_R2":
            lo = min(0, ymin - 0.05)
            hi = max(0.1, ymax + 0.06)
            ax.set_ylim(lo, hi)
            ax.axhline(0, color="#444444", linewidth=0.8)
            offset = (hi - lo) * 0.025
        else:
            ax.set_ylim(0, ymax * 1.20)
            offset = ax.get_ylim()[1] * 0.025

        for xi, v in zip(x, vals):
            ax.text(xi, v + (offset if v >= 0 else -offset), fmt.format(v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6.7)

    fig.suptitle("Expanded ML model screening under balanced literature-level grouped validation", fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.012, "Mechanism-aware descriptors + literature-group-weighted training; methods sorted by pooled RMSE.", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=[0.02, 0.06, 1, 0.93])
    save_fig(fig, "95_expanded_model_screen_metrics.png")


def plot_residual_correlation(corr: pd.DataFrame):
    if corr.empty:
        return
    labels = [METHOD_LABELS.get(str(c), str(c)) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(val) > 0.65 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson residual correlation")
    ax.set_title("Residual correlation among expanded base models")
    fig.tight_layout()
    save_fig(fig, "96_expanded_model_residual_correlation.png")


def plot_pairwise_gain(pair_gain: pd.DataFrame):
    if pair_gain.empty:
        return
    data = pair_gain.copy().sort_values("RMSE_Gain_vs_Best_Single", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(data))
    vals = data["RMSE_Gain_vs_Best_Single"].astype(float).values
    labels = [METHOD_LABELS.get(m, m) for m in data["Fusion_Method"].astype(str)]
    bars = ax.barh(y, vals, color="#8DA9C4", edgecolor="black", linewidth=0.45, alpha=0.9)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("RMSE gain vs best single model (positive = fusion improves)")
    ax.set_title("Pairwise validation-selected fusion gain on grouped test predictions")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    for yi, v in zip(y, vals):
        ax.text(v + (0.03 if v >= 0 else -0.03), yi, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)
    fig.tight_layout()
    save_fig(fig, "97_expanded_model_pairwise_fusion_gain.png")


def plot_paper_rmse_heatmap(paper_rmse_matrix: pd.DataFrame, max_papers: int = 30):
    if paper_rmse_matrix.empty:
        return
    data = paper_rmse_matrix.copy()
    # Keep important methods and limit paper count for readability.
    keep_methods = [m for m in METHOD_ORDER if m in data.columns]
    data = data[keep_methods]
    if data.shape[0] > max_papers:
        data = data.iloc[:max_papers]

    fig, ax = plt.subplots(figsize=(min(18, 0.68 * data.shape[1] + 4), max(6, 0.32 * data.shape[0] + 2)))
    im = ax.imshow(data.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels([METHOD_LABELS.get(str(c), str(c)) for c in data.columns], rotation=40, ha="right")
    ax.set_yticklabels(data.index.astype(str))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=5.8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Paper-level RMSE")
    ax.set_title("Paper-level RMSE matrix for expanded models")
    fig.tight_layout()
    save_fig(fig, "98_expanded_model_paper_level_rmse_heatmap.png")


def write_report(summary_df, split_df, corr, paper_rmse_df, win_count, pair_gain):
    path = TABLE_DIR / "31g_expanded_model_complementarity_report.txt"
    ranking = summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE", "Pooled_R2"], ascending=[True, True, False]).reset_index(drop=True)
    best = ranking.iloc[0]

    lines = []
    lines.append("=" * 100)
    lines.append("Step 31G expanded ML model complementarity screening report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Identify models that genuinely complement the CatBoost/HGB main line under balanced literature-level grouped validation.")
    lines.append("")
    lines.append("Strict workflow reused:")
    lines.append("- Step-31B mechanism-aware descriptors")
    lines.append("- Step-31F balanced grouped split")
    lines.append("- Literature-group-weighted training")
    lines.append("- Script-19 feature selection/preprocessing where applicable")
    lines.append("- Validation-RMSE-optimized non-negative fusion weights")
    lines.append("")
    lines.append("Expanded base model pool:")
    lines.append(", ".join(BASE_MODEL_NAMES))
    lines.append("")
    lines.append("Split summary:")
    cols = ["Seed", "n_train", "n_valid", "n_test", "n_train_groups", "n_valid_groups", "n_test_groups", "Balanced_Split_Score"]
    cols = [c for c in cols if c in split_df.columns]
    lines.append(split_df[cols].to_string(index=False))
    lines.append("")
    lines.append("Performance summary:")
    keep_cols = ["Method", "R2_mean", "R2_std", "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std", "Pooled_R2", "Pooled_RMSE", "Pooled_MAE", "Pooled_Bias", "n_pooled", "n_literature_groups_pooled"]
    lines.append(summary_df[keep_cols].to_string(index=False))
    lines.append("")
    lines.append("Ranking by pooled RMSE:")
    lines.append(ranking[["Method", "Pooled_R2", "Pooled_RMSE", "Pooled_MAE", "Pooled_Bias"]].to_string(index=False))
    lines.append("")
    lines.append("Current Step-31G best candidate:")
    lines.append(f"{best['Method']} | Pooled_R2={best['Pooled_R2']:.4f}, Pooled_RMSE={best['Pooled_RMSE']:.4f}, Pooled_MAE={best['Pooled_MAE']:.4f}")
    lines.append("")
    if not corr.empty:
        lines.append("Residual-correlation interpretation:")
        # Focus on CatBoost correlations.
        cb = "Selected_CatBoost"
        if cb in corr.columns:
            vals = corr[cb].drop(index=[cb], errors="ignore").sort_values()
            lines.append("Lowest residual correlations with CatBoost:")
            lines.append(vals.head(5).to_string())
            lines.append("Highest residual correlations with CatBoost:")
            lines.append(vals.tail(5).to_string())
        lines.append("")
    if not win_count.empty:
        lines.append("Model win-count by paper:")
        lines.append(win_count.to_string(index=False))
        lines.append("")
    if not pair_gain.empty:
        lines.append("Pairwise fusion gain vs best single model:")
        lines.append(pair_gain.to_string(index=False))
        lines.append("")
    lines.append("Next decision rule:")
    lines.append("1. If a new model improves pooled RMSE and has lower residual correlation with CatBoost/HGB, it is a valid complement candidate.")
    lines.append("2. If XGBoost/LightGBM or another candidate wins, Step 31H should tune only the promising subset, not every model blindly.")
    lines.append("3. If no fusion improves over the best single model, keep the best single model for all-domain grouped prediction and move to tuning/applicability-domain analysis.")
    lines.append("=" * 100)
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[TEXT] {path}")


# ============================================================
# 7. Main
# ============================================================

def main():
    log("\n" + "=" * 100)
    log("Step 31G: expanded model complementarity screening")
    log("=" * 100)
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"SEEDS = {SEEDS}")
    log(f"K_OPTIONS = {K_OPTIONS}")
    log(f"XGBoost available = {HAS_XGBOOST}")
    log(f"LightGBM available = {HAS_LIGHTGBM}")
    log("=" * 100)

    # Reuse script-19 feature preparation exactly, then apply Step-31B/31F feature integration.
    log("\nPreparing script-19 feature sets...")
    y, X_raw, X_eng, X_mix_base = sf19.prepare_feature_sets()
    y = y.reset_index(drop=True)
    X_mix_base = X_mix_base.reset_index(drop=True)

    log("\nAppending Step-31B mechanism-aware descriptors using real 31F integration function...")
    X_mix = s31f.append_mechanism31b_features(X_mix_base, expected_n=len(y))
    X_mix = X_mix.reset_index(drop=True)

    groups = s31f.load_group_labels(expected_n=len(y))
    s31f.set_balanced_split_context(X_mix, y, groups)

    all_splits = []
    all_candidates = []
    all_tests = []
    all_predictions = []

    for seed in SEEDS:
        result = run_one_seed(X_mix=X_mix, y=y, groups=groups, seed=seed)
        all_splits.append(result["split"])
        all_candidates.append(result["candidate"])
        all_tests.append(result["test"])
        all_predictions.append(result["prediction"])

    split_df = pd.concat(all_splits, ignore_index=True)
    candidate_df = pd.concat(all_candidates, ignore_index=True)
    test_df = pd.concat(all_tests, ignore_index=True)
    pred_df = pd.concat(all_predictions, ignore_index=True)
    summary_df = summarize_results(test_df, pred_df)

    # Complementarity diagnostics.
    corr = compute_residual_correlation(pred_df)
    paper_rmse_df, paper_rmse_matrix = compute_paper_level_rmse(pred_df)
    win_by_paper, win_count = compute_model_win_count(paper_rmse_df)
    pair_gain = compute_pairwise_fusion_gain(summary_df)

    # Save tables.
    out_tables = {
        "31g_expanded_model_screen_split_summary.csv": split_df,
        "31g_expanded_model_screen_validation_candidates.csv": candidate_df,
        "31g_expanded_model_screen_test_results.csv": test_df,
        "31g_expanded_model_screen_predictions.csv": pred_df,
        "31g_expanded_model_screen_summary.csv": summary_df,
        "31g_residual_correlation_matrix.csv": corr,
        "31g_paper_level_rmse_long.csv": paper_rmse_df,
        "31g_paper_level_rmse_matrix.csv": paper_rmse_matrix,
        "31g_model_win_by_paper.csv": win_by_paper,
        "31g_model_win_count_by_paper.csv": win_count,
        "31g_pairwise_fusion_gain.csv": pair_gain,
    }

    for filename, df in out_tables.items():
        path = TABLE_DIR / filename
        if isinstance(df, pd.DataFrame):
            df.to_csv(path, index=True if filename in {"31g_residual_correlation_matrix.csv", "31g_paper_level_rmse_matrix.csv"} else False, encoding="utf-8-sig")
            log(f"[TABLE] {path}")

    log("\n=== Step 31G expanded model screening summary ===")
    print(summary_df.to_string(index=False))

    write_report(summary_df, split_df, corr, paper_rmse_df, win_count, pair_gain)

    log("\nGenerating figures...")
    plot_metrics(summary_df)
    plot_residual_correlation(corr)
    plot_pairwise_gain(pair_gain)
    plot_paper_rmse_heatmap(paper_rmse_matrix)

    log("\n" + "=" * 100)
    log("[DONE] Step 31G expanded model complementarity screening completed.")
    log("=" * 100)
    log("Please send these outputs:")
    log("1. outputs/tables/31g_expanded_model_screen_summary.csv")
    log("2. outputs/tables/31g_expanded_model_complementarity_report.txt")
    log("3. outputs/tables/31g_residual_correlation_matrix.csv")
    log("4. outputs/tables/31g_pairwise_fusion_gain.csv")
    log("5. outputs/figures/95_expanded_model_screen_metrics.png")
    log("6. outputs/figures/96_expanded_model_residual_correlation.png")
    log("7. outputs/figures/97_expanded_model_pairwise_fusion_gain.png")
    log("=" * 100)


if __name__ == "__main__":
    main()
