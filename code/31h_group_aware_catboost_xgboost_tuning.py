# -*- coding: utf-8 -*-
"""
31h_group_aware_catboost_xgboost_tuning.py

Purpose
-------
Group-aware hyperparameter tuning for the Step-31G best complementary route:
CatBoost + XGBoost weighted 2Model under balanced literature-level grouped
validation.

This script intentionally continues the exact project workflow instead of
rebuilding a simplified pipeline:
    - script 19: original feature engineering / preprocessing / feature selection utilities
    - script 31F: Step-31B mechanism-aware descriptors, balanced grouped split,
      and literature-group sample weights
    - script 31G result: CatBoost-XGBoost weighted 2Model was the only pairwise
      fusion that improved over the best single CatBoost in pooled RMSE.

Main question
-------------
Can group-aware tuning of CatBoost and XGBoost improve all-domain grouped
validation performance beyond Step 31G?

Strict rules
------------
1. 文献编号 is used only for literature-level grouped splitting.
2. 文献编号 is never used as an input feature.
3. Step-31B mechanism-aware feature representation is reused.
4. Step-31F balanced grouped split logic is reused.
5. Literature-group-weighted training is reused.
6. Hyperparameters and fusion weights are selected only by validation RMSE.
7. Test labels are never used for feature selection, tuning, weight search, or
   model selection.

Outputs
-------
Tables:
    outputs/tables/31h_tuned_catboost_xgboost_split_summary.csv
    outputs/tables/31h_tuned_catboost_xgboost_validation_candidates.csv
    outputs/tables/31h_tuned_catboost_xgboost_test_results.csv
    outputs/tables/31h_tuned_catboost_xgboost_predictions.csv
    outputs/tables/31h_tuned_catboost_xgboost_summary.csv
    outputs/tables/31h_tuned_catboost_xgboost_selected_params.csv
    outputs/tables/31h_tuned_catboost_xgboost_decision.txt

Figures:
    outputs/figures/99_tuned_catboost_xgboost_metrics.png/pdf
    outputs/figures/100_tuned_catboost_xgboost_pooled_predicted_vs_actual.png/pdf
    outputs/figures/101_tuned_catboost_xgboost_weight_stability.png/pdf
    outputs/figures/102_tuned_catboost_xgboost_param_selection.png/pdf
"""

from __future__ import annotations

import ast
import math
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
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

try:
    from catboost import CatBoostRegressor
except Exception as e:
    raise ImportError(
        "CatBoost is required. Install in the project .venv if needed: "
        ".\\.venv\\Scripts\\python.exe -m pip install catboost\n"
        f"Original error: {repr(e)}"
    )

try:
    from xgboost import XGBRegressor
except Exception as e:
    raise ImportError(
        "XGBoost is required for Step 31H. Install in the project .venv: "
        ".\\.venv\\Scripts\\python.exe -m pip install xgboost\n"
        f"Original error: {repr(e)}"
    )


# ============================================================
# 1. Settings inherited from script 19 / 31F
# ============================================================

SEEDS = list(getattr(sf19, "SEEDS", [7, 21, 42, 66, 88]))
K_OPTIONS = list(getattr(sf19, "K_OPTIONS", ["all"]))

# We keep the candidate space deliberately focused on Step-31G's proven winner:
# CatBoost-XGBoost. The grid is not exhaustive; it is a grouped-generalization-
# oriented, conservative tuning set.
CATBOOST_CONFIGS = [
    {
        "Config_ID": "CB_script19_default",
        "Source": "script19_factory",
        "Params": None,
    },
    {
        "Config_ID": "CB_conservative_d4_lr003_l2_10",
        "Source": "tuned31h",
        "Params": dict(iterations=900, learning_rate=0.03, depth=4, l2_leaf_reg=10.0,
                       random_strength=3.0, bagging_temperature=0.5, rsm=0.85),
    },
    {
        "Config_ID": "CB_conservative_d4_lr002_l2_20",
        "Source": "tuned31h",
        "Params": dict(iterations=1200, learning_rate=0.02, depth=4, l2_leaf_reg=20.0,
                       random_strength=5.0, bagging_temperature=0.8, rsm=0.80),
    },
    {
        "Config_ID": "CB_shallow_d3_lr004_l2_10",
        "Source": "tuned31h",
        "Params": dict(iterations=800, learning_rate=0.04, depth=3, l2_leaf_reg=10.0,
                       random_strength=5.0, bagging_temperature=1.0, rsm=0.85),
    },
    {
        "Config_ID": "CB_mid_d5_lr003_l2_15",
        "Source": "tuned31h",
        "Params": dict(iterations=900, learning_rate=0.03, depth=5, l2_leaf_reg=15.0,
                       random_strength=4.0, bagging_temperature=0.6, rsm=0.85),
    },
    {
        "Config_ID": "CB_mid_d5_lr005_l2_20",
        "Source": "tuned31h",
        "Params": dict(iterations=650, learning_rate=0.05, depth=5, l2_leaf_reg=20.0,
                       random_strength=7.0, bagging_temperature=1.0, rsm=0.75),
    },
    {
        "Config_ID": "CB_d6_lr003_l2_30",
        "Source": "tuned31h",
        "Params": dict(iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=30.0,
                       random_strength=8.0, bagging_temperature=1.2, rsm=0.70),
    },
    {
        "Config_ID": "CB_d4_lr006_l2_6",
        "Source": "tuned31h",
        "Params": dict(iterations=500, learning_rate=0.06, depth=4, l2_leaf_reg=6.0,
                       random_strength=2.0, bagging_temperature=0.4, rsm=0.90),
    },
]

XGBOOST_CONFIGS = [
    {
        "Config_ID": "XGB_31g_default",
        "Source": "screen31g_default",
        "Params": dict(n_estimators=600, learning_rate=0.035, max_depth=3,
                       min_child_weight=3.0, subsample=0.80, colsample_bytree=0.80,
                       reg_alpha=0.05, reg_lambda=5.0, gamma=0.0),
    },
    {
        "Config_ID": "XGB_shallow_d2_lr003_reg10",
        "Source": "tuned31h",
        "Params": dict(n_estimators=850, learning_rate=0.03, max_depth=2,
                       min_child_weight=5.0, subsample=0.80, colsample_bytree=0.80,
                       reg_alpha=0.05, reg_lambda=10.0, gamma=0.0),
    },
    {
        "Config_ID": "XGB_d3_lr0025_reg10_gamma01",
        "Source": "tuned31h",
        "Params": dict(n_estimators=1000, learning_rate=0.025, max_depth=3,
                       min_child_weight=5.0, subsample=0.75, colsample_bytree=0.75,
                       reg_alpha=0.10, reg_lambda=10.0, gamma=0.10),
    },
    {
        "Config_ID": "XGB_d3_lr004_reg8",
        "Source": "tuned31h",
        "Params": dict(n_estimators=700, learning_rate=0.04, max_depth=3,
                       min_child_weight=4.0, subsample=0.85, colsample_bytree=0.80,
                       reg_alpha=0.05, reg_lambda=8.0, gamma=0.0),
    },
    {
        "Config_ID": "XGB_d4_lr003_reg15_gamma02",
        "Source": "tuned31h",
        "Params": dict(n_estimators=850, learning_rate=0.03, max_depth=4,
                       min_child_weight=6.0, subsample=0.75, colsample_bytree=0.70,
                       reg_alpha=0.15, reg_lambda=15.0, gamma=0.20),
    },
    {
        "Config_ID": "XGB_d2_lr005_reg6",
        "Source": "tuned31h",
        "Params": dict(n_estimators=550, learning_rate=0.05, max_depth=2,
                       min_child_weight=4.0, subsample=0.85, colsample_bytree=0.85,
                       reg_alpha=0.05, reg_lambda=6.0, gamma=0.0),
    },
    {
        "Config_ID": "XGB_d3_lr002_reg20_gamma03",
        "Source": "tuned31h",
        "Params": dict(n_estimators=1200, learning_rate=0.02, max_depth=3,
                       min_child_weight=8.0, subsample=0.70, colsample_bytree=0.70,
                       reg_alpha=0.20, reg_lambda=20.0, gamma=0.30),
    },
    {
        "Config_ID": "XGB_d4_lr004_reg12",
        "Source": "tuned31h",
        "Params": dict(n_estimators=650, learning_rate=0.04, max_depth=4,
                       min_child_weight=5.0, subsample=0.80, colsample_bytree=0.75,
                       reg_alpha=0.10, reg_lambda=12.0, gamma=0.10),
    },
]

# Only top validation candidates of each base model are crossed for fusion.
# This prevents an expensive and unstable all-by-all grid while preserving a
# validation-driven search.
TOP_N_PER_BASE_FOR_FUSION = 3

METHOD_ORDER = [
    "Tuned_CatBoost",
    "Tuned_XGBoost",
    "Tuned_CatBoost_XGBoost_Weighted_2Model",
]

METHOD_LABELS = {
    "Tuned_CatBoost": "Tuned CatBoost",
    "Tuned_XGBoost": "Tuned XGBoost",
    "Tuned_CatBoost_XGBoost_Weighted_2Model": "Tuned CatBoost-XGB 2Model",
}

COLORS = {
    "Tuned_CatBoost": "#9B72B0",
    "Tuned_XGBoost": "#D28C3D",
    "Tuned_CatBoost_XGBoost_Weighted_2Model": "#D09A6A",
    "CatBoost": "#9B72B0",
    "XGBoost": "#D28C3D",
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


def params_to_text(params):
    if params is None:
        return "script19_factory_default"
    return str({k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v) for k, v in params.items()})


# ============================================================
# 3. Model factories and fitting wrappers
# ============================================================

def make_catboost_model(config: dict, seed: int):
    if config["Source"] == "script19_factory":
        return sf19.MODEL_FACTORIES["CatBoost"](seed)

    p = dict(config["Params"])
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        **p,
    )


def make_xgboost_model(config: dict, seed: int):
    p = dict(config["Params"])
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
        **p,
    )


def build_pipeline(model):
    # Reuse script-19 preprocessing. This preserves categorical/numeric handling.
    # The model receives the preprocessed matrix.
    return model


def fit_predict_model(model_family: str, config: dict, X_train, y_train, X_target, seed: int, sample_weight=None):
    preprocessor, _, _ = sf19.build_preprocessor(X_train)

    if model_family == "CatBoost":
        model = make_catboost_model(config, seed)
    elif model_family == "XGBoost":
        model = make_xgboost_model(config, seed)
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    if sample_weight is None:
        pipe.fit(X_train, y_train)
    else:
        try:
            pipe.fit(X_train, y_train, model__sample_weight=np.asarray(sample_weight, dtype=float))
        except TypeError:
            log(f"[WARN] {model_family} config {config['Config_ID']} did not accept sample_weight; fitting without weights.")
            pipe.fit(X_train, y_train)

    pred = pipe.predict(X_target)
    return np.asarray(pred, dtype=float)


def select_columns_from_importance(imp_df: pd.DataFrame, X_mix: pd.DataFrame, selected_k):
    cols = sf19.select_top_features(imp_df, X_mix, selected_k)
    return [c for c in cols if c in X_mix.columns]


# ============================================================
# 4. Validation candidate evaluation and test prediction
# ============================================================

def evaluate_validation_candidates(X_train_k, y_train, X_valid_k, y_valid, seed, selected_k, selected_cols, sample_weight_train=None):
    rows = []
    valid_pred_cache = {}

    # CatBoost configs
    for config in CATBOOST_CONFIGS:
        cid = config["Config_ID"]
        log(f"    CatBoost validation config: {cid}")
        pred = fit_predict_model(
            model_family="CatBoost",
            config=config,
            X_train=X_train_k,
            y_train=y_train,
            X_target=X_valid_k,
            seed=seed,
            sample_weight=sample_weight_train,
        )
        valid_pred_cache[("CatBoost", cid)] = pred
        m = regression_metrics(y_valid, pred)
        rows.append({
            "Seed": seed,
            "Method": "Tuned_CatBoost",
            "Candidate_Type": "single",
            "Base_Models": "CatBoost",
            "Selected_K": selected_k,
            "Selected_Features": ";".join(selected_cols),
            "CatBoost_Config_ID": cid,
            "XGBoost_Config_ID": "",
            "CatBoost_Params": params_to_text(config["Params"]),
            "XGBoost_Params": "",
            "Validation_R2": m["R2"],
            "Validation_RMSE": m["RMSE"],
            "Validation_MAE": m["MAE"],
            "Validation_MSE": m["MSE"],
            "Validation_Bias": m["Bias"],
            "Weights": "",
        })

    # XGBoost configs
    for config in XGBOOST_CONFIGS:
        cid = config["Config_ID"]
        log(f"    XGBoost validation config: {cid}")
        pred = fit_predict_model(
            model_family="XGBoost",
            config=config,
            X_train=X_train_k,
            y_train=y_train,
            X_target=X_valid_k,
            seed=seed,
            sample_weight=sample_weight_train,
        )
        valid_pred_cache[("XGBoost", cid)] = pred
        m = regression_metrics(y_valid, pred)
        rows.append({
            "Seed": seed,
            "Method": "Tuned_XGBoost",
            "Candidate_Type": "single",
            "Base_Models": "XGBoost",
            "Selected_K": selected_k,
            "Selected_Features": ";".join(selected_cols),
            "CatBoost_Config_ID": "",
            "XGBoost_Config_ID": cid,
            "CatBoost_Params": "",
            "XGBoost_Params": params_to_text(config["Params"]),
            "Validation_R2": m["R2"],
            "Validation_RMSE": m["RMSE"],
            "Validation_MAE": m["MAE"],
            "Validation_MSE": m["MSE"],
            "Validation_Bias": m["Bias"],
            "Weights": "",
        })

    tmp = pd.DataFrame(rows)
    top_cb = (
        tmp[tmp["Method"] == "Tuned_CatBoost"]
        .sort_values("Validation_RMSE", ascending=True)
        .head(TOP_N_PER_BASE_FOR_FUSION)
    )
    top_xgb = (
        tmp[tmp["Method"] == "Tuned_XGBoost"]
        .sort_values("Validation_RMSE", ascending=True)
        .head(TOP_N_PER_BASE_FOR_FUSION)
    )

    # Cross only top validation candidates for fusion.
    for _, cb_row in top_cb.iterrows():
        for _, xgb_row in top_xgb.iterrows():
            cb_id = cb_row["CatBoost_Config_ID"]
            xgb_id = xgb_row["XGBoost_Config_ID"]
            pred_dict = {
                "CatBoost": valid_pred_cache[("CatBoost", cb_id)],
                "XGBoost": valid_pred_cache[("XGBoost", xgb_id)],
            }
            weights, _ = sf19.search_best_weights(pred_dict, y_valid, seed)
            pred_fused = sf19.apply_weights(pred_dict, weights)
            m = regression_metrics(y_valid, pred_fused)
            rows.append({
                "Seed": seed,
                "Method": "Tuned_CatBoost_XGBoost_Weighted_2Model",
                "Candidate_Type": "fusion",
                "Base_Models": "CatBoost;XGBoost",
                "Selected_K": selected_k,
                "Selected_Features": ";".join(selected_cols),
                "CatBoost_Config_ID": cb_id,
                "XGBoost_Config_ID": xgb_id,
                "CatBoost_Params": cb_row["CatBoost_Params"],
                "XGBoost_Params": xgb_row["XGBoost_Params"],
                "Validation_R2": m["R2"],
                "Validation_RMSE": m["RMSE"],
                "Validation_MAE": m["MAE"],
                "Validation_MSE": m["MSE"],
                "Validation_Bias": m["Bias"],
                "Weights": weight_dict_to_text(weights),
            })

    return pd.DataFrame(rows)


def get_config_by_id(family: str, config_id: str):
    configs = CATBOOST_CONFIGS if family == "CatBoost" else XGBOOST_CONFIGS
    for c in configs:
        if c["Config_ID"] == config_id:
            return c
    raise KeyError(f"Cannot find {family} config_id={config_id}")


def test_predict_from_candidate(best, X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full=None):
    method = best["Method"]

    if method == "Tuned_CatBoost":
        config = get_config_by_id("CatBoost", best["CatBoost_Config_ID"])
        return fit_predict_model("CatBoost", config, X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full)

    if method == "Tuned_XGBoost":
        config = get_config_by_id("XGBoost", best["XGBoost_Config_ID"])
        return fit_predict_model("XGBoost", config, X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full)

    if method == "Tuned_CatBoost_XGBoost_Weighted_2Model":
        cb_config = get_config_by_id("CatBoost", best["CatBoost_Config_ID"])
        xgb_config = get_config_by_id("XGBoost", best["XGBoost_Config_ID"])
        weights = parse_weight_text(best["Weights"])
        pred_dict = {
            "CatBoost": fit_predict_model("CatBoost", cb_config, X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full),
            "XGBoost": fit_predict_model("XGBoost", xgb_config, X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full),
        }
        return sf19.apply_weights(pred_dict, weights)

    raise ValueError(f"Unknown method: {method}")


def run_one_seed(X_mix: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int):
    log("\n" + "=" * 100)
    log(f"31H group-aware CatBoost-XGBoost tuning seed = {seed}")
    log("=" * 100)

    train_idx, valid_idx, test_idx, split_info = s31f.grouped_split_indices(groups, seed)
    log(f"Rows: train={split_info['n_train']}, valid={split_info['n_valid']}, test={split_info['n_test']}")
    log(f"Groups: train={split_info['n_train_groups']}, valid={split_info['n_valid_groups']}, test={split_info['n_test_groups']}")

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

    all_candidate_dfs = []

    for selected_k in K_OPTIONS:
        selected_cols = select_columns_from_importance(imp_df, X_mix, selected_k)
        if len(selected_cols) < 3:
            continue

        log("-" * 100)
        log(f"Seed={seed} | K={selected_k} | n_features={len(selected_cols)}")
        log("-" * 100)

        candidate_df_k = evaluate_validation_candidates(
            X_train_k=X_train[selected_cols].reset_index(drop=True),
            y_train=y_train,
            X_valid_k=X_valid[selected_cols].reset_index(drop=True),
            y_valid=y_valid,
            seed=seed,
            selected_k=selected_k,
            selected_cols=selected_cols,
            sample_weight_train=sample_weight_train,
        )
        all_candidate_dfs.append(candidate_df_k)

    candidate_df = pd.concat(all_candidate_dfs, ignore_index=True)

    test_rows = []
    pred_rows = []

    for method in METHOD_ORDER:
        sub = candidate_df[candidate_df["Method"] == method].copy()
        if sub.empty:
            continue

        best = sub.sort_values("Validation_RMSE", ascending=True).iloc[0]
        selected_cols = str(best["Selected_Features"]).split(";")
        selected_cols = [c for c in selected_cols if c in X_mix.columns]

        log("=" * 100)
        log(
            f"Final grouped test | seed={seed} | method={method} | "
            f"K={best['Selected_K']} | n_features={len(selected_cols)} | "
            f"CB={best['CatBoost_Config_ID']} | XGB={best['XGBoost_Config_ID']}"
        )
        log("=" * 100)

        pred_test = test_predict_from_candidate(
            best=best,
            X_train_full_k=X_train_full[selected_cols].reset_index(drop=True),
            y_train_full=y_train_full,
            X_test_k=X_test[selected_cols].reset_index(drop=True),
            seed=seed,
            sample_weight_train_full=sample_weight_train_full,
        )
        metrics = regression_metrics(y_test, pred_test)

        test_rows.append({
            "Seed": seed,
            "Method": method,
            "Candidate_Type": best["Candidate_Type"],
            "Base_Models": best["Base_Models"],
            "Selected_K": best["Selected_K"],
            "Selected_Features": ";".join(selected_cols),
            "CatBoost_Config_ID": best["CatBoost_Config_ID"],
            "XGBoost_Config_ID": best["XGBoost_Config_ID"],
            "CatBoost_Params": best["CatBoost_Params"],
            "XGBoost_Params": best["XGBoost_Params"],
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
                "Candidate_Type": best["Candidate_Type"],
                "Base_Models": best["Base_Models"],
                "Selected_K": best["Selected_K"],
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
# 5. Summary, reporting and plotting
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
        m = regression_metrics(sub["Observed"], sub["Predicted"])
        pooled_rows.append({
            "Method": method,
            "Pooled_R2": m["R2"],
            "Pooled_RMSE": m["RMSE"],
            "Pooled_MAE": m["MAE"],
            "Pooled_MSE": m["MSE"],
            "Pooled_Bias": m["Bias"],
            "n_pooled": len(sub),
            "n_literature_groups_pooled": sub["文献编号"].nunique(),
        })

    summary = summary.merge(pd.DataFrame(pooled_rows), on="Method", how="left")
    summary["Method"] = pd.Categorical(summary["Method"], categories=METHOD_ORDER, ordered=True)
    return summary.sort_values("Method").reset_index(drop=True)


def collect_selected_params(test_df: pd.DataFrame):
    cols = [
        "Seed", "Method", "Selected_K", "CatBoost_Config_ID", "XGBoost_Config_ID",
        "CatBoost_Params", "XGBoost_Params", "Validation_RMSE", "Test_RMSE", "Weights"
    ]
    return test_df[[c for c in cols if c in test_df.columns]].copy()


def write_decision_report(summary_df, split_df, selected_params_df):
    path = TABLE_DIR / "31h_tuned_catboost_xgboost_decision.txt"
    ranking = summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE", "Pooled_R2"], ascending=[True, True, False]).reset_index(drop=True)
    best = ranking.iloc[0]

    lines = []
    lines.append("=" * 100)
    lines.append("Step 31H group-aware CatBoost-XGBoost tuning report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Tune the Step-31G best complementary route, CatBoost-XGBoost weighted 2Model, under balanced literature-level grouped validation.")
    lines.append("")
    lines.append("Strict workflow reused:")
    lines.append("- Step-31B mechanism-aware descriptors")
    lines.append("- Step-31F balanced grouped split")
    lines.append("- Literature-group-weighted training")
    lines.append("- Script-19 feature selection/preprocessing where applicable")
    lines.append("- Validation-RMSE-selected hyperparameters and non-negative fusion weights")
    lines.append("")
    lines.append("Tuning scope:")
    lines.append(f"CatBoost configs: {len(CATBOOST_CONFIGS)}")
    lines.append(f"XGBoost configs : {len(XGBOOST_CONFIGS)}")
    lines.append(f"Top configs crossed for fusion per base model: {TOP_N_PER_BASE_FOR_FUSION}")
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
    lines.append("Current Step-31H best candidate:")
    lines.append(f"{best['Method']} | Pooled_R2={best['Pooled_R2']:.4f}, Pooled_RMSE={best['Pooled_RMSE']:.4f}, Pooled_MAE={best['Pooled_MAE']:.4f}")
    lines.append("")
    lines.append("Selected parameter/configuration records:")
    lines.append(selected_params_df.to_string(index=False))
    lines.append("")
    lines.append("Decision guide:")
    lines.append("1. If tuned CatBoost-XGBoost improves over Step 31G, keep it as the grouped-generalization main candidate.")
    lines.append("2. If tuning does not improve, keep Step 31G CatBoost-XGBoost as the grouped model and move to target transformation / applicability domain.")
    lines.append("3. Do not compare against random-split R2 as the same task; random split measures literature-internal interpolation.")
    lines.append("=" * 100)
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[TEXT] {path}")


def plot_metrics(summary_df: pd.DataFrame):
    data = summary_df.copy().sort_values("Pooled_RMSE", ascending=True).reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6))
    specs = [
        ("Pooled_R2", "Pooled $R^2$", "{:.3f}", "max"),
        ("Pooled_RMSE", "Pooled RMSE", "{:.2f}", "min"),
        ("Pooled_MAE", "Pooled MAE", "{:.2f}", "min"),
    ]
    for ax, (col, ylabel, fmt, rule) in zip(axes, specs):
        vals = data[col].astype(float).values
        methods = data["Method"].astype(str).tolist()
        x = np.arange(len(data))
        colors = [COLORS.get(m, COLORS["gray"]) for m in methods]
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.45, alpha=0.90)
        best_idx = int(np.nanargmax(vals)) if rule == "max" else int(np.nanargmin(vals))
        bars[best_idx].set_edgecolor(COLORS["red"])
        bars[best_idx].set_linewidth(1.4)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        ymin, ymax = np.nanmin(vals), np.nanmax(vals)
        if col == "Pooled_R2":
            lo = min(0, ymin - 0.05)
            hi = max(0.12, ymax + 0.06)
            ax.set_ylim(lo, hi)
            ax.axhline(0, color="#444444", linewidth=0.8)
            offset = (hi - lo) * 0.025
        else:
            ax.set_ylim(0, ymax * 1.20)
            offset = ax.get_ylim()[1] * 0.025
        for xi, v in zip(x, vals):
            ax.text(xi, v + offset, fmt.format(v), ha="center", va="bottom", fontsize=7)
    fig.suptitle("Group-aware tuning of CatBoost-XGBoost under balanced grouped validation", fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, 0.01, "Mechanism-aware descriptors + literature-group-weighted training; hyperparameters selected by validation RMSE.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0.02, 0.06, 1, 0.92])
    save_fig(fig, "99_tuned_catboost_xgboost_metrics.png")


def plot_pooled_prediction(pred_df: pd.DataFrame, summary_df: pd.DataFrame):
    methods = list(
        summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE"], ascending=[True, True])["Method"].astype(str)
    )
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        sub = pred_df[pred_df["Method"] == method].copy()
        color = COLORS.get(method, COLORS["gray"])
        ax.scatter(sub["Observed"], sub["Predicted"], s=18, color=color, alpha=0.68, edgecolor="none")
        ax.plot([0, 100], [0, 100], color=COLORS["red"], linestyle=(0, (4, 3)), linewidth=1.0)
        m = regression_metrics(sub["Observed"], sub["Predicted"])
        ax.text(0.06, 0.92, f"$R^2$ = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nMAE = {m['MAE']:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, boxstyle="round,pad=0.3"))
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Observed F removal (%)")
        ax.set_ylabel("Predicted F removal (%)")
        ax.grid(color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    fig.suptitle("Pooled predicted-vs-observed plots after group-aware tuning", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.92])
    save_fig(fig, "100_tuned_catboost_xgboost_pooled_predicted_vs_actual.png")


def plot_weight_stability(test_df: pd.DataFrame):
    sub = test_df[test_df["Method"] == "Tuned_CatBoost_XGBoost_Weighted_2Model"].copy()
    if sub.empty:
        return
    rows = []
    for _, r in sub.iterrows():
        weights = parse_weight_text(r["Weights"])
        for base_model, weight in weights.items():
            rows.append({"Seed": r["Seed"], "BaseModel": base_model, "Weight": float(weight)})
    wdf = pd.DataFrame(rows)
    if wdf.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for base_model, sub_m in wdf.groupby("BaseModel"):
        sub_m = sub_m.sort_values("Seed")
        ax.plot(sub_m["Seed"], sub_m["Weight"], marker="o", linewidth=1.6, markersize=4,
                color=COLORS.get(base_model, COLORS["gray"]), label=base_model)
        for _, rr in sub_m.iterrows():
            ax.text(rr["Seed"], rr["Weight"] + 0.025, f"{rr['Weight']:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Validation-optimized fusion weight")
    ax.set_ylim(-0.05, 1.12)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False)
    ax.set_title("Tuned CatBoost-XGBoost fusion weight stability")
    fig.tight_layout()
    save_fig(fig, "101_tuned_catboost_xgboost_weight_stability.png")


def plot_param_selection(selected_params_df: pd.DataFrame):
    if selected_params_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    # CatBoost config counts
    cb = selected_params_df[selected_params_df["CatBoost_Config_ID"].astype(str) != ""].copy()
    cb_counts = cb["CatBoost_Config_ID"].value_counts().sort_values(ascending=True)
    axes[0].barh(range(len(cb_counts)), cb_counts.values, color=COLORS["CatBoost"], edgecolor="black", linewidth=0.45, alpha=0.9)
    axes[0].set_yticks(range(len(cb_counts)))
    axes[0].set_yticklabels(cb_counts.index)
    axes[0].set_xlabel("Selection count")
    axes[0].set_title("Selected CatBoost configs")
    axes[0].grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.8)

    xgb = selected_params_df[selected_params_df["XGBoost_Config_ID"].astype(str) != ""].copy()
    xgb_counts = xgb["XGBoost_Config_ID"].value_counts().sort_values(ascending=True)
    axes[1].barh(range(len(xgb_counts)), xgb_counts.values, color=COLORS["XGBoost"], edgecolor="black", linewidth=0.45, alpha=0.9)
    axes[1].set_yticks(range(len(xgb_counts)))
    axes[1].set_yticklabels(xgb_counts.index)
    axes[1].set_xlabel("Selection count")
    axes[1].set_title("Selected XGBoost configs")
    axes[1].grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.8)

    fig.suptitle("Group-aware selected hyperparameter configurations", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.92])
    save_fig(fig, "102_tuned_catboost_xgboost_param_selection.png")


# ============================================================
# 6. Main
# ============================================================

def main():
    log("\n" + "=" * 100)
    log("Step 31H: group-aware CatBoost-XGBoost tuning")
    log("=" * 100)
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"SEEDS = {SEEDS}")
    log(f"K_OPTIONS = {K_OPTIONS}")
    log(f"CatBoost configs = {len(CATBOOST_CONFIGS)}")
    log(f"XGBoost configs = {len(XGBOOST_CONFIGS)}")
    log("=" * 100)

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
    selected_params_df = collect_selected_params(test_df)

    out_tables = {
        "31h_tuned_catboost_xgboost_split_summary.csv": split_df,
        "31h_tuned_catboost_xgboost_validation_candidates.csv": candidate_df,
        "31h_tuned_catboost_xgboost_test_results.csv": test_df,
        "31h_tuned_catboost_xgboost_predictions.csv": pred_df,
        "31h_tuned_catboost_xgboost_summary.csv": summary_df,
        "31h_tuned_catboost_xgboost_selected_params.csv": selected_params_df,
    }

    for filename, df in out_tables.items():
        path = TABLE_DIR / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        log(f"[TABLE] {path}")

    log("\n=== Step 31H group-aware tuning summary ===")
    print(summary_df.to_string(index=False))

    write_decision_report(summary_df, split_df, selected_params_df)

    log("\nGenerating figures...")
    plot_metrics(summary_df)
    plot_pooled_prediction(pred_df, summary_df)
    plot_weight_stability(test_df)
    plot_param_selection(selected_params_df)

    log("\n" + "=" * 100)
    log("[DONE] Step 31H group-aware CatBoost-XGBoost tuning completed.")
    log("=" * 100)
    log("Please send:")
    log("1. outputs/tables/31h_tuned_catboost_xgboost_summary.csv")
    log("2. outputs/tables/31h_tuned_catboost_xgboost_decision.txt")
    log("3. outputs/tables/31h_tuned_catboost_xgboost_selected_params.csv")
    log("4. outputs/figures/99_tuned_catboost_xgboost_metrics.png")
    log("5. outputs/figures/101_tuned_catboost_xgboost_weight_stability.png")
    log("=" * 100)


if __name__ == "__main__":
    main()
