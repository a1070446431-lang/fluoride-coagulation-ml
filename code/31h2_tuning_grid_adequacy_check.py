# -*- coding: utf-8 -*-
"""
31h2_tuning_grid_adequacy_check.py

Purpose
-------
Step 31H2 checks whether Step 31H failed because the tuning grid was too
conservative rather than because CatBoost-XGBoost tuning has no value.

This is not a simplified workflow. It reuses the real project scripts:
    - script 19: feature engineering / preprocessing / feature selection utilities
    - script 31F: mechanism-aware descriptors, balanced grouped split,
      literature-group sample weights
    - script 31G: exact CatBoost/XGBoost baseline route and validation-RMSE fusion logic

Strict rules
------------
1. 文献编号 is used only for grouped splitting.
2. 文献编号 is never used as an input feature.
3. Step-31B mechanism-aware representation is reused.
4. Step-31F balanced grouped split is reused.
5. Literature-group-weighted training is reused.
6. Feature ranking is performed only on grouped training data.
7. Hyperparameters and fusion weights are selected only by validation RMSE.
8. Test labels are never used for feature selection, tuning, weight search, or model selection.

Outputs
-------
Tables:
    outputs/tables/31h2_tuning_grid_adequacy_split_summary.csv
    outputs/tables/31h2_tuning_grid_adequacy_validation_candidates.csv
    outputs/tables/31h2_tuning_grid_adequacy_test_results.csv
    outputs/tables/31h2_tuning_grid_adequacy_predictions.csv
    outputs/tables/31h2_tuning_grid_adequacy_summary.csv
    outputs/tables/31h2_tuning_grid_adequacy_selected_params.csv
    outputs/tables/31h2_tuning_grid_adequacy_config_catalog.csv
    outputs/tables/31h2_tuning_grid_adequacy_decision.txt

Figures:
    outputs/figures/106_31h2_tuning_grid_adequacy_metrics.png/pdf
    outputs/figures/107_31h2_tuning_grid_adequacy_pooled_predicted_vs_actual.png/pdf
    outputs/figures/108_31h2_tuning_grid_adequacy_weight_stability.png/pdf
    outputs/figures/109_31h2_tuning_grid_adequacy_param_selection.png/pdf
    outputs/figures/110_31h2_validation_test_rmse_transfer.png/pdf
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


def log(msg: str):
    print(msg, flush=True)


def find_file_upwards(start_path: Path, filename: str) -> Path:
    start_path = start_path.resolve()
    search_roots = [start_path.parent] + list(start_path.parents)
    for root in search_roots:
        for p in [root / filename, root / "src" / filename]:
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


s31g = load_module_from_file("31g_expanded_model_complementarity_screening.py", "s31g_real")
s31f = s31g.s31f
sf19 = s31g.sf19

try:
    s31h = load_module_from_file("31h_group_aware_catboost_xgboost_tuning.py", "s31h_real")
except Exception as e:
    s31h = None
    log(f"[WARN] Could not import 31H reference script. 31H2 can still run. Error: {repr(e)}")

PROJECT_ROOT = Path(s31g.PROJECT_ROOT)
FIG_DIR = Path(s31g.FIG_DIR)
TABLE_DIR = Path(s31g.TABLE_DIR)
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
        "XGBoost is required. Install in the project .venv: "
        ".\\.venv\\Scripts\\python.exe -m pip install xgboost\n"
        f"Original error: {repr(e)}"
    )


SEEDS = list(getattr(sf19, "SEEDS", [7, 21, 42, 66, 88]))
K_OPTIONS = list(getattr(sf19, "K_OPTIONS", ["all"]))
TOP_N_PER_BASE_FOR_FUSION = 5

BASELINE_METHODS = [
    "31G_Baseline_CatBoost",
    "31G_Baseline_XGBoost",
    "31G_Baseline_CatBoost_XGBoost_Weighted_2Model",
]

TUNED_METHODS = [
    "31H2_Tuned_CatBoost",
    "31H2_Tuned_XGBoost",
    "31H2_Tuned_CatBoost_XGBoost_Weighted_2Model",
]

METHOD_ORDER = BASELINE_METHODS + TUNED_METHODS

METHOD_LABELS = {
    "31G_Baseline_CatBoost": "31G baseline CatBoost",
    "31G_Baseline_XGBoost": "31G baseline XGBoost",
    "31G_Baseline_CatBoost_XGBoost_Weighted_2Model": "31G baseline CB-XGB",
    "31H2_Tuned_CatBoost": "31H2 tuned CatBoost",
    "31H2_Tuned_XGBoost": "31H2 tuned XGBoost",
    "31H2_Tuned_CatBoost_XGBoost_Weighted_2Model": "31H2 tuned CB-XGB",
}

COLORS = {
    "31G_Baseline_CatBoost": "#9B72B0",
    "31G_Baseline_XGBoost": "#D28C3D",
    "31G_Baseline_CatBoost_XGBoost_Weighted_2Model": "#B184C6",
    "31H2_Tuned_CatBoost": "#6F4E9A",
    "31H2_Tuned_XGBoost": "#B56F23",
    "31H2_Tuned_CatBoost_XGBoost_Weighted_2Model": "#8C6BB1",
    "CatBoost": "#9B72B0",
    "XGBoost": "#D28C3D",
    "grid": "#E6E6E6",
    "red": "#FF0000",
    "gray": "#666666",
    "dark": "#222222",
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
# Structured 31H2 configuration grid
# ============================================================

CATBOOST_CONFIGS = [
    {"Config_ID": "CB_script19_default", "Grid_Block": "31G_baseline", "Params": None},

    {"Config_ID": "CB_cons_d3_lr004_l2_10_rs5", "Grid_Block": "conservative",
     "Params": dict(iterations=800, learning_rate=0.04, depth=3, l2_leaf_reg=10.0,
                    random_strength=5.0, bagging_temperature=1.0, rsm=0.85)},
    {"Config_ID": "CB_cons_d4_lr003_l2_10_rs3", "Grid_Block": "conservative",
     "Params": dict(iterations=900, learning_rate=0.03, depth=4, l2_leaf_reg=10.0,
                    random_strength=3.0, bagging_temperature=0.5, rsm=0.85)},
    {"Config_ID": "CB_cons_d4_lr002_l2_20_rs5", "Grid_Block": "conservative",
     "Params": dict(iterations=1200, learning_rate=0.02, depth=4, l2_leaf_reg=20.0,
                    random_strength=5.0, bagging_temperature=0.8, rsm=0.80)},
    {"Config_ID": "CB_cons_d5_lr003_l2_15_rs4", "Grid_Block": "conservative",
     "Params": dict(iterations=900, learning_rate=0.03, depth=5, l2_leaf_reg=15.0,
                    random_strength=4.0, bagging_temperature=0.6, rsm=0.85)},

    {"Config_ID": "CB_near_d5_lr005_l2_6_rs2", "Grid_Block": "31G_neighborhood",
     "Params": dict(iterations=700, learning_rate=0.05, depth=5, l2_leaf_reg=6.0,
                    random_strength=2.0, bagging_temperature=0.5, rsm=0.90)},
    {"Config_ID": "CB_near_d6_lr004_l2_6_rs3", "Grid_Block": "31G_neighborhood",
     "Params": dict(iterations=850, learning_rate=0.04, depth=6, l2_leaf_reg=6.0,
                    random_strength=3.0, bagging_temperature=0.8, rsm=0.85)},
    {"Config_ID": "CB_near_d6_lr005_l2_3_rs2", "Grid_Block": "31G_neighborhood",
     "Params": dict(iterations=650, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                    random_strength=2.0, bagging_temperature=0.4, rsm=0.90)},

    {"Config_ID": "CB_mid_d5_lr007_l2_3_rs1", "Grid_Block": "medium_complexity",
     "Params": dict(iterations=550, learning_rate=0.07, depth=5, l2_leaf_reg=3.0,
                    random_strength=1.0, bagging_temperature=0.3, rsm=0.95)},
    {"Config_ID": "CB_mid_d6_lr006_l2_3_rs1", "Grid_Block": "medium_complexity",
     "Params": dict(iterations=600, learning_rate=0.06, depth=6, l2_leaf_reg=3.0,
                    random_strength=1.0, bagging_temperature=0.3, rsm=0.95)},
    {"Config_ID": "CB_mid_d7_lr004_l2_6_rs2", "Grid_Block": "medium_complexity",
     "Params": dict(iterations=750, learning_rate=0.04, depth=7, l2_leaf_reg=6.0,
                    random_strength=2.0, bagging_temperature=0.7, rsm=0.85)},
    {"Config_ID": "CB_mid_d6_lr008_l2_6_rs2", "Grid_Block": "medium_complexity",
     "Params": dict(iterations=450, learning_rate=0.08, depth=6, l2_leaf_reg=6.0,
                    random_strength=2.0, bagging_temperature=0.5, rsm=0.90)},

    {"Config_ID": "CB_expr_d7_lr006_l2_3_rs1", "Grid_Block": "controlled_expressive",
     "Params": dict(iterations=550, learning_rate=0.06, depth=7, l2_leaf_reg=3.0,
                    random_strength=1.0, bagging_temperature=0.5, rsm=0.90)},
    {"Config_ID": "CB_expr_d8_lr004_l2_6_rs2", "Grid_Block": "controlled_expressive",
     "Params": dict(iterations=650, learning_rate=0.04, depth=8, l2_leaf_reg=6.0,
                    random_strength=2.0, bagging_temperature=0.8, rsm=0.80)},
]

XGBOOST_CONFIGS = [
    {"Config_ID": "XGB_31g_default", "Grid_Block": "31G_baseline",
     "Params": dict(n_estimators=600, learning_rate=0.035, max_depth=3,
                    min_child_weight=3.0, subsample=0.80, colsample_bytree=0.80,
                    reg_alpha=0.05, reg_lambda=5.0, gamma=0.0)},

    {"Config_ID": "XGB_cons_d2_lr003_lam10", "Grid_Block": "conservative",
     "Params": dict(n_estimators=850, learning_rate=0.03, max_depth=2,
                    min_child_weight=5.0, subsample=0.80, colsample_bytree=0.80,
                    reg_alpha=0.05, reg_lambda=10.0, gamma=0.0)},
    {"Config_ID": "XGB_cons_d3_lr0025_lam10_g01", "Grid_Block": "conservative",
     "Params": dict(n_estimators=1000, learning_rate=0.025, max_depth=3,
                    min_child_weight=5.0, subsample=0.75, colsample_bytree=0.75,
                    reg_alpha=0.10, reg_lambda=10.0, gamma=0.10)},
    {"Config_ID": "XGB_cons_d3_lr002_lam20_g03", "Grid_Block": "conservative",
     "Params": dict(n_estimators=1200, learning_rate=0.02, max_depth=3,
                    min_child_weight=8.0, subsample=0.70, colsample_bytree=0.70,
                    reg_alpha=0.20, reg_lambda=20.0, gamma=0.30)},

    {"Config_ID": "XGB_near_d3_lr004_lam5", "Grid_Block": "31G_neighborhood",
     "Params": dict(n_estimators=700, learning_rate=0.04, max_depth=3,
                    min_child_weight=3.0, subsample=0.85, colsample_bytree=0.85,
                    reg_alpha=0.03, reg_lambda=5.0, gamma=0.0)},
    {"Config_ID": "XGB_near_d4_lr0035_lam5", "Grid_Block": "31G_neighborhood",
     "Params": dict(n_estimators=750, learning_rate=0.035, max_depth=4,
                    min_child_weight=3.0, subsample=0.85, colsample_bytree=0.80,
                    reg_alpha=0.03, reg_lambda=5.0, gamma=0.0)},
    {"Config_ID": "XGB_near_d4_lr004_lam8_g01", "Grid_Block": "31G_neighborhood",
     "Params": dict(n_estimators=650, learning_rate=0.04, max_depth=4,
                    min_child_weight=4.0, subsample=0.80, colsample_bytree=0.80,
                    reg_alpha=0.05, reg_lambda=8.0, gamma=0.10)},

    {"Config_ID": "XGB_mid_d4_lr006_lam3", "Grid_Block": "medium_complexity",
     "Params": dict(n_estimators=500, learning_rate=0.06, max_depth=4,
                    min_child_weight=2.0, subsample=0.90, colsample_bytree=0.90,
                    reg_alpha=0.02, reg_lambda=3.0, gamma=0.0)},
    {"Config_ID": "XGB_mid_d5_lr004_lam5", "Grid_Block": "medium_complexity",
     "Params": dict(n_estimators=650, learning_rate=0.04, max_depth=5,
                    min_child_weight=3.0, subsample=0.85, colsample_bytree=0.85,
                    reg_alpha=0.03, reg_lambda=5.0, gamma=0.05)},
    {"Config_ID": "XGB_mid_d5_lr006_lam3", "Grid_Block": "medium_complexity",
     "Params": dict(n_estimators=450, learning_rate=0.06, max_depth=5,
                    min_child_weight=3.0, subsample=0.85, colsample_bytree=0.85,
                    reg_alpha=0.02, reg_lambda=3.0, gamma=0.0)},
    {"Config_ID": "XGB_mid_d4_lr008_lam3", "Grid_Block": "medium_complexity",
     "Params": dict(n_estimators=350, learning_rate=0.08, max_depth=4,
                    min_child_weight=2.0, subsample=0.90, colsample_bytree=0.90,
                    reg_alpha=0.02, reg_lambda=3.0, gamma=0.0)},

    {"Config_ID": "XGB_expr_d6_lr0035_lam5", "Grid_Block": "controlled_expressive",
     "Params": dict(n_estimators=650, learning_rate=0.035, max_depth=6,
                    min_child_weight=3.0, subsample=0.80, colsample_bytree=0.80,
                    reg_alpha=0.05, reg_lambda=5.0, gamma=0.05)},
    {"Config_ID": "XGB_expr_d6_lr005_lam3", "Grid_Block": "controlled_expressive",
     "Params": dict(n_estimators=450, learning_rate=0.05, max_depth=6,
                    min_child_weight=2.0, subsample=0.85, colsample_bytree=0.85,
                    reg_alpha=0.03, reg_lambda=3.0, gamma=0.0)},
]


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
    if not vals:
        return ""

    def sort_key(s):
        s = str(s)
        if s.lower() == "raw":
            return (-1, 0)
        if s.lower() == "all":
            return (999999, 0)
        try:
            return (int(float(s)), 0)
        except Exception:
            return (999998, s)

    return ",".join(sorted(set(vals), key=sort_key))


def params_to_text(params):
    if params is None:
        return "script19_factory_default"
    clean = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        else:
            clean[k] = v
    return str(clean)


def get_config_by_id(family: str, config_id: str):
    configs = CATBOOST_CONFIGS if family == "CatBoost" else XGBOOST_CONFIGS
    for c in configs:
        if c["Config_ID"] == config_id:
            return c
    raise KeyError(f"Cannot find {family} config_id={config_id}")


def selected_cols_from_row(row, X_mix):
    cols = str(row["Selected_Features"]).split(";")
    return [c for c in cols if c in X_mix.columns]


def make_catboost_model(config: dict, seed: int):
    if config["Config_ID"] == "CB_script19_default" or config["Params"] is None:
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


def fit_predict_config(model_family: str, config: dict, X_train, y_train, X_target, seed: int, sample_weight=None):
    preprocessor, _, _ = sf19.build_preprocessor(X_train)
    if model_family == "CatBoost":
        model = make_catboost_model(config, seed)
    elif model_family == "XGBoost":
        model = make_xgboost_model(config, seed)
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    if sample_weight is None:
        pipe.fit(X_train, y_train)
    else:
        try:
            pipe.fit(X_train, y_train, model__sample_weight=np.asarray(sample_weight, dtype=float))
        except TypeError:
            log(f"[WARN] {model_family} config {config['Config_ID']} did not accept sample_weight; fitting without weights.")
            pipe.fit(X_train, y_train)
    return np.asarray(pipe.predict(X_target), dtype=float)


def fit_predict_31g_baseline(model_name: str, X_train, y_train, X_target, seed: int, sample_weight=None):
    return s31g.fit_predict_model(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_target=X_target,
        seed=seed,
        sample_weight=sample_weight,
    )


def select_columns_from_importance(imp_df: pd.DataFrame, X_mix: pd.DataFrame, selected_k):
    return s31g.select_columns_from_importance(imp_df, X_mix, selected_k)


def evaluate_validation_candidates(X_train_k, y_train, X_valid_k, y_valid, seed, selected_k, selected_cols, sample_weight_train=None):
    rows = []
    pred_cache = {}

    for model_name, method in [("CatBoost", "31G_Baseline_CatBoost"), ("XGBoost", "31G_Baseline_XGBoost")]:
        log(f"    31G baseline validation model: {model_name}")
        pred = fit_predict_31g_baseline(model_name, X_train_k, y_train, X_valid_k, seed, sample_weight_train)
        pred_cache[("31G", model_name, "baseline")] = pred
        m = regression_metrics(y_valid, pred)
        rows.append({
            "Seed": seed, "Method": method, "Candidate_Type": "baseline_single", "Base_Models": model_name,
            "Selected_K": selected_k, "Selected_Features": ";".join(selected_cols),
            "CatBoost_Config_ID": "CB_script19_default" if model_name == "CatBoost" else "",
            "XGBoost_Config_ID": "XGB_31g_default" if model_name == "XGBoost" else "",
            "CatBoost_Grid_Block": "31G_baseline" if model_name == "CatBoost" else "",
            "XGBoost_Grid_Block": "31G_baseline" if model_name == "XGBoost" else "",
            "CatBoost_Params": "script19_factory_default" if model_name == "CatBoost" else "",
            "XGBoost_Params": params_to_text(XGBOOST_CONFIGS[0]["Params"]) if model_name == "XGBoost" else "",
            "Validation_R2": m["R2"], "Validation_RMSE": m["RMSE"], "Validation_MAE": m["MAE"],
            "Validation_MSE": m["MSE"], "Validation_Bias": m["Bias"], "Weights": "",
        })

    baseline_pred_dict = {
        "CatBoost": pred_cache[("31G", "CatBoost", "baseline")],
        "XGBoost": pred_cache[("31G", "XGBoost", "baseline")],
    }
    weights, _ = sf19.search_best_weights(baseline_pred_dict, y_valid, seed)
    pred_fused = sf19.apply_weights(baseline_pred_dict, weights)
    m = regression_metrics(y_valid, pred_fused)
    rows.append({
        "Seed": seed, "Method": "31G_Baseline_CatBoost_XGBoost_Weighted_2Model", "Candidate_Type": "baseline_fusion",
        "Base_Models": "CatBoost;XGBoost", "Selected_K": selected_k, "Selected_Features": ";".join(selected_cols),
        "CatBoost_Config_ID": "CB_script19_default", "XGBoost_Config_ID": "XGB_31g_default",
        "CatBoost_Grid_Block": "31G_baseline", "XGBoost_Grid_Block": "31G_baseline",
        "CatBoost_Params": "script19_factory_default", "XGBoost_Params": params_to_text(XGBOOST_CONFIGS[0]["Params"]),
        "Validation_R2": m["R2"], "Validation_RMSE": m["RMSE"], "Validation_MAE": m["MAE"],
        "Validation_MSE": m["MSE"], "Validation_Bias": m["Bias"], "Weights": weight_dict_to_text(weights),
    })

    for config in CATBOOST_CONFIGS:
        cid = config["Config_ID"]
        log(f"    31H2 CatBoost validation config: {cid}")
        pred = fit_predict_config("CatBoost", config, X_train_k, y_train, X_valid_k, seed, sample_weight_train)
        pred_cache[("H2", "CatBoost", cid)] = pred
        m = regression_metrics(y_valid, pred)
        rows.append({
            "Seed": seed, "Method": "31H2_Tuned_CatBoost", "Candidate_Type": "tuned_single", "Base_Models": "CatBoost",
            "Selected_K": selected_k, "Selected_Features": ";".join(selected_cols),
            "CatBoost_Config_ID": cid, "XGBoost_Config_ID": "",
            "CatBoost_Grid_Block": config["Grid_Block"], "XGBoost_Grid_Block": "",
            "CatBoost_Params": params_to_text(config["Params"]), "XGBoost_Params": "",
            "Validation_R2": m["R2"], "Validation_RMSE": m["RMSE"], "Validation_MAE": m["MAE"],
            "Validation_MSE": m["MSE"], "Validation_Bias": m["Bias"], "Weights": "",
        })

    for config in XGBOOST_CONFIGS:
        cid = config["Config_ID"]
        log(f"    31H2 XGBoost validation config: {cid}")
        pred = fit_predict_config("XGBoost", config, X_train_k, y_train, X_valid_k, seed, sample_weight_train)
        pred_cache[("H2", "XGBoost", cid)] = pred
        m = regression_metrics(y_valid, pred)
        rows.append({
            "Seed": seed, "Method": "31H2_Tuned_XGBoost", "Candidate_Type": "tuned_single", "Base_Models": "XGBoost",
            "Selected_K": selected_k, "Selected_Features": ";".join(selected_cols),
            "CatBoost_Config_ID": "", "XGBoost_Config_ID": cid,
            "CatBoost_Grid_Block": "", "XGBoost_Grid_Block": config["Grid_Block"],
            "CatBoost_Params": "", "XGBoost_Params": params_to_text(config["Params"]),
            "Validation_R2": m["R2"], "Validation_RMSE": m["RMSE"], "Validation_MAE": m["MAE"],
            "Validation_MSE": m["MSE"], "Validation_Bias": m["Bias"], "Weights": "",
        })

    tmp = pd.DataFrame(rows)
    top_cb = tmp[tmp["Method"] == "31H2_Tuned_CatBoost"].sort_values("Validation_RMSE", ascending=True).head(TOP_N_PER_BASE_FOR_FUSION)
    top_xgb = tmp[tmp["Method"] == "31H2_Tuned_XGBoost"].sort_values("Validation_RMSE", ascending=True).head(TOP_N_PER_BASE_FOR_FUSION)

    if not (top_cb["CatBoost_Config_ID"] == "CB_script19_default").any():
        default_cb = tmp[(tmp["Method"] == "31H2_Tuned_CatBoost") & (tmp["CatBoost_Config_ID"] == "CB_script19_default")]
        top_cb = pd.concat([top_cb, default_cb], ignore_index=True).drop_duplicates(subset=["CatBoost_Config_ID"], keep="first")
    if not (top_xgb["XGBoost_Config_ID"] == "XGB_31g_default").any():
        default_xgb = tmp[(tmp["Method"] == "31H2_Tuned_XGBoost") & (tmp["XGBoost_Config_ID"] == "XGB_31g_default")]
        top_xgb = pd.concat([top_xgb, default_xgb], ignore_index=True).drop_duplicates(subset=["XGBoost_Config_ID"], keep="first")

    for _, cb_row in top_cb.iterrows():
        for _, xgb_row in top_xgb.iterrows():
            cb_id = cb_row["CatBoost_Config_ID"]
            xgb_id = xgb_row["XGBoost_Config_ID"]
            pred_dict = {"CatBoost": pred_cache[("H2", "CatBoost", cb_id)], "XGBoost": pred_cache[("H2", "XGBoost", xgb_id)]}
            weights, _ = sf19.search_best_weights(pred_dict, y_valid, seed)
            pred_fused = sf19.apply_weights(pred_dict, weights)
            m = regression_metrics(y_valid, pred_fused)
            rows.append({
                "Seed": seed, "Method": "31H2_Tuned_CatBoost_XGBoost_Weighted_2Model", "Candidate_Type": "tuned_fusion",
                "Base_Models": "CatBoost;XGBoost", "Selected_K": selected_k, "Selected_Features": ";".join(selected_cols),
                "CatBoost_Config_ID": cb_id, "XGBoost_Config_ID": xgb_id,
                "CatBoost_Grid_Block": cb_row["CatBoost_Grid_Block"], "XGBoost_Grid_Block": xgb_row["XGBoost_Grid_Block"],
                "CatBoost_Params": cb_row["CatBoost_Params"], "XGBoost_Params": xgb_row["XGBoost_Params"],
                "Validation_R2": m["R2"], "Validation_RMSE": m["RMSE"], "Validation_MAE": m["MAE"],
                "Validation_MSE": m["MSE"], "Validation_Bias": m["Bias"], "Weights": weight_dict_to_text(weights),
            })

    return pd.DataFrame(rows)


def test_predict_from_candidate(best, X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full=None):
    method = best["Method"]

    if method == "31G_Baseline_CatBoost":
        return fit_predict_31g_baseline("CatBoost", X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full)
    if method == "31G_Baseline_XGBoost":
        return fit_predict_31g_baseline("XGBoost", X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full)
    if method == "31G_Baseline_CatBoost_XGBoost_Weighted_2Model":
        pred_dict = {
            "CatBoost": fit_predict_31g_baseline("CatBoost", X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full),
            "XGBoost": fit_predict_31g_baseline("XGBoost", X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full),
        }
        return sf19.apply_weights(pred_dict, parse_weight_text(best["Weights"]))

    if method == "31H2_Tuned_CatBoost":
        return fit_predict_config("CatBoost", get_config_by_id("CatBoost", best["CatBoost_Config_ID"]), X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full)
    if method == "31H2_Tuned_XGBoost":
        return fit_predict_config("XGBoost", get_config_by_id("XGBoost", best["XGBoost_Config_ID"]), X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full)
    if method == "31H2_Tuned_CatBoost_XGBoost_Weighted_2Model":
        cb_config = get_config_by_id("CatBoost", best["CatBoost_Config_ID"])
        xgb_config = get_config_by_id("XGBoost", best["XGBoost_Config_ID"])
        pred_dict = {
            "CatBoost": fit_predict_config("CatBoost", cb_config, X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full),
            "XGBoost": fit_predict_config("XGBoost", xgb_config, X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full),
        }
        return sf19.apply_weights(pred_dict, parse_weight_text(best["Weights"]))

    raise ValueError(f"Unknown method: {method}")


def run_one_seed(X_mix: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int):
    log("\n" + "=" * 100)
    log(f"31H2 tuning-grid adequacy check seed = {seed}")
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
    imp_df = s31f.rank_features_by_rf_importance_weighted(X_train, y_train, seed, sample_weight=sample_weight_train)

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
        selected_cols = selected_cols_from_row(best, X_mix)
        log("=" * 100)
        log(f"Final grouped test | seed={seed} | method={method} | K={best['Selected_K']} | n_features={len(selected_cols)} | CB={best['CatBoost_Config_ID']} | XGB={best['XGBoost_Config_ID']}")
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
            "Seed": seed, "Method": method, "Candidate_Type": best["Candidate_Type"], "Base_Models": best["Base_Models"],
            "Selected_K": best["Selected_K"], "Selected_Features": ";".join(selected_cols),
            "CatBoost_Config_ID": best["CatBoost_Config_ID"], "XGBoost_Config_ID": best["XGBoost_Config_ID"],
            "CatBoost_Grid_Block": best["CatBoost_Grid_Block"], "XGBoost_Grid_Block": best["XGBoost_Grid_Block"],
            "CatBoost_Params": best["CatBoost_Params"], "XGBoost_Params": best["XGBoost_Params"],
            "Validation_R2": best["Validation_R2"], "Validation_RMSE": best["Validation_RMSE"], "Validation_MAE": best["Validation_MAE"],
            "Weights": best["Weights"], "Test_R2": metrics["R2"], "Test_RMSE": metrics["RMSE"],
            "Test_MAE": metrics["MAE"], "Test_MSE": metrics["MSE"], "Test_Bias": metrics["Bias"],
            "n_test": len(y_test), "n_test_groups": test_groups.nunique(),
        })
        for local_i, (obs, pred, gid, data_idx) in enumerate(zip(y_test, pred_test, test_groups, test_idx)):
            pred_rows.append({
                "Seed": seed, "Method": method, "Candidate_Type": best["Candidate_Type"], "Base_Models": best["Base_Models"],
                "Selected_K": best["Selected_K"], "Data_Index": int(data_idx), "Local_Test_Index": int(local_i),
                "文献编号": gid, "Observed": float(obs), "Predicted": float(pred), "Residual": float(obs - pred),
            })

    return {"split": pd.DataFrame([split_info]), "candidate": candidate_df, "test": pd.DataFrame(test_rows), "prediction": pd.DataFrame(pred_rows)}


def summarize_results(test_df: pd.DataFrame, pred_df: pd.DataFrame):
    summary = test_df.groupby("Method", as_index=False).agg(
        R2_mean=("Test_R2", "mean"), R2_std=("Test_R2", "std"),
        RMSE_mean=("Test_RMSE", "mean"), RMSE_std=("Test_RMSE", "std"),
        MAE_mean=("Test_MAE", "mean"), MAE_std=("Test_MAE", "std"),
        MSE_mean=("Test_MSE", "mean"), MSE_std=("Test_MSE", "std"),
        Bias_mean=("Test_Bias", "mean"), Bias_std=("Test_Bias", "std"),
        Selected_K=("Selected_K", format_selected_k), n_seed=("Seed", "nunique"), n_test_total=("n_test", "sum"),
    )
    pooled_rows = []
    for method, sub in pred_df.groupby("Method"):
        m = regression_metrics(sub["Observed"], sub["Predicted"])
        pooled_rows.append({
            "Method": method, "Pooled_R2": m["R2"], "Pooled_RMSE": m["RMSE"], "Pooled_MAE": m["MAE"],
            "Pooled_MSE": m["MSE"], "Pooled_Bias": m["Bias"], "n_pooled": len(sub),
            "n_literature_groups_pooled": sub["文献编号"].nunique(),
        })
    summary = summary.merge(pd.DataFrame(pooled_rows), on="Method", how="left")
    summary["Method"] = pd.Categorical(summary["Method"], categories=METHOD_ORDER, ordered=True)
    return summary.sort_values("Method").reset_index(drop=True)


def collect_selected_params(test_df: pd.DataFrame):
    cols = [
        "Seed", "Method", "Candidate_Type", "Selected_K", "CatBoost_Config_ID", "XGBoost_Config_ID",
        "CatBoost_Grid_Block", "XGBoost_Grid_Block", "CatBoost_Params", "XGBoost_Params",
        "Validation_RMSE", "Validation_MAE", "Test_RMSE", "Test_MAE", "Weights"
    ]
    return test_df[[c for c in cols if c in test_df.columns]].copy()


def build_config_catalog():
    rows = []
    for c in CATBOOST_CONFIGS:
        rows.append({"Model_Family": "CatBoost", "Config_ID": c["Config_ID"], "Grid_Block": c["Grid_Block"], "Params": params_to_text(c["Params"])})
    for c in XGBOOST_CONFIGS:
        rows.append({"Model_Family": "XGBoost", "Config_ID": c["Config_ID"], "Grid_Block": c["Grid_Block"], "Params": params_to_text(c["Params"])})
    return pd.DataFrame(rows)


def write_decision_report(summary_df, split_df, selected_params_df, config_catalog):
    path = TABLE_DIR / "31h2_tuning_grid_adequacy_decision.txt"
    ranking = summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE", "Pooled_R2"], ascending=[True, True, False]).reset_index(drop=True)
    best = ranking.iloc[0]
    baseline_method = "31G_Baseline_CatBoost_XGBoost_Weighted_2Model"
    h2_method = "31H2_Tuned_CatBoost_XGBoost_Weighted_2Model"
    baseline_row = summary_df[summary_df["Method"].astype(str) == baseline_method]
    h2_row = summary_df[summary_df["Method"].astype(str) == h2_method]

    lines = []
    lines.append("=" * 100)
    lines.append("Step 31H2 tuning-grid adequacy check report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Check whether Step 31H failed because the tuning grid was too conservative.")
    lines.append("")
    lines.append("Strict workflow reused:")
    lines.append("- Step-31B mechanism-aware descriptors")
    lines.append("- Step-31F balanced grouped split")
    lines.append("- Literature-group-weighted training")
    lines.append("- Script-19 feature selection/preprocessing where applicable")
    lines.append("- Step-31G exact CatBoost/XGBoost baseline")
    lines.append("- Validation-RMSE-selected hyperparameters and non-negative fusion weights")
    lines.append("")
    lines.append("Tuning-grid design:")
    lines.append(f"CatBoost configs: {len(CATBOOST_CONFIGS)}")
    lines.append(f"XGBoost configs : {len(XGBOOST_CONFIGS)}")
    lines.append("Grid blocks: 31G_baseline, conservative, 31G_neighborhood, medium_complexity, controlled_expressive")
    lines.append(f"Top configs crossed for fusion per base model: {TOP_N_PER_BASE_FOR_FUSION}, with 31G default forced into the fusion candidate pool.")
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
    lines.append("Current Step-31H2 best candidate:")
    lines.append(f"{best['Method']} | Pooled_R2={best['Pooled_R2']:.4f}, Pooled_RMSE={best['Pooled_RMSE']:.4f}, Pooled_MAE={best['Pooled_MAE']:.4f}")
    lines.append("")
    if not baseline_row.empty and not h2_row.empty:
        b = baseline_row.iloc[0]
        h = h2_row.iloc[0]
        rmse_gain = float(b["Pooled_RMSE"] - h["Pooled_RMSE"])
        r2_gain = float(h["Pooled_R2"] - b["Pooled_R2"])
        mae_gain = float(b["Pooled_MAE"] - h["Pooled_MAE"])
        lines.append("Direct 31H2 tuned-fusion vs 31G baseline-fusion comparison:")
        lines.append(f"31G baseline fusion: R2={b['Pooled_R2']:.4f}, RMSE={b['Pooled_RMSE']:.4f}, MAE={b['Pooled_MAE']:.4f}")
        lines.append(f"31H2 tuned fusion  : R2={h['Pooled_R2']:.4f}, RMSE={h['Pooled_RMSE']:.4f}, MAE={h['Pooled_MAE']:.4f}")
        lines.append(f"RMSE gain = {rmse_gain:.4f}; R2 gain = {r2_gain:.4f}; MAE gain = {mae_gain:.4f}")
        if rmse_gain > 0 and r2_gain > 0:
            lines.append("Decision: 31H2 tuning improved over the exact 31G baseline fusion.")
        else:
            lines.append("Decision: 31H2 tuning did not improve over the exact 31G baseline fusion.")
        lines.append("")
    lines.append("Selected parameter/configuration records:")
    lines.append(selected_params_df.to_string(index=False))
    lines.append("")
    lines.append("Config catalog:")
    lines.append(config_catalog.to_string(index=False))
    lines.append("")
    lines.append("Next decision guide:")
    lines.append("1. If 31H2 improves over the 31G baseline, keep the improved CatBoost-XGBoost route.")
    lines.append("2. If 31H2 does not improve, stop ordinary CatBoost-XGBoost tuning and move to group-aware feature selection/ablation.")
    lines.append("3. If tuned fusion collapses to boundary weights, report it as evidence that global two-model fusion is limited.")
    lines.append("=" * 100)
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[TEXT] {path}")


def plot_metrics(summary_df: pd.DataFrame):
    data = summary_df.copy().sort_values("Pooled_RMSE", ascending=True).reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    specs = [("Pooled_R2", "Pooled $R^2$", "{:.3f}", "max"), ("Pooled_RMSE", "Pooled RMSE", "{:.2f}", "min"), ("Pooled_MAE", "Pooled MAE", "{:.2f}", "min")]
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
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=30, ha="right")
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
            ax.text(xi, v + (offset if v >= 0 else -offset), fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    fig.suptitle("31H2 tuning-grid adequacy check under balanced grouped validation", fontsize=13.5, fontweight="bold", y=0.995)
    fig.text(0.5, 0.01, "Mechanism-aware descriptors + literature-group-weighted training; 31G baseline retained as explicit comparator.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0.02, 0.06, 1, 0.92])
    save_fig(fig, "106_31h2_tuning_grid_adequacy_metrics.png")


def plot_pooled_prediction(pred_df: pd.DataFrame, summary_df: pd.DataFrame):
    methods = list(summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE"], ascending=[True, True])["Method"].astype(str).head(4))
    n = len(methods)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.6, 5.2 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    for ax, method in zip(axes, methods):
        sub = pred_df[pred_df["Method"] == method].copy()
        sub_plot = sub.sample(900, random_state=42) if len(sub) > 900 else sub
        color = COLORS.get(method, COLORS["gray"])
        ax.scatter(sub_plot["Observed"], sub_plot["Predicted"], s=16, color=color, alpha=0.68, edgecolor="none")
        ax.plot([0, 100], [0, 100], color=COLORS["red"], linestyle=(0, (4, 3)), linewidth=1.0)
        m = regression_metrics(sub["Observed"], sub["Predicted"])
        ax.text(0.06, 0.92, f"$R^2$ = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nMAE = {m['MAE']:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, boxstyle="round,pad=0.3"))
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Observed F removal (%)")
        ax.set_ylabel("Predicted F removal (%)")
        ax.grid(color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    for ax in axes[len(methods):]:
        ax.axis("off")
    fig.suptitle("31H2 pooled predicted-vs-observed plots", fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    save_fig(fig, "107_31h2_tuning_grid_adequacy_pooled_predicted_vs_actual.png")


def plot_weight_stability(test_df: pd.DataFrame):
    rows = []
    for _, row in test_df.iterrows():
        weights = parse_weight_text(row["Weights"])
        if not weights:
            continue
        for base_model, weight in weights.items():
            rows.append({"Seed": row["Seed"], "Method": row["Method"], "BaseModel": base_model, "Weight": float(weight)})
    wdf = pd.DataFrame(rows)
    if wdf.empty:
        return
    methods = [m for m in ["31G_Baseline_CatBoost_XGBoost_Weighted_2Model", "31H2_Tuned_CatBoost_XGBoost_Weighted_2Model"] if m in wdf["Method"].unique()]
    fig, axes = plt.subplots(1, len(methods), figsize=(6.4 * len(methods), 4.6), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        sub = wdf[wdf["Method"] == method].copy()
        for base_model, sub_m in sub.groupby("BaseModel"):
            sub_m = sub_m.sort_values("Seed")
            ax.plot(sub_m["Seed"], sub_m["Weight"], marker="o", linewidth=1.6, markersize=4, color=COLORS.get(base_model, COLORS["gray"]), label=base_model)
            for _, rr in sub_m.iterrows():
                ax.text(rr["Seed"], rr["Weight"] + 0.025, f"{rr['Weight']:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xlabel("Random seed")
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_ylim(-0.05, 1.12)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Validation-optimized fusion weight")
    fig.suptitle("31H2 fusion weight stability", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.92])
    save_fig(fig, "108_31h2_tuning_grid_adequacy_weight_stability.png")


def plot_param_selection(selected_params_df: pd.DataFrame):
    tuned = selected_params_df[selected_params_df["Method"].astype(str).str.contains("31H2")].copy()
    if tuned.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    cb = tuned[tuned["CatBoost_Config_ID"].astype(str) != ""]
    cb_counts = cb["CatBoost_Config_ID"].value_counts().sort_values(ascending=True)
    axes[0, 0].barh(range(len(cb_counts)), cb_counts.values, color=COLORS["CatBoost"], edgecolor="black", linewidth=0.45, alpha=0.9)
    axes[0, 0].set_yticks(range(len(cb_counts)))
    axes[0, 0].set_yticklabels(cb_counts.index)
    axes[0, 0].set_title("Selected CatBoost configs")
    axes[0, 0].set_xlabel("Selection count")
    axes[0, 0].grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    xgb = tuned[tuned["XGBoost_Config_ID"].astype(str) != ""]
    xgb_counts = xgb["XGBoost_Config_ID"].value_counts().sort_values(ascending=True)
    axes[0, 1].barh(range(len(xgb_counts)), xgb_counts.values, color=COLORS["XGBoost"], edgecolor="black", linewidth=0.45, alpha=0.9)
    axes[0, 1].set_yticks(range(len(xgb_counts)))
    axes[0, 1].set_yticklabels(xgb_counts.index)
    axes[0, 1].set_title("Selected XGBoost configs")
    axes[0, 1].set_xlabel("Selection count")
    axes[0, 1].grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    cb_block = cb["CatBoost_Grid_Block"].value_counts().sort_values(ascending=True)
    axes[1, 0].barh(range(len(cb_block)), cb_block.values, color=COLORS["CatBoost"], edgecolor="black", linewidth=0.45, alpha=0.9)
    axes[1, 0].set_yticks(range(len(cb_block)))
    axes[1, 0].set_yticklabels(cb_block.index)
    axes[1, 0].set_title("Selected CatBoost grid blocks")
    axes[1, 0].set_xlabel("Selection count")
    axes[1, 0].grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    xgb_block = xgb["XGBoost_Grid_Block"].value_counts().sort_values(ascending=True)
    axes[1, 1].barh(range(len(xgb_block)), xgb_block.values, color=COLORS["XGBoost"], edgecolor="black", linewidth=0.45, alpha=0.9)
    axes[1, 1].set_yticks(range(len(xgb_block)))
    axes[1, 1].set_yticklabels(xgb_block.index)
    axes[1, 1].set_title("Selected XGBoost grid blocks")
    axes[1, 1].set_xlabel("Selection count")
    axes[1, 1].grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    fig.suptitle("31H2 selected hyperparameter configurations and grid blocks", fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    save_fig(fig, "109_31h2_tuning_grid_adequacy_param_selection.png")


def plot_validation_test_transfer(test_df: pd.DataFrame):
    data = test_df.copy()
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    for method, sub in data.groupby("Method"):
        ax.scatter(sub["Validation_RMSE"], sub["Test_RMSE"], s=60, color=COLORS.get(str(method), COLORS["gray"]), edgecolor="black", linewidth=0.45, alpha=0.8, label=METHOD_LABELS.get(str(method), str(method)))
        for _, r in sub.iterrows():
            ax.text(r["Validation_RMSE"], r["Test_RMSE"] + 0.18, str(int(r["Seed"])), ha="center", va="bottom", fontsize=7)
    lim_min = min(data["Validation_RMSE"].min(), data["Test_RMSE"].min()) - 1
    lim_max = max(data["Validation_RMSE"].max(), data["Test_RMSE"].max()) + 1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color=COLORS["red"], linestyle=(0, (4, 3)), linewidth=1.0)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Validation RMSE")
    ax.set_ylabel("Grouped test RMSE")
    ax.set_title("Validation-to-test transfer under grouped tuning")
    ax.grid(color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    save_fig(fig, "110_31h2_validation_test_rmse_transfer.png")


def main():
    log("\n" + "=" * 100)
    log("Step 31H2: tuning-grid adequacy check")
    log("=" * 100)
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"SEEDS = {SEEDS}")
    log(f"K_OPTIONS = {K_OPTIONS}")
    log(f"CatBoost configs = {len(CATBOOST_CONFIGS)}")
    log(f"XGBoost configs = {len(XGBOOST_CONFIGS)}")
    log(f"Top configs crossed per base = {TOP_N_PER_BASE_FOR_FUSION}")
    log("=" * 100)
    log("\nPreparing script-19 feature sets...")
    y, X_raw, X_eng, X_mix_base = sf19.prepare_feature_sets()
    y = y.reset_index(drop=True)
    X_mix_base = X_mix_base.reset_index(drop=True)
    log("\nAppending Step-31B mechanism-aware descriptors using real 31F integration function...")
    X_mix = s31f.append_mechanism31b_features(X_mix_base, expected_n=len(y)).reset_index(drop=True)
    groups = s31f.load_group_labels(expected_n=len(y))
    s31f.set_balanced_split_context(X_mix, y, groups)
    all_splits, all_candidates, all_tests, all_predictions = [], [], [], []
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
    config_catalog = build_config_catalog()
    out_tables = {
        "31h2_tuning_grid_adequacy_split_summary.csv": split_df,
        "31h2_tuning_grid_adequacy_validation_candidates.csv": candidate_df,
        "31h2_tuning_grid_adequacy_test_results.csv": test_df,
        "31h2_tuning_grid_adequacy_predictions.csv": pred_df,
        "31h2_tuning_grid_adequacy_summary.csv": summary_df,
        "31h2_tuning_grid_adequacy_selected_params.csv": selected_params_df,
        "31h2_tuning_grid_adequacy_config_catalog.csv": config_catalog,
    }
    for filename, df in out_tables.items():
        path = TABLE_DIR / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        log(f"[TABLE] {path}")
    log("\n=== Step 31H2 tuning-grid adequacy summary ===")
    print(summary_df.to_string(index=False))
    write_decision_report(summary_df, split_df, selected_params_df, config_catalog)
    log("\nGenerating figures...")
    plot_metrics(summary_df)
    plot_pooled_prediction(pred_df, summary_df)
    plot_weight_stability(test_df)
    plot_param_selection(selected_params_df)
    plot_validation_test_transfer(test_df)
    log("\n" + "=" * 100)
    log("[DONE] Step 31H2 tuning-grid adequacy check completed.")
    log("=" * 100)
    log("Please send these outputs:")
    log("1. outputs/tables/31h2_tuning_grid_adequacy_summary.csv")
    log("2. outputs/tables/31h2_tuning_grid_adequacy_decision.txt")
    log("3. outputs/tables/31h2_tuning_grid_adequacy_selected_params.csv")
    log("4. outputs/tables/31h2_tuning_grid_adequacy_config_catalog.csv")
    log("5. outputs/figures/106_31h2_tuning_grid_adequacy_metrics.png")
    log("6. outputs/figures/108_31h2_tuning_grid_adequacy_weight_stability.png")
    log("7. outputs/figures/110_31h2_validation_test_rmse_transfer.png")
    log("=" * 100)


if __name__ == "__main__":
    main()
