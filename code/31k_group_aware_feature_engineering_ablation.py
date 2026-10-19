# -*- coding: utf-8 -*-
"""
31k_group_aware_feature_engineering_ablation.py

Purpose
-------
Group-aware feature engineering / feature-set ablation under the same strict
balanced literature-level grouped-validation framework used in Steps 31F--31I.

This script is designed to answer a specific question raised after Step 31H2:
Should the old Step-19 feature engineering / feature selection route be updated
for grouped generalization?

It is NOT a simplified workflow. It reuses the real project scripts:
    - script 19: original feature engineering, preprocessing, feature-selection utilities
    - script 31F: Step-31B mechanism-aware descriptors, balanced grouped split,
      and literature-group sample weights
    - script 31G: CatBoost/XGBoost default model wrappers and validation-RMSE
      non-negative weighted fusion

Main idea
---------
Step 31H2 showed that ordinary CatBoost/XGBoost hyperparameter tuning did not
improve over the Step-31G baseline. Therefore, this step moves upstream to test
whether the feature representation itself should be revised for grouped
cross-literature generalization.

Strict rules
------------
1. 文献编号 is used only for literature-level grouped splitting.
2. 文献编号 is never used as an input feature.
3. Test labels are never used for feature engineering, feature selection,
   validation-RMSE weight search, model selection, or feature-space selection.
4. Feature ranking is performed only on grouped training data.
5. Literature-group-weighted training is reused.
6. Balanced grouped split is reused from 31F.
7. The Step-31G baseline feature space is explicitly retained for comparison.

Compared feature spaces
-----------------------
A. S19_original_mix
   The original script-19 mixed feature table.
B. S19_no_fine_categories
   Script-19 mixed feature table after removing over-specific raw fine labels
   混凝剂 and 干扰离子.
C. Core_numeric_only
   Safe numeric process variables only; leakage/source columns removed.
D. Core_numeric_plus_31B_mechanism
   Safe numeric variables plus Step-31B mechanism-aware descriptors.
E. Current31F_noFine_plus_31B_mechanism
   The exact 31F/31G current feature space: script-19 X_mix after removing
   over-specific fine labels and appending mechanism-aware descriptors.
F. Current31F_minus_coagulant_mechanism
   Current31F feature space with coagulant-mechanism columns removed.
G. Current31F_minus_ion_mechanism
   Current31F feature space with ion/interference-mechanism columns removed.
H. Current31F_minus_interaction_mechanism
   Current31F feature space with pH/dose/initial-F interaction-style mechanism
   columns removed.
I. Current31F_mechanism_only
   Step-31B mechanism-aware descriptors only.

Candidate models
----------------
1. CatBoost
2. XGBoost
3. CatBoost-XGBoost validation-RMSE weighted 2Model

Outputs
-------
Tables:
    outputs/tables/31k_feature_space_audit.csv
    outputs/tables/31k_feature_block_audit.csv
    outputs/tables/31k_feature_ablation_split_summary.csv
    outputs/tables/31k_feature_ablation_validation_candidates.csv
    outputs/tables/31k_feature_ablation_test_results.csv
    outputs/tables/31k_feature_ablation_predictions.csv
    outputs/tables/31k_feature_ablation_summary.csv
    outputs/tables/31k_feature_selection_frequency.csv
    outputs/tables/31k_feature_ablation_decision.txt

Figures:
    outputs/figures/111_31k_feature_ablation_metrics.png/pdf
    outputs/figures/112_31k_feature_space_rmse_heatmap.png/pdf
    outputs/figures/113_31k_feature_selection_frequency.png/pdf
    outputs/figures/114_31k_feature_ablation_pooled_predicted_vs_actual.png/pdf
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

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# ============================================================
# 0. Logging and dynamic imports
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


# Reuse Step 31G; it imports real 31F and script-19.
s31g = load_module_from_file("31g_expanded_model_complementarity_screening.py", "s31g_real")
s31f = s31g.s31f
sf19 = s31g.sf19

PROJECT_ROOT = Path(s31g.PROJECT_ROOT)
FIG_DIR = Path(s31g.FIG_DIR)
TABLE_DIR = Path(s31g.TABLE_DIR)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MECHANISM31B_PATH = PROCESSED_DIR / "model_ready_v1_with_source_mechanism31b.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Settings
# ============================================================

SEEDS = list(getattr(sf19, "SEEDS", [7, 21, 42, 66, 88]))
K_OPTIONS_FROM_19 = list(getattr(sf19, "K_OPTIONS", [12, 18, 25, 35, 45, "all"]))
# Ensure grouped feature-selection sensitivity is actually evaluated even if
# script-19 in the local project only defines a short K list.
K_OPTIONS = []
for k in [12, 18, 25, 35, 45, 60, "all"]:
    if k not in K_OPTIONS:
        K_OPTIONS.append(k)
for k in K_OPTIONS_FROM_19:
    if k not in K_OPTIONS:
        K_OPTIONS.append(k)

BASE_MODELS = ["CatBoost", "XGBoost"]
METHODS = ["CatBoost", "XGBoost", "CatBoost_XGBoost_Weighted_2Model"]

RAW_FINE_CATEGORY_COLS = ["混凝剂", "干扰离子"]

BAD_FEATURE_KEYWORDS = [
    "文献编号", "source", "paper", "literature", "reference", "doi", "id", "index", "行号", "序号", "编号",
    "氟去除率", "去除率", "removal", "target", "label",
    "出水", "effluent", "remaining", "残余",
]

# Human-readable feature-space order.
FEATURE_SPACE_ORDER = [
    "S19_original_mix",
    "S19_no_fine_categories",
    "Core_numeric_only",
    "Core_numeric_plus_31B_mechanism",
    "Current31F_noFine_plus_31B_mechanism",
    "Current31F_mechanism_only",
    "Current31F_minus_coagulant_mechanism",
    "Current31F_minus_ion_mechanism",
    "Current31F_minus_interaction_mechanism",
]

FEATURE_SPACE_LABELS = {
    "S19_original_mix": "S19 original",
    "S19_no_fine_categories": "S19 no fine cat.",
    "Core_numeric_only": "Core numeric",
    "Core_numeric_plus_31B_mechanism": "Core + 31B mech.",
    "Current31F_noFine_plus_31B_mechanism": "Current 31F",
    "Current31F_mechanism_only": "31B mech. only",
    "Current31F_minus_coagulant_mechanism": "31F - coagulant mech.",
    "Current31F_minus_ion_mechanism": "31F - ion mech.",
    "Current31F_minus_interaction_mechanism": "31F - interaction mech.",
}

METHOD_LABELS = {
    "CatBoost": "CatBoost",
    "XGBoost": "XGBoost",
    "CatBoost_XGBoost_Weighted_2Model": "CB-XGB 2Model",
}

COLORS = {
    "CatBoost": "#9B72B0",
    "XGBoost": "#D28C3D",
    "CatBoost_XGBoost_Weighted_2Model": "#B184C6",
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

def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Cannot read CSV: {path}")


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


def is_bad_feature_name(col: str) -> bool:
    s = str(col).lower().replace(" ", "")
    return any(k.lower().replace(" ", "") in s for k in BAD_FEATURE_KEYWORDS)


def clean_feature_table(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy().reset_index(drop=True)
    bad_cols = [c for c in X2.columns if is_bad_feature_name(c)]
    if bad_cols:
        X2 = X2.drop(columns=bad_cols, errors="ignore")
    # Remove duplicated column names safely.
    if X2.columns.duplicated().any():
        X2 = X2.loc[:, ~X2.columns.duplicated()].copy()
    return X2


def safe_numeric_columns(X: pd.DataFrame) -> list[str]:
    cols = []
    for c in X.columns:
        if c in RAW_FINE_CATEGORY_COLS:
            continue
        if is_bad_feature_name(c):
            continue
        if pd.api.types.is_numeric_dtype(X[c]):
            cols.append(c)
    return cols


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


# ============================================================
# 3. Feature-space construction and block definitions
# ============================================================

def load_mechanism_features(expected_n: int):
    if not MECHANISM31B_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {MECHANISM31B_PATH}. Run Step 31B first."
        )
    df_mech = read_csv_smart(MECHANISM31B_PATH).reset_index(drop=True)
    if len(df_mech) != expected_n:
        raise ValueError(
            f"Row mismatch: expected {expected_n}, mechanism31b has {len(df_mech)}."
        )
    mechanism_cols = s31f.load_mechanism31b_feature_columns(df_mech)
    mechanism_cols = [c for c in mechanism_cols if c in df_mech.columns and not is_bad_feature_name(c)]
    X_mech = df_mech[mechanism_cols].copy().reset_index(drop=True)
    return X_mech, mechanism_cols, df_mech


def classify_mechanism_blocks(mechanism_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in mechanism_cols:
        s = str(c).lower()
        tags = []
        if any(k in s for k in ["混凝", "coagul", "metal", "金属", "铝", "铁", "钙", "镁", "锆", "稀土", "al", "fe", "ca", "mg", "zr"]):
            tags.append("coagulant_mechanism")
        if any(k in s for k in ["干扰", "离子", "ion", "so4", "cl", "no3", "po4", "hco3", "co3", "硫酸", "氯", "硝酸", "磷酸", "碳酸"]):
            tags.append("ion_mechanism")
        if any(k in s for k in ["ph", "剂量", "dose", "初始", "initial", "ratio", "relative", "相对", "log", "interaction", "交互", "乘", "×", "x_"]):
            tags.append("interaction_mechanism")
        if not tags:
            tags.append("other_mechanism")
        rows.append({"Feature": c, "Block": ";".join(tags)})
    return pd.DataFrame(rows)


def columns_in_block(block_df: pd.DataFrame, block_keyword: str) -> list[str]:
    if block_df.empty:
        return []
    mask = block_df["Block"].astype(str).str.contains(block_keyword, regex=False)
    return block_df.loc[mask, "Feature"].astype(str).tolist()


def make_feature_spaces(X_mix_base: pd.DataFrame, expected_n: int):
    X_base_raw = clean_feature_table(X_mix_base)
    X_base_no_fine = X_base_raw.drop(columns=[c for c in RAW_FINE_CATEGORY_COLS if c in X_base_raw.columns], errors="ignore")

    X_mech, mechanism_cols, df_mech = load_mechanism_features(expected_n)
    block_df = classify_mechanism_blocks(mechanism_cols)

    core_num_cols = safe_numeric_columns(X_base_raw)
    X_core_numeric = X_base_raw[core_num_cols].copy().reset_index(drop=True)

    # Exact current 31F feature space, built by the real 31F integration function.
    X_current31f = s31f.append_mechanism31b_features(X_base_raw, expected_n=expected_n)
    X_current31f = clean_feature_table(X_current31f).reset_index(drop=True)

    # Mechanism-only and core+mechanism variants.
    X_mech_only = X_mech.copy().reset_index(drop=True)
    X_core_plus_mech = pd.concat([X_core_numeric.reset_index(drop=True), X_mech.reset_index(drop=True)], axis=1)
    X_core_plus_mech = clean_feature_table(X_core_plus_mech)

    coagulant_cols = columns_in_block(block_df, "coagulant_mechanism")
    ion_cols = columns_in_block(block_df, "ion_mechanism")
    interaction_cols = columns_in_block(block_df, "interaction_mechanism")

    feature_spaces = {
        "S19_original_mix": X_base_raw,
        "S19_no_fine_categories": clean_feature_table(X_base_no_fine),
        "Core_numeric_only": clean_feature_table(X_core_numeric),
        "Core_numeric_plus_31B_mechanism": X_core_plus_mech,
        "Current31F_noFine_plus_31B_mechanism": X_current31f,
        "Current31F_mechanism_only": clean_feature_table(X_mech_only),
        "Current31F_minus_coagulant_mechanism": clean_feature_table(X_current31f.drop(columns=coagulant_cols, errors="ignore")),
        "Current31F_minus_ion_mechanism": clean_feature_table(X_current31f.drop(columns=ion_cols, errors="ignore")),
        "Current31F_minus_interaction_mechanism": clean_feature_table(X_current31f.drop(columns=interaction_cols, errors="ignore")),
    }

    # Remove invalid/too-small feature spaces.
    feature_spaces = {k: v for k, v in feature_spaces.items() if v.shape[1] >= 3}

    audit_rows = []
    for name, X in feature_spaces.items():
        audit_rows.append({
            "Feature_Space": name,
            "Label": FEATURE_SPACE_LABELS.get(name, name),
            "n_features": X.shape[1],
            "n_numeric_features": int(sum(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)),
            "n_non_numeric_features": int(sum(not pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)),
            "contains_raw_coagulant_label": int("混凝剂" in X.columns),
            "contains_raw_interfering_ion_label": int("干扰离子" in X.columns),
            "n_mechanism_cols_present": int(sum(c in X.columns for c in mechanism_cols)),
            "columns": ";".join(map(str, X.columns)),
        })

    block_audit = block_df.copy()
    block_count = (
        block_audit.assign(Block_Exploded=block_audit["Block"].str.split(";"))
        .explode("Block_Exploded")
        .groupby("Block_Exploded", as_index=False)
        .agg(n_features=("Feature", "nunique"))
    )
    block_audit = block_audit.merge(block_count, left_on="Block", right_on="Block_Exploded", how="left")

    return feature_spaces, pd.DataFrame(audit_rows), block_df


# ============================================================
# 4. Model evaluation
# ============================================================

def select_columns_from_importance(imp_df: pd.DataFrame, X: pd.DataFrame, selected_k):
    cols = sf19.select_top_features(imp_df, X, selected_k)
    return [c for c in cols if c in X.columns and not is_bad_feature_name(c)]


def fit_predict_base(model_name, X_train, y_train, X_target, seed, sample_weight=None):
    return s31g.fit_predict_model(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_target=X_target,
        seed=seed,
        sample_weight=sample_weight,
    )


def evaluate_validation_candidates(X_train_k, y_train, X_valid_k, y_valid, seed, feature_space, selected_k, selected_cols, sample_weight_train):
    rows = []
    valid_preds = {}

    for model_name in BASE_MODELS:
        pred = fit_predict_base(model_name, X_train_k, y_train, X_valid_k, seed, sample_weight_train)
        valid_preds[model_name] = pred
        m = regression_metrics(y_valid, pred)
        rows.append({
            "Seed": seed,
            "Feature_Space": feature_space,
            "Method": model_name,
            "Selected_K": selected_k,
            "Selected_Features": ";".join(selected_cols),
            "Validation_R2": m["R2"],
            "Validation_RMSE": m["RMSE"],
            "Validation_MAE": m["MAE"],
            "Validation_MSE": m["MSE"],
            "Validation_Bias": m["Bias"],
            "Weights": "",
        })

    weights, _ = sf19.search_best_weights(valid_preds, y_valid, seed)
    pred_fused = sf19.apply_weights(valid_preds, weights)
    m = regression_metrics(y_valid, pred_fused)
    rows.append({
        "Seed": seed,
        "Feature_Space": feature_space,
        "Method": "CatBoost_XGBoost_Weighted_2Model",
        "Selected_K": selected_k,
        "Selected_Features": ";".join(selected_cols),
        "Validation_R2": m["R2"],
        "Validation_RMSE": m["RMSE"],
        "Validation_MAE": m["MAE"],
        "Validation_MSE": m["MSE"],
        "Validation_Bias": m["Bias"],
        "Weights": weight_dict_to_text(weights),
    })

    return rows


def test_predict_method(method, weights_text, X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full):
    if method in BASE_MODELS:
        return fit_predict_base(method, X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full)

    if method == "CatBoost_XGBoost_Weighted_2Model":
        weights = parse_weight_text(weights_text)
        pred_dict = {
            "CatBoost": fit_predict_base("CatBoost", X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full),
            "XGBoost": fit_predict_base("XGBoost", X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full),
        }
        return sf19.apply_weights(pred_dict, weights)

    raise ValueError(f"Unknown method: {method}")


def run_one_seed_one_feature_space(X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int, feature_space: str):
    log("\n" + "=" * 100)
    log(f"31K feature ablation | seed={seed} | feature_space={feature_space} | n_features={X.shape[1]}")
    log("=" * 100)

    train_idx, valid_idx, test_idx, split_info = s31f.grouped_split_indices(groups, seed)
    train_valid_idx = np.concatenate([train_idx, valid_idx])

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_valid = X.iloc[valid_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    X_train_full = X.iloc[train_valid_idx].reset_index(drop=True)

    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_valid = y.iloc[valid_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    y_train_full = y.iloc[train_valid_idx].reset_index(drop=True)

    train_groups = groups.iloc[train_idx].reset_index(drop=True)
    train_full_groups = groups.iloc[train_valid_idx].reset_index(drop=True)
    test_groups = groups.iloc[test_idx].reset_index(drop=True)

    sample_weight_train = s31f.make_literature_group_weights(train_groups)
    sample_weight_train_full = s31f.make_literature_group_weights(train_full_groups)

    # Feature ranking only on training data for this feature space.
    imp_df = s31f.rank_features_by_rf_importance_weighted(
        X_train,
        y_train,
        seed,
        sample_weight=sample_weight_train,
    )

    candidate_rows = []
    valid_k_seen = set()
    for selected_k in K_OPTIONS:
        selected_cols = select_columns_from_importance(imp_df, X, selected_k)
        if len(selected_cols) < 3:
            continue
        # Avoid duplicate K results when K exceeds n_features and returns all columns.
        key = tuple(selected_cols)
        if key in valid_k_seen:
            continue
        valid_k_seen.add(key)

        log(f"  Feature_Space={feature_space} | Seed={seed} | K={selected_k} | selected_features={len(selected_cols)}")
        X_train_k = X_train[selected_cols].reset_index(drop=True)
        X_valid_k = X_valid[selected_cols].reset_index(drop=True)
        rows = evaluate_validation_candidates(
            X_train_k=X_train_k,
            y_train=y_train,
            X_valid_k=X_valid_k,
            y_valid=y_valid,
            seed=seed,
            feature_space=feature_space,
            selected_k=selected_k,
            selected_cols=selected_cols,
            sample_weight_train=sample_weight_train,
        )
        candidate_rows.extend(rows)

    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        raise RuntimeError(f"No validation candidates produced for {feature_space}, seed={seed}")

    test_rows = []
    pred_rows = []

    for method in METHODS:
        sub = candidate_df[candidate_df["Method"] == method].copy()
        if sub.empty:
            continue
        best = sub.sort_values(["Validation_RMSE", "Validation_MAE", "Validation_R2"], ascending=[True, True, False]).iloc[0]
        selected_cols = str(best["Selected_Features"]).split(";")
        selected_cols = [c for c in selected_cols if c in X.columns]

        X_train_full_k = X_train_full[selected_cols].reset_index(drop=True)
        X_test_k = X_test[selected_cols].reset_index(drop=True)

        pred_test = test_predict_method(
            method=method,
            weights_text=best["Weights"],
            X_train_full_k=X_train_full_k,
            y_train_full=y_train_full,
            X_test_k=X_test_k,
            seed=seed,
            sample_weight_train_full=sample_weight_train_full,
        )
        m = regression_metrics(y_test, pred_test)

        test_rows.append({
            "Seed": seed,
            "Feature_Space": feature_space,
            "Method": method,
            "Selected_K": best["Selected_K"],
            "n_Selected_Features": len(selected_cols),
            "Selected_Features": ";".join(selected_cols),
            "Validation_R2": best["Validation_R2"],
            "Validation_RMSE": best["Validation_RMSE"],
            "Validation_MAE": best["Validation_MAE"],
            "Weights": best["Weights"],
            "Test_R2": m["R2"],
            "Test_RMSE": m["RMSE"],
            "Test_MAE": m["MAE"],
            "Test_MSE": m["MSE"],
            "Test_Bias": m["Bias"],
            "n_test": len(y_test),
            "n_test_groups": test_groups.nunique(),
        })

        for local_i, (obs, pred, gid, data_idx) in enumerate(zip(y_test, pred_test, test_groups, test_idx)):
            pred_rows.append({
                "Seed": seed,
                "Feature_Space": feature_space,
                "Method": method,
                "Selected_K": best["Selected_K"],
                "Data_Index": int(data_idx),
                "Local_Test_Index": int(local_i),
                "文献编号": gid,
                "Observed": float(obs),
                "Predicted": float(pred),
                "Residual": float(obs - pred),
            })

    split_info = dict(split_info)
    split_info["Feature_Space"] = feature_space
    split_info["n_features_in_space"] = int(X.shape[1])

    return {
        "split": pd.DataFrame([split_info]),
        "candidate": candidate_df,
        "test": pd.DataFrame(test_rows),
        "prediction": pd.DataFrame(pred_rows),
    }


# ============================================================
# 5. Summary and diagnostics
# ============================================================

def summarize_results(test_df: pd.DataFrame, pred_df: pd.DataFrame):
    summary = (
        test_df.groupby(["Feature_Space", "Method"], as_index=False)
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
            n_Selected_Features_mean=("n_Selected_Features", "mean"),
            n_seed=("Seed", "nunique"),
            n_test_total=("n_test", "sum"),
        )
    )

    pooled_rows = []
    for (fs, method), sub in pred_df.groupby(["Feature_Space", "Method"]):
        m = regression_metrics(sub["Observed"], sub["Predicted"])
        pooled_rows.append({
            "Feature_Space": fs,
            "Method": method,
            "Pooled_R2": m["R2"],
            "Pooled_RMSE": m["RMSE"],
            "Pooled_MAE": m["MAE"],
            "Pooled_MSE": m["MSE"],
            "Pooled_Bias": m["Bias"],
            "n_pooled": len(sub),
            "n_literature_groups_pooled": sub["文献编号"].nunique(),
        })
    summary = summary.merge(pd.DataFrame(pooled_rows), on=["Feature_Space", "Method"], how="left")
    summary["Feature_Space"] = pd.Categorical(summary["Feature_Space"], categories=FEATURE_SPACE_ORDER, ordered=True)
    summary["Method"] = pd.Categorical(summary["Method"], categories=METHODS, ordered=True)
    summary = summary.sort_values(["Pooled_RMSE", "Pooled_MAE", "Pooled_R2"], ascending=[True, True, False]).reset_index(drop=True)
    return summary


def compute_feature_selection_frequency(test_df: pd.DataFrame):
    rows = []
    for _, row in test_df.iterrows():
        features = str(row["Selected_Features"]).split(";")
        for f in features:
            if f.strip():
                rows.append({
                    "Feature_Space": row["Feature_Space"],
                    "Method": row["Method"],
                    "Seed": row["Seed"],
                    "Feature": f,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    freq = (
        df.groupby(["Feature_Space", "Method", "Feature"], as_index=False)
        .agg(n_selected=("Seed", "nunique"))
        .sort_values(["Feature_Space", "Method", "n_selected"], ascending=[True, True, False])
    )
    return freq


# ============================================================
# 6. Figures and report
# ============================================================

def plot_metrics(summary_df: pd.DataFrame):
    data = summary_df.copy().sort_values(["Pooled_RMSE", "Pooled_MAE"], ascending=[True, True]).head(18)
    labels = [f"{FEATURE_SPACE_LABELS.get(str(fs), str(fs))}\n{METHOD_LABELS.get(str(m), str(m))}" for fs, m in zip(data["Feature_Space"], data["Method"])]
    x = np.arange(len(data))

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6))
    specs = [
        ("Pooled_R2", "Pooled $R^2$", "{:.3f}", True),
        ("Pooled_RMSE", "Pooled RMSE", "{:.2f}", False),
        ("Pooled_MAE", "Pooled MAE", "{:.2f}", False),
    ]
    for ax, (col, ylabel, fmt, high_better) in zip(axes, specs):
        vals = data[col].astype(float).values
        bar_colors = [COLORS.get(str(m), COLORS["gray"]) for m in data["Method"]]
        bars = ax.bar(x, vals, color=bar_colors, edgecolor="black", linewidth=0.45, alpha=0.88)
        best_idx = int(np.nanargmax(vals)) if high_better else int(np.nanargmin(vals))
        bars[best_idx].set_edgecolor(COLORS["red"])
        bars[best_idx].set_linewidth(1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=38, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        if col == "Pooled_R2":
            ymin, ymax = np.nanmin(vals), np.nanmax(vals)
            ax.set_ylim(min(0, ymin - 0.04), max(0.20, ymax + 0.05))
            ax.axhline(0, color="#444444", linewidth=0.8)
            offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.025
        else:
            ax.set_ylim(0, np.nanmax(vals) * 1.18)
            offset = ax.get_ylim()[1] * 0.02
        for xi, v in zip(x, vals):
            ax.text(xi, v + (offset if v >= 0 else -offset), fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=6.6)
    fig.suptitle("31K group-aware feature engineering and feature-set ablation", fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.01, "All feature spaces use the same balanced literature-level grouped split; features and weights selected only on validation data.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0.01, 0.07, 1, 0.93])
    save_fig(fig, "111_31k_feature_ablation_metrics.png")


def plot_feature_space_rmse_heatmap(summary_df: pd.DataFrame):
    pivot = summary_df.pivot_table(index="Feature_Space", columns="Method", values="Pooled_RMSE", aggfunc="mean")
    pivot = pivot.reindex([fs for fs in FEATURE_SPACE_ORDER if fs in pivot.index])
    pivot = pivot[[m for m in METHODS if m in pivot.columns]]
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, max(4.8, 0.45 * len(pivot) + 1.8)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels([METHOD_LABELS.get(str(c), str(c)) for c in pivot.columns], rotation=25, ha="right")
    ax.set_yticklabels([FEATURE_SPACE_LABELS.get(str(i), str(i)) for i in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pooled RMSE")
    ax.set_title("Feature-space RMSE heatmap under grouped validation")
    fig.tight_layout()
    save_fig(fig, "112_31k_feature_space_rmse_heatmap.png")


def plot_feature_selection_frequency(freq_df: pd.DataFrame, summary_df: pd.DataFrame):
    if freq_df.empty:
        return
    best = summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE"], ascending=[True, True]).iloc[0]
    fs = str(best["Feature_Space"])
    method = str(best["Method"])
    sub = freq_df[(freq_df["Feature_Space"].astype(str) == fs) & (freq_df["Method"].astype(str) == method)].copy()
    sub = sub.sort_values("n_selected", ascending=False).head(20)
    if sub.empty:
        return
    sub = sub.sort_values("n_selected", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    y = np.arange(len(sub))
    ax.barh(y, sub["n_selected"].values, color="#8DA9C4", edgecolor="black", linewidth=0.45, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["Feature"].astype(str).values)
    ax.set_xlabel("Selection count across seeds")
    ax.set_title(f"Top selected features for best 31K route\n{FEATURE_SPACE_LABELS.get(fs, fs)} + {METHOD_LABELS.get(method, method)}")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    for yi, v in zip(y, sub["n_selected"].values):
        ax.text(v + 0.05, yi, str(int(v)), va="center", ha="left", fontsize=7)
    fig.tight_layout()
    save_fig(fig, "113_31k_feature_selection_frequency.png")


def plot_pooled_predictions(pred_df: pd.DataFrame, summary_df: pd.DataFrame):
    top = summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE"], ascending=[True, True]).head(4)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 10.5), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (_, row) in zip(axes, top.iterrows()):
        fs = str(row["Feature_Space"])
        method = str(row["Method"])
        sub = pred_df[(pred_df["Feature_Space"].astype(str) == fs) & (pred_df["Method"].astype(str) == method)].copy()
        sub_plot = sub.sample(900, random_state=42) if len(sub) > 900 else sub
        ax.scatter(sub_plot["Observed"], sub_plot["Predicted"], s=16, color=COLORS.get(method, COLORS["gray"]), alpha=0.68, edgecolor="none")
        ax.plot([0, 100], [0, 100], color=COLORS["red"], linestyle=(0, (4, 3)), linewidth=1.0)
        m = regression_metrics(sub["Observed"], sub["Predicted"])
        ax.text(0.06, 0.92, f"$R^2$ = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nMAE = {m['MAE']:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, boxstyle="round,pad=0.3"))
        ax.set_title(f"{FEATURE_SPACE_LABELS.get(fs, fs)}\n{METHOD_LABELS.get(method, method)}")
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Observed F removal (%)")
        ax.set_ylabel("Predicted F removal (%)")
        ax.grid(color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    fig.suptitle("Top pooled predicted-vs-observed plots after 31K feature ablation", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    save_fig(fig, "114_31k_feature_ablation_pooled_predicted_vs_actual.png")


def write_decision_report(summary_df, audit_df, block_df, freq_df):
    path = TABLE_DIR / "31k_feature_ablation_decision.txt"
    ranking = summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE", "Pooled_R2"], ascending=[True, True, False]).reset_index(drop=True)
    best = ranking.iloc[0]

    # Current31F baseline line for comparison.
    base = summary_df[
        (summary_df["Feature_Space"].astype(str) == "Current31F_noFine_plus_31B_mechanism") &
        (summary_df["Method"].astype(str) == "CatBoost_XGBoost_Weighted_2Model")
    ].copy()
    base_line = "Not found"
    if not base.empty:
        b = base.iloc[0]
        rmse_gain = float(b["Pooled_RMSE"] - best["Pooled_RMSE"])
        r2_gain = float(best["Pooled_R2"] - b["Pooled_R2"])
        base_line = (
            f"Current31F CB-XGB baseline: R2={b['Pooled_R2']:.4f}, RMSE={b['Pooled_RMSE']:.4f}, MAE={b['Pooled_MAE']:.4f}; "
            f"Best-vs-baseline RMSE gain={rmse_gain:.4f}, R2 gain={r2_gain:.4f}."
        )

    lines = []
    lines.append("=" * 100)
    lines.append("Step 31K group-aware feature engineering / feature-set ablation report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Test whether the Step-19 / Step-31F feature representation should be updated for all-domain grouped generalization.")
    lines.append("")
    lines.append("Strict workflow reused:")
    lines.append("- script-19 feature engineering / preprocessing / feature selection utilities")
    lines.append("- Step-31B mechanism-aware descriptors")
    lines.append("- Step-31F balanced literature-level grouped split")
    lines.append("- Literature-group-weighted training")
    lines.append("- Step-31G CatBoost/XGBoost defaults and validation-RMSE fusion")
    lines.append("- Test labels are never used for feature selection or model/feature-space selection")
    lines.append("")
    lines.append("Feature-space audit:")
    lines.append(audit_df[["Feature_Space", "n_features", "n_numeric_features", "n_non_numeric_features", "contains_raw_coagulant_label", "contains_raw_interfering_ion_label", "n_mechanism_cols_present"]].to_string(index=False))
    lines.append("")
    lines.append("Mechanism block counts:")
    if not block_df.empty:
        block_counts = (
            block_df.assign(Block_Exploded=block_df["Block"].str.split(";"))
            .explode("Block_Exploded")
            .groupby("Block_Exploded", as_index=False)
            .agg(n_features=("Feature", "nunique"))
            .sort_values("n_features", ascending=False)
        )
        lines.append(block_counts.to_string(index=False))
    lines.append("")
    lines.append("Performance ranking by pooled RMSE:")
    keep = ["Feature_Space", "Method", "Selected_K", "n_Selected_Features_mean", "Pooled_R2", "Pooled_RMSE", "Pooled_MAE", "Pooled_Bias", "n_pooled"]
    lines.append(ranking[keep].to_string(index=False))
    lines.append("")
    lines.append("Current 31K best candidate:")
    lines.append(f"{best['Feature_Space']} + {best['Method']} | Pooled_R2={best['Pooled_R2']:.4f}, Pooled_RMSE={best['Pooled_RMSE']:.4f}, Pooled_MAE={best['Pooled_MAE']:.4f}")
    lines.append(base_line)
    lines.append("")
    lines.append("Decision guide:")
    lines.append("1. If a revised feature space improves over Current31F CB-XGB, update the Step-19/31F feature engineering route for grouped generalization.")
    lines.append("2. If no feature space improves over Current31F CB-XGB, keep the current feature representation and move to non-tree model screening / grouped OOF stacking.")
    lines.append("3. If mechanism-block removal improves performance, the removed block should be treated as noisy or over-specific under cross-literature validation, not automatically discarded from chemical interpretation.")
    lines.append("4. Raw fine labels are diagnostic only; if they improve random or grouped metrics, still check unseen-category risk before using them as final paper features.")
    lines.append("=" * 100)
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[TEXT] {path}")


# ============================================================
# 7. Main
# ============================================================

def main():
    log("\n" + "=" * 100)
    log("Step 31K: group-aware feature engineering / feature-set ablation")
    log("=" * 100)
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"SEEDS = {SEEDS}")
    log(f"K_OPTIONS = {K_OPTIONS}")
    log("=" * 100)

    log("Preparing script-19 feature sets...")
    y, X_raw, X_eng, X_mix_base = sf19.prepare_feature_sets()
    y = y.reset_index(drop=True).astype(float)
    X_mix_base = X_mix_base.reset_index(drop=True)

    log("Building feature spaces...")
    feature_spaces, audit_df, block_df = make_feature_spaces(X_mix_base, expected_n=len(y))

    # Use the exact Current31F feature table as the balancing context so all
    # feature spaces share the same 31F balanced grouped splits.
    if "Current31F_noFine_plus_31B_mechanism" not in feature_spaces:
        raise RuntimeError("Current31F feature space not found; cannot set fair split context.")
    X_split_context = feature_spaces["Current31F_noFine_plus_31B_mechanism"]
    groups = s31f.load_group_labels(expected_n=len(y))
    s31f.set_balanced_split_context(X_split_context, y, groups)

    audit_path = TABLE_DIR / "31k_feature_space_audit.csv"
    block_path = TABLE_DIR / "31k_feature_block_audit.csv"
    audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")
    block_df.to_csv(block_path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {audit_path}")
    log(f"[TABLE] {block_path}")

    all_splits = []
    all_candidates = []
    all_tests = []
    all_predictions = []

    ordered_spaces = [fs for fs in FEATURE_SPACE_ORDER if fs in feature_spaces]
    for fs in ordered_spaces:
        X = feature_spaces[fs].reset_index(drop=True)
        for seed in SEEDS:
            result = run_one_seed_one_feature_space(X=X, y=y, groups=groups, seed=seed, feature_space=fs)
            all_splits.append(result["split"])
            all_candidates.append(result["candidate"])
            all_tests.append(result["test"])
            all_predictions.append(result["prediction"])

    split_df = pd.concat(all_splits, ignore_index=True)
    candidate_df = pd.concat(all_candidates, ignore_index=True)
    test_df = pd.concat(all_tests, ignore_index=True)
    pred_df = pd.concat(all_predictions, ignore_index=True)
    summary_df = summarize_results(test_df, pred_df)
    freq_df = compute_feature_selection_frequency(test_df)

    out_tables = {
        "31k_feature_ablation_split_summary.csv": split_df,
        "31k_feature_ablation_validation_candidates.csv": candidate_df,
        "31k_feature_ablation_test_results.csv": test_df,
        "31k_feature_ablation_predictions.csv": pred_df,
        "31k_feature_ablation_summary.csv": summary_df,
        "31k_feature_selection_frequency.csv": freq_df,
    }
    for filename, df in out_tables.items():
        path = TABLE_DIR / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        log(f"[TABLE] {path}")

    log("\n=== Step 31K feature ablation summary ===")
    print(summary_df.to_string(index=False))

    write_decision_report(summary_df, audit_df, block_df, freq_df)

    log("Generating figures...")
    plot_metrics(summary_df)
    plot_feature_space_rmse_heatmap(summary_df)
    plot_feature_selection_frequency(freq_df, summary_df)
    plot_pooled_predictions(pred_df, summary_df)

    log("\n" + "=" * 100)
    log("[DONE] Step 31K group-aware feature engineering / feature-set ablation completed.")
    log("=" * 100)
    log("Please send these outputs:")
    log("1. outputs/tables/31k_feature_ablation_summary.csv")
    log("2. outputs/tables/31k_feature_ablation_decision.txt")
    log("3. outputs/tables/31k_feature_space_audit.csv")
    log("4. outputs/tables/31k_feature_selection_frequency.csv")
    log("5. outputs/figures/111_31k_feature_ablation_metrics.png")
    log("6. outputs/figures/112_31k_feature_space_rmse_heatmap.png")
    log("7. outputs/figures/113_31k_feature_selection_frequency.png")
    log("=" * 100)


if __name__ == "__main__":
    main()
