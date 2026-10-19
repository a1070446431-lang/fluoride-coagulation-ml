# -*- coding: utf-8 -*-
"""
31n_krr_grouped_validation_robustness.py

Purpose
-------
Robustness validation after Step 31L/31M/31M2 showed that KRR_Poly
(polynomial Kernel Ridge Regression; 多项式核岭回归) is the current best
all-domain grouped-validation candidate.

This script answers four strict questions:
1. Is KRR_Poly robust beyond the current Step-31F balanced grouped split?
2. Does the result change under other fully grouped validation forms?
3. Are there candidate models with residual directions opposite to, or at least
   weakly correlated with, KRR_Poly?
4. Which literature groups dominate the remaining error?

Validation forms included
-------------------------
A. Balanced_5Seed_Current31F
   The same balanced grouped validation seeds used in the current 31F-31M route.

B. RepeatedBalanced_Extended
   More seeds using the same balanced grouped split algorithm.

C. RepeatedOrdinary_GroupedSplit
   Fully grouped repeated train/validation/test splitting without balancing.

D. GroupKFold_FullyGrouped
   Fully grouped K-fold test folds. Validation groups are selected only from
   the remaining training-side groups.

E. LOLO_LeaveOneLiteratureOut
   Leave-one-literature-out test evaluation. Validation groups are selected only
   from the non-test literature groups. This is the strictest per-paper test.

Important strictness rules
--------------------------
- 文献编号 is used only for splitting and diagnostics, never as a model input.
- Feature ranking is fitted only on training literature groups.
- Model/config/fusion weights are selected only on validation literature groups.
- Test labels are never used for feature/config/model/weight selection.
- The confirmed Current31F_noFine_plus_31B_mechanism feature route is reused.
- Total runtime is printed at the end.

Core model set
--------------
KRR_Poly, CatBoost, XGBoost, CB-XGB, KRR_CatBoost_Weighted_2Model,
KRR_XGBoost_Weighted_2Model.

Extra residual-correlation screen
---------------------------------
GPR_Matern and PLS are added for selected robustness screens to look for
low/negative residual correlation with KRR_Poly. These are not forced into the
final route unless they show both performance and complementarity.

Outputs
-------
Tables:
    outputs/tables/31n_grouped_validation_split_summary.csv
    outputs/tables/31n_grouped_validation_test_results.csv
    outputs/tables/31n_grouped_validation_predictions.csv
    outputs/tables/31n_grouped_validation_summary.csv
    outputs/tables/31n_grouped_validation_paper_metrics.csv
    outputs/tables/31n_grouped_validation_residual_correlation_by_strategy.csv
    outputs/tables/31n_low_residual_correlation_screen.csv
    outputs/tables/31n_grouped_validation_decision.txt

Figures:
    outputs/figures/127_31n_grouped_validation_strategy_metrics.png/pdf
    outputs/figures/128_31n_krr_vs_baselines_by_strategy.png/pdf
    outputs/figures/129_31n_lolo_paper_level_rmse.png/pdf
    outputs/figures/130_31n_residual_correlation_screen.png/pdf
"""

from __future__ import annotations

import math
import time
import warnings
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


# ============================================================
# 0. Logging and dynamic import of real project scripts
# ============================================================

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
    log(f"Loading real project script: {filename}")
    log(str(module_path))
    log("=" * 100)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Step 31M2 imports/reuses 31M, 31L, 31K, 31F, and script-19.
s31m2 = load_module_from_file("31m2_single_tree_krr_pairwise_fusion_fixed.py", "s31m2_real_for_31n")
s31m = s31m2.s31m
s31l = s31m2.s31l
s31k = s31m2.s31k
s31f = s31m2.s31f
sf19 = s31m2.sf19

PROJECT_ROOT = Path(s31m2.PROJECT_ROOT)
FIG_DIR = Path(s31m2.FIG_DIR)
TABLE_DIR = Path(s31m2.TABLE_DIR)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Settings
# ============================================================

BEST_FEATURE_SPACE = "Current31F_noFine_plus_31B_mechanism"

# Keep the current main 5 seeds exactly visible.
BALANCED_CURRENT_SEEDS = [7, 21, 42, 66, 88]

# More fully grouped balanced splits. Includes current seeds for continuity.
REPEATED_BALANCED_SEEDS = [3, 5, 7, 11, 21, 31, 42, 53, 66, 77, 88, 101, 123, 202, 314]

# Ordinary grouped split seeds, intentionally independent from balanced scoring.
ORDINARY_GROUPED_SEEDS = [3, 5, 7, 11, 21, 31, 42, 53, 66, 77, 88, 101, 123, 202, 314]

GROUP_KFOLD_N_SPLITS = 5
LOLO_VALID_FRAC_OF_NONTEST_GROUPS = 0.20
VALID_SELECTION_CANDIDATES = 500

# Exact GPR is computationally heavier. To keep the script rigorous but feasible,
# extra non-tree residual-correlation candidates are run on the two repeated split
# strategies and not on every LOLO fold.
INCLUDE_EXTRA_ANTI_CORR_MODELS_IN = {"Balanced_5Seed_Current31F", "RepeatedBalanced_Extended"}
EXTRA_ANTI_CORR_MODELS = ["GPR_Matern", "PLS"]

CORE_METHODS = [
    "KRR_Poly",
    "CatBoost",
    "XGBoost",
    "CB-XGB",
    "KRR_CatBoost_Weighted_2Model",
    "KRR_XGBoost_Weighted_2Model",
]

METHOD_ORDER = [
    "KRR_Poly",
    "KRR_CatBoost_Weighted_2Model",
    "KRR_XGBoost_Weighted_2Model",
    "GPR_Matern",
    "PLS",
    "CB-XGB",
    "CatBoost",
    "XGBoost",
]

STRATEGY_ORDER = [
    "Balanced_5Seed_Current31F",
    "RepeatedBalanced_Extended",
    "RepeatedOrdinary_GroupedSplit",
    "GroupKFold_FullyGrouped",
    "LOLO_LeaveOneLiteratureOut",
]

COLORS = {
    "KRR_Poly": "#76B083",
    "KRR_CatBoost_Weighted_2Model": "#67A9CF",
    "KRR_XGBoost_Weighted_2Model": "#B2ABD2",
    "GPR_Matern": "#BC7C95",
    "PLS": "#9B72B0",
    "CB-XGB": "#808080",
    "CatBoost": "#9B72B0",
    "XGBoost": "#D28C3D",
    "Balanced_5Seed_Current31F": "#76B083",
    "RepeatedBalanced_Extended": "#67A9CF",
    "RepeatedOrdinary_GroupedSplit": "#FDB863",
    "GroupKFold_FullyGrouped": "#B2ABD2",
    "LOLO_LeaveOneLiteratureOut": "#BC7C95",
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
# 2. Metrics, saving, and splitting utilities
# ============================================================

def regression_metrics_safe(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "Bias": np.nan}
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    if len(y_true) < 2 or np.nanstd(y_true) == 0:
        r2 = np.nan
    else:
        r2 = r2_score(y_true, y_pred)
    return {
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "RMSE": float(rmse),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "Bias": float(np.mean(y_true - y_pred)),
    }


def weight_dict_to_text(weight_dict: dict) -> str:
    if not weight_dict:
        return ""
    return ";".join([f"{k}:{float(v):.6f}" for k, v in weight_dict.items()])


def save_fig(fig, filename: str):
    png_path = FIG_DIR / filename
    pdf_path = FIG_DIR / filename.replace(".png", ".pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] {png_path}")
    log(f"[FIG] {pdf_path}")


def groups_to_indices(groups: pd.Series, selected_groups: Iterable[str]) -> np.ndarray:
    g = pd.Series(groups).astype(str).reset_index(drop=True)
    selected_groups = set(map(str, selected_groups))
    return np.where(g.isin(selected_groups).values)[0]


def validate_group_split(groups: pd.Series, train_idx, valid_idx, test_idx):
    g = pd.Series(groups).astype(str).reset_index(drop=True)
    train_g = set(g.iloc[train_idx])
    valid_g = set(g.iloc[valid_idx])
    test_g = set(g.iloc[test_idx])
    if train_g & valid_g:
        raise RuntimeError("Group leakage detected: train and validation overlap.")
    if train_g & test_g:
        raise RuntimeError("Group leakage detected: train and test overlap.")
    if valid_g & test_g:
        raise RuntimeError("Group leakage detected: validation and test overlap.")


def split_info_from_indices(strategy: str, split_id: str, seed_or_fold, groups: pd.Series,
                            train_idx, valid_idx, test_idx, extra: dict | None = None) -> dict:
    g = pd.Series(groups).astype(str).reset_index(drop=True)
    info = {
        "Strategy": strategy,
        "Split_ID": split_id,
        "Seed_or_Fold": seed_or_fold,
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "n_test": int(len(test_idx)),
        "n_train_groups": int(g.iloc[train_idx].nunique()),
        "n_valid_groups": int(g.iloc[valid_idx].nunique()),
        "n_test_groups": int(g.iloc[test_idx].nunique()),
        "train_groups": ";".join(sorted(set(g.iloc[train_idx]))),
        "valid_groups": ";".join(sorted(set(g.iloc[valid_idx]))),
        "test_groups": ";".join(sorted(set(g.iloc[test_idx]))),
    }
    if extra:
        info.update(extra)
    return info


def select_validation_groups_from_pool(y: pd.Series, groups: pd.Series, pool_groups: List[str], seed: int,
                                       valid_frac: float = LOLO_VALID_FRAC_OF_NONTEST_GROUPS,
                                       n_candidates: int = VALID_SELECTION_CANDIDATES) -> set:
    """Select validation groups from the non-test pool.

    The selection is based only on non-test groups and target distribution in the
    train/valid side. Test labels/groups are not used except for being excluded.
    """
    g = pd.Series(groups).astype(str).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True).astype(float)
    pool_groups = list(sorted(set(map(str, pool_groups))))
    n_pool = len(pool_groups)
    n_valid = int(max(1, math.ceil(n_pool * valid_frac)))
    if n_pool <= 2:
        raise ValueError("Too few groups in pool for validation selection.")
    n_valid = min(n_valid, n_pool - 1)

    pool_mask = g.isin(pool_groups).values
    target_mean = float(y[pool_mask].mean())
    target_std = float(y[pool_mask].std(ddof=0)) if y[pool_mask].std(ddof=0) > 0 else 1.0
    target_valid_row_frac = valid_frac

    rng = np.random.RandomState(seed + 41000)
    best_score = None
    best_valid = None
    for _ in range(n_candidates):
        valid_groups = set(rng.choice(pool_groups, size=n_valid, replace=False))
        valid_mask = g.isin(valid_groups).values
        train_mask = pool_mask & (~valid_mask)
        if train_mask.sum() == 0 or valid_mask.sum() == 0:
            continue
        valid_mean = float(y[valid_mask].mean())
        train_mean = float(y[train_mask].mean())
        valid_row_frac = valid_mask.sum() / pool_mask.sum()
        # Balance target mean and row fraction, but do not over-optimize.
        score = abs(valid_mean - target_mean) / target_std + 0.5 * abs(train_mean - target_mean) / target_std + abs(valid_row_frac - target_valid_row_frac)
        if best_score is None or score < best_score:
            best_score = score
            best_valid = valid_groups
    if best_valid is None:
        raise RuntimeError("Failed to select validation groups.")
    return best_valid


def ordinary_grouped_split_indices(groups: pd.Series, seed: int, valid_frac: float = 0.20, test_frac: float = 0.20):
    g = pd.Series(groups).astype(str).reset_index(drop=True)
    unique_groups = np.array(sorted(g.unique()))
    rng = np.random.RandomState(seed + 51000)
    perm = list(rng.permutation(unique_groups))
    n_groups = len(perm)
    n_test = int(max(1, math.ceil(n_groups * test_frac)))
    n_valid = int(max(1, math.ceil((n_groups - n_test) * valid_frac)))
    n_valid = min(n_valid, n_groups - n_test - 1)
    test_groups = set(perm[:n_test])
    valid_groups = set(perm[n_test:n_test + n_valid])
    train_groups = set(perm[n_test + n_valid:])
    train_idx = groups_to_indices(groups, train_groups)
    valid_idx = groups_to_indices(groups, valid_groups)
    test_idx = groups_to_indices(groups, test_groups)
    validate_group_split(groups, train_idx, valid_idx, test_idx)
    info = split_info_from_indices("RepeatedOrdinary_GroupedSplit", f"ordinary_seed_{seed}", seed, groups, train_idx, valid_idx, test_idx)
    return train_idx, valid_idx, test_idx, info


def groupkfold_splits(y: pd.Series, groups: pd.Series, n_splits: int = GROUP_KFOLD_N_SPLITS):
    g = pd.Series(groups).astype(str).reset_index(drop=True)
    dummy_X = np.zeros((len(g), 1))
    gkf = GroupKFold(n_splits=n_splits)
    for fold_id, (train_valid_idx, test_idx) in enumerate(gkf.split(dummy_X, y, groups=g), start=1):
        test_groups = set(g.iloc[test_idx])
        pool_groups = sorted(set(g.iloc[train_valid_idx]))
        valid_groups = select_validation_groups_from_pool(y=y, groups=groups, pool_groups=pool_groups, seed=7000 + fold_id)
        train_groups = set(pool_groups) - set(valid_groups)
        train_idx = groups_to_indices(groups, train_groups)
        valid_idx = groups_to_indices(groups, valid_groups)
        validate_group_split(groups, train_idx, valid_idx, test_idx)
        info = split_info_from_indices("GroupKFold_FullyGrouped", f"groupkfold_{fold_id}", fold_id, groups, train_idx, valid_idx, test_idx)
        yield train_idx, valid_idx, test_idx, info


def lolo_splits(y: pd.Series, groups: pd.Series):
    g = pd.Series(groups).astype(str).reset_index(drop=True)
    unique_groups = sorted(g.unique())
    for i, test_group in enumerate(unique_groups, start=1):
        test_groups = {test_group}
        pool_groups = sorted(set(unique_groups) - test_groups)
        valid_groups = select_validation_groups_from_pool(y=y, groups=groups, pool_groups=pool_groups, seed=9000 + i)
        train_groups = set(pool_groups) - set(valid_groups)
        train_idx = groups_to_indices(groups, train_groups)
        valid_idx = groups_to_indices(groups, valid_groups)
        test_idx = groups_to_indices(groups, test_groups)
        validate_group_split(groups, train_idx, valid_idx, test_idx)
        info = split_info_from_indices("LOLO_LeaveOneLiteratureOut", f"lolo_{test_group}", test_group, groups, train_idx, valid_idx, test_idx,
                                       extra={"LOLO_Test_Group": test_group})
        yield train_idx, valid_idx, test_idx, info


# ============================================================
# 3. Strict one-split modeling workflow
# ============================================================

def add_prediction_rows(pred_rows: List[dict], strategy: str, split_id: str, seed_or_fold, method: str,
                        y_test: pd.Series, pred: np.ndarray, test_groups: pd.Series, test_idx: np.ndarray):
    for local_i, (obs, pp, gid, data_idx) in enumerate(zip(y_test, pred, test_groups, test_idx)):
        pred_rows.append({
            "Strategy": strategy,
            "Split_ID": split_id,
            "Seed_or_Fold": seed_or_fold,
            "Method": method,
            "Data_Index": int(data_idx),
            "Local_Test_Index": int(local_i),
            "文献编号": gid,
            "Observed": float(obs),
            "Predicted": float(pp),
            "Residual": float(obs - pp),
        })


def add_test_row(test_rows: List[dict], strategy: str, split_id: str, seed_or_fold, method: str,
                 model_type: str, config_name: str, validation_metrics: dict, weights: dict,
                 y_test: pd.Series, pred_test: np.ndarray, n_test_groups: int,
                 selected_k="mixed", n_features=np.nan, selected_features="mixed"):
    mt = regression_metrics_safe(y_test, pred_test)
    test_rows.append({
        "Strategy": strategy,
        "Split_ID": split_id,
        "Seed_or_Fold": seed_or_fold,
        "Method": method,
        "Model_Type": model_type,
        "Selected_K": selected_k,
        "n_Selected_Features": n_features,
        "Selected_Features": selected_features,
        "Config_Name": config_name,
        "Validation_R2": validation_metrics.get("R2", np.nan),
        "Validation_RMSE": validation_metrics.get("RMSE", np.nan),
        "Validation_MAE": validation_metrics.get("MAE", np.nan),
        "Weights": weight_dict_to_text(weights),
        "Test_R2": mt["R2"],
        "Test_RMSE": mt["RMSE"],
        "Test_MAE": mt["MAE"],
        "Test_MSE": mt["MSE"],
        "Test_Bias": mt["Bias"],
        "n_test": len(y_test),
        "n_test_groups": n_test_groups,
    })



def fit_predict_selected_spec_31n(
    spec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_target: pd.DataFrame,
    seed: int,
    sample_weight=None,
    train_groups: pd.Series | None = None,
) -> np.ndarray:
    """31N-safe predictor for SelectedSpec objects.

    Step 31M's fit_predict_selected_spec supports the core 31M models. Step 31N
    additionally uses PLS for residual-correlation screening, so this wrapper
    routes any non-tree screening model through the real Step-31L non-tree
    predictor while preserving the exact selected columns and config object.
    """
    core_supported_by_31m = {"CatBoost", "XGBoost", "KRR_Poly", "GPR_Matern", "CB-XGB"}
    if spec.name in core_supported_by_31m:
        return s31m.fit_predict_selected_spec(
            spec=spec,
            X_train=X_train,
            y_train=y_train,
            X_target=X_target,
            seed=seed,
            sample_weight=sample_weight,
            train_groups=train_groups,
        )

    # Extra non-tree candidates used only for residual-correlation screening.
    if spec.name in {"PLS", "GPR_RBF", "KRR_RBF", "KRR_Poly", "GPR_Matern"}:
        pred, _ = s31l.fit_predict_non_tree(
            cfg=spec.config_obj,
            X_train=X_train[spec.selected_cols].reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            X_target=X_target[spec.selected_cols].reset_index(drop=True),
            seed=seed,
            sample_weight=sample_weight,
            train_groups=train_groups.reset_index(drop=True) if train_groups is not None else None,
        )
        return np.clip(pred, 0, 100)

    raise ValueError(f"Unknown SelectedSpec name in 31N wrapper: {spec.name}")

def run_one_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                  train_idx: np.ndarray, valid_idx: np.ndarray, test_idx: np.ndarray,
                  split_info: dict, include_extra_models: bool = False):
    strategy = split_info["Strategy"]
    split_id = split_info["Split_ID"]
    seed_or_fold = split_info["Seed_or_Fold"]

    log("\n" + "=" * 100)
    log(f"31N robustness | Strategy={strategy} | Split={split_id}")
    log("=" * 100)

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

    # Training-only feature ranking.
    imp_df = s31f.rank_features_by_rf_importance_weighted(
        X_train, y_train, int(abs(hash(str(seed_or_fold))) % 100000), sample_weight=sample_weight_train
    )

    candidate_parts = []
    specs = {}
    val_preds = {}

    # Tree route: CatBoost, XGBoost, CB-XGB.
    tree_specs, tree_val_preds, tree_cand = s31m.select_tree_and_cbxgb_specs(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        imp_df=imp_df,
        seed=int(abs(hash(str(seed_or_fold))) % 100000),
        sample_weight_train=sample_weight_train,
    )
    specs.update(tree_specs)
    val_preds.update(tree_val_preds)
    candidate_parts.append(tree_cand)

    # KRR_Poly route.
    krr_spec, krr_valid_pred, krr_cand = s31m.select_non_tree_spec(
        family="KRR_Poly",
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        imp_df=imp_df,
        seed=int(abs(hash(str(seed_or_fold))) % 100000),
        sample_weight_train=sample_weight_train,
        train_groups=train_groups,
    )
    specs["KRR_Poly"] = krr_spec
    val_preds["KRR_Poly"] = krr_valid_pred
    candidate_parts.append(krr_cand)

    # Optional extra non-tree models for residual-correlation screening.
    if include_extra_models:
        for family in EXTRA_ANTI_CORR_MODELS:
            try:
                spec, valid_pred, cand = s31m.select_non_tree_spec(
                    family=family,
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    imp_df=imp_df,
                    seed=int(abs(hash(str(seed_or_fold))) % 100000),
                    sample_weight_train=sample_weight_train,
                    train_groups=train_groups,
                )
                specs[family] = spec
                val_preds[family] = valid_pred
                candidate_parts.append(cand)
            except Exception as e:
                log(f"[WARN] Extra model {family} skipped for {strategy}/{split_id}: {e}")

    candidate_df = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()

    base_methods_to_predict = ["KRR_Poly", "CatBoost", "XGBoost", "CB-XGB"]
    if include_extra_models:
        base_methods_to_predict += EXTRA_ANTI_CORR_MODELS

    base_test_preds: Dict[str, np.ndarray] = {}
    test_rows, pred_rows, weight_rows = [], [], []

    for base in base_methods_to_predict:
        if base not in specs:
            continue
        spec = specs[base]
        pred = fit_predict_selected_spec_31n(
            spec=spec,
            X_train=X_train_full,
            y_train=y_train_full,
            X_target=X_test,
            seed=int(abs(hash(str(seed_or_fold))) % 100000),
            sample_weight=sample_weight_train_full,
            train_groups=train_full_groups,
        )
        pred = np.clip(pred, 0, 100)
        base_test_preds[base] = pred
        add_test_row(
            test_rows, strategy, split_id, seed_or_fold, base,
            model_type="base_or_reference",
            config_name=spec.config_name,
            validation_metrics={"R2": spec.validation_r2, "RMSE": spec.validation_rmse, "MAE": spec.validation_mae},
            weights=spec.weights or {},
            y_test=y_test,
            pred_test=pred,
            n_test_groups=test_groups.nunique(),
            selected_k=spec.selected_k,
            n_features=len(spec.selected_cols),
            selected_features=";".join(spec.selected_cols),
        )
        add_prediction_rows(pred_rows, strategy, split_id, seed_or_fold, base, y_test, pred, test_groups, test_idx)

    # KRR + CatBoost and KRR + XGBoost pairwise fusions.
    pairwise_pools = {
        "KRR_CatBoost_Weighted_2Model": ["KRR_Poly", "CatBoost"],
        "KRR_XGBoost_Weighted_2Model": ["KRR_Poly", "XGBoost"],
    }
    for method, pool in pairwise_pools.items():
        if not all(m in val_preds and m in base_test_preds for m in pool):
            continue
        pred_dict_valid = {m: val_preds[m] for m in pool}
        pred_dict_test = {m: base_test_preds[m] for m in pool}
        weights, _ = sf19.search_best_weights(pred_dict_valid, y_valid, int(abs(hash(str(seed_or_fold))) % 100000))
        pred_valid_fused = np.clip(sf19.apply_weights(pred_dict_valid, weights), 0, 100)
        pred_test_fused = np.clip(sf19.apply_weights(pred_dict_test, weights), 0, 100)
        mv = regression_metrics_safe(y_valid, pred_valid_fused)
        add_test_row(
            test_rows, strategy, split_id, seed_or_fold, method,
            model_type="validation_weighted_pairwise_fusion",
            config_name="+".join(pool),
            validation_metrics=mv,
            weights=weights,
            y_test=y_test,
            pred_test=pred_test_fused,
            n_test_groups=test_groups.nunique(),
        )
        for k, v in weights.items():
            weight_rows.append({"Strategy": strategy, "Split_ID": split_id, "Seed_or_Fold": seed_or_fold, "Method": method, "Base_Model": k, "Weight": float(v)})
        add_prediction_rows(pred_rows, strategy, split_id, seed_or_fold, method, y_test, pred_test_fused, test_groups, test_idx)

    split_df = pd.DataFrame([split_info])
    return {
        "split": split_df,
        "candidate": candidate_df,
        "test": pd.DataFrame(test_rows),
        "prediction": pd.DataFrame(pred_rows),
        "weights": pd.DataFrame(weight_rows),
    }


# ============================================================
# 4. Summaries and diagnostics
# ============================================================

def summarize_by_strategy(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, method), sub in pred_df.groupby(["Strategy", "Method"]):
        m = regression_metrics_safe(sub["Observed"], sub["Predicted"])
        split_sub = test_df[(test_df["Strategy"] == strategy) & (test_df["Method"] == method)]
        rows.append({
            "Strategy": strategy,
            "Method": method,
            "Pooled_R2": m["R2"],
            "Pooled_RMSE": m["RMSE"],
            "Pooled_MAE": m["MAE"],
            "Pooled_MSE": m["MSE"],
            "Pooled_Bias": m["Bias"],
            "Split_R2_mean": split_sub["Test_R2"].mean(),
            "Split_R2_std": split_sub["Test_R2"].std(),
            "Split_RMSE_mean": split_sub["Test_RMSE"].mean(),
            "Split_RMSE_std": split_sub["Test_RMSE"].std(),
            "Split_MAE_mean": split_sub["Test_MAE"].mean(),
            "Split_MAE_std": split_sub["Test_MAE"].std(),
            "n_splits": split_sub["Split_ID"].nunique(),
            "n_pooled": len(sub),
            "n_test_literature_groups": sub["文献编号"].nunique(),
        })
    out = pd.DataFrame(rows)
    out["Strategy_Order"] = out["Strategy"].map({s: i for i, s in enumerate(STRATEGY_ORDER)}).fillna(999)
    out["Method_Order"] = out["Method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(999)
    return out.sort_values(["Strategy_Order", "Pooled_RMSE", "Method_Order"]).drop(columns=["Strategy_Order", "Method_Order"]).reset_index(drop=True)


def compute_paper_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, method, paper), sub in pred_df.groupby(["Strategy", "Method", "文献编号"]):
        m = regression_metrics_safe(sub["Observed"], sub["Predicted"])
        rows.append({
            "Strategy": strategy,
            "Method": method,
            "文献编号": paper,
            "Paper_R2": m["R2"],
            "Paper_RMSE": m["RMSE"],
            "Paper_MAE": m["MAE"],
            "Paper_Bias": m["Bias"],
            "n_rows": len(sub),
            "Observed_mean": float(sub["Observed"].mean()),
            "Predicted_mean": float(sub["Predicted"].mean()),
        })
    return pd.DataFrame(rows)


def residual_correlation_by_strategy(pred_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    corr_rows = []
    low_rows = []
    for strategy, sub in pred_df.groupby("Strategy"):
        piv = sub.pivot_table(index=["Split_ID", "Data_Index"], columns="Method", values="Residual", aggfunc="mean")
        corr = piv.corr(method="pearson")
        for m1 in corr.index:
            row = {"Strategy": strategy, "Method": m1}
            for m2 in corr.columns:
                row[m2] = corr.loc[m1, m2]
            corr_rows.append(row)
        if "KRR_Poly" in corr.index:
            for method in corr.columns:
                if method == "KRR_Poly":
                    continue
                low_rows.append({
                    "Strategy": strategy,
                    "Reference_Method": "KRR_Poly",
                    "Candidate_Method": method,
                    "Residual_Correlation_with_KRR": corr.loc["KRR_Poly", method],
                    "Interpretation": "negative_or_low_is_more_complementary",
                })
    corr_df = pd.DataFrame(corr_rows)
    low_df = pd.DataFrame(low_rows)
    if len(low_df):
        low_df = low_df.sort_values(["Strategy", "Residual_Correlation_with_KRR"]).reset_index(drop=True)
    return corr_df, low_df


def macro_strategy_summary(paper_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, method), sub in paper_df.groupby(["Strategy", "Method"]):
        rows.append({
            "Strategy": strategy,
            "Method": method,
            "PaperMacro_RMSE_mean": sub["Paper_RMSE"].mean(),
            "PaperMacro_RMSE_median": sub["Paper_RMSE"].median(),
            "PaperMacro_MAE_mean": sub["Paper_MAE"].mean(),
            "PaperMacro_MAE_median": sub["Paper_MAE"].median(),
            "PaperMacro_Bias_mean": sub["Paper_Bias"].mean(),
            "n_papers": sub["文献编号"].nunique(),
        })
    return pd.DataFrame(rows)


# ============================================================
# 5. Figures
# ============================================================

def plot_strategy_metrics(summary_df: pd.DataFrame):
    # Keep only core methods for readable strategy-level comparison.
    methods = ["KRR_Poly", "KRR_CatBoost_Weighted_2Model", "CB-XGB", "CatBoost", "XGBoost"]
    data = summary_df[summary_df["Method"].isin(methods)].copy()
    strategies = [s for s in STRATEGY_ORDER if s in data["Strategy"].unique()]
    x = np.arange(len(methods))
    width = 0.15

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    specs = [("Pooled_R2", "Pooled $R^2$", "{:.3f}"), ("Pooled_RMSE", "Pooled RMSE", "{:.2f}"), ("Pooled_MAE", "Pooled MAE", "{:.2f}")]
    for ax, (col, ylabel, fmt) in zip(axes, specs):
        for i, strategy in enumerate(strategies):
            sub = data[data["Strategy"] == strategy].set_index("Method")
            vals = [sub.loc[m, col] if m in sub.index else np.nan for m in methods]
            offset = (i - (len(strategies)-1)/2) * width
            bars = ax.bar(x + offset, vals, width=width, label=strategy, color=COLORS.get(strategy, COLORS["gray"]), edgecolor="black", linewidth=0.35, alpha=0.85)
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x()+b.get_width()/2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=6, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        if col == "Pooled_R2":
            ax.axhline(0, color="black", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("31N grouped-validation robustness across fully grouped strategies", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01, "All strategies keep literature groups non-overlapping; models/configs selected without test labels.", ha="center", fontsize=8)
    save_fig(fig, "127_31n_grouped_validation_strategy_metrics.png")


def plot_krr_vs_baselines(summary_df: pd.DataFrame):
    methods = ["KRR_Poly", "KRR_CatBoost_Weighted_2Model", "CB-XGB", "CatBoost", "XGBoost"]
    data = summary_df[summary_df["Method"].isin(methods)].copy()
    strategies = [s for s in STRATEGY_ORDER if s in data["Strategy"].unique()]
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    for method in methods:
        vals = []
        for strategy in strategies:
            sub = data[(data["Strategy"] == strategy) & (data["Method"] == method)]
            vals.append(float(sub["Pooled_RMSE"].iloc[0]) if len(sub) else np.nan)
        ax.plot(strategies, vals, marker="o", linewidth=1.5, label=method, color=COLORS.get(method, None))
        for x0, y0 in zip(strategies, vals):
            if np.isfinite(y0):
                ax.text(x0, y0, f"{y0:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Pooled RMSE")
    ax.set_title("KRR_Poly vs tree baselines under different fully grouped validations")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    ax.set_xticklabels(strategies, rotation=28, ha="right")
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, "128_31n_krr_vs_baselines_by_strategy.png")


def plot_lolo_paper_rmse(paper_df: pd.DataFrame):
    data = paper_df[(paper_df["Strategy"] == "LOLO_LeaveOneLiteratureOut") & (paper_df["Method"].isin(["KRR_Poly", "CB-XGB", "CatBoost"]))].copy()
    if data.empty:
        return
    # Show top 20 KRR high-error papers.
    krr = data[data["Method"] == "KRR_Poly"].sort_values("Paper_RMSE", ascending=False).head(20)
    top_papers = krr["文献编号"].tolist()
    methods = ["KRR_Poly", "CB-XGB", "CatBoost"]
    x = np.arange(len(top_papers))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 5.2))
    for i, method in enumerate(methods):
        sub = data[(data["Method"] == method) & (data["文献编号"].isin(top_papers))].set_index("文献编号")
        vals = [sub.loc[p, "Paper_RMSE"] if p in sub.index else np.nan for p in top_papers]
        bars = ax.bar(x + (i-1)*width, vals, width=width, label=method, color=COLORS.get(method, COLORS["gray"]), edgecolor="black", linewidth=0.35, alpha=0.85)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.1f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(top_papers, rotation=45, ha="right")
    ax.set_ylabel("LOLO paper-level RMSE")
    ax.set_title("Highest-error LOLO literature groups")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False)
    save_fig(fig, "129_31n_lolo_paper_level_rmse.png")


def plot_residual_correlation_screen(low_corr_df: pd.DataFrame):
    data = low_corr_df.copy()
    if data.empty:
        return
    # Show lowest correlation candidates by strategy.
    data = data.sort_values(["Strategy", "Residual_Correlation_with_KRR"])
    strategies = [s for s in STRATEGY_ORDER if s in data["Strategy"].unique()]
    fig, axes = plt.subplots(len(strategies), 1, figsize=(10, max(3.2, 2.6 * len(strategies))), sharex=True)
    if len(strategies) == 1:
        axes = [axes]
    for ax, strategy in zip(axes, strategies):
        sub = data[data["Strategy"] == strategy].head(8).copy()
        labels = sub["Candidate_Method"].tolist()
        vals = sub["Residual_Correlation_with_KRR"].astype(float).values
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, vals, color="#9EBCDA", edgecolor="black", linewidth=0.35)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(strategy)
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        for b, v in zip(bars, vals):
            ax.text(v + 0.01, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", fontsize=8)
    axes[-1].set_xlabel("Pearson residual correlation with KRR_Poly")
    fig.suptitle("Residual-correlation screen: lower/negative values would indicate complementarity", fontsize=14, fontweight="bold")
    save_fig(fig, "130_31n_residual_correlation_screen.png")


# ============================================================
# 6. Decision report
# ============================================================

def write_decision_report(summary_df: pd.DataFrame, macro_df: pd.DataFrame, low_corr_df: pd.DataFrame, split_df: pd.DataFrame):
    path = TABLE_DIR / "31n_grouped_validation_decision.txt"
    lines = []
    lines.append("=" * 100)
    lines.append("Step 31N KRR_Poly grouped-validation robustness report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Evaluate whether KRR_Poly remains robust under multiple fully grouped validation forms and screen for residual-complementary models.")
    lines.append("")
    lines.append("Validation forms included:")
    for s in STRATEGY_ORDER:
        n = split_df[split_df["Strategy"] == s]["Split_ID"].nunique() if "Strategy" in split_df else 0
        lines.append(f"- {s}: {n} splits/folds")
    lines.append("")
    lines.append("Pooled performance summary:")
    display_cols = ["Strategy", "Method", "Pooled_R2", "Pooled_RMSE", "Pooled_MAE", "Pooled_Bias", "n_splits", "n_pooled"]
    lines.append(summary_df[display_cols].to_string(index=False))
    lines.append("")
    lines.append("Paper-level macro summary:")
    if not macro_df.empty:
        lines.append(macro_df.to_string(index=False))
    lines.append("")
    lines.append("Residual-correlation screen with KRR_Poly:")
    if not low_corr_df.empty:
        lines.append(low_corr_df.to_string(index=False))
        min_corr = low_corr_df["Residual_Correlation_with_KRR"].min()
        if pd.notna(min_corr) and min_corr < 0:
            lines.append("")
            lines.append("Decision note: at least one candidate showed negative residual correlation with KRR_Poly. It may be considered for a cautious validation-only fusion test.")
        else:
            lines.append("")
            lines.append("Decision note: no tested candidate showed negative residual correlation with KRR_Poly. Low positive correlation, not anti-correlation, is the best available complementarity signal.")
    else:
        lines.append("No residual-correlation table available.")
    lines.append("")
    lines.append("Recommended interpretation guide:")
    lines.append("1. If KRR_Poly remains best or near-best across balanced, ordinary, GroupKFold, and LOLO, keep it as grouped-generalization main candidate.")
    lines.append("2. If KRR_Poly collapses under ordinary/LOLO while trees are stronger, the main model choice must be validation-form-specific.")
    lines.append("3. If all fully grouped forms remain far below R2=0.5, proceed to applicability-domain analysis as the technical safeguard rather than forcing more global fusion.")
    lines.append("4. Paper-level macro metrics should be reported beside pooled metrics, because large literature groups can dominate pooled results.")
    lines.append("=" * 100)

    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[REPORT] {path}")


# ============================================================
# 7. Main workflow
# ============================================================

def main():
    start_time = time.time()
    log("\n" + "=" * 100)
    log("Step 31N: KRR_Poly grouped-validation robustness")
    log("=" * 100)
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"BEST_FEATURE_SPACE = {BEST_FEATURE_SPACE}")
    log("Planned validation forms: " + ", ".join(STRATEGY_ORDER))
    log("=" * 100)

    y, X_raw, X_eng, X_mix_base = sf19.prepare_feature_sets()
    y = y.reset_index(drop=True).astype(float)
    X_mix_base = X_mix_base.reset_index(drop=True)

    log("Building confirmed 31K/31F feature route...")
    feature_spaces, audit_df, block_df = s31k.make_feature_spaces(X_mix_base, expected_n=len(y))
    if BEST_FEATURE_SPACE not in feature_spaces:
        raise RuntimeError(f"Cannot find required feature space: {BEST_FEATURE_SPACE}")
    X = feature_spaces[BEST_FEATURE_SPACE].reset_index(drop=True)

    groups = s31f.load_group_labels(expected_n=len(y))
    groups = pd.Series(groups).astype(str).reset_index(drop=True)

    # Needed by the 31F balanced split scorer.
    s31f.set_balanced_split_context(X, y, groups)

    all_splits, all_candidates, all_tests, all_predictions, all_weights = [], [], [], [], []

    # A. Current 5 balanced seeds.
    for seed in BALANCED_CURRENT_SEEDS:
        train_idx, valid_idx, test_idx, info = s31f.grouped_split_indices(groups, seed)
        info = dict(info)
        info["Strategy"] = "Balanced_5Seed_Current31F"
        info["Split_ID"] = f"balanced5_seed_{seed}"
        info["Seed_or_Fold"] = seed
        include_extra = info["Strategy"] in INCLUDE_EXTRA_ANTI_CORR_MODELS_IN
        result = run_one_split(X, y, groups, train_idx, valid_idx, test_idx, info, include_extra_models=include_extra)
        all_splits.append(result["split"]); all_candidates.append(result["candidate"]); all_tests.append(result["test"]); all_predictions.append(result["prediction"]); all_weights.append(result["weights"])

    # B. Extended balanced seeds.
    for seed in REPEATED_BALANCED_SEEDS:
        train_idx, valid_idx, test_idx, info = s31f.grouped_split_indices(groups, seed)
        info = dict(info)
        info["Strategy"] = "RepeatedBalanced_Extended"
        info["Split_ID"] = f"balanced_ext_seed_{seed}"
        info["Seed_or_Fold"] = seed
        include_extra = info["Strategy"] in INCLUDE_EXTRA_ANTI_CORR_MODELS_IN
        result = run_one_split(X, y, groups, train_idx, valid_idx, test_idx, info, include_extra_models=include_extra)
        all_splits.append(result["split"]); all_candidates.append(result["candidate"]); all_tests.append(result["test"]); all_predictions.append(result["prediction"]); all_weights.append(result["weights"])

    # C. Ordinary repeated fully grouped splits.
    for seed in ORDINARY_GROUPED_SEEDS:
        train_idx, valid_idx, test_idx, info = ordinary_grouped_split_indices(groups, seed)
        result = run_one_split(X, y, groups, train_idx, valid_idx, test_idx, info, include_extra_models=False)
        all_splits.append(result["split"]); all_candidates.append(result["candidate"]); all_tests.append(result["test"]); all_predictions.append(result["prediction"]); all_weights.append(result["weights"])

    # D. GroupKFold fully grouped.
    for train_idx, valid_idx, test_idx, info in groupkfold_splits(y, groups, n_splits=GROUP_KFOLD_N_SPLITS):
        result = run_one_split(X, y, groups, train_idx, valid_idx, test_idx, info, include_extra_models=False)
        all_splits.append(result["split"]); all_candidates.append(result["candidate"]); all_tests.append(result["test"]); all_predictions.append(result["prediction"]); all_weights.append(result["weights"])

    # E. Leave-one-literature-out.
    for train_idx, valid_idx, test_idx, info in lolo_splits(y, groups):
        result = run_one_split(X, y, groups, train_idx, valid_idx, test_idx, info, include_extra_models=False)
        all_splits.append(result["split"]); all_candidates.append(result["candidate"]); all_tests.append(result["test"]); all_predictions.append(result["prediction"]); all_weights.append(result["weights"])

    split_df = pd.concat(all_splits, ignore_index=True)
    cand_df = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    test_df = pd.concat(all_tests, ignore_index=True)
    pred_df = pd.concat(all_predictions, ignore_index=True)
    weights_df = pd.concat(all_weights, ignore_index=True) if any(len(x) for x in all_weights) else pd.DataFrame(columns=["Strategy", "Split_ID", "Seed_or_Fold", "Method", "Base_Model", "Weight"])

    summary_df = summarize_by_strategy(test_df, pred_df)
    paper_df = compute_paper_metrics(pred_df)
    macro_df = macro_strategy_summary(paper_df)
    corr_df, low_corr_df = residual_correlation_by_strategy(pred_df)

    outputs = {
        "31n_grouped_validation_split_summary.csv": split_df,
        "31n_grouped_validation_validation_candidates.csv": cand_df,
        "31n_grouped_validation_test_results.csv": test_df,
        "31n_grouped_validation_predictions.csv": pred_df,
        "31n_grouped_validation_fusion_weights.csv": weights_df,
        "31n_grouped_validation_summary.csv": summary_df,
        "31n_grouped_validation_paper_metrics.csv": paper_df,
        "31n_grouped_validation_paper_macro_summary.csv": macro_df,
        "31n_grouped_validation_residual_correlation_by_strategy.csv": corr_df,
        "31n_low_residual_correlation_screen.csv": low_corr_df,
    }
    for filename, df in outputs.items():
        path = TABLE_DIR / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        log(f"[TABLE] {path}")

    log("\n=== Step 31N summary ===")
    print(summary_df.to_string(index=False))
    log("\n=== Lowest residual correlations with KRR_Poly ===")
    print(low_corr_df.groupby("Strategy").head(5).to_string(index=False) if len(low_corr_df) else "No correlation rows.")

    write_decision_report(summary_df, macro_df, low_corr_df, split_df)

    log("Generating figures...")
    plot_strategy_metrics(summary_df)
    plot_krr_vs_baselines(summary_df)
    plot_lolo_paper_rmse(paper_df)
    plot_residual_correlation_screen(low_corr_df)

    elapsed = time.time() - start_time
    log("\n" + "=" * 100)
    log("[DONE] Step 31N grouped-validation robustness completed.")
    log(f"[TIME] Total runtime: {elapsed:.2f} s ({elapsed/60:.2f} min)")
    log("=" * 100)


if __name__ == "__main__":
    main()
