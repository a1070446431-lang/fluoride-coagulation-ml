# -*- coding: utf-8 -*-
"""
31o_grouped_oof_residual_anti_correlation_screen.py

Purpose
-------
Residual anti-correlation / residual-correction screening after Steps 31L-31M2.

Current evidence:
- KRR_Poly（多项式核岭回归） is the best all-domain grouped-validation candidate.
- Standard non-negative global fusion with CatBoost/XGBoost/CB-XGB/GPR did not
  improve over KRR_Poly.
- Residuals of KRR_Poly and tree models are positively correlated, so ordinary
  weighted fusion has limited value.

This script therefore tests a stricter and more targeted route:
    grouped OOF residual learning（分组折外残差学习）

Strict rules
------------
1. Test residuals are NEVER used for training, candidate selection, residual-model
   selection, residual-feature selection, or shrinkage selection.
2. Base KRR_Poly configuration is selected only by validation RMSE, following the
   real Step-31L/31M workflow.
3. Base-model OOF predictions on train+valid literature groups are generated with
   GroupKFold（分组K折）, so residual targets for residual learning are out-of-fold.
4. Residual-correction candidates are selected only by grouped CV on train+valid
   OOF residuals.
5. Final selected residual model is trained on all train+valid OOF residuals and
   then applied to held-out test literature groups.
6. 文献编号 is used only for grouped splitting, grouped OOF, and literature-group
   sample weighting; it is never used as a model input feature.
7. Total runtime is printed at the end.

Candidate residual-correction models
------------------------------------
- Ridge residual model（岭回归残差模型）
- ElasticNet residual model（弹性网络残差模型）
- Huber residual model（Huber稳健残差模型）
- KNN residual model（K近邻残差模型）
- Kernel Ridge residual model（核岭残差模型）
- SVR residual model（支持向量残差模型）
- HGB residual model（直方图梯度提升残差模型）
- RandomForest / ExtraTrees residual models（随机森林/极端随机树残差模型）

Residual feature sets
---------------------
A. base_preds_only：only OOF base predictions and disagreement signals.
B. base_preds_plus_krr_features：base predictions + compact KRR selected features.

Outputs
-------
Tables:
    outputs/tables/31o_residual_learning_candidate_cv.csv
    outputs/tables/31o_residual_learning_test_results.csv
    outputs/tables/31o_residual_learning_predictions.csv
    outputs/tables/31o_residual_learning_summary.csv
    outputs/tables/31o_residual_learning_selected_configs.csv
    outputs/tables/31o_oof_residual_correlation_screen.csv
    outputs/tables/31o_test_residual_correlation_matrix.csv
    outputs/tables/31o_residual_learning_decision.txt

Figures:
    outputs/figures/131_31o_residual_learning_metrics.png/pdf
    outputs/figures/132_31o_cv_to_test_gain_transfer.png/pdf
    outputs/figures/133_31o_oof_residual_correlation_screen.png/pdf
    outputs/figures/134_31o_residual_learning_pooled_predicted_vs_actual.png/pdf
"""

from __future__ import annotations

import math
import time
import warnings
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor

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
    log(f"Loading real project script: {filename}")
    log(str(module_path))
    log("=" * 100)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # needed for Python 3.12 + dataclass
    spec.loader.exec_module(module)
    return module


# Reuse the fixed real Step-31M script. It imports 31L/31K/31G/31F/script-19.
s31m = load_module_from_file("31m_hybrid_krr_tree_fusion_oof_stacking_fixed.py", "s31m_real_for_31o")
s31l = s31m.s31l
s31k = s31m.s31k
s31g = s31m.s31g
s31f = s31m.s31f
sf19 = s31m.sf19

PROJECT_ROOT = Path(s31m.PROJECT_ROOT)
FIG_DIR = Path(s31m.FIG_DIR)
TABLE_DIR = Path(s31m.TABLE_DIR)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Settings
# ============================================================

SEEDS = list(getattr(sf19, "SEEDS", [7, 21, 42, 66, 88]))
BEST_FEATURE_SPACE = "Current31F_noFine_plus_31B_mechanism"
K_OPTIONS = list(getattr(s31m, "K_OPTIONS", [12, 18, 25, 35, 45, 60, "all"]))

MAX_RESIDUAL_RAW_FEATURES = 25  # controlled to avoid overfitting in 27 literature groups
BASE_FOR_RESIDUAL = "KRR_Poly"
BASE_MODEL_NAMES = ["KRR_Poly", "CatBoost", "XGBoost", "CB-XGB", "GPR_Matern"]

METHOD_ORDER = [
    "KRR_Poly",
    "KRR_ResidualCorrected_BestOOF",
    "KRR_ResidualCorrected_Ridge",
    "KRR_ResidualCorrected_Huber",
    "KRR_ResidualCorrected_ElasticNet",
    "KRR_ResidualCorrected_KRR_RBF",
    "KRR_ResidualCorrected_KNN",
    "KRR_ResidualCorrected_SVR_RBF",
    "KRR_ResidualCorrected_HGB",
    "KRR_ResidualCorrected_RF",
    "KRR_ResidualCorrected_ET",
    "CB-XGB",
    "CatBoost",
    "XGBoost",
    "GPR_Matern",
]

COLORS = {
    "KRR_Poly": "#76B083",
    "KRR_ResidualCorrected_BestOOF": "#D95F02",
    "KRR_ResidualCorrected_Ridge": "#7570B3",
    "KRR_ResidualCorrected_Huber": "#E7298A",
    "KRR_ResidualCorrected_ElasticNet": "#66A61E",
    "KRR_ResidualCorrected_KRR_RBF": "#E6AB02",
    "KRR_ResidualCorrected_KNN": "#A6761D",
    "KRR_ResidualCorrected_SVR_RBF": "#1B9E77",
    "KRR_ResidualCorrected_HGB": "#386CB0",
    "KRR_ResidualCorrected_RF": "#F0027F",
    "KRR_ResidualCorrected_ET": "#BF5B17",
    "CB-XGB": "#808080",
    "CatBoost": "#9B72B0",
    "XGBoost": "#D28C3D",
    "GPR_Matern": "#B06D84",
    "red": "#FF0000",
    "grid": "#E6E6E6",
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
# 2. Generic utilities
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


def save_fig(fig, filename: str):
    png = FIG_DIR / filename
    fig.savefig(png, bbox_inches="tight")
    pdf = png.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] {png}")


def corr_safe(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def summarize_results(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in test_df.groupby("Method"):
        pooled = pred_df[pred_df["Method"] == method]
        pooled_m = regression_metrics(pooled["Observed"], pooled["Predicted"])
        rows.append({
            "Method": method,
            "R2_mean": sub["R2"].mean(),
            "R2_std": sub["R2"].std(ddof=0),
            "RMSE_mean": sub["RMSE"].mean(),
            "RMSE_std": sub["RMSE"].std(ddof=0),
            "MAE_mean": sub["MAE"].mean(),
            "MAE_std": sub["MAE"].std(ddof=0),
            "Pooled_R2": pooled_m["R2"],
            "Pooled_RMSE": pooled_m["RMSE"],
            "Pooled_MAE": pooled_m["MAE"],
            "Pooled_Bias": pooled_m["Bias"],
            "n_pooled": len(pooled),
            "n_literature_groups_pooled": pooled["文献编号"].nunique(),
        })
    out = pd.DataFrame(rows)
    out["_order"] = out["Method"].apply(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 999)
    out = out.sort_values(["Pooled_RMSE", "_order"], ascending=[True, True]).drop(columns="_order")
    return out.reset_index(drop=True)


def compute_residual_corr(pred_df: pd.DataFrame) -> pd.DataFrame:
    pivot = pred_df.pivot_table(
        index=["Seed", "Data_Index"],
        columns="Method",
        values="Residual",
        aggfunc="mean",
    )
    corr = pivot.corr(method="pearson")
    order = [m for m in METHOD_ORDER if m in corr.index]
    return corr.reindex(index=order, columns=order)


# ============================================================
# 3. Residual-correction candidates
# ============================================================

@dataclass
class ResidualConfig:
    method_group: str
    config_name: str
    model_kind: str
    feature_mode: str
    params: Dict[str, Any]
    shrinkage: float


def make_residual_configs() -> list[ResidualConfig]:
    configs: list[ResidualConfig] = []
    feature_modes = ["base_preds_only", "base_preds_plus_krr_features"]
    shrinkages = [0.25, 0.50, 0.75, 1.00]

    def add(group, kind, feature_mode, params, shrinkage):
        pname = "_".join(f"{k}{str(v).replace('.', 'p')}" for k, v in params.items())
        configs.append(ResidualConfig(
            method_group=group,
            config_name=f"{group}__{feature_mode}__{pname}__sh{str(shrinkage).replace('.', 'p')}",
            model_kind=kind,
            feature_mode=feature_mode,
            params=params,
            shrinkage=shrinkage,
        ))

    for fm in feature_modes:
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            for sh in shrinkages:
                add("Ridge", "ridge", fm, {"alpha": alpha}, sh)

        for alpha in [0.001, 0.01, 0.1, 1.0]:
            for l1_ratio in [0.1, 0.5, 0.8]:
                for sh in [0.25, 0.5, 0.75]:
                    add("ElasticNet", "elasticnet", fm, {"alpha": alpha, "l1_ratio": l1_ratio}, sh)

        for alpha in [0.0001, 0.001, 0.01, 0.1]:
            for epsilon in [1.35, 1.75, 2.0]:
                for sh in [0.25, 0.5, 0.75]:
                    add("Huber", "huber", fm, {"alpha": alpha, "epsilon": epsilon}, sh)

    # More flexible residual correctors: use compact feature set only to avoid
    # fitting arbitrary high-dimensional residual noise.
    fm = "base_preds_plus_krr_features"
    for n_neighbors in [10, 20, 35, 50]:
        for weights in ["uniform", "distance"]:
            for sh in [0.25, 0.5, 0.75]:
                add("KNN", "knn", fm, {"n_neighbors": n_neighbors, "weights": weights}, sh)

    for alpha in [1.0, 10.0, 30.0]:
        for gamma in [0.001, 0.01, 0.05, 0.1]:
            for sh in [0.25, 0.5, 0.75]:
                add("KRR_RBF", "krr_rbf", fm, {"alpha": alpha, "gamma": gamma}, sh)

    for C in [1.0, 5.0, 10.0]:
        for epsilon in [2.0, 5.0, 8.0]:
            for sh in [0.25, 0.5, 0.75]:
                add("SVR_RBF", "svr_rbf", fm, {"C": C, "epsilon": epsilon, "gamma": "scale"}, sh)

    for max_leaf_nodes in [7, 15, 31]:
        for l2_regularization in [0.1, 1.0, 10.0]:
            for sh in [0.25, 0.5, 0.75]:
                add("HGB", "hgb", fm, {"max_leaf_nodes": max_leaf_nodes, "l2_regularization": l2_regularization}, sh)

    for max_depth in [3, 5, None]:
        for min_samples_leaf in [3, 8, 15]:
            for sh in [0.25, 0.5]:
                add("RF", "rf", fm, {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf}, sh)
                add("ET", "et", fm, {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf}, sh)

    return configs


def build_residual_feature_frame(
    X: pd.DataFrame,
    base_pred_dict: Dict[str, np.ndarray],
    krr_selected_cols: List[str],
    feature_mode: str,
) -> pd.DataFrame:
    df = pd.DataFrame(index=np.arange(len(X)))
    # Base OOF/test prediction signals.
    for name in ["KRR_Poly", "CatBoost", "XGBoost", "CB-XGB", "GPR_Matern"]:
        if name in base_pred_dict:
            df[f"pred_{name}"] = np.asarray(base_pred_dict[name], dtype=float)

    if "KRR_Poly" in base_pred_dict and "CatBoost" in base_pred_dict:
        df["diff_KRR_minus_CatBoost"] = np.asarray(base_pred_dict["KRR_Poly"]) - np.asarray(base_pred_dict["CatBoost"])
    if "KRR_Poly" in base_pred_dict and "XGBoost" in base_pred_dict:
        df["diff_KRR_minus_XGBoost"] = np.asarray(base_pred_dict["KRR_Poly"]) - np.asarray(base_pred_dict["XGBoost"])
    if "CatBoost" in base_pred_dict and "XGBoost" in base_pred_dict:
        df["diff_CatBoost_minus_XGBoost"] = np.asarray(base_pred_dict["CatBoost"]) - np.asarray(base_pred_dict["XGBoost"])

    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if pred_cols:
        pred_values = df[pred_cols].values.astype(float)
        df["pred_mean"] = np.nanmean(pred_values, axis=1)
        df["pred_std"] = np.nanstd(pred_values, axis=1)
        df["pred_min"] = np.nanmin(pred_values, axis=1)
        df["pred_max"] = np.nanmax(pred_values, axis=1)

    if feature_mode == "base_preds_plus_krr_features":
        # Use the selected KRR features as compact mechanism/operating descriptors.
        # This is not a simplification; it controls residual-learning overfitting.
        raw_cols = [c for c in krr_selected_cols if c in X.columns][:MAX_RESIDUAL_RAW_FEATURES]
        if raw_cols:
            raw = X[raw_cols].reset_index(drop=True).copy()
            # Prefix to avoid collisions with base prediction columns.
            raw.columns = [f"feat_{str(c)}" for c in raw.columns]
            df = pd.concat([df.reset_index(drop=True), raw.reset_index(drop=True)], axis=1)
    elif feature_mode != "base_preds_only":
        raise ValueError(f"Unknown residual feature mode: {feature_mode}")

    return df.reset_index(drop=True)


def fit_predict_residual_model(
    cfg: ResidualConfig,
    X_train_meta: pd.DataFrame,
    residual_train: pd.Series,
    X_target_meta: pd.DataFrame,
    seed: int,
    sample_weight=None,
) -> np.ndarray:
    Z_train, Z_target = s31l.preprocess_fit_transform(
        X_train_meta.reset_index(drop=True),
        X_target_meta.reset_index(drop=True),
    )
    residual_train = pd.Series(residual_train).reset_index(drop=True).astype(float)
    sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    kind = cfg.model_kind
    p = cfg.params

    if kind == "ridge":
        model = Ridge(alpha=float(p["alpha"]), random_state=seed)
    elif kind == "elasticnet":
        model = ElasticNet(alpha=float(p["alpha"]), l1_ratio=float(p["l1_ratio"]), max_iter=50000, random_state=seed)
    elif kind == "huber":
        model = HuberRegressor(alpha=float(p["alpha"]), epsilon=float(p["epsilon"]), max_iter=1000)
    elif kind == "knn":
        n_neighbors = int(min(int(p["n_neighbors"]), max(1, len(residual_train))))
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=str(p["weights"]))
    elif kind == "krr_rbf":
        model = KernelRidge(kernel="rbf", alpha=float(p["alpha"]), gamma=float(p["gamma"]))
    elif kind == "svr_rbf":
        model = SVR(kernel="rbf", C=float(p["C"]), epsilon=float(p["epsilon"]), gamma=p.get("gamma", "scale"))
    elif kind == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.035,
            max_leaf_nodes=int(p["max_leaf_nodes"]),
            min_samples_leaf=10,
            l2_regularization=float(p["l2_regularization"]),
            random_state=seed,
        )
    elif kind == "rf":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=p["max_depth"],
            min_samples_leaf=int(p["min_samples_leaf"]),
            max_features=0.7,
            random_state=seed,
            n_jobs=-1,
        )
    elif kind == "et":
        model = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=p["max_depth"],
            min_samples_leaf=int(p["min_samples_leaf"]),
            max_features=0.7,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown residual model kind: {kind}")

    if sample_weight is not None:
        try:
            model.fit(Z_train, residual_train, sample_weight=sample_weight)
        except TypeError:
            model.fit(Z_train, residual_train)
    else:
        model.fit(Z_train, residual_train)

    correction = np.asarray(model.predict(Z_target), dtype=float)
    correction = np.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
    return correction * float(cfg.shrinkage)


def grouped_cv_select_residual_correction(
    X_meta: pd.DataFrame,
    y_true: pd.Series,
    base_pred: np.ndarray,
    groups: pd.Series,
    configs: list[ResidualConfig],
    seed: int,
) -> Tuple[ResidualConfig, pd.DataFrame, Dict[str, np.ndarray]]:
    """Select residual correction model with grouped CV on OOF residuals only."""
    y_true = pd.Series(y_true).reset_index(drop=True).astype(float)
    groups = pd.Series(groups).reset_index(drop=True).astype(str)
    base_pred = np.asarray(base_pred, dtype=float)
    base_residual = y_true.values - base_pred

    n_splits = min(5, groups.nunique())
    if n_splits < 2:
        raise RuntimeError("Not enough groups for grouped residual-CV.")
    gkf = GroupKFold(n_splits=n_splits)

    rows = []
    oof_final_by_config: Dict[str, np.ndarray] = {}
    oof_corr_by_config: Dict[str, np.ndarray] = {}

    base_oof_m = regression_metrics(y_true, base_pred)

    for i, cfg in enumerate(configs, start=1):
        log(f"    [ResidualCV] {i}/{len(configs)} | {cfg.config_name}")
        oof_corr = np.full(len(y_true), np.nan, dtype=float)
        fold_rows = []
        failed = False
        err_msg = ""

        X_cfg = build_residual_feature_frame(
            X=X_meta["__X_original__"] if "__X_original__" in X_meta.attrs else X_meta,
            base_pred_dict=X_meta.attrs.get("base_pred_dict", {}),
            krr_selected_cols=X_meta.attrs.get("krr_selected_cols", []),
            feature_mode=cfg.feature_mode,
        ) if False else None

        # X_meta is already built according to cfg.feature_mode outside; choose columns here.
        X_cfg = X_meta[cfg.feature_mode].reset_index(drop=True)

        for fold_id, (tr, va) in enumerate(gkf.split(X_cfg, y_true, groups=groups), start=1):
            try:
                sw_tr = s31f.make_literature_group_weights(groups.iloc[tr].reset_index(drop=True))
                correction = fit_predict_residual_model(
                    cfg=cfg,
                    X_train_meta=X_cfg.iloc[tr].reset_index(drop=True),
                    residual_train=pd.Series(base_residual[tr]),
                    X_target_meta=X_cfg.iloc[va].reset_index(drop=True),
                    seed=seed + 31000 + fold_id,
                    sample_weight=sw_tr,
                )
                oof_corr[va] = correction
                final_pred = np.clip(base_pred[va] + correction, 0, 100)
                final_resid = y_true.values[va] - final_pred
                base_resid_va = base_residual[va]
                m_final = regression_metrics(y_true.values[va], final_pred)
                m_base = regression_metrics(y_true.values[va], base_pred[va])
                fold_rows.append({
                    "Fold": fold_id,
                    "Base_RMSE": m_base["RMSE"],
                    "Final_RMSE": m_final["RMSE"],
                    "RMSE_Gain_vs_Base": m_base["RMSE"] - m_final["RMSE"],
                    "Base_R2": m_base["R2"],
                    "Final_R2": m_final["R2"],
                    "R2_Gain_vs_Base": m_final["R2"] - m_base["R2"],
                    "Corr_FinalResidual_vs_BaseResidual": corr_safe(final_resid, base_resid_va),
                    "Corr_PredCorrection_vs_BaseResidual": corr_safe(correction, base_resid_va),
                })
            except Exception as e:
                failed = True
                err_msg = repr(e)
                break

        if failed or np.isnan(oof_corr).any():
            rows.append({
                "Seed": seed,
                "Residual_Group": cfg.method_group,
                "Config_Name": cfg.config_name,
                "Feature_Mode": cfg.feature_mode,
                "Model_Kind": cfg.model_kind,
                "Params": str(cfg.params),
                "Shrinkage": cfg.shrinkage,
                "Status": "failed",
                "Error": err_msg,
                "OOF_Base_RMSE": base_oof_m["RMSE"],
                "OOF_Final_RMSE": np.nan,
                "OOF_RMSE_Gain_vs_Base": np.nan,
                "OOF_Base_R2": base_oof_m["R2"],
                "OOF_Final_R2": np.nan,
                "OOF_R2_Gain_vs_Base": np.nan,
                "OOF_Corr_FinalResidual_vs_BaseResidual": np.nan,
                "OOF_Corr_PredCorrection_vs_BaseResidual": np.nan,
            })
            continue

        final_oof_pred = np.clip(base_pred + oof_corr, 0, 100)
        final_oof_resid = y_true.values - final_oof_pred
        m_final_all = regression_metrics(y_true, final_oof_pred)
        fold_df = pd.DataFrame(fold_rows)

        row = {
            "Seed": seed,
            "Residual_Group": cfg.method_group,
            "Config_Name": cfg.config_name,
            "Feature_Mode": cfg.feature_mode,
            "Model_Kind": cfg.model_kind,
            "Params": str(cfg.params),
            "Shrinkage": cfg.shrinkage,
            "Status": "ok",
            "Error": "",
            "OOF_Base_RMSE": base_oof_m["RMSE"],
            "OOF_Final_RMSE": m_final_all["RMSE"],
            "OOF_RMSE_Gain_vs_Base": base_oof_m["RMSE"] - m_final_all["RMSE"],
            "OOF_Base_R2": base_oof_m["R2"],
            "OOF_Final_R2": m_final_all["R2"],
            "OOF_R2_Gain_vs_Base": m_final_all["R2"] - base_oof_m["R2"],
            "OOF_Corr_FinalResidual_vs_BaseResidual": corr_safe(final_oof_resid, base_residual),
            "OOF_Corr_PredCorrection_vs_BaseResidual": corr_safe(oof_corr, base_residual),
            "Fold_Final_RMSE_mean": fold_df["Final_RMSE"].mean(),
            "Fold_Final_RMSE_std": fold_df["Final_RMSE"].std(ddof=0),
            "Fold_RMSE_Gain_mean": fold_df["RMSE_Gain_vs_Base"].mean(),
            "Fold_RMSE_Gain_std": fold_df["RMSE_Gain_vs_Base"].std(ddof=0),
        }
        rows.append(row)
        oof_final_by_config[cfg.config_name] = final_oof_pred
        oof_corr_by_config[cfg.config_name] = oof_corr

    cv_df = pd.DataFrame(rows)
    ok = cv_df[cv_df["Status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("All residual correction candidates failed.")

    # Primary selection: OOF final RMSE. Secondary: lower residual correlation with base residual.
    ok = ok.sort_values(
        ["OOF_Final_RMSE", "OOF_Corr_FinalResidual_vs_BaseResidual", "Fold_Final_RMSE_std"],
        ascending=[True, True, True],
    )
    best_row = ok.iloc[0]
    best_cfg = next(c for c in configs if c.config_name == best_row["Config_Name"])
    return best_cfg, cv_df, {"final": oof_final_by_config, "correction": oof_corr_by_config}


# ============================================================
# 4. One split run
# ============================================================

def fit_predict_base_specs_for_test(specs, X_train_full, y_train_full, groups_train_full, X_test, seed, sample_weight_train_full):
    pred = {}
    for name in ["CatBoost", "XGBoost", "KRR_Poly", "GPR_Matern"]:
        if name not in specs:
            continue
        pred[name] = s31m.fit_predict_selected_spec(
            spec=specs[name],
            X_train=X_train_full,
            y_train=y_train_full,
            X_target=X_test,
            seed=seed + 123,
            sample_weight=sample_weight_train_full,
            train_groups=groups_train_full,
        )
    if "CB-XGB" in specs:
        pred["CB-XGB"] = s31m.fit_predict_selected_spec(
            spec=specs["CB-XGB"],
            X_train=X_train_full,
            y_train=y_train_full,
            X_target=X_test,
            seed=seed + 123,
            sample_weight=sample_weight_train_full,
            train_groups=groups_train_full,
        )
    return {k: np.clip(v, 0, 100) for k, v in pred.items()}


def run_one_seed(X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int):
    log("\n" + "=" * 100)
    log(f"31O grouped OOF residual learning | seed={seed}")
    log("=" * 100)

    train_idx, valid_idx, test_idx, split_info = s31f.grouped_split_indices(groups, seed)
    train_valid_idx = np.concatenate([train_idx, valid_idx])

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_valid = X.iloc[valid_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    X_train_full = X.iloc[train_valid_idx].reset_index(drop=True)

    y_train = y.iloc[train_idx].reset_index(drop=True).astype(float)
    y_valid = y.iloc[valid_idx].reset_index(drop=True).astype(float)
    y_test = y.iloc[test_idx].reset_index(drop=True).astype(float)
    y_train_full = y.iloc[train_valid_idx].reset_index(drop=True).astype(float)

    train_groups = groups.iloc[train_idx].reset_index(drop=True).astype(str)
    train_full_groups = groups.iloc[train_valid_idx].reset_index(drop=True).astype(str)
    test_groups = groups.iloc[test_idx].reset_index(drop=True).astype(str)

    sample_weight_train = s31f.make_literature_group_weights(train_groups)
    sample_weight_train_full = s31f.make_literature_group_weights(train_full_groups)

    # Feature ranking only on training groups.
    imp_df = s31f.rank_features_by_rf_importance_weighted(
        X_train, y_train, seed, sample_weight=sample_weight_train
    )

    # Select base specs by validation only.
    tree_specs, _, tree_cand = s31m.select_tree_and_cbxgb_specs(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        imp_df=imp_df,
        seed=seed,
        sample_weight_train=sample_weight_train,
    )
    krr_spec, _, krr_cand = s31m.select_non_tree_spec(
        family="KRR_Poly",
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        imp_df=imp_df,
        seed=seed,
        sample_weight_train=sample_weight_train,
        train_groups=train_groups,
    )
    gpr_spec, _, gpr_cand = s31m.select_non_tree_spec(
        family="GPR_Matern",
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        imp_df=imp_df,
        seed=seed,
        sample_weight_train=sample_weight_train,
        train_groups=train_groups,
    )

    specs = {**tree_specs, "KRR_Poly": krr_spec, "GPR_Matern": gpr_spec}
    candidate_df = pd.concat([tree_cand, krr_cand, gpr_cand], ignore_index=True)

    # Grouped OOF base predictions on train+valid only.
    oof_base, oof_audit = s31m.build_oof_base_predictions(
        specs=specs,
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        groups_train_full=train_full_groups,
        seed=seed,
    )
    # Construct CB-XGB OOF using the validation-selected CB-XGB weights.
    cbxgb_weights = specs.get("CB-XGB").weights if "CB-XGB" in specs else {"CatBoost": 0.5, "XGBoost": 0.5}
    if "CatBoost" in oof_base and "XGBoost" in oof_base:
        oof_base["CB-XGB"] = np.clip(sf19.apply_weights({"CatBoost": oof_base["CatBoost"], "XGBoost": oof_base["XGBoost"]}, cbxgb_weights), 0, 100)

    # Test predictions from base models fitted on train+valid.
    test_base = fit_predict_base_specs_for_test(
        specs=specs,
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        groups_train_full=train_full_groups,
        X_test=X_test,
        seed=seed,
        sample_weight_train_full=sample_weight_train_full,
    )

    # Build residual feature matrices for all candidate feature modes.
    X_meta_modes = {}
    for mode in ["base_preds_only", "base_preds_plus_krr_features"]:
        X_meta_modes[mode] = build_residual_feature_frame(
            X=X_train_full,
            base_pred_dict=oof_base,
            krr_selected_cols=krr_spec.selected_cols,
            feature_mode=mode,
        )
    X_test_meta_modes = {}
    for mode in ["base_preds_only", "base_preds_plus_krr_features"]:
        X_test_meta_modes[mode] = build_residual_feature_frame(
            X=X_test,
            base_pred_dict=test_base,
            krr_selected_cols=krr_spec.selected_cols,
            feature_mode=mode,
        )

    residual_configs = make_residual_configs()
    best_cfg, residual_cv_df, oof_pred_info = grouped_cv_select_residual_correction(
        X_meta=X_meta_modes,
        y_true=y_train_full,
        base_pred=oof_base["KRR_Poly"],
        groups=train_full_groups,
        configs=residual_configs,
        seed=seed,
    )

    # Train selected residual model on all train+valid OOF residuals, then apply to held-out test papers.
    base_residual_full = y_train_full.values - np.asarray(oof_base["KRR_Poly"], dtype=float)
    sw_resid_full = s31f.make_literature_group_weights(train_full_groups)
    correction_test = fit_predict_residual_model(
        cfg=best_cfg,
        X_train_meta=X_meta_modes[best_cfg.feature_mode],
        residual_train=pd.Series(base_residual_full),
        X_target_meta=X_test_meta_modes[best_cfg.feature_mode],
        seed=seed + 8888,
        sample_weight=sw_resid_full,
    )
    final_test = np.clip(np.asarray(test_base["KRR_Poly"], dtype=float) + correction_test, 0, 100)

    # Also produce grouped-CV selected representatives by residual group for test diagnostics.
    ok_cv = residual_cv_df[residual_cv_df["Status"] == "ok"].copy()
    group_best_rows = []
    group_test_preds = {}
    for group_name, sub in ok_cv.groupby("Residual_Group"):
        best_row = sub.sort_values(["OOF_Final_RMSE", "OOF_Corr_FinalResidual_vs_BaseResidual"], ascending=[True, True]).iloc[0]
        cfg = next(c for c in residual_configs if c.config_name == best_row["Config_Name"])
        corr = fit_predict_residual_model(
            cfg=cfg,
            X_train_meta=X_meta_modes[cfg.feature_mode],
            residual_train=pd.Series(base_residual_full),
            X_target_meta=X_test_meta_modes[cfg.feature_mode],
            seed=seed + 9000,
            sample_weight=sw_resid_full,
        )
        method = f"KRR_ResidualCorrected_{group_name}"
        group_test_preds[method] = np.clip(np.asarray(test_base["KRR_Poly"]) + corr, 0, 100)
        group_best_rows.append({
            "Seed": seed,
            "Residual_Group": group_name,
            "Selected_Config_Name": cfg.config_name,
            "Feature_Mode": cfg.feature_mode,
            "Model_Kind": cfg.model_kind,
            "Params": str(cfg.params),
            "Shrinkage": cfg.shrinkage,
            "OOF_Final_RMSE": best_row["OOF_Final_RMSE"],
            "OOF_RMSE_Gain_vs_Base": best_row["OOF_RMSE_Gain_vs_Base"],
            "OOF_Corr_FinalResidual_vs_BaseResidual": best_row["OOF_Corr_FinalResidual_vs_BaseResidual"],
        })

    # Test results and predictions.
    test_rows, pred_rows = [], []
    all_test_methods = {
        "KRR_Poly": test_base["KRR_Poly"],
        "CatBoost": test_base.get("CatBoost"),
        "XGBoost": test_base.get("XGBoost"),
        "CB-XGB": test_base.get("CB-XGB"),
        "GPR_Matern": test_base.get("GPR_Matern"),
        "KRR_ResidualCorrected_BestOOF": final_test,
        **group_test_preds,
    }
    all_test_methods = {k: v for k, v in all_test_methods.items() if v is not None}

    # Selected config audit for BestOOF.
    selected_rows = [{
        "Seed": seed,
        "Selected_For": "KRR_ResidualCorrected_BestOOF",
        "Residual_Group": best_cfg.method_group,
        "Selected_Config_Name": best_cfg.config_name,
        "Feature_Mode": best_cfg.feature_mode,
        "Model_Kind": best_cfg.model_kind,
        "Params": str(best_cfg.params),
        "Shrinkage": best_cfg.shrinkage,
    }] + group_best_rows

    for method, pred in all_test_methods.items():
        pred = np.clip(np.asarray(pred, dtype=float), 0, 100)
        m = regression_metrics(y_test, pred)
        test_rows.append({
            "Seed": seed,
            "Method": method,
            "R2": m["R2"],
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
            "MSE": m["MSE"],
            "Bias": m["Bias"],
            "n_test": len(y_test),
            "n_test_groups": test_groups.nunique(),
            "Selected_Config": best_cfg.config_name if method == "KRR_ResidualCorrected_BestOOF" else "",
        })
        for local_i, global_i in enumerate(test_idx):
            pred_rows.append({
                "Seed": seed,
                "Data_Index": int(global_i),
                "文献编号": test_groups.iloc[local_i],
                "Method": method,
                "Observed": float(y_test.iloc[local_i]),
                "Predicted": float(pred[local_i]),
                "Residual": float(y_test.iloc[local_i] - pred[local_i]),
            })

    # OOF correlation screen on train+valid only.
    oof_corr_rows = []
    base_resid = y_train_full.values - np.asarray(oof_base["KRR_Poly"], dtype=float)
    for method_name, pred in {
        "KRR_Poly": oof_base.get("KRR_Poly"),
        "CatBoost": oof_base.get("CatBoost"),
        "XGBoost": oof_base.get("XGBoost"),
        "CB-XGB": oof_base.get("CB-XGB"),
        "GPR_Matern": oof_base.get("GPR_Matern"),
    }.items():
        if pred is None:
            continue
        resid = y_train_full.values - np.asarray(pred, dtype=float)
        oof_corr_rows.append({
            "Seed": seed,
            "Method": method_name,
            "OOF_RMSE": regression_metrics(y_train_full, pred)["RMSE"],
            "OOF_R2": regression_metrics(y_train_full, pred)["R2"],
            "OOF_Corr_Residual_vs_KRRResidual": corr_safe(resid, base_resid),
        })
    for _, row in residual_cv_df[residual_cv_df["Status"] == "ok"].iterrows():
        oof_corr_rows.append({
            "Seed": seed,
            "Method": f"ResidualCandidate::{row['Residual_Group']}::{row['Config_Name']}",
            "OOF_RMSE": row["OOF_Final_RMSE"],
            "OOF_R2": row["OOF_Final_R2"],
            "OOF_Corr_Residual_vs_KRRResidual": row["OOF_Corr_FinalResidual_vs_BaseResidual"],
        })

    split_row = {
        "Seed": seed,
        **split_info,
        "Best_Residual_Group": best_cfg.method_group,
        "Best_Residual_Config": best_cfg.config_name,
        "Best_Residual_Feature_Mode": best_cfg.feature_mode,
    }

    return {
        "split": pd.DataFrame([split_row]),
        "candidate_cv": residual_cv_df,
        "test": pd.DataFrame(test_rows),
        "prediction": pd.DataFrame(pred_rows),
        "selected": pd.DataFrame(selected_rows),
        "oof_corr_screen": pd.DataFrame(oof_corr_rows),
        "base_candidate": candidate_df,
        "oof_audit": oof_audit,
    }


# ============================================================
# 5. Reporting and plotting
# ============================================================

def plot_metrics(summary_df: pd.DataFrame):
    rank = summary_df.sort_values("Pooled_RMSE", ascending=True).copy()
    top = rank.head(12).copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    metrics = [("Pooled_R2", "Pooled $R^2$"), ("Pooled_RMSE", "Pooled RMSE"), ("Pooled_MAE", "Pooled MAE")]
    best_method = rank.iloc[0]["Method"]
    for ax, (col, ylabel) in zip(axes, metrics):
        vals = top[col].values
        xs = np.arange(len(top))
        colors = [COLORS.get(m, "#999999") for m in top["Method"]]
        bars = ax.bar(xs, vals, color=colors, edgecolor="#333333", linewidth=0.5)
        for b, v, m in zip(bars, vals, top["Method"]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.3f}" if "R2" in col else f"{v:.2f}", ha="center", va="bottom", fontsize=7)
            if m == best_method:
                b.set_edgecolor(COLORS["red"])
                b.set_linewidth(1.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(top["Method"], rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
        ax.set_title(ylabel)
    fig.suptitle("31O grouped OOF residual learning and anti-correlation screen", fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02, "Residual models are selected only by grouped OOF residual CV; test residuals are never used for training/selection.", ha="center", fontsize=8)
    save_fig(fig, "131_31o_residual_learning_metrics.png")


def plot_cv_to_test_gain(selected_df: pd.DataFrame, test_df: pd.DataFrame):
    # Compare OOF gain of selected residual config with held-out test gain.
    base = test_df[test_df["Method"] == "KRR_Poly"][["Seed", "RMSE", "R2"]].rename(columns={"RMSE": "Test_Base_RMSE", "R2": "Test_Base_R2"})
    corrected = test_df[test_df["Method"] == "KRR_ResidualCorrected_BestOOF"][["Seed", "RMSE", "R2"]].rename(columns={"RMSE": "Test_Corrected_RMSE", "R2": "Test_Corrected_R2"})
    cfg = selected_df[selected_df["Selected_For"] == "KRR_ResidualCorrected_BestOOF"].copy()
    merged = cfg.merge(base, on="Seed", how="left").merge(corrected, on="Seed", how="left")
    merged["Test_RMSE_Gain_vs_Base"] = merged["Test_Base_RMSE"] - merged["Test_Corrected_RMSE"]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.scatter(merged.get("OOF_RMSE_Gain_vs_Base", np.zeros(len(merged))), merged["Test_RMSE_Gain_vs_Base"], s=60, color=COLORS["KRR_ResidualCorrected_BestOOF"], edgecolor="#333333", linewidth=0.5)
    for _, r in merged.iterrows():
        ax.text(r.get("OOF_RMSE_Gain_vs_Base", 0), r["Test_RMSE_Gain_vs_Base"], str(int(r["Seed"])), fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("Grouped OOF RMSE gain vs KRR")
    ax.set_ylabel("Held-out test RMSE gain vs KRR")
    ax.set_title("Validation-to-test transfer of residual correction")
    ax.grid(color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    save_fig(fig, "132_31o_cv_to_test_gain_transfer.png")


def plot_oof_corr_screen(oof_corr_df: pd.DataFrame):
    # Show lowest mean OOF residual correlations among method groups / candidates.
    df = oof_corr_df.copy()
    df["Group_Label"] = df["Method"].astype(str).str.replace(r"ResidualCandidate::", "", regex=True).str.split("::").str[0]
    agg = df.groupby("Group_Label", as_index=False).agg(
        Mean_OOF_Corr=("OOF_Corr_Residual_vs_KRRResidual", "mean"),
        Std_OOF_Corr=("OOF_Corr_Residual_vs_KRRResidual", "std"),
        Mean_OOF_RMSE=("OOF_RMSE", "mean"),
        n=("Seed", "count"),
    ).sort_values("Mean_OOF_Corr", ascending=True).head(18)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = np.arange(len(agg))
    bars = ax.bar(xs, agg["Mean_OOF_Corr"], color="#9ECAE1", edgecolor="#333333", linewidth=0.5)
    ax.axhline(0, color=COLORS["red"], linestyle="--", linewidth=1.0)
    for b, v in zip(bars, agg["Mean_OOF_Corr"]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(agg["Group_Label"], rotation=45, ha="right")
    ax.set_ylabel("Mean OOF residual correlation vs KRR residual")
    ax.set_title("Residual anti-correlation / low-correlation screen")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    save_fig(fig, "133_31o_oof_residual_correlation_screen.png")


def plot_predicted_vs_actual(pred_df: pd.DataFrame, summary_df: pd.DataFrame):
    top_methods = summary_df.sort_values("Pooled_RMSE")["Method"].head(4).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, method in zip(axes, top_methods):
        sub = pred_df[pred_df["Method"] == method]
        m = regression_metrics(sub["Observed"], sub["Predicted"])
        ax.scatter(sub["Observed"], sub["Predicted"], s=13, alpha=0.62, color=COLORS.get(method, "#888888"), edgecolors="none")
        ax.plot([0, 100], [0, 100], color=COLORS["red"], linestyle="--", linewidth=1.0)
        ax.set_title(method)
        ax.text(0.05, 0.92, f"$R^2$ = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nMAE = {m['MAE']:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=8)
        ax.grid(color=COLORS["grid"], linewidth=0.5, alpha=0.7)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
    for ax in axes[::2]:
        ax.set_ylabel("Predicted F removal (%)")
    for ax in axes[-2:]:
        ax.set_xlabel("Observed F removal (%)")
    fig.suptitle("Top pooled predicted-vs-observed plots after grouped OOF residual learning", fontsize=13, fontweight="bold")
    save_fig(fig, "134_31o_residual_learning_pooled_predicted_vs_actual.png")


def write_decision_report(summary_df, selected_df, oof_corr_df, test_corr, path: Path):
    rank = summary_df.sort_values("Pooled_RMSE", ascending=True).copy()
    best = rank.iloc[0]
    krr = rank[rank["Method"] == "KRR_Poly"].iloc[0] if "KRR_Poly" in set(rank["Method"]) else None
    best_corr = rank[rank["Method"] == "KRR_ResidualCorrected_BestOOF"].iloc[0] if "KRR_ResidualCorrected_BestOOF" in set(rank["Method"]) else None

    lines = []
    lines.append("=" * 100)
    lines.append("Step 31O grouped OOF residual-learning / anti-correlation screen report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Test whether residual-correction models trained only on grouped OOF residuals can create a genuinely complementary route to KRR_Poly.")
    lines.append("")
    lines.append("Strict workflow:")
    lines.append("- Base KRR_Poly/CatBoost/XGBoost/CB-XGB/GPR_Matern configs are selected only by validation RMSE.")
    lines.append("- Residual targets are train+valid grouped OOF residuals: observed - KRR_Poly_OOF_prediction.")
    lines.append("- Residual-correction configs are selected by grouped CV on OOF residuals only.")
    lines.append("- Test residuals are never used for training or selecting residual correction models.")
    lines.append("- 文献编号 is never used as an input feature.")
    lines.append("")
    lines.append("Performance ranking:")
    keep = ["Method", "Pooled_R2", "Pooled_RMSE", "Pooled_MAE", "Pooled_Bias", "R2_mean", "RMSE_mean", "MAE_mean", "n_pooled"]
    lines.append(rank[keep].to_string(index=False))
    lines.append("")
    lines.append("Best candidate:")
    lines.append(f"{best['Method']} | Pooled_R2={best['Pooled_R2']:.4f}, RMSE={best['Pooled_RMSE']:.4f}, MAE={best['Pooled_MAE']:.4f}")
    if krr is not None:
        lines.append("")
        lines.append("KRR_Poly reference:")
        lines.append(f"KRR_Poly | Pooled_R2={krr['Pooled_R2']:.4f}, RMSE={krr['Pooled_RMSE']:.4f}, MAE={krr['Pooled_MAE']:.4f}")
        lines.append(f"Best-vs-KRR RMSE gain = {krr['Pooled_RMSE'] - best['Pooled_RMSE']:.4f}")
        lines.append(f"Best-vs-KRR R2 gain = {best['Pooled_R2'] - krr['Pooled_R2']:.4f}")
    if best_corr is not None and krr is not None:
        lines.append("")
        lines.append("Grouped-OOF residual correction route:")
        lines.append(f"KRR_ResidualCorrected_BestOOF | Pooled_R2={best_corr['Pooled_R2']:.4f}, RMSE={best_corr['Pooled_RMSE']:.4f}, MAE={best_corr['Pooled_MAE']:.4f}")
        lines.append(f"Residual-correction RMSE gain vs KRR = {krr['Pooled_RMSE'] - best_corr['Pooled_RMSE']:.4f}")
    lines.append("")
    lines.append("Selected residual correction configs:")
    show_cols = ["Seed", "Selected_For", "Residual_Group", "Selected_Config_Name", "Feature_Mode", "Shrinkage"]
    show_cols = [c for c in show_cols if c in selected_df.columns]
    lines.append(selected_df[show_cols].to_string(index=False))
    lines.append("")
    lines.append("Lowest OOF residual correlations vs KRR residual:")
    oof_show = oof_corr_df.copy()
    oof_show["Group_Label"] = oof_show["Method"].astype(str).str.replace(r"ResidualCandidate::", "", regex=True).str.split("::").str[0]
    corr_rank = oof_show.groupby("Group_Label", as_index=False).agg(
        Mean_OOF_Corr=("OOF_Corr_Residual_vs_KRRResidual", "mean"),
        Mean_OOF_RMSE=("OOF_RMSE", "mean"),
        n=("Seed", "count"),
    ).sort_values("Mean_OOF_Corr", ascending=True).head(15)
    lines.append(corr_rank.to_string(index=False))
    lines.append("")
    lines.append("Decision guide:")
    lines.append("1. If residual correction improves over KRR_Poly on held-out grouped test, keep it for robustness validation.")
    lines.append("2. If residual correction improves OOF but not grouped test, treat it as overfitting-prone and do not use as final model.")
    lines.append("3. If no OOF residual model yields negative/low residual correlation and test improvement, keep KRR_Poly as main grouped candidate.")
    lines.append("4. Do not use test-residual-derived correction; this script does not train on test residuals.")
    lines.append("=" * 100)
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[TEXT] {path}")


# ============================================================
# 6. Main
# ============================================================

def main():
    start_time = time.time()
    log("\n" + "=" * 100)
    log("Step 31O: grouped OOF residual-learning anti-correlation screen")
    log("=" * 100)
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"SEEDS = {SEEDS}")
    log(f"BEST_FEATURE_SPACE = {BEST_FEATURE_SPACE}")
    log("=" * 100)

    y, X_raw, X_eng, X_mix_base = sf19.prepare_feature_sets()
    y = y.reset_index(drop=True).astype(float)
    X_mix_base = X_mix_base.reset_index(drop=True)

    log("Building 31K feature spaces and selecting confirmed Current31F route...")
    feature_spaces, audit_df, block_df = s31k.make_feature_spaces(X_mix_base, expected_n=len(y))
    if BEST_FEATURE_SPACE not in feature_spaces:
        raise RuntimeError(f"Cannot find required feature space: {BEST_FEATURE_SPACE}")
    X = feature_spaces[BEST_FEATURE_SPACE].reset_index(drop=True)

    groups = s31f.load_group_labels(expected_n=len(y))
    groups = groups.reset_index(drop=True).astype(str)
    s31f.set_balanced_split_context(X, y, groups)

    all_splits, all_cv, all_tests, all_predictions = [], [], [], []
    all_selected, all_oof_corr, all_base_cand, all_oof_audit = [], [], [], []
    for seed in SEEDS:
        result = run_one_seed(X=X, y=y, groups=groups, seed=seed)
        all_splits.append(result["split"])
        all_cv.append(result["candidate_cv"])
        all_tests.append(result["test"])
        all_predictions.append(result["prediction"])
        all_selected.append(result["selected"])
        all_oof_corr.append(result["oof_corr_screen"])
        all_base_cand.append(result["base_candidate"])
        all_oof_audit.append(result["oof_audit"])

    split_df = pd.concat(all_splits, ignore_index=True)
    cv_df = pd.concat(all_cv, ignore_index=True)
    test_df = pd.concat(all_tests, ignore_index=True)
    pred_df = pd.concat(all_predictions, ignore_index=True)
    selected_df = pd.concat(all_selected, ignore_index=True)
    oof_corr_df = pd.concat(all_oof_corr, ignore_index=True)
    base_cand_df = pd.concat(all_base_cand, ignore_index=True)
    oof_audit_df = pd.concat(all_oof_audit, ignore_index=True)

    summary_df = summarize_results(test_df, pred_df)
    test_corr = compute_residual_corr(pred_df)

    paths = {
        "split": TABLE_DIR / "31o_residual_learning_split_summary.csv",
        "cv": TABLE_DIR / "31o_residual_learning_candidate_cv.csv",
        "test": TABLE_DIR / "31o_residual_learning_test_results.csv",
        "pred": TABLE_DIR / "31o_residual_learning_predictions.csv",
        "summary": TABLE_DIR / "31o_residual_learning_summary.csv",
        "selected": TABLE_DIR / "31o_residual_learning_selected_configs.csv",
        "oof_corr": TABLE_DIR / "31o_oof_residual_correlation_screen.csv",
        "test_corr": TABLE_DIR / "31o_test_residual_correlation_matrix.csv",
        "base_candidates": TABLE_DIR / "31o_base_model_validation_candidates.csv",
        "oof_audit": TABLE_DIR / "31o_oof_base_prediction_audit.csv",
        "decision": TABLE_DIR / "31o_residual_learning_decision.txt",
    }

    split_df.to_csv(paths["split"], index=False, encoding="utf-8-sig")
    cv_df.to_csv(paths["cv"], index=False, encoding="utf-8-sig")
    test_df.to_csv(paths["test"], index=False, encoding="utf-8-sig")
    pred_df.to_csv(paths["pred"], index=False, encoding="utf-8-sig")
    summary_df.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    selected_df.to_csv(paths["selected"], index=False, encoding="utf-8-sig")
    oof_corr_df.to_csv(paths["oof_corr"], index=False, encoding="utf-8-sig")
    test_corr.to_csv(paths["test_corr"], encoding="utf-8-sig")
    base_cand_df.to_csv(paths["base_candidates"], index=False, encoding="utf-8-sig")
    oof_audit_df.to_csv(paths["oof_audit"], index=False, encoding="utf-8-sig")

    for p in paths.values():
        if p.suffix.lower() == ".csv":
            log(f"[TABLE] {p}")

    plot_metrics(summary_df)
    plot_cv_to_test_gain(selected_df, test_df)
    plot_oof_corr_screen(oof_corr_df)
    plot_predicted_vs_actual(pred_df, summary_df)
    write_decision_report(summary_df, selected_df, oof_corr_df, test_corr, paths["decision"])

    log("\n" + "=" * 100)
    log("Step 31O completed.")
    log("Please inspect:")
    log("1. outputs/tables/31o_residual_learning_summary.csv")
    log("2. outputs/tables/31o_residual_learning_decision.txt")
    log("3. outputs/tables/31o_oof_residual_correlation_screen.csv")
    log("4. outputs/figures/131_31o_residual_learning_metrics.png")
    log("5. outputs/figures/133_31o_oof_residual_correlation_screen.png")
    elapsed = time.time() - start_time
    log(f"[TIME] Total runtime: {elapsed:.2f} s ({elapsed/60:.2f} min)")
    log("=" * 100)


if __name__ == "__main__":
    main()
