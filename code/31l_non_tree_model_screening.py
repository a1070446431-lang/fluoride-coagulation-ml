# -*- coding: utf-8 -*-
"""
31l_non_tree_model_screening.py

Purpose
-------
Screen non-tree machine-learning models under the same strict balanced
literature-level grouped-validation framework used in Steps 31F--31K.

This script answers the next question after Step 31K:
    Do non-tree models, especially Gaussian Process (GP), Kernel Ridge (KRR),
    and PLS, provide a useful alternative or complementary error pattern for
    all-domain grouped generalization?

It is NOT a simplified workflow. It reuses the real project scripts:
    - Step 31K: confirmed grouped-generalization feature route
      Current31F_noFine_plus_31B_mechanism.
    - Step 31F: balanced literature-level grouped split and literature-group
      sample weights.
    - script 19: feature engineering, preprocessing, and feature-selection
      utilities.

Strict rules
------------
1. 文献编号 is used only for literature-level grouped splitting and weighting.
2. 文献编号 is never used as an input feature.
3. Test labels are never used for feature selection, model/config selection,
   or target scaling.
4. Feature ranking is performed only on grouped training data.
5. Candidate configs are selected only by validation RMSE.
6. Final test predictions are produced after retraining on train+valid groups.
7. Current 31K/31G CB-XGB baseline is retained as an explicit comparator when
   the Step-31K summary exists.
8. Total runtime is printed at the end.

Models screened
---------------
1. PLS Regression（偏最小二乘回归）
2. Kernel Ridge Regression, RBF kernel（RBF核岭回归）
3. Kernel Ridge Regression, polynomial kernel（多项式核岭回归）
4. Gaussian Process Regression, RBF kernel（RBF高斯过程回归）
5. Gaussian Process Regression, Matern kernel（Matern高斯过程回归）

Notes on sample weighting
-------------------------
- Kernel Ridge supports sample_weight in recent sklearn versions; the script
  attempts to pass literature-group weights and audits the mode actually used.
- Gaussian Process does not directly support sample_weight, so this script uses
  a heteroscedastic alpha array: alpha_i = base_noise / sample_weight_i. This
  gives higher-weight literature groups lower assumed noise.
- PLS does not support sample_weight. To avoid simply ignoring literature-group
  imbalance, this script uses deterministic literature-balanced resampling only
  within the training groups. The test set is never resampled.

Outputs
-------
Tables:
    outputs/tables/31l_non_tree_model_config_audit.csv
    outputs/tables/31l_non_tree_model_split_summary.csv
    outputs/tables/31l_non_tree_model_validation_candidates.csv
    outputs/tables/31l_non_tree_model_test_results.csv
    outputs/tables/31l_non_tree_model_predictions.csv
    outputs/tables/31l_non_tree_model_summary.csv
    outputs/tables/31l_non_tree_model_residual_correlation.csv
    outputs/tables/31l_non_tree_model_decision.txt

Figures:
    outputs/figures/115_31l_non_tree_model_metrics.png/pdf
    outputs/figures/116_31l_non_tree_model_residual_correlation.png/pdf
    outputs/figures/117_31l_non_tree_model_pooled_predicted_vs_actual.png/pdf
    outputs/figures/118_31l_non_tree_model_config_selection.png/pdf
"""

from __future__ import annotations

import ast
import math
import time
import warnings
import importlib.util
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import sparse

from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 0. Logging, imports, and project paths
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
    spec.loader.exec_module(module)
    return module


# Reuse Step 31K; it imports real 31G/31F/script-19.
s31k = load_module_from_file("31k_group_aware_feature_engineering_ablation.py", "s31k_real")
s31g = s31k.s31g
s31f = s31k.s31f
sf19 = s31k.sf19

PROJECT_ROOT = Path(s31k.PROJECT_ROOT)
FIG_DIR = Path(s31k.FIG_DIR)
TABLE_DIR = Path(s31k.TABLE_DIR)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Settings
# ============================================================

SEEDS = list(getattr(sf19, "SEEDS", [7, 21, 42, 66, 88]))
K_OPTIONS = [12, 18, 25, 35, 45, 60, "all"]
GP_ALLOWED_K = {12, 18, 25}  # deliberate computational guard for exact GP.
BEST_FEATURE_SPACE = "Current31F_noFine_plus_31B_mechanism"

CURRENT_BASELINE_NAME = "31K/31G baseline CB-XGB"

COLORS = {
    "PLS": "#9B72B0",
    "KRR_RBF": "#6E9BC3",
    "KRR_Poly": "#76B083",
    "GPR_RBF": "#D28C3D",
    "GPR_Matern": "#B06D84",
    "Baseline": "#999999",
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
# 2. Utilities
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
        if s.lower() == "all":
            return (999999, "")
        try:
            return (int(float(s)), "")
        except Exception:
            return (999998, s)
    return ",".join(sorted(set(vals), key=sort_key))


def select_columns_from_importance(imp_df: pd.DataFrame, X: pd.DataFrame, selected_k):
    cols = sf19.select_top_features(imp_df, X, selected_k)
    return [c for c in cols if c in X.columns and not s31k.is_bad_feature_name(c)]


def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def preprocess_fit_transform(X_train: pd.DataFrame, X_target: pd.DataFrame):
    """Use the real script-19 preprocessor, then densify and scale for non-tree models."""
    preprocessor, _, _ = sf19.build_preprocessor(X_train)
    Z_train = to_dense(preprocessor.fit_transform(X_train))
    Z_target = to_dense(preprocessor.transform(X_target))

    scaler = StandardScaler(with_mean=True, with_std=True)
    Z_train = scaler.fit_transform(Z_train)
    Z_target = scaler.transform(Z_target)

    # Guard against numerical issues from constant columns.
    Z_train = np.nan_to_num(Z_train, nan=0.0, posinf=0.0, neginf=0.0)
    Z_target = np.nan_to_num(Z_target, nan=0.0, posinf=0.0, neginf=0.0)
    return Z_train, Z_target


def clip_prediction(y_pred):
    # Keep F removal predictions within physically meaningful percent scale.
    return np.clip(np.asarray(y_pred, dtype=float).ravel(), 0.0, 100.0)


def group_balanced_resample_indices(groups: pd.Series, seed: int, target_per_group: int | None = None):
    """Deterministic literature-balanced resampling for estimators without sample_weight.

    This approximates equal literature-group contribution without using test labels.
    """
    rng = np.random.default_rng(seed)
    groups = pd.Series(groups).reset_index(drop=True)
    group_counts = groups.value_counts()
    if target_per_group is None:
        # Median keeps size controlled while preventing very small groups from vanishing.
        target_per_group = int(max(5, round(group_counts.median())))
    idx_all = []
    for g in sorted(group_counts.index.astype(str)):
        idx = np.where(groups.astype(str).values == str(g))[0]
        replace = len(idx) < target_per_group
        chosen = rng.choice(idx, size=target_per_group, replace=replace)
        idx_all.extend(chosen.tolist())
    rng.shuffle(idx_all)
    return np.asarray(idx_all, dtype=int), target_per_group


# ============================================================
# 3. Model config definitions
# ============================================================

@dataclass
class ModelConfig:
    model_family: str
    config_name: str
    params: dict
    allowed_k: set | None = None
    weighting_mode: str = ""


def make_model_configs() -> list[ModelConfig]:
    configs = []

    # PLS: chemometrics-style latent-variable benchmark.
    for n in [2, 3, 5, 8, 12, 20]:
        configs.append(ModelConfig(
            model_family="PLS",
            config_name=f"PLS_n{n}",
            params={"n_components": n},
            allowed_k=None,
            weighting_mode="group_balanced_resampling",
        ))

    # KRR-RBF: nonlinear kernel benchmark.
    for alpha, gamma in [
        (1.0, None),
        (5.0, None),
        (10.0, None),
        (1.0, 0.01),
        (5.0, 0.01),
        (10.0, 0.01),
        (5.0, 0.05),
        (10.0, 0.05),
    ]:
        gtxt = "auto" if gamma is None else str(gamma).replace(".", "p")
        configs.append(ModelConfig(
            model_family="KRR_RBF",
            config_name=f"KRR_RBF_a{str(alpha).replace('.', 'p')}_g{gtxt}",
            params={"kernel": "rbf", "alpha": alpha, "gamma": gamma},
            allowed_k=None,
            weighting_mode="sample_weight_if_supported",
        ))

    # KRR-polynomial: lower-dimensional nonlinear interactions.
    for alpha, degree in [(1.0, 2), (5.0, 2), (10.0, 2), (5.0, 3), (10.0, 3)]:
        configs.append(ModelConfig(
            model_family="KRR_Poly",
            config_name=f"KRR_Poly_d{degree}_a{str(alpha).replace('.', 'p')}",
            params={"kernel": "polynomial", "alpha": alpha, "degree": degree, "coef0": 1.0, "gamma": None},
            allowed_k={12, 18, 25, 35},
            weighting_mode="sample_weight_if_supported",
        ))

    # Exact GP is computationally expensive; use compact top-K features only.
    # Heteroscedastic alpha approximates literature-group weighting.
    for length_scale, noise in [(1.0, 4.0), (2.0, 4.0), (5.0, 9.0)]:
        configs.append(ModelConfig(
            model_family="GPR_RBF",
            config_name=f"GPR_RBF_ls{str(length_scale).replace('.', 'p')}_noise{str(noise).replace('.', 'p')}",
            params={"kernel_type": "rbf", "length_scale": length_scale, "noise": noise},
            allowed_k=GP_ALLOWED_K,
            weighting_mode="heteroscedastic_alpha_from_group_weight",
        ))
    for nu, length_scale, noise in [(1.5, 2.0, 4.0), (2.5, 2.0, 4.0), (1.5, 5.0, 9.0)]:
        configs.append(ModelConfig(
            model_family="GPR_Matern",
            config_name=f"GPR_Matern_nu{str(nu).replace('.', 'p')}_ls{str(length_scale).replace('.', 'p')}_noise{str(noise).replace('.', 'p')}",
            params={"kernel_type": "matern", "nu": nu, "length_scale": length_scale, "noise": noise},
            allowed_k=GP_ALLOWED_K,
            weighting_mode="heteroscedastic_alpha_from_group_weight",
        ))

    return configs


def fit_predict_non_tree(
    cfg: ModelConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_target: pd.DataFrame,
    seed: int,
    sample_weight=None,
    train_groups: pd.Series | None = None,
):
    """Fit one config and predict target rows."""
    sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    if cfg.model_family == "PLS":
        if train_groups is None:
            raise ValueError("PLS group-balanced resampling requires train_groups.")
        idx_resampled, target_per_group = group_balanced_resample_indices(train_groups, seed)
        X_fit = X_train.iloc[idx_resampled].reset_index(drop=True)
        y_fit = y_train.iloc[idx_resampled].reset_index(drop=True)
        Z_fit, Z_target = preprocess_fit_transform(X_fit, X_target)
        max_comp = int(min(cfg.params["n_components"], Z_fit.shape[1], max(1, len(y_fit) - 1)))
        model = PLSRegression(n_components=max_comp, scale=False)
        model.fit(Z_fit, y_fit)
        pred = model.predict(Z_target).ravel()
        return clip_prediction(pred), {"effective_n_components": max_comp, "target_per_group": target_per_group, "fit_weighting_mode": cfg.weighting_mode}

    Z_train, Z_target = preprocess_fit_transform(X_train, X_target)

    if cfg.model_family in {"KRR_RBF", "KRR_Poly"}:
        model = KernelRidge(**cfg.params)
        fit_mode = "sample_weight"
        try:
            model.fit(Z_train, y_train, sample_weight=sample_weight)
        except TypeError:
            fit_mode = "unweighted_fallback_no_sample_weight_support"
            model.fit(Z_train, y_train)
        pred = model.predict(Z_target)
        return clip_prediction(pred), {"fit_weighting_mode": fit_mode}

    if cfg.model_family in {"GPR_RBF", "GPR_Matern"}:
        length_scale = float(cfg.params["length_scale"])
        noise = float(cfg.params["noise"])
        if cfg.params["kernel_type"] == "rbf":
            kernel = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds="fixed")
        else:
            kernel = C(1.0, constant_value_bounds="fixed") * Matern(length_scale=length_scale, nu=float(cfg.params["nu"]), length_scale_bounds="fixed")

        if sample_weight is None:
            alpha = noise
        else:
            sw = np.asarray(sample_weight, dtype=float)
            sw = np.clip(sw, 0.05, 20.0)
            # Higher literature weight = lower assumed observation noise.
            alpha = noise / sw
            alpha = np.clip(alpha, 1e-4, 100.0)

        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            optimizer=None,
            normalize_y=True,
            random_state=seed,
        )
        model.fit(Z_train, y_train)
        pred = model.predict(Z_target)
        return clip_prediction(pred), {"fit_weighting_mode": cfg.weighting_mode}

    raise KeyError(f"Unknown model family: {cfg.model_family}")


# ============================================================
# 4. One split/feature-selection run
# ============================================================

def run_one_seed(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seed: int,
    configs: list[ModelConfig],
):
    log("\n" + "=" * 100)
    log(f"31L non-tree screening | seed={seed} | feature_space={BEST_FEATURE_SPACE} | n_features={X.shape[1]}")
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

    # Feature ranking only on grouped training rows.
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
        key = tuple(selected_cols)
        if key in valid_k_seen:
            continue
        valid_k_seen.add(key)

        log(f"  Seed={seed} | K={selected_k} | selected_features={len(selected_cols)}")
        X_train_k = X_train[selected_cols].reset_index(drop=True)
        X_valid_k = X_valid[selected_cols].reset_index(drop=True)

        for cfg in configs:
            if cfg.allowed_k is not None and selected_k not in cfg.allowed_k:
                continue
            log(f"    fitting validation config: {cfg.config_name}")
            try:
                pred_valid, fit_meta = fit_predict_non_tree(
                    cfg=cfg,
                    X_train=X_train_k,
                    y_train=y_train,
                    X_target=X_valid_k,
                    seed=seed,
                    sample_weight=sample_weight_train,
                    train_groups=train_groups,
                )
                m = regression_metrics(y_valid, pred_valid)
                status = "ok"
                err = ""
            except Exception as e:
                m = {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "Bias": np.nan}
                fit_meta = {}
                status = "failed"
                err = repr(e)
                log(f"    [WARN] {cfg.config_name} failed: {err}")

            candidate_rows.append({
                "Seed": seed,
                "Feature_Space": BEST_FEATURE_SPACE,
                "Model_Family": cfg.model_family,
                "Config_Name": cfg.config_name,
                "Selected_K": selected_k,
                "n_Selected_Features": len(selected_cols),
                "Selected_Features": ";".join(selected_cols),
                "Validation_R2": m["R2"],
                "Validation_RMSE": m["RMSE"],
                "Validation_MAE": m["MAE"],
                "Validation_MSE": m["MSE"],
                "Validation_Bias": m["Bias"],
                "Status": status,
                "Error": err,
                "Fit_Meta": str(fit_meta),
            })

    candidate_df = pd.DataFrame(candidate_rows)
    ok_candidates = candidate_df[candidate_df["Status"] == "ok"].copy()
    if ok_candidates.empty:
        raise RuntimeError(f"No valid non-tree candidates for seed {seed}")

    test_rows = []
    pred_rows = []

    for family in sorted(ok_candidates["Model_Family"].unique()):
        sub = ok_candidates[ok_candidates["Model_Family"] == family].copy()
        best = sub.sort_values(["Validation_RMSE", "Validation_MAE", "Validation_R2"], ascending=[True, True, False]).iloc[0]
        selected_cols = str(best["Selected_Features"]).split(";")
        selected_cols = [c for c in selected_cols if c in X.columns]
        cfg = next(c for c in configs if c.config_name == best["Config_Name"])

        X_train_full_k = X_train_full[selected_cols].reset_index(drop=True)
        X_test_k = X_test[selected_cols].reset_index(drop=True)
        pred_test, fit_meta = fit_predict_non_tree(
            cfg=cfg,
            X_train=X_train_full_k,
            y_train=y_train_full,
            X_target=X_test_k,
            seed=seed,
            sample_weight=sample_weight_train_full,
            train_groups=train_full_groups,
        )
        m = regression_metrics(y_test, pred_test)

        test_rows.append({
            "Seed": seed,
            "Feature_Space": BEST_FEATURE_SPACE,
            "Model_Family": family,
            "Config_Name": best["Config_Name"],
            "Selected_K": best["Selected_K"],
            "n_Selected_Features": len(selected_cols),
            "Selected_Features": ";".join(selected_cols),
            "Validation_R2": best["Validation_R2"],
            "Validation_RMSE": best["Validation_RMSE"],
            "Validation_MAE": best["Validation_MAE"],
            "Test_R2": m["R2"],
            "Test_RMSE": m["RMSE"],
            "Test_MAE": m["MAE"],
            "Test_MSE": m["MSE"],
            "Test_Bias": m["Bias"],
            "n_test": len(y_test),
            "n_test_groups": test_groups.nunique(),
            "Fit_Meta_Test": str(fit_meta),
        })

        for local_i, (obs, pred, gid, data_idx) in enumerate(zip(y_test, pred_test, test_groups, test_idx)):
            pred_rows.append({
                "Seed": seed,
                "Feature_Space": BEST_FEATURE_SPACE,
                "Model_Family": family,
                "Config_Name": best["Config_Name"],
                "Selected_K": best["Selected_K"],
                "Data_Index": int(data_idx),
                "Local_Test_Index": int(local_i),
                "文献编号": gid,
                "Observed": float(obs),
                "Predicted": float(pred),
                "Residual": float(obs - pred),
            })

    split_info = dict(split_info)
    split_info["Feature_Space"] = BEST_FEATURE_SPACE
    split_info["n_features_in_space"] = int(X.shape[1])

    return {
        "split": pd.DataFrame([split_info]),
        "candidate": candidate_df,
        "test": pd.DataFrame(test_rows),
        "prediction": pd.DataFrame(pred_rows),
    }


# ============================================================
# 5. Summary and baseline comparison
# ============================================================

def summarize_results(test_df: pd.DataFrame, pred_df: pd.DataFrame):
    summary = (
        test_df.groupby(["Model_Family"], as_index=False)
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
            Config_Name=("Config_Name", lambda x: ",".join(sorted(set(map(str, x))))),
            n_Selected_Features_mean=("n_Selected_Features", "mean"),
            n_seed=("Seed", "nunique"),
            n_test_total=("n_test", "sum"),
        )
    )

    pooled_rows = []
    for family, sub in pred_df.groupby("Model_Family"):
        m = regression_metrics(sub["Observed"], sub["Predicted"])
        pooled_rows.append({
            "Model_Family": family,
            "Pooled_R2": m["R2"],
            "Pooled_RMSE": m["RMSE"],
            "Pooled_MAE": m["MAE"],
            "Pooled_MSE": m["MSE"],
            "Pooled_Bias": m["Bias"],
            "n_pooled": len(sub),
            "n_literature_groups_pooled": sub["文献编号"].nunique(),
        })
    summary = summary.merge(pd.DataFrame(pooled_rows), on="Model_Family", how="left")
    summary = summary.sort_values(["Pooled_RMSE", "Pooled_MAE", "Pooled_R2"], ascending=[True, True, False]).reset_index(drop=True)
    return summary


def load_current_baseline_from_31k():
    path = TABLE_DIR / "31k_feature_ablation_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    mask = (
        (df["Feature_Space"].astype(str) == BEST_FEATURE_SPACE) &
        (df["Method"].astype(str) == "CatBoost_XGBoost_Weighted_2Model")
    )
    if mask.sum() == 0:
        return None
    row = df.loc[mask].iloc[0].to_dict()
    return {
        "Model_Family": CURRENT_BASELINE_NAME,
        "R2_mean": row.get("R2_mean", np.nan),
        "R2_std": row.get("R2_std", np.nan),
        "RMSE_mean": row.get("RMSE_mean", np.nan),
        "RMSE_std": row.get("RMSE_std", np.nan),
        "MAE_mean": row.get("MAE_mean", np.nan),
        "MAE_std": row.get("MAE_std", np.nan),
        "MSE_mean": row.get("MSE_mean", np.nan),
        "MSE_std": row.get("MSE_std", np.nan),
        "Bias_mean": row.get("Bias_mean", np.nan),
        "Bias_std": row.get("Bias_std", np.nan),
        "Selected_K": row.get("Selected_K", ""),
        "Config_Name": "From 31K summary",
        "n_Selected_Features_mean": row.get("n_Selected_Features_mean", np.nan),
        "n_seed": row.get("n_seed", np.nan),
        "n_test_total": row.get("n_test_total", np.nan),
        "Pooled_R2": row.get("Pooled_R2", np.nan),
        "Pooled_RMSE": row.get("Pooled_RMSE", np.nan),
        "Pooled_MAE": row.get("Pooled_MAE", np.nan),
        "Pooled_MSE": row.get("Pooled_MSE", np.nan),
        "Pooled_Bias": row.get("Pooled_Bias", np.nan),
        "n_pooled": row.get("n_pooled", np.nan),
        "n_literature_groups_pooled": row.get("n_literature_groups_pooled", np.nan),
    }


def add_baseline_to_summary(summary_df: pd.DataFrame):
    baseline = load_current_baseline_from_31k()
    if baseline is None:
        return summary_df.copy(), False
    out = pd.concat([pd.DataFrame([baseline]), summary_df.copy()], ignore_index=True)
    out = out.sort_values(["Pooled_RMSE", "Pooled_MAE", "Pooled_R2"], ascending=[True, True, False]).reset_index(drop=True)
    return out, True


def compute_residual_correlation(pred_df: pd.DataFrame):
    # Pivot by seed+data index so each row has residuals from all model families.
    piv = pred_df.pivot_table(
        index=["Seed", "Data_Index"],
        columns="Model_Family",
        values="Residual",
        aggfunc="mean",
    )
    corr = piv.corr(method="pearson")
    return corr.reset_index().rename(columns={"index": "Model_Family"})


# ============================================================
# 6. Figures and report
# ============================================================

def plot_metrics(summary_with_baseline: pd.DataFrame):
    data = summary_with_baseline.copy().sort_values(["Pooled_RMSE", "Pooled_MAE"], ascending=[True, True])
    x = np.arange(len(data))
    labels = data["Model_Family"].astype(str).tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    specs = [
        ("Pooled_R2", "Pooled $R^2$", "{:.3f}", True),
        ("Pooled_RMSE", "Pooled RMSE", "{:.2f}", False),
        ("Pooled_MAE", "Pooled MAE", "{:.2f}", False),
    ]
    for ax, (col, ylabel, fmt, high_better) in zip(axes, specs):
        vals = data[col].astype(float).values
        bar_colors = []
        for name in labels:
            if name == CURRENT_BASELINE_NAME:
                bar_colors.append(COLORS["gray"])
            else:
                bar_colors.append(COLORS.get(name, COLORS["gray"]))
        bars = ax.bar(x, vals, color=bar_colors, edgecolor="black", linewidth=0.45, alpha=0.88)
        best_idx = int(np.nanargmax(vals)) if high_better else int(np.nanargmin(vals))
        bars[best_idx].set_edgecolor(COLORS["red"])
        bars[best_idx].set_linewidth(1.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=34, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        if col == "Pooled_R2":
            ymin = min(-0.08, np.nanmin(vals) - 0.03)
            ymax = max(0.22, np.nanmax(vals) + 0.03)
            ax.set_ylim(ymin, ymax)
            ax.axhline(0, color="black", linewidth=0.7)
        else:
            ymin = max(0, np.nanmin(vals) - 1.5)
            ymax = np.nanmax(vals) + 1.5
            ax.set_ylim(ymin, ymax)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                va = "bottom" if v >= 0 else "top"
                ax.text(b.get_x() + b.get_width()/2, v, fmt.format(v), ha="center", va=va, fontsize=7)
    fig.suptitle("31L non-tree model screening under balanced literature-level grouped validation", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "Current31F_noFine_plus_31B_mechanism feature route; configs selected by validation RMSE.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0.02, 0.05, 1, 0.93])
    save_fig(fig, "115_31l_non_tree_model_metrics.png")


def plot_residual_correlation(corr_df: pd.DataFrame):
    if corr_df.empty or corr_df.shape[0] < 2:
        return
    corr = corr_df.set_index("Model_Family")
    labels = corr.index.astype(str).tolist()
    vals = corr.values.astype(float)

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    im = ax.imshow(vals, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(vals[i, j]) > 0.55 else "black")
    ax.set_title("Residual correlation among non-tree candidates")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson residual correlation")
    fig.tight_layout()
    save_fig(fig, "116_31l_non_tree_model_residual_correlation.png")


def plot_pooled_predictions(pred_df: pd.DataFrame, summary_df: pd.DataFrame):
    top = summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE"], ascending=[True, True]).head(4)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 10.5), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (_, row) in zip(axes, top.iterrows()):
        family = str(row["Model_Family"])
        sub = pred_df[pred_df["Model_Family"].astype(str) == family].copy()
        sub_plot = sub.sample(900, random_state=42) if len(sub) > 900 else sub
        ax.scatter(sub_plot["Observed"], sub_plot["Predicted"], s=16, color=COLORS.get(family, COLORS["gray"]), alpha=0.68, edgecolor="none")
        ax.plot([0, 100], [0, 100], color=COLORS["red"], linestyle=(0, (4, 3)), linewidth=1.0)
        m = regression_metrics(sub["Observed"], sub["Predicted"])
        ax.text(0.06, 0.92, f"$R^2$ = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nMAE = {m['MAE']:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, boxstyle="round,pad=0.3"))
        ax.set_title(family)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Observed F removal (%)")
        ax.set_ylabel("Predicted F removal (%)")
        ax.grid(color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    fig.suptitle("Top non-tree pooled predicted-vs-observed plots", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    save_fig(fig, "117_31l_non_tree_model_pooled_predicted_vs_actual.png")


def plot_config_selection(test_df: pd.DataFrame):
    if test_df.empty:
        return
    cfg_counts = (
        test_df.groupby(["Model_Family", "Config_Name"], as_index=False)
        .agg(n_selected=("Seed", "nunique"))
        .sort_values(["Model_Family", "n_selected"], ascending=[True, False])
    )
    # Top 3 per family for readability.
    plot_df = cfg_counts.groupby("Model_Family", group_keys=False).head(3).copy()
    plot_df["Label"] = plot_df["Model_Family"].astype(str) + " | " + plot_df["Config_Name"].astype(str)
    plot_df = plot_df.sort_values("n_selected", ascending=True)

    fig, ax = plt.subplots(figsize=(10.5, max(5.5, 0.35 * len(plot_df))))
    colors = [COLORS.get(f, COLORS["gray"]) for f in plot_df["Model_Family"]]
    bars = ax.barh(np.arange(len(plot_df)), plot_df["n_selected"], color=colors, edgecolor="black", linewidth=0.45, alpha=0.85)
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["Label"])
    ax.set_xlabel("Selection count across seeds")
    ax.set_title("Validation-selected non-tree hyperparameter configurations")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    for b, v in zip(bars, plot_df["n_selected"]):
        ax.text(v + 0.05, b.get_y() + b.get_height()/2, f"{int(v)}", ha="left", va="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "118_31l_non_tree_model_config_selection.png")


def write_decision_report(summary_df: pd.DataFrame, summary_with_baseline: pd.DataFrame, baseline_loaded: bool, config_audit: pd.DataFrame):
    path = TABLE_DIR / "31l_non_tree_model_decision.txt"
    rank = summary_with_baseline.sort_values(["Pooled_RMSE", "Pooled_MAE", "Pooled_R2"], ascending=[True, True, False]).reset_index(drop=True)
    best = rank.iloc[0]

    non_tree_best = summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE", "Pooled_R2"], ascending=[True, True, False]).iloc[0]
    baseline = None
    if baseline_loaded:
        base_rows = rank[rank["Model_Family"].astype(str) == CURRENT_BASELINE_NAME]
        if not base_rows.empty:
            baseline = base_rows.iloc[0]

    lines = []
    lines.append("=" * 100)
    lines.append("Step 31L non-tree model screening report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Test whether GP/KRR/PLS non-tree models provide a stronger or complementary route after 31K confirmed the current grouped feature representation.")
    lines.append("")
    lines.append("Strict workflow reused:")
    lines.append("- Current31F_noFine_plus_31B_mechanism feature route confirmed by 31K")
    lines.append("- Step-31F balanced literature-level grouped split")
    lines.append("- Literature-group weighting or justified equivalent where supported")
    lines.append("- Training-only RF feature ranking and validation-only config selection")
    lines.append("- Test labels never used in model/config selection")
    lines.append("")
    lines.append("Config audit:")
    lines.append(config_audit.to_string(index=False))
    lines.append("")
    lines.append("Performance ranking including 31K/31G baseline when available:")
    keep_cols = ["Model_Family", "Selected_K", "Config_Name", "n_Selected_Features_mean", "Pooled_R2", "Pooled_RMSE", "Pooled_MAE", "Pooled_Bias", "n_pooled"]
    lines.append(rank[keep_cols].to_string(index=False))
    lines.append("")
    lines.append("Best overall candidate:")
    lines.append(f"{best['Model_Family']} | Pooled_R2={best['Pooled_R2']:.4f}, RMSE={best['Pooled_RMSE']:.4f}, MAE={best['Pooled_MAE']:.4f}")
    lines.append("")
    lines.append("Best non-tree candidate:")
    lines.append(f"{non_tree_best['Model_Family']} | Pooled_R2={non_tree_best['Pooled_R2']:.4f}, RMSE={non_tree_best['Pooled_RMSE']:.4f}, MAE={non_tree_best['Pooled_MAE']:.4f}")
    if baseline is not None:
        lines.append("")
        lines.append("Comparison with current CB-XGB baseline:")
        lines.append(f"Baseline | R2={baseline['Pooled_R2']:.4f}, RMSE={baseline['Pooled_RMSE']:.4f}, MAE={baseline['Pooled_MAE']:.4f}")
        lines.append(f"Best non-tree RMSE gain vs baseline = {baseline['Pooled_RMSE'] - non_tree_best['Pooled_RMSE']:.4f}")
        lines.append(f"Best non-tree R2 gain vs baseline = {non_tree_best['Pooled_R2'] - baseline['Pooled_R2']:.4f}")
    lines.append("")
    lines.append("Decision guide:")
    lines.append("1. If a non-tree model clearly beats the CB-XGB baseline, include it in the next fusion/stacking candidate pool.")
    lines.append("2. If a non-tree model does not beat baseline but has low residual correlation, keep it for OOF stacking as a possible complementary model.")
    lines.append("3. If all non-tree models are worse and residuals are highly correlated, do not force GP/KRR/PLS into the final model; move to grouped OOF stacking with the proven CB-XGB route.")
    lines.append("4. GP models here are exact compact-feature GPR screens, not full GP-BT Bayesian optimization. They are appropriate as a diagnostic non-tree benchmark.")
    lines.append("=" * 100)
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[TEXT] {path}")


# ============================================================
# 7. Main
# ============================================================

def main():
    start = time.time()
    log("\n" + "=" * 100)
    log("Step 31L: non-tree model screening under grouped validation")
    log("=" * 100)
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"SEEDS = {SEEDS}")
    log(f"K_OPTIONS = {K_OPTIONS}")
    log(f"BEST_FEATURE_SPACE = {BEST_FEATURE_SPACE}")
    log("=" * 100)

    y, X_raw, X_eng, X_mix_base = sf19.prepare_feature_sets()
    y = y.reset_index(drop=True).astype(float)
    X_mix_base = X_mix_base.reset_index(drop=True)

    log("Building 31K feature spaces and selecting confirmed current 31F route...")
    feature_spaces, audit_df, block_df = s31k.make_feature_spaces(X_mix_base, expected_n=len(y))
    if BEST_FEATURE_SPACE not in feature_spaces:
        raise RuntimeError(f"Cannot find required feature space: {BEST_FEATURE_SPACE}")
    X = feature_spaces[BEST_FEATURE_SPACE].reset_index(drop=True)

    groups = s31f.load_group_labels(expected_n=len(y))
    s31f.set_balanced_split_context(X, y, groups)

    configs = make_model_configs()
    config_audit = pd.DataFrame([{
        "Model_Family": c.model_family,
        "Config_Name": c.config_name,
        "Params": str(c.params),
        "Allowed_K": "all" if c.allowed_k is None else ",".join(map(str, sorted(c.allowed_k))),
        "Weighting_Mode": c.weighting_mode,
    } for c in configs])
    config_path = TABLE_DIR / "31l_non_tree_model_config_audit.csv"
    config_audit.to_csv(config_path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {config_path}")

    all_splits, all_candidates, all_tests, all_predictions = [], [], [], []
    for seed in SEEDS:
        result = run_one_seed(X=X, y=y, groups=groups, seed=seed, configs=configs)
        all_splits.append(result["split"])
        all_candidates.append(result["candidate"])
        all_tests.append(result["test"])
        all_predictions.append(result["prediction"])

    split_df = pd.concat(all_splits, ignore_index=True)
    candidate_df = pd.concat(all_candidates, ignore_index=True)
    test_df = pd.concat(all_tests, ignore_index=True)
    pred_df = pd.concat(all_predictions, ignore_index=True)
    summary_df = summarize_results(test_df, pred_df)
    summary_with_baseline, baseline_loaded = add_baseline_to_summary(summary_df)
    corr_df = compute_residual_correlation(pred_df)

    out_tables = {
        "31l_non_tree_model_split_summary.csv": split_df,
        "31l_non_tree_model_validation_candidates.csv": candidate_df,
        "31l_non_tree_model_test_results.csv": test_df,
        "31l_non_tree_model_predictions.csv": pred_df,
        "31l_non_tree_model_summary.csv": summary_with_baseline,
        "31l_non_tree_model_non_tree_only_summary.csv": summary_df,
        "31l_non_tree_model_residual_correlation.csv": corr_df,
    }
    for filename, df in out_tables.items():
        path = TABLE_DIR / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        log(f"[TABLE] {path}")

    log("\n=== Step 31L summary including baseline when available ===")
    print(summary_with_baseline.to_string(index=False))

    write_decision_report(summary_df, summary_with_baseline, baseline_loaded, config_audit)

    log("Generating figures...")
    plot_metrics(summary_with_baseline)
    plot_residual_correlation(corr_df)
    plot_pooled_predictions(pred_df, summary_df)
    plot_config_selection(test_df)

    elapsed = time.time() - start
    log("\n" + "=" * 100)
    log("[DONE] Step 31L non-tree model screening completed.")
    log(f"[TIME] Total runtime: {elapsed:.2f} s ({elapsed/60:.2f} min)")
    log("=" * 100)
    log("Please send these outputs:")
    log("1. outputs/tables/31l_non_tree_model_summary.csv")
    log("2. outputs/tables/31l_non_tree_model_decision.txt")
    log("3. outputs/tables/31l_non_tree_model_residual_correlation.csv")
    log("4. outputs/figures/115_31l_non_tree_model_metrics.png")
    log("5. outputs/figures/116_31l_non_tree_model_residual_correlation.png")
    log("6. outputs/figures/117_31l_non_tree_model_pooled_predicted_vs_actual.png")
    log("=" * 100)


if __name__ == "__main__":
    main()
