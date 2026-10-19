# -*- coding: utf-8 -*-
"""
27_oof_stacking_benchmark.py

Step 27: Strict OOF stacking benchmark under the real script-19 workflow.

Purpose
-------
This script tests whether OOF stacking can outperform the current final model:

    Current final model:
        Selected CatBoost-HGB validation-weighted 2Model

Stacking benchmarks:
    1. CatBoost-HGB OOF Linear Stacking
    2. CatBoost-HGB OOF Ridge Stacking
    3. RF-CatBoost-HGB-ExtraTrees OOF Linear Stacking
    4. RF-CatBoost-HGB-ExtraTrees OOF Ridge Stacking

Strict rule
-----------
This script imports and reuses the real 19_selected_feature_fusion.py workflow:
    - same data source
    - same feature engineering
    - same mixed feature table
    - same random seeds
    - same train/validation/test split
    - same feature ranking and K search
    - same base-model parameters
    - same final test-set isolation

The only change is the fusion strategy:
    validation-weighted averaging vs OOF stacking.

Outputs
-------
Tables:
    outputs/tables/27_oof_stacking_candidate_results.csv
    outputs/tables/27_oof_stacking_test_results.csv
    outputs/tables/27_oof_stacking_predictions.csv
    outputs/tables/27_oof_stacking_meta_coefficients.csv
    outputs/tables/27_oof_stacking_summary.csv
    outputs/tables/27_oof_stacking_decision.txt

Figures:
    outputs/figures/77_oof_stacking_metrics_comparison.png/pdf
    outputs/figures/78_oof_stacking_pooled_predicted_vs_actual.png/pdf
    outputs/figures/79_oof_stacking_meta_coefficients.png/pdf
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

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# ============================================================
# 0. Import real script 19
# ============================================================

def log(msg: str):
    print(msg, flush=True)


def find_file_upwards(start_path: Path, filename: str) -> Path:
    start_path = start_path.resolve()
    search_roots = [start_path.parent] + list(start_path.parents)

    for root in search_roots:
        candidates = [
            root / filename,
            root / "src" / filename,
        ]
        for p in candidates:
            if p.exists():
                return p

    raise FileNotFoundError(f"Cannot find required file: {filename}")


def load_script19():
    module_path = find_file_upwards(Path(__file__), "19_selected_feature_fusion.py")

    log("=" * 100)
    log("Loading real 19_selected_feature_fusion.py for strict OOF stacking benchmark:")
    log(str(module_path))
    log("=" * 100)

    spec = importlib.util.spec_from_file_location(
        "selected_feature_fusion_19",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


sf19 = load_script19()

PROJECT_ROOT = Path(sf19.PROJECT_ROOT)
FIG_DIR = Path(sf19.FIG_DIR)
TABLE_DIR = Path(sf19.TABLE_DIR)
MODEL_DIR = Path(sf19.MODEL_DIR)

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Global settings
# ============================================================

OOF_N_SPLITS = 5

TWO_MODEL_NAMES = [
    "CatBoost",
    "HistGradientBoosting",
]

FOUR_MODEL_NAMES = [
    "RandomForest",
    "CatBoost",
    "HistGradientBoosting",
    "ExtraTrees",
]

METHOD_ORDER = [
    "Weighted_2Model",
    "OOF_Linear_2Model",
    "OOF_Ridge_2Model",
    "Weighted_4Model",
    "OOF_Linear_4Model",
    "OOF_Ridge_4Model",
]

METHOD_LABELS = {
    "Weighted_2Model": "Weighted 2Model",
    "OOF_Linear_2Model": "OOF Linear 2Model",
    "OOF_Ridge_2Model": "OOF Ridge 2Model",
    "Weighted_4Model": "Weighted 4Model",
    "OOF_Linear_4Model": "OOF Linear 4Model",
    "OOF_Ridge_4Model": "OOF Ridge 4Model",
}

COLORS = {
    "Weighted_2Model": "#63B7B2",
    "OOF_Linear_2Model": "#9B72B0",
    "OOF_Ridge_2Model": "#7E62A8",
    "Weighted_4Model": "#E28E68",
    "OOF_Linear_4Model": "#4C78A8",
    "OOF_Ridge_4Model": "#2F5F9F",
    "CatBoost": "#9B72B0",
    "HistGradientBoosting": "#63A97A",
    "RandomForest": "#4C78A8",
    "ExtraTrees": "#8C564B",
    "red": "#FF0000",
    "gray": "#666666",
    "dark": "#333333",
    "grid": "#E6E6E6",
    "magenta": "#EB4AA9",
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
# 2. Utility functions
# ============================================================

def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)

    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
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


def weight_dict_to_text(weight_dict: dict) -> str:
    return str({k: float(v) for k, v in weight_dict.items()})


def parse_weight_text(x):
    if pd.isna(x):
        return {}
    s = str(x).strip()
    if not s:
        return {}
    return ast.literal_eval(s)


def model_short_name(name: str) -> str:
    mapping = {
        "RandomForest": "RF",
        "CatBoost": "CatBoost",
        "HistGradientBoosting": "HGB",
        "ExtraTrees": "ExtraTrees",
    }
    return mapping.get(str(name), str(name))


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
# 3. OOF stacking core functions
# ============================================================

def check_required_models():
    missing = []

    for m in FOUR_MODEL_NAMES:
        if m not in sf19.MODEL_FACTORIES:
            missing.append(m)

    if missing:
        raise RuntimeError(
            "Missing base models in script 19 MODEL_FACTORIES: "
            + ", ".join(missing)
        )


def fit_predict_single_model(model_name, X_train, y_train, X_target, seed):
    """
    Reuse script-19 model factory and preprocessing.
    """
    _, pred = sf19.fit_predict_model_trainfull(
        model_name,
        X_train,
        y_train,
        X_target,
        seed,
    )
    return np.asarray(pred, dtype=float)


def make_oof_and_target_predictions(
    model_names: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_target: pd.DataFrame,
    seed: int,
    n_splits: int = OOF_N_SPLITS,
):
    """
    Generate OOF predictions on X_train and base-model predictions on X_target.

    OOF predictions are used to train the stacking meta-model.
    X_target can be validation set during K search or test set during final evaluation.
    """
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_target = X_target.reset_index(drop=True)

    n = len(X_train)
    n_splits_eff = min(n_splits, n)

    if n_splits_eff < 2:
        raise ValueError("Too few training samples for OOF stacking.")

    kf = KFold(
        n_splits=n_splits_eff,
        shuffle=True,
        random_state=seed,
    )

    oof = np.zeros((n, len(model_names)), dtype=float)

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
        log(f"            OOF fold {fold_id}/{n_splits_eff}")

        X_tr = X_train.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_train.iloc[tr_idx].reset_index(drop=True)
        X_va = X_train.iloc[va_idx].reset_index(drop=True)

        for j, model_name in enumerate(model_names):
            pred_va = fit_predict_single_model(
                model_name=model_name,
                X_train=X_tr,
                y_train=y_tr,
                X_target=X_va,
                seed=seed + fold_id * 100,
            )
            oof[va_idx, j] = pred_va

    target_preds = []

    for model_name in model_names:
        pred_target = fit_predict_single_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_target=X_target,
            seed=seed,
        )
        target_preds.append(pred_target)

    target_mat = np.column_stack(target_preds)

    return oof, target_mat


def fit_meta_and_predict(meta_kind: str, oof_mat, y_train, target_mat):
    y_train = np.asarray(y_train, dtype=float)

    if meta_kind == "Linear":
        meta = LinearRegression()
    elif meta_kind == "Ridge":
        meta = RidgeCV(
            alphas=np.logspace(-4, 4, 25),
            fit_intercept=True,
        )
    else:
        raise ValueError(f"Unknown meta model kind: {meta_kind}")

    meta.fit(oof_mat, y_train)
    pred = meta.predict(target_mat)

    return meta, np.asarray(pred, dtype=float)


def meta_coeff_records(meta, method: str, seed: int, selected_k, model_names: list[str]):
    rows = []

    coefs = getattr(meta, "coef_", None)
    intercept = getattr(meta, "intercept_", np.nan)

    if coefs is None:
        return rows

    for model_name, coef in zip(model_names, np.asarray(coefs).ravel()):
        rows.append({
            "Seed": seed,
            "Method": method,
            "Selected_K": selected_k,
            "Meta_Model": method,
            "Base_Model": model_name,
            "Coefficient": float(coef),
            "Intercept": float(intercept) if np.ndim(intercept) == 0 else np.nan,
            "Alpha": float(getattr(meta, "alpha_", np.nan)) if hasattr(meta, "alpha_") else np.nan,
        })

    return rows


# ============================================================
# 4. Per-seed benchmark
# ============================================================

def run_one_seed(X_mix: pd.DataFrame, y: pd.Series, seed: int):
    log("\n" + "=" * 100)
    log(f"Step 27 OOF stacking benchmark | seed = {seed}")
    log("=" * 100)

    idx = np.arange(len(y))

    train_valid_idx, test_idx = train_test_split(
        idx,
        test_size=sf19.TEST_SIZE,
        random_state=seed,
    )

    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=sf19.VALID_SIZE_IN_TRAIN,
        random_state=seed,
    )

    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_valid = y.iloc[valid_idx].reset_index(drop=True)
    y_train_full = y.iloc[train_valid_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    X_mix_train = X_mix.iloc[train_idx].reset_index(drop=True)
    X_mix_valid = X_mix.iloc[valid_idx].reset_index(drop=True)
    X_mix_train_full = X_mix.iloc[train_valid_idx].reset_index(drop=True)
    X_mix_test = X_mix.iloc[test_idx].reset_index(drop=True)

    log("Ranking features with the same RF-importance logic as script 19...")
    imp_df = sf19.rank_features_by_rf_importance(
        X_mix_train,
        y_train,
        seed,
    )

    candidate_rows = []
    test_rows = []
    prediction_rows = []
    meta_coef_rows = []

    # --------------------------------------------------------
    # Validation-stage K search
    # --------------------------------------------------------
    for k in sf19.K_OPTIONS:
        selected_cols = sf19.select_top_features(imp_df, X_mix, k)
        selected_cols = [c for c in selected_cols if c in X_mix.columns]

        if len(selected_cols) < 3:
            continue

        log("-" * 100)
        log(f"Seed {seed} | K = {k} | selected features = {len(selected_cols)}")
        log("-" * 100)

        X_train_k = X_mix_train[selected_cols].reset_index(drop=True)
        X_valid_k = X_mix_valid[selected_cols].reset_index(drop=True)

        # Generate OOF/validation predictions for 4 base models once.
        oof_4, valid_mat_4 = make_oof_and_target_predictions(
            model_names=FOUR_MODEL_NAMES,
            X_train=X_train_k,
            y_train=y_train,
            X_target=X_valid_k,
            seed=seed,
            n_splits=OOF_N_SPLITS,
        )

        name_to_col = {name: i for i, name in enumerate(FOUR_MODEL_NAMES)}
        idx_2 = [name_to_col[m] for m in TWO_MODEL_NAMES]

        oof_2 = oof_4[:, idx_2]
        valid_mat_2 = valid_mat_4[:, idx_2]

        valid_pred_dict_2 = {
            m: valid_mat_2[:, j]
            for j, m in enumerate(TWO_MODEL_NAMES)
        }

        valid_pred_dict_4 = {
            m: valid_mat_4[:, j]
            for j, m in enumerate(FOUR_MODEL_NAMES)
        }

        # Weighted 2Model, same as exact-ablation logic.
        weights_2, val_rmse_2 = sf19.search_best_weights(
            valid_pred_dict_2,
            y_valid,
            seed,
        )
        pred_valid_weighted_2 = sf19.apply_weights(valid_pred_dict_2, weights_2)
        metrics_w2 = regression_metrics(y_valid, pred_valid_weighted_2)

        candidate_rows.append({
            "Seed": seed,
            "Method": "Weighted_2Model",
            "Selected_K": k,
            "Selected_Features": ";".join(selected_cols),
            "Validation_RMSE": metrics_w2["RMSE"],
            "Validation_R2": metrics_w2["R2"],
            "Validation_MAE": metrics_w2["MAE"],
            "Weights": weight_dict_to_text(weights_2),
            "Meta_Alpha": np.nan,
        })

        # Weighted 4Model, for benchmark.
        weights_4, val_rmse_4 = sf19.search_best_weights(
            valid_pred_dict_4,
            y_valid,
            seed,
        )
        pred_valid_weighted_4 = sf19.apply_weights(valid_pred_dict_4, weights_4)
        metrics_w4 = regression_metrics(y_valid, pred_valid_weighted_4)

        candidate_rows.append({
            "Seed": seed,
            "Method": "Weighted_4Model",
            "Selected_K": k,
            "Selected_Features": ";".join(selected_cols),
            "Validation_RMSE": metrics_w4["RMSE"],
            "Validation_R2": metrics_w4["R2"],
            "Validation_MAE": metrics_w4["MAE"],
            "Weights": weight_dict_to_text(weights_4),
            "Meta_Alpha": np.nan,
        })

        # OOF stacking 2Model.
        for meta_kind in ["Linear", "Ridge"]:
            method_name = f"OOF_{meta_kind}_2Model"

            meta, pred_valid = fit_meta_and_predict(
                meta_kind=meta_kind,
                oof_mat=oof_2,
                y_train=y_train,
                target_mat=valid_mat_2,
            )

            metrics = regression_metrics(y_valid, pred_valid)

            candidate_rows.append({
                "Seed": seed,
                "Method": method_name,
                "Selected_K": k,
                "Selected_Features": ";".join(selected_cols),
                "Validation_RMSE": metrics["RMSE"],
                "Validation_R2": metrics["R2"],
                "Validation_MAE": metrics["MAE"],
                "Weights": "",
                "Meta_Alpha": float(getattr(meta, "alpha_", np.nan)) if hasattr(meta, "alpha_") else np.nan,
            })

        # OOF stacking 4Model.
        for meta_kind in ["Linear", "Ridge"]:
            method_name = f"OOF_{meta_kind}_4Model"

            meta, pred_valid = fit_meta_and_predict(
                meta_kind=meta_kind,
                oof_mat=oof_4,
                y_train=y_train,
                target_mat=valid_mat_4,
            )

            metrics = regression_metrics(y_valid, pred_valid)

            candidate_rows.append({
                "Seed": seed,
                "Method": method_name,
                "Selected_K": k,
                "Selected_Features": ";".join(selected_cols),
                "Validation_RMSE": metrics["RMSE"],
                "Validation_R2": metrics["R2"],
                "Validation_MAE": metrics["MAE"],
                "Weights": "",
                "Meta_Alpha": float(getattr(meta, "alpha_", np.nan)) if hasattr(meta, "alpha_") else np.nan,
            })

    candidate_df_seed = pd.DataFrame(candidate_rows)

    # --------------------------------------------------------
    # Final test evaluation
    # Each method selects its own best K by validation RMSE.
    # Then stacking meta-model is retrained using OOF predictions on train+valid.
    # --------------------------------------------------------
    for method in METHOD_ORDER:
        sub = candidate_df_seed[candidate_df_seed["Method"] == method].copy()

        if sub.empty:
            continue

        best_row = sub.sort_values("Validation_RMSE", ascending=True).iloc[0]

        selected_k = best_row["Selected_K"]
        selected_cols = str(best_row["Selected_Features"]).split(";")
        selected_cols = [c for c in selected_cols if c in X_mix.columns]

        if len(selected_cols) < 3:
            continue

        log("=" * 100)
        log(f"Seed {seed} | Final test evaluation | {method} | K={selected_k} | n_features={len(selected_cols)}")
        log("=" * 100)

        X_train_full_k = X_mix_train_full[selected_cols].reset_index(drop=True)
        X_test_k = X_mix_test[selected_cols].reset_index(drop=True)

        if method == "Weighted_2Model":
            weights = parse_weight_text(best_row["Weights"])

            test_pred_dict = {}
            for m in TWO_MODEL_NAMES:
                pred_test_m = fit_predict_single_model(
                    model_name=m,
                    X_train=X_train_full_k,
                    y_train=y_train_full,
                    X_target=X_test_k,
                    seed=seed,
                )
                test_pred_dict[m] = pred_test_m

            pred_test = sf19.apply_weights(test_pred_dict, weights)
            meta_alpha = np.nan

        elif method == "Weighted_4Model":
            weights = parse_weight_text(best_row["Weights"])

            test_pred_dict = {}
            for m in FOUR_MODEL_NAMES:
                pred_test_m = fit_predict_single_model(
                    model_name=m,
                    X_train=X_train_full_k,
                    y_train=y_train_full,
                    X_target=X_test_k,
                    seed=seed,
                )
                test_pred_dict[m] = pred_test_m

            pred_test = sf19.apply_weights(test_pred_dict, weights)
            meta_alpha = np.nan

        elif method in ["OOF_Linear_2Model", "OOF_Ridge_2Model"]:
            meta_kind = "Linear" if "Linear" in method else "Ridge"

            oof_2, test_mat_2 = make_oof_and_target_predictions(
                model_names=TWO_MODEL_NAMES,
                X_train=X_train_full_k,
                y_train=y_train_full,
                X_target=X_test_k,
                seed=seed,
                n_splits=OOF_N_SPLITS,
            )

            meta, pred_test = fit_meta_and_predict(
                meta_kind=meta_kind,
                oof_mat=oof_2,
                y_train=y_train_full,
                target_mat=test_mat_2,
            )

            meta_coef_rows.extend(
                meta_coeff_records(
                    meta=meta,
                    method=method,
                    seed=seed,
                    selected_k=selected_k,
                    model_names=TWO_MODEL_NAMES,
                )
            )

            meta_alpha = float(getattr(meta, "alpha_", np.nan)) if hasattr(meta, "alpha_") else np.nan

        elif method in ["OOF_Linear_4Model", "OOF_Ridge_4Model"]:
            meta_kind = "Linear" if "Linear" in method else "Ridge"

            oof_4, test_mat_4 = make_oof_and_target_predictions(
                model_names=FOUR_MODEL_NAMES,
                X_train=X_train_full_k,
                y_train=y_train_full,
                X_target=X_test_k,
                seed=seed,
                n_splits=OOF_N_SPLITS,
            )

            meta, pred_test = fit_meta_and_predict(
                meta_kind=meta_kind,
                oof_mat=oof_4,
                y_train=y_train_full,
                target_mat=test_mat_4,
            )

            meta_coef_rows.extend(
                meta_coeff_records(
                    meta=meta,
                    method=method,
                    seed=seed,
                    selected_k=selected_k,
                    model_names=FOUR_MODEL_NAMES,
                )
            )

            meta_alpha = float(getattr(meta, "alpha_", np.nan)) if hasattr(meta, "alpha_") else np.nan

        else:
            raise ValueError(f"Unknown method: {method}")

        metrics = regression_metrics(y_test, pred_test)

        test_rows.append({
            "Seed": seed,
            "Method": method,
            "Selected_K": selected_k,
            "Validation_RMSE": best_row["Validation_RMSE"],
            "Validation_R2": best_row["Validation_R2"],
            "Validation_MAE": best_row["Validation_MAE"],
            "Meta_Alpha": meta_alpha,
            **metrics,
            "n_test": len(y_test),
        })

        for yt, yp in zip(y_test, pred_test):
            prediction_rows.append({
                "Seed": seed,
                "Method": method,
                "Selected_K": selected_k,
                "Observed": float(yt),
                "Predicted": float(yp),
                "Residual": float(yt - yp),
            })

    return {
        "candidate": candidate_df_seed,
        "test": pd.DataFrame(test_rows),
        "prediction": pd.DataFrame(prediction_rows),
        "meta_coef": pd.DataFrame(meta_coef_rows),
    }


# ============================================================
# 5. Summary and decision
# ============================================================

def summarize_results(test_df: pd.DataFrame, prediction_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        test_df
        .groupby("Method", as_index=False)
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            Bias_mean=("Bias", "mean"),
            Bias_std=("Bias", "std"),
            Selected_K=("Selected_K", format_selected_k),
            n_seed=("Seed", "nunique"),
        )
    )

    pooled_rows = []

    for method, sub in prediction_df.groupby("Method"):
        metrics = regression_metrics(sub["Observed"], sub["Predicted"])

        pooled_rows.append({
            "Method": method,
            "Pooled_R2": metrics["R2"],
            "Pooled_RMSE": metrics["RMSE"],
            "Pooled_MAE": metrics["MAE"],
            "Pooled_MSE": metrics["MSE"],
            "Pooled_Bias": metrics["Bias"],
            "n_pooled": len(sub),
        })

    pooled_df = pd.DataFrame(pooled_rows)

    summary = summary.merge(pooled_df, on="Method", how="left")

    summary["Method"] = pd.Categorical(
        summary["Method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    summary = summary.sort_values("Method").reset_index(drop=True)

    return summary


def write_decision_report(summary_df: pd.DataFrame, test_df: pd.DataFrame):
    path = TABLE_DIR / "27_oof_stacking_decision.txt"

    lines = []
    lines.append("=" * 100)
    lines.append("Step 27 OOF stacking benchmark decision report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("This benchmark uses the real script-19 workflow.")
    lines.append("Only the fusion strategy was changed:")
    lines.append("validation-RMSE weighted averaging vs OOF Linear/Ridge stacking.")
    lines.append("")

    keep_cols = [
        "Method",
        "R2_mean",
        "R2_std",
        "RMSE_mean",
        "RMSE_std",
        "MAE_mean",
        "MAE_std",
        "Pooled_R2",
        "Pooled_RMSE",
        "Pooled_MAE",
        "Selected_K",
        "n_pooled",
    ]

    lines.append("Summary:")
    lines.append(summary_df[keep_cols].to_string(index=False))
    lines.append("")

    base = summary_df[summary_df["Method"].astype(str) == "Weighted_2Model"]

    if base.empty:
        lines.append("Weighted_2Model baseline not found. Decision cannot be made.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    base = base.iloc[0]

    lines.append("Comparison against current final Weighted_2Model:")
    lines.append("-" * 100)

    for method in METHOD_ORDER:
        if method == "Weighted_2Model":
            continue

        row = summary_df[summary_df["Method"].astype(str) == method]

        if row.empty:
            continue

        row = row.iloc[0]

        d_r2 = row["Pooled_R2"] - base["Pooled_R2"]
        d_rmse = row["Pooled_RMSE"] - base["Pooled_RMSE"]
        d_mae = row["Pooled_MAE"] - base["Pooled_MAE"]

        # Seed-level improvement count.
        merged = test_df[test_df["Method"].isin(["Weighted_2Model", method])].copy()
        pivot = merged.pivot_table(
            index="Seed",
            columns="Method",
            values="RMSE",
            aggfunc="first",
        )

        if "Weighted_2Model" in pivot.columns and method in pivot.columns:
            n_improve = int((pivot[method] < pivot["Weighted_2Model"]).sum())
            n_total = int(pivot[[method, "Weighted_2Model"]].dropna().shape[0])
        else:
            n_improve = 0
            n_total = 0

        lines.append(
            f"{method}: ΔR2={d_r2:+.4f}, ΔRMSE={d_rmse:+.4f}, "
            f"ΔMAE={d_mae:+.4f}, seed RMSE improved={n_improve}/{n_total}"
        )

    lines.append("")
    lines.append("Decision rule:")
    lines.append("Adopt stacking only if it reduces pooled RMSE by at least 0.20, does not increase pooled MAE,")
    lines.append("and improves seed-level RMSE in at least 3 out of 5 seeds.")
    lines.append("")

    # Best stacking candidate.
    stacking = summary_df[
        summary_df["Method"].astype(str).str.contains("OOF", case=False, na=False)
    ].copy()

    if not stacking.empty:
        best = stacking.sort_values(["Pooled_RMSE", "Pooled_MAE"], ascending=[True, True]).iloc[0]

        d_rmse = best["Pooled_RMSE"] - base["Pooled_RMSE"]
        d_mae = best["Pooled_MAE"] - base["Pooled_MAE"]

        merged = test_df[test_df["Method"].isin(["Weighted_2Model", str(best["Method"])])].copy()
        pivot = merged.pivot_table(
            index="Seed",
            columns="Method",
            values="RMSE",
            aggfunc="first",
        )

        if "Weighted_2Model" in pivot.columns and str(best["Method"]) in pivot.columns:
            n_improve = int((pivot[str(best["Method"])] < pivot["Weighted_2Model"]).sum())
            n_total = int(pivot[[str(best["Method"]), "Weighted_2Model"]].dropna().shape[0])
        else:
            n_improve = 0
            n_total = 0

        lines.append(f"Best stacking candidate: {best['Method']}")
        lines.append(f"ΔRMSE vs Weighted_2Model = {d_rmse:+.4f}")
        lines.append(f"ΔMAE  vs Weighted_2Model = {d_mae:+.4f}")
        lines.append(f"Seed-level RMSE improvement = {n_improve}/{n_total}")
        lines.append("")

        if d_rmse <= -0.20 and d_mae <= 0 and n_improve >= 3:
            lines.append("Conclusion:")
            lines.append("OOF stacking provides a practically meaningful and stable improvement.")
            lines.append("It can be considered as a candidate final model, but interpretability and complexity must still be discussed.")
        else:
            lines.append("Conclusion:")
            lines.append("OOF stacking does not provide sufficient stable improvement over the current Weighted_2Model.")
            lines.append("Keep the CatBoost-HGB validation-weighted 2Model as the main model and present stacking as a benchmark/robustness check.")
    else:
        lines.append("No stacking candidate found.")

    lines.append("")
    lines.append("=" * 100)

    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[TEXT] {path}")


# ============================================================
# 6. Figures
# ============================================================

def plot_metrics(summary_df: pd.DataFrame):
    data = summary_df.copy()
    data["Method_str"] = data["Method"].astype(str)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9))

    specs = [
        ("Pooled_R2", "Pooled $R^2$", "{:.3f}"),
        ("Pooled_RMSE", "Pooled RMSE", "{:.2f}"),
        ("Pooled_MAE", "Pooled MAE", "{:.2f}"),
    ]

    for ax, (col, ylabel, fmt) in zip(axes, specs):
        x = np.arange(len(data))
        vals = data[col].astype(float).values
        colors = [COLORS.get(m, COLORS["gray"]) for m in data["Method_str"]]

        bars = ax.bar(
            x,
            vals,
            color=colors,
            edgecolor="black",
            linewidth=0.45,
            alpha=0.90,
        )

        for bar, method in zip(bars, data["Method_str"]):
            if method == "Weighted_2Model":
                bar.set_edgecolor(COLORS["red"])
                bar.set_linewidth(1.3)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [METHOD_LABELS.get(m, m) for m in data["Method_str"]],
            rotation=28,
            ha="right",
        )

        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)

        if col == "Pooled_R2":
            ax.set_ylim(0, min(1.0, max(0.90, np.nanmax(vals) * 1.10)))
            offset = ax.get_ylim()[1] * 0.018
        else:
            ax.set_ylim(0, np.nanmax(vals) * 1.22)
            offset = ax.get_ylim()[1] * 0.025

        for xi, v in zip(x, vals):
            ax.text(
                xi,
                v + offset,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle(
        "OOF stacking benchmark under the selected-feature fusion workflow",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.92])

    save_fig(fig, "77_oof_stacking_metrics_comparison.png")


def plot_pooled_predicted_vs_actual(pred_df: pd.DataFrame):
    methods_to_show = [
        "Weighted_2Model",
        "OOF_Linear_2Model",
        "OOF_Ridge_2Model",
        "OOF_Ridge_4Model",
    ]

    available = [m for m in methods_to_show if m in pred_df["Method"].unique()]

    n = len(available)
    ncols = 2
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11.5, 5.3 * nrows),
        sharex=True,
        sharey=True,
    )

    if nrows == 1:
        axes = np.array(axes).reshape(1, -1)

    axes_flat = axes.flatten()

    for ax, method in zip(axes_flat, available):
        sub = pred_df[pred_df["Method"] == method].copy()

        if len(sub) > 900:
            sub_plot = sub.sample(900, random_state=42)
        else:
            sub_plot = sub

        color = COLORS.get(method, COLORS["gray"])

        ax.scatter(
            sub_plot["Observed"],
            sub_plot["Predicted"],
            s=14,
            color=color,
            alpha=0.68,
            edgecolor="none",
        )

        ax.plot(
            [0, 100],
            [0, 100],
            color=COLORS["red"],
            linestyle=(0, (4, 3)),
            linewidth=1.0,
        )

        metrics = regression_metrics(sub["Observed"], sub["Predicted"])

        ax.text(
            0.06,
            0.90,
            f"$R^2$ = {metrics['R2']:.3f}\nRMSE = {metrics['RMSE']:.2f}\nMAE = {metrics['MAE']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.78,
                boxstyle="round,pad=0.3",
            )
        )

        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Observed F removal (%)")
        ax.set_ylabel("Predicted F removal (%)")

    for ax in axes_flat[len(available):]:
        ax.axis("off")

    fig.suptitle(
        "Pooled predicted-vs-observed comparison of weighted fusion and OOF stacking",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])

    save_fig(fig, "78_oof_stacking_pooled_predicted_vs_actual.png")


def plot_meta_coefficients(meta_df: pd.DataFrame):
    if meta_df.empty:
        log("[WARN] No meta coefficients to plot.")
        return

    methods = [
        "OOF_Linear_2Model",
        "OOF_Ridge_2Model",
        "OOF_Linear_4Model",
        "OOF_Ridge_4Model",
    ]

    data = meta_df[meta_df["Method"].isin(methods)].copy()

    if data.empty:
        log("[WARN] No selected meta coefficient rows.")
        return

    summary = (
        data
        .groupby(["Method", "Base_Model"], as_index=False)
        .agg(
            mean_coef=("Coefficient", "mean"),
            std_coef=("Coefficient", "std"),
            n=("Coefficient", "count"),
        )
    )

    n_methods = len(methods)

    fig, axes = plt.subplots(
        1,
        n_methods,
        figsize=(4.2 * n_methods, 4.8),
        sharey=False,
    )

    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        sub = summary[summary["Method"] == method].copy()

        if sub.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No coefficient data", ha="center", va="center")
            continue

        sub["Base_Model_Label"] = sub["Base_Model"].map(model_short_name)
        sub = sub.sort_values("mean_coef", ascending=True)

        y_pos = np.arange(len(sub))

        ax.barh(
            y_pos,
            sub["mean_coef"],
            xerr=sub["std_coef"].fillna(0),
            color=[COLORS.get(m, COLORS["gray"]) for m in sub["Base_Model"]],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            capsize=3,
        )

        ax.axvline(0, color=COLORS["red"], linestyle=(0, (4, 3)), linewidth=1.0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["Base_Model_Label"])
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xlabel("Mean meta coefficient")

        for yi, v in zip(y_pos, sub["mean_coef"]):
            ax.text(
                v,
                yi,
                f"{v:.2f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=7,
            )

    fig.suptitle(
        "OOF stacking meta-model coefficients across random seeds",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.93])

    save_fig(fig, "79_oof_stacking_meta_coefficients.png")


# ============================================================
# 7. Main
# ============================================================

def main():
    log("\n" + "=" * 100)
    log("Step 27: Strict OOF stacking benchmark started")
    log("=" * 100)
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"TABLE_DIR    = {TABLE_DIR}")
    log(f"FIG_DIR      = {FIG_DIR}")
    log(f"SEEDS        = {sf19.SEEDS}")
    log(f"K_OPTIONS    = {sf19.K_OPTIONS}")
    log(f"OOF folds    = {OOF_N_SPLITS}")
    log("=" * 100)

    check_required_models()

    log("\nPreparing feature sets using script 19...")
    y, X_raw, X_eng, X_mix = sf19.prepare_feature_sets()

    log("\nFeature sets:")
    log(f"Sample size      = {len(y)}")
    log(f"Raw features     = {X_raw.shape[1]}")
    log(f"Engineered       = {X_eng.shape[1]}")
    log(f"Mixed features   = {X_mix.shape[1]}")

    all_candidates = []
    all_tests = []
    all_predictions = []
    all_meta_coefs = []

    for seed in sf19.SEEDS:
        result = run_one_seed(X_mix=X_mix, y=y, seed=seed)

        all_candidates.append(result["candidate"])
        all_tests.append(result["test"])
        all_predictions.append(result["prediction"])

        if not result["meta_coef"].empty:
            all_meta_coefs.append(result["meta_coef"])

    candidate_df = pd.concat(all_candidates, ignore_index=True)
    test_df = pd.concat(all_tests, ignore_index=True)
    prediction_df = pd.concat(all_predictions, ignore_index=True)

    if all_meta_coefs:
        meta_coef_df = pd.concat(all_meta_coefs, ignore_index=True)
    else:
        meta_coef_df = pd.DataFrame()

    summary_df = summarize_results(test_df, prediction_df)

    # Save tables.
    candidate_path = TABLE_DIR / "27_oof_stacking_candidate_results.csv"
    test_path = TABLE_DIR / "27_oof_stacking_test_results.csv"
    pred_path = TABLE_DIR / "27_oof_stacking_predictions.csv"
    coef_path = TABLE_DIR / "27_oof_stacking_meta_coefficients.csv"
    summary_path = TABLE_DIR / "27_oof_stacking_summary.csv"

    candidate_df.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    prediction_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    meta_coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    log("\n[OK] Tables saved:")
    for p in [candidate_path, test_path, pred_path, coef_path, summary_path]:
        log(f"  - {p}")

    log("\n=== Step 27 summary ===")
    print(summary_df.to_string(index=False))

    write_decision_report(summary_df, test_df)

    log("\nGenerating figures...")
    plot_metrics(summary_df)
    plot_pooled_predicted_vs_actual(prediction_df)
    plot_meta_coefficients(meta_coef_df)

    log("\n" + "=" * 100)
    log("[DONE] Step 27 OOF stacking benchmark completed.")
    log("=" * 100)


if __name__ == "__main__":
    main()