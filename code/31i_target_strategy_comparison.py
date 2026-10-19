# -*- coding: utf-8 -*-
"""
31i_target_strategy_comparison.py

Purpose
-------
Target-strategy comparison under the same grouped-generalization framework used
in Step 31G/31H.

This is NOT a simplified pipeline. It reuses:
    - Step 31B mechanism-aware descriptors
    - Step 31F balanced literature-level grouped split
    - literature-group-weighted training
    - script-19 preprocessing / feature-selection utilities where applicable
    - Step 31G CatBoost/XGBoost model defaults and validation-RMSE fusion logic

Main question
-------------
Can changing the target representation improve all-domain grouped prediction of
F removal (%)?

Compared target strategies
--------------------------
1. Direct removal target: train on F removal (%) directly.
2. Logit-removal target: train on logit(F removal / 100), then inverse-transform.
3. Arcsin-sqrt removal target: train on arcsin(sqrt(F removal / 100)), then inverse-transform.
4. Effluent-F target, if the effluent F column exists: train on effluent F (mg/L),
   then convert predicted effluent F back to F removal (%) using initial F.

Strict rules
------------
1. 文献编号 is used only for literature-level grouped splitting.
2. 文献编号 is never used as an input feature.
3. Test labels are never used for feature selection, target transformation setup,
   weight search, or model selection.
4. Fusion weights are selected by validation RMSE on original F-removal (%) scale.
5. Final metrics are always computed on original F-removal (%) scale.

Outputs
-------
Tables:
    outputs/tables/31i_target_strategy_split_summary.csv
    outputs/tables/31i_target_strategy_validation_candidates.csv
    outputs/tables/31i_target_strategy_test_results.csv
    outputs/tables/31i_target_strategy_predictions.csv
    outputs/tables/31i_target_strategy_summary.csv
    outputs/tables/31i_target_strategy_audit.csv
    outputs/tables/31i_target_strategy_decision.txt

Figures:
    outputs/figures/103_target_strategy_metrics.png/pdf
    outputs/figures/104_target_strategy_pooled_predicted_vs_actual.png/pdf
    outputs/figures/105_target_strategy_weight_stability.png/pdf
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


# Reuse real Step 31G, which itself imports real 31F and script-19.
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

SEEDS = list(getattr(sf19, "SEEDS", [7, 21, 42, 66, 88]))
K_OPTIONS = list(getattr(sf19, "K_OPTIONS", ["all"]))
BASE_MODELS = ["CatBoost", "XGBoost"]
METHODS = ["CatBoost", "XGBoost", "CatBoost_XGBoost_Weighted_2Model"]

COLORS = {
    "CatBoost": "#9B72B0",
    "XGBoost": "#D7903E",
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
# Utility functions
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
    if s == "":
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
        if str(s).lower() == "raw":
            return (-1, 0)
        if str(s).lower() == "all":
            return (999999, 0)
        try:
            return (int(float(s)), 0)
        except Exception:
            return (999998, str(s))
    return ",".join(sorted(set(vals), key=sort_key))


def find_column_by_keywords(df: pd.DataFrame, keyword_groups: list[list[str]], preferred=None):
    cols = list(df.columns)
    if preferred:
        for c in preferred:
            if c in cols:
                return c
    for group in keyword_groups:
        for c in cols:
            s = str(c).lower().replace(" ", "")
            if all(k.lower().replace(" ", "") in s for k in group):
                return c
    return None


# ============================================================
# Target transformations
# ============================================================

EPS = 1e-4


def removal_to_logit(y):
    p = np.clip(np.asarray(y, dtype=float) / 100.0, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def logit_to_removal(z):
    z = np.asarray(z, dtype=float)
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return np.clip(p * 100.0, 0.0, 100.0)


def removal_to_arcsin(y):
    p = np.clip(np.asarray(y, dtype=float) / 100.0, 0.0, 1.0)
    return np.arcsin(np.sqrt(p))


def arcsin_to_removal(z):
    z = np.asarray(z, dtype=float)
    p = np.sin(np.clip(z, 0, np.pi / 2.0)) ** 2
    return np.clip(p * 100.0, 0.0, 100.0)


def effluent_to_removal(pred_effluent, initial_f):
    pred_effluent = np.asarray(pred_effluent, dtype=float)
    initial_f = np.asarray(initial_f, dtype=float)
    pred_effluent = np.clip(pred_effluent, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        removal = (initial_f - pred_effluent) / initial_f * 100.0
    removal[~np.isfinite(removal)] = np.nan
    return np.clip(removal, 0.0, 100.0)


def get_mode_train_target(mode: str, y_removal, effluent_f=None):
    if mode == "DirectRemoval":
        return pd.Series(np.asarray(y_removal, dtype=float))
    if mode == "LogitRemoval":
        return pd.Series(removal_to_logit(y_removal))
    if mode == "ArcsinRemoval":
        return pd.Series(removal_to_arcsin(y_removal))
    if mode == "EffluentF":
        if effluent_f is None:
            raise ValueError("EffluentF mode requires effluent-F target column.")
        return pd.Series(np.asarray(effluent_f, dtype=float))
    raise ValueError(f"Unknown target mode: {mode}")


def inverse_to_removal(mode: str, pred_model_scale, initial_f=None):
    if mode == "DirectRemoval":
        return np.clip(np.asarray(pred_model_scale, dtype=float), 0.0, 100.0)
    if mode == "LogitRemoval":
        return logit_to_removal(pred_model_scale)
    if mode == "ArcsinRemoval":
        return arcsin_to_removal(pred_model_scale)
    if mode == "EffluentF":
        if initial_f is None:
            raise ValueError("EffluentF inverse conversion requires initial F.")
        return effluent_to_removal(pred_model_scale, initial_f)
    raise ValueError(f"Unknown target mode: {mode}")


# ============================================================
# Model evaluation for one target mode
# ============================================================

def select_columns_from_importance(imp_df: pd.DataFrame, X_mix: pd.DataFrame, selected_k):
    return s31g.select_columns_from_importance(imp_df, X_mix, selected_k)


def evaluate_validation_candidates_for_mode(
    mode,
    X_train_k,
    y_train_model,
    y_valid_removal,
    valid_initial_f,
    X_valid_k,
    seed,
    selected_k,
    selected_cols,
    sample_weight_train=None,
):
    rows = []
    valid_pred_by_model = {}

    for model_name in BASE_MODELS:
        pred_model_scale = s31g.fit_predict_model(
            model_name=model_name,
            X_train=X_train_k,
            y_train=y_train_model,
            X_target=X_valid_k,
            seed=seed,
            sample_weight=sample_weight_train,
        )
        pred_removal = inverse_to_removal(mode, pred_model_scale, valid_initial_f)
        valid_pred_by_model[model_name] = pred_removal
        metrics = regression_metrics(y_valid_removal, pred_removal)
        rows.append({
            "Seed": seed,
            "Target_Mode": mode,
            "Method": f"{mode}_{model_name}",
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

    if all(m in valid_pred_by_model for m in BASE_MODELS):
        weights, _ = sf19.search_best_weights(valid_pred_by_model, y_valid_removal, seed)
        pred_fused = sf19.apply_weights(valid_pred_by_model, weights)
        metrics = regression_metrics(y_valid_removal, pred_fused)
        rows.append({
            "Seed": seed,
            "Target_Mode": mode,
            "Method": f"{mode}_CatBoost_XGBoost_Weighted_2Model",
            "Base_Models": "CatBoost;XGBoost",
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


def predict_test_for_best_candidate(mode, method, weights_text, X_train_full_k, y_train_full_model, X_test_k, test_initial_f, seed, sample_weight_train_full=None):
    base_preds = {}
    for model_name in BASE_MODELS:
        pred_model_scale = s31g.fit_predict_model(
            model_name=model_name,
            X_train=X_train_full_k,
            y_train=y_train_full_model,
            X_target=X_test_k,
            seed=seed,
            sample_weight=sample_weight_train_full,
        )
        base_preds[model_name] = inverse_to_removal(mode, pred_model_scale, test_initial_f)

    if method.endswith("_CatBoost"):
        return base_preds["CatBoost"]
    if method.endswith("_XGBoost"):
        return base_preds["XGBoost"]
    if method.endswith("_CatBoost_XGBoost_Weighted_2Model"):
        weights = parse_weight_text(weights_text)
        return sf19.apply_weights(base_preds, weights)
    raise ValueError(f"Unknown method for test prediction: {method}")


def run_one_seed_one_mode(X_mix, y_removal, y_effluent, initial_f, groups, seed: int, mode: str):
    log("\n" + "=" * 100)
    log(f"31I target-strategy comparison | seed={seed} | mode={mode}")
    log("=" * 100)

    train_idx, valid_idx, test_idx, split_info = s31f.grouped_split_indices(groups, seed)
    train_valid_idx = np.concatenate([train_idx, valid_idx])

    X_train = X_mix.iloc[train_idx].reset_index(drop=True)
    X_valid = X_mix.iloc[valid_idx].reset_index(drop=True)
    X_test = X_mix.iloc[test_idx].reset_index(drop=True)
    X_train_full = X_mix.iloc[train_valid_idx].reset_index(drop=True)

    y_train_removal = y_removal.iloc[train_idx].reset_index(drop=True)
    y_valid_removal = y_removal.iloc[valid_idx].reset_index(drop=True)
    y_test_removal = y_removal.iloc[test_idx].reset_index(drop=True)
    y_train_full_removal = y_removal.iloc[train_valid_idx].reset_index(drop=True)

    eff_train = None if y_effluent is None else y_effluent.iloc[train_idx].reset_index(drop=True)
    eff_train_full = None if y_effluent is None else y_effluent.iloc[train_valid_idx].reset_index(drop=True)

    initial_valid = initial_f.iloc[valid_idx].reset_index(drop=True) if initial_f is not None else None
    initial_test = initial_f.iloc[test_idx].reset_index(drop=True) if initial_f is not None else None

    train_groups_for_weight = groups.iloc[train_idx].reset_index(drop=True)
    train_full_groups_for_weight = groups.iloc[train_valid_idx].reset_index(drop=True)
    test_groups = groups.iloc[test_idx].reset_index(drop=True)

    sample_weight_train = s31f.make_literature_group_weights(train_groups_for_weight)
    sample_weight_train_full = s31f.make_literature_group_weights(train_full_groups_for_weight)

    y_train_model = get_mode_train_target(mode, y_train_removal, eff_train)
    y_train_full_model = get_mode_train_target(mode, y_train_full_removal, eff_train_full)

    # Feature ranking is performed only on grouped training data. For transformed
    # targets, rank on the transformed training target; for effluent target, rank
    # on effluent F. Final validation/test metrics remain on removal scale.
    imp_df = s31f.rank_features_by_rf_importance_weighted(
        X_train,
        y_train_model,
        seed,
        sample_weight=sample_weight_train,
    )

    candidate_rows = []
    for selected_k in K_OPTIONS:
        selected_cols = select_columns_from_importance(imp_df, X_mix, selected_k)
        if len(selected_cols) < 3:
            continue
        X_train_k = X_train[selected_cols].reset_index(drop=True)
        X_valid_k = X_valid[selected_cols].reset_index(drop=True)
        rows = evaluate_validation_candidates_for_mode(
            mode=mode,
            X_train_k=X_train_k,
            y_train_model=y_train_model,
            y_valid_removal=y_valid_removal,
            valid_initial_f=initial_valid,
            X_valid_k=X_valid_k,
            seed=seed,
            selected_k=selected_k,
            selected_cols=selected_cols,
            sample_weight_train=sample_weight_train,
        )
        candidate_rows.extend(rows)

    candidate_df = pd.DataFrame(candidate_rows)
    test_rows = []
    pred_rows = []

    for method in sorted(candidate_df["Method"].unique()):
        sub = candidate_df[candidate_df["Method"] == method].copy()
        best = sub.sort_values("Validation_RMSE", ascending=True).iloc[0]
        selected_cols = str(best["Selected_Features"]).split(";")
        selected_cols = [c for c in selected_cols if c in X_mix.columns]

        X_train_full_k = X_train_full[selected_cols].reset_index(drop=True)
        X_test_k = X_test[selected_cols].reset_index(drop=True)
        pred_test = predict_test_for_best_candidate(
            mode=mode,
            method=method,
            weights_text=best["Weights"],
            X_train_full_k=X_train_full_k,
            y_train_full_model=y_train_full_model,
            X_test_k=X_test_k,
            test_initial_f=initial_test,
            seed=seed,
            sample_weight_train_full=sample_weight_train_full,
        )
        metrics = regression_metrics(y_test_removal, pred_test)

        test_rows.append({
            "Seed": seed,
            "Target_Mode": mode,
            "Method": method,
            "Base_Models": best.get("Base_Models", ""),
            "Selected_K": best["Selected_K"],
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
            "n_test": len(y_test_removal),
            "n_test_groups": test_groups.nunique(),
        })

        for local_i, (obs, pred, gid, data_idx) in enumerate(zip(y_test_removal, pred_test, test_groups, test_idx)):
            pred_rows.append({
                "Seed": seed,
                "Target_Mode": mode,
                "Method": method,
                "Base_Models": best.get("Base_Models", ""),
                "Selected_K": best["Selected_K"],
                "Data_Index": int(data_idx),
                "Local_Test_Index": int(local_i),
                "文献编号": gid,
                "Observed": float(obs),
                "Predicted": float(pred),
                "Residual": float(obs - pred),
            })

    split_info = dict(split_info)
    split_info["Target_Mode"] = mode
    return {
        "split": pd.DataFrame([split_info]),
        "candidate": candidate_df,
        "test": pd.DataFrame(test_rows),
        "prediction": pd.DataFrame(pred_rows),
    }


# ============================================================
# Summary and figures
# ============================================================

def summarize_results(test_df: pd.DataFrame, pred_df: pd.DataFrame):
    summary = (
        test_df.groupby(["Target_Mode", "Method"], as_index=False)
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
    for (mode, method), sub in pred_df.groupby(["Target_Mode", "Method"]):
        metrics = regression_metrics(sub["Observed"], sub["Predicted"])
        pooled_rows.append({
            "Target_Mode": mode,
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
    summary = summary.merge(pooled_df, on=["Target_Mode", "Method"], how="left")
    summary = summary.sort_values(["Pooled_RMSE", "Pooled_MAE", "Pooled_R2"], ascending=[True, True, False]).reset_index(drop=True)
    return summary


def short_label(method: str):
    return (method.replace("DirectRemoval_", "Direct-")
                  .replace("LogitRemoval_", "Logit-")
                  .replace("ArcsinRemoval_", "Arcsin-")
                  .replace("EffluentF_", "EffluentF-")
                  .replace("CatBoost_XGBoost_Weighted_2Model", "CB-XGB 2Model"))


def plot_metrics(summary_df: pd.DataFrame):
    data = summary_df.copy().head(15)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    specs = [
        ("Pooled_R2", "Pooled $R^2$", "{:.3f}", True),
        ("Pooled_RMSE", "Pooled RMSE", "{:.2f}", False),
        ("Pooled_MAE", "Pooled MAE", "{:.2f}", False),
    ]
    labels = [short_label(m) for m in data["Method"]]
    x = np.arange(len(data))
    for ax, (col, ylabel, fmt, high_better) in zip(axes, specs):
        vals = data[col].astype(float).values
        colors = [COLORS.get("CatBoost_XGBoost_Weighted_2Model" if "Weighted" in m else "CatBoost", COLORS["gray"]) for m in data["Method"]]
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.45, alpha=0.88)
        best_idx = int(np.nanargmax(vals)) if high_better else int(np.nanargmin(vals))
        bars[best_idx].set_edgecolor(COLORS["red"])
        bars[best_idx].set_linewidth(1.5)
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        ymin, ymax = np.nanmin(vals), np.nanmax(vals)
        if col == "Pooled_R2":
            ax.set_ylim(min(-0.15, ymin - 0.03), max(0.20, ymax + 0.04))
            offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.025
        else:
            ax.set_ylim(0, ymax * 1.20)
            offset = ax.get_ylim()[1] * 0.02
        for xi, v in zip(x, vals):
            ax.text(xi, v + (offset if v >= 0 else -offset), fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    fig.suptitle("Target-strategy comparison under balanced literature-level grouped validation", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "All metrics are computed on original F removal (%) scale; fusion weights are selected by validation RMSE.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0.01, 0.06, 1, 0.93])
    save_fig(fig, "103_target_strategy_metrics.png")


def plot_pooled_prediction(pred_df: pd.DataFrame, summary_df: pd.DataFrame):
    methods = list(summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE"], ascending=[True, True])["Method"].head(4))
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 10.5), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, method in zip(axes, methods):
        sub = pred_df[pred_df["Method"] == method].copy()
        sub_plot = sub.sample(900, random_state=42) if len(sub) > 900 else sub
        color = COLORS.get("CatBoost_XGBoost_Weighted_2Model" if "Weighted" in method else "CatBoost", COLORS["gray"])
        ax.scatter(sub_plot["Observed"], sub_plot["Predicted"], s=16, color=color, alpha=0.68, edgecolor="none")
        ax.plot([0, 100], [0, 100], color=COLORS["red"], linestyle=(0, (4, 3)), linewidth=1.0)
        metrics = regression_metrics(sub["Observed"], sub["Predicted"])
        ax.text(0.06, 0.92, f"$R^2$ = {metrics['R2']:.3f}\nRMSE = {metrics['RMSE']:.2f}\nMAE = {metrics['MAE']:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, boxstyle="round,pad=0.3"))
        ax.set_title(short_label(method))
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Observed F removal (%)")
        ax.set_ylabel("Predicted F removal (%)")
        ax.grid(color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    fig.suptitle("Pooled predicted-vs-observed plots for target-strategy candidates", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    save_fig(fig, "104_target_strategy_pooled_predicted_vs_actual.png")


def collect_weight_rows(test_df: pd.DataFrame):
    rows = []
    for _, row in test_df.iterrows():
        weights = parse_weight_text(row["Weights"])
        if not weights:
            continue
        for base_model, weight in weights.items():
            rows.append({"Seed": row["Seed"], "Target_Mode": row["Target_Mode"], "Method": row["Method"], "BaseModel": base_model, "Weight": float(weight)})
    return pd.DataFrame(rows)


def plot_weight_stability(test_df: pd.DataFrame):
    weight_df = collect_weight_rows(test_df)
    if weight_df.empty:
        return
    top_methods = [m for m in test_df["Method"].unique() if "Weighted" in m][:4]
    fig, axes = plt.subplots(1, len(top_methods), figsize=(5.2 * len(top_methods), 4.3), sharey=True)
    if len(top_methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, top_methods):
        sub = weight_df[weight_df["Method"] == method].copy()
        for base_model, sub_m in sub.groupby("BaseModel"):
            sub_m = sub_m.sort_values("Seed")
            ax.plot(sub_m["Seed"], sub_m["Weight"], marker="o", linewidth=1.5, markersize=4, label=base_model,
                    color=COLORS.get(base_model, COLORS["gray"]))
            for _, r in sub_m.iterrows():
                ax.text(r["Seed"], r["Weight"] + 0.025, f"{r['Weight']:.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_title(short_label(method))
        ax.set_xlabel("Random seed")
        ax.set_ylim(-0.05, 1.12)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Validation-optimized fusion weight")
    fig.suptitle("Fusion weight stability across target strategies", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.92])
    save_fig(fig, "105_target_strategy_weight_stability.png")


def write_report(summary_df, split_df, audit_rows):
    path = TABLE_DIR / "31i_target_strategy_decision.txt"
    ranking = summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE", "Pooled_R2"], ascending=[True, True, False]).reset_index(drop=True)
    best = ranking.iloc[0]
    lines = []
    lines.append("=" * 100)
    lines.append("Step 31I target-strategy comparison report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Evaluate whether target transformation or effluent-F target modeling improves all-domain grouped prediction of F removal (%).")
    lines.append("")
    lines.append("Strict workflow reused:")
    lines.append("- Step-31B mechanism-aware descriptors")
    lines.append("- Step-31F balanced grouped split")
    lines.append("- Literature-group-weighted training")
    lines.append("- Step-31G CatBoost/XGBoost defaults and validation-RMSE fusion logic")
    lines.append("- All final metrics computed on original F removal (%) scale")
    lines.append("")
    lines.append("Audit:")
    lines.append(pd.DataFrame(audit_rows).to_string(index=False))
    lines.append("")
    lines.append("Split summary:")
    lines.append(split_df[["Seed", "Target_Mode", "n_train", "n_valid", "n_test", "n_train_groups", "n_valid_groups", "n_test_groups"]].head(20).to_string(index=False))
    lines.append("")
    lines.append("Performance summary:")
    keep = ["Target_Mode", "Method", "R2_mean", "R2_std", "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std", "Pooled_R2", "Pooled_RMSE", "Pooled_MAE", "Pooled_Bias", "n_pooled"]
    lines.append(summary_df[keep].to_string(index=False))
    lines.append("")
    lines.append("Ranking by pooled RMSE:")
    lines.append(ranking[["Target_Mode", "Method", "Pooled_R2", "Pooled_RMSE", "Pooled_MAE", "Pooled_Bias"]].to_string(index=False))
    lines.append("")
    lines.append("Current Step-31I best candidate:")
    lines.append(f"{best['Method']} | Target_Mode={best['Target_Mode']} | Pooled_R2={best['Pooled_R2']:.4f}, Pooled_RMSE={best['Pooled_RMSE']:.4f}, Pooled_MAE={best['Pooled_MAE']:.4f}")
    lines.append("")
    lines.append("Decision guide:")
    lines.append("1. If a target strategy improves over Step 31G, keep it as the next grouped-generalization candidate.")
    lines.append("2. If no strategy improves over Step 31G, retain Step 31G CatBoost-XGBoost and move to applicability-domain analysis or targeted data expansion.")
    lines.append("3. Effluent-F modeling is only valid if effluent F is a measured/credible column and is excluded from input features.")
    lines.append("=" * 100)
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[TEXT] {path}")


# ============================================================
# Main
# ============================================================

def main():
    log("\n" + "=" * 100)
    log("Step 31I: target-strategy comparison")
    log("=" * 100)
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"SEEDS = {SEEDS}")

    log("Preparing script-19 feature sets...")
    y_removal, X_raw, X_eng, X_mix_base = sf19.prepare_feature_sets()
    y_removal = y_removal.reset_index(drop=True).astype(float)
    X_mix_base = X_mix_base.reset_index(drop=True)

    log("Appending Step-31B mechanism-aware descriptors using real 31F integration...")
    X_mix = s31f.append_mechanism31b_features(X_mix_base, expected_n=len(y_removal)).reset_index(drop=True)

    # Load full processed table to find initial-F and effluent-F columns.
    if not MECHANISM31B_PATH.exists():
        raise FileNotFoundError(f"Cannot find {MECHANISM31B_PATH}. Run Step 31B first.")
    df_full = read_csv_smart(MECHANISM31B_PATH).reset_index(drop=True)
    if len(df_full) != len(y_removal):
        raise ValueError("Row mismatch between script-19 target and mechanism31b table.")

    initial_col = find_column_by_keywords(
        df_full,
        keyword_groups=[["初始", "氟"], ["初始", "F"], ["initial", "f"]],
        preferred=["初始氟_mg_L", "初始 F-(mg/L)", "初始F_mg_L"],
    )
    effluent_col = find_column_by_keywords(
        df_full,
        keyword_groups=[["出水", "氟"], ["出水", "F"], ["effluent", "f"]],
        preferred=["出水氟_mg_L", "出水 F-(mg/L)", "出水F_mg_L"],
    )

    y_effluent = None
    initial_f = None
    target_modes = ["DirectRemoval", "LogitRemoval", "ArcsinRemoval"]
    audit_rows = []
    audit_rows.append({"Item": "Initial_F_column_detected", "Value": initial_col if initial_col else "NOT_FOUND"})
    audit_rows.append({"Item": "Effluent_F_column_detected", "Value": effluent_col if effluent_col else "NOT_FOUND"})

    if initial_col is not None:
        initial_f = pd.to_numeric(df_full[initial_col], errors="coerce").reset_index(drop=True)
    if effluent_col is not None and initial_f is not None:
        y_effluent = pd.to_numeric(df_full[effluent_col], errors="coerce").reset_index(drop=True)
        valid_eff = np.isfinite(y_effluent) & np.isfinite(initial_f) & (initial_f > 0)
        audit_rows.append({"Item": "Effluent_F_valid_rows", "Value": int(valid_eff.sum())})
        if int(valid_eff.sum()) == len(y_removal):
            target_modes.append("EffluentF")
        else:
            audit_rows.append({"Item": "EffluentF_mode", "Value": "SKIPPED_due_to_missing_or_invalid_rows"})
    else:
        audit_rows.append({"Item": "EffluentF_mode", "Value": "SKIPPED_due_to_missing_column"})

    # Remove direct effluent-F columns from feature input if present. This is
    # essential for EffluentF target modeling and harmless if the column is absent.
    leakage_drop_cols = [c for c in X_mix.columns if any(k.lower() in str(c).lower() for k in ["出水", "effluent"])]
    if leakage_drop_cols:
        X_mix = X_mix.drop(columns=leakage_drop_cols)
    audit_rows.append({"Item": "Feature_columns_after_leakage_drop", "Value": X_mix.shape[1]})
    audit_rows.append({"Item": "Dropped_effluent_like_feature_columns", "Value": ";".join(leakage_drop_cols) if leakage_drop_cols else ""})
    audit_rows.append({"Item": "Target_modes_tested", "Value": ";".join(target_modes)})

    groups = s31f.load_group_labels(expected_n=len(y_removal))
    s31f.set_balanced_split_context(X_mix, y_removal, groups)

    all_splits = []
    all_candidates = []
    all_tests = []
    all_predictions = []

    for mode in target_modes:
        for seed in SEEDS:
            result = run_one_seed_one_mode(X_mix, y_removal, y_effluent, initial_f, groups, seed, mode)
            all_splits.append(result["split"])
            all_candidates.append(result["candidate"])
            all_tests.append(result["test"])
            all_predictions.append(result["prediction"])

    split_df = pd.concat(all_splits, ignore_index=True)
    candidate_df = pd.concat(all_candidates, ignore_index=True)
    test_df = pd.concat(all_tests, ignore_index=True)
    pred_df = pd.concat(all_predictions, ignore_index=True)
    summary_df = summarize_results(test_df, pred_df)
    audit_df = pd.DataFrame(audit_rows)

    out_tables = {
        "31i_target_strategy_split_summary.csv": split_df,
        "31i_target_strategy_validation_candidates.csv": candidate_df,
        "31i_target_strategy_test_results.csv": test_df,
        "31i_target_strategy_predictions.csv": pred_df,
        "31i_target_strategy_summary.csv": summary_df,
        "31i_target_strategy_audit.csv": audit_df,
    }
    for filename, df in out_tables.items():
        path = TABLE_DIR / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        log(f"[TABLE] {path}")

    log("\n=== Step 31I target-strategy summary ===")
    print(summary_df.to_string(index=False))

    write_report(summary_df, split_df, audit_rows)
    plot_metrics(summary_df)
    plot_pooled_prediction(pred_df, summary_df)
    plot_weight_stability(test_df)

    log("\n" + "=" * 100)
    log("[DONE] Step 31I target-strategy comparison completed.")
    log("=" * 100)
    log("Send these results:")
    log("1. outputs/tables/31i_target_strategy_summary.csv")
    log("2. outputs/tables/31i_target_strategy_decision.txt")
    log("3. outputs/tables/31i_target_strategy_audit.csv")
    log("4. outputs/figures/103_target_strategy_metrics.png")
    log("5. outputs/figures/104_target_strategy_pooled_predicted_vs_actual.png")
    log("6. outputs/figures/105_target_strategy_weight_stability.png")
    log("=" * 100)


if __name__ == "__main__":
    main()
