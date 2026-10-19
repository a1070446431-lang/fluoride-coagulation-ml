# -*- coding: utf-8 -*-
"""
31m2_single_tree_krr_pairwise_fusion.py

Purpose
-------
Strict follow-up to Step 31M.

Step 31M tested:
- KRR_Poly single model
- KRR + CB-XGB weighted 2Model
- KRR + CatBoost + XGBoost weighted 3Model
- KRR/GPR/tree hybrid fusions
- grouped OOF stacking

However, Step 31M did NOT separately test whether keeping exactly one tree model
(CatBoost or XGBoost) and fusing it with KRR_Poly can outperform KRR_Poly alone.

This script fills that gap under the same strict grouped-validation route:
- Current31F_noFine_plus_31B_mechanism feature route confirmed by Step 31K
- Step 31F balanced literature-level grouped split
- literature-group weighted training
- training-only feature ranking
- validation-only model/config/weight selection
- test labels are never used for feature selection, model/config selection, or fusion-weight search

Candidate methods
-----------------
Base references:
- KRR_Poly（多项式核岭回归）
- CatBoost（类别梯度提升）
- XGBoost（极端梯度提升）
- CB-XGB（CatBoost-XGBoost 加权二模型, reference only）

New pairwise fusions:
- KRR_CatBoost_Weighted_2Model
- KRR_XGBoost_Weighted_2Model
- KRR_BestSingleTree_Weighted_2Model
  where the single tree is chosen per seed by validation RMSE between CatBoost and XGBoost.

Outputs
-------
Tables:
    outputs/tables/31m2_single_tree_krr_pairwise_split_summary.csv
    outputs/tables/31m2_single_tree_krr_pairwise_validation_candidates.csv
    outputs/tables/31m2_single_tree_krr_pairwise_test_results.csv
    outputs/tables/31m2_single_tree_krr_pairwise_predictions.csv
    outputs/tables/31m2_single_tree_krr_pairwise_weights.csv
    outputs/tables/31m2_single_tree_krr_pairwise_summary.csv
    outputs/tables/31m2_single_tree_krr_pairwise_residual_correlation.csv
    outputs/tables/31m2_single_tree_krr_pairwise_decision.txt

Figures:
    outputs/figures/123_31m2_single_tree_krr_pairwise_metrics.png/pdf
    outputs/figures/124_31m2_single_tree_krr_pairwise_weight_stability.png/pdf
    outputs/figures/125_31m2_single_tree_krr_pairwise_pooled_predicted_vs_actual.png/pdf
    outputs/figures/126_31m2_single_tree_krr_pairwise_residual_correlation.png/pdf

Total runtime is printed at the end.
"""

from __future__ import annotations

import math
import time
import warnings
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 0. Logging and dynamic import of real Step 31M module
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


# Step 31M already imports/reuses Step 31L, 31K, 31G, 31F, and script-19.
# Importing it here does not execute its main() because it is protected by __main__.
s31m = load_module_from_file("31m_hybrid_krr_tree_fusion_oof_stacking_fixed.py", "s31m_real_for_31m2")
s31l = s31m.s31l
s31k = s31m.s31k
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
K_OPTIONS = list(getattr(s31m, "K_OPTIONS", [12, 18, 25, 35, 45, 60, "all"]))
BEST_FEATURE_SPACE = "Current31F_noFine_plus_31B_mechanism"

BASE_METHODS = ["KRR_Poly", "CatBoost", "XGBoost", "CB-XGB"]
PAIRWISE_POOLS = {
    "KRR_CatBoost_Weighted_2Model": ["KRR_Poly", "CatBoost"],
    "KRR_XGBoost_Weighted_2Model": ["KRR_Poly", "XGBoost"],
    # KRR_BestSingleTree is constructed per seed because the best single tree
    # must be selected only from validation RMSE within that seed.
}

METHOD_ORDER = [
    "KRR_Poly",
    "KRR_CatBoost_Weighted_2Model",
    "KRR_XGBoost_Weighted_2Model",
    "KRR_BestSingleTree_Weighted_2Model",
    "CatBoost",
    "XGBoost",
    "CB-XGB",
]

COLORS = {
    "KRR_Poly": "#76B083",
    "CatBoost": "#9B72B0",
    "XGBoost": "#D28C3D",
    "CB-XGB": "#808080",
    "KRR_CatBoost_Weighted_2Model": "#67A9CF",
    "KRR_XGBoost_Weighted_2Model": "#B2ABD2",
    "KRR_BestSingleTree_Weighted_2Model": "#FDB863",
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
# 2. Small utilities
# ============================================================

def save_fig(fig, filename: str):
    png_path = FIG_DIR / filename
    pdf_path = FIG_DIR / filename.replace(".png", ".pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] {png_path}")
    log(f"[FIG] {pdf_path}")


def weight_dict_to_text(weight_dict: dict) -> str:
    if not weight_dict:
        return ""
    return ";".join([f"{k}:{float(v):.6f}" for k, v in weight_dict.items()])


def parse_weight_text(text) -> dict:
    if text is None or (isinstance(text, float) and np.isnan(text)) or str(text).strip() == "":
        return {}
    out = {}
    for part in str(text).split(";"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k.strip()] = float(v)
    return out


def sorted_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    out = df.copy()
    out["_order"] = out["Method"].map(order_map).fillna(999)
    out = out.sort_values(["Pooled_RMSE", "Pooled_MAE", "_order"], ascending=[True, True, True]).drop(columns=["_order"])
    return out.reset_index(drop=True)


def summarize_results(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        test_df.groupby(["Method"], as_index=False)
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
            Config_Name=("Config_Name", lambda x: ",".join(sorted(set(map(str, x))))),
            n_seed=("Seed", "nunique"),
            n_test_total=("n_test", "sum"),
        )
    )
    pooled = []
    for method, sub in pred_df.groupby("Method"):
        m = s31m.regression_metrics(sub["Observed"], sub["Predicted"])
        pooled.append({
            "Method": method,
            "Pooled_R2": m["R2"],
            "Pooled_RMSE": m["RMSE"],
            "Pooled_MAE": m["MAE"],
            "Pooled_MSE": m["MSE"],
            "Pooled_Bias": m["Bias"],
            "n_pooled": len(sub),
            "n_literature_groups_pooled": sub["文献编号"].nunique(),
        })
    summary = summary.merge(pd.DataFrame(pooled), on="Method", how="left")
    return sorted_method_summary(summary)


def compute_residual_correlation(pred_df: pd.DataFrame) -> pd.DataFrame:
    piv = pred_df.pivot_table(index=["Seed", "Data_Index"], columns="Method", values="Residual", aggfunc="mean")
    corr = piv.corr(method="pearson")
    return corr.reset_index().rename(columns={"index": "Method"})


def add_prediction_rows(pred_rows: List[dict], seed: int, method: str, y_test: pd.Series, pred: np.ndarray, test_groups: pd.Series, test_idx: np.ndarray):
    for local_i, (obs, pp, gid, data_idx) in enumerate(zip(y_test, pred, test_groups, test_idx)):
        pred_rows.append({
            "Seed": seed,
            "Method": method,
            "Data_Index": int(data_idx),
            "Local_Test_Index": int(local_i),
            "文献编号": gid,
            "Observed": float(obs),
            "Predicted": float(pp),
            "Residual": float(obs - pp),
        })


def add_test_row(test_rows: List[dict], seed: int, method: str, model_type: str, config_name: str,
                 validation_metrics: dict, weights: dict, y_test: pd.Series, pred_test: np.ndarray,
                 n_test_groups: int, selected_k="mixed", n_features=np.nan, selected_features="mixed_base_specific"):
    mt = s31m.regression_metrics(y_test, pred_test)
    test_rows.append({
        "Seed": seed,
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


# ============================================================
# 3. One-seed strict workflow
# ============================================================

def run_one_seed(X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int):
    log("\n" + "=" * 100)
    log(f"31M2 single-tree + KRR_Poly pairwise fusion | seed={seed}")
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

    # Feature ranking is done only on training literature groups.
    imp_df = s31f.rank_features_by_rf_importance_weighted(
        X_train, y_train, seed, sample_weight=sample_weight_train
    )

    candidate_parts = []
    specs = {}
    val_preds = {}

    # Reuse Step 31M real tree selection, including CatBoost, XGBoost, and CB-XGB reference.
    tree_specs, tree_val_preds, tree_cand = s31m.select_tree_and_cbxgb_specs(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        imp_df=imp_df,
        seed=seed,
        sample_weight_train=sample_weight_train,
    )
    specs.update(tree_specs)
    val_preds.update(tree_val_preds)
    candidate_parts.append(tree_cand)

    # Reuse Step 31L KRR_Poly selection.
    krr_spec, krr_valid_pred, krr_cand = s31m.select_non_tree_spec(
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
    specs["KRR_Poly"] = krr_spec
    val_preds["KRR_Poly"] = krr_valid_pred
    candidate_parts.append(krr_cand)

    candidate_df = pd.concat(candidate_parts, ignore_index=True)

    # Final base predictions retrained on train+valid groups only.
    base_test_preds: Dict[str, np.ndarray] = {}
    test_rows, pred_rows, weight_rows = [], [], []

    for base in ["KRR_Poly", "CatBoost", "XGBoost", "CB-XGB"]:
        if base not in specs:
            continue
        pred = s31m.fit_predict_selected_spec(
            spec=specs[base],
            X_train=X_train_full,
            y_train=y_train_full,
            X_target=X_test,
            seed=seed,
            sample_weight=sample_weight_train_full,
            train_groups=train_full_groups,
        )
        pred = np.clip(pred, 0, 100)
        base_test_preds[base] = pred
        spec = specs[base]
        add_test_row(
            test_rows=test_rows,
            seed=seed,
            method=base,
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
        add_prediction_rows(pred_rows, seed, base, y_test, pred, test_groups, test_idx)

    # Pairwise fusions: KRR + CatBoost and KRR + XGBoost.
    for method, pool in PAIRWISE_POOLS.items():
        pred_dict_valid = {m: val_preds[m] for m in pool if m in val_preds}
        pred_dict_test = {m: base_test_preds[m] for m in pool if m in base_test_preds}
        if len(pred_dict_valid) != 2:
            raise RuntimeError(f"Missing validation predictions for {method}: {pool}")
        weights, _ = sf19.search_best_weights(pred_dict_valid, y_valid, seed)
        pred_valid_fused = np.clip(sf19.apply_weights(pred_dict_valid, weights), 0, 100)
        pred_test_fused = np.clip(sf19.apply_weights(pred_dict_test, weights), 0, 100)
        mv = s31m.regression_metrics(y_valid, pred_valid_fused)

        add_test_row(
            test_rows=test_rows,
            seed=seed,
            method=method,
            model_type="validation_weighted_pairwise_fusion",
            config_name="+".join(pool),
            validation_metrics=mv,
            weights=weights,
            y_test=y_test,
            pred_test=pred_test_fused,
            n_test_groups=test_groups.nunique(),
        )
        for k, v in weights.items():
            weight_rows.append({"Seed": seed, "Method": method, "Base_Model": k, "Weight": float(v), "Best_Single_Tree": ""})
        add_prediction_rows(pred_rows, seed, method, y_test, pred_test_fused, test_groups, test_idx)

    # KRR + best single tree, where the single tree is selected only by validation RMSE.
    tree_val_rmse = {
        m: float(specs[m].validation_rmse)
        for m in ["CatBoost", "XGBoost"]
        if m in specs
    }
    best_tree = min(tree_val_rmse, key=tree_val_rmse.get)
    method = "KRR_BestSingleTree_Weighted_2Model"
    pool = ["KRR_Poly", best_tree]
    pred_dict_valid = {m: val_preds[m] for m in pool}
    pred_dict_test = {m: base_test_preds[m] for m in pool}
    weights, _ = sf19.search_best_weights(pred_dict_valid, y_valid, seed)
    pred_valid_fused = np.clip(sf19.apply_weights(pred_dict_valid, weights), 0, 100)
    pred_test_fused = np.clip(sf19.apply_weights(pred_dict_test, weights), 0, 100)
    mv = s31m.regression_metrics(y_valid, pred_valid_fused)
    weights_with_label = dict(weights)

    add_test_row(
        test_rows=test_rows,
        seed=seed,
        method=method,
        model_type="validation_selected_single_tree_pairwise_fusion",
        config_name=f"KRR_Poly+{best_tree}",
        validation_metrics=mv,
        weights=weights_with_label,
        y_test=y_test,
        pred_test=pred_test_fused,
        n_test_groups=test_groups.nunique(),
    )
    for k, v in weights.items():
        weight_rows.append({"Seed": seed, "Method": method, "Base_Model": k, "Weight": float(v), "Best_Single_Tree": best_tree})
    add_prediction_rows(pred_rows, seed, method, y_test, pred_test_fused, test_groups, test_idx)

    split_row = dict(split_info)
    split_row["Best_Single_Tree_by_validation_RMSE"] = best_tree
    split_row["CatBoost_validation_RMSE"] = tree_val_rmse.get("CatBoost", np.nan)
    split_row["XGBoost_validation_RMSE"] = tree_val_rmse.get("XGBoost", np.nan)

    return {
        "split": pd.DataFrame([split_row]),
        "candidate": candidate_df,
        "test": pd.DataFrame(test_rows),
        "prediction": pd.DataFrame(pred_rows),
        "weights": pd.DataFrame(weight_rows),
    }


# ============================================================
# 4. Figures and report
# ============================================================

def plot_metrics(summary_df: pd.DataFrame):
    data = summary_df.copy().sort_values(["Pooled_RMSE", "Pooled_MAE"], ascending=[True, True])
    x = np.arange(len(data))
    labels = data["Method"].astype(str).tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.0))
    specs = [
        ("Pooled_R2", "Pooled $R^2$", "{:.3f}", True),
        ("Pooled_RMSE", "Pooled RMSE", "{:.2f}", False),
        ("Pooled_MAE", "Pooled MAE", "{:.2f}", False),
    ]
    for ax, (col, ylabel, fmt, high_better) in zip(axes, specs):
        vals = data[col].astype(float).values
        colors = [COLORS.get(m, COLORS["gray"]) for m in labels]
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.45, alpha=0.88)
        best_idx = int(np.nanargmax(vals)) if high_better else int(np.nanargmin(vals))
        bars[best_idx].set_edgecolor(COLORS["red"])
        bars[best_idx].set_linewidth(1.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=38, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        if col == "Pooled_R2":
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_ylim(min(-0.08, np.nanmin(vals)-0.03), max(0.28, np.nanmax(vals)+0.03))
        else:
            ax.set_ylim(max(0, np.nanmin(vals)-1.2), np.nanmax(vals)+1.3)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x()+b.get_width()/2, v, fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    fig.suptitle("31M2 single-tree + KRR pairwise fusion under grouped validation", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "Current31F feature route; single tree and fusion weights selected only by validation RMSE.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0.02, 0.05, 1, 0.93])
    save_fig(fig, "123_31m2_single_tree_krr_pairwise_metrics.png")


def plot_weight_stability(weights_df: pd.DataFrame):
    if weights_df.empty:
        return
    methods = [m for m in [
        "KRR_CatBoost_Weighted_2Model",
        "KRR_XGBoost_Weighted_2Model",
        "KRR_BestSingleTree_Weighted_2Model",
    ] if m in set(weights_df["Method"])]
    if not methods:
        return
    fig, axes = plt.subplots(1, len(methods), figsize=(5.5*len(methods), 4.4), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        sub = weights_df[weights_df["Method"] == method].copy()
        for base in sorted(sub["Base_Model"].unique()):
            s = sub[sub["Base_Model"] == base].sort_values("Seed")
            ax.plot(s["Seed"], s["Weight"], marker="o", linewidth=1.6, label=base, color=COLORS.get(base, None))
            for _, r in s.iterrows():
                ax.text(r["Seed"], r["Weight"] + 0.025, f"{r['Weight']:.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_title(method)
        ax.set_xlabel("Random seed")
        ax.set_ylim(-0.05, 1.12)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        ax.legend(frameon=False, loc="best")
    axes[0].set_ylabel("Validation-optimized fusion weight")
    fig.suptitle("31M2 validation-selected pairwise fusion weight stability", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.92])
    save_fig(fig, "124_31m2_single_tree_krr_pairwise_weight_stability.png")


def plot_pooled_predictions(pred_df: pd.DataFrame, summary_df: pd.DataFrame):
    top_methods = summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE"]).head(4)["Method"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.2), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, method in zip(axes, top_methods):
        sub = pred_df[pred_df["Method"] == method].copy()
        m = s31m.regression_metrics(sub["Observed"], sub["Predicted"])
        ax.scatter(sub["Observed"], sub["Predicted"], s=16, alpha=0.65, color=COLORS.get(method, COLORS["gray"]), edgecolors="none")
        ax.plot([0, 100], [0, 100], linestyle="--", color=COLORS["red"], linewidth=1.0)
        ax.set_title(method)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.grid(color=COLORS["grid"], linewidth=0.6, alpha=0.75)
        ax.text(0.06, 0.88, f"$R^2$ = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nMAE = {m['MAE']:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=8)
        ax.set_xlabel("Observed F removal (%)")
        ax.set_ylabel("Predicted F removal (%)")
    for j in range(len(top_methods), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Top pooled predicted-vs-observed plots after 31M2 pairwise fusion", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.94])
    save_fig(fig, "125_31m2_single_tree_krr_pairwise_pooled_predicted_vs_actual.png")


def plot_residual_correlation(corr_df: pd.DataFrame, summary_df: pd.DataFrame):
    top_methods = summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE"]).head(7)["Method"].tolist()
    if corr_df.empty:
        return
    corr = corr_df.set_index("Method")
    keep = [m for m in top_methods if m in corr.index and m in corr.columns]
    if len(keep) < 2:
        return
    mat = corr.loc[keep, keep].astype(float).values
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(keep)))
    ax.set_yticks(np.arange(len(keep)))
    ax.set_xticklabels(keep, rotation=45, ha="right")
    ax.set_yticklabels(keep)
    for i in range(len(keep)):
        for j in range(len(keep)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(mat[i, j]) > 0.55 else "black")
    ax.set_title("Residual correlation among best 31M2 candidates", fontsize=13)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson residual correlation")
    fig.tight_layout()
    save_fig(fig, "126_31m2_single_tree_krr_pairwise_residual_correlation.png")


def write_decision_report(summary_df: pd.DataFrame, weights_df: pd.DataFrame, split_df: pd.DataFrame):
    path = TABLE_DIR / "31m2_single_tree_krr_pairwise_decision.txt"
    lines = []
    lines.append("=" * 100)
    lines.append("Step 31M2 single-tree + KRR_Poly pairwise fusion report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Fill the specific gap left by Step 31M: test KRR_Poly + one tree model pairwise fusions.")
    lines.append("")
    lines.append("Strict workflow reused:")
    lines.append("- Current31F_noFine_plus_31B_mechanism feature route confirmed by 31K")
    lines.append("- Step-31F balanced literature-level grouped split")
    lines.append("- Literature-group weighted training")
    lines.append("- Training-only RF feature ranking")
    lines.append("- Validation-only single-tree selection and fusion-weight search")
    lines.append("- Test labels never used for feature/model/weight selection")
    lines.append("")
    lines.append("Performance ranking:")
    show_cols = ["Method", "Pooled_R2", "Pooled_RMSE", "Pooled_MAE", "Pooled_Bias", "R2_mean", "RMSE_mean", "MAE_mean", "n_pooled"]
    lines.append(summary_df[show_cols].to_string(index=False))
    lines.append("")
    if "KRR_Poly" in set(summary_df["Method"]):
        krr = summary_df[summary_df["Method"] == "KRR_Poly"].iloc[0]
        best = summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE"]).iloc[0]
        lines.append("KRR_Poly reference:")
        lines.append(f"KRR_Poly | R2={krr['Pooled_R2']:.4f}, RMSE={krr['Pooled_RMSE']:.4f}, MAE={krr['Pooled_MAE']:.4f}")
        lines.append("Best 31M2 candidate:")
        lines.append(f"{best['Method']} | R2={best['Pooled_R2']:.4f}, RMSE={best['Pooled_RMSE']:.4f}, MAE={best['Pooled_MAE']:.4f}")
        lines.append(f"Best-vs-KRR RMSE gain = {krr['Pooled_RMSE'] - best['Pooled_RMSE']:.4f}")
        lines.append(f"Best-vs-KRR R2 gain = {best['Pooled_R2'] - krr['Pooled_R2']:.4f}")
    lines.append("")
    lines.append("Best single tree selected by validation RMSE per seed:")
    tree_cols = [c for c in ["Seed", "Best_Single_Tree_by_validation_RMSE", "CatBoost_validation_RMSE", "XGBoost_validation_RMSE"] if c in split_df.columns]
    lines.append(split_df[tree_cols].to_string(index=False))
    lines.append("")
    lines.append("Fusion-weight summary:")
    if weights_df.empty:
        lines.append("No fusion weights available.")
    else:
        wsum = weights_df.groupby(["Method", "Base_Model"], as_index=False).agg(
            Weight_mean=("Weight", "mean"),
            Weight_std=("Weight", "std"),
            n_seed=("Seed", "nunique"),
            Best_Single_Tree=("Best_Single_Tree", lambda x: ",".join(sorted(set([str(v) for v in x if str(v).strip() != ""]))))
        )
        lines.append(wsum.to_string(index=False))
    lines.append("")
    lines.append("Decision guide:")
    lines.append("1. If KRR + one tree improves over KRR_Poly, carry that pair to robustness testing.")
    lines.append("2. If no pair improves over KRR_Poly, keep KRR_Poly as the grouped-generalization main candidate.")
    lines.append("3. If pairwise fusion reduces bias but worsens RMSE/MAE, report it as calibration help but not as the final prediction route.")
    lines.append("4. Do not select any model based on test-label-derived weights; this script uses validation-only weights.")
    lines.append("=" * 100)
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[TEXT] {path}")


# ============================================================
# 5. Main
# ============================================================

def main():
    start_time = time.time()
    log("\n" + "=" * 100)
    log("Step 31M2: single-tree + KRR_Poly pairwise fusion")
    log("=" * 100)
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"SEEDS = {SEEDS}")
    log(f"K_OPTIONS = {K_OPTIONS}")
    log(f"BEST_FEATURE_SPACE = {BEST_FEATURE_SPACE}")
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
    s31f.set_balanced_split_context(X, y, groups)

    all_splits, all_candidates, all_tests, all_predictions, all_weights = [], [], [], [], []
    for seed in SEEDS:
        result = run_one_seed(X=X, y=y, groups=groups, seed=seed)
        all_splits.append(result["split"])
        all_candidates.append(result["candidate"])
        all_tests.append(result["test"])
        all_predictions.append(result["prediction"])
        all_weights.append(result["weights"])

    split_df = pd.concat(all_splits, ignore_index=True)
    cand_df = pd.concat(all_candidates, ignore_index=True)
    test_df = pd.concat(all_tests, ignore_index=True)
    pred_df = pd.concat(all_predictions, ignore_index=True)
    weights_df = pd.concat(all_weights, ignore_index=True) if any(len(x) for x in all_weights) else pd.DataFrame(columns=["Seed", "Method", "Base_Model", "Weight", "Best_Single_Tree"])

    summary_df = summarize_results(test_df, pred_df)
    corr_df = compute_residual_correlation(pred_df)

    outputs = {
        "31m2_single_tree_krr_pairwise_split_summary.csv": split_df,
        "31m2_single_tree_krr_pairwise_validation_candidates.csv": cand_df,
        "31m2_single_tree_krr_pairwise_test_results.csv": test_df,
        "31m2_single_tree_krr_pairwise_predictions.csv": pred_df,
        "31m2_single_tree_krr_pairwise_weights.csv": weights_df,
        "31m2_single_tree_krr_pairwise_summary.csv": summary_df,
        "31m2_single_tree_krr_pairwise_residual_correlation.csv": corr_df,
    }
    for filename, df in outputs.items():
        path = TABLE_DIR / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        log(f"[TABLE] {path}")

    log("\n=== Step 31M2 summary ===")
    print(summary_df.to_string(index=False))

    write_decision_report(summary_df, weights_df, split_df)

    log("Generating figures...")
    plot_metrics(summary_df)
    plot_weight_stability(weights_df)
    plot_pooled_predictions(pred_df, summary_df)
    plot_residual_correlation(corr_df, summary_df)

    elapsed = time.time() - start_time
    log("\n" + "=" * 100)
    log("[DONE] Step 31M2 single-tree + KRR_Poly pairwise fusion completed.")
    log(f"[TIME] Total runtime: {elapsed:.2f} s ({elapsed/60:.2f} min)")
    log("=" * 100)
    log("Please send these outputs:")
    log("1. outputs/tables/31m2_single_tree_krr_pairwise_summary.csv")
    log("2. outputs/tables/31m2_single_tree_krr_pairwise_decision.txt")
    log("3. outputs/tables/31m2_single_tree_krr_pairwise_weights.csv")
    log("4. outputs/tables/31m2_single_tree_krr_pairwise_residual_correlation.csv")
    log("5. outputs/figures/123_31m2_single_tree_krr_pairwise_metrics.png")
    log("6. outputs/figures/124_31m2_single_tree_krr_pairwise_weight_stability.png")
    log("7. outputs/figures/125_31m2_single_tree_krr_pairwise_pooled_predicted_vs_actual.png")
    log("8. outputs/figures/126_31m2_single_tree_krr_pairwise_residual_correlation.png")
    log("=" * 100)


if __name__ == "__main__":
    main()
