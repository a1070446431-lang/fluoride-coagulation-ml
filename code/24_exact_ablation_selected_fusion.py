# -*- coding: utf-8 -*-
"""
24_exact_ablation_selected_fusion.py

严格消融实验：基于真实 19_selected_feature_fusion.py 的最小改动版。

核心原则：
    不重新发明流程；
    不重新定义数据处理；
    不重新定义特征工程；
    不重新定义模型参数；
    不重新定义随机种子和划分逻辑；
    不重新定义 K 搜索和权重搜索逻辑。

本脚本动态导入真实的 19_selected_feature_fusion.py，然后复用其中：
    prepare_feature_sets()
    rank_features_by_rf_importance()
    select_top_features()
    fit_predict_model()
    fit_predict_model_trainfull()
    search_best_weights()
    apply_weights()
    regression_metrics()
    MODEL_FACTORIES
    SEEDS
    TEST_SIZE
    VALID_SIZE_IN_TRAIN
    K_OPTIONS
    N_RANDOM_WEIGHTS

唯一新增：
    Selected_CatBoost_HGB_Weighted
    即 CatBoost + HistGradientBoosting 的 2Model 加权融合。

主要对比：
    Selected_CatBoost
    Selected_HistGradientBoosting
    Selected_CatBoost_HGB_Weighted
    Selected_Weighted_4Model

输出：
    outputs/tables/24_exact_ablation_test_results.csv
    outputs/tables/24_exact_ablation_candidate_results.csv
    outputs/tables/24_exact_ablation_weights.csv
    outputs/tables/24_exact_ablation_predictions.csv
    outputs/tables/24_exact_ablation_summary.csv
    outputs/tables/24_exact_ablation_decision.txt

    outputs/figures/72_exact_ablation_metrics_2model_vs_4model.png/pdf
    outputs/figures/73_exact_ablation_weight_stability.png/pdf
    outputs/figures/74_exact_ablation_pooled_predicted_vs_actual.png/pdf
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# ============================================================
# 0. 导入真实 19 脚本
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

    raise FileNotFoundError(f"没有找到文件：{filename}")


def load_script19():
    current_file = Path(__file__)
    module_path = find_file_upwards(current_file, "19_selected_feature_fusion.py")

    log("=" * 100)
    log("已找到真实 19_selected_feature_fusion.py，本脚本将基于它做严格消融：")
    log(str(module_path))
    log("=" * 100)

    spec = importlib.util.spec_from_file_location(
        "selected_feature_fusion_19",
        module_path
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
# 1. 画图风格
# ============================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

COLORS = {
    "cat": "#9B72B0",
    "hgb": "#59A14F",
    "two": "#63B7B2",
    "four": "#E28E68",
    "rf": "#4C78A8",
    "extra": "#8C564B",
    "dark": "#333333",
    "pink": "#E84BA5",
    "red": "#FF0000",
    "gray": "#666666",
    "grid": "#E6E6E6",
}

METHOD_ORDER = [
    "Selected_CatBoost",
    "Selected_HistGradientBoosting",
    "Selected_CatBoost_HGB_Weighted",
    "Selected_Weighted_4Model",
]

METHOD_LABELS = {
    "Selected_CatBoost": "CatBoost",
    "Selected_HistGradientBoosting": "HGB",
    "Selected_CatBoost_HGB_Weighted": "Weighted 2Model\nCatBoost-HGB",
    "Selected_Weighted_4Model": "Weighted 4Model",
}

METHOD_COLORS = {
    "Selected_CatBoost": COLORS["cat"],
    "Selected_HistGradientBoosting": COLORS["hgb"],
    "Selected_CatBoost_HGB_Weighted": COLORS["two"],
    "Selected_Weighted_4Model": COLORS["four"],
}


# ============================================================
# 2. 基础指标与保存工具
# ============================================================

def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "MSE": mse,
    }


def save_fig(fig, filename: str):
    png_path = FIG_DIR / filename
    pdf_path = FIG_DIR / filename.replace(".png", ".pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] {png_path}")
    log(f"[FIG] {pdf_path}")


def parse_weight_text(weight_text):
    if pd.isna(weight_text) or str(weight_text).strip() == "":
        return {}
    return ast.literal_eval(str(weight_text))


def pretty_model_name(x: str) -> str:
    x = str(x)
    mapping = {
        "RandomForest": "RF",
        "CatBoost": "CatBoost",
        "HistGradientBoosting": "HGB",
        "ExtraTrees": "ExtraTrees",
    }
    return mapping.get(x, x)


# ============================================================
# 3. 严格复用 19 流程的核心评估函数
# ============================================================

def evaluate_exact_ablation(X_raw: pd.DataFrame, X_mix: pd.DataFrame, y: pd.Series):
    all_test_records = []
    all_candidate_records = []
    all_weight_records = []
    all_prediction_records = []
    all_feature_rank_records = []

    for seed in sf19.SEEDS:
        log("\n" + "=" * 100)
        log(f"严格消融 seed = {seed}")
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

        X_raw_train_full = X_raw.iloc[train_valid_idx].reset_index(drop=True)
        X_raw_test = X_raw.iloc[test_idx].reset_index(drop=True)

        X_mix_train = X_mix.iloc[train_idx].reset_index(drop=True)
        X_mix_valid = X_mix.iloc[valid_idx].reset_index(drop=True)
        X_mix_train_full = X_mix.iloc[train_valid_idx].reset_index(drop=True)
        X_mix_test = X_mix.iloc[test_idx].reset_index(drop=True)

        # ----------------------------------------------------
        # 1. 保留 19 原始 Raw compact baseline 逻辑
        # ----------------------------------------------------
        for model_name in ["RandomForest", "HistGradientBoosting", "ExtraTrees"]:
            if model_name not in sf19.MODEL_FACTORIES:
                continue

            _, pred_test = sf19.fit_predict_model_trainfull(
                model_name,
                X_raw_train_full,
                y_train_full,
                X_raw_test,
                seed,
            )

            metrics = regression_metrics(y_test, pred_test)
            method_name = f"RawCompact_{model_name}"

            all_test_records.append({
                "Seed": seed,
                "Method": method_name,
                "Selected_K": "raw",
                "Validation_RMSE": np.nan,
                **metrics,
            })

            for yt, yp in zip(y_test, pred_test):
                all_prediction_records.append({
                    "Seed": seed,
                    "Method": method_name,
                    "Observed": yt,
                    "Predicted": yp,
                })

        # ----------------------------------------------------
        # 2. 完全复用 19 的 RF 重要性排序
        # ----------------------------------------------------
        imp_df = sf19.rank_features_by_rf_importance(X_mix_train, y_train, seed)
        imp_df["Seed"] = seed
        all_feature_rank_records.append(imp_df)

        log("该 seed 前10个重要特征：")
        print(imp_df.head(10)[["Feature", "Importance"]])

        candidate_rows = []

        # ----------------------------------------------------
        # 3. 完全复用 19 的 K_OPTIONS 搜索
        # ----------------------------------------------------
        for k in sf19.K_OPTIONS:
            selected_cols = sf19.select_top_features(imp_df, X_mix, k)

            X_train_k = X_mix_train[selected_cols]
            X_valid_k = X_mix_valid[selected_cols]

            valid_preds = {}

            # 3.1 单模型验证集性能：与 19 一致
            for model_name in sf19.MODEL_FACTORIES.keys():
                _, pred_valid, _ = sf19.fit_predict_model(
                    model_name,
                    X_train_k,
                    y_train,
                    X_valid_k,
                    X_valid_k,
                    seed,
                )

                valid_preds[model_name] = pred_valid
                val_metrics = regression_metrics(y_valid, pred_valid)

                candidate_rows.append({
                    "Seed": seed,
                    "Kind": f"Selected_{model_name}",
                    "Selected_K": k,
                    "Selected_Features": ";".join(selected_cols),
                    "Validation_RMSE": val_metrics["RMSE"],
                    "Validation_R2": val_metrics["R2"],
                    "Weights": "",
                })

            # 3.2 保留 19 的 RF + CatBoost 加权融合
            if "RandomForest" in valid_preds and "CatBoost" in valid_preds:
                sub_valid = {
                    "RandomForest": valid_preds["RandomForest"],
                    "CatBoost": valid_preds["CatBoost"],
                }

                weights, val_rmse = sf19.search_best_weights(sub_valid, y_valid, seed)

                candidate_rows.append({
                    "Seed": seed,
                    "Kind": "Selected_RF_CatBoost_Weighted",
                    "Selected_K": k,
                    "Selected_Features": ";".join(selected_cols),
                    "Validation_RMSE": val_rmse,
                    "Validation_R2": np.nan,
                    "Weights": str(weights),
                })

            # 3.3 新增：CatBoost + HGB 2Model 严格消融
            if "CatBoost" in valid_preds and "HistGradientBoosting" in valid_preds:
                sub_valid = {
                    "CatBoost": valid_preds["CatBoost"],
                    "HistGradientBoosting": valid_preds["HistGradientBoosting"],
                }

                weights, val_rmse = sf19.search_best_weights(sub_valid, y_valid, seed)

                candidate_rows.append({
                    "Seed": seed,
                    "Kind": "Selected_CatBoost_HGB_Weighted",
                    "Selected_K": k,
                    "Selected_Features": ";".join(selected_cols),
                    "Validation_RMSE": val_rmse,
                    "Validation_R2": np.nan,
                    "Weights": str(weights),
                })

            # 3.4 保留 19 的四模型加权融合
            use_models = [
                m for m in ["RandomForest", "CatBoost", "HistGradientBoosting", "ExtraTrees"]
                if m in valid_preds
            ]

            if len(use_models) >= 3:
                sub_valid = {m: valid_preds[m] for m in use_models}
                weights, val_rmse = sf19.search_best_weights(sub_valid, y_valid, seed)

                candidate_rows.append({
                    "Seed": seed,
                    "Kind": "Selected_Weighted_4Model",
                    "Selected_K": k,
                    "Selected_Features": ";".join(selected_cols),
                    "Validation_RMSE": val_rmse,
                    "Validation_R2": np.nan,
                    "Weights": str(weights),
                })

        candidate_df_seed = pd.DataFrame(candidate_rows)
        all_candidate_records.append(candidate_df_seed)

        # ----------------------------------------------------
        # 4. 每种 Kind 选验证集 RMSE 最小 K，再 train+valid 重训后测试
        # ----------------------------------------------------
        for kind in candidate_df_seed["Kind"].unique():
            sub = candidate_df_seed[candidate_df_seed["Kind"] == kind].copy()
            best_row = sub.sort_values("Validation_RMSE", ascending=True).iloc[0]

            selected_k = best_row["Selected_K"]
            selected_cols = str(best_row["Selected_Features"]).split(";")
            selected_cols = [c for c in selected_cols if c in X_mix.columns]

            if not selected_cols:
                continue

            X_train_full_k = X_mix_train_full[selected_cols]
            X_test_k = X_mix_test[selected_cols]

            # 4.1 单模型：与 19 一致
            model_name = kind.replace("Selected_", "")

            if model_name in sf19.MODEL_FACTORIES:
                _, pred_test = sf19.fit_predict_model_trainfull(
                    model_name,
                    X_train_full_k,
                    y_train_full,
                    X_test_k,
                    seed,
                )

                metrics = regression_metrics(y_test, pred_test)

                all_test_records.append({
                    "Seed": seed,
                    "Method": kind,
                    "Selected_K": selected_k,
                    "Validation_RMSE": best_row["Validation_RMSE"],
                    **metrics,
                })

                for yt, yp in zip(y_test, pred_test):
                    all_prediction_records.append({
                        "Seed": seed,
                        "Method": kind,
                        "Observed": yt,
                        "Predicted": yp,
                    })

            # 4.2 RF + CatBoost：保留 19
            elif kind == "Selected_RF_CatBoost_Weighted":
                weights = parse_weight_text(best_row["Weights"])

                test_pred_dict = {}
                for m in ["RandomForest", "CatBoost"]:
                    if m not in sf19.MODEL_FACTORIES:
                        continue

                    _, pred_test_m = sf19.fit_predict_model_trainfull(
                        m,
                        X_train_full_k,
                        y_train_full,
                        X_test_k,
                        seed,
                    )
                    test_pred_dict[m] = pred_test_m

                pred_test = sf19.apply_weights(test_pred_dict, weights)
                metrics = regression_metrics(y_test, pred_test)

                all_test_records.append({
                    "Seed": seed,
                    "Method": kind,
                    "Selected_K": selected_k,
                    "Validation_RMSE": best_row["Validation_RMSE"],
                    **metrics,
                })

                for m, w in weights.items():
                    all_weight_records.append({
                        "Seed": seed,
                        "Method": kind,
                        "Model": m,
                        "Weight": w,
                        "Selected_K": selected_k,
                    })

                for yt, yp in zip(y_test, pred_test):
                    all_prediction_records.append({
                        "Seed": seed,
                        "Method": kind,
                        "Observed": yt,
                        "Predicted": yp,
                    })

            # 4.3 新增：CatBoost + HGB 2Model
            elif kind == "Selected_CatBoost_HGB_Weighted":
                weights = parse_weight_text(best_row["Weights"])

                test_pred_dict = {}
                for m in ["CatBoost", "HistGradientBoosting"]:
                    if m not in sf19.MODEL_FACTORIES:
                        continue

                    _, pred_test_m = sf19.fit_predict_model_trainfull(
                        m,
                        X_train_full_k,
                        y_train_full,
                        X_test_k,
                        seed,
                    )
                    test_pred_dict[m] = pred_test_m

                pred_test = sf19.apply_weights(test_pred_dict, weights)
                metrics = regression_metrics(y_test, pred_test)

                all_test_records.append({
                    "Seed": seed,
                    "Method": kind,
                    "Selected_K": selected_k,
                    "Validation_RMSE": best_row["Validation_RMSE"],
                    **metrics,
                })

                for m, w in weights.items():
                    all_weight_records.append({
                        "Seed": seed,
                        "Method": kind,
                        "Model": m,
                        "Weight": w,
                        "Selected_K": selected_k,
                    })

                for yt, yp in zip(y_test, pred_test):
                    all_prediction_records.append({
                        "Seed": seed,
                        "Method": kind,
                        "Observed": yt,
                        "Predicted": yp,
                    })

            # 4.4 四模型：完全保留 19
            elif kind == "Selected_Weighted_4Model":
                weights = parse_weight_text(best_row["Weights"])

                test_pred_dict = {}
                for m in weights.keys():
                    if m not in sf19.MODEL_FACTORIES:
                        continue

                    _, pred_test_m = sf19.fit_predict_model_trainfull(
                        m,
                        X_train_full_k,
                        y_train_full,
                        X_test_k,
                        seed,
                    )
                    test_pred_dict[m] = pred_test_m

                pred_test = sf19.apply_weights(test_pred_dict, weights)
                metrics = regression_metrics(y_test, pred_test)

                all_test_records.append({
                    "Seed": seed,
                    "Method": kind,
                    "Selected_K": selected_k,
                    "Validation_RMSE": best_row["Validation_RMSE"],
                    **metrics,
                })

                for m, w in weights.items():
                    all_weight_records.append({
                        "Seed": seed,
                        "Method": kind,
                        "Model": m,
                        "Weight": w,
                        "Selected_K": selected_k,
                    })

                for yt, yp in zip(y_test, pred_test):
                    all_prediction_records.append({
                        "Seed": seed,
                        "Method": kind,
                        "Observed": yt,
                        "Predicted": yp,
                    })

        log(f"seed={seed} 严格消融完成。")

    test_df = pd.DataFrame(all_test_records)
    candidate_df = pd.concat(all_candidate_records, ignore_index=True)
    weight_df = pd.DataFrame(all_weight_records)
    prediction_df = pd.DataFrame(all_prediction_records)
    feature_rank_df = pd.concat(all_feature_rank_records, ignore_index=True)

    return test_df, candidate_df, weight_df, prediction_df, feature_rank_df


# ============================================================
# 4. 汇总：包含多 seed 均值 + pooled 指标
# ============================================================

def format_selected_k(values):
    """
    Format Selected_K values safely.

    Selected_K may contain mixed types:
        integers such as 12, 18, 25, 35, 45
        strings such as 'raw' and 'all'

    Python cannot directly sort int and str together, so all values are
    converted to strings and then sorted with a custom rule.
    """
    vals = []

    for v in values:
        if pd.isna(v):
            continue

        if isinstance(v, float) and v.is_integer():
            vals.append(str(int(v)))
        else:
            vals.append(str(v))

    if len(vals) == 0:
        return ""

    def sort_key(s):
        s = str(s)

        if s.lower() == "raw":
            return (-2, 0)

        if s.lower() == "all":
            return (999999, 0)

        try:
            return (int(float(s)), 0)
        except Exception:
            return (999998, s)

    vals_unique = sorted(set(vals), key=sort_key)

    return ",".join(vals_unique)


def summarize_exact_ablation(test_df: pd.DataFrame, prediction_df: pd.DataFrame):
    summary = test_df.groupby("Method").agg(
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MSE_mean=("MSE", "mean"),
        MSE_std=("MSE", "std"),
        Mean_Selected_K=("Selected_K", format_selected_k),
        n_seed=("Seed", "nunique"),
    ).reset_index()

    pooled_rows = []

    for method, sub in prediction_df.groupby("Method"):
        metrics = regression_metrics(sub["Observed"], sub["Predicted"])

        pooled_rows.append({
            "Method": method,
            "Pooled_R2": metrics["R2"],
            "Pooled_MAE": metrics["MAE"],
            "Pooled_RMSE": metrics["RMSE"],
            "Pooled_MSE": metrics["MSE"],
            "n_pooled": len(sub),
        })

    pooled_df = pd.DataFrame(pooled_rows)

    summary = summary.merge(pooled_df, on="Method", how="left")

    summary = summary.sort_values(
        ["Pooled_R2", "Pooled_RMSE"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return summary

def write_decision_text(summary_df: pd.DataFrame):
    path = TABLE_DIR / "24_exact_ablation_decision.txt"

    lines = []
    lines.append("=" * 100)
    lines.append("24 exact ablation decision report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("This ablation is based on the real 19_selected_feature_fusion.py workflow.")
    lines.append("Only an additional CatBoost + HGB weighted 2Model was added.")
    lines.append("All other preprocessing, feature engineering, feature selection, splitting, model parameters,")
    lines.append("K search, validation-RMSE weight search, and final test evaluation follow script 19.")
    lines.append("")

    keep = summary_df[summary_df["Method"].isin(METHOD_ORDER)].copy()
    lines.append("Key comparison:")
    lines.append(keep[[
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
        "n_pooled",
    ]].to_string(index=False))
    lines.append("")

    two = summary_df[summary_df["Method"] == "Selected_CatBoost_HGB_Weighted"]
    four = summary_df[summary_df["Method"] == "Selected_Weighted_4Model"]

    if not two.empty and not four.empty:
        two = two.iloc[0]
        four = four.iloc[0]

        d_r2 = two["Pooled_R2"] - four["Pooled_R2"]
        d_rmse = two["Pooled_RMSE"] - four["Pooled_RMSE"]
        d_mae = two["Pooled_MAE"] - four["Pooled_MAE"]

        lines.append("Decision check: 2Model - 4Model")
        lines.append(f"Delta Pooled R2   = {d_r2:+.4f}")
        lines.append(f"Delta Pooled RMSE = {d_rmse:+.4f}")
        lines.append(f"Delta Pooled MAE  = {d_mae:+.4f}")
        lines.append("")

        if abs(d_r2) < 0.005 and abs(d_rmse) < 0.30 and abs(d_mae) < 0.30:
            lines.append("Suggested interpretation:")
            lines.append("The 2Model performs nearly equivalently to the 4Model. It can be considered as a parsimonious alternative,")
            lines.append("but the final decision should also consider seed-level stability and residual behavior.")
        elif d_r2 >= 0 and d_rmse <= 0:
            lines.append("Suggested interpretation:")
            lines.append("The 2Model is not worse than the 4Model under the exact 19 workflow. The final model may be simplified to CatBoost-HGB weighted fusion.")
        else:
            lines.append("Suggested interpretation:")
            lines.append("The 4Model still has measurable advantage under the exact 19 workflow. Keep the 4Model,")
            lines.append("and explain RF/ExtraTrees as candidate learners automatically down-weighted by validation-RMSE optimization.")

    lines.append("")
    lines.append("=" * 100)

    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[TEXT] {path}")


# ============================================================
# 5. 作图
# ============================================================

def plot_metrics(summary_df: pd.DataFrame):
    data = summary_df[summary_df["Method"].isin(METHOD_ORDER)].copy()
    data["Method"] = pd.Categorical(data["Method"], categories=METHOD_ORDER, ordered=True)
    data = data.sort_values("Method")

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9))

    metric_specs = [
        ("Pooled_R2", "Pooled $R^2$", "higher is better", "{:.3f}"),
        ("Pooled_RMSE", "Pooled RMSE", "lower is better", "{:.2f}"),
        ("Pooled_MAE", "Pooled MAE", "lower is better", "{:.2f}"),
    ]

    for ax, (col, title, note, fmt) in zip(axes, metric_specs):
        x = np.arange(len(data))
        vals = data[col].values
        colors = [METHOD_COLORS.get(str(m), COLORS["gray"]) for m in data["Method"].astype(str)]

        ax.bar(
            x,
            vals,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.92,
        )

        labels = [METHOD_LABELS.get(str(m), str(m)) for m in data["Method"].astype(str)]

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(title)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

        if col == "Pooled_R2":
            ax.set_ylim(0, min(1.0, max(0.9, np.nanmax(vals) * 1.10)))
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
                fontsize=8,
            )

        ax.text(
            0.98,
            0.96,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=COLORS["gray"],
        )

    fig.suptitle(
        "Exact ablation based on selected feature fusion workflow",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    fig.text(
        0.5,
        -0.02,
        "All models use the same preprocessing, feature engineering, feature selection, splitting, and validation-RMSE weight search as script 19.",
        ha="center",
        fontsize=9,
        color=COLORS["gray"],
    )

    fig.tight_layout(rect=[0.02, 0.04, 1, 0.93])
    save_fig(fig, "72_exact_ablation_metrics_2model_vs_4model.png")


def plot_weight_stability(weight_df: pd.DataFrame):
    keep_methods = [
        "Selected_CatBoost_HGB_Weighted",
        "Selected_Weighted_4Model",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.4), sharey=True)

    title_map = {
        "Selected_CatBoost_HGB_Weighted": "CatBoost-HGB 2Model weights",
        "Selected_Weighted_4Model": "RF-CatBoost-HGB-ExtraTrees 4Model weights",
    }

    color_map = {
        "RandomForest": COLORS["rf"],
        "CatBoost": COLORS["cat"],
        "HistGradientBoosting": COLORS["hgb"],
        "ExtraTrees": COLORS["extra"],
    }

    for ax, method in zip(axes, keep_methods):
        sub = weight_df[weight_df["Method"] == method].copy()

        if sub.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No weight data for {method}", ha="center", va="center")
            continue

        for model, g in sub.groupby("Model"):
            g = g.sort_values("Seed")
            ax.plot(
                g["Seed"],
                g["Weight"],
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=color_map.get(model, COLORS["gray"]),
                label=pretty_model_name(model),
            )

            for _, row in g.iterrows():
                ax.text(
                    row["Seed"],
                    row["Weight"] + 0.025,
                    f"{row['Weight']:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.set_title(title_map[method], fontsize=12, fontweight="bold")
        ax.set_xlabel("Random seed")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("Validation-optimized fusion weight")

    fig.suptitle(
        "Exact ablation of fusion weight stability",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.93])
    save_fig(fig, "73_exact_ablation_weight_stability.png")


def plot_pooled_prediction(prediction_df: pd.DataFrame):
    methods = [
        "Selected_CatBoost_HGB_Weighted",
        "Selected_Weighted_4Model",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharex=True, sharey=True)

    for ax, method in zip(axes, methods):
        sub = prediction_df[prediction_df["Method"] == method].copy()

        if sub.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No prediction data for {method}", ha="center", va="center")
            continue

        if len(sub) > 900:
            sub_plot = sub.sample(900, random_state=42)
        else:
            sub_plot = sub

        color = METHOD_COLORS.get(method, COLORS["gray"])

        ax.scatter(
            sub_plot["Observed"],
            sub_plot["Predicted"],
            s=22,
            alpha=0.62,
            color=color,
            edgecolor="white",
            linewidth=0.3,
        )

        ax.plot(
            [0, 100],
            [0, 100],
            color=COLORS["red"],
            linestyle=(0, (4, 3)),
            linewidth=1.1,
        )

        metrics = regression_metrics(sub["Observed"], sub["Predicted"])

        ax.text(
            0.06,
            0.90,
            f"$R^2$ = {metrics['R2']:.3f}\nRMSE = {metrics['RMSE']:.2f}\nMAE = {metrics['MAE']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.32",
                facecolor="white",
                edgecolor="#BBBBBB",
                alpha=0.85,
            ),
        )

        ax.set_title(METHOD_LABELS.get(method, method), fontsize=12, fontweight="bold")
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Observed F removal (%)")
        ax.set_ylabel("Predicted F removal (%)")
        ax.grid(alpha=0.15)

    fig.suptitle(
        "Pooled predicted-vs-observed comparison under exact ablation",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.93])
    save_fig(fig, "74_exact_ablation_pooled_predicted_vs_actual.png")


# ============================================================
# 6. 主函数
# ============================================================

def main():
    log("\n" + "=" * 100)
    log("24 严格消融实验开始：基于真实 19_selected_feature_fusion.py")
    log("=" * 100)
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"FIG_DIR      = {FIG_DIR}")
    log(f"TABLE_DIR    = {TABLE_DIR}")
    log(f"SEEDS        = {sf19.SEEDS}")
    log(f"K_OPTIONS    = {sf19.K_OPTIONS}")
    log(f"候选模型      = {list(sf19.MODEL_FACTORIES.keys())}")
    log("=" * 100)

    # 关键：完全复用 19 的特征准备函数
    y, X_raw, X_eng, X_mix = sf19.prepare_feature_sets()

    log("\n=== 特征集确认，来自真实 19 流程 ===")
    log(f"样本数：{len(y)}")
    log(f"Raw compact 特征数：{X_raw.shape[1]}")
    log(f"Engineered compact 特征数：{X_eng.shape[1]}")
    log(f"Raw + Engineered 混合特征数：{X_mix.shape[1]}")

    test_df, candidate_df, weight_df, prediction_df, feature_rank_df = evaluate_exact_ablation(
        X_raw=X_raw,
        X_mix=X_mix,
        y=y,
    )

    summary_df = summarize_exact_ablation(test_df, prediction_df)

    # 保存表格
    test_path = TABLE_DIR / "24_exact_ablation_test_results.csv"
    cand_path = TABLE_DIR / "24_exact_ablation_candidate_results.csv"
    weight_path = TABLE_DIR / "24_exact_ablation_weights.csv"
    pred_path = TABLE_DIR / "24_exact_ablation_predictions.csv"
    feat_path = TABLE_DIR / "24_exact_ablation_feature_ranking_by_seed.csv"
    summary_path = TABLE_DIR / "24_exact_ablation_summary.csv"

    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    candidate_df.to_csv(cand_path, index=False, encoding="utf-8-sig")
    weight_df.to_csv(weight_path, index=False, encoding="utf-8-sig")
    prediction_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    feature_rank_df.to_csv(feat_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    log("\n=== 24 严格消融性能汇总 ===")
    print(summary_df.to_string(index=False))

    write_decision_text(summary_df)

    log("\n=== 生成严格消融图 72–74 ===")
    plot_metrics(summary_df)
    plot_weight_stability(weight_df)
    plot_pooled_prediction(prediction_df)

    log("\n输出表格：")
    for p in [
        test_path,
        cand_path,
        weight_path,
        pred_path,
        feat_path,
        summary_path,
        TABLE_DIR / "24_exact_ablation_decision.txt",
    ]:
        log(f"  - {p}")

    log("\n输出图片：")
    for f in [
        "72_exact_ablation_metrics_2model_vs_4model.png",
        "73_exact_ablation_weight_stability.png",
        "74_exact_ablation_pooled_predicted_vs_actual.png",
    ]:
        log(f"  - {FIG_DIR / f}")

    log("\n" + "=" * 100)
    log("24 严格消融实验完成。")
    log("=" * 100)


if __name__ == "__main__":
    main()