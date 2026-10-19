# -*- coding: utf-8 -*-
"""
31m_hybrid_krr_tree_fusion_oof_stacking.py

Purpose
-------
Hybrid fusion and grouped OOF stacking after Step 31L showed that
KRR_Poly（多项式核岭回归）outperformed the current CB-XGB tree baseline under
balanced literature-level grouped validation.

This script answers two strict questions:
1. Can KRR_Poly be fused with the CatBoost-XGBoost route to further improve
   all-domain grouped generalization?
2. Does grouped OOF stacking with simple regularized meta-models provide extra
   benefit beyond validation-RMSE weighted fusion?

It is NOT a simplified workflow. It reuses the real project scripts:
- Step 31K: confirmed grouped-generalization feature route
  Current31F_noFine_plus_31B_mechanism.
- Step 31F: balanced literature-level grouped split and literature-group
  sample weights.
- Step 31G: CatBoost / XGBoost default wrappers and validation-RMSE fusion.
- Step 31L: KRR / GPR / PLS non-tree model implementations and configs.
- script 19: feature engineering, preprocessing, and feature-selection tools.

Strict rules
------------
1. 文献编号 is used only for literature-level grouped splitting and weighting.
2. 文献编号 is never used as an input feature.
3. Test labels are never used for feature selection, model/config selection,
   fusion-weight search, or meta-model selection.
4. Feature ranking is performed only on grouped training data.
5. Base-model configs are selected only by validation RMSE.
6. Fusion weights are selected only by validation RMSE.
7. OOF stacking meta-models are trained only on grouped OOF predictions from
   train+valid groups, and evaluated only on held-out test literature groups.
8. Meta-models are intentionally simple: Ridge（岭回归） and ElasticNet（弹性网络）.
9. Total runtime is printed at the end.

Candidate base models
---------------------
- CatBoost（类别梯度提升）
- XGBoost（极端梯度提升）
- CB-XGB weighted 2Model（CatBoost-XGBoost 加权二模型）
- KRR_Poly（多项式核岭回归）
- GPR_Matern（Matern核高斯过程）

Candidate fusion / stacking methods
-----------------------------------
- KRR_Poly single model
- GPR_Matern single model
- CB-XGB weighted 2Model baseline
- KRR + CB-XGB weighted 2Model
- KRR + CatBoost + XGBoost weighted 3Model
- KRR + GPR + CB-XGB weighted 3Model
- KRR + GPR + CatBoost + XGBoost weighted 4Model
- OOF_Ridge over CatBoost/XGBoost/KRR/GPR predictions
- OOF_ElasticNet over CatBoost/XGBoost/KRR/GPR predictions

Outputs
-------
Tables:
    outputs/tables/31m_hybrid_fusion_split_summary.csv
    outputs/tables/31m_hybrid_fusion_validation_candidates.csv
    outputs/tables/31m_hybrid_fusion_test_results.csv
    outputs/tables/31m_hybrid_fusion_predictions.csv
    outputs/tables/31m_hybrid_fusion_weights.csv
    outputs/tables/31m_hybrid_fusion_oof_meta_results.csv
    outputs/tables/31m_hybrid_fusion_summary.csv
    outputs/tables/31m_hybrid_fusion_residual_correlation.csv
    outputs/tables/31m_hybrid_fusion_decision.txt

Figures:
    outputs/figures/119_31m_hybrid_fusion_metrics.png/pdf
    outputs/figures/120_31m_hybrid_fusion_weight_stability.png/pdf
    outputs/figures/121_31m_hybrid_fusion_pooled_predicted_vs_actual.png/pdf
    outputs/figures/122_31m_hybrid_fusion_residual_correlation.png/pdf
"""

from __future__ import annotations

import math
import time
import warnings
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    log(f"Loading real project script: {filename}")
    log(str(module_path))
    log("=" * 100)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    # Important for Python 3.12 + @dataclass:
    # dataclass internally looks up sys.modules[cls.__module__].
    # If an imported script is executed before being registered in sys.modules,
    # it can raise: AttributeError: 'NoneType' object has no attribute '__dict__'.
    sys.modules[module_name] = module

    spec.loader.exec_module(module)
    return module


# Reuse Step 31L; it imports real 31K/31G/31F/script-19.
s31l = load_module_from_file("31l_non_tree_model_screening.py", "s31l_real")
s31k = s31l.s31k
s31g = s31l.s31g
s31f = s31l.s31f
sf19 = s31l.sf19

PROJECT_ROOT = Path(s31l.PROJECT_ROOT)
FIG_DIR = Path(s31l.FIG_DIR)
TABLE_DIR = Path(s31l.TABLE_DIR)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Settings
# ============================================================

SEEDS = list(getattr(sf19, "SEEDS", [7, 21, 42, 66, 88]))
K_OPTIONS = [12, 18, 25, 35, 45, 60, "all"]
BEST_FEATURE_SPACE = "Current31F_noFine_plus_31B_mechanism"

BASE_TREE_MODELS = ["CatBoost", "XGBoost"]
BASE_SPEC_NAMES = ["CatBoost", "XGBoost", "CB-XGB", "KRR_Poly", "GPR_Matern"]

FUSION_POOLS = {
    "KRR_CBXGB_Weighted_2Model": ["KRR_Poly", "CB-XGB"],
    "KRR_CatBoost_XGBoost_Weighted_3Model": ["KRR_Poly", "CatBoost", "XGBoost"],
    "KRR_GPR_CBXGB_Weighted_3Model": ["KRR_Poly", "GPR_Matern", "CB-XGB"],
    "KRR_GPR_CatBoost_XGBoost_Weighted_4Model": ["KRR_Poly", "GPR_Matern", "CatBoost", "XGBoost"],
}

METHOD_ORDER = [
    "KRR_Poly",
    "CB-XGB",
    "GPR_Matern",
    "KRR_CBXGB_Weighted_2Model",
    "KRR_CatBoost_XGBoost_Weighted_3Model",
    "KRR_GPR_CBXGB_Weighted_3Model",
    "KRR_GPR_CatBoost_XGBoost_Weighted_4Model",
    "OOF_Ridge_BasePreds",
    "OOF_ElasticNet_BasePreds",
    "CatBoost",
    "XGBoost",
]

COLORS = {
    "CatBoost": "#9B72B0",
    "XGBoost": "#D28C3D",
    "CB-XGB": "#808080",
    "KRR_Poly": "#76B083",
    "GPR_Matern": "#B06D84",
    "KRR_CBXGB_Weighted_2Model": "#67A9CF",
    "KRR_CatBoost_XGBoost_Weighted_3Model": "#B2ABD2",
    "KRR_GPR_CBXGB_Weighted_3Model": "#FDB863",
    "KRR_GPR_CatBoost_XGBoost_Weighted_4Model": "#A6D854",
    "OOF_Ridge_BasePreds": "#5E81AC",
    "OOF_ElasticNet_BasePreds": "#BF616A",
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

@dataclass
class SelectedSpec:
    name: str
    model_family: str
    config_name: str
    config_obj: object | None
    selected_k: object
    selected_cols: List[str]
    validation_rmse: float
    validation_mae: float
    validation_r2: float
    weights: Dict[str, float] | None = None


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


def select_columns_from_importance(imp_df: pd.DataFrame, X: pd.DataFrame, selected_k):
    cols = sf19.select_top_features(imp_df, X, selected_k)
    return [c for c in cols if c in X.columns and not s31k.is_bad_feature_name(c)]


def sorted_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    out = df.copy()
    out["_order"] = out["Method"].map(order_map).fillna(999)
    out = out.sort_values(["Pooled_RMSE", "Pooled_MAE", "_order"], ascending=[True, True, True]).drop(columns=["_order"])
    return out.reset_index(drop=True)


def build_meta_features(pred_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Meta-features based only on base predictions and disagreements."""
    keys = [k for k in ["CatBoost", "XGBoost", "KRR_Poly", "GPR_Matern"] if k in pred_dict]
    arr = np.column_stack([np.asarray(pred_dict[k], dtype=float).ravel() for k in keys])
    df = pd.DataFrame(arr, columns=[f"pred_{k}" for k in keys])
    df["pred_mean"] = np.mean(arr, axis=1)
    df["pred_std"] = np.std(arr, axis=1)
    df["pred_range"] = np.max(arr, axis=1) - np.min(arr, axis=1)
    if "CatBoost" in pred_dict and "XGBoost" in pred_dict:
        df["abs_CatBoost_minus_XGBoost"] = np.abs(np.asarray(pred_dict["CatBoost"]) - np.asarray(pred_dict["XGBoost"]))
    if "KRR_Poly" in pred_dict and "CatBoost" in pred_dict:
        df["abs_KRR_minus_CatBoost"] = np.abs(np.asarray(pred_dict["KRR_Poly"]) - np.asarray(pred_dict["CatBoost"]))
    if "KRR_Poly" in pred_dict and "XGBoost" in pred_dict:
        df["abs_KRR_minus_XGBoost"] = np.abs(np.asarray(pred_dict["KRR_Poly"]) - np.asarray(pred_dict["XGBoost"]))
    if "KRR_Poly" in pred_dict and "GPR_Matern" in pred_dict:
        df["abs_KRR_minus_GPR"] = np.abs(np.asarray(pred_dict["KRR_Poly"]) - np.asarray(pred_dict["GPR_Matern"]))
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ============================================================
# 3. Base model selection and prediction
# ============================================================


def fit_predict_tree(model_name, X_train, y_train, X_target, seed, sample_weight=None):
    return s31k.fit_predict_base(model_name, X_train, y_train, X_target, seed, sample_weight)


def fit_predict_selected_spec(
    spec: SelectedSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_target: pd.DataFrame,
    seed: int,
    sample_weight=None,
    train_groups: pd.Series | None = None,
) -> np.ndarray:
    """Fit selected base/fusion spec and predict target rows."""
    if spec.name in {"CatBoost", "XGBoost"}:
        return np.clip(fit_predict_tree(
            spec.name,
            X_train[spec.selected_cols].reset_index(drop=True),
            y_train.reset_index(drop=True),
            X_target[spec.selected_cols].reset_index(drop=True),
            seed,
            sample_weight,
        ), 0, 100)

    if spec.name in {"KRR_Poly", "GPR_Matern"}:
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

    if spec.name == "CB-XGB":
        weights = spec.weights or {"CatBoost": 0.5, "XGBoost": 0.5}
        pred_dict = {}
        for model_name in ["CatBoost", "XGBoost"]:
            pred_dict[model_name] = np.clip(fit_predict_tree(
                model_name,
                X_train[spec.selected_cols].reset_index(drop=True),
                y_train.reset_index(drop=True),
                X_target[spec.selected_cols].reset_index(drop=True),
                seed,
                sample_weight,
            ), 0, 100)
        return np.clip(sf19.apply_weights(pred_dict, weights), 0, 100)

    raise ValueError(f"Unknown SelectedSpec name: {spec.name}")


def select_tree_and_cbxgb_specs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    imp_df: pd.DataFrame,
    seed: int,
    sample_weight_train,
) -> Tuple[Dict[str, SelectedSpec], Dict[str, np.ndarray], pd.DataFrame]:
    rows = []
    pred_store = {}

    for selected_k in K_OPTIONS:
        selected_cols = select_columns_from_importance(imp_df, X_train, selected_k)
        if len(selected_cols) < 3:
            continue
        key = ("K", str(selected_k), tuple(selected_cols))
        log(f"  [Tree] seed={seed} | K={selected_k} | n_features={len(selected_cols)}")
        Xtr = X_train[selected_cols].reset_index(drop=True)
        Xva = X_valid[selected_cols].reset_index(drop=True)
        valid_preds = {}
        for model_name in ["CatBoost", "XGBoost"]:
            pred = np.clip(fit_predict_tree(model_name, Xtr, y_train, Xva, seed, sample_weight_train), 0, 100)
            valid_preds[model_name] = pred
            pred_store[(model_name, selected_k, tuple(selected_cols))] = pred
            m = regression_metrics(y_valid, pred)
            rows.append({
                "Seed": seed,
                "Candidate_Group": "Tree",
                "Method": model_name,
                "Selected_K": selected_k,
                "n_Selected_Features": len(selected_cols),
                "Selected_Features": ";".join(selected_cols),
                "Config_Name": "31G_default",
                "Validation_R2": m["R2"],
                "Validation_RMSE": m["RMSE"],
                "Validation_MAE": m["MAE"],
                "Validation_MSE": m["MSE"],
                "Validation_Bias": m["Bias"],
                "Weights": "",
            })

        weights, _ = sf19.search_best_weights(valid_preds, y_valid, seed)
        fused = np.clip(sf19.apply_weights(valid_preds, weights), 0, 100)
        pred_store[("CB-XGB", selected_k, tuple(selected_cols))] = fused
        m = regression_metrics(y_valid, fused)
        rows.append({
            "Seed": seed,
            "Candidate_Group": "TreeFusion",
            "Method": "CB-XGB",
            "Selected_K": selected_k,
            "n_Selected_Features": len(selected_cols),
            "Selected_Features": ";".join(selected_cols),
            "Config_Name": "31G_validation_weighted",
            "Validation_R2": m["R2"],
            "Validation_RMSE": m["RMSE"],
            "Validation_MAE": m["MAE"],
            "Validation_MSE": m["MSE"],
            "Validation_Bias": m["Bias"],
            "Weights": weight_dict_to_text(weights),
        })

    cand = pd.DataFrame(rows)
    specs, val_preds = {}, {}
    for method in ["CatBoost", "XGBoost", "CB-XGB"]:
        sub = cand[cand["Method"] == method].copy()
        if sub.empty:
            continue
        best = sub.sort_values(["Validation_RMSE", "Validation_MAE", "Validation_R2"], ascending=[True, True, False]).iloc[0]
        selected_cols = [c for c in str(best["Selected_Features"]).split(";") if c in X_train.columns]
        selected_k = best["Selected_K"]
        weights = parse_weight_text(best["Weights"])
        specs[method] = SelectedSpec(
            name=method,
            model_family=method,
            config_name=str(best["Config_Name"]),
            config_obj=None,
            selected_k=selected_k,
            selected_cols=selected_cols,
            validation_rmse=float(best["Validation_RMSE"]),
            validation_mae=float(best["Validation_MAE"]),
            validation_r2=float(best["Validation_R2"]),
            weights=weights if method == "CB-XGB" else None,
        )
        val_preds[method] = pred_store[(method, selected_k, tuple(selected_cols))]
    return specs, val_preds, cand


def select_non_tree_spec(
    family: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    imp_df: pd.DataFrame,
    seed: int,
    sample_weight_train,
    train_groups: pd.Series,
) -> Tuple[SelectedSpec, np.ndarray, pd.DataFrame]:
    configs = [c for c in s31l.make_model_configs() if c.model_family == family]
    if not configs:
        raise RuntimeError(f"No non-tree configs found for family={family}")

    rows = []
    pred_store = {}
    valid_seen = set()

    for selected_k in K_OPTIONS:
        selected_cols = select_columns_from_importance(imp_df, X_train, selected_k)
        if len(selected_cols) < 3:
            continue
        key_cols = tuple(selected_cols)
        if key_cols in valid_seen and selected_k != "all":
            continue
        valid_seen.add(key_cols)

        for cfg in configs:
            if cfg.allowed_k is not None and selected_k not in cfg.allowed_k:
                continue
            # Guard exact GPR to compact K only through 31L config itself.
            log(f"  [NonTree] seed={seed} | {family} | {cfg.config_name} | K={selected_k} | n_features={len(selected_cols)}")
            try:
                pred, fit_meta = s31l.fit_predict_non_tree(
                    cfg=cfg,
                    X_train=X_train[selected_cols].reset_index(drop=True),
                    y_train=y_train.reset_index(drop=True),
                    X_target=X_valid[selected_cols].reset_index(drop=True),
                    seed=seed,
                    sample_weight=sample_weight_train,
                    train_groups=train_groups.reset_index(drop=True),
                )
                pred = np.clip(pred, 0, 100)
                m = regression_metrics(y_valid, pred)
                status, err = "ok", ""
                pred_store[(cfg.config_name, selected_k, key_cols)] = pred
            except Exception as e:
                m = {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "Bias": np.nan}
                status, err = "failed", repr(e)
                fit_meta = {}
                log(f"    [WARN] {family} failed: {err}")

            rows.append({
                "Seed": seed,
                "Candidate_Group": "NonTree",
                "Method": family,
                "Selected_K": selected_k,
                "n_Selected_Features": len(selected_cols),
                "Selected_Features": ";".join(selected_cols),
                "Config_Name": cfg.config_name,
                "Validation_R2": m["R2"],
                "Validation_RMSE": m["RMSE"],
                "Validation_MAE": m["MAE"],
                "Validation_MSE": m["MSE"],
                "Validation_Bias": m["Bias"],
                "Weights": "",
                "Status": status,
                "Error": err,
                "Fit_Meta": str(fit_meta),
            })

    cand = pd.DataFrame(rows)
    ok = cand[cand.get("Status", "ok") == "ok"].copy()
    ok = ok.dropna(subset=["Validation_RMSE"])
    if ok.empty:
        raise RuntimeError(f"No valid non-tree candidate for {family}, seed={seed}")
    best = ok.sort_values(["Validation_RMSE", "Validation_MAE", "Validation_R2"], ascending=[True, True, False]).iloc[0]
    cfg = next(c for c in configs if c.config_name == best["Config_Name"])
    selected_cols = [c for c in str(best["Selected_Features"]).split(";") if c in X_train.columns]
    selected_k = best["Selected_K"]
    spec = SelectedSpec(
        name=family,
        model_family=family,
        config_name=str(best["Config_Name"]),
        config_obj=cfg,
        selected_k=selected_k,
        selected_cols=selected_cols,
        validation_rmse=float(best["Validation_RMSE"]),
        validation_mae=float(best["Validation_MAE"]),
        validation_r2=float(best["Validation_R2"]),
        weights=None,
    )
    val_pred = pred_store[(spec.config_name, selected_k, tuple(selected_cols))]
    return spec, val_pred, cand


# ============================================================
# 4. OOF stacking utilities
# ============================================================


def choose_grouped_meta_model(X_meta: pd.DataFrame, y: pd.Series, groups: pd.Series, kind: str) -> Tuple[Pipeline, dict, pd.DataFrame]:
    """Select a simple meta-model config using grouped CV on OOF meta-features."""
    Xv = X_meta.values.astype(float)
    yv = np.asarray(y, dtype=float)
    groups = pd.Series(groups).astype(str).reset_index(drop=True)
    unique_groups = groups.nunique()
    n_splits = min(5, unique_groups)
    if n_splits < 2:
        raise RuntimeError("Not enough groups for grouped meta-model selection")
    gkf = GroupKFold(n_splits=n_splits)

    candidates = []
    if kind == "ridge":
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]:
            candidates.append({"kind": "ridge", "alpha": alpha})
    elif kind == "elasticnet":
        for alpha in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
            for l1_ratio in [0.05, 0.1, 0.3, 0.5, 0.8]:
                candidates.append({"kind": "elasticnet", "alpha": alpha, "l1_ratio": l1_ratio})
    else:
        raise ValueError(kind)

    rows = []
    for cfg in candidates:
        fold_rmses = []
        for tr, va in gkf.split(Xv, yv, groups=groups):
            if cfg["kind"] == "ridge":
                model = Ridge(alpha=cfg["alpha"], random_state=42)
            else:
                model = ElasticNet(alpha=cfg["alpha"], l1_ratio=cfg["l1_ratio"], max_iter=50000, random_state=42)
            pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
            pipe.fit(Xv[tr], yv[tr])
            pred = np.clip(pipe.predict(Xv[va]), 0, 100)
            fold_rmses.append(regression_metrics(yv[va], pred)["RMSE"])
        rows.append({**cfg, "GroupedCV_RMSE": float(np.mean(fold_rmses)), "GroupedCV_RMSE_std": float(np.std(fold_rmses))})

    score_df = pd.DataFrame(rows).sort_values(["GroupedCV_RMSE", "GroupedCV_RMSE_std"], ascending=[True, True]).reset_index(drop=True)
    best = score_df.iloc[0].to_dict()
    if best["kind"] == "ridge":
        final_model = Ridge(alpha=float(best["alpha"]), random_state=42)
    else:
        final_model = ElasticNet(alpha=float(best["alpha"]), l1_ratio=float(best["l1_ratio"]), max_iter=50000, random_state=42)
    final_pipe = Pipeline([("scaler", StandardScaler()), ("model", final_model)])
    final_pipe.fit(Xv, yv)
    return final_pipe, best, score_df


def build_oof_base_predictions(
    specs: Dict[str, SelectedSpec],
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    groups_train_full: pd.Series,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    base_names = ["CatBoost", "XGBoost", "KRR_Poly", "GPR_Matern"]
    n = len(y_train_full)
    oof = {b: np.full(n, np.nan, dtype=float) for b in base_names if b in specs}
    groups_train_full = groups_train_full.reset_index(drop=True).astype(str)
    unique_groups = groups_train_full.nunique()
    n_splits = min(5, unique_groups)
    if n_splits < 2:
        raise RuntimeError("Not enough literature groups for grouped OOF stacking.")

    gkf = GroupKFold(n_splits=n_splits)
    audit_rows = []
    fold_id = 0
    for inner_tr, inner_va in gkf.split(X_train_full, y_train_full, groups=groups_train_full):
        fold_id += 1
        X_inner_tr = X_train_full.iloc[inner_tr].reset_index(drop=True)
        y_inner_tr = y_train_full.iloc[inner_tr].reset_index(drop=True)
        g_inner_tr = groups_train_full.iloc[inner_tr].reset_index(drop=True)
        X_inner_va = X_train_full.iloc[inner_va].reset_index(drop=True)
        g_inner_va = groups_train_full.iloc[inner_va].reset_index(drop=True)
        sw_inner = s31f.make_literature_group_weights(g_inner_tr)

        log(f"    [OOF] seed={seed} | fold={fold_id}/{n_splits} | train_groups={g_inner_tr.nunique()} | holdout_groups={g_inner_va.nunique()}")
        for base in list(oof.keys()):
            pred = fit_predict_selected_spec(
                spec=specs[base],
                X_train=X_inner_tr,
                y_train=y_inner_tr,
                X_target=X_inner_va,
                seed=seed + 1000 + fold_id,
                sample_weight=sw_inner,
                train_groups=g_inner_tr,
            )
            oof[base][inner_va] = pred
            audit_rows.append({
                "Seed": seed,
                "OOF_Fold": fold_id,
                "Base_Model": base,
                "n_inner_train": len(inner_tr),
                "n_inner_valid": len(inner_va),
                "n_inner_train_groups": g_inner_tr.nunique(),
                "n_inner_valid_groups": g_inner_va.nunique(),
            })

    # Fill rare missing values with each base model's OOF median; this should usually not occur.
    for base, arr in oof.items():
        if np.isnan(arr).any():
            med = np.nanmedian(arr)
            arr[np.isnan(arr)] = med
            oof[base] = arr
    return oof, pd.DataFrame(audit_rows)


# ============================================================
# 5. One seed run
# ============================================================


def run_one_seed(X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int):
    log("\n" + "=" * 100)
    log(f"31M hybrid KRR-tree fusion + grouped OOF stacking | seed={seed}")
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

    # Feature ranking only on training groups.
    imp_df = s31f.rank_features_by_rf_importance_weighted(
        X_train, y_train, seed, sample_weight=sample_weight_train
    )

    all_candidate_rows = []
    specs = {}
    val_preds = {}

    tree_specs, tree_val_preds, tree_cand = select_tree_and_cbxgb_specs(
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
    all_candidate_rows.append(tree_cand)

    for family in ["KRR_Poly", "GPR_Matern"]:
        spec, pred_valid, cand = select_non_tree_spec(
            family=family,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            imp_df=imp_df,
            seed=seed,
            sample_weight_train=sample_weight_train,
            train_groups=train_groups,
        )
        specs[family] = spec
        val_preds[family] = pred_valid
        all_candidate_rows.append(cand)

    candidate_df = pd.concat(all_candidate_rows, ignore_index=True)

    # Final test base predictions after retraining on train+valid groups.
    base_test_preds = {}
    test_rows = []
    pred_rows = []
    weight_rows = []

    for base in ["CatBoost", "XGBoost", "CB-XGB", "KRR_Poly", "GPR_Matern"]:
        if base not in specs:
            continue
        pred = fit_predict_selected_spec(
            spec=specs[base],
            X_train=X_train_full,
            y_train=y_train_full,
            X_target=X_test,
            seed=seed,
            sample_weight=sample_weight_train_full,
            train_groups=train_full_groups,
        )
        base_test_preds[base] = pred
        m = regression_metrics(y_test, pred)
        spec = specs[base]
        test_rows.append({
            "Seed": seed,
            "Method": base,
            "Model_Type": "base_or_nested_base",
            "Selected_K": spec.selected_k,
            "n_Selected_Features": len(spec.selected_cols),
            "Selected_Features": ";".join(spec.selected_cols),
            "Config_Name": spec.config_name,
            "Validation_R2": spec.validation_r2,
            "Validation_RMSE": spec.validation_rmse,
            "Validation_MAE": spec.validation_mae,
            "Weights": weight_dict_to_text(spec.weights or {}),
            "Test_R2": m["R2"],
            "Test_RMSE": m["RMSE"],
            "Test_MAE": m["MAE"],
            "Test_MSE": m["MSE"],
            "Test_Bias": m["Bias"],
            "n_test": len(y_test),
            "n_test_groups": test_groups.nunique(),
        })
        for local_i, (obs, pp, gid, data_idx) in enumerate(zip(y_test, pred, test_groups, test_idx)):
            pred_rows.append({
                "Seed": seed,
                "Method": base,
                "Data_Index": int(data_idx),
                "Local_Test_Index": int(local_i),
                "文献编号": gid,
                "Observed": float(obs),
                "Predicted": float(pp),
                "Residual": float(obs - pp),
            })

    # Validation-selected weighted fusions.
    for method, pool in FUSION_POOLS.items():
        pred_dict_valid = {m: val_preds[m] for m in pool if m in val_preds}
        pred_dict_test = {m: base_test_preds[m] for m in pool if m in base_test_preds}
        if len(pred_dict_valid) < 2:
            continue
        weights, _ = sf19.search_best_weights(pred_dict_valid, y_valid, seed)
        pred_valid_fused = np.clip(sf19.apply_weights(pred_dict_valid, weights), 0, 100)
        pred_test_fused = np.clip(sf19.apply_weights(pred_dict_test, weights), 0, 100)
        mv = regression_metrics(y_valid, pred_valid_fused)
        mt = regression_metrics(y_test, pred_test_fused)
        test_rows.append({
            "Seed": seed,
            "Method": method,
            "Model_Type": "validation_weighted_fusion",
            "Selected_K": "mixed",
            "n_Selected_Features": np.nan,
            "Selected_Features": "mixed_base_specific",
            "Config_Name": "+".join(pool),
            "Validation_R2": mv["R2"],
            "Validation_RMSE": mv["RMSE"],
            "Validation_MAE": mv["MAE"],
            "Weights": weight_dict_to_text(weights),
            "Test_R2": mt["R2"],
            "Test_RMSE": mt["RMSE"],
            "Test_MAE": mt["MAE"],
            "Test_MSE": mt["MSE"],
            "Test_Bias": mt["Bias"],
            "n_test": len(y_test),
            "n_test_groups": test_groups.nunique(),
        })
        for k, v in weights.items():
            weight_rows.append({"Seed": seed, "Method": method, "Base_Model": k, "Weight": float(v)})
        for local_i, (obs, pp, gid, data_idx) in enumerate(zip(y_test, pred_test_fused, test_groups, test_idx)):
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

    # Grouped OOF stacking on train+valid groups.
    meta_rows = []
    try:
        oof_preds, oof_audit = build_oof_base_predictions(specs, X_train_full, y_train_full, train_full_groups, seed)
        test_meta_base = {b: base_test_preds[b] for b in oof_preds.keys()}
        X_meta_oof = build_meta_features(oof_preds)
        X_meta_test = build_meta_features(test_meta_base)

        for kind, method_name in [("ridge", "OOF_Ridge_BasePreds"), ("elasticnet", "OOF_ElasticNet_BasePreds")]:
            meta_model, best_cfg, score_df = choose_grouped_meta_model(X_meta_oof, y_train_full, train_full_groups, kind)
            pred_test_meta = np.clip(meta_model.predict(X_meta_test.values.astype(float)), 0, 100)
            mt = regression_metrics(y_test, pred_test_meta)
            test_rows.append({
                "Seed": seed,
                "Method": method_name,
                "Model_Type": "grouped_oof_stacking",
                "Selected_K": "base_specific",
                "n_Selected_Features": X_meta_oof.shape[1],
                "Selected_Features": ";".join(list(X_meta_oof.columns)),
                "Config_Name": str(best_cfg),
                "Validation_R2": np.nan,
                "Validation_RMSE": float(best_cfg.get("GroupedCV_RMSE", np.nan)),
                "Validation_MAE": np.nan,
                "Weights": "",
                "Test_R2": mt["R2"],
                "Test_RMSE": mt["RMSE"],
                "Test_MAE": mt["MAE"],
                "Test_MSE": mt["MSE"],
                "Test_Bias": mt["Bias"],
                "n_test": len(y_test),
                "n_test_groups": test_groups.nunique(),
            })
            for local_i, (obs, pp, gid, data_idx) in enumerate(zip(y_test, pred_test_meta, test_groups, test_idx)):
                pred_rows.append({
                    "Seed": seed,
                    "Method": method_name,
                    "Data_Index": int(data_idx),
                    "Local_Test_Index": int(local_i),
                    "文献编号": gid,
                    "Observed": float(obs),
                    "Predicted": float(pp),
                    "Residual": float(obs - pp),
                })
            tmp = score_df.copy()
            tmp["Seed"] = seed
            tmp["Stacking_Method"] = method_name
            meta_rows.append(tmp)
        meta_audit_df = pd.concat(meta_rows, ignore_index=True) if meta_rows else pd.DataFrame()
        oof_audit["Seed"] = seed
    except Exception as e:
        log(f"[WARN] grouped OOF stacking failed for seed={seed}: {repr(e)}")
        meta_audit_df = pd.DataFrame([{"Seed": seed, "Error": repr(e)}])
        oof_audit = pd.DataFrame([{"Seed": seed, "Error": repr(e)}])

    split_info = dict(split_info)
    split_info["Feature_Space"] = BEST_FEATURE_SPACE
    split_info["n_features_in_space"] = int(X.shape[1])

    return {
        "split": pd.DataFrame([split_info]),
        "candidate": candidate_df,
        "test": pd.DataFrame(test_rows),
        "prediction": pd.DataFrame(pred_rows),
        "weights": pd.DataFrame(weight_rows),
        "meta": meta_audit_df,
        "oof_audit": oof_audit,
    }


# ============================================================
# 6. Summary, figures, report
# ============================================================


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
        m = regression_metrics(sub["Observed"], sub["Predicted"])
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


def plot_metrics(summary_df: pd.DataFrame):
    data = summary_df.copy().sort_values(["Pooled_RMSE", "Pooled_MAE"], ascending=[True, True])
    # Keep figure readable while ensuring the best methods are shown.
    data = data.head(10).copy()
    x = np.arange(len(data))
    labels = data["Method"].astype(str).tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    specs = [("Pooled_R2", "Pooled $R^2$", "{:.3f}", True), ("Pooled_RMSE", "Pooled RMSE", "{:.2f}", False), ("Pooled_MAE", "Pooled MAE", "{:.2f}", False)]
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
            ax.set_ylim(min(-0.05, np.nanmin(vals)-0.03), max(0.28, np.nanmax(vals)+0.03))
        else:
            ax.set_ylim(max(0, np.nanmin(vals)-1.5), np.nanmax(vals)+1.5)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x()+b.get_width()/2, v, fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    fig.suptitle("31M hybrid KRR-tree fusion and grouped OOF stacking", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "Current31F feature route; base models/configs selected by validation RMSE; meta-models selected by grouped OOF CV.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0.02, 0.05, 1, 0.93])
    save_fig(fig, "119_31m_hybrid_fusion_metrics.png")


def plot_weight_stability(weights_df: pd.DataFrame):
    if weights_df.empty:
        return
    methods = [m for m in FUSION_POOLS.keys() if m in set(weights_df["Method"])]
    if not methods:
        return
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5.5*n, 4.5), sharey=True)
    if n == 1:
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
        ax.legend(frameon=False)
    axes[0].set_ylabel("Validation-optimized fusion weight")
    fig.suptitle("31M validation-selected hybrid fusion weight stability", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.93])
    save_fig(fig, "120_31m_hybrid_fusion_weight_stability.png")


def plot_pooled_predictions(pred_df: pd.DataFrame, summary_df: pd.DataFrame):
    top = summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE"], ascending=[True, True]).head(4)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 10.5), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (_, row) in zip(axes, top.iterrows()):
        method = str(row["Method"])
        sub = pred_df[pred_df["Method"].astype(str) == method].copy()
        sub_plot = sub.sample(900, random_state=42) if len(sub) > 900 else sub
        ax.scatter(sub_plot["Observed"], sub_plot["Predicted"], s=16, color=COLORS.get(method, COLORS["gray"]), alpha=0.68, edgecolor="none")
        ax.plot([0, 100], [0, 100], color=COLORS["red"], linestyle=(0, (4, 3)), linewidth=1.0)
        m = regression_metrics(sub["Observed"], sub["Predicted"])
        ax.text(0.06, 0.92, f"$R^2$ = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nMAE = {m['MAE']:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, boxstyle="round,pad=0.3"))
        ax.set_title(method)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Observed F removal (%)")
        ax.set_ylabel("Predicted F removal (%)")
        ax.grid(color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    fig.suptitle("Top pooled predicted-vs-observed plots after 31M hybrid fusion", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    save_fig(fig, "121_31m_hybrid_fusion_pooled_predicted_vs_actual.png")


def plot_residual_correlation(corr_df: pd.DataFrame, summary_df: pd.DataFrame):
    if corr_df.empty or corr_df.shape[0] < 2:
        return
    # Show best 8 methods by RMSE for readability.
    keep_methods = summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE"], ascending=[True, True]).head(8)["Method"].astype(str).tolist()
    corr = corr_df.set_index("Method")
    keep_methods = [m for m in keep_methods if m in corr.index and m in corr.columns]
    if len(keep_methods) < 2:
        return
    vals = corr.loc[keep_methods, keep_methods].values.astype(float)
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    im = ax.imshow(vals, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(keep_methods)))
    ax.set_yticks(np.arange(len(keep_methods)))
    ax.set_xticklabels(keep_methods, rotation=45, ha="right")
    ax.set_yticklabels(keep_methods)
    for i in range(len(keep_methods)):
        for j in range(len(keep_methods)):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(vals[i, j]) > 0.55 else "black")
    ax.set_title("Residual correlation among best 31M candidates")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson residual correlation")
    fig.tight_layout()
    save_fig(fig, "122_31m_hybrid_fusion_residual_correlation.png")


def write_decision_report(summary_df, test_df, weights_df, meta_df):
    path = TABLE_DIR / "31m_hybrid_fusion_decision.txt"
    rank = summary_df.sort_values(["Pooled_RMSE", "Pooled_MAE", "Pooled_R2"], ascending=[True, True, False]).reset_index(drop=True)
    best = rank.iloc[0]
    base_rows = rank[rank["Method"].astype(str) == "KRR_Poly"]
    krr = base_rows.iloc[0] if not base_rows.empty else None
    cbx_rows = rank[rank["Method"].astype(str) == "CB-XGB"]
    cbx = cbx_rows.iloc[0] if not cbx_rows.empty else None

    lines = []
    lines.append("=" * 100)
    lines.append("Step 31M hybrid KRR-tree fusion and grouped OOF stacking report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Test whether KRR_Poly can improve further after being combined with the CB-XGB tree route, and whether grouped OOF stacking adds stable value.")
    lines.append("")
    lines.append("Strict workflow reused:")
    lines.append("- Current31F_noFine_plus_31B_mechanism feature route confirmed by 31K")
    lines.append("- Step-31F balanced literature-level grouped split")
    lines.append("- Literature-group weighted training")
    lines.append("- Training-only RF feature ranking")
    lines.append("- Validation-only base config and fusion-weight selection")
    lines.append("- Grouped OOF meta-model training without test-label use")
    lines.append("")
    lines.append("Performance ranking:")
    keep_cols = ["Method", "Pooled_R2", "Pooled_RMSE", "Pooled_MAE", "Pooled_Bias", "R2_mean", "RMSE_mean", "MAE_mean", "n_pooled"]
    lines.append(rank[keep_cols].to_string(index=False))
    lines.append("")
    lines.append("Best 31M candidate:")
    lines.append(f"{best['Method']} | Pooled_R2={best['Pooled_R2']:.4f}, RMSE={best['Pooled_RMSE']:.4f}, MAE={best['Pooled_MAE']:.4f}")
    if krr is not None:
        lines.append("")
        lines.append("KRR_Poly reference:")
        lines.append(f"KRR_Poly | Pooled_R2={krr['Pooled_R2']:.4f}, RMSE={krr['Pooled_RMSE']:.4f}, MAE={krr['Pooled_MAE']:.4f}")
        lines.append(f"Best-vs-KRR RMSE gain = {krr['Pooled_RMSE'] - best['Pooled_RMSE']:.4f}")
        lines.append(f"Best-vs-KRR R2 gain = {best['Pooled_R2'] - krr['Pooled_R2']:.4f}")
    if cbx is not None:
        lines.append("")
        lines.append("CB-XGB reference:")
        lines.append(f"CB-XGB | Pooled_R2={cbx['Pooled_R2']:.4f}, RMSE={cbx['Pooled_RMSE']:.4f}, MAE={cbx['Pooled_MAE']:.4f}")
        lines.append(f"Best-vs-CB-XGB RMSE gain = {cbx['Pooled_RMSE'] - best['Pooled_RMSE']:.4f}")
        lines.append(f"Best-vs-CB-XGB R2 gain = {best['Pooled_R2'] - cbx['Pooled_R2']:.4f}")
    lines.append("")
    lines.append("Fusion-weight summary:")
    if weights_df.empty:
        lines.append("No validation-weighted fusion rows were generated.")
    else:
        lines.append(weights_df.groupby(["Method", "Base_Model"]).agg(Weight_mean=("Weight", "mean"), Weight_std=("Weight", "std"), n_seed=("Seed", "nunique")).reset_index().to_string(index=False))
    lines.append("")
    lines.append("Meta-model selection summary:")
    if meta_df.empty:
        lines.append("No OOF meta-model records available.")
    else:
        show_cols = [c for c in ["Seed", "Stacking_Method", "kind", "alpha", "l1_ratio", "GroupedCV_RMSE", "GroupedCV_RMSE_std"] if c in meta_df.columns]
        lines.append(meta_df[show_cols].sort_values(show_cols[:2] if len(show_cols) >= 2 else show_cols).head(30).to_string(index=False))
    lines.append("")
    lines.append("Decision guide:")
    lines.append("1. If a hybrid fusion/OOF method improves over KRR_Poly and CB-XGB, carry it to the next robustness step.")
    lines.append("2. If KRR_Poly remains best, keep KRR_Poly as the new grouped-generalization main candidate and use CB-XGB as tree baseline.")
    lines.append("3. If OOF stacking improves only validation but not grouped test, treat it as overfitting-prone and do not use as final main model.")
    lines.append("4. If all methods remain far below all-domain R2=0.5, continue to additional robustness/modeling checks before using applicability domain as the final technical safeguard.")
    lines.append("=" * 100)
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[TEXT] {path}")


# ============================================================
# 7. Main
# ============================================================


def main():
    start_time = time.time()
    log("\n" + "=" * 100)
    log("Step 31M: hybrid KRR-tree fusion and grouped OOF stacking")
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

    all_splits, all_candidates, all_tests, all_predictions, all_weights, all_meta, all_oof_audits = [], [], [], [], [], [], []
    for seed in SEEDS:
        result = run_one_seed(X=X, y=y, groups=groups, seed=seed)
        all_splits.append(result["split"])
        all_candidates.append(result["candidate"])
        all_tests.append(result["test"])
        all_predictions.append(result["prediction"])
        all_weights.append(result["weights"])
        all_meta.append(result["meta"])
        all_oof_audits.append(result["oof_audit"])

    split_df = pd.concat(all_splits, ignore_index=True)
    cand_df = pd.concat(all_candidates, ignore_index=True)
    test_df = pd.concat(all_tests, ignore_index=True)
    pred_df = pd.concat(all_predictions, ignore_index=True)
    weights_df = pd.concat(all_weights, ignore_index=True) if any(len(x) for x in all_weights) else pd.DataFrame(columns=["Seed", "Method", "Base_Model", "Weight"])
    meta_df = pd.concat(all_meta, ignore_index=True) if any(len(x) for x in all_meta) else pd.DataFrame()
    oof_audit_df = pd.concat(all_oof_audits, ignore_index=True) if any(len(x) for x in all_oof_audits) else pd.DataFrame()

    summary_df = summarize_results(test_df, pred_df)
    corr_df = compute_residual_correlation(pred_df)

    out_tables = {
        "31m_hybrid_fusion_split_summary.csv": split_df,
        "31m_hybrid_fusion_validation_candidates.csv": cand_df,
        "31m_hybrid_fusion_test_results.csv": test_df,
        "31m_hybrid_fusion_predictions.csv": pred_df,
        "31m_hybrid_fusion_weights.csv": weights_df,
        "31m_hybrid_fusion_oof_meta_results.csv": meta_df,
        "31m_hybrid_fusion_oof_audit.csv": oof_audit_df,
        "31m_hybrid_fusion_summary.csv": summary_df,
        "31m_hybrid_fusion_residual_correlation.csv": corr_df,
    }
    for filename, df in out_tables.items():
        path = TABLE_DIR / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        log(f"[TABLE] {path}")

    log("\n=== Step 31M summary ===")
    print(summary_df.to_string(index=False))

    write_decision_report(summary_df, test_df, weights_df, meta_df)

    log("Generating figures...")
    plot_metrics(summary_df)
    plot_weight_stability(weights_df)
    plot_pooled_predictions(pred_df, summary_df)
    plot_residual_correlation(corr_df, summary_df)

    elapsed = time.time() - start_time
    log("\n" + "=" * 100)
    log("[DONE] Step 31M hybrid KRR-tree fusion and grouped OOF stacking completed.")
    log(f"[TIME] Total runtime: {elapsed:.2f} s ({elapsed/60:.2f} min)")
    log("=" * 100)
    log("Please send these outputs:")
    log("1. outputs/tables/31m_hybrid_fusion_summary.csv")
    log("2. outputs/tables/31m_hybrid_fusion_decision.txt")
    log("3. outputs/tables/31m_hybrid_fusion_weights.csv")
    log("4. outputs/tables/31m_hybrid_fusion_residual_correlation.csv")
    log("5. outputs/figures/119_31m_hybrid_fusion_metrics.png")
    log("6. outputs/figures/120_31m_hybrid_fusion_weight_stability.png")
    log("7. outputs/figures/121_31m_hybrid_fusion_pooled_predicted_vs_actual.png")
    log("8. outputs/figures/122_31m_hybrid_fusion_residual_correlation.png")
    log("=" * 100)


if __name__ == "__main__":
    main()
