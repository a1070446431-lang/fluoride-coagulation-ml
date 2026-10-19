# -*- coding: utf-8 -*-
"""
31d_group_weighted_mechanism_validation.py

Purpose
-------
Grouped validation with mechanism-aware features plus literature-group-weighted training.

This script uses:
    data/processed/model_ready_v1.csv
        - final 1011 x 26 model-ready table
        - no literature ID, safe for training

    data/processed/model_ready_v1_with_source.csv
        - final 1011 x 27 table
        - contains 文献编号 only for grouped splitting

    data/processed/model_ready_v1_with_source_mechanism31b.csv
        - final 1011-row table with Step-31B mechanism-aware descriptors
        - 文献编号 is still used only for grouped splitting, not as an input feature

Main question
-------------
Does literature-group-weighted training improve grouped validation after Step-31B mechanism-aware feature generalization?

Candidate methods
-----------------
1. Selected_Weighted_4Model
2. CatBoost_HGB_Weighted_2Model
3. RF_CatBoost_Weighted
4. Selected_HistGradientBoosting
5. Selected_CatBoost
6. Selected_RandomForest
7. Selected_ExtraTrees

Strict rules
------------
1. 文献编号 is used only for grouped splitting.
2. 文献编号 is not used as an input feature.
3. Feature engineering, feature selection, base model parameters,
   random seeds, and validation-RMSE weight search are reused from script 19.
4. The intended modeling changes are: (i) Step-31B mechanism-aware input representation;
   (ii) training sample weights so each literature group contributes approximately equally.
   Splits, seeds, candidate models, validation-RMSE weight search, and metrics are otherwise reused.
4. Group overlap among train / validation / test is forbidden.

Outputs
-------
Tables:
    outputs/tables/31d_grouped_mechanism_screen_group_distribution.csv
    outputs/tables/31d_grouped_mechanism_screen_split_summary.csv
    outputs/tables/31d_grouped_mechanism_screen_validation_candidates.csv
    outputs/tables/31d_grouped_mechanism_screen_test_results.csv
    outputs/tables/31d_grouped_mechanism_screen_predictions.csv
    outputs/tables/31d_grouped_mechanism_screen_summary.csv
    outputs/tables/31d_grouped_mechanism_screen_decision.txt

Figures:
    outputs/figures/86_group_weighted_mechanism_screen_metrics.png/pdf
    outputs/figures/87_group_weighted_mechanism_screen_pooled_predicted_vs_actual.png/pdf
    outputs/figures/88_group_weighted_mechanism_screen_weight_stability.png/pdf
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

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ============================================================
# 0. Basic logging
# ============================================================

def log(msg: str):
    print(msg, flush=True)


# ============================================================
# 1. Load real script 19
# ============================================================

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
    log("Loading real 19_selected_feature_fusion.py")
    log(str(module_path))
    log("=" * 100)

    spec = importlib.util.spec_from_file_location("sf19_real", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


sf19 = load_script19()

PROJECT_ROOT = Path(sf19.PROJECT_ROOT)
FIG_DIR = Path(sf19.FIG_DIR)
TABLE_DIR = Path(sf19.TABLE_DIR)

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_READY_PATH = PROCESSED_DIR / "model_ready_v1.csv"
MODEL_READY_WITH_SOURCE_PATH = PROCESSED_DIR / "model_ready_v1_with_source.csv"
MECHANISM31B_PATH = PROCESSED_DIR / "model_ready_v1_with_source_mechanism31b.csv"
MECHANISM_COLS_PATH = TABLE_DIR / "31b_mechanism_feature_columns.csv"

# Strict category-generalization setting for Step 31D.
# False means the over-specific raw labels are removed from model input and
# replaced by the Step-31B mechanism-aware descriptors.
# This avoids letting models rely on literature-specific raw category names.
KEEP_RAW_FINE_CATEGORIES = False
RAW_FINE_CATEGORY_COLS_TO_REPLACE = ["混凝剂", "干扰离子"]


# ============================================================
# 2. Settings
# ============================================================

SEEDS = list(getattr(sf19, "SEEDS", [7, 21, 42, 66, 88]))
TEST_SIZE = float(getattr(sf19, "TEST_SIZE", 0.20))
VALID_SIZE_IN_TRAIN = float(getattr(sf19, "VALID_SIZE_IN_TRAIN", 0.25))
K_OPTIONS = list(getattr(sf19, "K_OPTIONS", ["all"]))

FOUR_MODEL_NAMES = [
    "RandomForest",
    "CatBoost",
    "HistGradientBoosting",
    "ExtraTrees",
]

TWO_MODEL_NAMES = [
    "CatBoost",
    "HistGradientBoosting",
]

RF_CATBOOST_NAMES = [
    "RandomForest",
    "CatBoost",
]

SINGLE_MODEL_TO_METHOD = {
    "CatBoost": "Selected_CatBoost",
    "HistGradientBoosting": "Selected_HistGradientBoosting",
    "RandomForest": "Selected_RandomForest",
    "ExtraTrees": "Selected_ExtraTrees",
}

METHOD_ORDER = [
    "Selected_Weighted_4Model",
    "CatBoost_HGB_Weighted_2Model",
    "RF_CatBoost_Weighted",
    "Selected_HistGradientBoosting",
    "Selected_CatBoost",
    "Selected_RandomForest",
    "Selected_ExtraTrees",
]

METHOD_LABELS = {
    "Selected_Weighted_4Model": "Weighted 4Model",
    "CatBoost_HGB_Weighted_2Model": "CatBoost-HGB 2Model",
    "RF_CatBoost_Weighted": "RF-CatBoost",
    "Selected_HistGradientBoosting": "Selected HGB",
    "Selected_CatBoost": "Selected CatBoost",
    "Selected_RandomForest": "Selected RF",
    "Selected_ExtraTrees": "Selected ExtraTrees",
}

MODEL_LABELS = {
    "RandomForest": "RF",
    "CatBoost": "CatBoost",
    "HistGradientBoosting": "HGB",
    "ExtraTrees": "ExtraTrees",
}

COLORS = {
    "Selected_Weighted_4Model": "#E28E68",
    "CatBoost_HGB_Weighted_2Model": "#63B7B2",
    "RF_CatBoost_Weighted": "#5B84B1",
    "Selected_HistGradientBoosting": "#63A97A",
    "Selected_CatBoost": "#9B72B0",
    "Selected_RandomForest": "#4C78A8",
    "Selected_ExtraTrees": "#A779A5",
    "RandomForest": "#4C78A8",
    "CatBoost": "#9B72B0",
    "HistGradientBoosting": "#63A97A",
    "ExtraTrees": "#8C564B",
    "red": "#FF0000",
    "gray": "#666666",
    "dark": "#222222",
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
# 3. General utilities
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
    rmse = math.sqrt(mse)

    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "Bias": float(np.mean(y_true - y_pred)),
    }


def weight_dict_to_text(weight_dict: dict) -> str:
    clean = {}
    for k, v in weight_dict.items():
        clean[str(k)] = float(v)
    return str(clean)


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

    if len(vals) == 0:
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


def normalize_group_value(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if s == "":
        return np.nan

    try:
        f = float(s)
        if f.is_integer():
            return f"Paper_{int(f):03d}"
    except Exception:
        pass

    return s.replace(" ", "_")



# ============================================================
# 4. Step-31B mechanism-aware feature integration
# ============================================================

def load_mechanism31b_feature_columns(df_mech: pd.DataFrame) -> list[str]:
    """Load the exact Step-31B mechanism feature-column list.

    The preferred source is outputs/tables/31b_mechanism_feature_columns.csv,
    created by Step 31B. If it is missing, fall back to columns beginning with
    "机理_" in the mechanism31b data table.
    """
    if MECHANISM_COLS_PATH.exists():
        cols_df = read_csv_smart(MECHANISM_COLS_PATH)
        first_col = cols_df.columns[0]
        cols = cols_df[first_col].dropna().astype(str).str.strip().tolist()
    else:
        cols = [c for c in df_mech.columns if str(c).startswith("机理_")]

    # Safety filter: never let target/source columns enter the model.
    forbidden_keywords = [
        "文献编号", "source", "paper", "literature", "reference",
        "氟去除率", "去除率_pct", "removal", "出水", "effluent",
    ]
    clean_cols = []
    for c in cols:
        s = str(c)
        s_lower = s.lower()
        if any(k.lower() in s_lower for k in forbidden_keywords):
            continue
        if s in df_mech.columns:
            clean_cols.append(s)

    clean_cols = list(dict.fromkeys(clean_cols))

    if len(clean_cols) == 0:
        raise ValueError(
            "No valid Step-31B mechanism feature columns were found. "
            "Run 31b_build_mechanism_generalized_features.py first."
        )

    return clean_cols


def append_mechanism31b_features(X_mix: pd.DataFrame, expected_n: int) -> pd.DataFrame:
    """Build the Step-31D feature table.

    This keeps the real script-19 X_mix construction and then changes only the
    category representation by adding Step-31B mechanism-aware descriptors.
    By default, over-specific raw labels 混凝剂 and 干扰离子 are removed from the
    model input to prevent literature-specific category memorization.
    """
    if not MECHANISM31B_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {MECHANISM31B_PATH}. "
            "Run 31b_build_mechanism_generalized_features.py first."
        )

    df_mech = read_csv_smart(MECHANISM31B_PATH)

    if len(df_mech) != expected_n:
        raise ValueError(
            f"Row mismatch: script-19 y has {expected_n} rows, "
            f"but mechanism31b data has {len(df_mech)} rows."
        )

    mechanism_cols = load_mechanism31b_feature_columns(df_mech)

    X_base = X_mix.copy().reset_index(drop=True)
    base_cols_before = list(X_base.columns)

    removed_cols = []
    if not KEEP_RAW_FINE_CATEGORIES:
        for c in RAW_FINE_CATEGORY_COLS_TO_REPLACE:
            if c in X_base.columns:
                X_base = X_base.drop(columns=[c])
                removed_cols.append(c)

    # Avoid accidental duplicate names.
    append_cols = [c for c in mechanism_cols if c not in X_base.columns]
    duplicate_skipped = [c for c in mechanism_cols if c in X_base.columns]

    X_mech = df_mech[append_cols].copy().reset_index(drop=True)
    X_out = pd.concat([X_base.reset_index(drop=True), X_mech], axis=1)

    duplicated = X_out.columns[X_out.columns.duplicated()].tolist()
    if duplicated:
        log(f"[WARN] Duplicate columns after Step-31B append; keeping first occurrence: {duplicated}")
        X_out = X_out.loc[:, ~X_out.columns.duplicated()].copy()

    # Save an audit log for reproducibility.
    audit_rows = []
    audit_rows.append({"Item": "X_mix_base_n_columns", "Value": len(base_cols_before)})
    audit_rows.append({"Item": "KEEP_RAW_FINE_CATEGORIES", "Value": KEEP_RAW_FINE_CATEGORIES})
    audit_rows.append({"Item": "Removed_raw_fine_category_columns", "Value": ";".join(removed_cols) if removed_cols else ""})
    audit_rows.append({"Item": "Mechanism31B_columns_loaded", "Value": len(mechanism_cols)})
    audit_rows.append({"Item": "Mechanism31B_columns_appended", "Value": len(append_cols)})
    audit_rows.append({"Item": "Mechanism31B_duplicate_skipped", "Value": ";".join(duplicate_skipped) if duplicate_skipped else ""})
    audit_rows.append({"Item": "X_mix_31D_n_columns", "Value": X_out.shape[1]})

    audit_path = TABLE_DIR / "31d_mechanism_feature_integration_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False, encoding="utf-8-sig")

    selected_cols_path = TABLE_DIR / "31d_mechanism_feature_columns_used.csv"
    pd.DataFrame({"mechanism_feature_column_used": append_cols}).to_csv(
        selected_cols_path, index=False, encoding="utf-8-sig"
    )

    log("\n[OK] Step-31B mechanism-aware features integrated into script-19 X_mix.")
    log(f"[OK] Base X_mix columns       : {len(base_cols_before)}")
    log(f"[OK] Removed raw fine labels : {removed_cols}")
    log(f"[OK] Mechanism columns loaded: {len(mechanism_cols)}")
    log(f"[OK] Mechanism columns appended: {len(append_cols)}")
    log(f"[OK] Final Step-31D X_mix shape: {X_out.shape}")
    log(f"[TABLE] {audit_path}")
    log(f"[TABLE] {selected_cols_path}")

    return X_out


# ============================================================
# 5. Group labels
# ============================================================

def load_group_labels(expected_n: int) -> pd.Series:
    if not MODEL_READY_WITH_SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {MODEL_READY_WITH_SOURCE_PATH}. "
            "Run 29c first to generate model_ready_v1_with_source.csv."
        )

    df_source = read_csv_smart(MODEL_READY_WITH_SOURCE_PATH)

    if len(df_source) != expected_n:
        raise ValueError(
            f"Row mismatch: model_ready_v1 has {expected_n} rows, "
            f"model_ready_v1_with_source has {len(df_source)} rows."
        )

    possible_cols = [
        "文献编号",
        "source_id",
        "paper_id",
        "literature_id",
        "reference_id",
    ]

    group_col = None
    for c in possible_cols:
        if c in df_source.columns:
            group_col = c
            break

    if group_col is None:
        raise ValueError(
            "No literature group column found in model_ready_v1_with_source.csv. "
            "Expected column: 文献编号."
        )

    groups = df_source[group_col].map(normalize_group_value).reset_index(drop=True)

    if groups.isna().any():
        missing_n = int(groups.isna().sum())
        raise ValueError(f"文献编号 has {missing_n} missing values. Complete it before grouped validation.")

    dist = groups.value_counts().rename_axis("文献编号").reset_index(name="n_rows")
    dist["fraction"] = dist["n_rows"] / len(groups)

    dist_path = TABLE_DIR / "31d_grouped_mechanism_screen_group_distribution.csv"
    dist.to_csv(dist_path, index=False, encoding="utf-8-sig")

    log(f"[OK] Group labels loaded from column: {group_col}")
    log(f"[OK] Rows: {len(groups)}")
    log(f"[OK] Literature groups: {groups.nunique()}")
    log(f"[TABLE] {dist_path}")
    log("\nTop literature groups:")
    print(dist.head(20).to_string(index=False))

    if groups.nunique() < 5:
        raise ValueError("Too few literature groups for grouped validation.")

    return groups


# ============================================================
# 5. Group split
# ============================================================

def grouped_split_indices(groups: pd.Series, seed: int):
    all_idx = np.arange(len(groups))

    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=seed,
    )

    train_valid_idx_local, test_idx_local = next(
        gss_test.split(
            np.zeros((len(all_idx), 1)),
            np.zeros(len(all_idx)),
            groups=groups,
        )
    )

    train_valid_idx = all_idx[train_valid_idx_local]
    test_idx = all_idx[test_idx_local]

    groups_train_valid = groups.iloc[train_valid_idx].reset_index(drop=True)

    gss_valid = GroupShuffleSplit(
        n_splits=1,
        test_size=VALID_SIZE_IN_TRAIN,
        random_state=seed + 1000,
    )

    train_idx_local, valid_idx_local = next(
        gss_valid.split(
            np.zeros((len(train_valid_idx), 1)),
            np.zeros(len(train_valid_idx)),
            groups=groups_train_valid,
        )
    )

    train_idx = train_valid_idx[train_idx_local]
    valid_idx = train_valid_idx[valid_idx_local]

    train_groups = set(groups.iloc[train_idx].astype(str))
    valid_groups = set(groups.iloc[valid_idx].astype(str))
    test_groups = set(groups.iloc[test_idx].astype(str))

    if train_groups & valid_groups:
        raise RuntimeError("Group leakage detected: train and validation overlap.")

    if train_groups & test_groups:
        raise RuntimeError("Group leakage detected: train and test overlap.")

    if valid_groups & test_groups:
        raise RuntimeError("Group leakage detected: validation and test overlap.")

    split_info = {
        "Seed": seed,
        "n_train": len(train_idx),
        "n_valid": len(valid_idx),
        "n_test": len(test_idx),
        "n_train_groups": len(train_groups),
        "n_valid_groups": len(valid_groups),
        "n_test_groups": len(test_groups),
        "train_groups": ";".join(sorted(train_groups)),
        "valid_groups": ";".join(sorted(valid_groups)),
        "test_groups": ";".join(sorted(test_groups)),
    }

    return train_idx, valid_idx, test_idx, split_info



# ============================================================
# 5B. Literature-group sample weights
# ============================================================

def make_literature_group_weights(group_labels: pd.Series) -> np.ndarray:
    """Give each literature group approximately equal total training weight.

    For samples from a group with n_g rows, each sample receives 1 / n_g.
    The vector is then rescaled so that mean weight = 1. This keeps model
    hyperparameters on roughly the same scale while reducing domination by
    large literature groups.
    """
    g = pd.Series(group_labels).astype(str).reset_index(drop=True)
    counts = g.value_counts()
    w = g.map(lambda x: 1.0 / float(counts[x])).astype(float).values
    w = w / np.mean(w)
    return w


def fit_predict_model_weighted(model_name, X_train, y_train, X_target, seed, sample_weight=None):
    if model_name not in sf19.MODEL_FACTORIES:
        raise KeyError(f"{model_name} not found in sf19.MODEL_FACTORIES.")

    preprocessor, _, _ = sf19.build_preprocessor(X_train)
    model = sf19.MODEL_FACTORIES[model_name](seed)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    if sample_weight is None:
        pipe.fit(X_train, y_train)
    else:
        pipe.fit(X_train, y_train, model__sample_weight=np.asarray(sample_weight, dtype=float))

    pred = pipe.predict(X_target)
    return np.asarray(pred, dtype=float)


def rank_features_by_rf_importance_weighted(X_train: pd.DataFrame, y_train: pd.Series, seed: int, sample_weight=None):
    """Script-19 RF feature ranking, with optional literature-group sample weights."""
    preprocessor, numeric_cols, categorical_cols = sf19.build_preprocessor(X_train)

    try:
        rf = sf19.RandomForestRegressor(
            n_estimators=700,
            max_depth=None,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(
            n_estimators=700,
            max_depth=None,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", rf),
    ])

    if sample_weight is None:
        pipe.fit(X_train, y_train)
    else:
        pipe.fit(X_train, y_train, model__sample_weight=np.asarray(sample_weight, dtype=float))

    fitted_pre = pipe.named_steps["preprocessor"]
    fitted_model = pipe.named_steps["model"]

    encoded_names = sf19.get_feature_names_from_preprocessor(fitted_pre)
    importances = fitted_model.feature_importances_

    if len(encoded_names) != len(importances):
        encoded_names = [f"feature_{i}" for i in range(len(importances))]

    original_cols = list(X_train.columns)
    agg = {}

    for enc_name, imp in zip(encoded_names, importances):
        source_name = sf19.source_from_encoded_name(enc_name, original_cols)
        agg[source_name] = agg.get(source_name, 0.0) + float(imp)

    imp_df = pd.DataFrame({
        "Feature": list(agg.keys()),
        "Importance": list(agg.values()),
    })

    bad_keywords = [
        "行号", "序号", "编号", "index", "id", "doi",
        "出水", "effluent", "remaining", "残余", "去除率", "removal",
    ]

    imp_df = imp_df[
        ~imp_df["Feature"].astype(str).str.lower().apply(
            lambda x: any(k in x for k in bad_keywords)
        )
    ].copy()

    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    return imp_df

# ============================================================
# 6. Model wrappers from script 19
# ============================================================

def fit_predict_single_model(model_name, X_train, y_train, X_target, seed, sample_weight=None):
    return fit_predict_model_weighted(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_target=X_target,
        seed=seed,
        sample_weight=sample_weight,
    )


def evaluate_validation_candidates(
    X_train_k,
    y_train,
    X_valid_k,
    y_valid,
    seed,
    selected_k,
    selected_cols,
    sample_weight_train=None,
):
    rows = []
    valid_pred_by_model = {}

    # Single models
    for model_name in FOUR_MODEL_NAMES:
        if model_name not in sf19.MODEL_FACTORIES:
            continue

        pred_valid = fit_predict_single_model(
            model_name=model_name,
            X_train=X_train_k,
            y_train=y_train,
            X_target=X_valid_k,
            seed=seed,
            sample_weight=sample_weight_train,
        )

        valid_pred_by_model[model_name] = pred_valid

        method = SINGLE_MODEL_TO_METHOD[model_name]
        metrics = regression_metrics(y_valid, pred_valid)

        rows.append({
            "Seed": seed,
            "Method": method,
            "Selected_K": selected_k,
            "Selected_Features": ";".join(selected_cols),
            "Validation_R2": metrics["R2"],
            "Validation_RMSE": metrics["RMSE"],
            "Validation_MAE": metrics["MAE"],
            "Validation_MSE": metrics["MSE"],
            "Validation_Bias": metrics["Bias"],
            "Weights": "",
        })

    # Weighted 4Model
    pred_dict_4 = {
        m: valid_pred_by_model[m]
        for m in FOUR_MODEL_NAMES
        if m in valid_pred_by_model
    }

    if len(pred_dict_4) >= 3:
        weights_4, _ = sf19.search_best_weights(
            pred_dict_4,
            y_valid,
            seed,
        )
        pred_4 = sf19.apply_weights(pred_dict_4, weights_4)
        metrics_4 = regression_metrics(y_valid, pred_4)

        rows.append({
            "Seed": seed,
            "Method": "Selected_Weighted_4Model",
            "Selected_K": selected_k,
            "Selected_Features": ";".join(selected_cols),
            "Validation_R2": metrics_4["R2"],
            "Validation_RMSE": metrics_4["RMSE"],
            "Validation_MAE": metrics_4["MAE"],
            "Validation_MSE": metrics_4["MSE"],
            "Validation_Bias": metrics_4["Bias"],
            "Weights": weight_dict_to_text(weights_4),
        })

    # CatBoost-HGB weighted 2Model
    pred_dict_2 = {
        m: valid_pred_by_model[m]
        for m in TWO_MODEL_NAMES
        if m in valid_pred_by_model
    }

    if len(pred_dict_2) == 2:
        weights_2, _ = sf19.search_best_weights(
            pred_dict_2,
            y_valid,
            seed,
        )
        pred_2 = sf19.apply_weights(pred_dict_2, weights_2)
        metrics_2 = regression_metrics(y_valid, pred_2)

        rows.append({
            "Seed": seed,
            "Method": "CatBoost_HGB_Weighted_2Model",
            "Selected_K": selected_k,
            "Selected_Features": ";".join(selected_cols),
            "Validation_R2": metrics_2["R2"],
            "Validation_RMSE": metrics_2["RMSE"],
            "Validation_MAE": metrics_2["MAE"],
            "Validation_MSE": metrics_2["MSE"],
            "Validation_Bias": metrics_2["Bias"],
            "Weights": weight_dict_to_text(weights_2),
        })

    # RF-CatBoost weighted model
    pred_dict_rfcat = {
        m: valid_pred_by_model[m]
        for m in RF_CATBOOST_NAMES
        if m in valid_pred_by_model
    }

    if len(pred_dict_rfcat) == 2:
        weights_rfcat, _ = sf19.search_best_weights(
            pred_dict_rfcat,
            y_valid,
            seed,
        )
        pred_rfcat = sf19.apply_weights(pred_dict_rfcat, weights_rfcat)
        metrics_rfcat = regression_metrics(y_valid, pred_rfcat)

        rows.append({
            "Seed": seed,
            "Method": "RF_CatBoost_Weighted",
            "Selected_K": selected_k,
            "Selected_Features": ";".join(selected_cols),
            "Validation_R2": metrics_rfcat["R2"],
            "Validation_RMSE": metrics_rfcat["RMSE"],
            "Validation_MAE": metrics_rfcat["MAE"],
            "Validation_MSE": metrics_rfcat["MSE"],
            "Validation_Bias": metrics_rfcat["Bias"],
            "Weights": weight_dict_to_text(weights_rfcat),
        })

    return rows


def test_predict_method(method, weights_text, X_train_full_k, y_train_full, X_test_k, seed, sample_weight_train_full=None):
    if method == "Selected_CatBoost":
        return fit_predict_single_model("CatBoost", X_train_full_k, y_train_full, X_test_k, seed, sample_weight=sample_weight_train_full)

    if method == "Selected_HistGradientBoosting":
        return fit_predict_single_model("HistGradientBoosting", X_train_full_k, y_train_full, X_test_k, seed, sample_weight=sample_weight_train_full)

    if method == "Selected_RandomForest":
        return fit_predict_single_model("RandomForest", X_train_full_k, y_train_full, X_test_k, seed, sample_weight=sample_weight_train_full)

    if method == "Selected_ExtraTrees":
        return fit_predict_single_model("ExtraTrees", X_train_full_k, y_train_full, X_test_k, seed, sample_weight=sample_weight_train_full)

    if method == "Selected_Weighted_4Model":
        weights = parse_weight_text(weights_text)
        pred_dict = {}
        for m in FOUR_MODEL_NAMES:
            if m in sf19.MODEL_FACTORIES:
                pred_dict[m] = fit_predict_single_model(m, X_train_full_k, y_train_full, X_test_k, seed, sample_weight=sample_weight_train_full)
        return sf19.apply_weights(pred_dict, weights)

    if method == "CatBoost_HGB_Weighted_2Model":
        weights = parse_weight_text(weights_text)
        pred_dict = {}
        for m in TWO_MODEL_NAMES:
            pred_dict[m] = fit_predict_single_model(m, X_train_full_k, y_train_full, X_test_k, seed, sample_weight=sample_weight_train_full)
        return sf19.apply_weights(pred_dict, weights)

    if method == "RF_CatBoost_Weighted":
        weights = parse_weight_text(weights_text)
        pred_dict = {}
        for m in RF_CATBOOST_NAMES:
            pred_dict[m] = fit_predict_single_model(m, X_train_full_k, y_train_full, X_test_k, seed, sample_weight=sample_weight_train_full)
        return sf19.apply_weights(pred_dict, weights)

    raise ValueError(f"Unknown method: {method}")


# ============================================================
# 7. One seed run
# ============================================================

def run_one_seed(X_mix: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int):
    log("\n" + "=" * 100)
    log(f"Grouped mechanism-aware screening seed = {seed}")
    log("=" * 100)

    train_idx, valid_idx, test_idx, split_info = grouped_split_indices(groups, seed)

    log(
        f"Rows: train={split_info['n_train']}, "
        f"valid={split_info['n_valid']}, "
        f"test={split_info['n_test']}"
    )
    log(
        f"Groups: train={split_info['n_train_groups']}, "
        f"valid={split_info['n_valid_groups']}, "
        f"test={split_info['n_test_groups']}"
    )

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

    sample_weight_train = make_literature_group_weights(train_groups_for_weight)
    sample_weight_train_full = make_literature_group_weights(train_full_groups_for_weight)

    log(
        "Group-weighted training: each literature group has approximately equal total weight. "
        f"train weight range=({sample_weight_train.min():.3f}, {sample_weight_train.max():.3f}); "
        f"train-full weight range=({sample_weight_train_full.min():.3f}, {sample_weight_train_full.max():.3f})"
    )

    # Feature ranking only on training set, with literature-group weights.
    log("Ranking features using grouped training set only, with literature-group weights...")
    imp_df = rank_features_by_rf_importance_weighted(
        X_train,
        y_train,
        seed,
        sample_weight=sample_weight_train,
    )

    candidate_rows = []

    for selected_k in K_OPTIONS:
        selected_cols = sf19.select_top_features(imp_df, X_mix, selected_k)
        selected_cols = [c for c in selected_cols if c in X_mix.columns]

        if len(selected_cols) < 3:
            continue

        log("-" * 100)
        log(f"Seed={seed} | K={selected_k} | n_features={len(selected_cols)}")
        log("-" * 100)

        X_train_k = X_train[selected_cols].reset_index(drop=True)
        X_valid_k = X_valid[selected_cols].reset_index(drop=True)

        rows = evaluate_validation_candidates(
            X_train_k=X_train_k,
            y_train=y_train,
            X_valid_k=X_valid_k,
            y_valid=y_valid,
            seed=seed,
            selected_k=selected_k,
            selected_cols=selected_cols,
            sample_weight_train=sample_weight_train,
        )

        candidate_rows.extend(rows)

    candidate_df = pd.DataFrame(candidate_rows)

    test_rows = []
    pred_rows = []

    for method in METHOD_ORDER:
        sub = candidate_df[candidate_df["Method"] == method].copy()

        if sub.empty:
            continue

        best = sub.sort_values("Validation_RMSE", ascending=True).iloc[0]

        selected_k = best["Selected_K"]
        selected_cols = str(best["Selected_Features"]).split(";")
        selected_cols = [c for c in selected_cols if c in X_mix.columns]

        X_train_full_k = X_train_full[selected_cols].reset_index(drop=True)
        X_test_k = X_test[selected_cols].reset_index(drop=True)

        log("=" * 100)
        log(
            f"Final grouped test | seed={seed} | method={method} "
            f"| K={selected_k} | n_features={len(selected_cols)}"
        )
        log("=" * 100)

        pred_test = test_predict_method(
            method=method,
            weights_text=best["Weights"],
            X_train_full_k=X_train_full_k,
            y_train_full=y_train_full,
            X_test_k=X_test_k,
            seed=seed,
            sample_weight_train_full=sample_weight_train_full,
        )

        metrics = regression_metrics(y_test, pred_test)

        test_rows.append({
            "Seed": seed,
            "Method": method,
            "Selected_K": selected_k,
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
            "n_test": len(y_test),
            "n_test_groups": test_groups.nunique(),
        })

        for obs, pred, gid in zip(y_test, pred_test, test_groups):
            pred_rows.append({
                "Seed": seed,
                "Method": method,
                "Selected_K": selected_k,
                "文献编号": gid,
                "Observed": float(obs),
                "Predicted": float(pred),
                "Residual": float(obs - pred),
            })

    return {
        "split": pd.DataFrame([split_info]),
        "candidate": candidate_df,
        "test": pd.DataFrame(test_rows),
        "prediction": pd.DataFrame(pred_rows),
    }


# ============================================================
# 8. Summary
# ============================================================

def summarize_results(test_df: pd.DataFrame, pred_df: pd.DataFrame):
    summary = (
        test_df
        .groupby("Method", as_index=False)
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

    for method, sub in pred_df.groupby("Method"):
        metrics = regression_metrics(sub["Observed"], sub["Predicted"])

        pooled_rows.append({
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
    summary = summary.merge(pooled_df, on="Method", how="left")

    summary["Method"] = pd.Categorical(
        summary["Method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    summary = summary.sort_values("Method").reset_index(drop=True)

    return summary


def write_decision_report(summary_df, split_df, group_dist):
    path = TABLE_DIR / "31d_grouped_mechanism_screen_decision.txt"

    ranking = summary_df.sort_values(
        ["Pooled_RMSE", "Pooled_MAE", "Pooled_R2"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    best = ranking.iloc[0]

    lines = []
    lines.append("=" * 100)
    lines.append("Step 31D grouped validation with mechanism-aware features report")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Screen candidate models under literature-level grouped validation after Step-31B mechanism-aware category generalization and literature-group-weighted training.")
    lines.append("")
    lines.append("Data:")
    lines.append("model_ready_v1.csv = final 1011 x 26 legacy-structure table, without 文献编号.")
    lines.append("model_ready_v1_with_source.csv = same rows plus 文献编号, used only for grouped splitting.")
    lines.append("model_ready_v1_with_source_mechanism31b.csv = same rows plus Step-31B mechanism-aware descriptors. Training uses literature-group sample weights; 文献编号 remains excluded from input features.")
    lines.append("")
    lines.append("Group summary:")
    lines.append(f"n literature groups : {int(group_dist['文献编号'].nunique())}")
    lines.append(f"largest group       : {int(group_dist['n_rows'].max())}")
    lines.append(f"smallest group      : {int(group_dist['n_rows'].min())}")
    lines.append(f"median group size   : {float(group_dist['n_rows'].median()):.2f}")
    lines.append("")
    lines.append("Split summary:")
    lines.append(split_df[[
        "Seed", "n_train", "n_valid", "n_test",
        "n_train_groups", "n_valid_groups", "n_test_groups"
    ]].to_string(index=False))
    lines.append("")
    lines.append("Performance summary:")
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
        "Pooled_Bias",
        "n_pooled",
        "n_literature_groups_pooled",
    ]
    lines.append(summary_df[keep_cols].to_string(index=False))
    lines.append("")
    lines.append("Ranking by pooled RMSE:")
    lines.append(ranking[[
        "Method", "Pooled_R2", "Pooled_RMSE", "Pooled_MAE", "Pooled_Bias"
    ]].to_string(index=False))
    lines.append("")
    lines.append("Current grouped-screen best candidate:")
    lines.append(
        f"{best['Method']} | "
        f"Pooled_R2={best['Pooled_R2']:.4f}, "
        f"Pooled_RMSE={best['Pooled_RMSE']:.4f}, "
        f"Pooled_MAE={best['Pooled_MAE']:.4f}"
    )
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("1. If the grouped result remains close to random-split performance, the model has useful cross-literature robustness.")
    lines.append("2. If grouped performance drops clearly, random-split performance should be reported as interpolation performance only.")
    lines.append("3. The next strict ablation should focus on the best grouped-screen candidates, not blindly follow random-split ranking.")
    lines.append("=" * 100)

    path.write_text("\n".join(lines), encoding="utf-8")

    log(f"[TEXT] {path}")


# ============================================================
# 9. Figures
# ============================================================

def plot_metrics(summary_df: pd.DataFrame):
    data = summary_df.copy()
    data["Method_str"] = data["Method"].astype(str)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    specs = [
        ("Pooled_R2", "Pooled $R^2$", "{:.3f}"),
        ("Pooled_RMSE", "Pooled RMSE", "{:.2f}"),
        ("Pooled_MAE", "Pooled MAE", "{:.2f}"),
    ]

    for ax, (col, ylabel, fmt) in zip(axes, specs):
        vals = data[col].astype(float).values
        x = np.arange(len(data))
        bar_colors = [COLORS.get(m, COLORS["gray"]) for m in data["Method_str"]]

        bars = ax.bar(
            x,
            vals,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.45,
            alpha=0.90,
        )

        best_idx = int(np.nanargmin(vals)) if col != "Pooled_R2" else int(np.nanargmax(vals))
        bars[best_idx].set_edgecolor(COLORS["red"])
        bars[best_idx].set_linewidth(1.4)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [METHOD_LABELS.get(m, m) for m in data["Method_str"]],
            rotation=32,
            ha="right",
        )

        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)

        if col == "Pooled_R2":
            ymin = max(0, np.nanmin(vals) - 0.15)
            ymax = min(1.0, np.nanmax(vals) + 0.08)
            ax.set_ylim(ymin, ymax)
            offset = (ymax - ymin) * 0.025
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
        "Grouped validation with mechanism-aware features under literature-level splits",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.01,
        "All methods reuse script-19 settings; mechanism descriptors are combined with literature-group-weighted training.",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    fig.tight_layout(rect=[0.02, 0.06, 1, 0.93])

    save_fig(fig, "86_group_weighted_mechanism_screen_metrics.png")


def plot_pooled_prediction(pred_df: pd.DataFrame, summary_df: pd.DataFrame):
    ranking = summary_df.sort_values(
        ["Pooled_RMSE", "Pooled_MAE", "Pooled_R2"],
        ascending=[True, True, False],
    )

    methods = list(ranking["Method"].astype(str).head(4))

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 10.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, method in zip(axes, methods):
        sub = pred_df[pred_df["Method"] == method].copy()

        if len(sub) > 900:
            sub_plot = sub.sample(900, random_state=42)
        else:
            sub_plot = sub

        color = COLORS.get(method, COLORS["gray"])

        ax.scatter(
            sub_plot["Observed"],
            sub_plot["Predicted"],
            s=16,
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
            0.92,
            f"$R^2$ = {metrics['R2']:.3f}\nRMSE = {metrics['RMSE']:.2f}\nMAE = {metrics['MAE']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.80,
                boxstyle="round,pad=0.3",
            )
        )

        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Observed F removal (%)")
        ax.set_ylabel("Predicted F removal (%)")
        ax.grid(color=COLORS["grid"], linewidth=0.6, alpha=0.7)

    fig.suptitle(
        "Pooled predicted-vs-observed plots for grouped-screen top candidates",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])

    save_fig(fig, "87_group_weighted_mechanism_screen_pooled_predicted_vs_actual.png")


def collect_weight_rows(test_df: pd.DataFrame):
    rows = []

    for _, row in test_df.iterrows():
        weights = parse_weight_text(row["Weights"])

        if not weights:
            continue

        for base_model, weight in weights.items():
            rows.append({
                "Seed": row["Seed"],
                "Method": row["Method"],
                "BaseModel": base_model,
                "Weight": float(weight),
            })

    return pd.DataFrame(rows)


def plot_weight_stability(test_df: pd.DataFrame):
    weight_df = collect_weight_rows(test_df)

    if weight_df.empty:
        log("[WARN] No weights found. Skip weight stability figure.")
        return

    methods = [
        "Selected_Weighted_4Model",
        "CatBoost_HGB_Weighted_2Model",
        "RF_CatBoost_Weighted",
    ]

    methods = [m for m in methods if m in weight_df["Method"].unique()]

    if len(methods) == 0:
        log("[WARN] No weighted methods found. Skip weight stability figure.")
        return

    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(5.4 * len(methods), 4.5),
        sharey=True,
    )

    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        sub = weight_df[weight_df["Method"] == method].copy()

        for base_model, sub_m in sub.groupby("BaseModel"):
            sub_m = sub_m.sort_values("Seed")
            ax.plot(
                sub_m["Seed"],
                sub_m["Weight"],
                marker="o",
                linewidth=1.5,
                markersize=4,
                color=COLORS.get(base_model, COLORS["gray"]),
                label=MODEL_LABELS.get(base_model, base_model),
            )

            for _, r in sub_m.iterrows():
                ax.text(
                    r["Seed"],
                    r["Weight"] + 0.025,
                    f"{r['Weight']:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )

        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xlabel("Random seed")
        ax.set_ylim(-0.05, 1.12)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        ax.legend(frameon=False, loc="best")

    axes[0].set_ylabel("Validation-optimized fusion weight")

    fig.suptitle(
        "Fusion weight stability under grouped validation",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.92])

    save_fig(fig, "88_group_weighted_mechanism_screen_weight_stability.png")


# ============================================================
# 10. Main
# ============================================================

def main():
    log("\n" + "=" * 100)
    log("Step 31D: grouped validation with mechanism-aware features")
    log("=" * 100)
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"MODEL_READY_PATH = {MODEL_READY_PATH}")
    log(f"MODEL_READY_WITH_SOURCE_PATH = {MODEL_READY_WITH_SOURCE_PATH}")
    log(f"SEEDS = {SEEDS}")
    log("=" * 100)

    if not MODEL_READY_PATH.exists():
        raise FileNotFoundError(f"Cannot find {MODEL_READY_PATH}")

    # Strict check: current model_ready_v1.csv must be the 26-column legacy structure.
    model_ready_df = read_csv_smart(MODEL_READY_PATH)

    if model_ready_df.shape[1] != 26:
        raise ValueError(
            f"Current model_ready_v1.csv has {model_ready_df.shape[1]} columns, "
            "but the strict final dataset should have 26 columns. "
            "Run 29c exact legacy rebuild first."
        )

    if "文献编号" in model_ready_df.columns:
        raise ValueError("文献编号 must not be included in model_ready_v1.csv.")

    log(f"[OK] model_ready_v1.csv shape = {model_ready_df.shape}")

    # Use script 19 to build the exact feature table.
    log("\nPreparing feature sets by reusing script 19...")
    y, X_raw, X_eng, X_mix = sf19.prepare_feature_sets()

    y = y.reset_index(drop=True)
    X_mix = X_mix.reset_index(drop=True)

    log(f"[OK] y length = {len(y)}")
    log(f"[OK] Original script-19 X_mix shape = {X_mix.shape}")

    log("\nIntegrating Step-31B mechanism-aware descriptors...")
    X_mix = append_mechanism31b_features(X_mix=X_mix, expected_n=len(y))

    groups = load_group_labels(expected_n=len(y))

    group_dist = (
        groups.value_counts()
        .rename_axis("文献编号")
        .reset_index(name="n_rows")
    )
    group_dist["fraction"] = group_dist["n_rows"] / len(groups)

    all_splits = []
    all_candidates = []
    all_tests = []
    all_predictions = []

    for seed in SEEDS:
        result = run_one_seed(
            X_mix=X_mix,
            y=y,
            groups=groups,
            seed=seed,
        )

        all_splits.append(result["split"])
        all_candidates.append(result["candidate"])
        all_tests.append(result["test"])
        all_predictions.append(result["prediction"])

    split_df = pd.concat(all_splits, ignore_index=True)
    candidate_df = pd.concat(all_candidates, ignore_index=True)
    test_df = pd.concat(all_tests, ignore_index=True)
    pred_df = pd.concat(all_predictions, ignore_index=True)

    summary_df = summarize_results(test_df, pred_df)

    # Save tables.
    split_path = TABLE_DIR / "31d_grouped_mechanism_screen_split_summary.csv"
    candidate_path = TABLE_DIR / "31d_grouped_mechanism_screen_validation_candidates.csv"
    test_path = TABLE_DIR / "31d_grouped_mechanism_screen_test_results.csv"
    pred_path = TABLE_DIR / "31d_grouped_mechanism_screen_predictions.csv"
    summary_path = TABLE_DIR / "31d_grouped_mechanism_screen_summary.csv"

    split_df.to_csv(split_path, index=False, encoding="utf-8-sig")
    candidate_df.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    log("\n[OK] Tables saved:")
    for p in [split_path, candidate_path, test_path, pred_path, summary_path]:
        log(f"  - {p}")

    log("\n=== Grouped validation with mechanism-aware features summary ===")
    print(summary_df.to_string(index=False))

    write_decision_report(
        summary_df=summary_df,
        split_df=split_df,
        group_dist=group_dist,
    )

    log("\nGenerating figures...")
    plot_metrics(summary_df)
    plot_pooled_prediction(pred_df, summary_df)
    plot_weight_stability(test_df)

    log("\n" + "=" * 100)
    log("[DONE] Grouped validation with mechanism-aware features completed.")
    log("=" * 100)
    log("Next:")
    log("Send these results:")
    log("1. outputs/tables/31d_grouped_mechanism_screen_summary.csv")
    log("2. outputs/tables/31d_grouped_mechanism_screen_decision.txt")
    log("3. outputs/figures/86_group_weighted_mechanism_screen_metrics.png")
    log("4. outputs/figures/88_group_weighted_mechanism_screen_weight_stability.png")
    log("=" * 100)


if __name__ == "__main__":
    main()