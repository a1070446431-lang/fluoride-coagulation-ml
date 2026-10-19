# -*- coding: utf-8 -*-
"""
23_ablation_2model_vs_4model.py

Ablation experiment for the fluoride coagulation ML project.

Goal
----
Test whether the final Selected Weighted 4Model can be simplified to a
CatBoost + HGB 2-model weighted ensemble.

Compared methods
----------------
1. CatBoost
2. HGB
3. Weighted 2Model: CatBoost + HGB, with its own selected feature subset
4. Weighted 2Model same-K: CatBoost + HGB using the same selected feature number as 4Model
5. Weighted 4Model: RF + CatBoost + HGB + ExtraTrees

Protocol
--------
For each random seed:
    1. Split data into train/validation/test = 60/20/20
    2. Fit preprocessing only on train set
    3. Rank encoded features using ExtraTrees on train set
    4. Search selected feature number K using validation RMSE
    5. Search non-negative ensemble weights using validation RMSE
    6. Evaluate final selected models on test set only

Important
---------
The test set is never used for selecting K or ensemble weights.

Outputs
-------
Tables:
    outputs/tables/23_ablation_seed_results.csv
    outputs/tables/23_ablation_summary.csv
    outputs/tables/23_ablation_predictions.csv
    outputs/tables/23_ablation_weights.csv
    outputs/tables/23_ablation_validation_candidates.csv
    outputs/tables/23_ablation_selected_features.csv

Figures:
    outputs/figures/69_ablation_metrics_2model_vs_4model.png/pdf
    outputs/figures/70_ablation_weight_stability.png/pdf
    outputs/figures/71_ablation_pooled_prediction_2model_vs_4model.png/pdf
"""

from pathlib import Path
import re
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

warnings.filterwarnings("ignore")


# ============================================================
# 1. Paths and global settings
# ============================================================

def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    candidates = [current, current.parent, current.parent.parent]

    for p in candidates:
        if (p / "data").exists() and (p / "outputs").exists():
            return p

    if current.name.lower() == "src":
        return current.parent

    raise FileNotFoundError("Cannot locate project root.")


PROJECT_ROOT = find_project_root()

DATA_PATH_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "model_ready_v1.csv",
    PROJECT_ROOT / "data" / "processed" / "model_ready-v1.csv",
    PROJECT_ROOT / "data" / "processed" / "model_ready.csv",
]

DATA_PATH = None
for p in DATA_PATH_CANDIDATES:
    if p.exists():
        DATA_PATH = p
        break

if DATA_PATH is None:
    raise FileNotFoundError("No model-ready data file found in data/processed/.")

FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [7, 21, 42, 66, 88]
RANDOM_STATE = 42

# Candidate selected encoded-feature numbers.
# The script will automatically remove values larger than the actual encoded feature count.
K_CANDIDATES = [8, 12, 16, 20, 24, 32, 40, 50, 64]

# Weight grid settings
WEIGHT_STEP_2MODEL = 0.01
WEIGHT_STEP_4MODEL = 0.05


# ============================================================
# 2. Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.linewidth": 1.0,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

COLORS = {
    "CatBoost": "#9B72B0",
    "HGB": "#62A77B",
    "Weighted 2Model": "#4FA6A6",
    "Weighted 2Model same-K": "#7AA6D8",
    "Weighted 4Model": "#E28E68",
    "RF": "#8172B3",
    "ExtraTrees": "#A66A5D",
    "gray": "#666666",
    "dark": "#3A3A3A",
    "magenta": "#E84BA5",
    "red": "#FF0000",
    "grid": "#E5E5E5",
}


# ============================================================
# 3. Basic utilities
# ============================================================

def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Cannot read csv file: {path}")


def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("²", "2")
    s = s.replace("％", "%")
    s = s.replace("%", "pct")
    s = s.replace("−", "-")
    s = s.replace("×", "x")
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def safe_numeric(x) -> pd.Series:
    return pd.to_numeric(pd.Series(x), errors="coerce")


def convert_target_to_percent(y: pd.Series) -> pd.Series:
    y = safe_numeric(y)
    valid = y.dropna()

    if len(valid) == 0:
        return y

    if valid.quantile(0.99) <= 1.5:
        print("[INFO] Target appears to be fraction scale. Converted to percentage.")
        return y * 100.0

    return y


def metric_values(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return np.nan, np.nan, np.nan, np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)

    return r2, rmse, mae, mse


def rmse_score(y_true, y_pred) -> float:
    return metric_values(y_true, y_pred)[1]


def weights_to_string(model_names, weights) -> str:
    return "; ".join([f"{m}={w:.3f}" for m, w in zip(model_names, weights)])


# ============================================================
# 4. Target detection and leakage filtering
# ============================================================

def is_likely_derived_removal_column(col: str) -> bool:
    """
    Identify removal-related columns that are derived or predicted values.

    Important:
    Do NOT treat '氟去除率_pct' as derived.
    """
    n = normalize_name(col)

    # True target column should not be excluded
    true_target_names = [
        "氟去除率_pct",
        "氟去除率",
        "除氟率",
        "removal_pct",
        "f_removal_pct",
        "fluoride_removal_pct",
    ]

    if n in [normalize_name(x) for x in true_target_names]:
        return False

    derived_keywords = [
        "反算",
        "出液度反算",
        "预测",
        "pred",
        "prediction",
        "predicted",
        "residual",
        "残差",
        "error",
        "calculated_removal",
        "estimated_removal",
    ]

    return any(k in n for k in derived_keywords)


def detect_target_column(df: pd.DataFrame) -> str:
    """
    Detect the true fluoride-removal target column.

    Priority:
    1. Prefer 氟去除率_pct
    2. Avoid derived columns such as 出液度反算去除率_p
    """
    exact_candidates = [
        "氟去除率_pct",
        "氟去除率",
        "氟去除率 (%)",
        "氟去除率(%)",
        "除氟率",
        "除氟率 (%)",
        "除氟率(%)",
        "真实氟去除率_pct",
        "真实氟去除率",
        "F removal",
        "F removal (%)",
        "F_removal_pct",
        "Fluoride removal",
        "Fluoride removal (%)",
        "Removal",
        "Removal (%)",
        "Removal rate",
        "Removal rate (%)",
        "target",
        "y",
    ]

    # 1. Exact match first, and do not accidentally exclude 氟去除率_pct
    for cand in exact_candidates:
        cand_norm = normalize_name(cand)

        for c in df.columns:
            c_norm = normalize_name(c)

            if c_norm == cand_norm:
                vals = safe_numeric(df[c])
                if vals.notna().sum() >= 10:
                    print(f"[OK] Target column detected by exact match: {c}")
                    return c

    # 2. Prefer columns containing 氟去除率 / 除氟率, but avoid derived columns
    candidates = []

    for c in df.columns:
        n = normalize_name(c)

        if is_likely_derived_removal_column(c):
            continue

        vals = safe_numeric(df[c])
        valid_n = vals.notna().sum()

        if valid_n < 10:
            continue

        score = 0

        if "氟去除率" in n:
            score += 100
        if "除氟率" in n:
            score += 90
        if "removal" in n:
            score += 70
        if "pct" in n or "percent" in n:
            score += 20
        if "真实" in n or "true" in n:
            score += 10

        # Avoid suspicious columns
        if "反算" in n or "预测" in n or "pred" in n:
            score -= 100

        if score > 0:
            candidates.append((score, c, valid_n))

    if candidates:
        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
        selected = candidates[0][1]
        print(f"[OK] Target column detected by fuzzy priority: {selected}")
        return selected

    # 3. Debug information if still failed
    print("[ERROR] Target column was not detected.")
    print("[INFO] Available columns are:")
    for c in df.columns:
        print(f"    - {c}")

    raise ValueError(
        "Target column not found. Please check removal-rate column name. "
        "Expected something like '氟去除率_pct'."
    )

def is_bad_feature_column(col: str, target_col: str) -> bool:
    if col == target_col:
        return True

    n = normalize_name(col)

    bad_keywords = [
        # prediction / model-result columns
        "pred", "prediction", "预测",
        "residual", "残差",
        "error", "误差",
        "r2", "rmse", "mae", "mse",
        "score", "rank",

        # target / target-derived columns
        "氟去除率",
        "除氟率",
        "去除率",
        "removal",
        "反算",

        # effluent fluoride concentration is output information
        "出水氟",
        "出液氟",
        "effluent_f",
        "outlet_f",
        "final_f",
        "residual_f",
        "残余氟",

        # metadata
        "原始行号",
        "行号",
        "编号",
        "序号",
        "样本编号",
        "数据编号",
        "source",
        "paper",
        "doi",
        "year",
        "文献",
        "来源",
        "年份",
    ]

    return any(k in n for k in bad_keywords)


# ============================================================
# 5. Feature engineering
# ============================================================

def find_col(df: pd.DataFrame, keywords):
    """
    Find the first column whose normalized name contains any keyword.
    """
    for c in df.columns:
        n = normalize_name(c)
        for k in keywords:
            if normalize_name(k) in n:
                return c
    return None


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add compact mechanism-based engineered features.

    These are used for ablation because the current strongest model path
    is feature selection + weighted fusion, not raw variables only.
    """
    out = df.copy()

    dose_col = find_col(out, ["剂量_mM", "dose_mm", "dose"])
    init_f_col = find_col(out, ["初始氟_mg_L", "initial_f", "influent_f", "进水氟"])
    ph_col = find_col(out, ["进水pH", "influent_ph", "initial_ph"])
    temp_col = find_col(out, ["水温_C", "temperature", "temp"])
    rt_col = find_col(out, ["停留时间_min", "retention", "residence"])
    ion_conc_col = find_col(out, ["干扰离子浓度_mmol_L", "interfering", "ion_conc"])
    free_f_col = find_col(out, ["游离氟占比", "free_fluoride_fraction"])

    if dose_col is not None:
        dose = safe_numeric(out[dose_col])
        out["Eng_log1p_dose_mM"] = np.log1p(np.clip(dose, a_min=0, a_max=None))

    if init_f_col is not None:
        init_f = safe_numeric(out[init_f_col])
        out["Eng_log1p_initial_F"] = np.log1p(np.clip(init_f, a_min=0, a_max=None))

    if dose_col is not None and init_f_col is not None:
        dose = safe_numeric(out[dose_col])
        init_f = safe_numeric(out[init_f_col])
        out["Eng_relative_dosing_strength"] = dose / (init_f + 1e-6)
        out["Eng_dose_x_initial_F"] = dose * init_f

    if ph_col is not None:
        ph = safe_numeric(out[ph_col])
        out["Eng_pH_distance_to_neutral"] = np.abs(ph - 7.0)
        out["Eng_pH_weak_acidic"] = ((ph >= 5.0) & (ph < 7.0)).astype(float)
        out["Eng_pH_near_neutral"] = ((ph >= 6.5) & (ph <= 7.5)).astype(float)
        out["Eng_pH_alkaline"] = (ph > 7.5).astype(float)

    if ph_col is not None and dose_col is not None:
        ph = safe_numeric(out[ph_col])
        dose = safe_numeric(out[dose_col])
        out["Eng_pH_x_dose"] = ph * dose

    if ph_col is not None and init_f_col is not None:
        ph = safe_numeric(out[ph_col])
        init_f = safe_numeric(out[init_f_col])
        out["Eng_pH_x_initial_F"] = ph * init_f

    if rt_col is not None:
        rt = safe_numeric(out[rt_col])
        out["Eng_log1p_retention_time"] = np.log1p(np.clip(rt, a_min=0, a_max=None))

    if ion_conc_col is not None:
        ion_conc = safe_numeric(out[ion_conc_col])
        out["Eng_log1p_interfering_ion_conc"] = np.log1p(np.clip(ion_conc, a_min=0, a_max=None))
        out["Eng_has_interfering_ion"] = (ion_conc > 0).astype(float)

    if free_f_col is not None:
        free_f = safe_numeric(out[free_f_col])
        out["Eng_free_fluoride_fraction"] = free_f

    return out


def build_model_table(df: pd.DataFrame):
    target_col = detect_target_column(df)
    df = add_engineered_features(df)

    y = convert_target_to_percent(df[target_col])
    valid_mask = y.notna()

    df = df.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    feature_cols = []
    for c in df.columns:
        if is_bad_feature_column(c, target_col):
            continue

        # Avoid duplicate raw-original columns when cleaned columns exist
        n = normalize_name(c)
        if "原始" in n or "_original" in n:
            continue

        feature_cols.append(c)

    X = df[feature_cols].copy()

    # Remove all-empty columns
    keep_cols = []
    for c in X.columns:
        if X[c].notna().sum() > 0:
            keep_cols.append(c)

    X = X[keep_cols].copy()

    print(f"[OK] Target column: {target_col}")
    print(f"[OK] Feature columns before encoding: {len(X.columns)}")
    print("[INFO] Raw/engineered feature columns used:")
    for c in X.columns:
        print(f"    - {c}")

    return X, y, target_col


# ============================================================
# 6. Preprocessing and feature ranking
# ============================================================

def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def split_numeric_categorical(X: pd.DataFrame):
    numeric_cols = []
    categorical_cols = []

    for c in X.columns:
        vals = pd.to_numeric(X[c], errors="coerce")
        ratio = vals.notna().mean()

        if ratio >= 0.80:
            numeric_cols.append(c)
            X[c] = vals
        else:
            categorical_cols.append(c)
            X[c] = X[c].astype(str).replace({"nan": np.nan, "None": np.nan})

    return X, numeric_cols, categorical_cols


def make_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder())
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No valid numeric or categorical feature columns found.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


def get_encoded_feature_names(preprocessor, numeric_cols, categorical_cols):
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
        names.extend([f"num__{c}" for c in numeric_cols])

        if categorical_cols:
            cat_pipe = preprocessor.named_transformers_.get("cat", None)
            if cat_pipe is not None:
                try:
                    ohe = cat_pipe.named_steps["onehot"]
                    cat_names = list(ohe.get_feature_names_out(categorical_cols))
                    names.extend([f"cat__{x}" for x in cat_names])
                except Exception:
                    names.extend([f"cat__{c}" for c in categorical_cols])

        return names


def rank_features_by_extratrees(X_train_enc, y_train, feature_names, seed):
    ranker = ExtraTreesRegressor(
        n_estimators=600,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=1,
        max_features=0.8
    )

    ranker.fit(X_train_enc, y_train)

    importances = np.asarray(ranker.feature_importances_, dtype=float)

    if len(importances) != X_train_enc.shape[1]:
        raise ValueError("Feature importance length does not match encoded feature count.")

    order = np.argsort(importances)[::-1]

    ranking_df = pd.DataFrame({
        "encoded_feature": np.array(feature_names)[order],
        "importance": importances[order],
        "rank": np.arange(1, len(order) + 1)
    })

    return order, ranking_df


def build_k_grid(n_features: int):
    ks = [k for k in K_CANDIDATES if k <= n_features]
    ks.append(n_features)
    ks = sorted(set(ks))
    return ks


# ============================================================
# 7. Models and weight optimization
# ============================================================

def make_base_models(seed):
    if CatBoostRegressor is None:
        raise ImportError(
            "CatBoost is not installed in the current environment. "
            "Please install catboost before running this script."
        )

    models = {
        "RF": RandomForestRegressor(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=0.7,
            bootstrap=True
        ),

        "CatBoost": CatBoostRegressor(
            iterations=800,
            learning_rate=0.06,
            depth=6,
            l2_leaf_reg=3.0,
            bagging_temperature=1.0,
            random_strength=5.0,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1
        ),

        "HGB": HistGradientBoostingRegressor(
            max_iter=600,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            random_state=seed
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=600,
            random_state=seed,
            n_jobs=-1,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=0.8,
            bootstrap=False
        ),
    }

    return models


def fit_predict_base_models(X_train, y_train, X_val, y_val, X_test, seed):
    models = make_base_models(seed)

    val_preds = {}
    test_preds = {}

    for name, model in models.items():
        print(f"        fitting {name} ...")

        if name == "CatBoost":
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
                early_stopping_rounds=80,
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        val_preds[name] = model.predict(X_val)
        test_preds[name] = model.predict(X_test)

    return val_preds, test_preds


def simplex_weight_grid(n_models: int, step: float):
    """
    Generate non-negative weights that sum to 1.
    """
    units = int(round(1.0 / step))

    if n_models == 2:
        for i in range(units + 1):
            yield np.array([i / units, (units - i) / units], dtype=float)
        return

    def compositions(total, parts):
        if parts == 1:
            yield [total]
        else:
            for i in range(total + 1):
                for rest in compositions(total - i, parts - 1):
                    yield [i] + rest

    for comp in compositions(units, n_models):
        yield np.array(comp, dtype=float) / units


def optimize_weights(y_val, pred_matrix, model_names, step):
    """
    pred_matrix shape: n_samples x n_models
    """
    y_val = np.asarray(y_val, dtype=float)
    pred_matrix = np.asarray(pred_matrix, dtype=float)

    best_rmse = np.inf
    best_w = None
    best_pred = None

    for w in simplex_weight_grid(len(model_names), step):
        pred = pred_matrix @ w
        cur_rmse = rmse_score(y_val, pred)

        if cur_rmse < best_rmse:
            best_rmse = cur_rmse
            best_w = w
            best_pred = pred

    return best_w, best_rmse, best_pred


# ============================================================
# 8. Main ablation workflow for one seed
# ============================================================

def run_one_seed(X: pd.DataFrame, y: pd.Series, seed: int):
    print("=" * 100)
    print(f"[SEED {seed}] Starting ablation experiment")
    print("=" * 100)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        random_state=seed
    )

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train, numeric_cols, categorical_cols = split_numeric_categorical(X_train)
    X_val, _, _ = split_numeric_categorical(X_val)
    X_test, _, _ = split_numeric_categorical(X_test)

    preprocessor = make_preprocessor(numeric_cols, categorical_cols)

    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)
    X_test_enc = preprocessor.transform(X_test)

    X_train_enc = np.asarray(X_train_enc, dtype=float)
    X_val_enc = np.asarray(X_val_enc, dtype=float)
    X_test_enc = np.asarray(X_test_enc, dtype=float)

    encoded_feature_names = get_encoded_feature_names(preprocessor, numeric_cols, categorical_cols)

    print(f"[SEED {seed}] Encoded feature count: {X_train_enc.shape[1]}")

    rank_order, ranking_df = rank_features_by_extratrees(
        X_train_enc,
        y_train,
        encoded_feature_names,
        seed
    )

    k_grid = build_k_grid(X_train_enc.shape[1])
    print(f"[SEED {seed}] K candidates: {k_grid}")

    best_2 = {
        "val_rmse": np.inf,
        "K": None,
        "weights": None,
        "val_preds": None,
        "test_preds": None,
        "base_val_preds": None,
        "base_test_preds": None,
        "selected_idx": None
    }

    best_4 = {
        "val_rmse": np.inf,
        "K": None,
        "weights": None,
        "val_preds": None,
        "test_preds": None,
        "base_val_preds": None,
        "base_test_preds": None,
        "selected_idx": None,
        "same_k_2_weights": None,
        "same_k_2_val_rmse": None,
        "same_k_2_test_pred": None
    }

    candidate_rows = []

    for K in k_grid:
        print(f"[SEED {seed}] Testing selected K = {K}")

        selected_idx = rank_order[:K]

        Xtr_k = X_train_enc[:, selected_idx]
        Xva_k = X_val_enc[:, selected_idx]
        Xte_k = X_test_enc[:, selected_idx]

        val_preds, test_preds = fit_predict_base_models(
            Xtr_k,
            y_train,
            Xva_k,
            y_val,
            Xte_k,
            seed
        )

        # 2Model: CatBoost + HGB
        two_names = ["CatBoost", "HGB"]
        two_val_mat = np.column_stack([val_preds[m] for m in two_names])
        two_test_mat = np.column_stack([test_preds[m] for m in two_names])

        w2, rmse2, _ = optimize_weights(
            y_val,
            two_val_mat,
            two_names,
            step=WEIGHT_STEP_2MODEL
        )

        pred2_test = two_test_mat @ w2

        # 4Model: RF + CatBoost + HGB + ExtraTrees
        four_names = ["RF", "CatBoost", "HGB", "ExtraTrees"]
        four_val_mat = np.column_stack([val_preds[m] for m in four_names])
        four_test_mat = np.column_stack([test_preds[m] for m in four_names])

        w4, rmse4, _ = optimize_weights(
            y_val,
            four_val_mat,
            four_names,
            step=WEIGHT_STEP_4MODEL
        )

        pred4_test = four_test_mat @ w4

        candidate_rows.append({
            "Seed": seed,
            "K": K,
            "Val_RMSE_2Model": rmse2,
            "Weights_2Model": weights_to_string(two_names, w2),
            "Val_RMSE_4Model": rmse4,
            "Weights_4Model": weights_to_string(four_names, w4),
        })

        if rmse2 < best_2["val_rmse"]:
            best_2.update({
                "val_rmse": rmse2,
                "K": K,
                "weights": w2,
                "test_preds": pred2_test,
                "base_val_preds": {m: val_preds[m] for m in two_names},
                "base_test_preds": {m: test_preds[m] for m in two_names},
                "selected_idx": selected_idx
            })

        if rmse4 < best_4["val_rmse"]:
            # also store two-model result under this same K
            best_4.update({
                "val_rmse": rmse4,
                "K": K,
                "weights": w4,
                "test_preds": pred4_test,
                "base_val_preds": {m: val_preds[m] for m in four_names},
                "base_test_preds": {m: test_preds[m] for m in four_names},
                "selected_idx": selected_idx,
                "same_k_2_weights": w2,
                "same_k_2_val_rmse": rmse2,
                "same_k_2_test_pred": pred2_test
            })

    print(f"[SEED {seed}] Best 2Model K = {best_2['K']}, Val RMSE = {best_2['val_rmse']:.4f}")
    print(f"[SEED {seed}] Best 4Model K = {best_4['K']}, Val RMSE = {best_4['val_rmse']:.4f}")

    # --------------------------------------------------------
    # Build predictions for this seed
    # --------------------------------------------------------
    y_test_arr = np.asarray(y_test, dtype=float)

    prediction_rows = []

    method_preds = {
        "CatBoost": best_2["base_test_preds"]["CatBoost"],
        "HGB": best_2["base_test_preds"]["HGB"],
        "Weighted 2Model": best_2["test_preds"],
        "Weighted 2Model same-K": best_4["same_k_2_test_pred"],
        "Weighted 4Model": best_4["test_preds"],
    }

    for method, pred in method_preds.items():
        for obs, yp in zip(y_test_arr, pred):
            prediction_rows.append({
                "Seed": seed,
                "Method": method,
                "Observed": obs,
                "Predicted": yp
            })

    # --------------------------------------------------------
    # Metrics for this seed
    # --------------------------------------------------------
    metric_rows = []

    for method, pred in method_preds.items():
        r2, rmse, mae, mse = metric_values(y_test_arr, pred)

        if method == "Weighted 2Model":
            selected_k = best_2["K"]
            val_rmse = best_2["val_rmse"]
            weight_text = weights_to_string(["CatBoost", "HGB"], best_2["weights"])
        elif method == "Weighted 2Model same-K":
            selected_k = best_4["K"]
            val_rmse = best_4["same_k_2_val_rmse"]
            weight_text = weights_to_string(["CatBoost", "HGB"], best_4["same_k_2_weights"])
        elif method == "Weighted 4Model":
            selected_k = best_4["K"]
            val_rmse = best_4["val_rmse"]
            weight_text = weights_to_string(["RF", "CatBoost", "HGB", "ExtraTrees"], best_4["weights"])
        else:
            selected_k = best_2["K"]
            val_rmse = np.nan
            weight_text = ""

        metric_rows.append({
            "Seed": seed,
            "Method": method,
            "Selected_K": selected_k,
            "Validation_RMSE": val_rmse,
            "Weights": weight_text,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MSE": mse,
            "n_test": len(y_test_arr)
        })

    # --------------------------------------------------------
    # Weight rows
    # --------------------------------------------------------
    weight_rows = []

    for model_name, w in zip(["CatBoost", "HGB"], best_2["weights"]):
        weight_rows.append({
            "Seed": seed,
            "Ensemble": "Weighted 2Model",
            "Selected_K": best_2["K"],
            "Model": model_name,
            "Weight": w
        })

    for model_name, w in zip(["CatBoost", "HGB"], best_4["same_k_2_weights"]):
        weight_rows.append({
            "Seed": seed,
            "Ensemble": "Weighted 2Model same-K",
            "Selected_K": best_4["K"],
            "Model": model_name,
            "Weight": w
        })

    for model_name, w in zip(["RF", "CatBoost", "HGB", "ExtraTrees"], best_4["weights"]):
        weight_rows.append({
            "Seed": seed,
            "Ensemble": "Weighted 4Model",
            "Selected_K": best_4["K"],
            "Model": model_name,
            "Weight": w
        })

    # --------------------------------------------------------
    # Selected features
    # --------------------------------------------------------
    selected_feature_rows = []

    for ensemble_name, best_obj in [
        ("Weighted 2Model", best_2),
        ("Weighted 4Model", best_4),
    ]:
        selected_idx = best_obj["selected_idx"]
        selected_names = np.array(encoded_feature_names)[selected_idx]

        for rank_i, feat in enumerate(selected_names, start=1):
            selected_feature_rows.append({
                "Seed": seed,
                "Ensemble": ensemble_name,
                "Selected_K": best_obj["K"],
                "Rank_in_selected_subset": rank_i,
                "Encoded_Feature": feat
            })

    return {
        "seed_results": pd.DataFrame(metric_rows),
        "predictions": pd.DataFrame(prediction_rows),
        "weights": pd.DataFrame(weight_rows),
        "candidates": pd.DataFrame(candidate_rows),
        "selected_features": pd.DataFrame(selected_feature_rows),
        "ranking": ranking_df.assign(Seed=seed)
    }


# ============================================================
# 9. Summary tables
# ============================================================

def build_summary(seed_results: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    method_order = [
        "CatBoost",
        "HGB",
        "Weighted 2Model",
        "Weighted 2Model same-K",
        "Weighted 4Model"
    ]

    for method in method_order:
        sub_seed = seed_results[seed_results["Method"] == method].copy()
        sub_pred = predictions[predictions["Method"] == method].copy()

        if sub_seed.empty or sub_pred.empty:
            continue

        pooled_r2, pooled_rmse, pooled_mae, pooled_mse = metric_values(
            sub_pred["Observed"],
            sub_pred["Predicted"]
        )

        rows.append({
            "Method": method,
            "R2_mean": sub_seed["R2"].mean(),
            "R2_std": sub_seed["R2"].std(ddof=1),
            "RMSE_mean": sub_seed["RMSE"].mean(),
            "RMSE_std": sub_seed["RMSE"].std(ddof=1),
            "MAE_mean": sub_seed["MAE"].mean(),
            "MAE_std": sub_seed["MAE"].std(ddof=1),
            "MSE_mean": sub_seed["MSE"].mean(),
            "MSE_std": sub_seed["MSE"].std(ddof=1),
            "Pooled_R2": pooled_r2,
            "Pooled_RMSE": pooled_rmse,
            "Pooled_MAE": pooled_mae,
            "Pooled_MSE": pooled_mse,
            "Mean_Selected_K": sub_seed["Selected_K"].mean(),
            "n_seed": sub_seed["Seed"].nunique(),
            "n_pooled": len(sub_pred)
        })

    return pd.DataFrame(rows)


def print_decision_hint(summary: pd.DataFrame):
    m2 = summary[summary["Method"] == "Weighted 2Model"]
    m4 = summary[summary["Method"] == "Weighted 4Model"]

    if m2.empty or m4.empty:
        return

    m2 = m2.iloc[0]
    m4 = m4.iloc[0]

    delta_r2 = m2["Pooled_R2"] - m4["Pooled_R2"]
    delta_rmse = m2["Pooled_RMSE"] - m4["Pooled_RMSE"]
    delta_mae = m2["Pooled_MAE"] - m4["Pooled_MAE"]

    print("=" * 100)
    print("[DECISION CHECK] Weighted 2Model vs Weighted 4Model")
    print(f"Delta Pooled R2   = 2Model - 4Model = {delta_r2:+.4f}")
    print(f"Delta Pooled RMSE = 2Model - 4Model = {delta_rmse:+.4f}")
    print(f"Delta Pooled MAE  = 2Model - 4Model = {delta_mae:+.4f}")
    print("-" * 100)

    if abs(delta_r2) < 0.005 and delta_rmse < 0.30 and abs(delta_mae) < 0.30:
        print("[SUGGESTION] 2Model is close enough to 4Model. It may be a good parsimonious final model.")
    elif delta_r2 >= 0 and delta_rmse <= 0:
        print("[SUGGESTION] 2Model is not worse than 4Model. Consider simplifying the final model to CatBoost-HGB fusion.")
    else:
        print("[SUGGESTION] 4Model still has measurable advantage. Keep 4Model and explain RF/ExtraTrees as automatically down-weighted candidate learners.")
    print("=" * 100)


# ============================================================
# 10. Plotting
# ============================================================

def method_display_name(method: str) -> str:
    mapping = {
        "CatBoost": "CatBoost",
        "HGB": "HGB",
        "Weighted 2Model": "Weighted 2Model",
        "Weighted 2Model same-K": "2Model same-K",
        "Weighted 4Model": "Weighted 4Model",
    }
    return mapping.get(method, method)


def plot_metrics(summary: pd.DataFrame):
    method_order = [
        "CatBoost",
        "HGB",
        "Weighted 2Model",
        "Weighted 2Model same-K",
        "Weighted 4Model"
    ]

    data = summary[summary["Method"].isin(method_order)].copy()
    data["Method"] = pd.Categorical(data["Method"], categories=method_order, ordered=True)
    data = data.sort_values("Method")

    labels = [method_display_name(m) for m in data["Method"].astype(str)]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    metrics = [
        ("Pooled_R2", "Pooled $R^2$", "higher is better"),
        ("Pooled_RMSE", "Pooled RMSE", "lower is better"),
        ("Pooled_MAE", "Pooled MAE", "lower is better"),
    ]

    bar_colors = [
        COLORS["CatBoost"],
        COLORS["HGB"],
        COLORS["Weighted 2Model"],
        COLORS["Weighted 2Model same-K"],
        COLORS["Weighted 4Model"],
    ]

    for ax, (metric_col, ylabel, note) in zip(axes, metrics):
        vals = data[metric_col].values
        x = np.arange(len(data))

        ax.bar(
            x,
            vals,
            color=bar_colors[:len(data)],
            edgecolor="none",
            alpha=0.88
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)

        ymax = np.nanmax(vals)
        ymin = np.nanmin(vals)

        if "R2" in metric_col:
            ax.set_ylim(0, min(1.0, max(0.9, ymax * 1.10)))
            text_offset = ax.get_ylim()[1] * 0.020
            fmt = "{:.3f}"
        else:
            ax.set_ylim(0, ymax * 1.22)
            text_offset = ax.get_ylim()[1] * 0.025
            fmt = "{:.2f}"

        for xi, v in zip(x, vals):
            ax.text(
                xi,
                v + text_offset,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=8
            )

        ax.text(
            0.98,
            0.96,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=COLORS["gray"]
        )

    fig.suptitle(
        "Ablation comparison between CatBoost-HGB 2Model and weighted 4Model",
        fontsize=14,
        fontweight="bold",
        y=0.995
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.995, 0.93])

    png_path = FIG_DIR / "69_ablation_metrics_2model_vs_4model.png"
    pdf_path = FIG_DIR / "69_ablation_metrics_2model_vs_4model.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[FIG] Saved: {png_path}")


def plot_weight_stability(weights: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.2), sharey=True)

    ensembles = ["Weighted 2Model", "Weighted 4Model"]
    titles = [
        "CatBoost-HGB 2Model weights",
        "RF-CatBoost-HGB-ExtraTrees 4Model weights"
    ]

    for ax, ensemble, title in zip(axes, ensembles, titles):
        sub = weights[weights["Ensemble"] == ensemble].copy()

        if sub.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No weight data for {ensemble}", ha="center", va="center")
            continue

        for model_name, g in sub.groupby("Model"):
            g = g.sort_values("Seed")
            color = COLORS.get(model_name, COLORS["gray"])

            ax.plot(
                g["Seed"],
                g["Weight"],
                marker="o",
                linewidth=2.0,
                markersize=5,
                label=model_name,
                color=color
            )

            for x, yv in zip(g["Seed"], g["Weight"]):
                ax.text(
                    x,
                    yv + 0.025,
                    f"{yv:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7
                )

        ax.set_title(title)
        ax.set_xlabel("Random seed")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
        ax.legend(frameon=False, loc="best")

    axes[0].set_ylabel("Validation-optimized fusion weight")

    fig.suptitle(
        "Weight stability in ablation ensembles",
        fontsize=14,
        fontweight="bold",
        y=0.995
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.995, 0.93])

    png_path = FIG_DIR / "70_ablation_weight_stability.png"
    pdf_path = FIG_DIR / "70_ablation_weight_stability.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[FIG] Saved: {png_path}")


def plot_pooled_prediction(predictions: pd.DataFrame):
    methods = ["Weighted 2Model", "Weighted 4Model"]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.5), sharex=True, sharey=True)

    for ax, method in zip(axes, methods):
        sub = predictions[predictions["Method"] == method].copy()

        if len(sub) > 900:
            sub_plot = sub.sample(900, random_state=42)
        else:
            sub_plot = sub

        color = COLORS.get(method, COLORS["Weighted 2Model"])

        ax.scatter(
            sub_plot["Observed"],
            sub_plot["Predicted"],
            s=15,
            alpha=0.70,
            color=color,
            edgecolor="none"
        )

        ax.plot(
            [0, 100],
            [0, 100],
            color=COLORS["red"],
            linestyle=(0, (4, 3)),
            linewidth=1.1
        )

        r2, rmse, mae, mse = metric_values(sub["Observed"], sub["Predicted"])

        ax.text(
            0.06,
            0.88,
            f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor="white",
                edgecolor="none",
                alpha=0.78
            )
        )

        ax.set_title(method_display_name(method))
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Observed F removal (%)")
        ax.set_ylabel("Predicted F removal (%)")
        ax.grid(False)

    fig.suptitle(
        "Pooled predicted-vs-observed comparison of 2Model and 4Model",
        fontsize=14,
        fontweight="bold",
        y=0.995
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.995, 0.93])

    png_path = FIG_DIR / "71_ablation_pooled_prediction_2model_vs_4model.png"
    pdf_path = FIG_DIR / "71_ablation_pooled_prediction_2model_vs_4model.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[FIG] Saved: {png_path}")


# ============================================================
# 11. Main
# ============================================================

def main():
    print("=" * 100)
    print("Starting 2Model vs 4Model ablation experiment")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data path    : {DATA_PATH}")
    print(f"Table dir    : {TABLE_DIR}")
    print(f"Figure dir   : {FIG_DIR}")
    print("=" * 100)

    df = read_csv_smart(DATA_PATH)
    print(f"[OK] Data loaded: {df.shape}")

    X, y, target_col = build_model_table(df)

    all_seed_results = []
    all_predictions = []
    all_weights = []
    all_candidates = []
    all_selected_features = []
    all_rankings = []

    for seed in SEEDS:
        result = run_one_seed(X, y, seed)

        all_seed_results.append(result["seed_results"])
        all_predictions.append(result["predictions"])
        all_weights.append(result["weights"])
        all_candidates.append(result["candidates"])
        all_selected_features.append(result["selected_features"])
        all_rankings.append(result["ranking"])

    seed_results = pd.concat(all_seed_results, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    weights = pd.concat(all_weights, ignore_index=True)
    candidates = pd.concat(all_candidates, ignore_index=True)
    selected_features = pd.concat(all_selected_features, ignore_index=True)
    rankings = pd.concat(all_rankings, ignore_index=True)

    summary = build_summary(seed_results, predictions)

    # Save tables
    seed_results_path = TABLE_DIR / "23_ablation_seed_results.csv"
    summary_path = TABLE_DIR / "23_ablation_summary.csv"
    predictions_path = TABLE_DIR / "23_ablation_predictions.csv"
    weights_path = TABLE_DIR / "23_ablation_weights.csv"
    candidates_path = TABLE_DIR / "23_ablation_validation_candidates.csv"
    selected_features_path = TABLE_DIR / "23_ablation_selected_features.csv"
    rankings_path = TABLE_DIR / "23_ablation_feature_ranking_by_seed.csv"

    seed_results.to_csv(seed_results_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    weights.to_csv(weights_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    selected_features.to_csv(selected_features_path, index=False, encoding="utf-8-sig")
    rankings.to_csv(rankings_path, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("[OK] Ablation tables saved:")
    print(f"    - {seed_results_path}")
    print(f"    - {summary_path}")
    print(f"    - {predictions_path}")
    print(f"    - {weights_path}")
    print(f"    - {candidates_path}")
    print(f"    - {selected_features_path}")
    print(f"    - {rankings_path}")
    print("=" * 100)

    print("[SUMMARY]")
    print(summary.to_string(index=False))

    print_decision_hint(summary)

    # Save figures
    plot_metrics(summary)
    plot_weight_stability(weights)
    plot_pooled_prediction(predictions)

    print("=" * 100)
    print("[DONE] 2Model vs 4Model ablation experiment completed successfully.")
    print("=" * 100)


if __name__ == "__main__":
    main()