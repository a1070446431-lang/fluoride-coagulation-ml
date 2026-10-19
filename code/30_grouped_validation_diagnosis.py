# -*- coding: utf-8 -*-
"""
30_grouped_validation_diagnosis.py

Purpose
-------
Grouped validation diagnosis for the fluoride coagulation/precipitation ML project.
This script does NOT change the model workflow and does NOT retrain the final model.
It diagnoses why grouped validation becomes difficult by checking:

1. Literature group size and target distribution.
2. Group-level train/valid/test split composition for each random seed.
3. Distribution shift between train and test for key variables:
   target, initial fluoride, coagulant dose, influent pH, etc.
4. Coagulant/interfering-ion/fluoride-form category coverage:
   whether categories in test papers are unseen in the training papers.
5. Paper-level extrapolation risk score.
6. If Step-29 grouped prediction records are available, paper-level error metrics.

Expected project files
----------------------
data/processed/model_ready_v1_with_source.csv   preferred; must contain 文献编号
or
data/processed/model_ready_v1.csv + data/raw/数据集修改最终.xlsx

Outputs
-------
outputs/tables/30_*.csv
outputs/tables/30_grouped_validation_diagnosis_report.txt
outputs/figures/69_*.png/pdf to 75_*.png/pdf

Run in PyCharm:
    Right click this file -> Run
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from scipy.stats import ks_2samp
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore")


# =============================================================================
# 0. Global configuration
# =============================================================================

SEEDS = [7, 21, 42, 66, 88]
TEST_SIZE = 0.20
VALID_SIZE_IN_TRAIN_VALID = 0.25   # 0.25 of 80% train_valid gives about 20% total validation

# Diagnostic variables. The script will identify real column names automatically.
NUMERIC_ROLE_KEYWORDS = {
    "Target_F_removal": ["氟去除率", "去除率", "removal", "re_%", "f_removal"],
    "Initial_F": ["初始", "influent", "initial", "f-", "fluoride"],
    "Dose": ["剂量", "dose", "dosage", "投加"],
    "Influent_pH": ["进水ph", "influentph", "initialph", "rawph", "ph"],
    "Temperature": ["水温", "temperature", "temp"],
    "Retention_time": ["停留", "retention", "residence", "time", "反应时间"],
    "Interfering_ion_conc": ["干扰离子浓度", "interfering", "ionconc", "coexisting"],
}

CATEGORICAL_ROLE_KEYWORDS = {
    "Coagulant": ["混凝剂", "coagulant"],
    "Interfering_ion": ["干扰离子", "interfering", "coexisting", "anion"],
    "Fluoride_form": ["氟离子分布形态", "氟形态", "fluorideform", "f_form"],
}

SOURCE_KEYWORDS = ["文献编号", "paper", "source", "serial", "serials", "literature", "doi", "study"]

COLORS = {
    "blue": "#4C78A8",
    "green": "#59A14F",
    "orange": "#F28E2B",
    "purple": "#B07AA1",
    "red": "#E15759",
    "gray": "#8C949B",
    "dark": "#222222",
    "light_gray": "#E6E8EB",
}

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


def log(msg: str = "") -> None:
    print(msg, flush=True)


# =============================================================================
# 1. Basic utilities
# =============================================================================

def find_project_root() -> Path:
    """Find project root whether this script is placed in root or src."""
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    candidates.append(Path.cwd().resolve())

    expanded = []
    for p in candidates:
        expanded.append(p)
        expanded.extend(list(p.parents))

    seen = set()
    for p in expanded:
        if p in seen:
            continue
        seen.add(p)
        if (p / "data").exists() and (p / "outputs").exists():
            return p
        if (p / "data").exists():
            return p

    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, **kwargs)

    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception:
            pass
    raise ValueError(f"Cannot read file: {path}")


def normalize_text(x) -> str:
    s = str(x).strip().lower()
    s = s.replace(" ", "").replace("_", "")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("−", "-").replace("⁻", "-")
    s = s.replace("％", "%")
    return s


def to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    def parse_one(v):
        if pd.isna(v):
            return np.nan
        text = str(v).strip()
        if text in ["", "-", "--", "nan", "NaN", "None", "无"]:
            return np.nan
        text = text.replace(",", "")
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not nums:
            return np.nan
        return float(nums[0])

    return s.apply(parse_one)


def save_fig(fig: plt.Figure, filename: str) -> None:
    png_path = FIG_DIR / filename
    pdf_path = FIG_DIR / filename.replace(".png", ".pdf")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax, fmt="{:.0f}", rotation=0, fontsize=8):
    for patch in ax.patches:
        h = patch.get_height()
        if not np.isfinite(h):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            h,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
        )


def clean_label(x: str) -> str:
    return str(x).replace("_", " ")


# =============================================================================
# 2. Load final data with source labels
# =============================================================================

def find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def find_source_col(df: pd.DataFrame) -> Optional[str]:
    exact_priority = ["文献编号", "Paper_ID", "paper_id", "source", "Source", "serials", "Serials"]
    for col in exact_priority:
        if col in df.columns:
            return col

    for col in df.columns:
        cn = normalize_text(col)
        if any(normalize_text(k) in cn for k in SOURCE_KEYWORDS):
            return col
    return None


def find_target_col(df: pd.DataFrame) -> str:
    exact_priority = ["氟去除率 (%)", "氟去除率", "F removal (%)", "F_removal", "Removal"]
    for col in exact_priority:
        if col in df.columns:
            return col

    for col in df.columns:
        cn = normalize_text(col)
        if ("去除率" in cn and "氟" in cn) or "removal" in cn:
            return col

    raise ValueError("Target column was not found. Please check the F removal column name.")


def standardize_paper_id(x) -> str:
    if pd.isna(x):
        return "Paper_Missing"
    text = str(x).strip()
    if text == "":
        return "Paper_Missing"
    # numeric literature id -> Paper_001
    try:
        val = float(text)
        if np.isfinite(val) and abs(val - round(val)) < 1e-9:
            return f"Paper_{int(round(val)):03d}"
    except Exception:
        pass
    nums = re.findall(r"\d+", text)
    if len(nums) == 1 and len(text) <= 20:
        return f"Paper_{int(nums[0]):03d}"
    return re.sub(r"\s+", "_", text)


def load_final_data_with_source() -> Tuple[pd.DataFrame, Path, str, str]:
    preferred = find_first_existing([
        DATA_PROCESSED_DIR / "model_ready_v1_with_source.csv",
        DATA_PROCESSED_DIR / "model_ready-v1_with_source.csv",
        DATA_PROCESSED_DIR / "model_ready_with_source.csv",
        DATA_PROCESSED_DIR / "model_ready_v1_source.csv",
    ])

    if preferred is not None:
        df = read_table(preferred)
        source_col = find_source_col(df)
        if source_col is None:
            raise ValueError(f"{preferred} exists but no source/paper column was found.")
        target_col = find_target_col(df)
        return df, preferred, target_col, source_col

    # Fallback: combine model_ready_v1.csv with final raw Excel source column by row order.
    model_ready = find_first_existing([
        DATA_PROCESSED_DIR / "model_ready_v1.csv",
        DATA_PROCESSED_DIR / "model_ready-v1.csv",
    ])
    raw_excel = find_first_existing([
        DATA_RAW_DIR / "数据集修改最终.xlsx",
        PROJECT_ROOT / "数据集修改最终.xlsx",
    ])

    if model_ready is None:
        raise FileNotFoundError(
            "Cannot find model_ready_v1_with_source.csv or model_ready_v1.csv under data/processed."
        )
    if raw_excel is None:
        raise FileNotFoundError(
            "Cannot find source labels. Please provide data/processed/model_ready_v1_with_source.csv "
            "or data/raw/数据集修改最终.xlsx."
        )

    df_model = read_table(model_ready)
    df_raw = read_table(raw_excel, header=1)
    raw_source_col = find_source_col(df_raw)
    if raw_source_col is None:
        raw_source_col = df_raw.columns[-1]

    if len(df_model) != len(df_raw):
        raise ValueError(
            f"Row mismatch: {model_ready.name} has {len(df_model)} rows, "
            f"but final Excel has {len(df_raw)} rows. Do not attach source labels by row order."
        )

    df = df_model.copy()
    df["文献编号"] = df_raw[raw_source_col].values
    target_col = find_target_col(df)
    return df, model_ready, target_col, "文献编号"


# =============================================================================
# 3. Column role identification
# =============================================================================

def find_column_by_role(df: pd.DataFrame, role: str, used: Optional[set] = None) -> Optional[str]:
    used = used or set()
    keywords = NUMERIC_ROLE_KEYWORDS.get(role, CATEGORICAL_ROLE_KEYWORDS.get(role, []))
    cols = [c for c in df.columns if c not in used]

    # Special rules to avoid wrong matching.
    if role == "Initial_F":
        for col in cols:
            cn = normalize_text(col)
            if ("初始" in cn or "initial" in cn or "influent" in cn) and (
                "f" in cn or "氟" in cn
            ) and "出水" not in cn and "effluent" not in cn:
                return col

    if role == "Dose":
        for col in cols:
            cn = normalize_text(col)
            if ("剂量" in cn or "dose" in cn or "dosage" in cn or "投加" in cn) and (
                "絮凝" not in cn and "flocculant" not in cn
            ):
                return col

    if role == "Influent_pH":
        for col in cols:
            cn = normalize_text(col)
            if "ph" in cn and ("进水" in cn or "influent" in cn or "initial" in cn or "raw" in cn):
                return col
        for col in cols:
            cn = normalize_text(col)
            if cn == "ph" or cn.endswith("ph"):
                return col

    if role == "Coagulant":
        for col in cols:
            cn = normalize_text(col)
            if "混凝剂" in cn or "coagulant" in cn:
                return col

    if role == "Interfering_ion":
        for col in cols:
            cn = normalize_text(col)
            if ("干扰离子" in cn or "interfering" in cn or "coexisting" in cn) and "浓度" not in cn and "conc" not in cn:
                return col

    # Generic fallback.
    for col in cols:
        cn = normalize_text(col)
        if all(normalize_text(k) in cn for k in keywords[:1]):
            return col
    for col in cols:
        cn = normalize_text(col)
        if any(normalize_text(k) in cn for k in keywords):
            return col
    return None


def identify_roles(df: pd.DataFrame, target_col: str, source_col: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    numeric_roles = {"Target_F_removal": target_col}
    used = {target_col, source_col}

    for role in ["Initial_F", "Dose", "Influent_pH", "Temperature", "Retention_time", "Interfering_ion_conc"]:
        col = find_column_by_role(df, role, used=used)
        if col is not None:
            numeric_roles[role] = col
            used.add(col)

    categorical_roles = {}
    for role in ["Coagulant", "Interfering_ion", "Fluoride_form"]:
        col = find_column_by_role(df, role, used=used)
        if col is not None:
            categorical_roles[role] = col
            used.add(col)

    return numeric_roles, categorical_roles


def prepare_analysis_dataframe(df_raw: pd.DataFrame, target_col: str, source_col: str) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    df = df_raw.copy().reset_index(drop=False).rename(columns={"index": "Original_Row_Index"})
    df["Paper_ID"] = df[source_col].apply(standardize_paper_id)

    y = to_numeric_series(df[target_col])
    valid_mask = y.notna()
    df = df.loc[valid_mask].copy().reset_index(drop=True)
    df[target_col] = y.loc[valid_mask].values

    numeric_roles, categorical_roles = identify_roles(df, target_col, source_col)

    # Create clean numeric columns for diagnostics.
    for role, col in numeric_roles.items():
        diag_col = f"Diag_{role}"
        df[diag_col] = to_numeric_series(df[col])

    for role, col in categorical_roles.items():
        diag_col = f"Diag_{role}"
        df[diag_col] = (
            df[col]
            .astype(str)
            .replace({"nan": "Missing", "NaN": "Missing", "None": "Missing", "": "Missing"})
            .fillna("Missing")
        )

    return df, numeric_roles, categorical_roles


# =============================================================================
# 4. Group split reconstruction
# =============================================================================

def make_group_splits(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    y = df["Diag_Target_F_removal"].values
    groups = df["Paper_ID"].values
    idx = np.arange(len(df))

    for seed in SEEDS:
        splitter_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_valid_idx, test_idx = next(splitter_test.split(idx, y, groups))

        train_valid_groups = groups[train_valid_idx]
        splitter_valid = GroupShuffleSplit(
            n_splits=1,
            test_size=VALID_SIZE_IN_TRAIN_VALID,
            random_state=seed,
        )
        rel_train_idx, rel_valid_idx = next(
            splitter_valid.split(train_valid_idx, y[train_valid_idx], train_valid_groups)
        )

        train_idx = train_valid_idx[rel_train_idx]
        valid_idx = train_valid_idx[rel_valid_idx]

        for split_name, part_idx in [
            ("Train", train_idx),
            ("Valid", valid_idx),
            ("Test", test_idx),
        ]:
            for i in part_idx:
                records.append({
                    "Seed": seed,
                    "Original_Row_Index": int(df.loc[i, "Original_Row_Index"]),
                    "Row_Index_After_Target_Filter": int(i),
                    "Paper_ID": df.loc[i, "Paper_ID"],
                    "Split": split_name,
                })

    return pd.DataFrame(records)


# =============================================================================
# 5. Statistics
# =============================================================================

def safe_std(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1))


def numeric_stats(x: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce")
    return {
        "n": int(x.notna().sum()),
        "missing_rate": float(x.isna().mean()),
        "mean": float(x.mean()) if x.notna().any() else np.nan,
        "std": safe_std(x),
        "median": float(x.median()) if x.notna().any() else np.nan,
        "min": float(x.min()) if x.notna().any() else np.nan,
        "max": float(x.max()) if x.notna().any() else np.nan,
        "q25": float(x.quantile(0.25)) if x.notna().any() else np.nan,
        "q75": float(x.quantile(0.75)) if x.notna().any() else np.nan,
    }


def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    return float((b.mean() - a.mean()) / pooled)


def range_overlap_ratio(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    a_min, a_max = float(a.min()), float(a.max())
    b_min, b_max = float(b.min()), float(b.max())
    union = max(a_max, b_max) - min(a_min, b_min)
    if union <= 0:
        return 1.0
    overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    return float(overlap / union)


def ks_test(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    if not SCIPY_AVAILABLE:
        return np.nan, np.nan
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    out = ks_2samp(a, b)
    return float(out.statistic), float(out.pvalue)


def split_summary(df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    merged = split_df.merge(df, on=["Original_Row_Index", "Paper_ID"], how="left")
    records = []

    for seed, seed_df in merged.groupby("Seed"):
        for split_name, part in seed_df.groupby("Split"):
            paper_count = part["Paper_ID"].nunique()
            row_count = len(part)
            rec = {
                "Seed": seed,
                "Split": split_name,
                "n_rows": row_count,
                "n_papers": paper_count,
                "row_fraction": row_count / len(df),
                "paper_fraction": paper_count / df["Paper_ID"].nunique(),
            }
            stats = numeric_stats(part["Diag_Target_F_removal"])
            rec.update({f"Target_{k}": v for k, v in stats.items()})
            records.append(rec)

    out = pd.DataFrame(records)
    order = {"Train": 0, "Valid": 1, "Test": 2}
    out["Split_Order"] = out["Split"].map(order)
    return out.sort_values(["Seed", "Split_Order"]).drop(columns="Split_Order")


def distribution_shift_table(df: pd.DataFrame, split_df: pd.DataFrame, numeric_roles: Dict[str, str]) -> pd.DataFrame:
    merged = split_df.merge(df, on=["Original_Row_Index", "Paper_ID"], how="left")
    records = []

    for seed, seed_df in merged.groupby("Seed"):
        train = seed_df[seed_df["Split"] == "Train"]
        valid = seed_df[seed_df["Split"] == "Valid"]
        test = seed_df[seed_df["Split"] == "Test"]

        for role in numeric_roles.keys():
            diag_col = f"Diag_{role}"
            if diag_col not in seed_df.columns:
                continue

            for compare_name, other in [("Valid_vs_Train", valid), ("Test_vs_Train", test)]:
                train_s = train[diag_col]
                other_s = other[diag_col]
                ks_stat, ks_p = ks_test(train_s, other_s)
                records.append({
                    "Seed": seed,
                    "Variable_Role": role,
                    "Original_Column": numeric_roles[role],
                    "Comparison": compare_name,
                    "Train_n": int(train_s.notna().sum()),
                    "Other_n": int(other_s.notna().sum()),
                    "Train_mean": float(pd.to_numeric(train_s, errors="coerce").mean()),
                    "Other_mean": float(pd.to_numeric(other_s, errors="coerce").mean()),
                    "Train_median": float(pd.to_numeric(train_s, errors="coerce").median()),
                    "Other_median": float(pd.to_numeric(other_s, errors="coerce").median()),
                    "SMD_Other_minus_Train": standardized_mean_difference(train_s, other_s),
                    "Abs_SMD": abs(standardized_mean_difference(train_s, other_s)) if np.isfinite(standardized_mean_difference(train_s, other_s)) else np.nan,
                    "Range_Overlap_Ratio": range_overlap_ratio(train_s, other_s),
                    "KS_Statistic": ks_stat,
                    "KS_pvalue": ks_p,
                })

    out = pd.DataFrame(records)
    return out


def category_coverage_table(df: pd.DataFrame, split_df: pd.DataFrame, categorical_roles: Dict[str, str]) -> pd.DataFrame:
    merged = split_df.merge(df, on=["Original_Row_Index", "Paper_ID"], how="left")
    records = []

    for seed, seed_df in merged.groupby("Seed"):
        train = seed_df[seed_df["Split"] == "Train"]
        valid = seed_df[seed_df["Split"] == "Valid"]
        test = seed_df[seed_df["Split"] == "Test"]

        for role in categorical_roles.keys():
            diag_col = f"Diag_{role}"
            if diag_col not in seed_df.columns:
                continue
            train_set = set(train[diag_col].astype(str).dropna())
            valid_set = set(valid[diag_col].astype(str).dropna())
            test_set = set(test[diag_col].astype(str).dropna())

            for compare_name, other, other_set in [
                ("Valid_vs_Train", valid, valid_set),
                ("Test_vs_Train", test, test_set),
            ]:
                unseen = sorted(list(other_set - train_set))
                other_values = other[diag_col].astype(str)
                unseen_sample_fraction = float(other_values.isin(unseen).mean()) if len(other_values) else np.nan
                records.append({
                    "Seed": seed,
                    "Variable_Role": role,
                    "Original_Column": categorical_roles[role],
                    "Comparison": compare_name,
                    "Train_n_categories": len(train_set),
                    "Other_n_categories": len(other_set),
                    "Unseen_n_categories": len(unseen),
                    "Unseen_categories": ";".join(unseen),
                    "Unseen_sample_fraction": unseen_sample_fraction,
                })
    return pd.DataFrame(records)


def paper_feature_summary(df: pd.DataFrame, numeric_roles: Dict[str, str], categorical_roles: Dict[str, str]) -> pd.DataFrame:
    records = []
    for paper, part in df.groupby("Paper_ID"):
        rec = {
            "Paper_ID": paper,
            "n_rows": len(part),
            "Target_mean": part["Diag_Target_F_removal"].mean(),
            "Target_std": part["Diag_Target_F_removal"].std(ddof=1),
            "Target_median": part["Diag_Target_F_removal"].median(),
            "Target_min": part["Diag_Target_F_removal"].min(),
            "Target_max": part["Diag_Target_F_removal"].max(),
            "Low_removal_frac_lt60": float((part["Diag_Target_F_removal"] < 60).mean()),
            "High_removal_frac_ge80": float((part["Diag_Target_F_removal"] >= 80).mean()),
        }
        for role in ["Initial_F", "Dose", "Influent_pH", "Temperature", "Retention_time", "Interfering_ion_conc"]:
            diag_col = f"Diag_{role}"
            if diag_col in part.columns:
                s = pd.to_numeric(part[diag_col], errors="coerce")
                rec[f"{role}_median"] = s.median()
                rec[f"{role}_min"] = s.min()
                rec[f"{role}_max"] = s.max()
                rec[f"{role}_missing_rate"] = s.isna().mean()
        for role in ["Coagulant", "Interfering_ion", "Fluoride_form"]:
            diag_col = f"Diag_{role}"
            if diag_col in part.columns:
                vals = sorted(part[diag_col].astype(str).dropna().unique().tolist())
                rec[f"{role}_n_categories"] = len(vals)
                rec[f"{role}_categories"] = ";".join(vals[:20])
        records.append(rec)

    out = pd.DataFrame(records).sort_values("n_rows", ascending=False).reset_index(drop=True)
    return out


def paper_split_table(split_df: pd.DataFrame) -> pd.DataFrame:
    out = split_df.groupby(["Seed", "Paper_ID", "Split"]).size().reset_index(name="n_rows")
    pivot = out.pivot_table(index=["Seed", "Paper_ID"], columns="Split", values="n_rows", fill_value=0).reset_index()
    for col in ["Train", "Valid", "Test"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["Assigned_Split"] = np.select(
        [pivot["Test"] > 0, pivot["Valid"] > 0, pivot["Train"] > 0],
        ["Test", "Valid", "Train"],
        default="Unknown",
    )
    return pivot.sort_values(["Seed", "Assigned_Split", "Paper_ID"])


def paper_extrapolation_risk(df: pd.DataFrame, split_df: pd.DataFrame, numeric_roles: Dict[str, str], categorical_roles: Dict[str, str]) -> pd.DataFrame:
    merged = split_df.merge(df, on=["Original_Row_Index", "Paper_ID"], how="left")
    records = []

    key_num_roles = [r for r in ["Target_F_removal", "Initial_F", "Dose", "Influent_pH"] if r in numeric_roles]
    key_cat_roles = [r for r in ["Coagulant", "Interfering_ion", "Fluoride_form"] if r in categorical_roles]

    for seed, seed_df in merged.groupby("Seed"):
        train = seed_df[seed_df["Split"] == "Train"]
        valid = seed_df[seed_df["Split"] == "Valid"]
        test = seed_df[seed_df["Split"] == "Test"]

        train_num_stats = {}
        for role in key_num_roles:
            col = f"Diag_{role}"
            s = pd.to_numeric(train[col], errors="coerce").dropna()
            train_num_stats[role] = {
                "mean": s.mean(),
                "std": s.std(ddof=1),
                "min": s.min(),
                "max": s.max(),
            }

        train_cat_sets = {}
        for role in key_cat_roles:
            col = f"Diag_{role}"
            train_cat_sets[role] = set(train[col].astype(str).dropna())

        for split_name, split_part in [("Valid", valid), ("Test", test)]:
            for paper, part in split_part.groupby("Paper_ID"):
                rec = {"Seed": seed, "Split": split_name, "Paper_ID": paper, "n_rows": len(part)}
                risk_score = 0.0

                for role in key_num_roles:
                    col = f"Diag_{role}"
                    s = pd.to_numeric(part[col], errors="coerce").dropna()
                    st = train_num_stats[role]
                    if len(s) == 0 or not np.isfinite(st["std"]) or st["std"] == 0:
                        z_mean = np.nan
                        out_of_range_frac = np.nan
                    else:
                        z_mean = float((s.mean() - st["mean"]) / st["std"])
                        out_of_range_frac = float(((s < st["min"]) | (s > st["max"])).mean())
                        risk_score += min(abs(z_mean), 5) * 1.0
                        risk_score += out_of_range_frac * 2.0
                    rec[f"{role}_mean_z_vs_train"] = z_mean
                    rec[f"{role}_out_of_train_range_frac"] = out_of_range_frac
                    rec[f"{role}_paper_mean"] = float(s.mean()) if len(s) else np.nan

                for role in key_cat_roles:
                    col = f"Diag_{role}"
                    vals = part[col].astype(str)
                    unseen_frac = float(~vals.isin(train_cat_sets[role]).mean()) if False else np.nan
                    # Correct calculation below. Kept explicit for readability.
                    unseen_mask = ~vals.isin(train_cat_sets[role])
                    unseen_frac = float(unseen_mask.mean()) if len(vals) else np.nan
                    unseen_categories = sorted(list(set(vals[unseen_mask]) - {"nan", "None"}))
                    rec[f"{role}_unseen_sample_frac"] = unseen_frac
                    rec[f"{role}_unseen_categories"] = ";".join(unseen_categories)
                    if np.isfinite(unseen_frac):
                        risk_score += unseen_frac * 3.0

                rec["Extrapolation_Risk_Score"] = risk_score
                records.append(rec)

    out = pd.DataFrame(records)
    return out.sort_values(["Split", "Extrapolation_Risk_Score"], ascending=[True, False])


# =============================================================================
# 6. Step-29 prediction reading and paper-level metrics
# =============================================================================

def find_prediction_file() -> Optional[Path]:
    candidates = []
    if TABLE_DIR.exists():
        for p in TABLE_DIR.glob("*.csv"):
            name = p.name.lower()
            if "29" in name and ("pred" in name or "prediction" in name):
                candidates.append(p)
        for p in TABLE_DIR.glob("*.xlsx"):
            name = p.name.lower()
            if "29" in name and ("pred" in name or "prediction" in name):
                candidates.append(p)

    priority_names = [
        "29_grouped_screen_predictions.csv",
        "29_grouped_validation_predictions.csv",
        "29_grouped_screen_prediction_records.csv",
        "29_grouped_predictions.csv",
    ]
    for name in priority_names:
        p = TABLE_DIR / name
        if p.exists():
            return p

    if candidates:
        return sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return None


def find_col_by_candidates(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        for col in df.columns:
            if normalize_text(col) == normalize_text(cand):
                return col
    for cand in candidates:
        cand_n = normalize_text(cand)
        for col in df.columns:
            if cand_n in normalize_text(col):
                return col
    return None


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    out = {
        "R2": r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan,
        "RMSE": math.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "Bias_mean_pred_minus_obs": float(np.mean(y_pred - y_true)),
        "n": len(y_true),
    }
    return out


def paper_metrics_from_predictions(df: pd.DataFrame, split_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Path], str]:
    pred_file = find_prediction_file()
    if pred_file is None:
        msg = (
            "No Step-29 prediction records were found. Paper-level metric diagnosis will be skipped.\n"
            "Expected a CSV/XLSX under outputs/tables with filename containing both '29' and 'pred'.\n"
            "If your Step-29 script only saved summary files, rerun Step-29 after saving row-level predictions "
            "or send me the Step-29 script and I will adapt this part."
        )
        return pd.DataFrame(), None, msg

    pred = read_table(pred_file)
    seed_col = find_col_by_candidates(pred, ["Seed", "seed", "随机种子"])
    method_col = find_col_by_candidates(pred, ["Method", "Model", "模型", "Kind"])
    obs_col = find_col_by_candidates(pred, ["Observed", "Actual", "y_true", "True", "实验值", "实测值"])
    pred_col = find_col_by_candidates(pred, ["Predicted", "Prediction", "y_pred", "预测值"])
    paper_col = find_col_by_candidates(pred, ["Paper_ID", "文献编号", "Paper", "Source", "source", "serials"])
    row_col = find_col_by_candidates(pred, ["Original_Row_Index", "Row_Index", "Index", "样本序号"])

    missing = []
    for name, col in [("seed", seed_col), ("observed", obs_col), ("predicted", pred_col)]:
        if col is None:
            missing.append(name)
    if missing:
        msg = f"Found prediction file {pred_file.name}, but required columns are missing: {missing}."
        return pd.DataFrame(), pred_file, msg

    pred_work = pred.copy()
    pred_work["Seed_std"] = pred_work[seed_col]
    pred_work["Observed_std"] = to_numeric_series(pred_work[obs_col])
    pred_work["Predicted_std"] = to_numeric_series(pred_work[pred_col])
    pred_work["Method_std"] = pred_work[method_col].astype(str) if method_col else "Unknown_Method"

    if paper_col is not None:
        pred_work["Paper_ID_std"] = pred_work[paper_col].apply(standardize_paper_id)
    elif row_col is not None:
        # Attach paper id using original row index.
        row_to_paper = df.set_index("Original_Row_Index")["Paper_ID"].to_dict()
        pred_work["Paper_ID_std"] = pred_work[row_col].map(row_to_paper)
    else:
        msg = (
            f"Found prediction file {pred_file.name}, but no Paper_ID/source or row index column was found. "
            "Cannot calculate paper-level metrics."
        )
        return pd.DataFrame(), pred_file, msg

    pred_work = pred_work.dropna(subset=["Observed_std", "Predicted_std", "Paper_ID_std"])

    records = []
    for (seed, method, paper), part in pred_work.groupby(["Seed_std", "Method_std", "Paper_ID_std"]):
        metrics = regression_metrics(part["Observed_std"], part["Predicted_std"])
        records.append({
            "Seed": seed,
            "Method": method,
            "Paper_ID": paper,
            **metrics,
            "Observed_mean": float(part["Observed_std"].mean()),
            "Predicted_mean": float(part["Predicted_std"].mean()),
        })

    out = pd.DataFrame(records)
    if out.empty:
        msg = f"Prediction file {pred_file.name} was read, but no valid prediction rows were available."
        return out, pred_file, msg

    return out.sort_values(["Method", "RMSE"], ascending=[True, False]), pred_file, f"Loaded prediction file: {pred_file}"


# =============================================================================
# 7. Plotting
# =============================================================================

def plot_group_size(paper_summary: pd.DataFrame):
    plot_df = paper_summary.sort_values("n_rows", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["n_rows"], color=COLORS["blue"], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Paper_ID"], rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Number of samples")
    ax.set_title("Literature group size distribution", fontsize=15, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for i, v in enumerate(plot_df["n_rows"]):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=7, rotation=90)
    fig.tight_layout()
    save_fig(fig, "69_literature_group_size_distribution.png")


def plot_paper_target_summary(paper_summary: pd.DataFrame):
    plot_df = paper_summary.sort_values("Target_mean", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    x = np.arange(len(plot_df))
    y = plot_df["Target_mean"].values
    yerr = plot_df["Target_std"].fillna(0).values
    ax.bar(x, y, yerr=yerr, capsize=2.5, color=COLORS["green"], alpha=0.84)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Paper_ID"], rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Mean F removal (%)")
    ax.set_title("Paper-level target distribution", fontsize=15, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for i, v in enumerate(y):
        if np.isfinite(v):
            ax.text(i, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
    fig.tight_layout()
    save_fig(fig, "70_paper_level_target_distribution.png")


def plot_seed_split_target_box(df: pd.DataFrame, split_df: pd.DataFrame):
    merged = split_df.merge(df, on=["Original_Row_Index", "Paper_ID"], how="left")
    fig, ax = plt.subplots(figsize=(13.5, 6.3))

    positions = []
    labels = []
    data = []
    colors = []
    split_colors = {"Train": COLORS["blue"], "Valid": COLORS["orange"], "Test": COLORS["red"]}

    pos = 1
    for seed in SEEDS:
        for split in ["Train", "Valid", "Test"]:
            vals = merged[(merged["Seed"] == seed) & (merged["Split"] == split)]["Diag_Target_F_removal"].dropna().values
            data.append(vals)
            positions.append(pos)
            labels.append(f"{seed}\n{split}")
            colors.append(split_colors[split])
            pos += 1
        pos += 0.8

    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.7, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor(COLORS["dark"])
    for med in bp["medians"]:
        med.set_color(COLORS["dark"])
        med.set_linewidth(1.2)

    # Annotate mean value above each box.
    for p, vals in zip(positions, data):
        if len(vals) == 0:
            continue
        ax.text(p, np.nanmean(vals) + 2, f"μ={np.nanmean(vals):.1f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("F removal (%)")
    ax.set_title("Target distribution under grouped train/valid/test splits", fontsize=15, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "71_grouped_split_target_distribution_by_seed.png")


def plot_numeric_shift_heatmap(shift_df: pd.DataFrame):
    plot_df = shift_df[shift_df["Comparison"] == "Test_vs_Train"].copy()
    if plot_df.empty:
        return

    # Keep important variables first.
    order = ["Target_F_removal", "Initial_F", "Dose", "Influent_pH", "Temperature", "Retention_time", "Interfering_ion_conc"]
    variables = [v for v in order if v in plot_df["Variable_Role"].unique()]
    seeds = sorted(plot_df["Seed"].unique())
    mat = np.full((len(variables), len(seeds)), np.nan)

    for i, var in enumerate(variables):
        for j, seed in enumerate(seeds):
            sub = plot_df[(plot_df["Variable_Role"] == var) & (plot_df["Seed"] == seed)]
            if not sub.empty:
                mat[i, j] = sub["SMD_Other_minus_Train"].iloc[0]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.set_yticks(np.arange(len(variables)))
    ax.set_yticklabels([clean_label(v) for v in variables])
    ax.set_xlabel("Random seed")
    ax.set_title("Train-test numeric distribution shift (SMD)", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("SMD: Test mean - Train mean")

    for i in range(len(variables)):
        for j in range(len(seeds)):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")

    fig.tight_layout()
    save_fig(fig, "72_train_test_numeric_shift_smd_heatmap.png")


def plot_category_coverage_heatmap(coverage_df: pd.DataFrame):
    plot_df = coverage_df[coverage_df["Comparison"] == "Test_vs_Train"].copy()
    if plot_df.empty:
        return

    variables = [v for v in ["Coagulant", "Interfering_ion", "Fluoride_form"] if v in plot_df["Variable_Role"].unique()]
    seeds = sorted(plot_df["Seed"].unique())
    mat = np.full((len(variables), len(seeds)), np.nan)

    for i, var in enumerate(variables):
        for j, seed in enumerate(seeds):
            sub = plot_df[(plot_df["Variable_Role"] == var) & (plot_df["Seed"] == seed)]
            if not sub.empty:
                mat[i, j] = sub["Unseen_sample_fraction"].iloc[0]

    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=max(0.5, np.nanmax(mat) if np.isfinite(mat).any() else 0.5), aspect="auto")
    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.set_yticks(np.arange(len(variables)))
    ax.set_yticklabels([clean_label(v) for v in variables])
    ax.set_xlabel("Random seed")
    ax.set_title("Unseen category fraction in grouped test papers", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of test samples with unseen categories")

    for i in range(len(variables)):
        for j in range(len(seeds)):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    save_fig(fig, "73_test_unseen_category_fraction_heatmap.png")


def plot_risk_score(risk_df: pd.DataFrame):
    plot_df = risk_df[risk_df["Split"] == "Test"].copy()
    if plot_df.empty:
        return
    # Average risk if a paper appears in test across multiple seeds.
    agg = plot_df.groupby("Paper_ID").agg(
        Mean_Risk=("Extrapolation_Risk_Score", "mean"),
        Max_Risk=("Extrapolation_Risk_Score", "max"),
        Test_Appearances=("Seed", "nunique"),
        n_rows_mean=("n_rows", "mean"),
    ).reset_index().sort_values("Mean_Risk", ascending=False).head(18)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(agg))
    ax.bar(x, agg["Mean_Risk"], color=COLORS["purple"], alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["Paper_ID"], rotation=55, ha="right")
    ax.set_ylabel("Mean extrapolation risk score")
    ax.set_title("Highest-risk test literature groups", fontsize=15, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for i, row in agg.iterrows():
        j = list(agg.index).index(i)
        ax.text(j, row["Mean_Risk"], f"{row['Mean_Risk']:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "74_high_risk_test_literature_groups.png")


def plot_paper_metrics(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        return

    # Prefer main methods if available.
    preferred_methods = [
        "Selected_CatBoost",
        "Selected_Weighted_4Model",
        "Weighted_2Model",
        "Selected_CatBoost_HGB_Weighted_2Model",
    ]
    available = metrics_df["Method"].astype(str).unique().tolist()
    method = None
    for m in preferred_methods:
        if m in available:
            method = m
            break
    if method is None:
        method = available[0]

    plot_df = metrics_df[metrics_df["Method"].astype(str) == method].copy()
    if plot_df.empty:
        return

    # Average across seeds for each paper.
    agg = plot_df.groupby("Paper_ID").agg(
        RMSE=("RMSE", "mean"),
        MAE=("MAE", "mean"),
        Bias=("Bias_mean_pred_minus_obs", "mean"),
        Observed_mean=("Observed_mean", "mean"),
        n=("n", "sum"),
    ).reset_index()
    agg = agg.sort_values("RMSE", ascending=False)

    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    sc = ax.scatter(
        agg["Observed_mean"],
        agg["RMSE"],
        s=np.clip(agg["n"].values * 8, 40, 360),
        alpha=0.72,
        color=COLORS["red"],
        edgecolor=COLORS["dark"],
        linewidth=0.5,
    )
    ax.set_xlabel("Paper-level mean observed F removal (%)")
    ax.set_ylabel("Paper-level RMSE")
    ax.set_title(f"Paper-level grouped test error diagnosis: {method}", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.25)

    for _, row in agg.head(10).iterrows():
        ax.text(row["Observed_mean"], row["RMSE"], row["Paper_ID"], fontsize=8, ha="left", va="bottom")

    fig.tight_layout()
    save_fig(fig, "75_paper_level_error_vs_target_mean.png")


# =============================================================================
# 8. Report generation
# =============================================================================

def format_top_rows(df: pd.DataFrame, cols: List[str], n: int = 8) -> str:
    if df.empty:
        return "None"
    show = df[cols].head(n).copy()
    return show.to_string(index=False)


def generate_report(
    df: pd.DataFrame,
    data_path: Path,
    target_col: str,
    source_col: str,
    numeric_roles: Dict[str, str],
    categorical_roles: Dict[str, str],
    paper_summary: pd.DataFrame,
    split_sum: pd.DataFrame,
    shift_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    pred_msg: str,
):
    n_papers = df["Paper_ID"].nunique()
    n_rows = len(df)

    group_sizes = paper_summary["n_rows"]
    largest = paper_summary.iloc[0]
    smallest = paper_summary.sort_values("n_rows").iloc[0]

    test_shift = shift_df[shift_df["Comparison"] == "Test_vs_Train"].copy()
    worst_shift = test_shift.sort_values("Abs_SMD", ascending=False).head(10) if not test_shift.empty else pd.DataFrame()

    test_cov = coverage_df[coverage_df["Comparison"] == "Test_vs_Train"].copy()
    worst_cov = test_cov.sort_values("Unseen_sample_fraction", ascending=False).head(10) if not test_cov.empty else pd.DataFrame()

    high_risk = risk_df[risk_df["Split"] == "Test"].sort_values("Extrapolation_Risk_Score", ascending=False).head(12)

    lines = []
    lines.append("30_grouped_validation_diagnosis report")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Data file: {data_path}")
    lines.append(f"Rows after valid target filtering: {n_rows}")
    lines.append(f"Number of literature groups: {n_papers}")
    lines.append(f"Target column: {target_col}")
    lines.append(f"Source column: {source_col}")
    lines.append("")
    lines.append("Identified numeric roles:")
    for k, v in numeric_roles.items():
        lines.append(f"  - {k}: {v}")
    lines.append("Identified categorical roles:")
    for k, v in categorical_roles.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Group-size diagnosis:")
    lines.append(f"  - Largest group: {largest['Paper_ID']} with n={int(largest['n_rows'])}")
    lines.append(f"  - Smallest group: {smallest['Paper_ID']} with n={int(smallest['n_rows'])}")
    lines.append(f"  - Median group size: {group_sizes.median():.1f}")
    lines.append(f"  - Mean group size: {group_sizes.mean():.1f}")
    lines.append("")
    lines.append("Split-size summary:")
    lines.append(split_sum.to_string(index=False))
    lines.append("")
    lines.append("Worst train-test numeric shifts by absolute SMD:")
    if not worst_shift.empty:
        lines.append(format_top_rows(
            worst_shift,
            ["Seed", "Variable_Role", "SMD_Other_minus_Train", "Range_Overlap_Ratio", "KS_Statistic", "KS_pvalue"],
            n=10,
        ))
    else:
        lines.append("None")
    lines.append("")
    lines.append("Worst train-test category coverage problems:")
    if not worst_cov.empty:
        lines.append(format_top_rows(
            worst_cov,
            ["Seed", "Variable_Role", "Unseen_sample_fraction", "Unseen_n_categories", "Unseen_categories"],
            n=10,
        ))
    else:
        lines.append("None")
    lines.append("")
    lines.append("Highest-risk test literature groups:")
    if not high_risk.empty:
        cols = ["Seed", "Paper_ID", "n_rows", "Extrapolation_Risk_Score"]
        extra = [c for c in high_risk.columns if c.endswith("_unseen_sample_frac")]
        cols.extend(extra[:3])
        lines.append(format_top_rows(high_risk, cols, n=12))
    else:
        lines.append("None")
    lines.append("")
    lines.append("Step-29 prediction diagnosis:")
    lines.append(pred_msg)
    if not metrics_df.empty:
        lines.append("Worst paper-level metrics:")
        lines.append(format_top_rows(
            metrics_df.sort_values("RMSE", ascending=False),
            ["Seed", "Method", "Paper_ID", "n", "R2", "RMSE", "MAE", "Bias_mean_pred_minus_obs"],
            n=15,
        ))
    lines.append("")
    lines.append("How to interpret this report:")
    lines.append("1. Large |SMD| values indicate that grouped test papers differ strongly from training papers.")
    lines.append("2. Low range-overlap ratios indicate extrapolation beyond the training range.")
    lines.append("3. High unseen-category fractions indicate that the model sees new coagulant/ion/form categories in test papers.")
    lines.append("4. Target distribution mismatch is not an input-feature problem, but it explains why validation becomes harder.")
    lines.append("5. If only a few test papers dominate the risk/error table, the collapse is partly group-composition driven.")
    lines.append("6. If most test papers have high risk/error, the problem is general cross-literature extrapolation.")

    report_path = TABLE_DIR / "30_grouped_validation_diagnosis_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 9. Main
# =============================================================================

def main():
    log("\n=== Step 30: Grouped validation diagnosis ===")
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"FIG_DIR      = {FIG_DIR}")
    log(f"TABLE_DIR    = {TABLE_DIR}")

    df_raw, data_path, target_col, source_col = load_final_data_with_source()
    log(f"\nLoaded data: {data_path}")
    log(f"Raw shape: {df_raw.shape}")
    log(f"Target column: {target_col}")
    log(f"Source column: {source_col}")

    df, numeric_roles, categorical_roles = prepare_analysis_dataframe(df_raw, target_col, source_col)
    log(f"Rows after target filtering: {df.shape[0]}")
    log(f"Number of papers: {df['Paper_ID'].nunique()}")

    log("\nIdentified numeric roles:")
    for k, v in numeric_roles.items():
        log(f"  {k}: {v}")
    log("Identified categorical roles:")
    for k, v in categorical_roles.items():
        log(f"  {k}: {v}")

    # Save column mapping.
    mapping_rows = []
    for role, col in numeric_roles.items():
        mapping_rows.append({"Type": "numeric", "Role": role, "Column": col})
    for role, col in categorical_roles.items():
        mapping_rows.append({"Type": "categorical", "Role": role, "Column": col})
    pd.DataFrame(mapping_rows).to_csv(TABLE_DIR / "30_column_role_mapping.csv", index=False, encoding="utf-8-sig")

    log("\nCreating grouped split assignments...")
    split_df = make_group_splits(df)
    split_df.to_csv(TABLE_DIR / "30_grouped_split_assignments_by_seed.csv", index=False, encoding="utf-8-sig")

    log("Calculating paper-level feature summary...")
    paper_summary = paper_feature_summary(df, numeric_roles, categorical_roles)
    paper_summary.to_csv(TABLE_DIR / "30_paper_level_feature_summary.csv", index=False, encoding="utf-8-sig")

    log("Calculating split summary...")
    split_sum = split_summary(df, split_df)
    split_sum.to_csv(TABLE_DIR / "30_grouped_split_size_and_target_summary.csv", index=False, encoding="utf-8-sig")

    log("Calculating numeric distribution shift...")
    shift_df = distribution_shift_table(df, split_df, numeric_roles)
    shift_df.to_csv(TABLE_DIR / "30_seed_level_numeric_distribution_shift.csv", index=False, encoding="utf-8-sig")

    log("Calculating categorical coverage...")
    coverage_df = category_coverage_table(df, split_df, categorical_roles)
    coverage_df.to_csv(TABLE_DIR / "30_seed_level_categorical_coverage.csv", index=False, encoding="utf-8-sig")

    log("Calculating paper split table...")
    paper_split = paper_split_table(split_df)
    paper_split.to_csv(TABLE_DIR / "30_paper_split_assignment_summary.csv", index=False, encoding="utf-8-sig")

    log("Calculating paper extrapolation risk...")
    risk_df = paper_extrapolation_risk(df, split_df, numeric_roles, categorical_roles)
    risk_df.to_csv(TABLE_DIR / "30_paper_level_extrapolation_risk.csv", index=False, encoding="utf-8-sig")

    log("Reading Step-29 prediction records if available...")
    metrics_df, pred_file, pred_msg = paper_metrics_from_predictions(df, split_df)
    if not metrics_df.empty:
        metrics_df.to_csv(TABLE_DIR / "30_paper_level_metrics_from_29_predictions.csv", index=False, encoding="utf-8-sig")
    else:
        (TABLE_DIR / "30_need_29_row_level_predictions.txt").write_text(pred_msg, encoding="utf-8")

    log("Generating figures...")
    plot_group_size(paper_summary)
    plot_paper_target_summary(paper_summary)
    plot_seed_split_target_box(df, split_df)
    plot_numeric_shift_heatmap(shift_df)
    plot_category_coverage_heatmap(coverage_df)
    plot_risk_score(risk_df)
    plot_paper_metrics(metrics_df)

    log("Generating text report...")
    generate_report(
        df=df,
        data_path=data_path,
        target_col=target_col,
        source_col=source_col,
        numeric_roles=numeric_roles,
        categorical_roles=categorical_roles,
        paper_summary=paper_summary,
        split_sum=split_sum,
        shift_df=shift_df,
        coverage_df=coverage_df,
        risk_df=risk_df,
        metrics_df=metrics_df,
        pred_msg=pred_msg,
    )

    log("\n=== Step 30 completed ===")
    log("Key tables saved to:")
    for name in [
        "30_column_role_mapping.csv",
        "30_grouped_split_assignments_by_seed.csv",
        "30_paper_level_feature_summary.csv",
        "30_grouped_split_size_and_target_summary.csv",
        "30_seed_level_numeric_distribution_shift.csv",
        "30_seed_level_categorical_coverage.csv",
        "30_paper_split_assignment_summary.csv",
        "30_paper_level_extrapolation_risk.csv",
        "30_grouped_validation_diagnosis_report.txt",
    ]:
        log(f"  {TABLE_DIR / name}")

    if not metrics_df.empty:
        log(f"  {TABLE_DIR / '30_paper_level_metrics_from_29_predictions.csv'}")
    else:
        log(f"  {TABLE_DIR / '30_need_29_row_level_predictions.txt'}")

    log("\nKey figures saved to:")
    for name in [
        "69_literature_group_size_distribution.png",
        "70_paper_level_target_distribution.png",
        "71_grouped_split_target_distribution_by_seed.png",
        "72_train_test_numeric_shift_smd_heatmap.png",
        "73_test_unseen_category_fraction_heatmap.png",
        "74_high_risk_test_literature_groups.png",
        "75_paper_level_error_vs_target_mean.png",
    ]:
        p = FIG_DIR / name
        if p.exists():
            log(f"  {p}")

    log("\nPlease send me these files after running:")
    log("  1) outputs/tables/30_grouped_validation_diagnosis_report.txt")
    log("  2) outputs/tables/30_seed_level_numeric_distribution_shift.csv")
    log("  3) outputs/tables/30_seed_level_categorical_coverage.csv")
    log("  4) outputs/tables/30_paper_level_extrapolation_risk.csv")
    log("  5) figures 69-75, especially 72, 73 and 74")


if __name__ == "__main__":
    main()
