# -*- coding: utf-8 -*-
"""
29c_rebuild_model_ready_v1_exact_legacy_structure.py

Purpose
-------
Rebuild final model_ready_v1.csv from the final Excel dataset while preserving
the exact legacy 26-column structure used by the previous 18/19 workflow.

This script corrects the earlier simplified 15-column rebuild.

Inputs
------
data/raw/数据集修改最终.xlsx
data/processed/model_ready_v1_legacy_1103.csv

Outputs
-------
data/processed/model_ready_v1.csv
    - final 1011-row model-ready table
    - exact same 26 columns as legacy model_ready_v1_legacy_1103.csv
    - does NOT include 文献编号

data/processed/model_ready_v1_with_source.csv
    - same 26 model-ready columns
    - plus 文献编号
    - only for grouped validation

data/processed/cleaned_full_with_flags.csv
    - final cleaned full table with 文献编号 for audit

outputs/tables/29c_exact_legacy_column_mapping.csv
outputs/tables/29c_exact_legacy_row_audit.csv
outputs/tables/29c_exact_legacy_group_distribution.csv
outputs/tables/29c_exact_legacy_summary.txt

Strict rule
-----------
文献编号 is NOT included in model_ready_v1.csv.
It is used only for grouped validation.
"""

from __future__ import annotations

import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Basic utilities and paths
# ============================================================

def log(msg: str):
    print(msg, flush=True)


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
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FINAL_EXCEL = RAW_DIR / "数据集修改最终.xlsx"
LEGACY_MODEL_READY = PROCESSED_DIR / "model_ready_v1_legacy_1103.csv"

MODEL_READY_OUT = PROCESSED_DIR / "model_ready_v1.csv"
MODEL_READY_WITH_SOURCE_OUT = PROCESSED_DIR / "model_ready_v1_with_source.csv"
CLEANED_FULL_OUT = PROCESSED_DIR / "cleaned_full_with_flags.csv"


# ============================================================
# 2. Legacy columns
# ============================================================

def read_csv_header(path: Path) -> list[str]:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                return next(reader)
        except Exception:
            continue

    raise ValueError(f"Cannot read header from {path}")


EXPECTED_LEGACY_COLUMNS = [
    "氟形态",
    "游离氟占比",
    "初始氟_mg_L",
    "干扰离子",
    "干扰离子浓度_mmol_L",
    "水温_C",
    "混凝剂",
    "剂量_mM",
    "进水pH",
    "停留时间_min",
    "出水pH",
    "氟去除率_pct",
    "出水氟_mg_L",
    "原始行号",
    "游离氟占比_原始",
    "初始氟_mg_L_原始",
    "干扰离子浓度_mmol_L_原始",
    "水温_C_原始",
    "剂量_mM_原始",
    "进水pH_原始",
    "停留时间_min_原始",
    "出水pH_原始",
    "氟去除率_pct_原始",
    "出水氟_mg_L_原始",
    "由浓度反算去除率_pct",
    "去除率差值_pct",
]


def get_legacy_columns() -> list[str]:
    if not LEGACY_MODEL_READY.exists():
        raise FileNotFoundError(
            f"Missing legacy model-ready file:\n{LEGACY_MODEL_READY}\n"
            "Please put model_ready_v1_legacy_1103.csv under data/processed."
        )

    legacy_cols = read_csv_header(LEGACY_MODEL_READY)

    if legacy_cols != EXPECTED_LEGACY_COLUMNS:
        log("[WARN] Legacy columns differ from the known expected columns.")
        log("The script will use the actual header from model_ready_v1_legacy_1103.csv.")

        log("\nActual legacy columns:")
        for i, c in enumerate(legacy_cols, start=1):
            log(f"  {i:02d}. {c}")

    return legacy_cols


# ============================================================
# 3. Robust column matching
# ============================================================

def normalize_name(s: str) -> str:
    s = str(s).strip()

    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("％", "%")
    s = s.replace("⁻", "-")
    s = s.replace("－", "-").replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.replace("℃", "C")
    s = s.replace("°C", "C").replace("°c", "c")

    s = s.lower()
    s = re.sub(r"\s+", "", s)

    s = s.replace("mg/l", "mgl")
    s = s.replace("mg·l-1", "mgl")
    s = s.replace("mg·l^-1", "mgl")
    s = s.replace("mg/l-1", "mgl")
    s = s.replace("mg/l^-1", "mgl")
    s = s.replace("mgl-1", "mgl")
    s = s.replace("mgl^-1", "mgl")

    s = s.replace("mmol/l", "mmoll")
    s = s.replace("mmol·l-1", "mmoll")
    s = s.replace("mmol·l^-1", "mmoll")
    s = s.replace("mmol/l-1", "mmoll")
    s = s.replace("mmol/l^-1", "mmoll")
    s = s.replace("mmoll-1", "mmoll")
    s = s.replace("mmoll^-1", "mmoll")

    # Critical fluoride column normalization.
    s = s.replace("f-", "氟")
    s = s.replace("f－", "氟")
    s = s.replace("f–", "氟")
    s = s.replace("f—", "氟")

    for prefix in ["初始", "进水", "原水", "出水", "残余", "剩余"]:
        s = s.replace(prefix + "f", prefix + "氟")

    for ch in [
        "_", "-", "/", "\\", "／",
        "(", ")", "[", "]", "{", "}",
        "：", ":", "；", ";", "，", ",",
        ".", "·", "•", " "
    ]:
        s = s.replace(ch, "")

    return s


COLUMN_CANDIDATES = {
    "氟形态": [
        "氟离子分布形态",
        "氟形态",
        "fluoride form",
    ],
    "游离氟占比": [
        "游离氟占比",
        "free fluoride fraction",
    ],
    "初始氟_mg_L": [
        "初始 F⁻(mg/L)",
        "初始 F-(mg/L)",
        "初始F-(mg/L)",
        "初始 F",
        "初始F",
        "初始 F-",
        "初始F-",
        "初始氟",
        "初始氟浓度",
        "初始氟_mg_L",
        "进水氟",
        "进水氟浓度",
        "原水氟",
        "原水氟浓度",
        "initial fluoride",
        "initial F",
        "initial F-",
        "influent fluoride",
        "influent F",
        "influent F-",
    ],
    "干扰离子": [
        "干扰离子",
        "共存离子",
        "竞争离子",
        "interfering ions",
        "coexisting ions",
    ],
    "干扰离子浓度_mmol_L": [
        "干扰离子浓度 (mmol/L)",
        "干扰离子浓度（mmol/L）",
        "干扰离子浓度",
        "干扰离子浓度_mmol_L",
        "共存离子浓度",
        "interfering ion concentration",
        "coexisting ion concentration",
    ],
    "水温_C": [
        "水温 (℃)",
        "水温（℃）",
        "水温",
        "温度",
        "temperature",
        "water temperature",
    ],
    "混凝剂": [
        "混凝剂",
        "coagulant",
        "coagulant type",
        "coagulant name",
    ],
    "剂量_mM": [
        "剂量 (mM)",
        "剂量（mM）",
        "剂量",
        "剂量_mM",
        "混凝剂剂量",
        "混凝剂投加量",
        "投加量",
        "dose",
        "dosage",
        "coagulant dose",
        "coagulant dosage",
    ],
    "进水pH": [
        "进水 pH",
        "进水pH",
        "初始pH",
        "原水pH",
        "influent pH",
        "initial pH",
        "solution pH",
        "pH",
    ],
    "停留时间_min": [
        "停留时间（min）",
        "停留时间 (min)",
        "停留时间",
        "水力停留时间",
        "反应时间",
        "contact time",
        "retention time",
        "reaction time",
    ],
    "出水pH": [
        "出水 pH",
        "出水pH",
        "effluent pH",
        "final pH",
    ],
    "氟去除率_pct": [
        "氟去除率 (%)",
        "氟去除率（%）",
        "氟去除率",
        "除氟率",
        "F removal",
        "fluoride removal",
        "removal rate",
        "removal efficiency",
        "RE",
    ],
    "出水氟_mg_L": [
        "出水 F⁻(mg/L)",
        "出水 F-(mg/L)",
        "出水F-(mg/L)",
        "出水 F",
        "出水F",
        "出水 F-",
        "出水F-",
        "出水氟",
        "出水氟浓度",
        "残余氟",
        "剩余氟",
        "effluent fluoride",
        "effluent F",
        "effluent F-",
        "residual fluoride",
        "remaining fluoride",
    ],
    "文献编号": [
        "文献编号",
        "文献来源",
        "来源编号",
        "source_id",
        "paper_id",
        "literature id",
        "reference id",
        "study id",
    ],
}


def find_source_column(raw_columns: list[str], standard_col: str) -> str | None:
    candidates = COLUMN_CANDIDATES.get(standard_col, [standard_col])
    norm_cols = {c: normalize_name(c) for c in raw_columns}
    norm_candidates = [normalize_name(c) for c in candidates]

    # Exact match.
    for cand in norm_candidates:
        for raw_col, raw_norm in norm_cols.items():
            if raw_norm == cand:
                return raw_col

    # Contains match.
    for cand in norm_candidates:
        for raw_col, raw_norm in norm_cols.items():
            if cand in raw_norm or raw_norm in cand:
                return raw_col

    return None


# ============================================================
# 4. Parsing functions
# ============================================================

def parse_numeric_value(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()

    if s == "":
        return np.nan

    if s.lower() in ["nan", "na", "none", "null"]:
        return np.nan

    if s in ["无", "/", "-", "—", "–"]:
        return np.nan

    if "室温" in s or "room" in s.lower():
        return 25.0

    s = s.replace("，", ",")
    s = s.replace("－", "-").replace("–", "-").replace("—", "-").replace("−", "-")

    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)

    if not nums:
        return np.nan

    vals = [float(v) for v in nums]

    # For ranges such as 5~7 or 5–7, use the mean of the first two numbers.
    if len(vals) >= 2 and any(sep in s for sep in ["~", "～", "至"]):
        return float(np.mean(vals[:2]))

    return vals[0]


def count_numbers(x) -> int:
    if pd.isna(x):
        return 0

    s = str(x)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)

    return len(nums)


def clean_text_value(x, default="未提供"):
    if pd.isna(x):
        return default

    s = str(x).strip()

    if s == "":
        return default

    if s.lower() in ["nan", "na", "none", "null"]:
        return default

    return s


def normalize_group_id(x):
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
# 5. Read final Excel
# ============================================================

def load_final_excel() -> pd.DataFrame:
    if not FINAL_EXCEL.exists():
        raise FileNotFoundError(
            f"Missing final Excel:\n{FINAL_EXCEL}\n"
            "Please put the final dataset at data/raw/数据集修改最终.xlsx."
        )

    # Current final Excel has two header rows:
    # row 1: grouped header
    # row 2: real column names
    df = pd.read_excel(FINAL_EXCEL, sheet_name="Sheet1", header=1)
    df = df.dropna(how="all").reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("Final Excel was read as an empty table.")

    return df


def build_mapping(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for std_col in [
        "氟形态",
        "游离氟占比",
        "初始氟_mg_L",
        "干扰离子",
        "干扰离子浓度_mmol_L",
        "水温_C",
        "混凝剂",
        "剂量_mM",
        "进水pH",
        "停留时间_min",
        "出水pH",
        "氟去除率_pct",
        "出水氟_mg_L",
        "文献编号",
    ]:
        src = find_source_column(list(raw_df.columns), std_col)
        rows.append({
            "standard_column": std_col,
            "source_column": "" if src is None else src,
            "found": src is not None,
        })

    return pd.DataFrame(rows)


def get_source_col(mapping_df: pd.DataFrame, standard_col: str) -> str:
    row = mapping_df[mapping_df["standard_column"] == standard_col]

    if row.empty:
        raise KeyError(f"No mapping row for {standard_col}")

    src = str(row.iloc[0]["source_column"]).strip()

    if src == "":
        raise ValueError(f"Column mapping failed for required column: {standard_col}")

    return src


# ============================================================
# 6. Rebuild exact legacy table
# ============================================================

BASE_LEGACY_COLUMNS = [
    "氟形态",
    "游离氟占比",
    "初始氟_mg_L",
    "干扰离子",
    "干扰离子浓度_mmol_L",
    "水温_C",
    "混凝剂",
    "剂量_mM",
    "进水pH",
    "停留时间_min",
    "出水pH",
    "氟去除率_pct",
    "出水氟_mg_L",
]

NUMERIC_COLUMNS = [
    "游离氟占比",
    "初始氟_mg_L",
    "干扰离子浓度_mmol_L",
    "水温_C",
    "剂量_mM",
    "进水pH",
    "停留时间_min",
    "出水pH",
    "氟去除率_pct",
    "出水氟_mg_L",
]

TEXT_COLUMNS = [
    "氟形态",
    "干扰离子",
    "混凝剂",
]

CRITICAL_COLUMNS = [
    "氟形态",
    "初始氟_mg_L",
    "混凝剂",
    "剂量_mM",
    "进水pH",
    "停留时间_min",
    "氟去除率_pct",
]


def rebuild_legacy_structure(raw_df: pd.DataFrame, mapping_df: pd.DataFrame):
    out = pd.DataFrame(index=raw_df.index)

    # Text columns.
    for col in TEXT_COLUMNS:
        src = get_source_col(mapping_df, col)
        out[col] = raw_df[src].map(clean_text_value)

    # Numeric columns and original backups.
    for col in NUMERIC_COLUMNS:
        src = get_source_col(mapping_df, col)
        out[f"{col}_原始"] = raw_df[src]
        out[col] = raw_df[src].map(parse_numeric_value)

    # If target was stored as 0-1 fraction, convert to percent.
    y_valid = out["氟去除率_pct"].dropna()
    if len(y_valid) > 0 and y_valid.quantile(0.95) <= 1.5:
        out["氟去除率_pct"] = out["氟去除率_pct"] * 100.0

    # Original Excel row index, consistent with old 02_clean_data.py.
    out["原始行号"] = np.arange(len(out)) + 3

    # Concentration-derived removal check.
    out["由浓度反算去除率_pct"] = (
        1 - out["出水氟_mg_L"] / out["初始氟_mg_L"]
    ) * 100.0

    out["去除率差值_pct"] = (
        out["由浓度反算去除率_pct"] - out["氟去除率_pct"]
    ).abs()

    # Literature group.
    group_src = get_source_col(mapping_df, "文献编号")
    out["文献编号"] = raw_df[group_src].map(normalize_group_id)
    out["文献编号_原始"] = raw_df[group_src]

    return out


def make_quality_flags(work_df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame({
        "原始行号": work_df["原始行号"],
        "文献编号": work_df["文献编号"],
    })

    flags["进水pH_异常"] = ~work_df["进水pH"].between(0, 14, inclusive="both")
    flags["出水pH_异常"] = ~work_df["出水pH"].between(0, 14, inclusive="both")
    flags["氟去除率_异常"] = ~work_df["氟去除率_pct"].between(0, 100, inclusive="both")
    flags["水温_异常"] = ~(
        work_df["水温_C"].isna() | work_df["水温_C"].between(0, 100, inclusive="both")
    )
    flags["停留时间_异常"] = ~work_df["停留时间_min"].between(0, 10000, inclusive="both")

    flags["剂量_疑似非mM单位"] = work_df["剂量_mM_原始"].astype(str).str.contains(
        "mg/L|mL/L|g/L|摩尔比", regex=True, na=False
    )

    flags["数值列含多个数字_需复核"] = False
    for col in NUMERIC_COLUMNS:
        flags["数值列含多个数字_需复核"] |= work_df[f"{col}_原始"].map(count_numbers).gt(1)

    flags["去除率与浓度不一致_差值大于5pct"] = work_df["去除率差值_pct"].gt(5)

    flag_cols = [c for c in flags.columns if c not in ["原始行号", "文献编号"]]
    flags["需要人工复核"] = flags[flag_cols].any(axis=1)

    return flags


def filter_model_ready(work_df: pd.DataFrame, legacy_cols: list[str]):
    row_audit = pd.DataFrame({
        "原始行号": work_df["原始行号"],
        "文献编号": work_df["文献编号"],
    })

    keep = pd.Series(True, index=work_df.index)

    for col in CRITICAL_COLUMNS:
        row_audit[f"{col}_缺失"] = work_df[col].isna()
        keep &= work_df[col].notna()

    row_audit["进水pH_有效"] = work_df["进水pH"].between(0, 14)
    row_audit["氟去除率_有效"] = work_df["氟去除率_pct"].between(0, 100)
    row_audit["水温_有效或缺失"] = work_df["水温_C"].isna() | work_df["水温_C"].between(0, 100)

    keep &= row_audit["进水pH_有效"]
    keep &= row_audit["氟去除率_有效"]
    keep &= row_audit["水温_有效或缺失"]

    before_duplicate_n = int(keep.sum())

    model_candidate = work_df.loc[keep, legacy_cols + ["文献编号"]].copy()

    duplicate_mask = model_candidate[legacy_cols].duplicated(keep="first")
    model_candidate = model_candidate.loc[~duplicate_mask].reset_index(drop=True)

    model_ready = model_candidate[legacy_cols].copy()
    model_ready_with_source = model_candidate[legacy_cols + ["文献编号"]].copy()

    row_audit["kept_before_duplicate_removal"] = keep
    row_audit["drop_reason"] = ""

    for col in CRITICAL_COLUMNS:
        row_audit.loc[row_audit[f"{col}_缺失"], "drop_reason"] += f"missing_{col};"

    row_audit.loc[~row_audit["进水pH_有效"], "drop_reason"] += "invalid_influent_pH;"
    row_audit.loc[~row_audit["氟去除率_有效"], "drop_reason"] += "invalid_removal;"
    row_audit.loc[~row_audit["水温_有效或缺失"], "drop_reason"] += "invalid_temperature;"

    row_audit["duplicate_rows_removed_after_filtering"] = 0
    row_audit.loc[row_audit["kept_before_duplicate_removal"], "duplicate_rows_removed_after_filtering"] = 0

    return model_ready, model_ready_with_source, row_audit, before_duplicate_n


# ============================================================
# 7. Backup and output
# ============================================================

def backup_file(path: Path, label: str):
    if not path.exists():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.parent / f"{path.stem}_{label}_{timestamp}{path.suffix}"
    shutil.copy2(path, backup_path)

    log(f"[BACKUP] {path.name} -> {backup_path.name}")


def write_outputs(
    model_ready: pd.DataFrame,
    model_ready_with_source: pd.DataFrame,
    cleaned_full: pd.DataFrame,
    mapping_df: pd.DataFrame,
    row_audit: pd.DataFrame,
    quality_flags: pd.DataFrame,
    legacy_cols: list[str],
):
    # Backup current files.
    backup_file(MODEL_READY_OUT, "before_exact_legacy_rebuild")
    backup_file(MODEL_READY_WITH_SOURCE_OUT, "before_exact_legacy_rebuild")
    backup_file(CLEANED_FULL_OUT, "before_exact_legacy_rebuild")

    # Write processed files.
    model_ready.to_csv(MODEL_READY_OUT, index=False, encoding="utf-8-sig")
    model_ready_with_source.to_csv(MODEL_READY_WITH_SOURCE_OUT, index=False, encoding="utf-8-sig")
    cleaned_full.to_csv(CLEANED_FULL_OUT, index=False, encoding="utf-8-sig")

    log(f"[WRITE] {MODEL_READY_OUT} | shape={model_ready.shape}")
    log(f"[WRITE] {MODEL_READY_WITH_SOURCE_OUT} | shape={model_ready_with_source.shape}")
    log(f"[WRITE] {CLEANED_FULL_OUT} | shape={cleaned_full.shape}")

    # Audit files.
    mapping_path = TABLE_DIR / "29c_exact_legacy_column_mapping.csv"
    row_audit_path = TABLE_DIR / "29c_exact_legacy_row_audit.csv"
    flags_path = TABLE_DIR / "29c_exact_legacy_quality_flags.csv"
    group_path = TABLE_DIR / "29c_exact_legacy_group_distribution.csv"
    summary_path = TABLE_DIR / "29c_exact_legacy_summary.txt"

    mapping_df.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    row_audit.to_csv(row_audit_path, index=False, encoding="utf-8-sig")
    quality_flags.to_csv(flags_path, index=False, encoding="utf-8-sig")

    group_dist = (
        model_ready_with_source["文献编号"]
        .value_counts()
        .rename_axis("文献编号")
        .reset_index(name="n_rows")
    )
    group_dist["fraction"] = group_dist["n_rows"] / len(model_ready_with_source)
    group_dist.to_csv(group_path, index=False, encoding="utf-8-sig")

    # Strict checks.
    same_cols = list(model_ready.columns) == legacy_cols
    has_source_in_model_ready = "文献编号" in model_ready.columns

    y = model_ready["氟去除率_pct"]

    lines = []
    lines.append("=" * 100)
    lines.append("29c exact legacy-structure rebuild summary")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Core result:")
    lines.append(f"model_ready_v1.csv shape             : {model_ready.shape}")
    lines.append(f"model_ready_v1_with_source.csv shape : {model_ready_with_source.shape}")
    lines.append(f"cleaned_full_with_flags.csv shape    : {cleaned_full.shape}")
    lines.append("")
    lines.append("Strict structure checks:")
    lines.append(f"model_ready columns exactly equal legacy columns : {same_cols}")
    lines.append(f"文献编号 included in model_ready_v1.csv            : {has_source_in_model_ready}")
    lines.append("")
    lines.append("Legacy columns used:")
    for i, c in enumerate(legacy_cols, start=1):
        lines.append(f"{i:02d}. {c}")
    lines.append("")
    lines.append("Column mapping:")
    lines.append(mapping_df.to_string(index=False))
    lines.append("")
    lines.append("Target summary:")
    lines.append(f"n      : {int(y.notna().sum())}")
    lines.append(f"mean   : {y.mean():.4f}")
    lines.append(f"std    : {y.std():.4f}")
    lines.append(f"min    : {y.min():.4f}")
    lines.append(f"median : {y.median():.4f}")
    lines.append(f"max    : {y.max():.4f}")
    lines.append("")
    lines.append("Literature group summary:")
    lines.append(f"n_groups       : {group_dist['文献编号'].nunique()}")
    lines.append(f"largest group  : {int(group_dist['n_rows'].max())}")
    lines.append(f"smallest group : {int(group_dist['n_rows'].min())}")
    lines.append(f"median group n : {group_dist['n_rows'].median():.2f}")
    lines.append("")
    lines.append("Top 20 groups:")
    lines.append(group_dist.head(20).to_string(index=False))
    lines.append("")
    lines.append("Important rule:")
    lines.append("model_ready_v1.csv preserves the legacy 26-column structure and does not include 文献编号.")
    lines.append("model_ready_v1_with_source.csv includes 文献编号 and is used only for grouped validation.")
    lines.append("=" * 100)

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    log(f"[TABLE] {mapping_path}")
    log(f"[TABLE] {row_audit_path}")
    log(f"[TABLE] {flags_path}")
    log(f"[TABLE] {group_path}")
    log(f"[TEXT]  {summary_path}")


# ============================================================
# 8. Main
# ============================================================

def main():
    log("\n" + "=" * 100)
    log("29c Rebuild model_ready_v1.csv using exact legacy 26-column structure")
    log("=" * 100)
    log(f"PROJECT_ROOT        : {PROJECT_ROOT}")
    log(f"FINAL_EXCEL         : {FINAL_EXCEL}")
    log(f"LEGACY_MODEL_READY  : {LEGACY_MODEL_READY}")
    log(f"PROCESSED_DIR       : {PROCESSED_DIR}")
    log(f"TABLE_DIR           : {TABLE_DIR}")
    log("=" * 100)

    legacy_cols = get_legacy_columns()

    log("\nLegacy model_ready columns:")
    for i, c in enumerate(legacy_cols, start=1):
        log(f"  {i:02d}. {c}")

    raw_df = load_final_excel()

    log("\nFinal Excel loaded:")
    log(f"shape = {raw_df.shape}")
    log("columns:")
    for i, c in enumerate(raw_df.columns, start=1):
        log(f"  {i:02d}. {c}")

    mapping_df = build_mapping(raw_df)

    log("\n=== Column mapping ===")
    print(mapping_df.to_string(index=False))

    required = [
        "氟形态",
        "游离氟占比",
        "初始氟_mg_L",
        "干扰离子",
        "干扰离子浓度_mmol_L",
        "水温_C",
        "混凝剂",
        "剂量_mM",
        "进水pH",
        "停留时间_min",
        "出水pH",
        "氟去除率_pct",
        "出水氟_mg_L",
        "文献编号",
    ]

    missing = []

    for c in required:
        row = mapping_df[mapping_df["standard_column"] == c]
        if row.empty or not bool(row.iloc[0]["found"]):
            missing.append(c)

    if missing:
        raise ValueError(
            "Missing required final Excel columns after mapping: "
            + ", ".join(missing)
        )

    work_df = rebuild_legacy_structure(raw_df, mapping_df)

    quality_flags = make_quality_flags(work_df)

    model_ready, model_ready_with_source, row_audit, before_duplicate_n = filter_model_ready(
        work_df=work_df,
        legacy_cols=legacy_cols,
    )

    # Full cleaned file: legacy 26 columns + 文献编号.
    cleaned_full = work_df[legacy_cols + ["文献编号"]].copy()

    # Final hard checks.
    if list(model_ready.columns) != legacy_cols:
        raise RuntimeError("Output model_ready_v1.csv columns do not match legacy columns.")

    if "文献编号" in model_ready.columns:
        raise RuntimeError("文献编号 must not appear in model_ready_v1.csv.")

    if len(model_ready) == 0:
        raise RuntimeError("No rows left after filtering. Check column mapping.")

    log("\n=== Row filtering ===")
    log(f"Raw final Excel rows                 : {len(raw_df)}")
    log(f"Rows before duplicate removal         : {before_duplicate_n}")
    log(f"Final model_ready_v1 rows             : {len(model_ready)}")
    log(f"Final model_ready_v1 columns          : {model_ready.shape[1]}")
    log(f"Final with-source rows                : {len(model_ready_with_source)}")
    log(f"Literature groups                     : {model_ready_with_source['文献编号'].nunique()}")

    write_outputs(
        model_ready=model_ready,
        model_ready_with_source=model_ready_with_source,
        cleaned_full=cleaned_full,
        mapping_df=mapping_df,
        row_audit=row_audit,
        quality_flags=quality_flags,
        legacy_cols=legacy_cols,
    )

    log("\n" + "=" * 100)
    log("[DONE] Exact legacy-structure model_ready_v1.csv rebuilt successfully.")
    log("=" * 100)
    log("Next:")
    log("Open outputs/tables/29c_exact_legacy_summary.txt and check:")
    log("1. model_ready_v1.csv shape should be 1011 x 26")
    log("2. columns exactly equal legacy columns should be True")
    log("3. 文献编号 included in model_ready_v1.csv should be False")
    log("4. model_ready_v1_with_source.csv should include 文献编号 for grouped validation")
    log("=" * 100)


if __name__ == "__main__":
    main()