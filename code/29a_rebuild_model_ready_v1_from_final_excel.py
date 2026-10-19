# -*- coding: utf-8 -*-
"""
29a_rebuild_model_ready_v1_from_final_excel.py

Purpose
-------
Rebuild the final model-ready dataset from the final Excel file.

Final raw data source:
    data/raw/数据集修改最终.xlsx

Main outputs:
    data/processed/model_ready_v1.csv
        - final model-ready table for model training
        - does NOT include literature/source group label

    data/processed/model_ready_v1_with_source.csv
        - same rows as model_ready_v1.csv
        - includes 文献编号 for grouped validation only

    data/processed/cleaned_full_with_flags.csv
        - cleaned full table with source label and auxiliary columns

Audit outputs:
    outputs/tables/29a_final_dataset_column_mapping.csv
    outputs/tables/29a_final_dataset_row_audit.csv
    outputs/tables/29a_final_dataset_group_distribution.csv
    outputs/tables/29a_final_dataset_summary.txt

Important rule
--------------
文献编号 / source_id is used only for grouped validation.
It must NOT be used as a model input feature.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

def log(msg: str):
    print(msg, flush=True)


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    candidates = [
        current,
        current.parent,
        current.parent.parent,
    ]

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

FINAL_EXCEL_CANDIDATES = [
    RAW_DIR / "数据集修改最终.xlsx",
    RAW_DIR / "final_dataset_with_literature_id.xlsx",
]


# ============================================================
# 2. Column detection
# ============================================================

def normalize_col(s: str) -> str:
    """
    Robust column-name normalizer.

    It treats Chinese 氟 and English F/F- in fluoride-related column names
    as comparable forms.
    """
    s = str(s)

    # Common full-width symbols
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("％", "%")
    s = s.replace("－", "-").replace("–", "-").replace("—", "-")
    s = s.replace("−", "-")
    s = s.replace("℃", "C")
    s = s.replace("°C", "C").replace("°c", "c")

    # Lowercase English letters
    s = s.lower()

    # Remove spaces
    s = re.sub(r"\s+", "", s)

    # Convert common concentration unit forms
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

    # Critical fluoride-column unification:
    # 初始 F-(mg/L), 出水 F-(mg/L), F removal, etc.
    s = s.replace("f-", "氟")
    s = s.replace("f－", "氟")
    s = s.replace("f–", "氟")
    s = s.replace("f—", "氟")

    # If F is attached to Chinese fluoride context, also convert it.
    s = s.replace("初始f", "初始氟")
    s = s.replace("进水f", "进水氟")
    s = s.replace("原水f", "原水氟")
    s = s.replace("出水f", "出水氟")
    s = s.replace("残余f", "残余氟")
    s = s.replace("剩余f", "剩余氟")

    # Remove brackets and separators
    remove_chars = [
        "_", "-", "/", "\\", "／",
        "(", ")", "[", "]", "{", "}",
        "：", ":", "；", ";", "，", ",",
        ".", "·", "•", " "
    ]

    for ch in remove_chars:
        s = s.replace(ch, "")

    return s


def find_col(df: pd.DataFrame, candidates: list[str], exclude: list[str] | None = None) -> str | None:
    exclude = exclude or []

    cols = list(df.columns)
    norm_map = {c: normalize_col(c) for c in cols}
    cand_norms = [normalize_col(c) for c in candidates]
    exclude_norms = [normalize_col(e) for e in exclude]

    # Exact match first.
    for cand in cand_norms:
        for col, ncol in norm_map.items():
            if ncol == cand:
                if any(ex in ncol for ex in exclude_norms):
                    continue
                return col

    # Contains match.
    for cand in cand_norms:
        for col, ncol in norm_map.items():
            if cand in ncol:
                if any(ex in ncol for ex in exclude_norms):
                    continue
                return col

    return None


COLUMN_CANDIDATES = {
    "混凝剂": [
        "混凝剂", "coagulant", "coagulant type", "coagulant name", "药剂", "除氟剂",
    ],
    "剂量_mM": [
        "剂量_mM", "混凝剂剂量_mM", "混凝剂投加量_mM", "投加量_mM",
        "coagulant dose mM", "coagulant dosage mM", "dose mM", "dosage mM",
        "coagulant dose", "dosage",
    ],
    "絮凝剂": [
        "絮凝剂", "flocculant", "polymer", "助凝剂",
    ],
    "絮凝剂剂量": [
        "絮凝剂剂量", "絮凝剂投加量", "助凝剂剂量", "flocculant dose", "polymer dose",
    ],
    "搅拌时间_min": [
        "搅拌时间", "快速搅拌时间", "慢速搅拌时间", "stirring time", "mixing time",
    ],
    "沉淀时间_min": [
        "沉淀时间", "静置时间", "settling time", "sedimentation time",
    ],
    "停留时间_min": [
        "停留时间", "水力停留时间", "反应时间", "retention time", "contact time", "reaction time",
    ],
    "进水pH": [
        "进水pH", "初始pH", "原水pH", "influent pH", "initial pH", "solution pH", "pH",
    ],
    "初始氟_mg_L": [
        "初始氟_mg_L",
        "初始氟",
        "初始氟浓度",
        "进水氟",
        "进水氟浓度",
        "原水氟",
        "原水氟浓度",
        "初始 F-(mg/L)",
        "初始F-(mg/L)",
        "初始 F-",
        "初始F-",
        "初始 F",
        "初始F",
        "进水 F-(mg/L)",
        "进水F-(mg/L)",
        "原水 F-(mg/L)",
        "原水F-(mg/L)",
        "initial fluoride",
        "initial fluoride concentration",
        "initial F",
        "initial F-",
        "initial F mg/L",
        "initial F- mg/L",
        "influent fluoride",
        "influent F",
        "influent F-",
        "F initial",
        "F- initial",
        "fluoride concentration",
    ],
    "水温_C": [
        "水温", "温度", "temperature", "water temperature",
    ],
    "干扰离子": [
        "干扰离子", "共存离子", "竞争离子", "interfering ions", "coexisting ions", "competing ions",
    ],
    "干扰离子浓度_mmol": [
        "干扰离子浓度", "共存离子浓度", "竞争离子浓度", "interfering ion concentration",
        "coexisting ion concentration",
    ],
    "氟形态": [
        "氟形态", "fluoride form", "F form",
    ],
    "游离氟占比": [
        "游离氟占比", "free fluoride fraction", "free F fraction",
    ],
    "出水pH": [
        "出水pH", "effluent pH", "final pH",
    ],
    "出水氟_mg_L": [
        "出水氟_mg_L",
        "出水氟",
        "出水氟浓度",
        "残余氟",
        "残余氟浓度",
        "剩余氟",
        "剩余氟浓度",
        "出水 F-(mg/L)",
        "出水F-(mg/L)",
        "出水 F-",
        "出水F-",
        "出水 F",
        "出水F",
        "effluent fluoride",
        "effluent F",
        "effluent F-",
        "effluent F mg/L",
        "effluent F- mg/L",
        "residual fluoride",
        "residual F",
        "residual F-",
        "remaining fluoride",
        "remaining F",
        "remaining F-",
    ],
    "氟去除率_pct": [
        "氟去除率", "除氟率", "F removal", "F removal rate", "fluoride removal",
        "removal rate", "removal efficiency", "RE",
    ],
    "文献编号": [
        "文献编号", "文献来源", "来源编号", "source_id", "source id", "paper_id",
        "paper id", "literature id", "reference id", "study id",
    ],
}


TARGET_EXCLUDE = [
    "出水", "effluent", "剩余", "残余", "remaining", "residual", "浓度",
]


# ============================================================
# 3. Excel loading
# ============================================================

def find_final_excel() -> Path:
    for p in FINAL_EXCEL_CANDIDATES:
        if p.exists():
            return p

    xlsx_files = sorted(RAW_DIR.glob("*.xlsx"))
    preferred = [p for p in xlsx_files if "最终" in p.name or "final" in p.name.lower()]

    if preferred:
        return preferred[0]

    if xlsx_files:
        return xlsx_files[0]

    raise FileNotFoundError(
        "No Excel file found under data/raw. "
        "Please put the final dataset at data/raw/数据集修改最终.xlsx."
    )


def candidate_score(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return -999

    score = 0

    group_col = find_col(df, COLUMN_CANDIDATES["文献编号"])
    target_col = find_col(
        df,
        COLUMN_CANDIDATES["氟去除率_pct"],
        exclude=["出水", "effluent", "残余", "剩余", "remaining", "residual"],
    )
    dose_col = find_col(df, COLUMN_CANDIDATES["剂量_mM"])
    ph_col = find_col(df, COLUMN_CANDIDATES["进水pH"])
    init_col = find_col(df, COLUMN_CANDIDATES["初始氟_mg_L"])

    if group_col is not None:
        score += 50
    if target_col is not None:
        score += 50
    if dose_col is not None:
        score += 15
    if ph_col is not None:
        score += 15
    if init_col is not None:
        score += 15

    score += min(len(df), 2000) // 20

    return score


def load_best_excel_sheet(excel_path: Path) -> tuple[pd.DataFrame, dict]:
    xls = pd.ExcelFile(excel_path)

    attempts = []

    for sheet in xls.sheet_names:
        for header in [0, 1, 2]:
            try:
                df = pd.read_excel(excel_path, sheet_name=sheet, header=header)
                df = df.dropna(how="all")
                df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

                # Drop fully empty columns.
                df = df.dropna(axis=1, how="all")

                if len(df) == 0 or df.shape[1] < 4:
                    continue

                score = candidate_score(df)

                attempts.append({
                    "sheet": sheet,
                    "header": header,
                    "shape": df.shape,
                    "score": score,
                    "df": df,
                })

            except Exception:
                continue

    if not attempts:
        raise RuntimeError(f"Cannot read any valid sheet from {excel_path}")

    attempts = sorted(attempts, key=lambda x: x["score"], reverse=True)
    best = attempts[0]

    if best["score"] < 80:
        log("[WARN] Best sheet score is low. Column detection may be incomplete.")

    meta = {
        "excel_path": str(excel_path),
        "sheet": best["sheet"],
        "header": best["header"],
        "shape": best["shape"],
        "score": best["score"],
    }

    return best["df"].copy(), meta


# ============================================================
# 4. Numeric cleaning
# ============================================================

def extract_first_number(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()

    if s == "":
        return np.nan

    # Common nonnumeric descriptions.
    low = s.lower()
    if low in ["无", "none", "nan", "na", "-", "/", "\\"]:
        return np.nan

    # Room temperature.
    if "室温" in s or "room" in low:
        return 25.0

    # Replace Chinese punctuation.
    s = s.replace("，", ",")
    s = s.replace("－", "-").replace("–", "-").replace("—", "-")
    s = s.replace("−", "-")

    # If range like 5-7, use mean.
    range_match = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)

    if len(range_match) == 0:
        return np.nan

    nums = [float(v) for v in range_match]

    if len(nums) >= 2 and ("-" in s or "~" in s or "～" in s):
        return float(np.mean(nums[:2]))

    return nums[0]


def to_numeric_series(s: pd.Series) -> pd.Series:
    return s.map(extract_first_number).astype(float)


def clean_target_percent(s: pd.Series) -> pd.Series:
    y = to_numeric_series(s)

    valid = y.dropna()

    if len(valid) == 0:
        return y

    # If removal rate is stored as fraction 0-1, convert to percent.
    if valid.quantile(0.95) <= 1.5:
        y = y * 100.0

    return y


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

    s = s.replace(" ", "_")
    return s


def clean_text_value(x, default="未提供"):
    if pd.isna(x):
        return default

    s = str(x).strip()

    if s == "":
        return default

    if s.lower() in ["nan", "na", "none", "null"]:
        return default

    return s


# ============================================================
# 5. Build standardized dataset
# ============================================================

def build_column_mapping(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for std_col, candidates in COLUMN_CANDIDATES.items():
        exclude = []

        if std_col == "氟去除率_pct":
            exclude = ["出水", "effluent", "残余", "剩余", "remaining", "residual"]

        col = find_col(df, candidates, exclude=exclude)

        rows.append({
            "standard_column": std_col,
            "source_column": col if col is not None else "",
            "found": col is not None,
        })

    return pd.DataFrame(rows)


def get_source_col(mapping_df: pd.DataFrame, std_col: str) -> str | None:
    row = mapping_df[mapping_df["standard_column"] == std_col]

    if row.empty:
        return None

    val = str(row.iloc[0]["source_column"]).strip()

    if val == "":
        return None

    return val


def build_standardized_table(raw_df: pd.DataFrame, mapping_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = pd.DataFrame(index=raw_df.index)

    row_audit = pd.DataFrame({
        "source_excel_row_index_0based": np.arange(len(raw_df)),
        "source_excel_row_index_1based": np.arange(len(raw_df)) + 1,
    })

    # Text/categorical fields.
    for std_col in ["混凝剂", "絮凝剂", "干扰离子", "氟形态"]:
        src = get_source_col(mapping_df, std_col)
        if src is not None:
            out[std_col] = raw_df[src].map(lambda x: clean_text_value(x, default="未提供"))
        else:
            out[std_col] = "未提供"

    # Numeric fields.
    numeric_cols = [
        "剂量_mM",
        "絮凝剂剂量",
        "搅拌时间_min",
        "沉淀时间_min",
        "停留时间_min",
        "进水pH",
        "初始氟_mg_L",
        "水温_C",
        "干扰离子浓度_mmol",
        "游离氟占比",
        "出水pH",
        "出水氟_mg_L",
    ]

    for std_col in numeric_cols:
        src = get_source_col(mapping_df, std_col)
        if src is not None:
            out[std_col] = to_numeric_series(raw_df[src])
        else:
            out[std_col] = np.nan

    # If no retention time is provided, use settling time first, then stirring + settling if available.
    if out["停留时间_min"].isna().all():
        if out["沉淀时间_min"].notna().any():
            out["停留时间_min"] = out["沉淀时间_min"]
        elif out["搅拌时间_min"].notna().any():
            out["停留时间_min"] = out["搅拌时间_min"]

    # Target.
    target_src = get_source_col(mapping_df, "氟去除率_pct")
    if target_src is None:
        raise ValueError(
            "Target column 氟去除率_pct was not detected. "
            "Please check the removal-rate column name in the final Excel."
        )

    out["氟去除率_pct"] = clean_target_percent(raw_df[target_src])

    # Literature/source group.
    group_src = get_source_col(mapping_df, "文献编号")
    if group_src is None:
        raise ValueError(
            "文献编号 column was not detected. "
            "Please make sure the final Excel has a literature ID column."
        )

    out["文献编号"] = raw_df[group_src].map(normalize_group_id)

    # Raw source row.
    out["source_excel_row_index_1based"] = np.arange(len(raw_df)) + 1

    # Row audit flags.
    row_audit["target_raw"] = raw_df[target_src]
    row_audit["target_clean"] = out["氟去除率_pct"]
    row_audit["group_raw"] = raw_df[group_src]
    row_audit["group_clean"] = out["文献编号"]
    row_audit["flag_missing_target"] = out["氟去除率_pct"].isna()
    row_audit["flag_target_out_of_0_100"] = ~out["氟去除率_pct"].between(0, 100, inclusive="both")
    row_audit["flag_missing_group"] = out["文献编号"].isna()

    return out, row_audit


def filter_model_rows(df: pd.DataFrame, row_audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = pd.Series(True, index=df.index)

    keep &= df["氟去除率_pct"].notna()
    keep &= df["氟去除率_pct"].between(0, 100, inclusive="both")
    keep &= df["文献编号"].notna()

    # Keep rows even if some inputs are missing. Pipelines can impute.
    # But remove rows with almost no useful input information.
    input_cols_for_presence = [
        "混凝剂",
        "剂量_mM",
        "进水pH",
        "初始氟_mg_L",
        "水温_C",
        "停留时间_min",
        "干扰离子",
        "干扰离子浓度_mmol",
    ]

    presence = pd.Series(0, index=df.index, dtype=int)

    for c in input_cols_for_presence:
        if c not in df.columns:
            continue

        if df[c].dtype == object:
            presence += df[c].astype(str).str.strip().ne("").astype(int)
            presence -= df[c].astype(str).str.strip().eq("未提供").astype(int)
        else:
            presence += df[c].notna().astype(int)

    keep &= presence >= 3

    filtered = df.loc[keep].reset_index(drop=True).copy()
    row_audit = row_audit.copy()
    row_audit["kept_in_model_ready"] = keep.values
    row_audit["drop_reason"] = ""

    row_audit.loc[row_audit["flag_missing_target"], "drop_reason"] += "missing_target;"
    row_audit.loc[row_audit["flag_target_out_of_0_100"], "drop_reason"] += "target_out_of_0_100;"
    row_audit.loc[row_audit["flag_missing_group"], "drop_reason"] += "missing_literature_id;"
    row_audit.loc[presence < 3, "drop_reason"] += "too_few_input_fields;"

    return filtered, row_audit


# ============================================================
# 6. Output construction
# ============================================================

MODEL_INPUT_COLUMNS = [
    "混凝剂",
    "剂量_mM",
    "絮凝剂",
    "絮凝剂剂量",
    "搅拌时间_min",
    "沉淀时间_min",
    "停留时间_min",
    "进水pH",
    "初始氟_mg_L",
    "水温_C",
    "干扰离子",
    "干扰离子浓度_mmol",
    "氟形态",
    "游离氟占比",
    "氟去除率_pct",
]

SOURCE_COLUMNS = [
    "文献编号",
    "source_excel_row_index_1based",
]

FULL_EXTRA_COLUMNS = [
    "出水pH",
    "出水氟_mg_L",
]


def backup_existing_file(path: Path):
    if not path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Main legacy backup requested by workflow.
    legacy_path = path.parent / "model_ready_v1_legacy_1103.csv"

    if path.name == "model_ready_v1.csv" and not legacy_path.exists():
        shutil.copy2(path, legacy_path)
        log(f"[BACKUP] {path} -> {legacy_path}")

    # Timestamped backup for safety.
    ts_path = path.parent / f"{path.stem}_backup_{timestamp}{path.suffix}"
    shutil.copy2(path, ts_path)
    log(f"[BACKUP] {path} -> {ts_path}")

    return ts_path


def write_outputs(filtered: pd.DataFrame, row_audit: pd.DataFrame, mapping_df: pd.DataFrame, meta: dict):
    # Ensure all expected columns exist.
    for c in MODEL_INPUT_COLUMNS + SOURCE_COLUMNS + FULL_EXTRA_COLUMNS:
        if c not in filtered.columns:
            filtered[c] = np.nan

    model_ready = filtered[MODEL_INPUT_COLUMNS].copy()

    # This file is safe for old model scripts because it does NOT contain 文献编号.
    model_ready_path = PROCESSED_DIR / "model_ready_v1.csv"
    backup_existing_file(model_ready_path)
    model_ready.to_csv(model_ready_path, index=False, encoding="utf-8-sig")
    log(f"[WRITE] {model_ready_path} | shape={model_ready.shape}")

    # This file is used for grouped validation.
    model_ready_with_source = filtered[MODEL_INPUT_COLUMNS + SOURCE_COLUMNS].copy()
    model_ready_with_source_path = PROCESSED_DIR / "model_ready_v1_with_source.csv"
    backup_existing_file(model_ready_with_source_path)
    model_ready_with_source.to_csv(model_ready_with_source_path, index=False, encoding="utf-8-sig")
    log(f"[WRITE] {model_ready_with_source_path} | shape={model_ready_with_source.shape}")

    # Full cleaned file for audit and future EDA.
    cleaned_full = filtered[MODEL_INPUT_COLUMNS + FULL_EXTRA_COLUMNS + SOURCE_COLUMNS].copy()
    cleaned_full_path = PROCESSED_DIR / "cleaned_full_with_flags.csv"
    backup_existing_file(cleaned_full_path)
    cleaned_full.to_csv(cleaned_full_path, index=False, encoding="utf-8-sig")
    log(f"[WRITE] {cleaned_full_path} | shape={cleaned_full.shape}")

    # Audit tables.
    mapping_path = TABLE_DIR / "29a_final_dataset_column_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {mapping_path}")

    row_audit_path = TABLE_DIR / "29a_final_dataset_row_audit.csv"
    row_audit.to_csv(row_audit_path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {row_audit_path}")

    group_dist = (
        model_ready_with_source["文献编号"]
        .value_counts()
        .rename_axis("文献编号")
        .reset_index(name="n_rows")
    )
    group_dist["fraction"] = group_dist["n_rows"] / len(model_ready_with_source)

    group_dist_path = TABLE_DIR / "29a_final_dataset_group_distribution.csv"
    group_dist.to_csv(group_dist_path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {group_dist_path}")

    summary_path = TABLE_DIR / "29a_final_dataset_summary.txt"

    lines = []
    lines.append("=" * 100)
    lines.append("29a final dataset rebuild summary")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Excel source:")
    lines.append(f"path   : {meta['excel_path']}")
    lines.append(f"sheet  : {meta['sheet']}")
    lines.append(f"header : {meta['header']}")
    lines.append(f"shape  : {meta['shape']}")
    lines.append(f"score  : {meta['score']}")
    lines.append("")
    lines.append("Output files:")
    lines.append(f"model_ready_v1.csv             : {model_ready.shape}")
    lines.append(f"model_ready_v1_with_source.csv : {model_ready_with_source.shape}")
    lines.append(f"cleaned_full_with_flags.csv    : {cleaned_full.shape}")
    lines.append("")
    lines.append("Important rule:")
    lines.append("model_ready_v1.csv does not include 文献编号, so it is safe for model training.")
    lines.append("model_ready_v1_with_source.csv includes 文献编号 and is used only for grouped validation.")
    lines.append("")
    lines.append("Column mapping:")
    lines.append(mapping_df.to_string(index=False))
    lines.append("")
    lines.append("Row filtering:")
    lines.append(f"raw rows                       : {len(row_audit)}")
    lines.append(f"kept rows                      : {int(row_audit['kept_in_model_ready'].sum())}")
    lines.append(f"dropped rows                   : {int((~row_audit['kept_in_model_ready']).sum())}")
    lines.append(f"missing target rows            : {int(row_audit['flag_missing_target'].sum())}")
    lines.append(f"target out of 0-100 rows        : {int(row_audit['flag_target_out_of_0_100'].sum())}")
    lines.append(f"missing literature-id rows      : {int(row_audit['flag_missing_group'].sum())}")
    lines.append("")
    lines.append("Target summary:")
    y = model_ready["氟去除率_pct"]
    lines.append(f"n      : {y.notna().sum()}")
    lines.append(f"mean   : {y.mean():.4f}")
    lines.append(f"std    : {y.std():.4f}")
    lines.append(f"min    : {y.min():.4f}")
    lines.append(f"median : {y.median():.4f}")
    lines.append(f"max    : {y.max():.4f}")
    lines.append("")
    lines.append("Literature group summary:")
    lines.append(f"n_groups       : {group_dist['文献编号'].nunique()}")
    lines.append(f"largest group  : {group_dist['n_rows'].max()}")
    lines.append(f"smallest group : {group_dist['n_rows'].min()}")
    lines.append(f"median group n : {group_dist['n_rows'].median():.2f}")
    lines.append("")
    lines.append("Top 20 groups:")
    lines.append(group_dist.head(20).to_string(index=False))
    lines.append("")
    lines.append("=" * 100)

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[TEXT] {summary_path}")


# ============================================================
# 7. Main
# ============================================================

def main():
    log("\n" + "=" * 100)
    log("29a Rebuild model_ready_v1.csv from final Excel")
    log("=" * 100)
    log(f"PROJECT_ROOT  = {PROJECT_ROOT}")
    log(f"RAW_DIR       = {RAW_DIR}")
    log(f"PROCESSED_DIR = {PROCESSED_DIR}")
    log(f"TABLE_DIR     = {TABLE_DIR}")
    log("=" * 100)

    excel_path = find_final_excel()
    log(f"\n[LOAD] Final Excel: {excel_path}")

    raw_df, meta = load_best_excel_sheet(excel_path)

    log("\n[OK] Selected Excel sheet:")
    for k, v in meta.items():
        log(f"  {k}: {v}")

    log("\nRaw columns:")
    for i, c in enumerate(raw_df.columns, start=1):
        log(f"  {i:02d}. {c}")

    mapping_df = build_column_mapping(raw_df)

    log("\n=== Column mapping ===")
    print(mapping_df.to_string(index=False))

    # Required columns check.
    required = [
        "混凝剂",
        "剂量_mM",
        "进水pH",
        "初始氟_mg_L",
        "氟去除率_pct",
        "文献编号",
    ]

    missing_required = []

    for c in required:
        row = mapping_df[mapping_df["standard_column"] == c]
        if row.empty or not bool(row.iloc[0]["found"]):
            missing_required.append(c)

    if missing_required:
        raise ValueError(
            "Missing required columns after automatic detection: "
            + ", ".join(missing_required)
            + "\nPlease check outputs/tables/29a_final_dataset_column_mapping.csv after rerun "
              "or rename the columns in the Excel file."
        )

    standardized, row_audit = build_standardized_table(raw_df, mapping_df)
    filtered, row_audit = filter_model_rows(standardized, row_audit)

    log("\n=== Row filtering ===")
    log(f"Raw rows     : {len(standardized)}")
    log(f"Kept rows    : {len(filtered)}")
    log(f"Dropped rows : {len(standardized) - len(filtered)}")

    if len(filtered) < 100:
        raise RuntimeError(
            f"Only {len(filtered)} rows were kept. This is suspicious. "
            "Please check column mapping and target/literature-id columns."
        )

    # Hard leakage check: 文献编号 should not be in model_ready_v1.csv.
    if "文献编号" in MODEL_INPUT_COLUMNS:
        raise RuntimeError("文献编号 must not be included in MODEL_INPUT_COLUMNS.")

    write_outputs(filtered, row_audit, mapping_df, meta)

    log("\n" + "=" * 100)
    log("[DONE] Final model_ready_v1.csv rebuilt successfully.")
    log("=" * 100)
    log("Next step:")
    log("1. Open outputs/tables/29a_final_dataset_summary.txt")
    log("2. Check kept rows, target summary, and literature group distribution")
    log("3. Then rerun 19_selected_feature_fusion.py on the final dataset")
    log("=" * 100)


if __name__ == "__main__":
    main()