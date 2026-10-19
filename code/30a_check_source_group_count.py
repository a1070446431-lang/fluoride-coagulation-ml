# -*- coding: utf-8 -*-
"""
30a_check_source_group_count.py

Purpose
-------
Check why grouped validation diagnosis reads fewer literature groups than expected.
This script only audits source labels and does not change any model/data file.

It compares:
1) data/raw/数据集修改最终.xlsx
2) data/processed/model_ready_v1_with_source.csv
3) data/processed/model_ready_v1.csv

Outputs:
outputs/tables/30a_source_group_audit_report.txt
outputs/tables/30a_raw_excel_source_counts.csv
outputs/tables/30a_processed_source_counts.csv
outputs/tables/30a_source_mismatch_rows.csv

Run in PyCharm:
Right click this file -> Run
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


SOURCE_CANDIDATES = ["文献编号", "Paper_ID", "paper_id", "source", "Source", "serials", "Serials"]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def find_project_root() -> Path:
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


def read_csv_smart(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot read CSV file: {path}. Last error: {last_err}")


def normalize_col_name(x) -> str:
    s = str(x).strip().lower()
    s = s.replace(" ", "").replace("_", "")
    s = s.replace("（", "(").replace("）", ")")
    return s


def find_source_col(df: pd.DataFrame) -> Optional[str]:
    for col in SOURCE_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        cn = normalize_col_name(col)
        if any(normalize_col_name(k) in cn for k in ["文献编号", "paper", "source", "serial", "literature", "doi", "study"]):
            return col
    return None


def standardize_paper_id(x) -> str:
    if pd.isna(x):
        return "Paper_Missing"
    text = str(x).strip()
    if text == "":
        return "Paper_Missing"
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


def read_raw_excel_with_source(path: Path) -> Tuple[pd.DataFrame, str, int]:
    """Try likely header rows and choose the one containing source column."""
    best = None
    errors = []
    for header in [1, 0, 2, None]:
        try:
            df = pd.read_excel(path, sheet_name=0, header=header)
            if header is None:
                df.columns = [f"Column_{i+1}" for i in range(df.shape[1])]
            source_col = find_source_col(df)
            if source_col is not None:
                return df, source_col, header if header is not None else -1
            best = (df, df.columns[-1], header if header is not None else -1)
        except Exception as e:
            errors.append(f"header={header}: {e}")
    if best is not None:
        return best
    raise RuntimeError("Cannot read raw Excel. " + "; ".join(errors))


def source_counts(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    out = df[[source_col]].copy()
    out["Source_Raw"] = out[source_col]
    out["Paper_ID_Standardized"] = out[source_col].apply(standardize_paper_id)
    counts = (
        out.groupby(["Paper_ID_Standardized", "Source_Raw"], dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["Paper_ID_Standardized", "Source_Raw"], kind="stable")
    )
    return counts


def main() -> None:
    project_root = find_project_root()
    processed_dir = project_root / "data" / "processed"
    raw_dir = project_root / "data" / "raw"
    table_dir = project_root / "outputs" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    raw_excel = raw_dir / "数据集修改最终.xlsx"
    processed_source_csv = processed_dir / "model_ready_v1_with_source.csv"
    processed_model_csv = processed_dir / "model_ready_v1.csv"

    report_lines = []
    report_lines.append("30a Source Group Audit Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Project root: {project_root}")
    report_lines.append("")

    # Raw Excel
    if raw_excel.exists():
        df_raw, raw_source_col, raw_header = read_raw_excel_with_source(raw_excel)
        raw_counts = source_counts(df_raw, raw_source_col)
        raw_unique = raw_counts["Paper_ID_Standardized"].nunique()
        raw_missing = int((df_raw[raw_source_col].isna() | (df_raw[raw_source_col].astype(str).str.strip() == "")).sum())
        raw_counts.to_csv(table_dir / "30a_raw_excel_source_counts.csv", index=False, encoding="utf-8-sig")

        report_lines.append("[Raw Excel]")
        report_lines.append(f"File: {raw_excel}")
        report_lines.append(f"Read header row setting: header={raw_header if raw_header != -1 else 'None'}")
        report_lines.append(f"Shape: {df_raw.shape}")
        report_lines.append(f"Source column: {raw_source_col}")
        report_lines.append(f"Unique standardized paper groups: {raw_unique}")
        report_lines.append(f"Missing source labels: {raw_missing}")
        report_lines.append("")
    else:
        df_raw = None
        raw_source_col = None
        raw_unique = None
        report_lines.append("[Raw Excel]")
        report_lines.append(f"MISSING: {raw_excel}")
        report_lines.append("")

    # Processed with source
    if processed_source_csv.exists():
        df_processed_source = read_csv_smart(processed_source_csv)
        processed_source_col = find_source_col(df_processed_source)
        if processed_source_col is None:
            processed_source_col = df_processed_source.columns[-1]
        processed_counts = source_counts(df_processed_source, processed_source_col)
        processed_unique = processed_counts["Paper_ID_Standardized"].nunique()
        processed_missing = int((df_processed_source[processed_source_col].isna() | (df_processed_source[processed_source_col].astype(str).str.strip() == "")).sum())
        processed_counts.to_csv(table_dir / "30a_processed_source_counts.csv", index=False, encoding="utf-8-sig")

        report_lines.append("[Processed CSV with source]")
        report_lines.append(f"File: {processed_source_csv}")
        report_lines.append(f"Shape: {df_processed_source.shape}")
        report_lines.append(f"Source column: {processed_source_col}")
        report_lines.append(f"Unique standardized paper groups: {processed_unique}")
        report_lines.append(f"Missing source labels: {processed_missing}")
        report_lines.append("")
    else:
        df_processed_source = None
        processed_source_col = None
        processed_unique = None
        report_lines.append("[Processed CSV with source]")
        report_lines.append(f"MISSING: {processed_source_csv}")
        report_lines.append("")

    # Processed model-ready without source
    if processed_model_csv.exists():
        df_model = read_csv_smart(processed_model_csv)
        report_lines.append("[Processed model-ready CSV]")
        report_lines.append(f"File: {processed_model_csv}")
        report_lines.append(f"Shape: {df_model.shape}")
        report_lines.append("")
    else:
        df_model = None
        report_lines.append("[Processed model-ready CSV]")
        report_lines.append(f"MISSING: {processed_model_csv}")
        report_lines.append("")

    # Row-wise comparison if possible.
    mismatch_df = pd.DataFrame()
    if df_raw is not None and df_processed_source is not None and len(df_raw) == len(df_processed_source):
        raw_std = df_raw[raw_source_col].apply(standardize_paper_id).reset_index(drop=True)
        processed_std = df_processed_source[processed_source_col].apply(standardize_paper_id).reset_index(drop=True)
        mismatch_mask = raw_std != processed_std
        mismatch_df = pd.DataFrame({
            "Row_1_based": np.arange(1, len(raw_std) + 1),
            "Raw_Excel_Source": df_raw[raw_source_col].reset_index(drop=True),
            "Raw_Excel_Standardized": raw_std,
            "Processed_Source": df_processed_source[processed_source_col].reset_index(drop=True),
            "Processed_Standardized": processed_std,
        }).loc[mismatch_mask]
        mismatch_df.to_csv(table_dir / "30a_source_mismatch_rows.csv", index=False, encoding="utf-8-sig")
        report_lines.append("[Row-wise source-label comparison]")
        report_lines.append(f"Comparable rows: {len(raw_std)}")
        report_lines.append(f"Mismatched rows: {len(mismatch_df)}")
        if len(mismatch_df) > 0:
            report_lines.append("First 20 mismatched rows saved in 30a_source_mismatch_rows.csv")
        report_lines.append("")
    elif df_raw is not None and df_processed_source is not None:
        report_lines.append("[Row-wise source-label comparison]")
        report_lines.append("Cannot compare row by row because row counts differ:")
        report_lines.append(f"Raw Excel rows: {len(df_raw)}")
        report_lines.append(f"Processed source CSV rows: {len(df_processed_source)}")
        report_lines.append("")

    # Diagnosis summary.
    report_lines.append("[Diagnosis]")
    if raw_unique is not None and processed_unique is not None:
        if raw_unique > processed_unique:
            report_lines.append(
                "Raw Excel has MORE paper groups than model_ready_v1_with_source.csv. "
                "This means the grouped-validation scripts are probably reading an older processed source file."
            )
            report_lines.append(
                "Action: rerun 29c_rebuild_model_ready_v1_exact_legacy_structure.py from the latest Excel, "
                "then rerun grouped validation screening and diagnosis."
            )
        elif raw_unique < processed_unique:
            report_lines.append(
                "Processed source CSV has MORE paper groups than raw Excel. Check whether the raw Excel path is the latest version."
            )
        else:
            report_lines.append(
                "Raw Excel and processed source CSV have the same unique group count. If the count is still unexpected, "
                "inspect 30a_raw_excel_source_counts.csv for duplicated or missing labels."
            )
    elif raw_unique is None and processed_unique is not None:
        report_lines.append("Only processed source CSV was found; grouped diagnosis will use that file.")
    elif raw_unique is not None and processed_unique is None:
        report_lines.append("Processed source CSV is missing; grouped diagnosis will need to attach Excel source labels by row order.")
    else:
        report_lines.append("No usable source-label file was found.")

    report_lines.append("")
    report_lines.append("Important note:")
    report_lines.append(
        "30_grouped_validation_diagnosis.py reads data/processed/model_ready_v1_with_source.csv first if it exists. "
        "It will NOT automatically use a newer raw Excel source-label column unless the processed source CSV is missing."
    )

    report_path = table_dir / "30a_source_group_audit_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    log("\n".join(report_lines))
    log("")
    log(f"Saved report: {report_path}")
    log(f"Saved counts: {table_dir / '30a_raw_excel_source_counts.csv'}")
    log(f"Saved counts: {table_dir / '30a_processed_source_counts.csv'}")
    if len(mismatch_df) > 0:
        log(f"Saved mismatches: {table_dir / '30a_source_mismatch_rows.csv'}")


if __name__ == "__main__":
    main()
