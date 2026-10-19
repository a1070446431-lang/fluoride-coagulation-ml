# -*- coding: utf-8 -*-
"""
34a_audit_and_fix_free_fluoride_fraction_scale.py

Purpose
-------
Audit and correct the archived scale inconsistency in the variable "游离氟占比"
without overwriting the current official model-ready datasets.

Scientific rule
---------------
The final source-aware dataset is first audited at the source-group level.
A source group is treated as percentage-scaled only when ALL non-missing values
of "游离氟占比" are > 1 and <= 100. Those values are divided by 100.

This script:
1) DOES NOT overwrite model_ready_v1.csv or model_ready_v1_with_source.csv.
2) Preserves the existing raw-backup field "游离氟占比_原始" unchanged.
3) Writes corrected candidate datasets for sensitivity analysis only.
4) Writes a source-group audit table and a text report.
5) Verifies that strict/source-aware matrices remain row-aligned.

Place this script in:
    fluoride_coagulation_ml_project/src/

Run directly in PyCharm.
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_NAME = "34a_audit_and_fix_free_fluoride_fraction_scale"

COL_FRACTION = "游离氟占比"
COL_RAW_BACKUP = "游离氟占比_原始"
COL_GROUP = "文献编号"
COL_ROW = "原始行号"

EXPECTED_RECORDS = 1011
EXPECTED_GROUPS = 27


def locate_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "data").exists() and (p / "outputs").exists():
            return p
    return here.parent.parent


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def fmt(x: float) -> str:
    return f"{x:.6g}"


def main() -> None:
    t0 = time.time()
    root = locate_root()

    strict_path = root / "data" / "processed" / "model_ready_v1.csv"
    source_path = root / "data" / "processed" / "model_ready_v1_with_source.csv"

    require_file(strict_path)
    require_file(source_path)

    out_tables = root / "outputs" / "jece_final_package" / "tables"
    out_reports = root / "outputs" / "jece_final_package" / "reports"
    out_logs = root / "outputs" / "jece_final_package" / "logs"

    for d in (out_tables, out_reports, out_logs):
        d.mkdir(parents=True, exist_ok=True)

    strict = pd.read_csv(strict_path)
    source = pd.read_csv(source_path)

    # ------------------------------------------------------------------
    # Basic structural verification
    # ------------------------------------------------------------------
    if len(strict) != len(source):
        raise RuntimeError(
            f"Row-count mismatch: strict={len(strict)}, source-aware={len(source)}"
        )

    if len(source) != EXPECTED_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDS} rows, observed {len(source)} rows."
        )

    required_source_cols = {COL_FRACTION, COL_GROUP, COL_ROW}
    missing = required_source_cols - set(source.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in source-aware matrix: {missing}")

    if COL_FRACTION not in strict.columns:
        raise RuntimeError(f"{COL_FRACTION} not found in strict matrix.")

    # Row alignment between the two archived matrices.
    if COL_ROW in strict.columns:
        same_row_order = (
            strict[COL_ROW].astype(str).reset_index(drop=True)
            .equals(source[COL_ROW].astype(str).reset_index(drop=True))
        )
        if not same_row_order:
            raise RuntimeError(
                "The strict and source-aware matrices are not aligned by 原始行号. "
                "Stop before any correction."
            )
    else:
        print(
            "[WARNING] Strict matrix has no 原始行号 column; row-order alignment "
            "cannot be independently verified."
        )

    n_groups = source[COL_GROUP].astype(str).nunique()
    if n_groups != EXPECTED_GROUPS:
        print(
            f"[WARNING] Expected {EXPECTED_GROUPS} source groups, observed {n_groups}."
        )

    # ------------------------------------------------------------------
    # Source-group scale audit
    # ------------------------------------------------------------------
    source[COL_FRACTION] = pd.to_numeric(source[COL_FRACTION], errors="coerce")
    strict[COL_FRACTION] = pd.to_numeric(strict[COL_FRACTION], errors="coerce")

    audit_rows = []
    percent_groups = []

    for group, g in source.groupby(COL_GROUP, sort=True):
        vals = g[COL_FRACTION].dropna()

        if len(vals) == 0:
            scale_class = "No data"
            correction = "None"
            min_before = np.nan
            median_before = np.nan
            max_before = np.nan
            gt1_n = 0
        else:
            min_before = float(vals.min())
            median_before = float(vals.median())
            max_before = float(vals.max())
            gt1_n = int((vals > 1).sum())

            all_gt1_le100 = bool((vals > 1).all() and (vals <= 100).all())
            all_le1 = bool((vals >= 0).all() and (vals <= 1).all())

            if all_gt1_le100:
                scale_class = "Percentage scale (0–100)"
                correction = "Divide by 100"
                percent_groups.append(str(group))
            elif all_le1:
                scale_class = "Fraction scale (0–1)"
                correction = "None"
            else:
                scale_class = "Mixed / unresolved"
                correction = "STOP — manual review"

        audit_rows.append(
            {
                "文献编号": group,
                "N": int(len(g)),
                "非缺失N": int(len(vals)),
                "最小值_修正前": min_before,
                "中位数_修正前": median_before,
                "最大值_修正前": max_before,
                ">1数量": gt1_n,
                "尺度判定": scale_class,
                "拟采取处理": correction,
            }
        )

    audit = pd.DataFrame(audit_rows)

    unresolved = audit[audit["尺度判定"] == "Mixed / unresolved"]
    if len(unresolved) > 0:
        print("\n[STOP] Mixed/unresolved source groups were detected:")
        print(
            unresolved[
                ["文献编号", "N", "最小值_修正前", "中位数_修正前", "最大值_修正前"]
            ].to_string(index=False)
        )
        raise RuntimeError(
            "Mixed-scale source groups exist. No corrected dataset has been written."
        )

    # ------------------------------------------------------------------
    # Correct a COPY of the two matrices only
    # ------------------------------------------------------------------
    source_fixed = source.copy()
    strict_fixed = strict.copy()

    mask_pct = source_fixed[COL_GROUP].astype(str).isin(percent_groups)
    corrected_rows = int(mask_pct.sum())

    source_fixed.loc[mask_pct, COL_FRACTION] = (
        source_fixed.loc[mask_pct, COL_FRACTION] / 100.0
    )

    # Strict matrix is aligned row-for-row with source-aware matrix.
    strict_fixed.loc[mask_pct.to_numpy(), COL_FRACTION] = (
        strict_fixed.loc[mask_pct.to_numpy(), COL_FRACTION] / 100.0
    )

    # Preserve the archived raw-value backup unchanged if present.
    if COL_RAW_BACKUP in source.columns:
        if not source_fixed[COL_RAW_BACKUP].equals(source[COL_RAW_BACKUP]):
            raise RuntimeError(f"{COL_RAW_BACKUP} was unexpectedly modified.")
    if COL_RAW_BACKUP in strict.columns:
        if not strict_fixed[COL_RAW_BACKUP].equals(strict[COL_RAW_BACKUP]):
            raise RuntimeError(f"{COL_RAW_BACKUP} was unexpectedly modified.")

    # ------------------------------------------------------------------
    # Post-correction verification
    # ------------------------------------------------------------------
    corrected_vals = source_fixed[COL_FRACTION].dropna()

    post_min = float(corrected_vals.min())
    post_max = float(corrected_vals.max())
    post_gt1 = int((corrected_vals > 1).sum())
    post_lt0 = int((corrected_vals < 0).sum())

    if post_gt1 != 0 or post_lt0 != 0:
        raise RuntimeError(
            "Post-correction physical-range check failed: "
            f">1 rows={post_gt1}, <0 rows={post_lt0}"
        )

    # Update audit with post-correction ranges.
    fixed_group_summary = (
        source_fixed.groupby(COL_GROUP)[COL_FRACTION]
        .agg(["min", "median", "max"])
        .reset_index()
        .rename(
            columns={
                "min": "最小值_修正后",
                "median": "中位数_修正后",
                "max": "最大值_修正后",
            }
        )
    )
    audit = audit.merge(fixed_group_summary, on=COL_GROUP, how="left")

    # ------------------------------------------------------------------
    # Write candidate datasets — official files are untouched.
    # ------------------------------------------------------------------
    source_out = (
        root
        / "data"
        / "processed"
        / "model_ready_v1_with_source_free_fraction_fixed_candidate.csv"
    )
    strict_out = (
        root
        / "data"
        / "processed"
        / "model_ready_v1_free_fraction_fixed_candidate.csv"
    )

    source_fixed.to_csv(source_out, index=False, encoding="utf-8-sig")
    strict_fixed.to_csv(strict_out, index=False, encoding="utf-8-sig")

    audit_out = out_tables / "34a_free_fluoride_fraction_source_group_scale_audit.csv"
    audit.to_csv(audit_out, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # Console/report summary
    # ------------------------------------------------------------------
    before_vals = source[COL_FRACTION].dropna()
    before_gt1 = int((before_vals > 1).sum())
    before_eq100 = int(np.isclose(before_vals, 100.0).sum())

    report_lines = [
        f"SCRIPT={SCRIPT_NAME}",
        f"records={len(source)}",
        f"source_groups={n_groups}",
        f"percentage_scale_groups={len(percent_groups)}",
        f"percentage_scale_group_ids={','.join(percent_groups)}",
        f"rows_corrected={corrected_rows}",
        f"before_min={fmt(float(before_vals.min()))}",
        f"before_median={fmt(float(before_vals.median()))}",
        f"before_max={fmt(float(before_vals.max()))}",
        f"before_gt1_rows={before_gt1}",
        f"before_eq100_rows={before_eq100}",
        f"after_min={fmt(post_min)}",
        f"after_median={fmt(float(corrected_vals.median()))}",
        f"after_max={fmt(post_max)}",
        f"after_gt1_rows={post_gt1}",
        f"after_lt0_rows={post_lt0}",
        f"official_files_overwritten=False",
        f"source_candidate={source_out}",
        f"strict_candidate={strict_out}",
        f"audit_table={audit_out}",
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
    ]

    runtime = time.time() - t0
    report_lines.append(f"runtime_s={runtime:.2f}")
    report = "\n".join(report_lines) + "\n"

    report_path = out_reports / f"{SCRIPT_NAME}_report.txt"
    log_path = out_logs / f"{SCRIPT_NAME}.log"
    report_path.write_text(report, encoding="utf-8")
    log_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 76)
    print("FREE-FLUORIDE FRACTION SCALE AUDIT")
    print("=" * 76)
    print(f"Final records:                  {len(source)}")
    print(f"Source groups:                  {n_groups}")
    print(f"Percentage-scale groups:        {len(percent_groups)}")
    print(f"Rows requiring ÷100 correction: {corrected_rows}")
    print(f"Rows > 1 before correction:     {before_gt1}")
    print(f"Rows = 100 before correction:   {before_eq100}")
    print(f"Rows > 1 after correction:      {post_gt1}")
    print(f"Final corrected range:          {post_min:.6g}–{post_max:.6g}")
    print("\nPercentage-scale groups:")
    print(", ".join(percent_groups))
    print("\nIMPORTANT:")
    print("The official model_ready_v1*.csv files were NOT overwritten.")
    print("Only candidate corrected datasets were generated for the next sensitivity step.")
    print(f"\nCompleted in {runtime:.2f} s")
    print("=" * 76)


if __name__ == "__main__":
    main()
