# -*- coding: utf-8 -*-
"""
34c_free_fluoride_fraction_grouped_validation_sensitivity.py

Purpose
-------
Check whether correcting the mixed scale of "游离氟占比" changes the TWO grouped-validation
results that are actually used in the manuscript:

1) Balanced_5Seed_Current31F + KRR_Poly
   -> exploratory best full-domain grouped result (historically pooled R² ≈ 0.250)

2) RepeatedBalanced_Extended + KRR_CatBoost_Weighted_2Model
   -> audited full-domain baseline used before applicability-domain analysis
      (historically R² ≈ 0.194, RMSE ≈ 27.513, MAE ≈ 22.894)

This script does NOT run GroupKFold, ordinary grouped split, LOLO, or AD.
It deliberately tests only the two manuscript-critical grouped branches first.

It does NOT overwrite any official dataset or historical result.

Place in:
    fluoride_coagulation_ml_project/src/

Prerequisites:
    - 34a has already generated:
        data/processed/model_ready_v1_free_fraction_fixed_candidate.csv
        data/processed/model_ready_v1_with_source_free_fraction_fixed_candidate.csv
    - Existing project scripts, including:
        31n_krr_grouped_validation_robustness_fixed2.py
"""

from __future__ import annotations

import importlib.util
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_NAME = "34c_free_fluoride_fraction_grouped_validation_sensitivity"

EXPECTED_RECORDS = 1011
EXPECTED_GROUPS = 27

# Manuscript reference values used only as an audit cross-check.
REFERENCE_BALANCED5_KRR_R2 = 0.250
REFERENCE_EXTENDED_KRR_CB_R2 = 0.194
REFERENCE_EXTENDED_KRR_CB_RMSE = 27.513
REFERENCE_EXTENDED_KRR_CB_MAE = 22.894

# Sensitivity thresholds: used only to flag material change, not to select models.
R2_TOL = 0.03
RMSE_TOL = 1.5
MAE_TOL = 1.5


def log(msg: str) -> None:
    print(msg, flush=True)


def locate_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "data").exists() and (p / "outputs").exists():
            return p
    return here.parent.parent


def find_file_upwards(start_path: Path, filename: str) -> Path:
    start_path = start_path.resolve()
    for root in [start_path.parent, *start_path.parents]:
        for candidate in (root / filename, root / "src" / filename):
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Cannot find required project script: {filename}")


def import_31n():
    path = find_file_upwards(
        Path(__file__),
        "31n_krr_grouped_validation_robustness_fixed2.py",
    )
    log("=" * 100)
    log("Loading established grouped-validation workflow:")
    log(str(path))
    log("=" * 100)

    spec = importlib.util.spec_from_file_location(
        "s31n_real_for_34c",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["s31n_real_for_34c"] = module
    spec.loader.exec_module(module)
    return module


def verify_candidate_files(
    official_strict: Path,
    corrected_strict: Path,
    official_source: Path,
    corrected_source: Path,
) -> None:
    for path in [official_strict, corrected_strict, official_source, corrected_source]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run 34a_audit_and_fix_free_fluoride_fraction_scale.py first if candidate files are missing."
            )

    os_df = pd.read_csv(official_strict)
    cs_df = pd.read_csv(corrected_strict)
    ow_df = pd.read_csv(official_source)
    cw_df = pd.read_csv(corrected_source)

    if not (len(os_df) == len(cs_df) == len(ow_df) == len(cw_df) == EXPECTED_RECORDS):
        raise RuntimeError(
            "Row counts are inconsistent: "
            f"official strict={len(os_df)}, corrected strict={len(cs_df)}, "
            f"official source={len(ow_df)}, corrected source={len(cw_df)}"
        )

    if "文献编号" not in ow_df.columns or "文献编号" not in cw_df.columns:
        raise RuntimeError("文献编号 is missing from a source-aware matrix.")

    if ow_df["文献编号"].astype(str).nunique() != EXPECTED_GROUPS:
        raise RuntimeError("Official source-aware matrix does not contain the expected 27 groups.")

    if cw_df["文献编号"].astype(str).nunique() != EXPECTED_GROUPS:
        raise RuntimeError("Corrected source-aware matrix does not contain the expected 27 groups.")

    if not ow_df["文献编号"].astype(str).equals(cw_df["文献编号"].astype(str)):
        raise RuntimeError("Source-group order changed between official and corrected source-aware matrices.")

    log("\nCandidate file verification passed:")
    log(f"  records       = {EXPECTED_RECORDS}")
    log(f"  source groups = {EXPECTED_GROUPS}")
    log("  source-group order is unchanged")


def patch_dataset_paths(s31n, strict_path: Path, source_path: Path) -> None:
    # The exact random/grouped project route ultimately reads model features through script 18.
    if not hasattr(s31n.sf19, "fe18"):
        raise RuntimeError("Expected sf19.fe18 was not found; cannot safely patch the model-ready input.")
    s31n.sf19.fe18.DATA_FILE = strict_path

    # Group labels are read by Step 31F from the source-aware file.
    s31n.s31f.MODEL_READY_WITH_SOURCE_PATH = source_path


def build_feature_and_group_data(s31n):
    y, X_raw, X_eng, X_mix_base = s31n.sf19.prepare_feature_sets()
    y = y.reset_index(drop=True).astype(float)
    X_mix_base = X_mix_base.reset_index(drop=True)

    feature_spaces, audit_df, block_df = s31n.s31k.make_feature_spaces(
        X_mix_base,
        expected_n=len(y),
    )

    if s31n.BEST_FEATURE_SPACE not in feature_spaces:
        raise RuntimeError(
            f"Required feature space not found: {s31n.BEST_FEATURE_SPACE}"
        )

    X = feature_spaces[s31n.BEST_FEATURE_SPACE].reset_index(drop=True)

    groups = s31n.s31f.load_group_labels(expected_n=len(y))
    groups = pd.Series(groups).astype(str).reset_index(drop=True)

    if len(y) != EXPECTED_RECORDS or len(groups) != EXPECTED_RECORDS:
        raise RuntimeError("Prepared feature/target/group rows do not equal 1011.")

    if groups.nunique() != EXPECTED_GROUPS:
        raise RuntimeError(f"Expected 27 groups, observed {groups.nunique()}.")

    s31n.s31f.set_balanced_split_context(X, y, groups)

    return y, X, groups


def run_manuscript_critical_grouped_branches(
    s31n,
    strict_path: Path,
    source_path: Path,
    label: str,
):
    log("\n" + "#" * 100)
    log(f"DATASET VERSION: {label}")
    log(f"strict matrix      : {strict_path}")
    log(f"source-aware matrix: {source_path}")
    log("#" * 100)

    patch_dataset_paths(s31n, strict_path, source_path)
    y, X, groups = build_feature_and_group_data(s31n)

    log(f"Feature space: {s31n.BEST_FEATURE_SPACE}")
    log(f"Feature shape: {X.shape}")
    log(f"Groups: {groups.nunique()}")

    tests = []
    preds = []

    # ------------------------------------------------------------------
    # A. Balanced 5Seed branch used for the exploratory full-domain maximum.
    # ------------------------------------------------------------------
    for seed in s31n.BALANCED_CURRENT_SEEDS:
        train_idx, valid_idx, test_idx, info = s31n.s31f.grouped_split_indices(groups, seed)
        info = dict(info)
        info["Strategy"] = "Balanced_5Seed_Current31F"
        info["Split_ID"] = f"balanced5_seed_{seed}"
        info["Seed_or_Fold"] = seed

        result = s31n.run_one_split(
            X,
            y,
            groups,
            train_idx,
            valid_idx,
            test_idx,
            info,
            include_extra_models=False,
        )
        tests.append(result["test"])
        preds.append(result["prediction"])

    # ------------------------------------------------------------------
    # B. RepeatedBalanced_Extended branch used for the audited baseline.
    # ------------------------------------------------------------------
    for seed in s31n.REPEATED_BALANCED_SEEDS:
        train_idx, valid_idx, test_idx, info = s31n.s31f.grouped_split_indices(groups, seed)
        info = dict(info)
        info["Strategy"] = "RepeatedBalanced_Extended"
        info["Split_ID"] = f"balanced_ext_seed_{seed}"
        info["Seed_or_Fold"] = seed

        result = s31n.run_one_split(
            X,
            y,
            groups,
            train_idx,
            valid_idx,
            test_idx,
            info,
            include_extra_models=False,
        )
        tests.append(result["test"])
        preds.append(result["prediction"])

    test_df = pd.concat(tests, ignore_index=True)
    pred_df = pd.concat(preds, ignore_index=True)

    summary = s31n.summarize_by_strategy(test_df, pred_df)

    test_df.insert(0, "Dataset_Version", label)
    pred_df.insert(0, "Dataset_Version", label)
    summary.insert(0, "Dataset_Version", label)

    return test_df, pred_df, summary


def choose_metric_column(df: pd.DataFrame, preferred: list[str]) -> str:
    for c in preferred:
        if c in df.columns:
            return c
    raise KeyError(
        f"Cannot find any metric column among {preferred}. "
        f"Existing columns: {list(df.columns)}"
    )


def extract_focus_rows(summary: pd.DataFrame) -> pd.DataFrame:
    r2_col = choose_metric_column(summary, ["Pooled_R2", "pooled_R2", "R2_pooled"])
    rmse_col = choose_metric_column(summary, ["Pooled_RMSE", "pooled_RMSE", "RMSE_pooled"])
    mae_col = choose_metric_column(summary, ["Pooled_MAE", "pooled_MAE", "MAE_pooled"])

    focus_specs = [
        (
            "Balanced_5Seed_Current31F",
            "KRR_Poly",
            "Exploratory full-domain grouped maximum",
        ),
        (
            "RepeatedBalanced_Extended",
            "KRR_CatBoost_Weighted_2Model",
            "Audited full-domain baseline for AD comparison",
        ),
    ]

    rows = []
    for strategy, method, role in focus_specs:
        sub = summary[
            (summary["Strategy"].astype(str) == strategy)
            & (summary["Method"].astype(str) == method)
        ]
        if len(sub) != 1:
            raise RuntimeError(
                f"Expected one row for strategy={strategy}, method={method}, observed {len(sub)}."
            )
        rr = sub.iloc[0]
        rows.append(
            {
                "Dataset_Version": rr["Dataset_Version"],
                "Role": role,
                "Strategy": strategy,
                "Method": method,
                "Pooled_R2": float(rr[r2_col]),
                "Pooled_RMSE": float(rr[rmse_col]),
                "Pooled_MAE": float(rr[mae_col]),
            }
        )
    return pd.DataFrame(rows)


def compare_focus(original_focus: pd.DataFrame, corrected_focus: pd.DataFrame) -> pd.DataFrame:
    keys = ["Role", "Strategy", "Method"]

    a = original_focus.rename(
        columns={
            "Pooled_R2": "Original_Pooled_R2",
            "Pooled_RMSE": "Original_Pooled_RMSE",
            "Pooled_MAE": "Original_Pooled_MAE",
        }
    ).drop(columns=["Dataset_Version"])

    b = corrected_focus.rename(
        columns={
            "Pooled_R2": "Corrected_Pooled_R2",
            "Pooled_RMSE": "Corrected_Pooled_RMSE",
            "Pooled_MAE": "Corrected_Pooled_MAE",
        }
    ).drop(columns=["Dataset_Version"])

    comp = a.merge(b, on=keys, how="inner")

    comp["Delta_Pooled_R2"] = comp["Corrected_Pooled_R2"] - comp["Original_Pooled_R2"]
    comp["Delta_Pooled_RMSE"] = comp["Corrected_Pooled_RMSE"] - comp["Original_Pooled_RMSE"]
    comp["Delta_Pooled_MAE"] = comp["Corrected_Pooled_MAE"] - comp["Original_Pooled_MAE"]

    comp["Sensitivity_Judgement"] = np.where(
        (comp["Delta_Pooled_R2"].abs() <= R2_TOL)
        & (comp["Delta_Pooled_RMSE"].abs() <= RMSE_TOL)
        & (comp["Delta_Pooled_MAE"].abs() <= MAE_TOL),
        "Stable under scale correction",
        "Materially affected; stop and review",
    )
    return comp


def main() -> None:
    t0 = time.time()
    root = locate_root()

    official_strict = root / "data" / "processed" / "model_ready_v1.csv"
    corrected_strict = (
        root
        / "data"
        / "processed"
        / "model_ready_v1_free_fraction_fixed_candidate.csv"
    )
    official_source = root / "data" / "processed" / "model_ready_v1_with_source.csv"
    corrected_source = (
        root
        / "data"
        / "processed"
        / "model_ready_v1_with_source_free_fraction_fixed_candidate.csv"
    )

    out_tables = root / "outputs" / "jece_final_package" / "tables"
    out_reports = root / "outputs" / "jece_final_package" / "reports"
    out_logs = root / "outputs" / "jece_final_package" / "logs"

    for d in (out_tables, out_reports, out_logs):
        d.mkdir(parents=True, exist_ok=True)

    verify_candidate_files(
        official_strict,
        corrected_strict,
        official_source,
        corrected_source,
    )

    s31n = import_31n()

    original_test, original_pred, original_summary = run_manuscript_critical_grouped_branches(
        s31n,
        official_strict,
        official_source,
        "Original_archived_scale",
    )

    corrected_test, corrected_pred, corrected_summary = run_manuscript_critical_grouped_branches(
        s31n,
        corrected_strict,
        corrected_source,
        "Corrected_free_fraction_scale",
    )

    all_summary = pd.concat(
        [original_summary, corrected_summary],
        ignore_index=True,
    )
    all_test = pd.concat(
        [original_test, corrected_test],
        ignore_index=True,
    )

    original_focus = extract_focus_rows(original_summary)
    corrected_focus = extract_focus_rows(corrected_summary)
    comparison = compare_focus(original_focus, corrected_focus)

    all_summary.to_csv(
        out_tables / "34c_grouped_scale_sensitivity_all_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_test.to_csv(
        out_tables / "34c_grouped_scale_sensitivity_split_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        out_tables / "34c_grouped_scale_sensitivity_key_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Reference-value cross-checks.
    row_best = comparison[
        comparison["Method"] == "KRR_Poly"
    ].iloc[0]
    row_audit = comparison[
        comparison["Method"] == "KRR_CatBoost_Weighted_2Model"
    ].iloc[0]

    all_stable = bool(
        (comparison["Sensitivity_Judgement"] == "Stable under scale correction").all()
    )

    report_lines = [
        "=" * 100,
        "34c FREE-FLUORIDE SCALE SENSITIVITY — MANUSCRIPT-CRITICAL GROUPED VALIDATION",
        "=" * 100,
        "",
        "A. Balanced_5Seed_Current31F + KRR_Poly",
        f"Original pooled R2    = {row_best['Original_Pooled_R2']:.6f}",
        f"Corrected pooled R2   = {row_best['Corrected_Pooled_R2']:.6f}",
        f"Delta pooled R2       = {row_best['Delta_Pooled_R2']:+.6f}",
        f"Original pooled RMSE  = {row_best['Original_Pooled_RMSE']:.6f}",
        f"Corrected pooled RMSE = {row_best['Corrected_Pooled_RMSE']:.6f}",
        f"Delta pooled RMSE     = {row_best['Delta_Pooled_RMSE']:+.6f}",
        f"Original pooled MAE   = {row_best['Original_Pooled_MAE']:.6f}",
        f"Corrected pooled MAE  = {row_best['Corrected_Pooled_MAE']:.6f}",
        f"Delta pooled MAE      = {row_best['Delta_Pooled_MAE']:+.6f}",
        f"Original R2 - manuscript reference 0.250 = {row_best['Original_Pooled_R2'] - REFERENCE_BALANCED5_KRR_R2:+.6f}",
        "",
        "B. RepeatedBalanced_Extended + KRR_CatBoost_Weighted_2Model",
        f"Original pooled R2    = {row_audit['Original_Pooled_R2']:.6f}",
        f"Corrected pooled R2   = {row_audit['Corrected_Pooled_R2']:.6f}",
        f"Delta pooled R2       = {row_audit['Delta_Pooled_R2']:+.6f}",
        f"Original pooled RMSE  = {row_audit['Original_Pooled_RMSE']:.6f}",
        f"Corrected pooled RMSE = {row_audit['Corrected_Pooled_RMSE']:.6f}",
        f"Delta pooled RMSE     = {row_audit['Delta_Pooled_RMSE']:+.6f}",
        f"Original pooled MAE   = {row_audit['Original_Pooled_MAE']:.6f}",
        f"Corrected pooled MAE  = {row_audit['Corrected_Pooled_MAE']:.6f}",
        f"Delta pooled MAE      = {row_audit['Delta_Pooled_MAE']:+.6f}",
        f"Original R2 - manuscript reference 0.194 = {row_audit['Original_Pooled_R2'] - REFERENCE_EXTENDED_KRR_CB_R2:+.6f}",
        f"Original RMSE - manuscript reference 27.513 = {row_audit['Original_Pooled_RMSE'] - REFERENCE_EXTENDED_KRR_CB_RMSE:+.6f}",
        f"Original MAE - manuscript reference 22.894 = {row_audit['Original_Pooled_MAE'] - REFERENCE_EXTENDED_KRR_CB_MAE:+.6f}",
        "",
        "Sensitivity thresholds:",
        f"|Delta R2| <= {R2_TOL}",
        f"|Delta RMSE| <= {RMSE_TOL}",
        f"|Delta MAE| <= {MAE_TOL}",
        "",
        (
            "DECISION: BOTH MANUSCRIPT-CRITICAL GROUPED RESULTS ARE STABLE UNDER SCALE CORRECTION."
            if all_stable
            else
            "DECISION: AT LEAST ONE MANUSCRIPT-CRITICAL GROUPED RESULT IS MATERIALLY AFFECTED. STOP BEFORE AD."
        ),
        "",
        "No applicability-domain result was changed in this step.",
    ]

    runtime = time.time() - t0
    report_lines += [
        "",
        f"runtime_s={runtime:.2f}",
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        "=" * 100,
    ]

    report = "\n".join(report_lines) + "\n"

    (out_reports / f"{SCRIPT_NAME}_decision.txt").write_text(
        report,
        encoding="utf-8",
    )
    (out_logs / f"{SCRIPT_NAME}.log").write_text(
        report,
        encoding="utf-8",
    )

    print("\n" + report)
    print("\nKey comparison:")
    print(comparison.to_string(index=False))
    print(f"\nCompleted in {runtime:.2f} s")


if __name__ == "__main__":
    main()
