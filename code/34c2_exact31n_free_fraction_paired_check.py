# -*- coding: utf-8 -*-
"""
34c2_exact31n_free_fraction_paired_check.py

使用真正的历史 31n_krr_grouped_validation_robustness.py，
在同一个 Python 进程中成对比较：
1. 原始正式数据
2. 游离氟占比尺度修正候选数据

只运行正文最关键的两套 grouped validation：
A. Balanced_5Seed_Current31F
B. RepeatedBalanced_Extended

不会覆盖任何正式数据或历史结果。
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_NAME = "34c2_exact31n_free_fraction_paired_check"

R2_TOL = 0.03
RMSE_TOL = 1.50
MAE_TOL = 1.50

FOCUS = [
    (
        "Balanced_5Seed_Current31F",
        "KRR_Poly",
    ),
    (
        "RepeatedBalanced_Extended",
        "KRR_CatBoost_Weighted_2Model",
    ),
]


def locate_root() -> Path:
    here = Path(__file__).resolve()

    for p in [here.parent, *here.parents]:
        if (p / "data").exists() and (p / "outputs").exists():
            return p

    raise RuntimeError("Cannot locate project root.")


def load_exact_31n(root: Path):
    path = root / "src" / "31n_krr_grouped_validation_robustness.py"

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find exact historical 31N script:\n{path}"
        )

    print("=" * 100)
    print("Loading EXACT historical 31N script")
    print(path)
    print("=" * 100)

    spec = importlib.util.spec_from_file_location(
        "exact31n_for_34c2",
        path,
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules["exact31n_for_34c2"] = module

    spec.loader.exec_module(module)

    return module


def patch_dataset_paths(
    m,
    strict_path: Path,
    source_path: Path,
):
    """
    只切换数据来源。
    31N 原有的：
    - 特征工程
    - 特征空间
    - 分组方式
    - 模型配置
    - 权重搜索
    全部保持不变。
    """

    if not hasattr(m.sf19, "fe18"):
        raise RuntimeError(
            "Cannot find sf19.fe18.DATA_FILE"
        )

    if not hasattr(
        m.s31f,
        "MODEL_READY_WITH_SOURCE_PATH",
    ):
        raise RuntimeError(
            "Cannot find "
            "s31f.MODEL_READY_WITH_SOURCE_PATH"
        )

    m.sf19.fe18.DATA_FILE = strict_path

    m.s31f.MODEL_READY_WITH_SOURCE_PATH = (
        source_path
    )


def prepare_data(m):
    y, X_raw, X_eng, X_mix = (
        m.sf19.prepare_feature_sets()
    )

    y = (
        y
        .reset_index(drop=True)
        .astype(float)
    )

    X_mix = X_mix.reset_index(drop=True)

    feature_spaces, _, _ = (
        m.s31k.make_feature_spaces(
            X_mix,
            expected_n=len(y),
        )
    )

    if m.BEST_FEATURE_SPACE not in feature_spaces:
        raise RuntimeError(
            "Cannot find feature space: "
            f"{m.BEST_FEATURE_SPACE}"
        )

    X = (
        feature_spaces[
            m.BEST_FEATURE_SPACE
        ]
        .reset_index(drop=True)
    )

    groups = m.s31f.load_group_labels(
        expected_n=len(y)
    )

    groups = (
        pd.Series(groups)
        .astype(str)
        .reset_index(drop=True)
    )

    if len(y) != 1011:
        raise RuntimeError(
            f"Expected 1011 rows, got {len(y)}"
        )

    if groups.nunique() != 27:
        raise RuntimeError(
            "Expected 27 source groups, "
            f"got {groups.nunique()}"
        )

    # 31F balanced split scorer requires this context
    m.s31f.set_balanced_split_context(
        X,
        y,
        groups,
    )

    return y, X, groups


def run_two_branches(
    m,
    strict_path: Path,
    source_path: Path,
    dataset_label: str,
):
    print()
    print("#" * 100)
    print("DATASET:", dataset_label)
    print("#" * 100)

    patch_dataset_paths(
        m,
        strict_path,
        source_path,
    )

    y, X, groups = prepare_data(m)

    print(
        "Feature space:",
        m.BEST_FEATURE_SPACE,
    )

    print(
        "Feature matrix:",
        X.shape,
    )

    print(
        "Source groups:",
        groups.nunique(),
    )

    all_tests = []
    all_predictions = []

    # =========================================================
    # A. Balanced_5Seed_Current31F
    # =========================================================

    for seed in m.BALANCED_CURRENT_SEEDS:

        train_idx, valid_idx, test_idx, info = (
            m.s31f.grouped_split_indices(
                groups,
                seed,
            )
        )

        info = dict(info)

        info["Strategy"] = (
            "Balanced_5Seed_Current31F"
        )

        info["Split_ID"] = (
            f"balanced5_seed_{seed}"
        )

        info["Seed_or_Fold"] = seed

        include_extra = (
            info["Strategy"]
            in
            m.INCLUDE_EXTRA_ANTI_CORR_MODELS_IN
        )

        result = m.run_one_split(
            X,
            y,
            groups,
            train_idx,
            valid_idx,
            test_idx,
            info,
            include_extra_models=include_extra,
        )

        all_tests.append(
            result["test"]
        )

        all_predictions.append(
            result["prediction"]
        )

    # =========================================================
    # B. RepeatedBalanced_Extended
    # =========================================================

    for seed in m.REPEATED_BALANCED_SEEDS:

        train_idx, valid_idx, test_idx, info = (
            m.s31f.grouped_split_indices(
                groups,
                seed,
            )
        )

        info = dict(info)

        info["Strategy"] = (
            "RepeatedBalanced_Extended"
        )

        info["Split_ID"] = (
            f"balanced_ext_seed_{seed}"
        )

        info["Seed_or_Fold"] = seed

        include_extra = (
            info["Strategy"]
            in
            m.INCLUDE_EXTRA_ANTI_CORR_MODELS_IN
        )

        result = m.run_one_split(
            X,
            y,
            groups,
            train_idx,
            valid_idx,
            test_idx,
            info,
            include_extra_models=include_extra,
        )

        all_tests.append(
            result["test"]
        )

        all_predictions.append(
            result["prediction"]
        )

    test_df = pd.concat(
        all_tests,
        ignore_index=True,
    )

    prediction_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    summary_df = (
        m.summarize_by_strategy(
            test_df,
            prediction_df,
        )
    )

    return summary_df


def extract_focus(
    df: pd.DataFrame,
    prefix: str,
):
    rows = []

    for strategy, method in FOCUS:

        sub = df[
            (df["Strategy"] == strategy)
            &
            (df["Method"] == method)
        ]

        if len(sub) != 1:
            raise RuntimeError(
                "Cannot uniquely identify "
                f"{strategy} / {method}"
            )

        r = sub.iloc[0]

        rows.append(
            {
                "Strategy": strategy,
                "Method": method,

                f"{prefix}_R2":
                    float(r["Pooled_R2"]),

                f"{prefix}_RMSE":
                    float(r["Pooled_RMSE"]),

                f"{prefix}_MAE":
                    float(r["Pooled_MAE"]),
            }
        )

    return pd.DataFrame(rows)


def main():
    start = time.time()

    root = locate_root()

    data_dir = (
        root
        / "data"
        / "processed"
    )

    official_strict = (
        data_dir
        / "model_ready_v1.csv"
    )

    official_source = (
        data_dir
        / "model_ready_v1_with_source.csv"
    )

    corrected_strict = (
        data_dir
        / "model_ready_v1_free_fraction_fixed_candidate.csv"
    )

    corrected_source = (
        data_dir
        / "model_ready_v1_with_source_free_fraction_fixed_candidate.csv"
    )

    historical_summary = (
        root
        / "outputs"
        / "tables"
        / "31n_grouped_validation_summary.csv"
    )

    for path in [
        official_strict,
        official_source,
        corrected_strict,
        corrected_source,
        historical_summary,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    # =========================================================
    # Load exact historical workflow
    # =========================================================

    m = load_exact_31n(root)

    print()
    print("BEST_FEATURE_SPACE:")
    print(m.BEST_FEATURE_SPACE)

    print()
    print(
        "Balanced seeds:",
        m.BALANCED_CURRENT_SEEDS,
    )

    print(
        "Extended seeds:",
        m.REPEATED_BALANCED_SEEDS,
    )

    # Important reproducibility audit
    print()
    print(
        "Current-process hash-derived model seeds:"
    )

    all_seeds = sorted(
        set(
            m.BALANCED_CURRENT_SEEDS
            +
            m.REPEATED_BALANCED_SEEDS
        )
    )

    for seed in all_seeds:

        derived_seed = int(
            abs(
                hash(str(seed))
            )
            % 100000
        )

        print(
            f"{seed:>4} -> "
            f"{derived_seed:>6}"
        )

    # =========================================================
    # Historical saved result
    # =========================================================

    historical_df = pd.read_csv(
        historical_summary
    )

    historical_focus = extract_focus(
        historical_df,
        "Historical",
    )

    # =========================================================
    # Current-process ORIGINAL data
    # =========================================================

    original_summary = run_two_branches(
        m,
        official_strict,
        official_source,
        "ORIGINAL SCALE",
    )

    original_focus = extract_focus(
        original_summary,
        "Original",
    )

    # =========================================================
    # Current-process CORRECTED data
    # =========================================================

    corrected_summary = run_two_branches(
        m,
        corrected_strict,
        corrected_source,
        "CORRECTED SCALE",
    )

    corrected_focus = extract_focus(
        corrected_summary,
        "Corrected",
    )

    # =========================================================
    # Merge
    # =========================================================

    comp = (
        historical_focus
        .merge(
            original_focus,
            on=[
                "Strategy",
                "Method",
            ]
        )
        .merge(
            corrected_focus,
            on=[
                "Strategy",
                "Method",
            ]
        )
    )

    # Historical reproducibility
    for metric in [
        "R2",
        "RMSE",
        "MAE",
    ]:

        comp[
            f"Original_minus_Historical_{metric}"
        ] = (
            comp[
                f"Original_{metric}"
            ]
            -
            comp[
                f"Historical_{metric}"
            ]
        )

    # Paired scale effect
    for metric in [
        "R2",
        "RMSE",
        "MAE",
    ]:

        comp[
            f"Corrected_minus_Original_{metric}"
        ] = (
            comp[
                f"Corrected_{metric}"
            ]
            -
            comp[
                f"Original_{metric}"
            ]
        )

    comp["Decision"] = np.where(

        (
            comp[
                "Corrected_minus_Original_R2"
            ].abs()
            <= R2_TOL
        )

        &

        (
            comp[
                "Corrected_minus_Original_RMSE"
            ].abs()
            <= RMSE_TOL
        )

        &

        (
            comp[
                "Corrected_minus_Original_MAE"
            ].abs()
            <= MAE_TOL
        ),

        "STABLE",

        "MATERIALLY AFFECTED",
    )

    # =========================================================
    # Save
    # =========================================================

    output_root = (
        root
        / "outputs"
        / "jece_final_package"
    )

    table_dir = (
        output_root
        / "tables"
    )

    report_dir = (
        output_root
        / "reports"
    )

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comp.to_csv(

        table_dir
        /
        "34c2_exact31n_free_fraction_paired_comparison.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # =========================================================
    # Console report
    # =========================================================

    print()
    print("=" * 100)
    print(
        "EXACT 31N FREE-FLUORIDE SCALE PAIRED CHECK"
    )
    print("=" * 100)

    for _, r in comp.iterrows():

        print()
        print(
            r["Strategy"],
            "|",
            r["Method"],
        )

        print()

        print(
            "Historical : "
            f"R2={r['Historical_R2']:.6f} "
            f"RMSE={r['Historical_RMSE']:.6f} "
            f"MAE={r['Historical_MAE']:.6f}"
        )

        print(
            "Original   : "
            f"R2={r['Original_R2']:.6f} "
            f"RMSE={r['Original_RMSE']:.6f} "
            f"MAE={r['Original_MAE']:.6f}"
        )

        print(
            "Corrected  : "
            f"R2={r['Corrected_R2']:.6f} "
            f"RMSE={r['Corrected_RMSE']:.6f} "
            f"MAE={r['Corrected_MAE']:.6f}"
        )

        print()

        print(
            "Original - Historical:"
        )

        print(
            f"dR2={r['Original_minus_Historical_R2']:+.6f}, "
            f"dRMSE={r['Original_minus_Historical_RMSE']:+.6f}, "
            f"dMAE={r['Original_minus_Historical_MAE']:+.6f}"
        )

        print()

        print(
            "Corrected - Original:"
        )

        print(
            f"dR2={r['Corrected_minus_Original_R2']:+.6f}, "
            f"dRMSE={r['Corrected_minus_Original_RMSE']:+.6f}, "
            f"dMAE={r['Corrected_minus_Original_MAE']:+.6f}"
        )

        print()

        print(
            "DECISION:",
            r["Decision"],
        )

    runtime = time.time() - start

    print()
    print(
        f"Completed in {runtime:.2f} s "
        f"({runtime/60:.2f} min)"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()