# -*- coding: utf-8 -*-
"""Finalize SI numbering, captions, cross-references, and gap audit for Paper 1.

Place this complete file in ``fluoride_coagulation_ml_project/src/`` and run it.
It reads the locked processed matrices and the six completed SI modules, creates
only Figure S1-S2 plus Table S1-S3, and maps all existing scientific outputs to
Supplementary Text S1-S7, Table S1-S20, and Figure S1-S18. Existing SI files,
main figures, model results, predictions, and data are never renamed or edited.

Dependencies: Python >= 3.9, numpy, pandas, matplotlib.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


STEM = "SI_final_numbering_caption_manifest_v1"
N_EXPECTED = 1011
SOURCE_EXPECTED = 27
PREDICTORS = [
    "氟形态", "游离氟占比", "初始氟_mg_L", "干扰离子", "干扰离子浓度_mmol_L",
    "水温_C", "混凝剂", "剂量_mM", "进水pH", "停留时间_min",
]
NUMERIC_ANALYSIS = [
    "游离氟占比", "初始氟_mg_L", "干扰离子浓度_mmol_L", "水温_C",
    "剂量_mM", "进水pH", "停留时间_min", "氟去除率_pct",
]
TARGET = "氟去除率_pct"
SOURCE = "文献编号"
ROW_ID = "原始行号"

DISPLAY = {
    "氟形态": "Fluoride form", "游离氟占比": "Free-fluoride fraction",
    "初始氟_mg_L": "Influent fluoride (mg L-1)", "干扰离子": "Interferent label",
    "干扰离子浓度_mmol_L": "Interferent concentration (mmol L-1)",
    "水温_C": "Water temperature (deg C)", "混凝剂": "Coagulant",
    "剂量_mM": "Coagulant dose (mM)", "进水pH": "Influent pH",
    "停留时间_min": "Contact time (min)", "出水pH": "Effluent pH",
    "氟去除率_pct": "Fluoride removal (%)", "出水氟_mg_L": "Effluent fluoride (mg L-1)",
    "原始行号": "Original row ID", "文献编号": "Source ID",
    "由浓度反算去除率_pct": "Removal reconstructed from concentrations (%)",
    "去除率差值_pct": "Removal-rate discrepancy (%)",
}

UNITS = {
    "游离氟占比": "fraction", "初始氟_mg_L": "mg L-1",
    "干扰离子浓度_mmol_L": "mmol L-1", "水温_C": "deg C",
    "剂量_mM": "mM", "进水pH": "dimensionless", "停留时间_min": "min",
    "出水pH": "dimensionless", "氟去除率_pct": "%", "出水氟_mg_L": "mg L-1",
    "由浓度反算去除率_pct": "%", "去除率差值_pct": "%",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_paths():
    root = Path(__file__).resolve().parents[1]
    cfg = root / "src" / "config.py"
    if not cfg.is_file():
        raise FileNotFoundError(f"Required project configuration not found: {cfg}")
    spec = importlib.util.spec_from_file_location("_si_final_config", cfg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configured_root = Path(module.PROJECT_ROOT).resolve()
    if configured_root != root:
        raise ValueError(f"config.py PROJECT_ROOT mismatch: {configured_root}; script root: {root}")
    processed = Path(module.PROCESSED_DIR)
    if not processed.is_absolute():
        processed = root / processed
    package = root / "outputs" / "jece_final_package"
    outputs = {name: package / name for name in ("figures", "tables", "reports", "logs")}
    for path in outputs.values():
        path.mkdir(parents=True, exist_ok=True)
    inputs = {
        "strict": processed / "model_ready_v1.csv",
        "source": processed / "model_ready_v1_with_source.csv",
    }
    missing = [str(p) for p in inputs.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required processed matrices are missing; no fallback data are permitted:\n" + "\n".join(missing))
    return root, cfg, inputs, outputs


def load_data(inputs):
    strict = pd.read_csv(inputs["strict"], encoding="utf-8-sig")
    source = pd.read_csv(inputs["source"], encoding="utf-8-sig")
    if strict.shape != (N_EXPECTED, 26) or source.shape != (N_EXPECTED, 27):
        raise ValueError(f"Expected 1011 x 26 and 1011 x 27; got {strict.shape} and {source.shape}")
    if source[SOURCE].nunique(dropna=False) != SOURCE_EXPECTED or source[SOURCE].isna().any():
        raise ValueError("The source-bearing matrix must contain 27 non-missing source groups.")
    if list(strict.columns) + [SOURCE] != list(source.columns):
        raise ValueError("The source-bearing matrix must equal the strict matrix plus final source-ID column.")
    if not strict.equals(source[strict.columns]):
        raise ValueError("The two processed matrices are not row-wise identical in their shared columns.")
    if not set(PREDICTORS + [TARGET, SOURCE, ROW_ID]).issubset(source.columns):
        raise ValueError("Required predictors, response, source ID, or row ID are missing.")
    if len(PREDICTORS) != 10 or len(set(PREDICTORS)) != 10:
        raise AssertionError("Exactly ten unique formal predictors are required.")
    forbidden = {SOURCE, ROW_ID, "出水pH", "出水氟_mg_L", "由浓度反算去除率_pct", "去除率差值_pct"}
    if forbidden.intersection(PREDICTORS) or TARGET in PREDICTORS:
        raise AssertionError("Leakage-control failure: a forbidden or target-derived field entered PREDICTORS.")
    return strict, source


def variable_role(column: str):
    if column in PREDICTORS:
        return "Formal predictor", "Yes", "Available before treatment/prediction"
    if column == TARGET:
        return "Primary response", "No", "Prediction target only"
    if column == SOURCE:
        return "Validation/diagnostic only", "No", "Grouped validation, LOLO, source diagnostics, and AD audit only"
    if column == ROW_ID:
        return "Audit/index only", "No", "Traceability only"
    if column.endswith("原始"):
        return "Raw-value audit only", "No", "Provenance and preprocessing audit only"
    if column in {"出水pH", "出水氟_mg_L"}:
        return "Post-treatment audit only", "No", "Unavailable at prediction time; excluded to prevent leakage"
    if column in {"由浓度反算去除率_pct", "去除率差值_pct"}:
        return "Target-derived audit only", "No", "Mathematically linked to the response; excluded to prevent leakage"
    return "Audit only", "No", "Not part of the formal prediction matrix"


def description(column: str) -> str:
    if column in DISPLAY:
        return DISPLAY[column]
    base = column.removesuffix("原始").rstrip("_")
    if column.endswith("原始"):
        return f"Original recorded value of {DISPLAY.get(base, base)}"
    return column


def make_table_s1(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, column in enumerate(data.columns, start=1):
        role, included, control = variable_role(column)
        rows.append({
            "storage_order": i,
            "variable": column,
            "display_name": description(column),
            "data_type": "categorical" if data[column].dtype == object else "numeric",
            "unit": UNITS.get(column.removesuffix("原始"), "not applicable"),
            "modeling_role": role,
            "formal_prediction_input": included,
            "leakage_or_provenance_control": control,
        })
    table = pd.DataFrame(rows)
    if len(table) != 27 or (table.formal_prediction_input == "Yes").sum() != 10:
        raise AssertionError("Table S1 must document all 27 stored fields and exactly 10 formal predictors.")
    return table


def make_table_s2(strict: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    duplicate_rows = int(data.duplicated(subset=PREDICTORS + [TARGET]).sum())
    checks = [
        ("strict_matrix_rows", len(strict), N_EXPECTED, len(strict) == N_EXPECTED),
        ("strict_matrix_columns", strict.shape[1], 26, strict.shape[1] == 26),
        ("source_matrix_rows", len(data), N_EXPECTED, len(data) == N_EXPECTED),
        ("source_matrix_columns", data.shape[1], 27, data.shape[1] == 27),
        ("source_groups", data[SOURCE].nunique(), SOURCE_EXPECTED, data[SOURCE].nunique() == SOURCE_EXPECTED),
        ("formal_predictors", len(PREDICTORS), 10, len(PREDICTORS) == 10),
        ("missing_source_ids", int(data[SOURCE].isna().sum()), 0, not data[SOURCE].isna().any()),
        ("missing_target_values", int(data[TARGET].isna().sum()), 0, not data[TARGET].isna().any()),
        ("target_below_0_pct", int((data[TARGET] < 0).sum()), 0, bool((data[TARGET] >= 0).all())),
        ("target_above_100_pct", int((data[TARGET] > 100).sum()), 0, bool((data[TARGET] <= 100).all())),
        ("duplicate_predictor_target_rows", duplicate_rows, "reported, not deleted", True),
        ("forbidden_fields_in_predictors", 0, 0, True),
        ("shared_columns_rowwise_identical", int(strict.equals(data[strict.columns])), 1, strict.equals(data[strict.columns])),
    ]
    for column in data.columns:
        checks.append((f"missing_values::{column}", int(data[column].isna().sum()), "reported", True))
    return pd.DataFrame(checks, columns=["audit_item", "observed", "expected_or_policy", "pass"])


def make_table_s3(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in NUMERIC_ANALYSIS:
        values = pd.to_numeric(data[column], errors="coerce")
        valid = values.dropna()
        sources = data.loc[values.notna(), SOURCE].nunique()
        q = valid.quantile([.05, .25, .50, .75, .95])
        rows.append({
            "variable": column, "display_name": DISPLAY[column], "unit": UNITS[column],
            "n": int(valid.size), "missing_n": int(values.isna().sum()), "n_sources": int(sources),
            "min": valid.min(), "p05": q.loc[.05], "q25": q.loc[.25], "median": q.loc[.50],
            "mean": valid.mean(), "q75": q.loc[.75], "p95": q.loc[.95], "max": valid.max(),
            "sd": valid.std(ddof=1), "iqr": q.loc[.75] - q.loc[.25],
        })
    return pd.DataFrame(rows)


def save_figure(fig, stem: Path):
    fig.savefig(stem.with_suffix(".png"), dpi=450, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def create_figure_s1(data: pd.DataFrame, table_s1: pd.DataFrame, path: Path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.bbox": "tight",
    })
    roles = ["Formal predictor", "Primary response", "Validation/diagnostic only",
             "Post-treatment audit only", "Target-derived audit only", "Raw-value audit only", "Audit/index only"]
    colors = {"Formal predictor": "#2878B5", "Primary response": "#D65F5F",
              "Validation/diagnostic only": "#7A5195", "Post-treatment audit only": "#E39C37",
              "Target-derived audit only": "#B7503C", "Raw-value audit only": "#8C8C8C",
              "Audit/index only": "#5C677D"}
    role_counts = table_s1.modeling_role.value_counts().reindex(roles, fill_value=0)
    missing_n = data.isna().sum()
    completeness = data.notna().mean().mul(100)
    order = table_s1.assign(
        completeness=table_s1.variable.map(completeness),
        missing_n=table_s1.variable.map(missing_n),
    ).sort_values(["completeness", "storage_order"], ascending=[True, True])

    fig = plt.figure(figsize=(12.0, 7.1))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], left=.12, right=.97, bottom=.12, top=.91, wspace=.34)
    ax, bx = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    y = np.arange(len(order))
    bar_colors = [colors[r] for r in order.modeling_role]
    ax.barh(y, order.completeness, color=bar_colors, height=.68, edgecolor="white", linewidth=.4)
    ax.set_yticks(y, [label.replace("Original recorded value of ", "Raw: ") for label in order.display_name])
    ax.invert_yaxis()
    ax.set_xlim(0, 132)
    ax.set_xlabel("Recorded values (%)")
    ax.set_title("a  Field completeness and modeling role", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#D9D9D9", linewidth=.6, alpha=.75)
    ax.set_axisbelow(True)
    for yy, value, nmiss in zip(y, order.completeness, order.missing_n):
        ax.text(value + .7, yy, f"{value:.1f}% (missing n={int(nmiss)})", va="center", fontsize=7.2)

    yy = np.arange(len(roles))
    bx.barh(yy, role_counts, color=[colors[r] for r in roles], height=.62)
    bx.set_yticks(yy, [r.replace("/", " / ") for r in roles])
    bx.invert_yaxis()
    bx.set_xlabel("Number of stored fields")
    bx.set_title("b  Leakage-controlled field allocation", loc="left", fontweight="bold")
    bx.set_xlim(0, max(role_counts.max() + 2, 12))
    bx.grid(axis="x", color="#D9D9D9", linewidth=.6, alpha=.75)
    bx.set_axisbelow(True)
    for y0, value in zip(yy, role_counts):
        bx.text(value + .18, y0, str(int(value)), va="center", fontweight="bold", fontsize=8.5)
    bx.text(.02, -.14, "10 formal predictors; source ID is validation-only;\npost-treatment and target-derived fields are excluded.",
            transform=bx.transAxes, ha="left", va="top", fontsize=8.3, color="#333333")
    fig.suptitle("Supplementary data architecture and leakage-control audit", fontsize=13, fontweight="bold", y=.975)
    save_figure(fig, path)


def create_figure_s2(table_s3: pd.DataFrame, path: Path):
    fig = plt.figure(figsize=(11.6, 6.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], left=.13, right=.97, bottom=.17, top=.88, wspace=.32)
    ax, bx = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    tab = table_s3.copy()
    labels = tab.display_name.str.replace(" (mg L-1)", "", regex=False).str.replace(" (mmol L-1)", "", regex=False)
    y = np.arange(len(tab))
    denominator = (tab.p95 - tab.p05).replace(0, np.nan)
    q25 = ((tab.q25 - tab.p05) / denominator).fillna(.5)
    med = ((tab["median"] - tab.p05) / denominator).fillna(.5)
    q75 = ((tab.q75 - tab.p05) / denominator).fillna(.5)
    ax.hlines(y, 0, 1, color="#B8C4CC", linewidth=2.2)
    ax.hlines(y, q25, q75, color="#2878B5", linewidth=8, alpha=.82)
    ax.scatter(med, y, s=42, color="#B7503C", edgecolor="white", linewidth=.7, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-.03, 1.03)
    ax.set_xticks([0, .25, .5, .75, 1], ["P5", "", "Within-variable robust scale", "", "P95"])
    ax.set_title("a  Robust within-variable distributions", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E1E1E1", linewidth=.6)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=42, color="#B7503C", label="Median")
    ax.plot([], [], color="#2878B5", linewidth=8, label="IQR")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.13), fontsize=8, ncol=2)

    coverage = tab.n_sources.astype(int)
    obs = tab.n.astype(int)
    bx.barh(y, coverage, color="#3E8E9E", height=.62)
    bx.set_yticks(y, [])
    bx.tick_params(axis="y", length=0)
    bx.invert_yaxis()
    bx.set_xlim(0, SOURCE_EXPECTED + 5.3)
    bx.set_xticks([0, 9, 18, 27])
    bx.set_xlabel("Sources with recorded values")
    bx.set_title("b  Cross-source measurement coverage", loc="left", fontweight="bold")
    bx.grid(axis="x", color="#E1E1E1", linewidth=.6)
    bx.set_axisbelow(True)
    for yy, nsrc, nobs in zip(y, coverage, obs):
        bx.text(nsrc + .35, yy, f"{nsrc}/27; n={nobs}", va="center", fontsize=8)
    fig.suptitle("Continuous-variable support and source coverage", fontsize=13, fontweight="bold", y=.965)
    save_figure(fig, path)


def table_specs(stem: str):
    return [
        (1, "Variable dictionary, modeling roles, and leakage controls", [f"{stem}_Table_S1_variable_dictionary.csv"], "Methods 2.1-2.2"),
        (2, "Processed-matrix quality-control audit", [f"{stem}_Table_S2_quality_control_audit.csv"], "Methods 2.1"),
        (3, "Descriptive statistics and source coverage for continuous variables", [f"{stem}_Table_S3_continuous_statistics.csv"], "Results 3.1"),
        (4, "Categorical-variable composition", ["SI_categorical_composition_source_coverage_v1_category_summary.csv", "SI_categorical_composition_source_coverage_v1_variable_summary.csv"], "Results 3.1"),
        (5, "Categorical levels across literature sources", ["SI_categorical_composition_source_coverage_v1_source_cross_long.csv", "SI_categorical_composition_source_coverage_v1_source_sizes.csv", "SI_categorical_composition_source_coverage_v1_fluoride_form_source_counts.csv", "SI_categorical_composition_source_coverage_v1_interferent_source_counts.csv", "SI_categorical_composition_source_coverage_v1_coagulant_source_counts.csv"], "Results 3.1"),
        (6, "Exact model configurations recovered from the analysis record", ["SI_model_parameters_and_search_spaces_v1_explicit_model_parameters.csv", "SI_model_parameters_and_search_spaces_v1_grouped_exact_configurations.csv", "SI_model_parameters_and_search_spaces_v1_non_tree_exact_configurations.csv"], "Methods 2.3"),
        (7, "Historical hyperparameter search spaces and listed ranges", ["SI_model_parameters_and_search_spaces_v1_historical_search_spaces.csv", "SI_model_parameters_and_search_spaces_v1_grouped_listed_value_ranges.csv"], "Methods 2.3"),
        (8, "Selected settings, ensemble weights, and configuration evidence", ["SI_model_parameters_and_search_spaces_v1_random_selected_settings.csv", "SI_model_parameters_and_search_spaces_v1_random_selected_weights.csv", "SI_model_parameters_and_search_spaces_v1_grouped_selected_settings.csv", "SI_model_parameters_and_search_spaces_v1_selection_crosscheck.csv", "SI_model_parameters_and_search_spaces_v1_source_manifest.csv", "SI_model_parameters_and_search_spaces_v1_source_code_evidence.csv"], "Methods 2.3-2.4"),
        (9, "Complete seed-level random-split metrics", ["SI_random_split_complete_results_v1_seed_metrics.csv", "SI_random_split_complete_results_v1_seed_summary.csv"], "Results 3.2"),
        (10, "Pooled random-split performance and residual statistics", ["SI_random_split_complete_results_v1_pooled_metrics.csv", "SI_random_split_complete_results_v1_residual_statistics.csv"], "Results 3.2"),
        (11, "Random-split candidates, prediction events, and selection audit", ["SI_random_split_complete_results_v1_complete_validation_candidates.csv", "SI_random_split_complete_results_v1_recorded_prediction_events.csv", "SI_random_split_complete_results_v1_validation_selection_audit.csv", "SI_random_split_complete_results_v1_metric_consistency_checks.csv"], "Results 3.2"),
        (12, "Grouped-validation split and pooled performance", ["SI_grouped_validation_lolo_complete_results_v1_original_split_summary.csv", "SI_grouped_validation_lolo_complete_results_v1_original_summary.csv", "SI_grouped_validation_lolo_complete_results_v1_original_test_results.csv", "SI_grouped_validation_lolo_complete_results_v1_recomputed_split_metrics.csv", "SI_grouped_validation_lolo_complete_results_v1_recomputed_pooled_metrics.csv"], "Results 3.3"),
        (13, "Source-level grouped-validation and LOLO metrics", ["SI_grouped_validation_lolo_complete_results_v1_original_paper_metrics.csv", "SI_grouped_validation_lolo_complete_results_v1_original_paper_macro_summary.csv", "SI_grouped_validation_lolo_complete_results_v1_recomputed_paper_metrics.csv"], "Results 3.3"),
        (14, "LOLO predictions and residual diagnostics", ["SI_grouped_validation_lolo_complete_results_v1_original_predictions.csv", "SI_grouped_validation_lolo_complete_results_v1_original_residual_correlation_by_strategy.csv"], "Results 3.3"),
        (15, "Grouped-validation configurations, weights, and source-isolation checks", ["SI_grouped_validation_lolo_complete_results_v1_original_validation_candidates.csv", "SI_grouped_validation_lolo_complete_results_v1_original_fusion_weights.csv", "SI_grouped_validation_lolo_complete_results_v1_source_isolation_audit.csv", "SI_grouped_validation_lolo_complete_results_v1_numeric_checks.csv"], "Methods 2.4; Results 3.3"),
        (16, "Exact selective-AD policy and retained predictions", ["SI_ad_threshold_coverage_sensitivity_audit_v1_exact_policy_metrics.csv", "SI_ad_threshold_coverage_sensitivity_audit_v1_retained_prediction_events.csv", "SI_ad_threshold_coverage_sensitivity_audit_v1_retained_unique_chemistry_rows.csv", "SI_ad_threshold_coverage_sensitivity_audit_v1_ad_component_summary.csv"], "Results 3.4"),
        (17, "AD threshold sensitivity, stability, transfer, and bootstrap uncertainty", ["SI_ad_threshold_coverage_sensitivity_audit_v1_threshold_sensitivity.csv", "SI_ad_threshold_coverage_sensitivity_audit_v1_threshold_paper_dominance.csv", "SI_ad_threshold_coverage_sensitivity_audit_v1_split_stability.csv", "SI_ad_threshold_coverage_sensitivity_audit_v1_strategy_method_transfer.csv", "SI_ad_threshold_coverage_sensitivity_audit_v1_event_level_bootstrap_summary.csv", "SI_ad_threshold_coverage_sensitivity_audit_v1_event_level_bootstrap_raw.csv", "SI_ad_threshold_coverage_sensitivity_audit_v1_numeric_consistency_checks.csv"], "Results 3.4"),
        (18, "Selective-domain chemistry and representativeness", ["SI_ad_threshold_coverage_sensitivity_audit_v1_numeric_operating_window.csv", "SI_ad_threshold_coverage_sensitivity_audit_v1_categorical_composition.csv", "SI_ad_threshold_coverage_sensitivity_audit_v1_representativeness.csv"], "Results 3.4"),
        (19, "SHAP importance across interpretation contexts", ["SI_shap_pdp_extended_interpretation_v1_feature_context_metrics.csv", "SI_shap_pdp_extended_interpretation_v1_feature_group_contributions.csv", "SI_shap_pdp_extended_interpretation_v1_context_rank_correlations.csv"], "Results 3.5"),
        (20, "PDP response profiles and empirical support", ["SI_shap_pdp_extended_interpretation_v1_pdp_surface_summary.csv", "SI_shap_pdp_extended_interpretation_v1_pdp_marginal_profiles.csv", "SI_shap_pdp_extended_interpretation_v1_pdp_paired_support_grid.csv"], "Results 3.5"),
    ]


def figure_specs(stem: str):
    return [
        (1, "Supplementary data architecture and leakage-control audit", f"{stem}_Figure_S1_data_roles_qc", "Methods 2.1-2.2", "Field-level completeness and modeling roles in the source-bearing matrix. (a) Percentage of recorded values for every stored field, with the missing count annotated and colors denoting analytical role. (b) Allocation of the 27 stored fields among formal prediction inputs, response, validation-only fields, and audit-only fields. Source identity, post-treatment measurements, target-derived quantities, raw values, and row identifiers were excluded from formal prediction inputs."),
        (2, "Continuous-variable support and source coverage", f"{stem}_Figure_S2_continuous_support_audit", "Results 3.1", "Support of the seven continuous formal predictors and the response. (a) Within-variable robust distributions normalized from the 5th to the 95th percentile; thick segments show the interquartile range and points show medians. (b) Number of literature sources and records contributing non-missing observations. Normalization in panel a is descriptive only and was not used for model fitting."),
        (3, "Fluoride-form composition and source coverage", "SI_categorical_composition_source_coverage_v1_fluoride_form_part1", "Results 3.1", "Observed fluoride-form categories, record frequencies, and literature-source coverage. Category labels reproduce the processed data; source identity is used only for coverage auditing."),
        (4, "Interferent composition and source coverage", "SI_categorical_composition_source_coverage_v1_interferent_part1", "Results 3.1", "Observed interferent categories, record frequencies, and literature-source coverage. Sparse and single-source levels are retained rather than merged."),
        (5, "Coagulant composition and source coverage, part 1", "SI_categorical_composition_source_coverage_v1_coagulant_part1", "Results 3.1", "Record frequency and source coverage for the first ranked set of coagulant categories. Rankings are descriptive and preserve the original categorical levels."),
        (6, "Coagulant composition and source coverage, part 2", "SI_categorical_composition_source_coverage_v1_coagulant_part2", "Results 3.1", "Record frequency and source coverage for the remaining coagulant categories, highlighting the long tail of source-specific formulations."),
        (7, "Seed-level stability of random-split models", "SI_random_split_complete_results_v1_seed_stability", "Results 3.2", "Performance across five prespecified random seeds for all evaluated random-split methods. Points represent seed-specific test results; summaries use the same locked prediction records as Figure 4."),
        (8, "Pooled random-split predictions for all methods", "SI_random_split_complete_results_v1_all_method_pooled_predictions", "Results 3.2", "Observed versus predicted fluoride removal for the complete set of random-split methods after pooling the five test sets. Pooled points are prediction events and are not treated as statistically independent samples."),
        (9, "Grouped-validation performance across strategies and models", "SI_grouped_validation_lolo_complete_results_v1_strategy_model_metrics", "Results 3.3", "Pooled source-isolated performance across validation strategies and candidate models, reconstructed from the locked grouped-validation prediction files."),
        (10, "LOLO error by held-out literature source", "SI_grouped_validation_lolo_complete_results_v1_lolo_source_rmse", "Results 3.3", "Leave-one-literature-out (LOLO) root-mean-square error for each held-out source, showing pronounced source-to-source heterogeneity."),
        (11, "Complete LOLO observed-versus-predicted results", "SI_grouped_validation_lolo_complete_results_v1_lolo_predictions", "Results 3.3", "Observed versus predicted fluoride removal under LOLO validation. Every prediction was generated while the corresponding literature source was excluded from model fitting."),
        (12, "Selective-AD threshold evidence", "SI_ad_threshold_coverage_sensitivity_audit_v1_threshold_evidence", "Results 3.4", "Sensitivity of retained coverage and predictive performance to the target-independent applicability-domain threshold. The prespecified P08 operating point is shown without optimizing on test errors."),
        (13, "Selective-AD stability, uncertainty, and representativeness", "SI_ad_threshold_coverage_sensitivity_audit_v1_stability_uncertainty_representativeness", "Results 3.4", "Split stability, bootstrap uncertainty, and representativeness diagnostics for the selectively retained domain."),
        (14, "Selective-AD chemistry window", "SI_ad_threshold_coverage_sensitivity_audit_v1_chemistry_window", "Results 3.4", "Numeric and categorical chemistry window represented by the retained prediction events and unique chemistry rows at the exact P08 policy."),
        (15, "Selective-AD transfer across strategies and methods", "SI_ad_threshold_coverage_sensitivity_audit_v1_strategy_method_transfer", "Results 3.4", "Transfer of the fixed selective-domain rule across validation strategies and modeling methods. Results distinguish a candidate source-specific reliability window from universal cross-literature validity."),
        (16, "SHAP importance across interpretation contexts", "SI_shap_pdp_extended_interpretation_v1_context_importance", "Results 3.5", "Comparison of feature and feature-group importance across the full dataset, Paper_010, and the retained selective domain. Source identity is used only to define diagnostic contexts and is not a model feature."),
        (17, "Empirical support for two-dimensional PDP interpretation", "SI_shap_pdp_extended_interpretation_v1_pdp_support", "Results 3.5", "Empirical data support underlying the paired partial-dependence grid. Sparse regions are explicitly identified to prevent extrapolative mechanistic interpretation."),
        (18, "Extended marginal PDP profiles", "SI_shap_pdp_extended_interpretation_v1_pdp_marginal_profiles", "Results 3.5", "Marginal partial-dependence profiles with empirical support information for the principal continuous predictors. Curves describe the fitted model and are not causal effects."),
    ]


def text_specs():
    return [
        (1, "Data structure, variable roles, and quality control", "Methods 2.1-2.2; Results 3.1"),
        (2, "Categorical composition and literature-source coverage", "Results 3.1"),
        (3, "Model configurations, hyperparameter spaces, and selection evidence", "Methods 2.3-2.4"),
        (4, "Complete random-split results and stability", "Results 3.2"),
        (5, "Grouped validation and leave-one-literature-out generalization", "Results 3.3"),
        (6, "Selective applicability-domain evidence and sensitivity", "Results 3.4"),
        (7, "Extended SHAP and partial-dependence interpretation", "Results 3.5"),
    ]


def build_manifests(outputs):
    table_rows = []
    for number, title, files, crossref in table_specs(STEM):
        resolved = [outputs["tables"] / f for f in files]
        table_rows.append({"number": f"Table S{number}", "title": title,
                           "component_files": ";".join(files), "n_component_files": len(files),
                           "all_files_present": all(p.is_file() for p in resolved), "main_text_cross_reference": crossref})
    figure_rows = []
    for number, title, base, crossref, caption in figure_specs(STEM):
        png, pdf = outputs["figures"] / f"{base}.png", outputs["figures"] / f"{base}.pdf"
        figure_rows.append({"number": f"Figure S{number}", "title": title, "file_stem": base,
                            "png_present": png.is_file(), "pdf_present": pdf.is_file(),
                            "caption": caption, "main_text_cross_reference": crossref})
    text_rows = [{"number": f"Supplementary Text S{n}", "title": title, "main_text_cross_reference": ref}
                 for n, title, ref in text_specs()]
    tables, figures, texts = pd.DataFrame(table_rows), pd.DataFrame(figure_rows), pd.DataFrame(text_rows)
    if len(tables) != 20 or len(figures) != 18 or len(texts) != 7:
        raise AssertionError("Final SI numbering must contain 20 tables, 18 figures, and 7 text sections.")
    if not tables.all_files_present.all():
        bad = tables.loc[~tables.all_files_present, ["number", "component_files"]]
        raise FileNotFoundError("Missing Table S component files:\n" + bad.to_string(index=False))
    if not (figures.png_present & figures.pdf_present).all():
        bad = figures.loc[~(figures.png_present & figures.pdf_present), ["number", "file_stem"]]
        raise FileNotFoundError("Missing Figure S PNG/PDF pairs:\n" + bad.to_string(index=False))
    return tables, figures, texts


def markdown_report(tables, figures, texts) -> str:
    lines = [
        "# Final Supplementary Information numbering and captions", "",
        "This manifest assigns final citation numbers without renaming or overwriting the underlying evidence files.", "",
        "## Supplementary Text", "",
    ]
    for row in texts.itertuples(index=False):
        lines += [f"### {row.number}. {row.title}", "", f"Main-text cross-reference: {row.main_text_cross_reference}.", ""]
    lines += ["## Supplementary Tables", ""]
    for row in tables.itertuples(index=False):
        lines += [f"### {row.number}. {row.title}", "", f"Files: `{row.component_files}`", "",
                  f"Main-text cross-reference: {row.main_text_cross_reference}.", ""]
    lines += ["## Supplementary Figures", ""]
    for row in figures.itertuples(index=False):
        lines += [f"### {row.number}. {row.title}", "", row.caption, "",
                  f"Files: `{row.file_stem}.png` and `{row.file_stem}.pdf`", "",
                  f"Main-text cross-reference: {row.main_text_cross_reference}.", ""]
    lines += ["## Interpretation guardrails", "",
              "- `Source ID` is used only for grouped validation, LOLO, source diagnostics, and AD auditing; it is never a prediction feature.",
              "- Effluent measurements, reconstructed removal, removal discrepancy, raw-value fields, and original row identifiers are excluded from formal model inputs.",
              "- Random-split pooled points are prediction events across five seeds, not independent experimental samples.",
              "- The P08 selective domain is target-independent, retains 52 prediction events representing 44 unique chemistry rows, and is entirely associated with Paper_010; it is therefore a candidate source-specific reliability window, not a universal cross-literature domain.",
              "- SHAP and PDP results describe the fitted model and are not interpreted as causal effects.", ""]
    return "\n".join(lines)


def main():
    started = time.perf_counter()
    root, cfg, inputs, outputs = project_paths()
    watched = [inputs["strict"], inputs["source"], cfg]
    main_figures = sorted(outputs["figures"].glob("Figure_[1-8]_*.png")) + sorted(outputs["figures"].glob("Figure_[1-8]_*.pdf"))
    before = {str(p.relative_to(root)): sha256(p) for p in watched + main_figures}
    strict, data = load_data(inputs)
    table_s1 = make_table_s1(data)
    table_s2 = make_table_s2(strict, data)
    table_s3 = make_table_s3(data)

    generated_tables = {
        f"{STEM}_Table_S1_variable_dictionary.csv": table_s1,
        f"{STEM}_Table_S2_quality_control_audit.csv": table_s2,
        f"{STEM}_Table_S3_continuous_statistics.csv": table_s3,
    }
    for name, table in generated_tables.items():
        table.to_csv(outputs["tables"] / name, index=False, encoding="utf-8-sig")

    fig1 = outputs["figures"] / f"{STEM}_Figure_S1_data_roles_qc"
    fig2 = outputs["figures"] / f"{STEM}_Figure_S2_continuous_support_audit"
    create_figure_s1(data, table_s1, fig1)
    create_figure_s2(table_s3, fig2)
    tables, figures, texts = build_manifests(outputs)

    tables.to_csv(outputs["tables"] / f"{STEM}_table_manifest.csv", index=False, encoding="utf-8-sig")
    figures.to_csv(outputs["tables"] / f"{STEM}_figure_manifest.csv", index=False, encoding="utf-8-sig")
    texts.to_csv(outputs["tables"] / f"{STEM}_text_manifest.csv", index=False, encoding="utf-8-sig")
    crossref = pd.concat([
        texts.assign(item_type="Supplementary Text").rename(columns={"number": "item"})[["item_type", "item", "title", "main_text_cross_reference"]],
        tables.assign(item_type="Table").rename(columns={"number": "item"})[["item_type", "item", "title", "main_text_cross_reference"]],
        figures.assign(item_type="Figure").rename(columns={"number": "item"})[["item_type", "item", "title", "main_text_cross_reference"]],
    ], ignore_index=True)
    crossref.to_csv(outputs["tables"] / f"{STEM}_cross_reference_manifest.csv", index=False, encoding="utf-8-sig")
    report = markdown_report(tables, figures, texts)
    (outputs["reports"] / f"{STEM}_numbering_captions.md").write_text(report, encoding="utf-8")

    after = {str(p.relative_to(root)): sha256(p) for p in watched + main_figures}
    unchanged = before == after
    if not unchanged:
        changed = [key for key in before if before[key] != after.get(key)]
        raise AssertionError("Protected inputs or main figures changed: " + ", ".join(changed))
    audit = {
        "stem": STEM, "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "project_root": str(root), "python": sys.version, "platform": platform.platform(),
        "matrix_shapes": {"strict": list(strict.shape), "source_bearing": list(data.shape)},
        "source_groups": int(data[SOURCE].nunique()), "formal_predictors": PREDICTORS,
        "counts": {"supplementary_text_sections": len(texts), "tables": len(tables), "figures": len(figures)},
        "all_table_components_present": bool(tables.all_files_present.all()),
        "all_figure_pairs_present": bool((figures.png_present & figures.pdf_present).all()),
        "protected_files_unchanged": unchanged, "protected_sha256": after,
        "archive_paths_used": False, "models_trained": False,
        "existing_si_files_renamed_or_overwritten": False,
    }
    (outputs["reports"] / f"{STEM}_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    log = [
        "PASS: final SI numbering is continuous and complete.",
        f"Supplementary Text: {len(texts)}; Tables: {len(tables)}; Figures: {len(figures)}.",
        f"Matrices: {strict.shape[0]} x {strict.shape[1]} and {data.shape[0]} x {data.shape[1]}; sources: {data[SOURCE].nunique()}.",
        "Created Figure S1-S2 and Table S1-S3; all other entries map existing evidence files.",
        "Existing SI outputs, processed data, model results, predictions, and main figures unchanged.",
        f"Outputs: {root / 'outputs' / 'jece_final_package'}",
    ]
    runtime = time.perf_counter() - started
    log.append(f"Total runtime: {runtime:.2f} s")
    (outputs["logs"] / f"{STEM}.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
