# -*- coding: utf-8 -*-
"""
32K Results and Discussion structure integration
Project: fluoride coagulation defluoridation ML project

Purpose
-------
This script reads the 32J integrated evidence tables and final 32I2 figures,
then generates a manuscript-ready Results and Discussion structure.

Important project rule
----------------------
This script follows the original project output style:
    tables  -> outputs/tables/
    figures -> outputs/figures/
    reports -> reports/

It does NOT create a separate outputs/32k_xxx/ directory.

How to use
----------
Put this file either in the project root or in src/, then run it in PyCharm.

Expected inputs
---------------
outputs/tables/32j_main_result_table.csv
outputs/tables/32j_final_evidence_ordered_manifest.csv
outputs/tables/32j_figure_table_manifest_project_order.csv
outputs/tables/32j_claim_guardrails_for_manuscript.csv
outputs/tables/32j_final_figure_file_check.csv

Expected final figures already present in outputs/figures/
194_32i2_evidence_chain_summary_clean.png/pdf
195_32i2_threshold_plateau_evidence_clean.png/pdf
196_32i2_retained_chemistry_window_clean.png/pdf
197_32i2_stability_transfer_boundary_clean.png/pdf

Outputs
-------
outputs/tables/32k_manuscript_results_structure.csv
outputs/tables/32k_figure_caption_table.csv
outputs/tables/32k_key_claims_and_evidence_map.csv
outputs/tables/32k_results_paragraph_blocks.csv
outputs/tables/32k_required_supplementary_items.csv
outputs/figures/198_32k_results_discussion_logic_map.png
outputs/figures/198_32k_results_discussion_logic_map.pdf
reports/32k_results_discussion_draft.md
reports/32k_run_log.txt
"""

from __future__ import annotations

import os
import re
import sys
import math
import textwrap
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


# =============================================================================
# 1. Path utilities
# =============================================================================

def detect_project_root() -> Path:
    """Detect project root when the script is placed in root or src/."""
    here = Path(__file__).resolve().parent

    candidates = [here]
    if here.name.lower() == "src":
        candidates.append(here.parent)
    candidates.extend(list(here.parents))

    for cand in candidates:
        if (cand / "outputs").exists() or (cand / "data").exists() or (cand / "src").exists():
            return cand

    return here


PROJECT_ROOT = detect_project_root()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUTS_DIR / "tables"
FIGURE_DIR = OUTPUTS_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "reports"

for d in [OUTPUTS_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def find_input_file(file_name: str) -> Path | None:
    """
    Search inputs by priority:
    1) outputs/tables or outputs/figures
    2) project root recursive search
    """
    direct_candidates = [
        TABLE_DIR / file_name,
        FIGURE_DIR / file_name,
        PROJECT_ROOT / file_name,
        PROJECT_ROOT / "tables" / file_name,
        PROJECT_ROOT / "figures" / file_name,
    ]
    for p in direct_candidates:
        if p.exists():
            return p

    matches = list(PROJECT_ROOT.rglob(file_name))
    if matches:
        # Prefer outputs/tables and outputs/figures if multiple exist.
        matches_sorted = sorted(
            matches,
            key=lambda x: (
                0 if "outputs" in str(x).lower() else 1,
                len(str(x))
            )
        )
        return matches_sorted[0]

    return None


def read_csv_required(file_name: str, log_lines: list[str]) -> pd.DataFrame:
    p = find_input_file(file_name)
    if p is None:
        raise FileNotFoundError(
            f"Required input not found: {file_name}\n"
            f"Please make sure it is in outputs/tables/ or the project folder."
        )
    log_lines.append(f"[INPUT] {file_name}: {p}")
    return pd.read_csv(p)


def safe_float(v, default=math.nan) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def fmt_num(v, nd=3, na="NA") -> str:
    if v is None:
        return na
    try:
        if pd.isna(v):
            return na
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def fmt_int(v, na="NA") -> str:
    try:
        if pd.isna(v):
            return na
        return str(int(round(float(v))))
    except Exception:
        return na


# =============================================================================
# 2. Load 32J outputs
# =============================================================================

def main() -> None:
    log_lines: list[str] = []
    log_lines.append("=" * 80)
    log_lines.append("32K Results and Discussion structure integration")
    log_lines.append(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Project root: {PROJECT_ROOT}")
    log_lines.append(f"Table output dir: {TABLE_DIR}")
    log_lines.append(f"Figure output dir: {FIGURE_DIR}")
    log_lines.append(f"Report output dir: {REPORT_DIR}")
    log_lines.append("=" * 80)

    required_csvs = {
        "main": "32j_main_result_table.csv",
        "evidence": "32j_final_evidence_ordered_manifest.csv",
        "fig_manifest": "32j_figure_table_manifest_project_order.csv",
        "guardrails": "32j_claim_guardrails_for_manuscript.csv",
        "fig_check": "32j_final_figure_file_check.csv",
    }

    main_df = read_csv_required(required_csvs["main"], log_lines)
    evidence_df = read_csv_required(required_csvs["evidence"], log_lines)
    fig_manifest_df = read_csv_required(required_csvs["fig_manifest"], log_lines)
    guardrails_df = read_csv_required(required_csvs["guardrails"], log_lines)
    fig_check_df = read_csv_required(required_csvs["fig_check"], log_lines)

    # Identify three evidence levels.
    def row_by_keyword(df: pd.DataFrame, col: str, keyword: str) -> pd.Series:
        mask = df[col].astype(str).str.lower().str.contains(keyword.lower(), na=False)
        if mask.any():
            return df.loc[mask].iloc[0]
        return pd.Series(dtype=object)

    all_domain = row_by_keyword(main_df, "Evidence_Level", "All-domain")
    predefined = row_by_keyword(main_df, "Evidence_Level", "Predefined")
    selective = row_by_keyword(main_df, "Evidence_Level", "selective")

    if selective.empty:
        raise ValueError("Could not locate the selective retained-subdomain row in 32j_main_result_table.csv.")

    sel_r2 = safe_float(selective.get("R2"))
    sel_rmse = safe_float(selective.get("RMSE"))
    sel_mae = safe_float(selective.get("MAE"))
    sel_cov = safe_float(selective.get("Coverage"))
    sel_n = selective.get("N_Retained", math.nan)
    sel_group = selective.get("Dominant_Group", "NA")
    sel_rule = selective.get("Rule", "NA")
    sel_strategy = selective.get("Strategy", "NA")
    sel_method = selective.get("Method", "NA")

    log_lines.append("")
    log_lines.append("[KEY RESULT]")
    log_lines.append(f"Selective AD retained-subdomain result: {sel_strategy} + {sel_method}")
    log_lines.append(f"Rule: {sel_rule}")
    log_lines.append(f"R2={fmt_num(sel_r2, 6)}, RMSE={fmt_num(sel_rmse, 6)}, MAE={fmt_num(sel_mae, 6)}, Coverage={fmt_num(sel_cov, 6)}, N={fmt_int(sel_n)}, Dominant group={sel_group}")

    # =============================================================================
    # 3. Build manuscript structure table
    # =============================================================================

    manuscript_rows = [
        {
            "Section_Order": 1,
            "Suggested_Section": "3.1 From all-domain grouped validation to selective AD retained subdomain（从全域分组验证到选择性适应域保留子域）",
            "Core_Message": (
                "先说明全域 grouped validation（分组验证）表现有限，再说明 predefined reliable domain（预定义可靠域）没有解决问题，"
                "最后引出 target-free selective AD（无目标泄露的选择性适应域）保留子域结果。"
            ),
            "Recommended_Figure_Table": "Main Fig. 1; Table 1",
            "Evidence_Source": "32j_main_result_table.csv; 194_32i2_evidence_chain_summary_clean.png/pdf",
            "Must_Write": (
                f"P08 = {sel_strategy} + {sel_method}; {sel_rule}; "
                f"R² = {fmt_num(sel_r2)}, RMSE = {fmt_num(sel_rmse)}, MAE = {fmt_num(sel_mae)}, "
                f"Coverage = {fmt_num(sel_cov)}, N retained = {fmt_int(sel_n)}."
            ),
            "Claim_Boundary": (
                "只能写 selective AD retained-subdomain result（选择性适应域保留子域结果），"
                "不能写 all-domain grouped-validation performance（全域分组验证性能）。"
            ),
        },
        {
            "Section_Order": 2,
            "Suggested_Section": "3.2 Threshold sensitivity and anti-cherry-picking evidence（阈值敏感性与反选择性报告证据）",
            "Core_Message": (
                "说明 NN_Risk threshold（最近邻风险阈值）附近存在一个较窄但非单点的平台，"
                "用于支持该阈值不是纯偶然挑选；同时必须披露 coverage（覆盖率）很低。"
            ),
            "Recommended_Figure_Table": "Main Fig. 2 or Figure S1; Table S1; Table S2",
            "Evidence_Source": "32h_threshold_sensitivity_all.csv; 32h_bootstrap_uncertainty_exact_policy.csv; 195_32i2_threshold_plateau_evidence_clean.png/pdf",
            "Must_Write": (
                f"Exact rule: {sel_rule}. The retained subset reached R² = {fmt_num(sel_r2)} with coverage = {fmt_num(sel_cov)}."
            ),
            "Claim_Boundary": (
                "可以说存在 threshold plateau evidence（阈值平台证据），但必须说明平台较窄、覆盖率低，不能说阈值是普适规则。"
            ),
        },
        {
            "Section_Order": 3,
            "Suggested_Section": "3.3 Chemistry window and retained-subdomain characterization（化学窗口与保留子域表征）",
            "Core_Message": (
                "解释保留样本集中在哪些初始氟浓度、pH、剂量、停留时间和混凝剂组成范围内，"
                "把模型结果落回混凝除氟机理。"
            ),
            "Recommended_Figure_Table": "Main Fig. 3; Table S3",
            "Evidence_Source": "32g_numeric_operating_window.csv; 32g_categorical_composition.csv; 196_32i2_retained_chemistry_window_clean.png/pdf",
            "Must_Write": (
                f"The retained subset was dominated by {sel_group}, so the chemical interpretation is literature-/chemistry-specific."
            ),
            "Claim_Boundary": (
                "只能解释为 retained chemistry window（保留化学窗口），不能外推为所有混凝剂、所有pH和所有文献条件下的普适规律。"
            ),
        },
        {
            "Section_Order": 4,
            "Suggested_Section": "3.4 Stability, transfer support, and interpretation boundary（稳定性、迁移支持与解释边界）",
            "Core_Message": (
                "说明该选择性规则在若干 strategy/model combination（策略/模型组合）上有部分迁移支持，"
                "但 split-level stability（分割层面稳定性）有限，且文献组高度集中。"
            ),
            "Recommended_Figure_Table": "Main Fig. 4 or Figure S2; Table S2",
            "Evidence_Source": "32h_exact_policy_split_metrics.csv; 32i_final_evidence_summary.csv; 197_32i2_stability_transfer_boundary_clean.png/pdf",
            "Must_Write": (
                "The accepted rule should be treated as an interpretation- and decision-support subdomain, not as a universal deployment rule."
            ),
            "Claim_Boundary": (
                "必须披露 Paper_010 dominance（Paper_010主导）和 split-level instability（分割层面不稳定），"
                "不能宣称已经获得跨文献全域泛化。"
            ),
        },
        {
            "Section_Order": 5,
            "Suggested_Section": "3.5 Environmental and engineering implication（环境与工程意义）",
            "Core_Message": (
                "把结果表述为：机器学习模型可以识别高可信局部工况窗口，为后续实验补点和工程调控提供候选区域，"
                "但当前还不能直接替代全域工程设计。"
            ),
            "Recommended_Figure_Table": "Table 1; Main Fig. 1-4",
            "Evidence_Source": "32J/32K integrated evidence",
            "Must_Write": (
                "The most defensible contribution is a target-free selective AD workflow for finding a high-confidence retained subdomain under grouped-validation pressure."
            ),
            "Claim_Boundary": (
                "工程意义应写成 experimental prioritization（实验优先级筛选）和 decision-support（决策支持），"
                "不要写成直接工程自动控制模型。"
            ),
        },
    ]
    manuscript_structure_df = pd.DataFrame(manuscript_rows)

    # =============================================================================
    # 4. Figure captions
    # =============================================================================

    caption_rows = [
        {
            "Item": "Main Fig. 1",
            "Source_File": "194_32i2_evidence_chain_summary_clean.png/pdf",
            "Suggested_Caption_CN": (
                "选择性适应域结果的证据链汇总。图中比较了全域分组验证、预定义可靠域和选择性适应域保留子域在 R²、RMSE、MAE 和覆盖率上的差异。"
                "结果显示，全域分组验证性能有限，预定义可靠域没有带来改善，而 target-free selective AD（无目标泄露的选择性适应域）规则在极低覆盖率下识别出一个高精度保留子域。"
            ),
            "Suggested_Caption_EN": (
                "Evidence-chain summary for the selective applicability-domain retained-subdomain result. "
                "The figure compares all-domain grouped validation, predefined reliable domain, and the selective AD retained subdomain in terms of R², RMSE, MAE, and coverage."
            ),
            "Placement": "Main text",
            "Boundary_Note": "Do not describe the selective subdomain R² as all-domain grouped-validation R².",
        },
        {
            "Item": "Main Fig. 2 / Figure S1",
            "Source_File": "195_32i2_threshold_plateau_evidence_clean.png/pdf",
            "Suggested_Caption_CN": (
                "被接受选择性适应域规则的阈值平台证据。图中展示 NN_Risk threshold（最近邻风险阈值）变化时，保留子域 R²、保留样本数和覆盖率的变化。"
                "该结果用于说明精确阈值附近存在非单点平台，但同时表明覆盖率较低。"
            ),
            "Suggested_Caption_EN": (
                "Threshold-plateau evidence for the accepted selective applicability-domain rule. "
                "The panels show retained-subdomain R², retained sample count, and coverage as the NN-risk threshold varies."
            ),
            "Placement": "Main text or Supplementary Information",
            "Boundary_Note": "Disclose the narrow plateau and low coverage.",
        },
        {
            "Item": "Main Fig. 3",
            "Source_File": "196_32i2_retained_chemistry_window_clean.png/pdf",
            "Suggested_Caption_CN": (
                "保留子域的化学窗口和适应域组成。图中总结了保留样本的初始氟浓度、投加量、进水 pH、停留时间、温度、干扰离子浓度、去除率和出水 pH 范围，"
                "以及混凝剂组成、文献组组成和适应域风险分量。"
            ),
            "Suggested_Caption_EN": (
                "Chemistry-window and AD-component characterization of the retained subdomain. "
                "The figure summarizes numerical operating ranges, retained coagulant compositions, literature-group composition, and median AD-component levels."
            ),
            "Placement": "Main text",
            "Boundary_Note": "Interpret as a chemistry-/literature-specific retained window rather than a universal mechanism.",
        },
        {
            "Item": "Main Fig. 4 / Figure S2",
            "Source_File": "197_32i2_stability_transfer_boundary_clean.png/pdf",
            "Suggested_Caption_CN": (
                "选择性保留子域结果的稳定性、迁移支持与解释边界。图中展示分割层面 R²、不同策略/模型组合的迁移支持，以及保留样本中文献组的主导性。"
            ),
            "Suggested_Caption_EN": (
                "Stability, transfer support, and interpretation boundary of the selective retained-subdomain result. "
                "The panels show split-level retained-subdomain R², transfer support across strategy/model combinations, and literature-group dominance."
            ),
            "Placement": "Main text or Supplementary Information",
            "Boundary_Note": "Disclose split-level instability and Paper_010 dominance.",
        },
        {
            "Item": "Main Fig. 5 / Figure S3",
            "Source_File": "198_32k_results_discussion_logic_map.png/pdf",
            "Suggested_Caption_CN": (
                "Results and Discussion（结果与讨论）写作逻辑图。该图把全域分组验证、预定义可靠域、选择性适应域保留子域和最终声明边界连接起来，"
                "用于论文组织和组会汇报。"
            ),
            "Suggested_Caption_EN": (
                "Logic map for organizing the Results and Discussion section. "
                "The map links all-domain grouped validation, predefined reliable domain, selective AD retained subdomain, and the final claim boundary."
            ),
            "Placement": "Internal use, graphical abstract candidate, or Supplementary Information",
            "Boundary_Note": "Use as an organization figure unless further polished for the main manuscript.",
        },
    ]
    caption_df = pd.DataFrame(caption_rows)

    # =============================================================================
    # 5. Claims and evidence map
    # =============================================================================

    claims_rows = [
        {
            "Claim_Order": 1,
            "Claim_Type": "Main defensible claim",
            "Claim": (
                "A target-free selective AD（无目标泄露的选择性适应域） workflow identified a high-confidence retained subdomain under grouped-validation（分组验证） pressure."
            ),
            "Evidence": "32F2 reproducibility audit; 32H threshold sensitivity; 32I final evidence integration; Fig. 1-2",
            "Allowed_Wording": (
                f"Under the accepted rule ({sel_rule}), the retained subdomain achieved R² = {fmt_num(sel_r2)}, RMSE = {fmt_num(sel_rmse)}, MAE = {fmt_num(sel_mae)}, with coverage = {fmt_num(sel_cov)}."
            ),
            "Forbidden_Wording": "The model achieved R² = 0.748 in all-domain grouped validation.",
        },
        {
            "Claim_Order": 2,
            "Claim_Type": "Mechanistic interpretation",
            "Claim": "The retained subdomain corresponds to a restricted chemistry and literature window.",
            "Evidence": "32G retained chemistry characterization; Fig. 3",
            "Allowed_Wording": f"The retained samples are dominated by {sel_group}, indicating a literature-/chemistry-specific operating window.",
            "Forbidden_Wording": "The rule is valid for all coagulants and all fluoride wastewater conditions.",
        },
        {
            "Claim_Order": 3,
            "Claim_Type": "Robustness qualification",
            "Claim": "The threshold shows partial support but still has limited stability and coverage.",
            "Evidence": "32H split metrics; 32H bootstrap uncertainty; Fig. 2 and Fig. 4",
            "Allowed_Wording": "The exact threshold is supported by a narrow non-single-point plateau and partial transfer support, but coverage remains low.",
            "Forbidden_Wording": "The threshold is universally stable and can be deployed without qualification.",
        },
        {
            "Claim_Order": 4,
            "Claim_Type": "Environmental implication",
            "Claim": "The method can prioritize reliable experimental regions for follow-up validation.",
            "Evidence": "Integrated 32J and 32K evidence",
            "Allowed_Wording": "The workflow provides an experimental-prioritization and decision-support tool for heterogeneous small environmental datasets.",
            "Forbidden_Wording": "The model is already sufficient for direct full-domain engineering control.",
        },
    ]
    claims_df = pd.DataFrame(claims_rows)

    # =============================================================================
    # 6. Paragraph blocks for manuscript drafting
    # =============================================================================

    paragraph_rows = [
        {
            "Paragraph_Order": 1,
            "Target_Section": "3.1 Evidence chain",
            "Paragraph_Function": "Opening result and problem framing",
            "Draft_Text_CN": (
                "首先，我们在严格的 grouped validation（分组验证）框架下评价模型的全域泛化能力。"
                "全域结果显示，模型在跨文献/跨分组条件下的整体预测能力仍然有限，说明简单依赖随机划分或全域平均指标容易高估模型在异质文献数据中的真实可靠性。"
                "因此，后续分析不再把单一全域 R² 作为唯一判断标准，而是进一步考察 applicability domain（适应域）筛选能否识别更可信的局部预测区域。"
            ),
        },
        {
            "Paragraph_Order": 2,
            "Target_Section": "3.1 Evidence chain",
            "Paragraph_Function": "Accepted selective AD result",
            "Draft_Text_CN": (
                f"在多轮适应域审计后，P08 规则被保留为最终可接受的 selective AD retained-subdomain（选择性适应域保留子域）结果。"
                f"该规则采用 {sel_strategy} 策略和 {sel_method} 模型，并设置 {sel_rule}。"
                f"在该保留子域中，模型达到 R² = {fmt_num(sel_r2)}、RMSE = {fmt_num(sel_rmse)}、MAE = {fmt_num(sel_mae)}，"
                f"但 coverage（覆盖率）仅为 {fmt_num(sel_cov)}，保留样本数为 {fmt_int(sel_n)}。"
                "因此，该结果应被解释为高可信局部子域的识别，而不是全域分组验证性能的提升。"
            ),
        },
        {
            "Paragraph_Order": 3,
            "Target_Section": "3.2 Threshold sensitivity",
            "Paragraph_Function": "Threshold plateau interpretation",
            "Draft_Text_CN": (
                "为了避免将单一阈值误判为偶然最优点，我们进一步进行了 threshold sensitivity analysis（阈值敏感性分析）。"
                "结果表明，在精确 NN_Risk threshold（最近邻风险阈值）附近，保留子域 R² 存在一个较窄的平台区间，说明该规则并非完全依赖单个孤立阈值。"
                "然而，随着阈值放宽，覆盖率增加的同时 R² 明显下降，表明高精度结果依赖较强的适应域筛选。"
            ),
        },
        {
            "Paragraph_Order": 4,
            "Target_Section": "3.3 Chemistry window",
            "Paragraph_Function": "Chemical interpretation",
            "Draft_Text_CN": (
                "保留子域的化学窗口显示，该高可信预测区域并非随机分布于全部实验空间，而是集中在特定的初始氟浓度、pH、投加量、停留时间和混凝剂组成范围内。"
                "这说明模型在该局部区域内可能捕捉到了更一致的混凝沉淀除氟规律，包括混凝剂水解、絮体形成、氟离子吸附/络合以及干扰离子竞争效应等共同作用。"
                f"但由于保留样本由 {sel_group} 主导，该机理解释必须限定在相应文献和化学窗口内。"
            ),
        },
        {
            "Paragraph_Order": 5,
            "Target_Section": "3.4 Stability and boundary",
            "Paragraph_Function": "Boundary disclosure",
            "Draft_Text_CN": (
                "稳定性与迁移支持分析进一步表明，该规则在若干 strategy/model combination（策略/模型组合）下具有一定迁移支持，"
                "但 split-level stability（分割层面稳定性）仍然有限，且保留样本存在明显的 literature-group dominance（文献组主导）。"
                "因此，本研究不将该规则表述为普适工程控制规则，而将其定位为一种用于小样本、异质环境数据的高可信子域识别和后续实验优先级筛选方法。"
            ),
        },
        {
            "Paragraph_Order": 6,
            "Target_Section": "3.5 Implication",
            "Paragraph_Function": "Engineering implication",
            "Draft_Text_CN": (
                "从工程应用角度看，该结果的意义不在于直接给出适用于所有工况的统一预测模型，而在于提供了一种从复杂文献数据中识别可靠局部工况窗口的方法。"
                "该窗口可用于指导后续实验补点、模型外推风险判断和混凝除氟工艺条件的初步筛选。"
                "未来需要围绕低覆盖区域和跨文献薄弱区域进行定向实验扩充，以逐步提高模型的全域泛化能力。"
            ),
        },
    ]
    paragraph_df = pd.DataFrame(paragraph_rows)

    # =============================================================================
    # 7. Supplementary items
    # =============================================================================

    supp_rows = [
        {
            "Supplementary_Item": "Table S1",
            "Source": "32h_bootstrap_uncertainty_exact_policy.csv",
            "Purpose": "Report bootstrap uncertainty（自助法不确定性） for exact P08 rule.",
            "Required_Boundary": "Report median and 95% CI; do not only report the best point estimate.",
        },
        {
            "Supplementary_Item": "Table S2",
            "Source": "32h_exact_policy_split_metrics.csv",
            "Purpose": "Report split-level stability（分割层面稳定性）.",
            "Required_Boundary": "Disclose unstable split-level behavior.",
        },
        {
            "Supplementary_Item": "Table S3",
            "Source": "32g_numeric_operating_window.csv; 32g_categorical_composition.csv",
            "Purpose": "Report retained chemistry window（保留化学窗口） and categorical composition（类别组成）.",
            "Required_Boundary": "Keep interpretation limited to the retained literature/chemistry window.",
        },
        {
            "Supplementary_Item": "Table S4",
            "Source": "32j_claim_guardrails_for_manuscript.csv",
            "Purpose": "Make claim boundaries transparent.",
            "Required_Boundary": "Separate allowed claims, required disclosures, and forbidden claims.",
        },
    ]
    supp_df = pd.DataFrame(supp_rows)

    # =============================================================================
    # 8. Save tables
    # =============================================================================

    outputs = {
        "32k_manuscript_results_structure.csv": manuscript_structure_df,
        "32k_figure_caption_table.csv": caption_df,
        "32k_key_claims_and_evidence_map.csv": claims_df,
        "32k_results_paragraph_blocks.csv": paragraph_df,
        "32k_required_supplementary_items.csv": supp_df,
    }

    for name, df in outputs.items():
        out_path = TABLE_DIR / name
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        log_lines.append(f"[OUTPUT TABLE] {out_path}")

    # =============================================================================
    # 9. Draw 32K logic map figure
    # =============================================================================

    draw_logic_map(
        all_domain=all_domain,
        predefined=predefined,
        selective=selective,
        out_png=FIGURE_DIR / "198_32k_results_discussion_logic_map.png",
        out_pdf=FIGURE_DIR / "198_32k_results_discussion_logic_map.pdf",
        log_lines=log_lines,
    )

    # =============================================================================
    # 10. Markdown report
    # =============================================================================

    report_md = build_markdown_report(
        main_df=main_df,
        evidence_df=evidence_df,
        fig_manifest_df=fig_manifest_df,
        guardrails_df=guardrails_df,
        manuscript_structure_df=manuscript_structure_df,
        caption_df=caption_df,
        claims_df=claims_df,
        paragraph_df=paragraph_df,
        supp_df=supp_df,
        key_row=selective,
    )

    report_path = REPORT_DIR / "32k_results_discussion_draft.md"
    report_path.write_text(report_md, encoding="utf-8")
    log_lines.append(f"[OUTPUT REPORT] {report_path}")

    # Save run log.
    log_path = REPORT_DIR / "32k_run_log.txt"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log_lines.append(f"[OUTPUT LOG] {log_path}")

    print("\n".join(log_lines))
    print("\n32K finished successfully.")
    print(f"Tables saved to: {TABLE_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Reports saved to: {REPORT_DIR}")


# =============================================================================
# Figure drawing
# =============================================================================

def draw_logic_map(all_domain: pd.Series,
                   predefined: pd.Series,
                   selective: pd.Series,
                   out_png: Path,
                   out_pdf: Path,
                   log_lines: list[str]) -> None:
    def label_from_row(row: pd.Series, title: str) -> str:
        if row.empty:
            return f"{title}\nnot found"
        return (
            f"{title}\n"
            f"R² = {fmt_num(row.get('R2'))}\n"
            f"RMSE = {fmt_num(row.get('RMSE'))}\n"
            f"MAE = {fmt_num(row.get('MAE'))}\n"
            f"Coverage = {fmt_num(row.get('Coverage'))}"
        )

    labels = [
        label_from_row(all_domain, "All-domain grouped validation"),
        label_from_row(predefined, "Predefined reliable domain"),
        label_from_row(selective, "Selective AD retained subdomain"),
    ]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Title
    ax.text(
        6.5, 5.55,
        "32K Results and Discussion logic map",
        ha="center", va="center", fontsize=18, fontweight="bold"
    )

    # Box settings
    boxes = [
        (0.7, 3.1, 3.1, 1.55, labels[0]),
        (4.95, 3.1, 3.1, 1.55, labels[1]),
        (9.2, 3.1, 3.1, 1.55, labels[2]),
    ]

    for x, y, w, h, text in boxes:
        rect = Rectangle((x, y), w, h, linewidth=1.4, edgecolor="black", facecolor="white")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    # Arrows
    arrow1 = FancyArrowPatch((3.85, 3.85), (4.9, 3.85), arrowstyle="->", mutation_scale=18, linewidth=1.4)
    arrow2 = FancyArrowPatch((8.1, 3.85), (9.15, 3.85), arrowstyle="->", mutation_scale=18, linewidth=1.4)
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)

    ax.text(4.37, 4.25, "AD audit", ha="center", va="center", fontsize=11)
    ax.text(8.62, 4.25, "Target-free\nselective filtering", ha="center", va="center", fontsize=11)

    # Boundary bar
    boundary_text = (
        "Claim boundary: the accepted result is a target-free selective AD retained-subdomain result, "
        "not all-domain grouped-validation performance."
    )
    rect = Rectangle((1.0, 1.35), 11.0, 1.0, linewidth=1.2, edgecolor="black", facecolor="white")
    ax.add_patch(rect)
    ax.text(6.5, 1.85, boundary_text, ha="center", va="center", fontsize=12, wrap=True)

    # Bottom notes
    notes = [
        "Use Fig. 194 for evidence-chain metrics",
        "Use Fig. 195 for threshold sensitivity",
        "Use Fig. 196 for chemistry window",
        "Use Fig. 197 for stability and boundary",
    ]
    for i, note in enumerate(notes):
        ax.text(1.0 + i * 3.05, 0.65, note, ha="left", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    log_lines.append(f"[OUTPUT FIGURE] {out_png}")
    log_lines.append(f"[OUTPUT FIGURE] {out_pdf}")


# =============================================================================
# Markdown report
# =============================================================================

def build_markdown_report(main_df: pd.DataFrame,
                          evidence_df: pd.DataFrame,
                          fig_manifest_df: pd.DataFrame,
                          guardrails_df: pd.DataFrame,
                          manuscript_structure_df: pd.DataFrame,
                          caption_df: pd.DataFrame,
                          claims_df: pd.DataFrame,
                          paragraph_df: pd.DataFrame,
                          supp_df: pd.DataFrame,
                          key_row: pd.Series) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    key = {
        "strategy": key_row.get("Strategy", "NA"),
        "method": key_row.get("Method", "NA"),
        "rule": key_row.get("Rule", "NA"),
        "r2": fmt_num(key_row.get("R2")),
        "rmse": fmt_num(key_row.get("RMSE")),
        "mae": fmt_num(key_row.get("MAE")),
        "coverage": fmt_num(key_row.get("Coverage")),
        "n": fmt_int(key_row.get("N_Retained")),
        "group": key_row.get("Dominant_Group", "NA"),
    }

    lines = []
    lines.append("# 32K Results and Discussion（结果与讨论）结构整理")
    lines.append("")
    lines.append(f"Generated at: {now}")
    lines.append("")
    lines.append("## 1. 当前可写的核心结果")
    lines.append("")
    lines.append(
        f"当前最终可写结果为 **P08 = {key['strategy']} + {key['method']} + {key['rule']}**。"
    )
    lines.append("")
    lines.append(
        f"该 selective AD retained-subdomain（选择性适应域保留子域）结果为："
        f"R² = **{key['r2']}**，RMSE = **{key['rmse']}**，MAE = **{key['mae']}**，"
        f"Coverage（覆盖率）= **{key['coverage']}**，N retained（保留样本数）= **{key['n']}**，"
        f"Dominant group（主导文献组）= **{key['group']}**。"
    )
    lines.append("")
    lines.append("> 论文中必须写成：target-free selective AD retained-subdomain result（无目标泄露的选择性适应域保留子域结果）。")
    lines.append("")
    lines.append("> 论文中不能写成：all-domain grouped-validation performance（全域分组验证性能）。")
    lines.append("")
    lines.append("## 2. Results and Discussion（结果与讨论）推荐结构")
    lines.append("")

    for _, r in manuscript_structure_df.iterrows():
        lines.append(f"### {int(r['Section_Order'])}. {r['Suggested_Section']}")
        lines.append("")
        lines.append(f"**核心信息：** {r['Core_Message']}")
        lines.append("")
        lines.append(f"**图表位置：** {r['Recommended_Figure_Table']}")
        lines.append("")
        lines.append(f"**必须写：** {r['Must_Write']}")
        lines.append("")
        lines.append(f"**声明边界：** {r['Claim_Boundary']}")
        lines.append("")

    lines.append("## 3. 可直接改写进论文的段落块")
    lines.append("")
    for _, r in paragraph_df.iterrows():
        lines.append(f"### Paragraph {int(r['Paragraph_Order'])}: {r['Target_Section']} - {r['Paragraph_Function']}")
        lines.append("")
        lines.append(r["Draft_Text_CN"])
        lines.append("")

    lines.append("## 4. 推荐图题与图注")
    lines.append("")
    for _, r in caption_df.iterrows():
        lines.append(f"### {r['Item']}")
        lines.append("")
        lines.append(f"**Source:** {r['Source_File']}")
        lines.append("")
        lines.append(f"**CN caption:** {r['Suggested_Caption_CN']}")
        lines.append("")
        lines.append(f"**EN caption:** {r['Suggested_Caption_EN']}")
        lines.append("")
        lines.append(f"**Boundary note:** {r['Boundary_Note']}")
        lines.append("")

    lines.append("## 5. Claims（声明）和 Evidence（证据）对应关系")
    lines.append("")
    for _, r in claims_df.iterrows():
        lines.append(f"### Claim {int(r['Claim_Order'])}: {r['Claim_Type']}")
        lines.append("")
        lines.append(f"- **Claim:** {r['Claim']}")
        lines.append(f"- **Evidence:** {r['Evidence']}")
        lines.append(f"- **Allowed wording:** {r['Allowed_Wording']}")
        lines.append(f"- **Forbidden wording:** {r['Forbidden_Wording']}")
        lines.append("")

    lines.append("## 6. 必须进入补充材料或正文说明的项目")
    lines.append("")
    for _, r in supp_df.iterrows():
        lines.append(f"- **{r['Supplementary_Item']}**: {r['Source']}")
        lines.append(f"  - Purpose: {r['Purpose']}")
        lines.append(f"  - Boundary: {r['Required_Boundary']}")
        lines.append("")

    lines.append("## 7. 32J guardrails（声明护栏）原始清单")
    lines.append("")
    for _, r in guardrails_df.iterrows():
        lines.append(f"- **{r.get('Category', 'NA')}**: {r.get('Statement', 'NA')}")
    lines.append("")

    lines.append("## 8. 下一步建议")
    lines.append("")
    lines.append(
        "下一步可以进入 32L：把 32K 形成的结构进一步转换为论文 Results and Discussion（结果与讨论）的正式中文初稿，"
        "并同步生成 Figure caption（图注）、Table caption（表注）和 manuscript checklist（论文检查清单）。"
    )
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        main()
    except Exception as exc:
        # Write failure log in the original project report location.
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        fail_log = REPORT_DIR / "32k_run_log.txt"
        fail_log.write_text(
            "32K failed.\n"
            f"Error type: {type(exc).__name__}\n"
            f"Error message: {exc}\n",
            encoding="utf-8"
        )
        print("32K failed.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error message: {exc}")
        print(f"Failure log saved to: {fail_log}")
        sys.exit(1)
