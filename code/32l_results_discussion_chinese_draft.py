# -*- coding: utf-8 -*-
"""
32L Results and Discussion Chinese draft integration
Project: fluoride coagulation defluoridation ML project

Purpose
-------
Generate a formal Chinese Results and Discussion draft based on the accepted
32J/32K evidence package. This script does NOT rerun models, does NOT change
metrics, and does NOT change the accepted selective AD rule.

Input convention
----------------
Run this file either from the project root or from src/.
Expected project structure:
    fluoride_coagulation_ml_project/
    ├─ outputs/
    │  ├─ tables/
    │  └─ figures/
    ├─ reports/
    └─ src/

Output convention
-----------------
Tables are saved to:  outputs/tables/
Figures are saved to: outputs/figures/  (32L does not create new figures by default)
Reports are saved to: reports/

This follows the existing project storage style and avoids creating a new
step-specific output root folder.
"""

from __future__ import annotations

import os
import re
import sys
import math
import json
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np


# =============================================================================
# Basic utilities
# =============================================================================

SCRIPT_NAME = "32l_results_discussion_chinese_draft.py"
STEP_NAME = "32L Results and Discussion formal Chinese draft"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_project_root() -> Path:
    """Find the project root from cwd or script location.

    The root is identified as a directory that contains outputs/, or data/, or src/.
    Preference is given to a directory containing outputs/tables and outputs/figures.
    """
    candidates: List[Path] = []

    cwd = Path.cwd().resolve()
    candidates.extend([cwd] + list(cwd.parents))

    try:
        script_path = Path(__file__).resolve()
        candidates.extend([script_path.parent] + list(script_path.parent.parents))
    except Exception:
        pass

    # Remove duplicates while keeping order
    unique_candidates: List[Path] = []
    seen = set()
    for c in candidates:
        if str(c) not in seen:
            unique_candidates.append(c)
            seen.add(str(c))

    # Best case: existing outputs/tables and outputs/figures
    for c in unique_candidates:
        if (c / "outputs" / "tables").exists() and (c / "outputs" / "figures").exists():
            return c

    # Accept root with outputs
    for c in unique_candidates:
        if (c / "outputs").exists():
            return c

    # Accept root with data and src
    for c in unique_candidates:
        if (c / "data").exists() and (c / "src").exists():
            return c

    # Fallback: if launched from src, use parent
    if cwd.name.lower() == "src":
        return cwd.parent

    return cwd


def read_csv_smart(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to read CSV: {path}. Last error: {last_err}")


def write_csv_smart(df: pd.DataFrame, path: Path) -> None:
    safe_mkdir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    safe_mkdir(path.parent)
    path.write_text(text, encoding="utf-8")


def fmt_num(x: Any, ndigits: int = 3, na: str = "NA") -> str:
    try:
        if pd.isna(x):
            return na
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x) if x is not None else na


def get_value(row: pd.Series, candidates: List[str], default: Any = None) -> Any:
    lower_map = {str(c).lower(): c for c in row.index}
    for name in candidates:
        if name in row.index:
            return row[name]
        key = name.lower()
        if key in lower_map:
            return row[lower_map[key]]
    return default


def find_file(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


# =============================================================================
# Evidence extraction
# =============================================================================


def load_inputs(table_dir: Path, report_dir: Path) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Load available 32J/32K tables."""
    log: List[str] = []
    inputs: Dict[str, pd.DataFrame] = {}

    expected_tables = {
        "32j_main_result_table": table_dir / "32j_main_result_table.csv",
        "32j_final_evidence_ordered_manifest": table_dir / "32j_final_evidence_ordered_manifest.csv",
        "32j_figure_table_manifest_project_order": table_dir / "32j_figure_table_manifest_project_order.csv",
        "32j_claim_guardrails_for_manuscript": table_dir / "32j_claim_guardrails_for_manuscript.csv",
        "32k_manuscript_results_structure": table_dir / "32k_manuscript_results_structure.csv",
        "32k_figure_caption_table": table_dir / "32k_figure_caption_table.csv",
        "32k_key_claims_and_evidence_map": table_dir / "32k_key_claims_and_evidence_map.csv",
        "32k_results_paragraph_blocks": table_dir / "32k_results_paragraph_blocks.csv",
        "32k_required_supplementary_items": table_dir / "32k_required_supplementary_items.csv",
    }

    for key, path in expected_tables.items():
        if path.exists():
            try:
                inputs[key] = read_csv_smart(path)
                log.append(f"[INPUT] {key}: {path}")
            except Exception as exc:
                log.append(f"[WARN] failed reading {key}: {path} | {exc}")
        else:
            log.append(f"[MISS] {key}: {path}")

    # Reports are optional. They are used only for the log.
    for report_name in ["32k_results_discussion_draft.md", "32k_run_log.txt", "32j_final_manuscript_integration_report.md"]:
        p = report_dir / report_name
        if p.exists():
            log.append(f"[INPUT REPORT] {report_name}: {p}")
        else:
            log.append(f"[MISS REPORT] {report_name}: {p}")

    return inputs, log


def extract_core_results(inputs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Extract main metrics from 32j_main_result_table.csv."""
    default = {
        "strategy": "RepeatedBalanced_Extended",
        "method": "XGBoost",
        "rule": "Zone=Outside; NN_Risk <= 0.814815",
        "r2": 0.747676,
        "rmse": 11.474674,
        "mae": 9.155856,
        "coverage": 0.014664,
        "n_retained": 52,
        "dominant_group": "Paper_010",
        "dominant_group_fraction": 1.0,
        "all_r2": 0.194,
        "all_rmse": 27.51,
        "all_mae": 22.89,
        "all_coverage": 1.0,
        "pre_r2": -0.160,
        "pre_rmse": 29.42,
        "pre_mae": 25.69,
        "pre_coverage": 0.380,
    }

    df = inputs.get("32j_main_result_table")
    if df is None or df.empty:
        return default

    # Identify rows robustly
    def row_where(patterns: List[str]) -> Optional[pd.Series]:
        text_cols = [c for c in df.columns if df[c].dtype == object or "Level" in c or "Rule" in c or "Method" in c]
        if not text_cols:
            text_cols = list(df.columns)
        combined = df[text_cols].astype(str).agg(" ".join, axis=1)
        mask = pd.Series(False, index=df.index)
        for pat in patterns:
            mask = mask | combined.str.contains(pat, case=False, na=False, regex=False)
        if mask.any():
            return df.loc[mask].iloc[-1]
        return None

    all_row = row_where(["All-domain", "all test rows", "full domain"])
    pre_row = row_where(["Predefined", "reliable domain"])
    sel_row = row_where(["Target-free", "selective", "NN_Risk", "retained subdomain"])

    def pull(row: Optional[pd.Series], prefix: str) -> None:
        if row is None:
            return
        if prefix == "sel":
            default["strategy"] = get_value(row, ["Strategy"], default["strategy"])
            default["method"] = get_value(row, ["Method"], default["method"])
            default["rule"] = get_value(row, ["Rule"], default["rule"])
            default["r2"] = get_value(row, ["R2", "R²", "r2"], default["r2"])
            default["rmse"] = get_value(row, ["RMSE", "rmse"], default["rmse"])
            default["mae"] = get_value(row, ["MAE", "mae"], default["mae"])
            default["coverage"] = get_value(row, ["Coverage", "coverage"], default["coverage"])
            default["n_retained"] = get_value(row, ["N_Retained", "N retained", "N"], default["n_retained"])
            default["dominant_group"] = get_value(row, ["Dominant_Group", "Dominant group"], default["dominant_group"])
            default["dominant_group_fraction"] = get_value(row, ["Dominant_Group_Fraction"], default["dominant_group_fraction"])
        else:
            default[f"{prefix}_r2"] = get_value(row, ["R2", "R²", "r2"], default[f"{prefix}_r2"])
            default[f"{prefix}_rmse"] = get_value(row, ["RMSE", "rmse"], default[f"{prefix}_rmse"])
            default[f"{prefix}_mae"] = get_value(row, ["MAE", "mae"], default[f"{prefix}_mae"])
            default[f"{prefix}_coverage"] = get_value(row, ["Coverage", "coverage"], default[f"{prefix}_coverage"])

    pull(all_row, "all")
    pull(pre_row, "pre")
    pull(sel_row, "sel")

    return default


# =============================================================================
# Draft construction
# =============================================================================


def make_section_outline(core: Dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "Section": "3.1",
            "Suggested_Title": "从全域分组验证到可靠性导向诊断",
            "Primary_Evidence": "Table 1; Fig. 194",
            "Core_Message": f"全域分组验证仅达到 R²={fmt_num(core['all_r2'])}，说明跨文献泛化仍然困难，不能只用随机划分性能判断模型价值。",
            "Claim_Boundary": "这是问题诊断，不是最终高性能声明。",
        },
        {
            "Section": "3.2",
            "Suggested_Title": "目标无关的选择性适应域保留子域构建",
            "Primary_Evidence": "Table 1; Fig. 194",
            "Core_Message": f"最终规则为 {core['rule']}，在保留子域内得到 R²={fmt_num(core['r2'], 6)}, RMSE={fmt_num(core['rmse'], 6)}, MAE={fmt_num(core['mae'], 6)}。",
            "Claim_Boundary": "必须写成 selective AD retained-subdomain result，不能写成 all-domain grouped-validation performance。",
        },
        {
            "Section": "3.3",
            "Suggested_Title": "阈值敏感性与反 cherry-picking 证据",
            "Primary_Evidence": "Fig. 195; 32H tables",
            "Core_Message": "NN_Risk 阈值附近存在局部平台，支持该规则不是单点偶然结果。",
            "Claim_Boundary": "必须披露平台较窄、coverage 很低，不能夸大为稳定大范围适应域。",
        },
        {
            "Section": "3.4",
            "Suggested_Title": "保留子域的混凝化学窗口与文献组成",
            "Primary_Evidence": "Fig. 196; 32G tables",
            "Core_Message": f"保留样本集中于特定化学/操作窗口，并由 {core['dominant_group']} 主导。",
            "Claim_Boundary": "这是特定文献/化学窗口的解释，不是普适工艺规律。",
        },
        {
            "Section": "3.5",
            "Suggested_Title": "稳定性、迁移支持与解释边界",
            "Primary_Evidence": "Fig. 197; 32H/32I tables",
            "Core_Message": "部分策略/模型组合在相同阈值下有迁移支持，但 split-level 稳定性和文献主导性仍限制外推。",
            "Claim_Boundary": "应主动声明该结果是高置信保留子域证据，而不是解决全部分组泛化问题。",
        },
        {
            "Section": "3.6",
            "Suggested_Title": "工程含义与后续实验设计方向",
            "Primary_Evidence": "Table 1; Fig. 196-197",
            "Core_Message": "该结果可用于指导后续在特定化学窗口中补充实验和开展主动学习，而不是直接给出全工况最优投药策略。",
            "Claim_Boundary": "工程建议必须限定在 AD retained subdomain 内。",
        },
    ]
    return pd.DataFrame(rows)


def make_claim_sentence_table(core: Dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "Use": "Allowed",
            "Manuscript_Sentence_CN": "在严格的分组验证条件下，模型在全域测试样本上的预测性能有限，表明不同文献来源之间存在明显的数据异质性和外推难度。",
            "Evidence": f"All-domain grouped validation: R²={fmt_num(core['all_r2'])}, RMSE={fmt_num(core['all_rmse'])}, MAE={fmt_num(core['all_mae'])}.",
            "Boundary": "不能把全域结果包装成高性能结果。",
        },
        {
            "Use": "Allowed",
            "Manuscript_Sentence_CN": f"在目标变量不参与筛选的前提下，选择性适应域规则（{core['rule']}）识别出一个高置信保留子域，在该子域内模型达到 R²={fmt_num(core['r2'], 6)}、RMSE={fmt_num(core['rmse'], 6)} 和 MAE={fmt_num(core['mae'], 6)}。",
            "Evidence": "32J main result table; Fig. 194.",
            "Boundary": "只能写为 target-free selective AD retained-subdomain result。",
        },
        {
            "Use": "Allowed with disclosure",
            "Manuscript_Sentence_CN": f"该保留子域仅覆盖全部样本的一小部分（coverage={fmt_num(core['coverage'], 6)}，N={fmt_num(core['n_retained'], 0)}），因此其意义在于识别高可信预测窗口，而不是替代全域模型性能评价。",
            "Evidence": "32J main result table; Fig. 194; Fig. 195.",
            "Boundary": "必须披露低覆盖率。",
        },
        {
            "Use": "Allowed with disclosure",
            "Manuscript_Sentence_CN": f"保留子域受 {core['dominant_group']} 明显主导，说明当前规则主要反映特定文献来源及其对应化学体系中的可预测区域。",
            "Evidence": "Fig. 196; Fig. 197.",
            "Boundary": "不能外推成所有混凝剂体系的普适结论。",
        },
        {
            "Use": "Forbidden",
            "Manuscript_Sentence_CN": "本模型已经在全域分组验证中实现了高精度泛化预测。",
            "Evidence": "Contradicted by all-domain R².",
            "Boundary": "禁止使用。",
        },
        {
            "Use": "Forbidden",
            "Manuscript_Sentence_CN": "NN_Risk 阈值筛选证明模型适用于所有混凝除氟工况。",
            "Evidence": "Contradicted by low coverage and Paper_010 dominance.",
            "Boundary": "禁止使用。",
        },
    ]
    return pd.DataFrame(rows)


def make_figure_caption_table(core: Dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "Figure": "Fig. 1 / Project Fig. 194",
            "Source_File": "194_32i2_evidence_chain_summary_clean.png/pdf",
            "Final_Caption_CN": "全域分组验证、预定义可靠域和选择性适应域保留子域的证据链比较。选择性适应域结果仅代表经过目标无关规则筛选后的保留子域性能，不代表全域分组验证性能。",
            "In_Text_Callout_CN": "如图1所示，全域分组验证与选择性保留子域之间存在明显差异，说明必须将模型全域泛化能力与高置信局部预测能力分开讨论。",
        },
        {
            "Figure": "Fig. 2 / Project Fig. 195",
            "Source_File": "195_32i2_threshold_plateau_evidence_clean.png/pdf",
            "Final_Caption_CN": "NN_Risk 阈值敏感性分析及接受规则的局部平台证据。虚线表示最终采用的阈值；该图用于评估阈值选择是否存在单点偶然性。",
            "In_Text_Callout_CN": "图2显示，接受阈值附近存在一定平台特征，但覆盖度仍然较低，因此该规则应解释为窄窗口下的高置信筛选规则。",
        },
        {
            "Figure": "Fig. 3 / Project Fig. 196",
            "Source_File": "196_32i2_retained_chemistry_window_clean.png/pdf",
            "Final_Caption_CN": "选择性适应域保留子域的化学窗口、混凝剂组成、文献组构成和适应域组件水平。该图用于解释保留样本对应的化学与数据结构特征。",
            "In_Text_Callout_CN": "图3进一步表明，保留子域并非随机分布于全部实验空间，而是集中于特定混凝剂组成和操作窗口。",
        },
        {
            "Figure": "Fig. 4 / Project Fig. 197",
            "Source_File": "197_32i2_stability_transfer_boundary_clean.png/pdf",
            "Final_Caption_CN": "选择性保留子域的分割稳定性、迁移支持和解释边界。该图同时展示了结果的有效证据和必须披露的局限性。",
            "In_Text_Callout_CN": "图4说明，虽然若干策略/模型组合支持该保留子域信号，但结果仍受到分割稳定性和文献组主导性的限制。",
        },
        {
            "Figure": "Fig. 5 / Project Fig. 198",
            "Source_File": "198_32k_results_discussion_logic_map.png/pdf",
            "Final_Caption_CN": "32K 形成的 Results and Discussion 逻辑结构图，用于展示从模型诊断、AD 筛选、阈值验证到化学解释和边界声明的论证链条。",
            "In_Text_Callout_CN": "该图可作为论文撰写或组会汇报的逻辑辅助图，不一定作为正式投稿主图。",
        },
    ]
    return pd.DataFrame(rows)


def make_limitations_table(core: Dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "Limitation": "低覆盖率",
            "Specific_Evidence": f"Coverage={fmt_num(core['coverage'], 6)}; N retained={fmt_num(core['n_retained'], 0)}",
            "How_To_Write": "该规则识别的是高置信保留子域，而不是大覆盖率适应域。",
            "Follow_Up_Work": "增加该化学窗口外的补充实验，并扩大低置信区域样本。",
        },
        {
            "Limitation": "文献组主导性",
            "Specific_Evidence": f"Dominant group={core['dominant_group']}",
            "How_To_Write": "保留子域具有明确的文献来源偏向，不能直接外推到所有文献和所有混凝剂体系。",
            "Follow_Up_Work": "按文献来源设计分层采样或跨文献外部验证实验。",
        },
        {
            "Limitation": "split-level stability 有限",
            "Specific_Evidence": "Fig. 197 shows split-level difference.",
            "How_To_Write": "部分划分支持高 R²，但不同划分之间仍存在波动。",
            "Follow_Up_Work": "进行更多重复分组验证和外部实验验证。",
        },
        {
            "Limitation": "全域分组泛化仍未解决",
            "Specific_Evidence": f"All-domain grouped-validation R²={fmt_num(core['all_r2'])}",
            "How_To_Write": "全域预测能力仍受数据异质性限制，本文的主要贡献在于识别可解释的高置信预测子域。",
            "Follow_Up_Work": "结合主动学习、分区建模和机理约束特征继续扩展模型。",
        },
    ]
    return pd.DataFrame(rows)


def make_chinese_draft(core: Dict[str, Any], section_outline: pd.DataFrame) -> str:
    """Create formal Chinese Results and Discussion draft."""
    all_r2 = fmt_num(core["all_r2"])
    all_rmse = fmt_num(core["all_rmse"])
    all_mae = fmt_num(core["all_mae"])
    pre_r2 = fmt_num(core["pre_r2"])
    pre_rmse = fmt_num(core["pre_rmse"])
    pre_mae = fmt_num(core["pre_mae"])
    pre_cov = fmt_num(core["pre_coverage"])
    r2 = fmt_num(core["r2"], 6)
    rmse = fmt_num(core["rmse"], 6)
    mae = fmt_num(core["mae"], 6)
    cov = fmt_num(core["coverage"], 6)
    nret = fmt_num(core["n_retained"], 0)
    dominant_group = str(core["dominant_group"])
    strategy = str(core["strategy"])
    method = str(core["method"])
    rule = str(core["rule"])

    draft = f"""
# 32L Results and Discussion 中文初稿

> 生成时间：{now_str()}  
> 说明：本文件由 `{SCRIPT_NAME}` 自动生成。该步骤不重新训练模型，不改变 32I2 最终规则和指标，仅将 32J/32K 证据链整理为论文 Results and Discussion（结果与讨论）的中文正文草稿。

## 3. Results and Discussion（结果与讨论）

### 3.1 全域分组验证揭示跨文献泛化难度

为避免随机划分造成相似实验条件同时进入训练集和测试集，本研究在模型评价阶段引入 grouped validation（分组验证）思路，将不同文献来源或实验组作为更严格的外推边界。与随机划分相比，grouped validation（分组验证）更接近真实应用中“用已有文献数据预测新文献或新体系”的任务，因此可以更直接地检验模型的跨来源泛化能力。

在 all-domain grouped validation（全域分组验证）下，模型性能为 R²={all_r2}、RMSE={all_rmse}、MAE={all_mae}。该结果说明，混凝沉淀除氟数据并不是一个容易通过单一模型直接全域泛化的数据集。造成这一现象的原因可能包括：不同文献之间实验体系不一致，混凝剂类型和投加量范围差异较大，进水 pH、初始氟浓度、干扰离子和反应时间等变量存在强耦合，同时部分关键机理变量存在记录缺失或表达方式不统一。因此，全域分组验证结果不应被简单理解为模型失败，而应作为后续进行 applicability domain（适应域）分析和选择性预测的出发点。

进一步地，predefined reliable domain（预定义可靠域）并未带来性能提升，其结果为 R²={pre_r2}、RMSE={pre_rmse}、MAE={pre_mae}，coverage（覆盖率）约为 {pre_cov}。这说明仅依靠固定规则预先划定可靠区域，并不足以有效识别真正可预测的样本空间。换言之，混凝除氟体系中的可靠预测区域并不只由单一距离、杠杆值或常规统计边界决定，而更可能由多种 AD component（适应域组件）共同约束。

### 3.2 目标无关选择性 AD 识别高置信保留子域

在全域分组验证和预定义可靠域均难以获得稳定高性能的基础上，本研究进一步构建 target-free selective AD（目标无关选择性适应域）筛选策略。这里的“目标无关”是指筛选规则不直接使用测试集真实去除率作为筛选依据，而是基于输入空间、模型预测结构和适应域风险指标识别高置信预测子域，从而降低 target leakage（目标泄露）的风险。

最终接受的选择性规则为：`{rule}`。在该规则下，{strategy} + {method} 在 retained subdomain（保留子域）中取得 R²={r2}、RMSE={rmse}、MAE={mae}。与全域分组验证相比，该保留子域中的预测误差显著降低，说明模型在特定输入空间内能够捕捉较稳定的混凝除氟规律。

但是，需要强调的是，该结果必须被表述为 **target-free selective AD retained-subdomain result（目标无关选择性适应域保留子域结果）**，而不能被表述为 all-domain grouped-validation performance（全域分组验证性能）。其原因在于，该规则仅保留了全部测试预测中的一小部分样本，coverage={cov}，retained prediction instances（保留预测实例数）为 N={nret}。因此，该结果的科学意义不是证明模型已经解决了所有混凝除氟工况下的泛化问题，而是证明在特定、可识别、目标无关的高置信输入子域内，模型预测具有明显更高的可靠性。

### 3.3 阈值敏感性分析支持该规则不是单点偶然结果

为检验 `NN_Risk <= 0.814815` 这一阈值是否只是偶然选择，本研究进一步开展 threshold sensitivity analysis（阈值敏感性分析）。结果显示，在接受阈值附近，retained-subdomain R²（保留子域 R²）存在局部平台区间，而不是仅在单一阈值点上突然出现高性能。这一结果为最终规则提供了 anti-cherry-picking（反选择性挑选）证据，即该规则并非完全依赖某一个孤立阈值点。

同时，阈值敏感性图也揭示了该规则的重要局限。随着 NN_Risk threshold（NN 风险阈值）放宽，保留样本数和 coverage（覆盖率）逐渐增加，但 R² 会明显下降。这说明高预测性能与低覆盖率之间存在权衡：更严格的 AD 筛选可以提高保留子域的预测可靠性，但会牺牲可应用样本范围。因此，本研究不将该规则解释为普适适应域边界，而是将其定位为一个 narrow but reproducible high-confidence retained subdomain（较窄但可复现的高置信保留子域）。

### 3.4 保留子域对应特定混凝化学窗口

为了进一步解释选择性 AD 筛选结果的物理化学含义，本研究对 retained subdomain（保留子域）中的水质条件、混凝剂组成、文献来源和 AD component levels（适应域组件水平）进行了分析。结果显示，保留样本并非随机分布于全部实验空间，而是集中在特定混凝剂组成和操作条件窗口中。这表明选择性 AD 筛选并不是简单地“挑选预测好的点”，而是在输入空间中识别出一个更容易被当前模型可靠学习的化学区域。

从混凝除氟机理角度看，这一现象具有合理性。混凝除氟过程受 coagulant hydrolysis（混凝剂水解）、fluoride adsorption（氟离子吸附）、complexation（络合）、co-precipitation（共沉淀）和 floc formation（絮体形成）共同影响。pH 决定铝盐或铁盐水解形态和絮体表面电荷，投加量决定有效水解产物和吸附位点数量，初始氟浓度决定污染负荷，干扰离子则可能通过竞争吸附或改变沉淀平衡影响去除效果。因此，模型在某些化学窗口内表现更可靠，符合混凝体系非线性和多因子耦合的基本特征。

然而，保留子域也表现出明显的 literature-group dominance（文献组主导性），其中 dominant group（主导文献组）为 {dominant_group}。这意味着当前保留子域的规律很大程度上受到特定文献来源和实验体系的影响。因此，该结果不能直接推广为所有混凝剂、所有水质条件和所有文献来源下的通用结论。更合理的表述是：当前模型在 {dominant_group} 所代表的相近化学与操作窗口内识别出了一个高置信预测区域。

### 3.5 稳定性、迁移支持与解释边界

为了评估该选择性 AD 规则的稳健性，本研究进一步比较了不同 split（数据划分）、不同 strategy/model combination（策略/模型组合）下的保留子域表现。结果表明，在若干组合中，相同或相近阈值规则仍能够保留 R² 较高的预测子域，说明该信号具有一定 transfer support（迁移支持）。这为选择性 AD 规则提供了额外证据：其并非完全依赖单一模型或单一运行结果。

但与此同时，split-level retained-subdomain R²（划分层面保留子域 R²）仍存在波动，并且文献组主导性较强。这提示我们，当前规则的稳定性尚不足以支撑全域外推声明。特别是在保留样本数量较少且主要来自单一文献组的情况下，模型的高性能更应被解释为特定数据子域中的可靠预测，而不是对所有未知文献体系的稳健泛化。

因此，本文在结果表述上采用较为保守的边界：本研究证明了 selective AD（选择性适应域）能够从异质性较强的小样本环境实验数据中识别高置信预测子域；但并不声称模型已经在 all-domain grouped validation（全域分组验证）上实现高精度泛化，也不声称当前规则可以无条件用于所有混凝沉淀除氟体系。

### 3.6 工程意义与后续研究方向

从工程应用角度看，本研究结果提示，在混凝沉淀除氟这类数据异质性强、文献来源复杂、实验条件不均衡的体系中，单纯追求全域模型 R² 可能并不是最合理的建模目标。更有价值的方向是同时回答两个问题：第一，模型在哪些条件下不可靠；第二，模型在哪些条件下可以提供高置信预测。选择性 AD 分析正是围绕这两个问题展开。

当前 high-confidence retained subdomain（高置信保留子域）可为后续实验设计提供依据。一方面，可以在该子域内开展重复实验或外部验证，检验模型预测的真实可靠性；另一方面，也可以围绕该子域边界设计主动学习实验，逐步扩大可靠预测范围。对于低置信区域，则应优先补充样本，尤其是不同文献来源、不同混凝剂体系、不同 pH 和干扰离子条件下的数据，以降低模型对单一文献组的依赖。

总体而言，本研究的核心贡献不在于宣称一个模型已经解决所有混凝除氟工况预测问题，而在于建立了一条从 grouped validation（分组验证）诊断、reliability-domain screening（可靠域筛选）、target-free selective AD（目标无关选择性适应域）到 chemistry-window interpretation（化学窗口解释）的证据链。该证据链为小样本、异质性环境实验数据中的机器学习建模提供了一种更审慎、更可解释的结果表达方式。

---

## 建议图表对应关系

- Fig. 1：194_32i2_evidence_chain_summary_clean，用于展示三层证据链。
- Fig. 2：195_32i2_threshold_plateau_evidence_clean，用于展示阈值敏感性和平台证据。
- Fig. 3：196_32i2_retained_chemistry_window_clean，用于展示保留子域化学窗口。
- Fig. 4：197_32i2_stability_transfer_boundary_clean，用于展示稳定性、迁移支持和边界。
- Fig. 5：198_32k_results_discussion_logic_map，可作为补充图或组会汇报逻辑图。

## 必须保留的声明边界

1. P08 / 选择性 AD 结果不能写成全域分组验证性能。
2. 必须披露 coverage={cov} 和 N={nret}。
3. 必须披露 dominant group={dominant_group}。
4. 必须说明该结果是高置信保留子域，而不是普适工艺规律。
5. 32I2 只是最终图件清理，不改变模型、不改变规则、不改变指标。
"""

    return textwrap.dedent(draft).strip() + "\n"


def make_run_log(project_root: Path, table_dir: Path, figure_dir: Path, report_dir: Path,
                 input_log: List[str], output_paths: List[Path], core: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"{STEP_NAME}")
    lines.append(f"Run time: {now_str()}")
    lines.append(f"Project root: {project_root}")
    lines.append(f"Table output dir: {table_dir}")
    lines.append(f"Figure output dir: {figure_dir}")
    lines.append(f"Report output dir: {report_dir}")
    lines.append("=" * 80)
    lines.extend(input_log)
    lines.append("=" * 80)
    lines.append("[KEY RESULT]")
    lines.append(f"Selective AD retained-subdomain result: {core['strategy']} + {core['method']}")
    lines.append(f"Rule: {core['rule']}")
    lines.append(
        f"R2={fmt_num(core['r2'], 6)}, RMSE={fmt_num(core['rmse'], 6)}, "
        f"MAE={fmt_num(core['mae'], 6)}, Coverage={fmt_num(core['coverage'], 6)}, "
        f"N={fmt_num(core['n_retained'], 0)}, Dominant group={core['dominant_group']}"
    )
    lines.append("=" * 80)
    for p in output_paths:
        lines.append(f"[OUTPUT] {p}")
    lines.append("=" * 80)
    lines.append("32L finished successfully.")
    lines.append("Tables saved to original project table directory.")
    lines.append("Reports saved to original project report directory.")
    lines.append("No model was rerun. No accepted 32I2 metric was changed.")
    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    project_root = find_project_root()
    table_dir = safe_mkdir(project_root / "outputs" / "tables")
    figure_dir = safe_mkdir(project_root / "outputs" / "figures")
    report_dir = safe_mkdir(project_root / "reports")

    inputs, input_log = load_inputs(table_dir, report_dir)
    core = extract_core_results(inputs)

    section_outline = make_section_outline(core)
    claim_sentence_table = make_claim_sentence_table(core)
    figure_caption_table = make_figure_caption_table(core)
    limitations_table = make_limitations_table(core)
    draft_text = make_chinese_draft(core, section_outline)

    output_paths: List[Path] = []

    out1 = table_dir / "32l_results_discussion_section_outline.csv"
    write_csv_smart(section_outline, out1)
    output_paths.append(out1)

    out2 = table_dir / "32l_formal_claim_sentence_table.csv"
    write_csv_smart(claim_sentence_table, out2)
    output_paths.append(out2)

    out3 = table_dir / "32l_final_figure_caption_chinese.csv"
    write_csv_smart(figure_caption_table, out3)
    output_paths.append(out3)

    out4 = table_dir / "32l_limitations_and_disclosures.csv"
    write_csv_smart(limitations_table, out4)
    output_paths.append(out4)

    draft_path = report_dir / "32l_results_discussion_chinese_draft.md"
    write_text(draft_path, draft_text)
    output_paths.append(draft_path)

    # Plain txt version for easy copy into Word/WPS
    txt_path = report_dir / "32l_results_discussion_chinese_draft.txt"
    write_text(txt_path, draft_text.replace("# ", "").replace("## ", "").replace("### ", ""))
    output_paths.append(txt_path)

    log_text = make_run_log(project_root, table_dir, figure_dir, report_dir, input_log, output_paths, core)
    log_path = report_dir / "32l_run_log.txt"
    write_text(log_path, log_text)
    output_paths.append(log_path)

    print(STEP_NAME)
    print(f"Run time: {now_str()}")
    print(f"Project root: {project_root}")
    print(f"Table output dir: {table_dir}")
    print(f"Figure output dir: {figure_dir}")
    print(f"Report output dir: {report_dir}")
    print("=" * 80)
    for line in input_log:
        print(line)
    print("=" * 80)
    print("[KEY RESULT]")
    print(f"Selective AD retained-subdomain result: {core['strategy']} + {core['method']}")
    print(f"Rule: {core['rule']}")
    print(
        f"R2={fmt_num(core['r2'], 6)}, RMSE={fmt_num(core['rmse'], 6)}, "
        f"MAE={fmt_num(core['mae'], 6)}, Coverage={fmt_num(core['coverage'], 6)}, "
        f"N={fmt_num(core['n_retained'], 0)}, Dominant group={core['dominant_group']}"
    )
    for p in output_paths:
        print(f"[OUTPUT] {p}")
    print("\n32L finished successfully.")
    print(f"Tables saved to: {table_dir}")
    print(f"Reports saved to: {report_dir}")


if __name__ == "__main__":
    main()
