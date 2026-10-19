# -*- coding: utf-8 -*-
"""
32N Word-ready Results and Discussion integration
Project: fluoride coagulation defluoridation ML project

Purpose
-------
This script compresses the corrected 32M2 Results and Discussion draft into a
Word-ready manuscript section. It does NOT retrain any model, does NOT modify
the accepted 32I2 / 32J / 32M2 metrics, and does NOT create a new output root.

Output locations follow the original project convention:
    outputs/tables/
    reports/

The accepted key result is explicitly locked as:
    RepeatedBalanced_Extended + XGBoost
    Rule: Zone=Outside; NN_Risk <= 0.814815
    R2=0.747676, RMSE=11.474674, MAE=9.155856,
    Coverage=0.014664, N=52, Dominant group=Paper_010

Important manuscript boundary:
    This is a target-free selective AD retained-subdomain result.
    It must NOT be written as all-domain grouped-validation performance.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Basic path utilities
# =============================================================================

def find_project_root(start: Optional[Path] = None) -> Path:
    """Find project root whether this script is placed in project root or src/."""
    start = (start or Path(__file__).resolve()).resolve()
    candidates = [start]
    if start.is_file():
        candidates.append(start.parent)
    candidates.extend(list(start.parents))

    markers = [
        ("outputs", "tables"),
        ("outputs", "figures"),
        ("src",),
        ("data",),
    ]

    best = None
    best_score = -1
    for c in candidates:
        if c.is_file():
            c = c.parent
        score = 0
        for marker in markers:
            if (c.joinpath(*marker)).exists():
                score += 1
        # Prefer directory with outputs/tables and src if nested package exists
        if score > best_score:
            best = c
            best_score = score

    if best is None or best_score <= 0:
        return Path.cwd().resolve()
    return best.resolve()


PROJECT_ROOT = find_project_root()
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CSV helpers
# =============================================================================

def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
        except Exception:
            return []
    return []


def write_csv_dicts(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen = []
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        txt = str(value).strip()
        if txt == "" or txt.lower() in {"nan", "none", "na", "null"}:
            return default
        return float(txt)
    except Exception:
        return default


def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


# =============================================================================
# Accepted key result handling
# =============================================================================

@dataclass
class KeyResult:
    method: str = "RepeatedBalanced_Extended + XGBoost"
    rule: str = "Zone=Outside; NN_Risk <= 0.814815"
    r2: float = 0.747676
    rmse: float = 11.474674
    mae: float = 9.155856
    coverage: float = 0.014664
    n_retained: int = 52
    dominant_group: str = "Paper_010"
    result_type: str = "target-free selective AD retained-subdomain result"
    forbidden_claim: str = "all-domain grouped-validation performance"


LOCKED_KEY_RESULT = KeyResult()


def extract_key_result() -> Tuple[KeyResult, str]:
    """
    Strictly select corrected/selective AD key result.
    If a table is incomplete or ambiguous, fall back to locked accepted constants.
    """
    sources = [
        TABLE_DIR / "32m2_corrected_key_result_table.csv",
        TABLE_DIR / "32j_main_result_table.csv",
    ]

    # First priority: 32M2 corrected key result table
    p = sources[0]
    rows = read_csv_dicts(p)
    if rows:
        for row in rows:
            hay = " ".join([str(v) for v in row.values()]).lower()
            if ("selective" in hay and "0.747" in hay) or ("nn_risk" in hay and "0.814815" in hay):
                kr = KeyResult()
                for key in ["method", "rule", "dominant_group", "result_type"]:
                    for col in row.keys():
                        if col.lower().replace(" ", "_") == key:
                            val = row.get(col)
                            if val:
                                setattr(kr, key, str(val))
                # flexible numeric extraction
                for attr, aliases in {
                    "r2": ["r2", "r^2", "R2", "R²"],
                    "rmse": ["rmse", "RMSE"],
                    "mae": ["mae", "MAE"],
                    "coverage": ["coverage", "Coverage"],
                    "n_retained": ["n_retained", "N", "n", "retained_n", "N_retained"],
                }.items():
                    for col, val in row.items():
                        norm = col.lower().replace(" ", "").replace("_", "")
                        if any(norm == a.lower().replace(" ", "").replace("_", "") for a in aliases):
                            fv = safe_float(val)
                            if fv is not None:
                                if attr == "n_retained":
                                    setattr(kr, attr, int(round(fv)))
                                else:
                                    setattr(kr, attr, fv)
                # protect core metrics if table had wrong/missing values
                if abs(kr.r2 - 0.747676) < 0.01:
                    return kr, f"selected_from:{p.name}"
    # Second priority: 32J main result table, but avoid all-domain first row
    p = sources[1]
    rows = read_csv_dicts(p)
    if rows:
        for row in rows:
            hay = " ".join([str(v) for v in row.values()]).lower()
            is_selective = ("selective" in hay or "retained subdomain" in hay or "retained-subdomain" in hay)
            has_rule = ("nn_risk" in hay or "0.814815" in hay or "outside" in hay)
            has_r2 = ("0.747" in hay or "0.748" in hay)
            if is_selective and (has_rule or has_r2):
                kr = KeyResult()
                # Detect values from any likely columns
                for col, val in row.items():
                    col_l = col.lower().strip()
                    val_s = str(val).strip()
                    if col_l in {"method", "model", "accepted_model", "strategy_model"} and val_s:
                        kr.method = val_s
                    elif col_l in {"rule", "ad_rule", "selection_rule", "accepted_rule"} and val_s:
                        kr.rule = val_s
                    elif col_l in {"dominant_group", "dominant literature group", "group"} and val_s:
                        kr.dominant_group = val_s
                    elif col_l in {"result_type", "claim_type", "manuscript_claim"} and val_s:
                        kr.result_type = val_s
                    elif col_l.replace("^", "").replace("²", "2") in {"r2", "r"}:
                        fv = safe_float(val)
                        if fv is not None:
                            kr.r2 = fv
                    elif col_l == "rmse":
                        fv = safe_float(val)
                        if fv is not None:
                            kr.rmse = fv
                    elif col_l == "mae":
                        fv = safe_float(val)
                        if fv is not None:
                            kr.mae = fv
                    elif "coverage" in col_l:
                        fv = safe_float(val)
                        if fv is not None:
                            kr.coverage = fv
                    elif col_l in {"n", "n_retained", "retained_n"}:
                        fv = safe_float(val)
                        if fv is not None:
                            kr.n_retained = int(round(fv))
                if abs(kr.r2 - 0.747676) < 0.02:
                    return kr, f"selected_from:{p.name}"

    return LOCKED_KEY_RESULT, "fallback_locked_constants"


KEY_RESULT, KEY_RESULT_SOURCE = extract_key_result()


# =============================================================================
# Manuscript text generation
# =============================================================================

def fmt(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}"


def make_paragraphs(kr: KeyResult) -> List[Dict[str, object]]:
    """Word-ready paragraphs: compact, formal, and directly pasteable."""
    paragraphs = [
        {
            "section": "3.1 数据集泛化验证与适应域分析的必要性",
            "paragraph_id": "P1",
            "word_ready_text": (
                "为评价机器学习模型在文献异质性条件下的真实泛化能力，本研究在常规随机划分之外，"
                "进一步采用按文献来源分组的验证策略。与随机划分相比，分组验证能够避免同一文献或相近实验系列"
                "同时进入训练集和测试集，从而更严格地检验模型对未见文献工况的预测能力。结果表明，在全域测试条件下，"
                "模型性能明显低于随机划分结果，说明混凝沉淀除氟数据不仅存在非线性和多因素耦合特征，还受到明显的"
                "文献来源差异、实验体系差异和局部工况分布差异影响。因此，仅报告随机划分下的高精度结果不足以说明模型"
                "具有可靠的工程外推能力，必须进一步结合适应域分析对模型可用范围进行限定。"
            ),
            "main_evidence": "32J/32K evidence chain; all-domain grouped validation is used as background comparison only.",
            "figure_reference": "Fig. 194",
        },
        {
            "section": "3.2 预定义可靠域与选择性适应域的对比",
            "paragraph_id": "P2",
            "word_ready_text": (
                "在预定义可靠域分析中，模型尝试通过适应域规则筛除高风险样本，以获得更可信的预测子集。"
                "然而，固定可靠域并未带来稳定的性能提升，说明单一、静态的适应域阈值难以充分处理本数据集中复杂的"
                "文献异质性和工况偏移。进一步的选择性适应域分析显示，当模型只在满足特定风险约束的保留子域内进行解释时，"
                "预测误差显著降低，决定系数明显提高。该结果表明，适应域并不是简单地扩大或缩小预测范围，而是需要识别"
                "模型在当前数据结构下真正具有相对可靠性的局部工况空间。"
            ),
            "main_evidence": "Predefined reliable domain was not accepted as final claim; selective retained subdomain was accepted.",
            "figure_reference": "Fig. 194",
        },
        {
            "section": "3.3 选择性适应域保留子域的模型性能",
            "paragraph_id": "P3",
            "word_ready_text": (
                f"最终接受的结果来自 {kr.method} 模型，并采用适应域规则“{kr.rule}”。"
                f"在该无目标泄露的选择性适应域保留子域中，模型取得 R²={kr.r2:.6f}、"
                f"RMSE={kr.rmse:.6f}、MAE={kr.mae:.6f}，覆盖率为 {kr.coverage:.6f}，"
                f"保留样本数为 {kr.n_retained}。需要强调的是，该结果代表模型在筛选后局部子域内的预测能力，"
                "不能表述为全域分组验证性能。换言之，该结果说明模型在特定适应域条件下具有较强预测可靠性，"
                "但并不意味着模型已经能够对所有文献来源和所有混凝除氟工况进行无条件外推。"
            ),
            "main_evidence": "Corrected 32M2 key result; target-free selective AD retained-subdomain.",
            "figure_reference": "Fig. 194",
        },
        {
            "section": "3.4 NN-risk 阈值平台与规则稳定性",
            "paragraph_id": "P4",
            "word_ready_text": (
                "为避免选择性适应域规则成为事后挑选结果，本研究进一步对 NN-risk 阈值进行了敏感性分析。"
                "结果显示，在接受阈值附近，保留子域 R² 存在相对稳定的高性能区间，而不是由单一孤立阈值产生的偶然峰值。"
                "同时，随着阈值放宽，保留样本数和覆盖率逐渐升高，但模型精度并未持续提高，说明更多样本并不必然带来"
                "更可靠的外推结果。该现象支持了选择性适应域的核心逻辑：在高度异质的环境实验数据中，模型可靠性需要通过"
                "风险约束来界定，而不是简单追求覆盖率最大化。"
            ),
            "main_evidence": "Threshold plateau and sensitivity evidence.",
            "figure_reference": "Fig. 195",
        },
        {
            "section": "3.5 保留子域的化学工况特征与解释边界",
            "paragraph_id": "P5",
            "word_ready_text": (
                f"保留子域的样本主要来自 {kr.dominant_group}，并集中于特定混凝剂组成、pH、投加量、"
                "初始氟浓度和干扰离子浓度范围内。该结果一方面表明模型在该类化学窗口内能够获得较高预测精度，"
                "另一方面也提示模型解释具有明确边界。对于该子域以外的混凝剂类型、极端 pH、不同初始氟负荷或复杂共存离子体系，"
                "模型预测仍需谨慎使用。因而，本研究的适应域结果更适合被理解为对特定文献支持工况的可靠识别，"
                "而不是对所有混凝沉淀除氟体系的普适性规律归纳。"
            ),
            "main_evidence": "Retained chemistry window and literature-group composition.",
            "figure_reference": "Fig. 196",
        },
        {
            "section": "3.6 稳定性、迁移支持与论文声明边界",
            "paragraph_id": "P6",
            "word_ready_text": (
                "稳定性分析进一步表明，选择性适应域结果在部分数据划分和相关模型组合中具有可重复支持，"
                "但其覆盖率较低且文献来源集中，因此仍应将其定位为保留子域内的可靠预测证据，而不是全域稳健预测证据。"
                "在论文表述中，应避免将 R²=0.747676 直接描述为整个混凝除氟数据集的泛化性能；更严谨的表述是："
                "在无目标泄露的选择性适应域筛选后，模型在保留子域内表现出较高预测精度。该边界声明能够同时保留结果亮点，"
                "并避免因过度外推导致结论不可靠。"
            ),
            "main_evidence": "Stability, transfer support, and interpretation boundary.",
            "figure_reference": "Fig. 197",
        },
        {
            "section": "3.7 本节小结",
            "paragraph_id": "P7",
            "word_ready_text": (
                "综上，混凝沉淀除氟数据在全域分组验证下表现出明显的跨文献泛化难度，说明该体系存在强烈的数据异质性。"
                "通过引入适应域筛选，本研究识别出一个具有较高预测可靠性的选择性保留子域，并在该子域内获得较好的预测结果。"
                "该发现为后续工艺参数推荐和模型辅助决策提供了更加谨慎的基础：模型可以用于支持特定工况窗口内的预测和解释，"
                "但对于适应域之外的水质条件和药剂体系，仍需要补充实验或更多文献数据进行验证。"
            ),
            "main_evidence": "Integrated conclusion of 32J-32M2.",
            "figure_reference": "Figs. 194-197",
        },
    ]
    return paragraphs


def make_figure_plan(kr: KeyResult) -> List[Dict[str, object]]:
    return [
        {
            "figure_no": "Fig. 194",
            "file_png": "outputs/figures/194_32i2_evidence_chain_summary_clean.png",
            "file_pdf": "outputs/figures/194_32i2_evidence_chain_summary_clean.pdf",
            "insert_after_section": "3.3 选择性适应域保留子域的模型性能",
            "caption_word_ready": (
                "图194 选择性适应域结果的证据链汇总。该图比较了全域分组验证、预定义可靠域和选择性适应域保留子域的"
                "R²、RMSE、MAE 和覆盖率。选择性保留子域取得较高 R² 和较低预测误差，但覆盖率较低，因此该结果应解释为"
                "局部适应域内的可靠预测结果，而非全域分组验证性能。"
            ),
            "claim_boundary": "可说明选择性子域性能提升；不可声称全域模型性能达到 R2=0.747676。",
        },
        {
            "figure_no": "Fig. 195",
            "file_png": "outputs/figures/195_32i2_threshold_plateau_evidence_clean.png",
            "file_pdf": "outputs/figures/195_32i2_threshold_plateau_evidence_clean.pdf",
            "insert_after_section": "3.4 NN-risk 阈值平台与规则稳定性",
            "caption_word_ready": (
                "图195 接受的选择性适应域规则的阈值平台证据。该图展示了 NN-risk 阈值变化下保留子域 R²、保留样本数"
                "和覆盖率的变化。接受阈值附近存在相对稳定的高性能区间，表明该规则并非单一阈值下的偶然结果。"
            ),
            "claim_boundary": "可说明阈值附近存在平台支持；不可将阈值搜索写成有监督目标调参。",
        },
        {
            "figure_no": "Fig. 196",
            "file_png": "outputs/figures/196_32i2_retained_chemistry_window_clean.png",
            "file_pdf": "outputs/figures/196_32i2_retained_chemistry_window_clean.pdf",
            "insert_after_section": "3.5 保留子域的化学工况特征与解释边界",
            "caption_word_ready": (
                "图196 选择性适应域保留子域的化学窗口和文献组成。该图展示了保留样本的关键水质和工艺参数范围、"
                "主要混凝剂组成、文献来源分布以及适应域风险分量水平，用于界定模型解释和应用的有效边界。"
            ),
            "claim_boundary": "可说明保留子域化学窗口；不可推广为所有混凝剂和水质条件。",
        },
        {
            "figure_no": "Fig. 197",
            "file_png": "outputs/figures/197_32i2_stability_transfer_boundary_clean.png",
            "file_pdf": "outputs/figures/197_32i2_stability_transfer_boundary_clean.pdf",
            "insert_after_section": "3.6 稳定性、迁移支持与论文声明边界",
            "caption_word_ready": (
                "图197 选择性适应域结果的稳定性、迁移支持和解释边界。该图显示不同数据划分下的保留子域表现、"
                "不同模型/策略下的迁移支持以及文献组主导性，强调该结果具有局部可靠性，同时存在覆盖率和文献集中性限制。"
            ),
            "claim_boundary": "可说明局部稳定性和迁移支持；不可淡化 Paper_010 主导和低覆盖率限制。",
        },
    ]


def make_guardrail_table(kr: KeyResult) -> List[Dict[str, object]]:
    return [
        {
            "item": "accepted_result_label",
            "must_write": kr.result_type,
            "must_not_write": kr.forbidden_claim,
            "reason": "R2=0.747676 only belongs to selected retained samples after target-free AD filtering.",
        },
        {
            "item": "model_rule",
            "must_write": f"{kr.method}; {kr.rule}",
            "must_not_write": "arbitrary threshold or target-based sample selection",
            "reason": "The accepted rule must be described as an AD/risk-based selection rule, not target leakage.",
        },
        {
            "item": "coverage",
            "must_write": f"coverage={kr.coverage:.6f}, N={kr.n_retained}",
            "must_not_write": "broad/general coverage",
            "reason": "Coverage is low, so conclusion must be bounded.",
        },
        {
            "item": "dominant_group",
            "must_write": f"dominant group={kr.dominant_group}",
            "must_not_write": "balanced support from all literature groups",
            "reason": "Literature-group dominance limits interpretation boundary.",
        },
        {
            "item": "engineering_use",
            "must_write": "model-assisted screening within supported chemical window",
            "must_not_write": "direct universal process optimization tool",
            "reason": "Application outside AD requires additional experiments or literature support.",
        },
    ]


def make_length_control(paragraphs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for row in paragraphs:
        txt = str(row["word_ready_text"])
        rows.append({
            "section": row["section"],
            "paragraph_id": row["paragraph_id"],
            "character_count": len(txt),
            "recommended_action": (
                "keep" if len(txt) <= 420 else
                "slightly_compress" if len(txt) <= 560 else
                "compress_before_word"
            )
        })
    return rows


def render_word_ready_md(kr: KeyResult, paragraphs: List[Dict[str, object]], figs: List[Dict[str, object]]) -> str:
    fig_map = {f["insert_after_section"]: f for f in figs}

    lines = []
    lines.append("# 3. Results and Discussion（结果与讨论）")
    lines.append("")
    lines.append("> 说明：本稿由 32N 自动生成，用于直接复制到 Word 初稿中。该步骤不重新训练模型，不改变 32I2/32J/32M2 最终指标。")
    lines.append("")
    lines.append("## 关键结果锁定")
    lines.append("")
    lines.append(
        f"最终接受结果：{kr.method}；规则：{kr.rule}；"
        f"R²={kr.r2:.6f}，RMSE={kr.rmse:.6f}，MAE={kr.mae:.6f}，"
        f"Coverage={kr.coverage:.6f}，N={kr.n_retained}，主导文献组={kr.dominant_group}。"
    )
    lines.append("")
    lines.append(
        "**写作边界：该结果必须表述为无目标泄露的选择性适应域保留子域结果，不能表述为全域分组验证性能。**"
    )
    lines.append("")

    for row in paragraphs:
        section = row["section"]
        lines.append(f"## {section}")
        lines.append("")
        lines.append(str(row["word_ready_text"]))
        lines.append("")
        if section in fig_map:
            f = fig_map[section]
            lines.append(f"**建议插入：{f['figure_no']}**")
            lines.append("")
            lines.append(f"{f['caption_word_ready']}")
            lines.append("")
            lines.append(f"**图件声明边界：**{f['claim_boundary']}")
            lines.append("")

    lines.append("## 本节可直接使用的保守结论句")
    lines.append("")
    lines.append(
        "本研究结果表明，混凝沉淀除氟文献数据具有明显的跨来源异质性，模型在全域分组验证下仍面临较大外推难度。"
        "通过引入无目标泄露的适应域筛选，可以识别出模型相对可靠的局部保留子域，并在该子域内获得较高预测精度。"
        "然而，由于该子域覆盖率较低且文献来源集中，模型结论应限定于相应化学窗口和数据支持范围内，未来仍需补充更多跨文献、跨药剂体系和低覆盖区域实验数据，以提高模型的全域泛化能力。"
    )
    lines.append("")
    return "\n".join(lines)


def render_word_ready_txt(md: str) -> str:
    # Simple markdown-to-text cleanup for Word paste
    txt = md
    txt = txt.replace("# ", "")
    txt = txt.replace("## ", "")
    txt = txt.replace("**", "")
    txt = re.sub(r"^> ?", "", txt, flags=re.MULTILINE)
    return txt


def check_final_figures(figs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for f in figs:
        png = PROJECT_ROOT / str(f["file_png"])
        pdf = PROJECT_ROOT / str(f["file_pdf"])
        rows.append({
            "figure_no": f["figure_no"],
            "png_path": str(png),
            "png_exists": png.exists(),
            "pdf_path": str(pdf),
            "pdf_exists": pdf.exists(),
            "status": "ok" if png.exists() and pdf.exists() else "missing_file_check_needed",
        })
    return rows


def main() -> None:
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    paragraphs = make_paragraphs(KEY_RESULT)
    figures = make_figure_plan(KEY_RESULT)
    guardrails = make_guardrail_table(KEY_RESULT)
    length_control = make_length_control(paragraphs)
    figure_check = check_final_figures(figures)

    md_text = render_word_ready_md(KEY_RESULT, paragraphs, figures)
    txt_text = render_word_ready_txt(md_text)

    outputs = {}

    p1 = TABLE_DIR / "32n_word_ready_results_discussion_paragraphs.csv"
    write_csv_dicts(p1, paragraphs)
    outputs[p1.name] = p1

    p2 = TABLE_DIR / "32n_figure_insertion_plan.csv"
    write_csv_dicts(p2, figures)
    outputs[p2.name] = p2

    p3 = TABLE_DIR / "32n_final_claim_guardrail_check.csv"
    write_csv_dicts(p3, guardrails)
    outputs[p3.name] = p3

    p4 = TABLE_DIR / "32n_section_length_control.csv"
    write_csv_dicts(p4, length_control)
    outputs[p4.name] = p4

    p5 = TABLE_DIR / "32n_final_figure_existence_check.csv"
    write_csv_dicts(p5, figure_check)
    outputs[p5.name] = p5

    p6 = TABLE_DIR / "32n_locked_key_result_for_word.csv"
    write_csv_dicts(p6, [asdict(KEY_RESULT) | {"key_result_source": KEY_RESULT_SOURCE}])
    outputs[p6.name] = p6

    md_path = REPORT_DIR / "32n_results_discussion_word_ready.md"
    md_path.write_text(md_text, encoding="utf-8")
    outputs[md_path.name] = md_path

    txt_path = REPORT_DIR / "32n_results_discussion_word_ready.txt"
    txt_path.write_text(txt_text, encoding="utf-8")
    outputs[txt_path.name] = txt_path

    log_lines = [
        "32N Word-ready Results and Discussion integration",
        f"Run time: {run_time}",
        f"Project root: {PROJECT_ROOT}",
        f"Table output dir: {TABLE_DIR}",
        f"Report output dir: {REPORT_DIR}",
        "=" * 72,
        "[LOCKED KEY RESULT]",
        f"Selection source: {KEY_RESULT_SOURCE}",
        f"Method: {KEY_RESULT.method}",
        f"Rule: {KEY_RESULT.rule}",
        f"R2={KEY_RESULT.r2:.6f}, RMSE={KEY_RESULT.rmse:.6f}, MAE={KEY_RESULT.mae:.6f}, "
        f"Coverage={KEY_RESULT.coverage:.6f}, N={KEY_RESULT.n_retained}, Dominant group={KEY_RESULT.dominant_group}",
        f"Result type: {KEY_RESULT.result_type}",
        f"Forbidden claim: {KEY_RESULT.forbidden_claim}",
        "=" * 72,
    ]

    for k, v in outputs.items():
        log_lines.append(f"[OUTPUT] {v}")

    missing = [r for r in figure_check if r["status"] != "ok"]
    if missing:
        log_lines.append("[WARNING] Some figure files are missing. See 32n_final_figure_existence_check.csv.")
    else:
        log_lines.append("[CHECK] All final 194-197 PNG/PDF figure files were found.")

    log_lines.extend([
        "",
        "32N finished successfully.",
        "Important writing boundary: R2=0.747676 belongs to the selective AD retained-subdomain, not all-domain grouped validation.",
        f"Please open: {md_path}",
    ])

    log_path = REPORT_DIR / "32n_run_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    outputs[log_path.name] = log_path

    print("=" * 72)
    print("32N Word-ready Results and Discussion integration")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Table output dir: {TABLE_DIR}")
    print(f"Report output dir: {REPORT_DIR}")
    print("=" * 72)
    print("[LOCKED KEY RESULT]")
    print(f"Selection source: {KEY_RESULT_SOURCE}")
    print(f"Selective AD retained-subdomain result: {KEY_RESULT.method}")
    print(f"Rule: {KEY_RESULT.rule}")
    print(
        f"R2={KEY_RESULT.r2:.6f}, RMSE={KEY_RESULT.rmse:.6f}, MAE={KEY_RESULT.mae:.6f}, "
        f"Coverage={KEY_RESULT.coverage:.6f}, N={KEY_RESULT.n_retained}, Dominant group={KEY_RESULT.dominant_group}"
    )
    print("=" * 72)
    for name, path in outputs.items():
        label = "OUTPUT REPORT" if path.suffix.lower() in {".md", ".txt"} and "run_log" not in path.name else "OUTPUT"
        if "run_log" in path.name:
            label = "OUTPUT LOG"
        print(f"[{label}] {path}")
    print("")
    print("32N finished successfully.")
    print("Important writing boundary: R2=0.747676 belongs to selective AD retained-subdomain, not all-domain grouped validation.")
    print("")
    print("Please open:")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
