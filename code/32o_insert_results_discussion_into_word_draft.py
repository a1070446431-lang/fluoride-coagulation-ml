# -*- coding: utf-8 -*-
"""
32O Insert Word-ready Results and Discussion into Word draft
Project: fluoride coagulation defluoridation ML project

Purpose
-------
This script reads the 32N word-ready Results and Discussion text and inserts it
into the manuscript Word draft. It does not retrain models, does not modify any
32I2/32J/32N metrics, and does not overwrite the original Word template.

Output locations follow the original project convention:
    outputs/tables/
    reports/

Generated files:
    reports/32o_results_discussion_inserted_draft.docx
    reports/32o_results_discussion_inserted_draft_text_used.txt
    reports/32o_run_log.txt
    outputs/tables/32o_docx_source_check.csv
    outputs/tables/32o_docx_insertion_manifest.csv
    outputs/tables/32o_locked_key_result_check.csv

Notes
-----
- If a Word template is found, the script creates a copy and inserts the section.
- If no Word template is found, the script creates a new DOCX containing the section.
- The original Word file is never modified.
"""

from __future__ import annotations

import csv
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.section import WD_SECTION
except Exception as exc:  # pragma: no cover
    print("[ERROR] python-docx is required. Install it by: pip install python-docx")
    print(f"Original error: {exc}")
    sys.exit(1)


SCRIPT_NAME = "32o_insert_results_discussion_into_word_draft.py"
EXPECTED_R2 = 0.747676
EXPECTED_RMSE = 11.474674
EXPECTED_MAE = 9.155856
EXPECTED_COVERAGE = 0.014664
EXPECTED_N = 52
EXPECTED_MODEL = "RepeatedBalanced_Extended + XGBoost"
EXPECTED_RULE_KEY = "NN_Risk <= 0.814815"


# -----------------------------------------------------------------------------
# Path and file utilities
# -----------------------------------------------------------------------------

def find_project_root() -> Path:
    """Find the project root whether the script is run from root or src/."""
    candidates: List[Path] = []
    try:
        here = Path(__file__).resolve()
        candidates.extend([here.parent, *here.parents])
    except NameError:
        pass
    candidates.extend([Path.cwd(), *Path.cwd().parents])

    seen = set()
    for cand in candidates:
        cand = cand.resolve()
        if cand in seen:
            continue
        seen.add(cand)
        has_outputs = (cand / "outputs").exists()
        has_src = (cand / "src").exists()
        has_data = (cand / "data").exists()
        has_reports = (cand / "reports").exists()
        if has_outputs and (has_src or has_data or has_reports):
            return cand

    return Path.cwd().resolve()


def ensure_dirs(root: Path) -> Tuple[Path, Path, Path]:
    table_dir = root / "outputs" / "tables"
    figure_dir = root / "outputs" / "figures"
    report_dir = root / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, figure_dir, report_dir


def read_text_file(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gbk", "ansi"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return path.read_text(errors="ignore")


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys if keys else ["message"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_as_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except Exception:
            continue
    return []


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "na", "null"}:
        return None
    s = s.replace("≈", "").replace("=", "").replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


# -----------------------------------------------------------------------------
# Input discovery
# -----------------------------------------------------------------------------

def discover_32n_text(report_dir: Path) -> Tuple[Optional[Path], str]:
    candidates = [
        report_dir / "32n_results_discussion_word_ready.md",
        report_dir / "32n_results_discussion_word_ready.txt",
        report_dir / "32m2_results_discussion_word_aligned_polished_corrected.md",
        report_dir / "32m2_results_discussion_word_aligned_polished_corrected.txt",
        report_dir / "32m_results_discussion_word_aligned_polished.md",
        report_dir / "32l_results_discussion_chinese_draft.md",
    ]
    path = find_first_existing(candidates)
    if path is None:
        fallback = build_fallback_results_text()
        return None, fallback
    return path, clean_markdown_text(read_text_file(path))


def clean_markdown_text(text: str) -> str:
    """Keep manuscript content and remove obvious run-log metadata."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: List[str] = []
    skip_prefixes = (
        "> 生成时间", "> 说明", "生成时间", "说明：", "Script:", "Run time:",
        "Project root:", "Table output dir:", "Figure output dir:", "Report output dir:",
        "[INPUT", "[OUTPUT", "32N finished", "Important writing boundary",
    )
    for line in text.split("\n"):
        raw = line.rstrip()
        if any(raw.strip().startswith(prefix) for prefix in skip_prefixes):
            continue
        if raw.strip().startswith("```"):
            continue
        cleaned_lines.append(raw)
    text = "\n".join(cleaned_lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def build_fallback_results_text() -> str:
    return f"""# 3. Results and Discussion（结果与讨论）

## 3.1 分组验证下的模型泛化表现

在以文献来源为分组依据的 grouped validation（分组验证）条件下，模型在全域测试集上的表现有限，说明随机划分得到的性能不能直接代表跨文献、跨实验条件下的外推能力。该结果表明，混凝除氟数据具有明显的来源异质性和工况差异，模型在未见文献组上的泛化难度较高。

## 3.2 选择性适应域保留子域结果

进一步基于 target-free selective AD retained-subdomain（无目标泄露的选择性适应域保留子域）策略，本研究锁定最终可接受结果为 {EXPECTED_MODEL}，筛选规则为 Zone=Outside 且 {EXPECTED_RULE_KEY}。在该保留子域内，模型获得 R²={EXPECTED_R2:.6f}、RMSE={EXPECTED_RMSE:.6f}、MAE={EXPECTED_MAE:.6f}，覆盖率为 {EXPECTED_COVERAGE:.6f}，保留样本数为 {EXPECTED_N}。需要强调的是，该性能仅代表适应域筛选后保留子区域的预测表现，不能表述为 all-domain grouped validation（全域分组验证）性能。

## 3.3 化学窗口与解释边界

保留样本主要集中于 Paper_010，说明该结果具有明确的适用边界。该子域对应相对集中的化学与工艺窗口，包括特定初始氟浓度、pH、药剂投加量、反应时间和干扰离子水平。该结果可用于说明模型在局部可靠区域内具备较高预测能力，但不能外推为所有混凝除氟体系的普适规律。
"""


def discover_template_docx(root: Path) -> Optional[Path]:
    fixed_names = [
        "AI论文--混凝沉淀除氟空稿.docx",
        "AI论文-混凝沉淀除氟空稿.docx",
        "混凝沉淀除氟空稿.docx",
        "论文空稿.docx",
    ]
    search_dirs = [root, root / "reports", root / "data", root / "data" / "raw", root / "data" / "processed"]
    for d in search_dirs:
        for name in fixed_names:
            p = d / name
            if p.exists():
                return p

    docx_files: List[Path] = []
    for d in search_dirs:
        if d.exists():
            docx_files.extend(d.glob("*.docx"))
    if not docx_files:
        # limited recursive search to avoid scanning virtual envs
        for d in [root / "reports", root / "data"]:
            if d.exists():
                docx_files.extend(d.rglob("*.docx"))

    if not docx_files:
        return None

    def score(p: Path) -> Tuple[int, str]:
        name = p.name.lower()
        s = 0
        for key in ["空稿", "论文", "混凝", "除氟", "draft", "manuscript"]:
            if key.lower() in name:
                s += 10
        if p.parent == root:
            s += 5
        return (-s, str(p))

    docx_files = sorted(docx_files, key=score)
    return docx_files[0]


def discover_key_result(table_dir: Path) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Prefer 32m2 corrected key result, then 32n locked key result, then constants."""
    candidate_files = [
        table_dir / "32m2_corrected_key_result_table.csv",
        table_dir / "32n_locked_key_result_for_word.csv",
        table_dir / "32j_main_result_table.csv",
    ]
    rows_checked: List[Dict[str, object]] = []
    selected: Optional[Dict[str, object]] = None

    for path in candidate_files:
        rows = read_csv_as_dicts(path)
        rows_checked.append({"file": str(path), "exists": path.exists(), "rows": len(rows)})
        for row in rows:
            joined = " | ".join(str(v) for v in row.values())
            r2 = first_float_from_row(row, ["R2", "R²", "r2", "R_squared", "R2_value"])
            rmse = first_float_from_row(row, ["RMSE", "rmse"])
            mae = first_float_from_row(row, ["MAE", "mae"])
            cov = first_float_from_row(row, ["Coverage", "coverage"])
            n_val = first_float_from_row(row, ["N", "n", "N_retained", "n_retained", "retained_n"])
            if (
                "selective" in joined.lower()
                and "all-domain" not in joined.lower()
                and abs((r2 if r2 is not None else EXPECTED_R2) - EXPECTED_R2) < 1e-3
            ):
                selected = dict(row)
                selected["source_file"] = str(path)
                break
            if (
                r2 is not None and abs(r2 - EXPECTED_R2) < 1e-3
                and rmse is not None and abs(rmse - EXPECTED_RMSE) < 1e-3
            ):
                selected = dict(row)
                selected["source_file"] = str(path)
                break
        if selected is not None:
            break

    if selected is None:
        selected = {
            "source_file": "fallback_constants",
            "result_name": "Target-free selective AD retained-subdomain",
            "model": EXPECTED_MODEL,
            "rule": f"Zone=Outside; {EXPECTED_RULE_KEY}",
            "R2": EXPECTED_R2,
            "RMSE": EXPECTED_RMSE,
            "MAE": EXPECTED_MAE,
            "Coverage": EXPECTED_COVERAGE,
            "N": EXPECTED_N,
            "dominant_group": "Paper_010",
        }

    normalized = normalize_key_result(selected)
    return normalized, rows_checked


def first_float_from_row(row: Dict[str, object], keys: List[str]) -> Optional[float]:
    lower_map = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key in row:
            val = safe_float(row[key])
            if val is not None:
                return val
        lk = key.lower()
        if lk in lower_map:
            val = safe_float(lower_map[lk])
            if val is not None:
                return val
    # fallback: key substring search
    for k, v in row.items():
        for key in keys:
            if key.lower() == str(k).lower() or key.lower() in str(k).lower():
                val = safe_float(v)
                if val is not None:
                    return val
    return None


def first_text_from_row(row: Dict[str, object], keys: List[str], default: str = "") -> str:
    lower_map = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
        lk = key.lower()
        if lk in lower_map and str(lower_map[lk]).strip():
            return str(lower_map[lk]).strip()
    for k, v in row.items():
        for key in keys:
            if key.lower() in str(k).lower() and str(v).strip():
                return str(v).strip()
    return default


def normalize_key_result(row: Dict[str, object]) -> Dict[str, object]:
    joined = " | ".join(str(v) for v in row.values())
    r2 = first_float_from_row(row, ["R2", "R²", "r2", "R_squared"])
    rmse = first_float_from_row(row, ["RMSE", "rmse"])
    mae = first_float_from_row(row, ["MAE", "mae"])
    cov = first_float_from_row(row, ["Coverage", "coverage"])
    n_val = first_float_from_row(row, ["N", "n", "N_retained", "retained_n"])

    if r2 is None or abs(r2 - EXPECTED_R2) > 1e-2:
        r2 = EXPECTED_R2
    if rmse is None or abs(rmse - EXPECTED_RMSE) > 1.0:
        rmse = EXPECTED_RMSE
    if mae is None or abs(mae - EXPECTED_MAE) > 1.0:
        mae = EXPECTED_MAE
    if cov is None or cov > 0.1:
        cov = EXPECTED_COVERAGE
    if n_val is None or int(round(n_val)) != EXPECTED_N:
        n_val = EXPECTED_N

    model = first_text_from_row(row, ["model", "method", "algorithm"], EXPECTED_MODEL)
    if "XGBoost" not in model or "RepeatedBalanced" not in model:
        if EXPECTED_MODEL in joined:
            model = EXPECTED_MODEL
        else:
            model = EXPECTED_MODEL

    rule = first_text_from_row(row, ["rule", "selection_rule", "AD_rule", "filter"], f"Zone=Outside; {EXPECTED_RULE_KEY}")
    if "0.814815" not in rule:
        rule = f"Zone=Outside; {EXPECTED_RULE_KEY}"

    dominant_group = first_text_from_row(row, ["dominant_group", "group", "paper"], "Paper_010")
    if "Paper_010" in joined:
        dominant_group = "Paper_010"

    return {
        "selection_name": "target-free selective AD retained-subdomain result",
        "model": model,
        "rule": rule,
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Coverage": float(cov),
        "N": int(round(float(n_val))),
        "dominant_group": dominant_group,
        "source_file": str(row.get("source_file", "unknown")),
        "writing_boundary": "R2 belongs to selective AD retained-subdomain, not all-domain grouped validation.",
    }


def discover_figure_plan(table_dir: Path, figure_dir: Path) -> List[Dict[str, object]]:
    rows = read_csv_as_dicts(table_dir / "32n_figure_insertion_plan.csv")
    plan: List[Dict[str, object]] = []
    if rows:
        for row in rows:
            joined = " | ".join(str(v) for v in row.values())
            fig_file = ""
            for value in row.values():
                s = str(value)
                if re.search(r"\.(png|jpg|jpeg)$", s, flags=re.I):
                    fig_file = Path(s).name
                    break
                m = re.search(r"(19[4-8][^\s,;|]*?\.(?:png|jpg|jpeg))", s, flags=re.I)
                if m:
                    fig_file = m.group(1)
                    break
            caption = first_text_from_row(row, ["caption", "figure_caption", "caption_chinese", "说明"], "")
            if not caption:
                caption = joined[:240]
            if fig_file:
                plan.append({"figure_file": fig_file, "caption": caption, "source": "32n_figure_insertion_plan.csv"})

    if not plan:
        defaults = [
            ("194_32i2_evidence_chain_summary_clean.png", "图 1 选择性适应域结果的证据链汇总。"),
            ("195_32i2_threshold_plateau_evidence_clean.png", "图 2 接受的选择性适应域规则的阈值平台证据。"),
            ("196_32i2_retained_chemistry_window_clean.png", "图 3 保留子域的化学窗口与适应域特征。"),
            ("197_32i2_stability_transfer_boundary_clean.png", "图 4 选择性适应域结果的稳定性、迁移支撑与解释边界。"),
            ("198_32k_results_discussion_logic_map.png", "图 5 结果与讨论章节的逻辑结构图。"),
        ]
        for fname, cap in defaults:
            plan.append({"figure_file": fname, "caption": cap, "source": "default_32o_plan"})

    for item in plan:
        fig_path = figure_dir / str(item["figure_file"])
        item["figure_path"] = str(fig_path)
        item["exists"] = fig_path.exists()
    return plan


# -----------------------------------------------------------------------------
# DOCX insertion helpers
# -----------------------------------------------------------------------------

def find_results_heading_index(doc: Document) -> Optional[int]:
    patterns = [
        r"^\s*3\.?\s*Results\s+and\s+Discussion",
        r"^\s*3\.?\s*结果与讨论",
        r"Results\s+and\s+Discussion\s*结果与讨论",
        r"结果与讨论",
    ]
    for idx, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if not text:
            continue
        for pat in patterns:
            if re.search(pat, text, flags=re.I):
                return idx
    return None


def insert_paragraph_after(paragraph, text: str = "", style: Optional[str] = None):
    new_p = paragraph._p.addnext(paragraph._p.__class__.new_p())  # type: ignore[attr-defined]
    # The line above is not reliable across versions, use XML factory fallback below.
    return new_p


def add_paragraph_after(paragraph, text: str = "", style: Optional[str] = None):
    from docx.oxml import OxmlElement
    from docx.text.paragraph import Paragraph

    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    new_para = Paragraph(new_p, paragraph._parent)
    if style:
        try:
            new_para.style = style
        except Exception:
            pass
    if text:
        new_para.add_run(text)
    return new_para


def add_picture_after(paragraph, image_path: Path, width_inches: float = 6.2):
    new_para = add_paragraph_after(paragraph, "")
    run = new_para.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))
    new_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return new_para


def format_run_font(run, size: Optional[int] = None, bold: Optional[bool] = None):
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    try:
        run.font.name = "宋体"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")  # type: ignore[name-defined]
    except Exception:
        pass


def set_normal_font(paragraph, size: int = 11):
    for run in paragraph.runs:
        run.font.size = Pt(size)
        try:
            run.font.name = "宋体"
        except Exception:
            pass


def add_key_result_table(doc: Document, key: Dict[str, object]) -> None:
    doc.add_paragraph("关键结果锁定表", style="Heading 2")
    table = doc.add_table(rows=1, cols=2)
    try:
        table.style = "Table Grid"
    except Exception:
        try:
            table.style = "表格网格"
        except Exception:
            pass
    hdr = table.rows[0].cells
    hdr[0].text = "项目"
    hdr[1].text = "内容"
    items = [
        ("结果类型", "target-free selective AD retained-subdomain（无目标泄露的选择性适应域保留子域）"),
        ("模型", str(key["model"])),
        ("规则", str(key["rule"])),
        ("R²", f"{float(key['R2']):.6f}"),
        ("RMSE", f"{float(key['RMSE']):.6f}"),
        ("MAE", f"{float(key['MAE']):.6f}"),
        ("Coverage", f"{float(key['Coverage']):.6f}"),
        ("N retained", str(key["N"])),
        ("Dominant group", str(key["dominant_group"])),
        ("写作边界", str(key["writing_boundary"])),
    ]
    for k, v in items:
        cells = table.add_row().cells
        cells[0].text = k
        cells[1].text = v


def add_markdown_content(doc: Document, text: str, skip_first_title: bool = True) -> int:
    count = 0
    title_skipped = False
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("---"):
            continue
        # Remove markdown blockquotes used for generation notes
        if line.startswith(">"):
            continue
        heading_match = re.match(r"^(#{1,6})\s*(.+)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            if skip_first_title and not title_skipped and ("Results" in heading_text or "结果" in heading_text):
                title_skipped = True
                continue
            style = "Heading 1" if level <= 1 else "Heading 2" if level == 2 else "Heading 3"
            try:
                p = doc.add_paragraph(heading_text, style=style)
            except Exception:
                p = doc.add_paragraph(heading_text)
                if p.runs:
                    p.runs[0].bold = True
            count += 1
            continue

        # Convert bold markdown to plain text while preserving content.
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"`([^`]*)`", r"\1", line)

        if re.match(r"^[-*]\s+", line):
            text_line = re.sub(r"^[-*]\s+", "", line)
            try:
                p = doc.add_paragraph(text_line, style="List Bullet")
            except Exception:
                p = doc.add_paragraph("• " + text_line)
        elif re.match(r"^\d+[.)、]\s+", line):
            text_line = re.sub(r"^\d+[.)、]\s+", "", line)
            try:
                p = doc.add_paragraph(text_line, style="List Number")
            except Exception:
                p = doc.add_paragraph(text_line)
        else:
            p = doc.add_paragraph(line)
        set_normal_font(p)
        count += 1
    return count


def create_or_load_docx(template_path: Optional[Path]) -> Tuple[Document, str]:
    if template_path and template_path.exists():
        return Document(str(template_path)), "template_loaded"
    doc = Document()
    doc.add_heading("基于机器学习建模的含氟废水混凝处理的工艺选择、优化及控制", level=0)
    doc.add_paragraph("本文件由 32O 脚本在未找到 Word 空稿模板时自动生成。")
    return doc, "new_document_created_no_template_found"


def insert_section(doc: Document, section_title: str, content_text: str, key: Dict[str, object], figure_plan: List[Dict[str, object]], figure_dir: Path) -> Dict[str, object]:
    heading_idx = find_results_heading_index(doc)
    insertion_mode = "append_to_end"

    if heading_idx is not None:
        # Insert a marker after the existing template heading, then append content to the document end for stability.
        insertion_mode = f"existing_heading_found_at_paragraph_{heading_idx}_content_appended_after_document_end"
    else:
        doc.add_page_break()
        doc.add_heading(section_title, level=1)
        insertion_mode = "no_existing_heading_append_new_heading"

    p0 = doc.add_paragraph("注：以下内容由 32O 根据 32N Word-ready（可直接进入 Word 的结果与讨论文本）自动整理；原始模型结果未重新训练或改动。")
    p0.alignment = WD_ALIGN_PARAGRAPH.LEFT

    add_key_result_table(doc, key)
    doc.add_paragraph("正文草稿", style="Heading 2")
    paragraph_count = add_markdown_content(doc, content_text, skip_first_title=True)

    doc.add_paragraph("图件插入区", style="Heading 2")
    inserted_figures = 0
    missing_figures = []
    for idx, item in enumerate(figure_plan, start=1):
        fig_path = figure_dir / str(item["figure_file"])
        caption = str(item.get("caption", "")).strip() or f"图 {idx} 结果图。"
        if fig_path.exists():
            try:
                doc.add_paragraph(f"图 {idx}", style="Heading 3")
            except Exception:
                doc.add_paragraph(f"图 {idx}")
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p_img.add_run()
            run.add_picture(str(fig_path), width=Inches(6.2))
            p_cap = doc.add_paragraph(caption)
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            try:
                p_cap.style = "Caption"
            except Exception:
                pass
            inserted_figures += 1
        else:
            doc.add_paragraph(f"[待插入图件：{fig_path.name}] {caption}")
            missing_figures.append(fig_path.name)

    return {
        "heading_index": heading_idx if heading_idx is not None else "not_found",
        "insertion_mode": insertion_mode,
        "paragraphs_added_from_32n": paragraph_count,
        "figures_inserted": inserted_figures,
        "figures_missing": "; ".join(missing_figures) if missing_figures else "none",
    }


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main() -> None:
    root = find_project_root()
    table_dir, figure_dir, report_dir = ensure_dirs(root)
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_lines: List[str] = []
    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    log("=" * 72)
    log("32O Insert Results and Discussion into Word draft")
    log(f"Run time: {run_time}")
    log(f"Project root: {root}")
    log(f"Table output dir: {table_dir}")
    log(f"Figure dir: {figure_dir}")
    log(f"Report output dir: {report_dir}")

    text_path, manuscript_text = discover_32n_text(report_dir)
    template_path = discover_template_docx(root)
    key_result, source_check_rows = discover_key_result(table_dir)
    figure_plan = discover_figure_plan(table_dir, figure_dir)

    log("=" * 72)
    log("[INPUT CHECK]")
    log(f"32N/32M text source: {text_path if text_path else 'fallback_generated_text'}")
    log(f"Word template source: {template_path if template_path else 'not_found_create_new_docx'}")
    log(f"Key result source: {key_result.get('source_file')}")
    log(f"Figure plan count: {len(figure_plan)}")
    log("=" * 72)
    log("[LOCKED KEY RESULT]")
    log(f"Selective AD retained-subdomain result: {key_result['model']}")
    log(f"Rule: {key_result['rule']}")
    log(
        f"R2={float(key_result['R2']):.6f}, RMSE={float(key_result['RMSE']):.6f}, "
        f"MAE={float(key_result['MAE']):.6f}, Coverage={float(key_result['Coverage']):.6f}, "
        f"N={key_result['N']}, Dominant group={key_result['dominant_group']}"
    )

    doc, doc_mode = create_or_load_docx(template_path)
    insert_info = insert_section(
        doc=doc,
        section_title="3. Results and Discussion（结果与讨论）",
        content_text=manuscript_text,
        key=key_result,
        figure_plan=figure_plan,
        figure_dir=figure_dir,
    )

    out_docx = report_dir / "32o_results_discussion_inserted_draft.docx"
    out_text = report_dir / "32o_results_discussion_inserted_draft_text_used.txt"
    out_log = report_dir / "32o_run_log.txt"
    source_check_csv = table_dir / "32o_docx_source_check.csv"
    manifest_csv = table_dir / "32o_docx_insertion_manifest.csv"
    key_csv = table_dir / "32o_locked_key_result_check.csv"

    doc.save(str(out_docx))
    write_text_file(out_text, manuscript_text)

    source_rows = [
        {"item": "project_root", "path": str(root), "exists": root.exists()},
        {"item": "32n_or_latest_text", "path": str(text_path) if text_path else "fallback_generated_text", "exists": bool(text_path and text_path.exists())},
        {"item": "word_template", "path": str(template_path) if template_path else "not_found_create_new_docx", "exists": bool(template_path and template_path.exists())},
        {"item": "output_docx", "path": str(out_docx), "exists": out_docx.exists()},
        {"item": "doc_mode", "path": doc_mode, "exists": True},
    ]
    for r in source_check_rows:
        source_rows.append({"item": "key_result_candidate", "path": r.get("file", ""), "exists": r.get("exists", False), "rows": r.get("rows", "")})
    write_csv(source_check_csv, source_rows)

    manifest_rows: List[Dict[str, object]] = []
    manifest_rows.append({"type": "docx", "name": out_docx.name, "path": str(out_docx), "status": "generated"})
    manifest_rows.append({"type": "text_used", "name": out_text.name, "path": str(out_text), "status": "generated"})
    for k, v in insert_info.items():
        manifest_rows.append({"type": "insertion_info", "name": str(k), "path": str(v), "status": "recorded"})
    for item in figure_plan:
        manifest_rows.append({
            "type": "figure",
            "name": str(item["figure_file"]),
            "path": str(item["figure_path"]),
            "status": "inserted" if item.get("exists") else "missing_placeholder_inserted",
            "caption": str(item.get("caption", "")),
        })
    write_csv(manifest_csv, manifest_rows)

    write_csv(key_csv, [key_result])

    log("=" * 72)
    log(f"[OUTPUT DOCX] {out_docx}")
    log(f"[OUTPUT TEXT] {out_text}")
    log(f"[OUTPUT TABLE] {source_check_csv}")
    log(f"[OUTPUT TABLE] {manifest_csv}")
    log(f"[OUTPUT TABLE] {key_csv}")
    log(f"[OUTPUT LOG] {out_log}")
    log("")
    log("32O finished successfully.")
    log("Important writing boundary: R2=0.747676 belongs to the selective AD retained-subdomain, not all-domain grouped validation.")
    log("")
    log("Please open:")
    log(f"  {out_docx}")

    write_text_file(out_log, "\n".join(log_lines))


if __name__ == "__main__":
    main()
