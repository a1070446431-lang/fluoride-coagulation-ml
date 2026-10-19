# -*- coding: utf-8 -*-
"""
32Q Final DOCX quality audit and light correction
Project: fluoride coagulation defluoridation ML project

Purpose
-------
This script audits the 32P Word document that already contains the Results and
Discussion text plus Figures 194-197. It does NOT retrain models and does NOT
change any 32I2 metrics. It only performs a final consistency check and writes a
reviewed DOCX copy.

Main checks
-----------
1. The locked selective-AD result is present and correctly stated.
2. R2=0.747676 is explicitly guarded as a selective retained-subdomain result,
   not an all-domain grouped-validation result.
3. The four final figures are present in outputs/figures and at least four inline
   images are present in the DOCX.
4. Common incorrect labels from intermediate scripts are repaired, especially
   "Selective AD retained-subdomain result: XGBoost" ->
   "Selective AD retained-subdomain result: RepeatedBalanced_Extended + XGBoost".

Output locations follow the original project convention:
    outputs/tables/
    reports/

Inputs expected
---------------
reports/32p_results_discussion_with_figures.docx
outputs/tables/32m2_corrected_key_result_table.csv     preferred
outputs/tables/32n_locked_key_result_for_word.csv      fallback
outputs/figures/194_32i2_evidence_chain_summary_clean.png
outputs/figures/195_32i2_threshold_plateau_evidence_clean.png
outputs/figures/196_32i2_retained_chemistry_window_clean.png
outputs/figures/197_32i2_stability_transfer_boundary_clean.png
"""

from __future__ import annotations

import csv
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    from docx import Document
except Exception as e:
    raise SystemExit(
        "[ERROR] python-docx is required. Install it by:\n"
        "    .\\.venv\\Scripts\\python.exe -m pip install python-docx\n"
        f"Original error: {e}"
    )


LOCKED = {
    "model_family": "RepeatedBalanced_Extended + XGBoost",
    "rule": "Zone=Outside; NN_Risk <= 0.814815",
    "r2": "0.747676",
    "rmse": "11.474674",
    "mae": "9.155856",
    "coverage": "0.014664",
    "n": "52",
    "dominant_group": "Paper_010",
}

REQUIRED_FIGURES = [
    "194_32i2_evidence_chain_summary_clean.png",
    "195_32i2_threshold_plateau_evidence_clean.png",
    "196_32i2_retained_chemistry_window_clean.png",
    "197_32i2_stability_transfer_boundary_clean.png",
]


def find_project_root() -> Path:
    """Find the project root whether this script is run from root or src/."""
    candidates: List[Path] = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    candidates.append(Path.cwd().resolve())
    expanded = []
    for c in candidates:
        expanded.extend([c, c.parent, c.parent.parent])
    for c in expanded:
        if (c / "outputs" / "tables").exists() and (c / "reports").exists():
            return c
    # create a conventional root at cwd if missing
    root = Path.cwd().resolve()
    (root / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    return root


def ensure_dirs(root: Path) -> Tuple[Path, Path, Path]:
    table_dir = root / "outputs" / "tables"
    fig_dir = root / "outputs" / "figures"
    report_dir = root / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, fig_dir, report_dir


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    # utf-8-sig handles Excel-friendly CSV created by earlier scripts.
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except Exception:
            continue
    return []


def find_key_result_from_tables(table_dir: Path) -> Dict[str, Any]:
    """Read key-result table when available; otherwise use locked constants."""
    candidates = [
        table_dir / "32m2_corrected_key_result_table.csv",
        table_dir / "32n_locked_key_result_for_word.csv",
        table_dir / "32j_main_result_table.csv",
    ]
    selected_source = "locked_constants_fallback"
    row_text = ""
    for p in candidates:
        rows = read_csv_rows(p)
        if not rows:
            continue
        # Prefer a row containing the exact accepted rule/model/R2.
        for row in rows:
            joined = " | ".join(str(v) for v in row.values())
            if (
                "RepeatedBalanced_Extended" in joined
                and "XGBoost" in joined
                and "0.747676" in joined
                and ("0.814815" in joined or "NN_Risk" in joined)
            ):
                selected_source = str(p)
                row_text = joined
                return {**LOCKED, "source": selected_source, "source_row_text": row_text}
        # Otherwise keep first table as diagnostic only, not as trusted key.
        selected_source = str(p)
        row_text = " | ".join(str(v) for v in rows[0].values())
    return {**LOCKED, "source": selected_source, "source_row_text": row_text}


def get_all_docx_text(doc: Document) -> str:
    parts: List[str] = []
    for p in doc.paragraphs:
        parts.append(p.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return "\n".join(parts)


def replace_in_paragraph(paragraph, replacements: Dict[str, str]) -> int:
    """Replace exact strings in a paragraph while preserving simple run structure.

    If a target spans multiple runs, set the full paragraph text in one run.
    This is acceptable for this final textual correction because only a few plain
    key-result labels are targeted.
    """
    original = paragraph.text
    new = original
    for old, repl in replacements.items():
        new = new.replace(old, repl)
    if new == original:
        return 0
    # Clear all runs and rewrite.
    for run in paragraph.runs:
        run.text = ""
    if paragraph.runs:
        paragraph.runs[0].text = new
    else:
        paragraph.add_run(new)
    return 1


def repair_docx_text(doc: Document) -> List[Dict[str, str]]:
    replacements = {
        "Selective AD retained-subdomain result: XGBoost":
            "Selective AD retained-subdomain result: RepeatedBalanced_Extended + XGBoost",
        "Selective AD retained-subdomain result：XGBoost":
            "Selective AD retained-subdomain result：RepeatedBalanced_Extended + XGBoost",
        "选择性适应域保留子域结果：XGBoost":
            "选择性适应域保留子域结果：RepeatedBalanced_Extended + XGBoost",
        "选择性适应域保留子域结果: XGBoost":
            "选择性适应域保留子域结果: RepeatedBalanced_Extended + XGBoost",
        "Rule: None (all test rows retained)":
            "Rule: Zone=Outside; NN_Risk <= 0.814815",
        "规则：None (all test rows retained)":
            "规则：Zone=Outside; NN_Risk <= 0.814815",
    }
    repair_log: List[Dict[str, str]] = []
    changed_count = 0
    for p in doc.paragraphs:
        before = p.text
        changed = replace_in_paragraph(p, replacements)
        if changed:
            changed_count += changed
            repair_log.append({"location": "paragraph", "before": before, "after": p.text})
    for ti, table in enumerate(doc.tables):
        for ri, row in enumerate(table.rows):
            for ci, cell in enumerate(row.cells):
                for p in cell.paragraphs:
                    before = p.text
                    changed = replace_in_paragraph(p, replacements)
                    if changed:
                        changed_count += changed
                        repair_log.append({
                            "location": f"table_{ti}_row_{ri}_cell_{ci}",
                            "before": before,
                            "after": p.text,
                        })
    if not repair_log:
        repair_log.append({"location": "none", "before": "no targeted issue found", "after": "no repair needed"})
    return repair_log


def audit_docx(doc: Document, full_text: str, fig_dir: Path, key: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    q: List[Dict[str, Any]] = []
    f: List[Dict[str, Any]] = []

    def add_check(item: str, status: bool, detail: str):
        q.append({"check_item": item, "status": "PASS" if status else "FAIL", "detail": detail})

    add_check("DOCX contains accepted R2", key["r2"] in full_text, f"Expected R2={key['r2']}")
    add_check("DOCX contains accepted RMSE", key["rmse"] in full_text, f"Expected RMSE={key['rmse']}")
    add_check("DOCX contains accepted MAE", key["mae"] in full_text, f"Expected MAE={key['mae']}")
    add_check("DOCX contains accepted coverage", key["coverage"] in full_text, f"Expected Coverage={key['coverage']}")
    add_check("DOCX contains retained N", re.search(r"\bN\s*=\s*52\b|样本量[^\d]*52|保留[^\d]*52", full_text) is not None,
              "Expected retained N=52")
    add_check("DOCX contains dominant group", key["dominant_group"] in full_text, f"Expected {key['dominant_group']}")
    add_check("DOCX contains accepted model family", key["model_family"] in full_text,
              f"Expected {key['model_family']}")
    add_check("DOCX contains accepted rule", ("0.814815" in full_text and "NN_Risk" in full_text),
              "Expected Zone=Outside; NN_Risk <= 0.814815")

    # Guardrail: no sentence should state that R2=0.747676 is all-domain grouped validation.
    bad_patterns = [
        r"0\.747676[^\n。；;]{0,80}all[- ]domain grouped validation",
        r"all[- ]domain grouped validation[^\n。；;]{0,80}0\.747676",
        r"0\.747676[^\n。；;]{0,80}全域分组验证",
        r"全域分组验证[^\n。；;]{0,80}0\.747676",
    ]
    bad_hits = []
    for pat in bad_patterns:
        bad_hits.extend(re.findall(pat, full_text, flags=re.IGNORECASE))
    add_check("No forbidden all-domain claim for R2=0.747676", len(bad_hits) == 0,
              "Forbidden matches: " + (" | ".join(bad_hits[:3]) if bad_hits else "none"))

    inline_shapes = len(doc.inline_shapes)
    add_check("At least four inline images in DOCX", inline_shapes >= 4, f"inline_shapes={inline_shapes}")

    for idx, name in enumerate(REQUIRED_FIGURES, start=194):
        p = fig_dir / name
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        f.append({
            "figure_no": idx,
            "file_name": name,
            "path": str(p),
            "exists": exists,
            "size_bytes": size,
            "status": "PASS" if exists and size > 0 else "FAIL",
        })
    add_check("All four figure files exist in outputs/figures",
              all(row["status"] == "PASS" for row in f),
              "; ".join(f"{row['figure_no']}={row['status']}" for row in f))
    return q, f


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys or ["message"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    root = find_project_root()
    table_dir, fig_dir, report_dir = ensure_dirs(root)
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    input_docx = report_dir / "32p_results_discussion_with_figures.docx"
    output_docx = report_dir / "32q_results_discussion_final_checked.docx"
    if not input_docx.exists():
        raise SystemExit(f"[ERROR] Input DOCX not found: {input_docx}")

    key = find_key_result_from_tables(table_dir)
    doc = Document(str(input_docx))

    repair_log = repair_docx_text(doc)
    full_text_after_repair = get_all_docx_text(doc)
    quality_rows, figure_rows = audit_docx(doc, full_text_after_repair, fig_dir, key)

    # Always write a reviewed DOCX copy. The original 32P file is not overwritten.
    doc.save(str(output_docx))

    # If save failed for some unexpected reason, keep a byte copy as fallback.
    if not output_docx.exists() or output_docx.stat().st_size == 0:
        shutil.copy2(input_docx, output_docx)

    pass_count = sum(1 for r in quality_rows if r.get("status") == "PASS")
    fail_count = sum(1 for r in quality_rows if r.get("status") == "FAIL")

    key_row = [{
        "model_family": key["model_family"],
        "rule": key["rule"],
        "R2": key["r2"],
        "RMSE": key["rmse"],
        "MAE": key["mae"],
        "Coverage": key["coverage"],
        "N": key["n"],
        "Dominant_group": key["dominant_group"],
        "source": key.get("source", ""),
        "writing_boundary": "R2=0.747676 is a selective AD retained-subdomain result, not all-domain grouped-validation performance.",
    }]

    package_rows = [
        {"item": "input_docx", "path": str(input_docx), "exists": input_docx.exists(), "size_bytes": input_docx.stat().st_size if input_docx.exists() else 0},
        {"item": "final_checked_docx", "path": str(output_docx), "exists": output_docx.exists(), "size_bytes": output_docx.stat().st_size if output_docx.exists() else 0},
        {"item": "quality_audit_csv", "path": str(table_dir / "32q_docx_quality_audit.csv"), "exists": True, "size_bytes": "written"},
        {"item": "figure_audit_csv", "path": str(table_dir / "32q_figure_presence_audit.csv"), "exists": True, "size_bytes": "written"},
        {"item": "run_log", "path": str(report_dir / "32q_run_log.txt"), "exists": True, "size_bytes": "written"},
    ]

    write_csv(table_dir / "32q_docx_quality_audit.csv", quality_rows)
    write_csv(table_dir / "32q_figure_presence_audit.csv", figure_rows)
    write_csv(table_dir / "32q_text_repair_log.csv", repair_log)
    write_csv(table_dir / "32q_locked_key_result_check.csv", key_row)
    write_csv(table_dir / "32q_final_docx_package_manifest.csv", package_rows)

    report_md = report_dir / "32q_final_docx_quality_audit_report.md"
    report_txt = report_dir / "32q_final_docx_quality_audit_report.txt"
    lines = [
        "# 32Q Final DOCX quality audit report",
        "",
        f"Run time: {started}",
        f"Project root: {root}",
        f"Input DOCX: {input_docx}",
        f"Final checked DOCX: {output_docx}",
        "",
        "## Locked key result",
        f"- Model family: {key['model_family']}",
        f"- Rule: {key['rule']}",
        f"- R2 = {key['r2']}",
        f"- RMSE = {key['rmse']}",
        f"- MAE = {key['mae']}",
        f"- Coverage = {key['coverage']}",
        f"- N = {key['n']}",
        f"- Dominant group = {key['dominant_group']}",
        "",
        "## Audit summary",
        f"- PASS checks: {pass_count}",
        f"- FAIL checks: {fail_count}",
        "- Writing boundary: R2=0.747676 belongs to the selective AD retained-subdomain, not all-domain grouped validation.",
        "",
        "## Figure insertion summary",
    ]
    for row in figure_rows:
        lines.append(f"- Figure {row['figure_no']}: {row['status']} | {row['file_name']} | {row['size_bytes']} bytes")
    lines += [
        "",
        "## Repair summary",
    ]
    for row in repair_log[:10]:
        lines.append(f"- {row['location']}: {row['before']} -> {row['after']}")
    lines += [
        "",
        "## Output files",
        f"- {output_docx}",
        f"- {table_dir / '32q_docx_quality_audit.csv'}",
        f"- {table_dir / '32q_figure_presence_audit.csv'}",
        f"- {table_dir / '32q_text_repair_log.csv'}",
        f"- {table_dir / '32q_locked_key_result_check.csv'}",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    report_txt.write_text("\n".join(lines), encoding="utf-8")

    log_lines = [
        "=" * 72,
        "32Q Final DOCX quality audit and light correction",
        f"Run time: {started}",
        f"Project root: {root}",
        f"Table output dir: {table_dir}",
        f"Report output dir: {report_dir}",
        "=" * 72,
        "[LOCKED KEY RESULT]",
        f"Selective AD retained-subdomain result: {key['model_family']}",
        f"Rule: {key['rule']}",
        f"R2={key['r2']}, RMSE={key['rmse']}, MAE={key['mae']}, Coverage={key['coverage']}, N={key['n']}, Dominant group={key['dominant_group']}",
        "=" * 72,
        f"[AUDIT] PASS={pass_count}; FAIL={fail_count}; inline_images={len(doc.inline_shapes)}",
    ]
    for row in quality_rows:
        log_lines.append(f"[{row['status']}] {row['check_item']} | {row['detail']}")
    log_lines += [
        "=" * 72,
        f"[OUTPUT DOCX] {output_docx}",
        f"[OUTPUT TABLE] {table_dir / '32q_docx_quality_audit.csv'}",
        f"[OUTPUT TABLE] {table_dir / '32q_figure_presence_audit.csv'}",
        f"[OUTPUT TABLE] {table_dir / '32q_text_repair_log.csv'}",
        f"[OUTPUT TABLE] {table_dir / '32q_locked_key_result_check.csv'}",
        f"[OUTPUT REPORT] {report_md}",
        f"[OUTPUT LOG] {report_dir / '32q_run_log.txt'}",
        "",
        "32Q finished successfully." if fail_count == 0 else "32Q finished with warnings. Please inspect FAIL rows in 32q_docx_quality_audit.csv.",
        "Important writing boundary: R2=0.747676 belongs to the selective AD retained-subdomain, not all-domain grouped validation.",
        "",
        "Please open:",
        f"    {output_docx}",
    ]
    (report_dir / "32q_run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
