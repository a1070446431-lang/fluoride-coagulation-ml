# -*- coding: utf-8 -*-
"""
32R Final Results-and-Discussion package and handoff check
Project: fluoride coagulation defluoridation ML project

Purpose
-------
This script packages and audits the final Results and Discussion outputs after 32Q2.
It does NOT retrain models, does NOT change any 32I2 metrics, and does NOT create a new
output root folder.

Output locations follow the existing project convention:
    outputs/tables/
    reports/

Main outputs
------------
outputs/tables/32r_final_results_discussion_package_manifest.csv
outputs/tables/32r_word_section_final_checklist.csv
outputs/tables/32r_figure_table_final_index.csv
outputs/tables/32r_submission_risk_guardrail.csv
reports/32r_results_discussion_final_for_word.docx
reports/32r_final_results_discussion_handoff.md
reports/32r_final_results_discussion_handoff.txt
reports/32r_run_log.txt
"""

from __future__ import annotations

import csv
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from xml.etree import ElementTree as ET


# -----------------------------
# General utilities
# -----------------------------

def find_project_root() -> Path:
    """Find project root from script path or current working directory."""
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    candidates.append(Path.cwd().resolve())

    for start in candidates:
        cur = start
        for _ in range(8):
            if (cur / "outputs").exists() and ((cur / "src").exists() or (cur / "reports").exists() or (cur / "data").exists()):
                return cur
            if cur.name.lower() == "src" and (cur.parent / "outputs").exists():
                return cur.parent
            if cur.parent == cur:
                break
            cur = cur.parent
    return Path.cwd().resolve()


def ensure_dirs(root: Path) -> Tuple[Path, Path, Path]:
    table_dir = root / "outputs" / "tables"
    figure_dir = root / "outputs" / "figures"
    report_dir = root / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, figure_dir, report_dir


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
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


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def strip_xml_text(xml_text: str) -> str:
    """Extract Word visible text from document.xml-like content."""
    try:
        root = ET.fromstring(xml_text.encode("utf-8"))
        texts = []
        for elem in root.iter():
            if elem.tag.endswith("}t") and elem.text:
                texts.append(elem.text)
        return "\n".join(texts)
    except Exception:
        return re.sub(r"<[^>]+>", " ", xml_text)


def extract_docx_text_and_media(docx_path: Path) -> Tuple[str, int, List[str]]:
    if not docx_path.exists():
        return "", 0, []
    text = ""
    media_files = []
    try:
        with zipfile.ZipFile(docx_path, "r") as z:
            names = z.namelist()
            if "word/document.xml" in names:
                xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
                text = strip_xml_text(xml)
            media_files = [n for n in names if n.startswith("word/media/")]
    except Exception:
        return "", 0, []
    return text, len(media_files), media_files


def safe_contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


# -----------------------------
# Key result extraction
# -----------------------------

def extract_locked_key_result(table_dir: Path) -> Dict[str, object]:
    """Prefer 32m2 corrected key result; fallback to locked constants."""
    candidates = [
        table_dir / "32m2_corrected_key_result_table.csv",
        table_dir / "32n_locked_key_result_for_word.csv",
        table_dir / "32q2_locked_key_result_check.csv",
        table_dir / "32j_main_result_table.csv",
    ]

    fallback = {
        "result_label": "Target-free selective AD retained subdomain",
        "model_family": "RepeatedBalanced_Extended + XGBoost",
        "rule": "Zone=Outside; NN_Risk <= 0.814815",
        "r2": "0.747676",
        "rmse": "11.474674",
        "mae": "9.155856",
        "coverage": "0.014664",
        "n_retained": "52",
        "dominant_group": "Paper_010",
        "source": "fallback_locked_constants",
    }

    for path in candidates:
        rows = read_csv_rows(path)
        if not rows:
            continue
        # Select row that looks like the accepted selective AD retained-subdomain result.
        for row in rows:
            combined = " | ".join(str(v) for v in row.values())
            if ("RepeatedBalanced_Extended" in combined and "XGBoost" in combined and
                ("0.747676" in combined or "0.74767" in combined or "0.814815" in combined)):
                result = fallback.copy()
                result["source"] = path.name
                # Try to map common possible column names.
                def get_any(names: List[str], default: str) -> str:
                    for n in names:
                        if n in row and str(row[n]).strip() not in ["", "nan", "None"]:
                            return str(row[n]).strip()
                    return default
                result["result_label"] = get_any(["result_label", "claim", "scenario", "result_type", "label"], result["result_label"])
                result["model_family"] = get_any(["model_family", "model", "accepted_model", "model_name", "pipeline"], result["model_family"])
                if result["model_family"] == "XGBoost":
                    result["model_family"] = "RepeatedBalanced_Extended + XGBoost"
                result["rule"] = get_any(["rule", "ad_rule", "selection_rule", "policy_rule"], result["rule"])
                result["r2"] = get_any(["R2", "r2", "R^2", "R²"], result["r2"])
                result["rmse"] = get_any(["RMSE", "rmse"], result["rmse"])
                result["mae"] = get_any(["MAE", "mae"], result["mae"])
                result["coverage"] = get_any(["coverage", "Coverage"], result["coverage"])
                result["n_retained"] = get_any(["N", "n", "n_retained", "retained_n", "N_retained"], result["n_retained"])
                result["dominant_group"] = get_any(["dominant_group", "Dominant group", "group", "source_group"], result["dominant_group"])
                return result
    return fallback


# -----------------------------
# Main pipeline
# -----------------------------

def main() -> None:
    root = find_project_root()
    table_dir, figure_dir, report_dir = ensure_dirs(root)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_lines: List[str] = []
    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    log("=" * 72)
    log("32R Final Results-and-Discussion package and handoff check")
    log(f"Run time: {now}")
    log(f"Project root: {root}")
    log(f"Table output dir: {table_dir}")
    log(f"Figure dir: {figure_dir}")
    log(f"Report output dir: {report_dir}")
    log("=" * 72)

    key = extract_locked_key_result(table_dir)
    log("[LOCKED KEY RESULT]")
    log(f"Source: {key.get('source')}")
    log(f"Selective AD retained-subdomain result: {key.get('model_family')}")
    log(f"Rule: {key.get('rule')}")
    log(
        f"R2={key.get('r2')}, RMSE={key.get('rmse')}, MAE={key.get('mae')}, "
        f"Coverage={key.get('coverage')}, N={key.get('n_retained')}, Dominant group={key.get('dominant_group')}"
    )

    source_docx = first_existing([
        report_dir / "32q2_results_discussion_final_checked.docx",
        report_dir / "32q_results_discussion_final_checked.docx",
        report_dir / "32p_results_discussion_with_figures.docx",
        report_dir / "32o_results_discussion_inserted_draft.docx",
    ])
    final_docx = report_dir / "32r_results_discussion_final_for_word.docx"
    copied = False
    if source_docx:
        shutil.copy2(source_docx, final_docx)
        copied = True

    docx_text, inline_images, media_files = extract_docx_text_and_media(final_docx)

    figure_specs = [
        ("194", "194_32i2_evidence_chain_summary_clean.png", "Evidence-chain summary for the selective AD retained-subdomain result"),
        ("195", "195_32i2_threshold_plateau_evidence_clean.png", "Threshold plateau evidence for the accepted selective AD rule"),
        ("196", "196_32i2_retained_chemistry_window_clean.png", "Retained-subdomain chemistry and AD characterization"),
        ("197", "197_32i2_stability_transfer_boundary_clean.png", "Stability, transfer support, and interpretation boundary"),
    ]

    figure_index_rows = []
    for fig_no, fname, desc in figure_specs:
        p = figure_dir / fname
        figure_index_rows.append({
            "figure_no": fig_no,
            "file_name": fname,
            "path": str(p),
            "exists": p.exists(),
            "description": desc,
            "word_insertion_status": "expected_inserted_in_32p_32q2_docx" if p.exists() else "missing_source_figure",
        })

    checklist_rows = []
    def add_check(item: str, status: bool, detail: str) -> None:
        checklist_rows.append({"check_item": item, "status": "PASS" if status else "FAIL", "detail": detail})

    add_check("final_docx_created", final_docx.exists(), str(final_docx))
    add_check("source_docx_found", source_docx is not None, str(source_docx) if source_docx else "not found")
    add_check("final_docx_copied_from_latest_checked_version", copied, f"source={source_docx}")
    add_check("contains_accepted_R2", safe_contains(docx_text, "0.747676"), "Expected R2=0.747676")
    add_check("contains_accepted_RMSE", safe_contains(docx_text, "11.474674"), "Expected RMSE=11.474674")
    add_check("contains_accepted_MAE", safe_contains(docx_text, "9.155856"), "Expected MAE=9.155856")
    add_check("contains_accepted_coverage", safe_contains(docx_text, "0.014664"), "Expected Coverage=0.014664")
    add_check("contains_retained_N", safe_contains(docx_text, "52"), "Expected retained N=52")
    add_check("contains_dominant_group", safe_contains(docx_text, "Paper_010"), "Expected Paper_010")
    add_check("contains_full_model_family", safe_contains(docx_text, "RepeatedBalanced_Extended + XGBoost"), "Avoid XGBoost-only shorthand")
    add_check("contains_selection_rule", safe_contains(docx_text, "NN_Risk <= 0.814815") or safe_contains(docx_text, "NN_Risk ≤ 0.814815"), "Expected NN_Risk <= 0.814815")
    add_check("has_at_least_four_inline_images", inline_images >= 4, f"inline/media images={inline_images}; media={'; '.join(media_files[:8])}")
    add_check("all_four_source_figures_exist", all((figure_dir / fname).exists() for _, fname, _ in figure_specs), "194-197 source figure files")

    # Forbid a specific dangerous sentence pattern: R2=0.747676 in same paragraph as all-domain grouped validation.
    paragraphs = [p.strip() for p in docx_text.split("\n") if p.strip()]
    mixed_risk = []
    for p in paragraphs:
        normalized = p.replace("R²", "R2").replace("Ｒ2", "R2")
        if "0.747676" in normalized and ("all-domain" in normalized.lower() or "全域" in normalized or "全域分组" in normalized):
            mixed_risk.append(p[:240])
    add_check("no_mixed_all_domain_claim_with_accepted_R2", len(mixed_risk) == 0, " | ".join(mixed_risk) if mixed_risk else "No mixed paragraph found")

    fail_count = sum(1 for r in checklist_rows if r["status"] == "FAIL")
    pass_count = sum(1 for r in checklist_rows if r["status"] == "PASS")

    guardrail_rows = [
        {
            "claim_type": "allowed_main_claim",
            "wording": "R2=0.747676 is the performance of the target-free selective AD retained-subdomain result.",
            "risk_level": "low",
            "note": "Use together with coverage=0.014664 and N=52.",
        },
        {
            "claim_type": "forbidden_claim",
            "wording": "Do not describe R2=0.747676 as all-domain grouped-validation performance.",
            "risk_level": "high",
            "note": "All-domain grouped validation is only the baseline comparison, not the accepted final result.",
        },
        {
            "claim_type": "required_boundary",
            "wording": "The accepted result is a selective AD retained-subdomain result with limited coverage and Paper_010 dominance.",
            "risk_level": "medium",
            "note": "This boundary must remain visible in the manuscript.",
        },
        {
            "claim_type": "model_naming",
            "wording": "Use RepeatedBalanced_Extended + XGBoost instead of XGBoost alone.",
            "risk_level": "medium",
            "note": "The data-splitting/selection strategy is part of the accepted result.",
        },
    ]

    package_rows = [
        {"artifact_type": "final_docx", "file_name": final_docx.name, "path": str(final_docx), "exists": final_docx.exists(), "role": "final Results and Discussion Word-ready document"},
        {"artifact_type": "source_docx", "file_name": source_docx.name if source_docx else "", "path": str(source_docx) if source_docx else "", "exists": source_docx.exists() if source_docx else False, "role": "latest checked DOCX used as source"},
        {"artifact_type": "key_result", "file_name": str(key.get("source")), "path": str(table_dir / str(key.get("source"))) if key.get("source") else "", "exists": (table_dir / str(key.get("source"))).exists() if key.get("source") else False, "role": "locked selective AD key result source"},
        {"artifact_type": "report_md", "file_name": "32r_final_results_discussion_handoff.md", "path": str(report_dir / "32r_final_results_discussion_handoff.md"), "exists": True, "role": "human-readable handoff report"},
    ]
    for row in figure_index_rows:
        package_rows.append({
            "artifact_type": "source_figure",
            "file_name": row["file_name"],
            "path": row["path"],
            "exists": row["exists"],
            "role": row["description"],
        })

    # Write tables.
    manifest_path = table_dir / "32r_final_results_discussion_package_manifest.csv"
    checklist_path = table_dir / "32r_word_section_final_checklist.csv"
    figure_index_path = table_dir / "32r_figure_table_final_index.csv"
    guardrail_path = table_dir / "32r_submission_risk_guardrail.csv"

    write_csv(manifest_path, package_rows)
    write_csv(checklist_path, checklist_rows)
    write_csv(figure_index_path, figure_index_rows)
    write_csv(guardrail_path, guardrail_rows)

    # Human-readable handoff.
    md = []
    md.append("# 32R Final Results and Discussion handoff\n")
    md.append(f"> Generated at: {now}\n")
    md.append("\n## Locked key result\n")
    md.append(f"- Accepted result: **{key.get('model_family')}**\n")
    md.append(f"- Selection rule: **{key.get('rule')}**\n")
    md.append(f"- R2 = **{key.get('r2')}**\n")
    md.append(f"- RMSE = **{key.get('rmse')}**\n")
    md.append(f"- MAE = **{key.get('mae')}**\n")
    md.append(f"- Coverage = **{key.get('coverage')}**\n")
    md.append(f"- Retained N = **{key.get('n_retained')}**\n")
    md.append(f"- Dominant group = **{key.get('dominant_group')}**\n")
    md.append("\n**Writing boundary:** R2=0.747676 belongs only to the target-free selective AD retained-subdomain result. It must not be described as all-domain grouped-validation performance.\n")
    md.append("\n## Final Word file\n")
    md.append(f"- `{final_docx}`\n")
    md.append("\n## Audit summary\n")
    md.append(f"- PASS: {pass_count}\n")
    md.append(f"- FAIL: {fail_count}\n")
    if fail_count:
        md.append("\n### Failed checks\n")
        for r in checklist_rows:
            if r["status"] == "FAIL":
                md.append(f"- {r['check_item']}: {r['detail']}\n")
    md.append("\n## Figure status\n")
    for r in figure_index_rows:
        md.append(f"- Figure {r['figure_no']}: `{r['file_name']}` | exists={r['exists']} | {r['description']}\n")
    md.append("\n## Next manuscript steps\n")
    md.append("1. Open `reports/32r_results_discussion_final_for_word.docx` and visually inspect the inserted figures and captions.\n")
    md.append("2. If layout is acceptable, copy this section into the full thesis/manuscript file or proceed to full manuscript assembly.\n")
    md.append("3. Keep the selective-AD boundary statement whenever reporting R2=0.747676.\n")

    handoff_md = report_dir / "32r_final_results_discussion_handoff.md"
    handoff_txt = report_dir / "32r_final_results_discussion_handoff.txt"
    handoff_md.write_text("".join(md), encoding="utf-8")
    handoff_txt.write_text("".join(md), encoding="utf-8")

    log("=" * 72)
    log(f"[AUDIT] PASS={pass_count}; FAIL={fail_count}; inline_images={inline_images}")
    for r in checklist_rows:
        log(f"[{r['status']}] {r['check_item']} | {r['detail']}")
    log("=" * 72)
    log(f"[OUTPUT DOCX] {final_docx}")
    log(f"[OUTPUT TABLE] {manifest_path}")
    log(f"[OUTPUT TABLE] {checklist_path}")
    log(f"[OUTPUT TABLE] {figure_index_path}")
    log(f"[OUTPUT TABLE] {guardrail_path}")
    log(f"[OUTPUT REPORT] {handoff_md}")
    log(f"[OUTPUT REPORT] {handoff_txt}")

    log_path = report_dir / "32r_run_log.txt"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"[OUTPUT LOG] {log_path}")

    if fail_count == 0:
        print("\n32R finished successfully.")
    else:
        print("\n32R finished with warnings. Please inspect FAIL rows in 32r_word_section_final_checklist.csv.")
    print("Important writing boundary: R2=0.747676 belongs to the selective AD retained-subdomain only.")
    print("\nPlease open:")
    print(f"  {final_docx}")


if __name__ == "__main__":
    main()
