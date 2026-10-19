# -*- coding: utf-8 -*-
"""
31a_unique_category_mapping_audit.py

Purpose
-------
Step 31A for the fluoride coagulation ML project.
This script DOES NOT train any model and DOES NOT change model_ready_v1.csv.
It only audits raw categorical labels and creates review tables for mechanism-aware
category mapping.

Outputs
-------
outputs/tables/31a_coagulant_mapping_audit.csv
outputs/tables/31a_interfering_ion_mapping_audit.csv
outputs/tables/31a_category_mapping_audit_report.txt

How to use
----------
1. Put this file under src/ in the PyCharm project.
2. Run it once.
3. Open the two CSV audit tables.
4. Review/correct the auto-inferred columns before any modeling step.

Important
---------
The auto-mapping is a first-pass audit, not the final scientific mapping.
Final mapping must be checked by the user before Step 31B.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SEEDS = [7, 21, 42, 66, 88]


def find_project_root() -> Path:
    """Find project root from current working directory or script location."""
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    candidates.append(Path.cwd().resolve())

    for start in candidates:
        for p in [start] + list(start.parents):
            if (p / "data" / "processed").exists() and (p / "outputs").exists():
                return p
    # Fallback: parent of src if the script is placed there
    try:
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path.cwd().resolve()


def read_model_data(project_root: Path) -> Tuple[pd.DataFrame, Path]:
    candidates = [
        project_root / "data" / "processed" / "model_ready_v1_with_source.csv",
        project_root / "data" / "processed" / "model_ready_v1.csv",
    ]
    for fp in candidates:
        if fp.exists():
            return pd.read_csv(fp), fp
    raise FileNotFoundError(
        "Cannot find model_ready_v1_with_source.csv or model_ready_v1.csv under data/processed."
    )


def norm_text(x) -> str:
    if pd.isna(x):
        return "缺失"
    s = str(x).strip()
    if s == "":
        return "缺失"
    # normalize common whitespace and separators, but keep scientific symbols
    s = re.sub(r"\s+", "", s)
    s = s.replace("，", ",").replace("；", ";").replace("＋", "+")
    s = s.replace("（", "(").replace("）", ")")
    return s


def contains_any(s: str, keywords: List[str]) -> bool:
    return any(k.lower() in s.lower() for k in keywords)


def coagulant_flags(raw: str) -> Dict[str, object]:
    s = norm_text(raw)
    sl = s.lower()

    # Element / chemistry flags. These are deliberately inclusive for audit purposes.
    contains_al = contains_any(s, ["铝", "al", "明矾", "矾", "pac", "pacl", "patc", "psaf", "fd-psaf", "csaf", "csam"])
    contains_fe = contains_any(s, ["铁", "fe", "pfs", "psf", "psfc", "psfcl", "psaf", "fd-psaf", "csaf", "pfasic"])
    contains_zr = contains_any(s, ["锆", "zr", "zxc", "pzs"])
    contains_ca = contains_any(s, ["钙", "ca", "石灰", "氧化钙", "氢氧化钙", "psfcl"])
    contains_mg = contains_any(s, ["镁", "mg", "csam"])
    contains_si = contains_any(s, ["硅", "si", "硅酸", "psf", "psfc", "psaf", "pfasic", "pzs"])
    contains_ti = contains_any(s, ["钛", "ti", "patc"])
    contains_ree = contains_any(s, ["镧", "la", "稀土", "铈", "ce", "psfcl"])

    is_polymeric = contains_any(s, ["聚合", "聚硅", "poly", "pfs", "pac", "pacl", "psf", "psfc", "psaf", "pfasic", "patc", "pzs"])
    is_composite = contains_any(s, ["复合", "+", "/", "-", "csaf", "csam", "pfasic", "psfc", "psfcl", "zxc", "fd-psaf"])
    is_sulfate = contains_any(s, ["硫酸", "sulfate", "so4"])
    is_chloride = contains_any(s, ["氯化", "chloride", "pacl", "pac", "patc"])
    is_oxide_hydroxide = contains_any(s, ["氧化", "氢氧化", "水合氧化"])
    is_commercial = contains_any(s, ["商业", "神美", "商品", "除氟剂"])
    is_none = s in {"无", "none", "nan", "缺失"}

    # First-pass family. This is intentionally conservative and reviewable.
    if is_none:
        family = "无/缺失"
    elif contains_zr:
        family = "锆基/含锆"
    elif contains_ca and not (contains_al or contains_fe or contains_zr):
        family = "钙基"
    elif contains_al and contains_fe:
        family = "铝铁复合"
    elif contains_al and contains_mg:
        family = "铝镁复合"
    elif contains_fe:
        family = "铁基/含铁"
    elif contains_al:
        family = "铝基/含铝"
    elif contains_mg:
        family = "镁基"
    else:
        family = "其他/需人工确认"

    notes = []
    for flag, name in [
        (contains_al, "Al"), (contains_fe, "Fe"), (contains_zr, "Zr"),
        (contains_ca, "Ca"), (contains_mg, "Mg"), (contains_si, "Si"),
        (contains_ti, "Ti"), (contains_ree, "REE/La"),
        (is_polymeric, "polymeric"), (is_composite, "composite"),
        (is_sulfate, "sulfate"), (is_chloride, "chloride"),
        (is_oxide_hydroxide, "oxide/hydroxide"), (is_commercial, "commercial"),
    ]:
        if flag:
            notes.append(name)

    return {
        "auto_coagulant_family": family,
        "contains_Al": int(contains_al),
        "contains_Fe": int(contains_fe),
        "contains_Zr": int(contains_zr),
        "contains_Ca": int(contains_ca),
        "contains_Mg": int(contains_mg),
        "contains_Si": int(contains_si),
        "contains_Ti": int(contains_ti),
        "contains_REE_or_La": int(contains_ree),
        "is_polymeric": int(is_polymeric),
        "is_composite": int(is_composite),
        "is_sulfate_based": int(is_sulfate),
        "is_chloride_based": int(is_chloride),
        "is_oxide_or_hydroxide": int(is_oxide_hydroxide),
        "is_commercial_product": int(is_commercial),
        "auto_mapping_notes": ";".join(notes),
        "needs_manual_review": int(family == "其他/需人工确认" or is_commercial),
    }


def ion_flags(raw: str) -> Dict[str, object]:
    s = norm_text(raw)
    sl = s.lower()

    is_none = s in {"无", "none", "nan", "缺失", "0"}
    has_carbonate = contains_any(s, ["co₃", "co3", "碳酸", "hco₃", "hco3", "碳酸氢"])
    has_bicarbonate = contains_any(s, ["hco₃", "hco3", "碳酸氢"])
    has_sulfate = contains_any(s, ["so₄", "so4", "硫酸"])
    has_phosphate = contains_any(s, ["po₄", "po4", "磷酸"])
    has_chloride = contains_any(s, ["cl⁻", "cl-", "氯离子", "氯"])
    has_nitrate = contains_any(s, ["no₃", "no3", "硝酸"])
    has_silicate = contains_any(s, ["sio₃", "sio3", "硅酸"])
    has_ca = contains_any(s, ["ca²", "ca2", "ca", "钙"])
    has_mg = contains_any(s, ["mg²", "mg2", "mg", "镁"])
    has_arsenic = contains_any(s, ["as(iii)", "as(v)", "as3", "as5", "砷"])
    has_nom_ha = contains_any(s, ["腐殖酸", "humic", "ha", "天然有机物", "nom", "海藻酸", "alginate"])
    has_bsa = contains_any(s, ["bsa", "蛋白"])
    has_oil = contains_any(s, ["油", "oil"])

    # Count mechanistic ion/organic categories, not exact chemical tokens.
    category_flags = [
        has_carbonate,
        has_sulfate,
        has_phosphate,
        has_chloride,
        has_nitrate,
        has_silicate,
        has_ca or has_mg,
        has_arsenic,
        has_nom_ha,
        has_bsa,
        has_oil,
    ]
    ion_type_count = int(sum(category_flags))
    strong_competing_anion_count = int(sum([has_carbonate, has_phosphate, has_silicate]))
    hardness_related = bool(has_ca or has_mg or has_bicarbonate)
    organic_related = bool(has_nom_ha or has_bsa or has_oil)

    if is_none:
        group = "无干扰/缺失"
    elif organic_related and ion_type_count == 1:
        group = "有机物干扰"
    elif organic_related and ion_type_count > 1:
        group = "有机物+无机离子复合干扰"
    elif has_arsenic:
        group = "砷物种干扰"
    elif strong_competing_anion_count >= 1 and hardness_related:
        group = "强竞争阴离子+硬度复合干扰"
    elif strong_competing_anion_count >= 1:
        group = "强竞争阴离子干扰"
    elif hardness_related:
        group = "硬度/碳酸盐相关干扰"
    elif ion_type_count >= 2:
        group = "多无机离子复合干扰"
    else:
        group = "其他/需人工确认"

    notes = []
    for flag, name in [
        (has_carbonate, "carbonate/bicarbonate"), (has_sulfate, "sulfate"),
        (has_phosphate, "phosphate"), (has_chloride, "chloride"),
        (has_nitrate, "nitrate"), (has_silicate, "silicate"),
        (has_ca, "Ca"), (has_mg, "Mg"), (has_arsenic, "As"),
        (has_nom_ha, "NOM/HA"), (has_bsa, "BSA/protein"), (has_oil, "oil/organic"),
    ]:
        if flag:
            notes.append(name)

    return {
        "auto_interference_group": group,
        "has_carbonate_or_bicarbonate": int(has_carbonate),
        "has_bicarbonate": int(has_bicarbonate),
        "has_sulfate": int(has_sulfate),
        "has_phosphate": int(has_phosphate),
        "has_chloride": int(has_chloride),
        "has_nitrate": int(has_nitrate),
        "has_silicate": int(has_silicate),
        "has_Ca": int(has_ca),
        "has_Mg": int(has_mg),
        "has_Ca_or_Mg_hardness": int(has_ca or has_mg),
        "has_arsenic": int(has_arsenic),
        "has_NOM_or_HA": int(has_nom_ha),
        "has_BSA_or_protein": int(has_bsa),
        "has_oil_or_other_organic": int(has_oil),
        "ion_mechanism_type_count": ion_type_count,
        "strong_competing_anion_count": strong_competing_anion_count,
        "has_hardness_related_ions": int(hardness_related),
        "has_organic_interference": int(organic_related),
        "auto_mapping_notes": ";".join(notes),
        "needs_manual_review": int(group == "其他/需人工确认"),
    }


def summarize_unique_category(
    df: pd.DataFrame,
    col: str,
    source_col: str | None,
    flag_func,
    raw_name: str,
) -> pd.DataFrame:
    tmp = df.copy()
    tmp["__norm"] = tmp[col].apply(norm_text)
    rows = []
    for val, sub in tmp.groupby("__norm", dropna=False):
        row = {
            raw_name: val,
            "n_rows": int(len(sub)),
            "row_fraction": float(len(sub) / len(tmp)),
        }
        if source_col and source_col in tmp.columns:
            papers = sorted(map(str, sub[source_col].dropna().unique().tolist()))
            row["n_papers"] = int(len(papers))
            row["paper_ids"] = ";".join(papers)
        else:
            row["n_papers"] = np.nan
            row["paper_ids"] = ""
        row.update(flag_func(val))
        row["manual_final_group"] = ""
        row["manual_notes"] = ""
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["n_rows", raw_name], ascending=[False, True]).reset_index(drop=True)
    return out


def find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    for c in cols:
        for k in candidates:
            if k.lower() in str(c).lower():
                return c
    return None


def main():
    project_root = find_project_root()
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df, data_file = read_model_data(project_root)

    coagulant_col = find_col(df, ["混凝剂", "Coagulant", "coagulant"])
    ion_col = find_col(df, ["干扰离子", "Interfering_ion", "interfering ion"])
    source_col = find_col(df, ["文献编号", "source", "Paper_ID", "paper", "literature"])

    if coagulant_col is None:
        raise ValueError("Cannot identify coagulant column. Expected column like '混凝剂'.")
    if ion_col is None:
        raise ValueError("Cannot identify interfering ion column. Expected column like '干扰离子'.")

    coagulant_audit = summarize_unique_category(
        df=df,
        col=coagulant_col,
        source_col=source_col,
        flag_func=coagulant_flags,
        raw_name="raw_coagulant_label",
    )
    ion_audit = summarize_unique_category(
        df=df,
        col=ion_col,
        source_col=source_col,
        flag_func=ion_flags,
        raw_name="raw_interfering_ion_label",
    )

    coagulant_out = tables_dir / "31a_coagulant_mapping_audit.csv"
    ion_out = tables_dir / "31a_interfering_ion_mapping_audit.csv"
    report_out = tables_dir / "31a_category_mapping_audit_report.txt"

    coagulant_audit.to_csv(coagulant_out, index=False, encoding="utf-8-sig")
    ion_audit.to_csv(ion_out, index=False, encoding="utf-8-sig")

    report_lines = []
    report_lines.append("31A category mapping audit report")
    report_lines.append("=" * 72)
    report_lines.append(f"Project root: {project_root}")
    report_lines.append(f"Data file: {data_file}")
    report_lines.append(f"Data shape: {df.shape}")
    report_lines.append(f"Coagulant column: {coagulant_col}")
    report_lines.append(f"Interfering ion column: {ion_col}")
    report_lines.append(f"Source column: {source_col}")
    if source_col:
        report_lines.append(f"Number of paper groups: {df[source_col].nunique(dropna=True)}")
    report_lines.append("")
    report_lines.append("Coagulant audit summary:")
    report_lines.append(f"  - Unique raw coagulant labels: {len(coagulant_audit)}")
    report_lines.append(f"  - Auto-family counts:")
    for k, v in coagulant_audit["auto_coagulant_family"].value_counts(dropna=False).items():
        report_lines.append(f"      {k}: {v}")
    report_lines.append(f"  - Labels needing manual review: {int(coagulant_audit['needs_manual_review'].sum())}")
    report_lines.append("")
    report_lines.append("Interfering-ion audit summary:")
    report_lines.append(f"  - Unique raw interfering-ion labels: {len(ion_audit)}")
    report_lines.append(f"  - Auto-group counts:")
    for k, v in ion_audit["auto_interference_group"].value_counts(dropna=False).items():
        report_lines.append(f"      {k}: {v}")
    report_lines.append(f"  - Labels needing manual review: {int(ion_audit['needs_manual_review'].sum())}")
    report_lines.append("")
    report_lines.append("Files saved:")
    report_lines.append(f"  - {coagulant_out}")
    report_lines.append(f"  - {ion_out}")
    report_lines.append("")
    report_lines.append("Important interpretation:")
    report_lines.append("1. This script is only an audit step. It does not modify model data.")
    report_lines.append("2. Auto-mapping is a first-pass guess and must be manually reviewed.")
    report_lines.append("3. Step 31B should only start after the two mapping tables are checked.")
    report_lines.append("4. Do not use these auto labels directly as final paper conclusions before review.")

    report_out.write_text("\n".join(report_lines), encoding="utf-8")

    print("31A category mapping audit completed.")
    print(f"Project root: {project_root}")
    print(f"Data file: {data_file}")
    print(f"Coagulant audit: {coagulant_out}")
    print(f"Interfering ion audit: {ion_out}")
    print(f"Report: {report_out}")
    print("\nOpen the two CSV files and review auto-mapped categories before Step 31B.")


if __name__ == "__main__":
    main()
