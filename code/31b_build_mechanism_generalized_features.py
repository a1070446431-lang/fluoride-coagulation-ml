# -*- coding: utf-8 -*-
"""
31B - Build mechanism-aware generalized feature table for grouped validation
混凝除氟项目：构建机理感知类别泛化特征表

Purpose / 目的：
1) 不改变原始数据，不删除样本。
2) 在 model_ready_v1_with_source.csv 基础上新增混凝剂与干扰离子的机理泛化特征。
3) 输出 raw category（原始类别） vs mechanism category（机理类别）的 unseen-category 诊断。
4) 为下一步 31C grouped validation（分组验证）重跑模型准备输入表。

Important / 重要说明：
- 本脚本不训练模型，不替代 29 的模型流程。
- 31C 若要严格复用 29_grouped_validation_candidate_screening.py 的模型、划分、权重搜索和评价逻辑，
  需要在 29 脚本基础上只替换输入特征表。
"""

from __future__ import annotations

import re
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


SEEDS = [7, 21, 42, 66, 88]
TARGET_COL_CANDIDATES = ["氟去除率_pct", "氟去除率", "F_removal_pct", "Removal_pct"]
SOURCE_COL_CANDIDATES = ["文献编号", "Paper_ID", "paper_id", "source", "Source"]

COL_CANDIDATES = {
    "coagulant": ["混凝剂", "coagulant", "Coagulant"],
    "ion": ["干扰离子", "interfering_ion", "Interfering_ion"],
    "dose": ["剂量_mM", "剂量", "coagulant_dose_mM", "Dose_mM"],
    "initial_f": ["初始氟_mg_L", "初始F_mg_L", "Initial_F_mg_L"],
    "ph": ["进水pH", "influent_pH", "pH"],
    "temp": ["水温_C", "temperature_C", "Temperature_C"],
    "hrt": ["停留时间_min", "retention_time_min", "Retention_time_min"],
    "ion_conc": ["干扰离子浓度_mmol_L", "interfering_ion_conc_mmol_L", "Interfering_ion_conc_mmol_L"],
    "free_f_ratio": ["游离氟占比", "free_f_ratio", "Free_f_ratio"],
    "fluoride_form": ["氟形态", "fluoride_form", "Fluoride_form"],
}


def find_project_root() -> Path:
    """Find project root by locating data/processed directory."""
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents]
    for p in candidates:
        if (p / "data" / "processed").exists():
            return p
    return Path.cwd()


def pick_col(df: pd.DataFrame, candidates: List[str], role: str, required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Cannot find column for role={role}. Candidates={candidates}. Existing columns={list(df.columns)}")
    return None


def normalize_text(x) -> str:
    if pd.isna(x):
        return "无"
    s = str(x).strip()
    if not s or s.lower() in ["nan", "none", "null", "na"]:
        return "无"
    # unify brackets and separators but keep chemistry symbols
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("，", ",").replace("；", ";")
    s = re.sub(r"\s+", "", s)
    return s


def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def coagulant_descriptors(x) -> Dict[str, object]:
    text = normalize_text(x)
    low = text.lower()

    no_coag = text in ["无", "0", "未添加", "不添加", "空白"]

    contains_al = bool(re.search(r"铝|明矾|al(?!g)", low)) or "pac" in low or "pacl" in low or "paf" in low or "psaf" in low or "pfasic" in low or "csa" in low
    contains_fe = bool(re.search(r"铁|fe|ferr", low)) or "psf" in low or "pfs" in low or "psfc" in low or "psfcl" in low or "psaf" in low or "pfasic" in low
    contains_zr = bool(re.search(r"锆|zr|zircon", low)) or "zxc" in low or "pzs" in low
    contains_ca = bool(re.search(r"钙|石灰|氧化钙|氢氧化钙", text))
    contains_mg = bool(re.search(r"镁|mg", low))
    contains_si = bool(re.search(r"硅|sio|silic", low)) or "ps" in low or "pfasic" in low or "psaf" in low or "psf" in low or "pzs" in low
    contains_ti = bool(re.search(r"钛|ti", low))
    contains_ree = bool(re.search(r"镧|稀土", text)) or "psfcl" in low

    is_polymeric = bool(re.search(r"聚|poly|paf|pac|pacl|psaf|psf|psfc|psfcl|pfasic|pzs", low))
    is_commercial = bool(re.search(r"商业|神美|除氟剂", text))
    is_composite = bool(re.search(r"复合|/|\+|掺杂|改性", text)) or sum([
        contains_al, contains_fe, contains_zr, contains_ca, contains_mg, contains_si, contains_ti, contains_ree
    ]) >= 2 or is_commercial
    is_sulfate = bool(re.search(r"硫酸|sulfate|csa", low))
    is_chloride = bool(re.search(r"氯化|chloride|pacl|pac", low))
    is_oxide_hydroxide = bool(re.search(r"氧化|氢氧化|水合氧化|oxide|hydroxide", low))

    if no_coag:
        family = "无/缺失"
    elif is_commercial and not any([contains_al, contains_fe, contains_zr, contains_ca, contains_mg, contains_si, contains_ti, contains_ree]):
        # 31A manual review item: 商业复合除氟混凝剂(神美除氟剂A)
        family = "商业复合/成分未知"
    elif contains_al and contains_fe:
        family = "铝铁复合"
    elif contains_al and contains_mg:
        family = "铝镁复合"
    elif contains_zr:
        family = "锆基/含锆"
    elif contains_ca and contains_al:
        family = "钙铝复合"
    elif contains_ca:
        family = "钙基"
    elif contains_mg:
        family = "镁基"
    elif contains_fe:
        family = "铁基/含铁"
    elif contains_al:
        family = "铝基/含铝"
    else:
        family = "其他/需人工确认"

    return {
        "机理_混凝剂大类": family,
        "机理_含Al": int(contains_al),
        "机理_含Fe": int(contains_fe),
        "机理_含Zr": int(contains_zr),
        "机理_含Ca": int(contains_ca),
        "机理_含Mg": int(contains_mg),
        "机理_含Si": int(contains_si),
        "机理_含Ti": int(contains_ti),
        "机理_含La或稀土": int(contains_ree),
        "机理_聚合型混凝剂": int(is_polymeric),
        "机理_复合型混凝剂": int(is_composite),
        "机理_硫酸盐体系混凝剂": int(is_sulfate),
        "机理_氯化物体系混凝剂": int(is_chloride),
        "机理_氧化物氢氧化物体系": int(is_oxide_hydroxide),
        "机理_商业复合药剂": int(is_commercial),
    }


def ion_descriptors(x) -> Dict[str, object]:
    text = normalize_text(x)
    low = text.lower()

    no_ion = text in ["无", "0", "未添加", "不添加", "空白"] or "无干扰" in text or "没有干扰" in text

    has_hco3 = bool(re.search(r"hco3|hco₃|碳酸氢|重碳酸", low))
    has_co3 = bool(re.search(r"co3|co₃|碳酸根|碳酸盐", low))
    has_sulfate = bool(re.search(r"so4|so₄|硫酸", low))
    has_phosphate = bool(re.search(r"po4|po₄|磷酸", low))
    has_chloride = bool(re.search(r"cl|cl⁻|氯", low))
    has_nitrate = bool(re.search(r"no3|no₃|硝酸", low))
    has_silicate = bool(re.search(r"sio3|sio₃|sio2|sio₂|硅酸|硅", low))
    has_ca = bool(re.search(r"ca|ca²|钙", low))
    has_mg = bool(re.search(r"mg|mg²|镁", low))
    has_arsenic = bool(re.search(r"as\(?iii\)?|as\(?v\)?|砷", low))
    has_nom = bool(re.search(r"腐殖酸|海藻酸|天然有机物|nom|humic|alginate", low))
    has_bsa = bool(re.search(r"bsa|蛋白", low))
    has_oil = bool(re.search(r"油|oil", low))

    if no_ion:
        has_hco3 = has_co3 = has_sulfate = has_phosphate = has_chloride = has_nitrate = False
        has_silicate = has_ca = has_mg = has_arsenic = has_nom = has_bsa = has_oil = False

    has_carbonate = has_hco3 or has_co3
    has_hardness = has_ca or has_mg
    has_organic = has_nom or has_bsa or has_oil

    # 强竞争阴离子：按氟去除竞争吸附/络合影响更强的 PO4/CO3/HCO3/SiO3 处理；SO4 单独保留为硫酸盐干扰。
    strong_competing_count = int(has_carbonate) + int(has_phosphate) + int(has_silicate)
    inorganic_flag_count = sum([has_carbonate, has_sulfate, has_phosphate, has_chloride, has_nitrate, has_silicate, has_hardness, has_arsenic])
    mechanism_type_count = sum([has_carbonate, has_sulfate, has_phosphate, has_chloride, has_nitrate, has_silicate, has_hardness, has_arsenic, has_organic])

    if no_ion:
        group = "无干扰/缺失"
    elif has_organic and inorganic_flag_count > 0:
        group = "有机物+无机离子复合干扰"
    elif has_organic:
        group = "有机物干扰"
    elif has_arsenic:
        group = "砷物种干扰"
    elif strong_competing_count > 0 and has_hardness:
        group = "强竞争阴离子+硬度复合干扰"
    elif strong_competing_count > 0:
        group = "强竞争阴离子干扰"
    elif has_hardness or has_carbonate:
        group = "硬度/碳酸盐相关干扰"
    elif has_sulfate or has_chloride or has_nitrate:
        # 31A manual review items: NO3-, SO4(2-), Cl- are assigned here.
        group = "常见无机阴离子干扰"
    else:
        group = "其他/需人工确认"

    return {
        "机理_干扰离子组": group,
        "机理_是否有干扰离子": int(not no_ion),
        "机理_含碳酸盐或碳酸氢盐": int(has_carbonate),
        "机理_含HCO3": int(has_hco3),
        "机理_含CO3": int(has_co3),
        "机理_含SO4": int(has_sulfate),
        "机理_含PO4": int(has_phosphate),
        "机理_含Cl": int(has_chloride),
        "机理_含NO3": int(has_nitrate),
        "机理_含硅酸盐": int(has_silicate),
        "机理_含Ca": int(has_ca),
        "机理_含Mg": int(has_mg),
        "机理_含Ca或Mg硬度": int(has_hardness),
        "机理_含As物种": int(has_arsenic),
        "机理_含NOM或腐殖酸": int(has_nom),
        "机理_含BSA或蛋白": int(has_bsa),
        "机理_含油类有机物": int(has_oil),
        "机理_有机物干扰": int(has_organic),
        "机理_强竞争阴离子种类数": int(strong_competing_count),
        "机理_干扰机理种类数": int(mechanism_type_count),
    }


def add_numeric_mechanism_features(df: pd.DataFrame, cols: Dict[str, str | None]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    dose = to_numeric_safe(df[cols["dose"]]) if cols.get("dose") else pd.Series(np.nan, index=df.index)
    init_f = to_numeric_safe(df[cols["initial_f"]]) if cols.get("initial_f") else pd.Series(np.nan, index=df.index)
    ph = to_numeric_safe(df[cols["ph"]]) if cols.get("ph") else pd.Series(np.nan, index=df.index)
    temp = to_numeric_safe(df[cols["temp"]]) if cols.get("temp") else pd.Series(np.nan, index=df.index)
    hrt = to_numeric_safe(df[cols["hrt"]]) if cols.get("hrt") else pd.Series(np.nan, index=df.index)
    ion_conc = to_numeric_safe(df[cols["ion_conc"]]) if cols.get("ion_conc") else pd.Series(np.nan, index=df.index)
    free_f = to_numeric_safe(df[cols["free_f_ratio"]]) if cols.get("free_f_ratio") else pd.Series(np.nan, index=df.index)

    eps = 1e-6

    out["机理_log1p_剂量"] = np.log1p(dose.clip(lower=0))
    out["机理_log1p_初始氟"] = np.log1p(init_f.clip(lower=0))
    out["机理_log1p_干扰离子浓度"] = np.log1p(ion_conc.clip(lower=0))
    out["机理_相对投加强度_剂量除以初始氟"] = dose / (init_f.abs() + eps)
    out["机理_剂量乘初始氟"] = dose * init_f
    out["机理_pH乘剂量"] = ph * dose
    out["机理_pH乘初始氟"] = ph * init_f
    out["机理_pH距中性距离"] = (ph - 7).abs()
    out["机理_pH弱酸至中性"] = ((ph >= 5.5) & (ph <= 7.5)).astype(int)
    out["机理_pH酸性"] = (ph < 5.5).astype(int)
    out["机理_pH碱性"] = (ph > 8.0).astype(int)
    out["机理_pH极端"] = ((ph < 4.0) | (ph > 10.0)).astype(int)
    out["机理_温度缺失"] = temp.isna().astype(int)
    out["机理_pH缺失"] = ph.isna().astype(int)
    out["机理_剂量缺失"] = dose.isna().astype(int)
    out["机理_初始氟缺失"] = init_f.isna().astype(int)
    out["机理_停留时间缺失"] = hrt.isna().astype(int)
    out["机理_干扰离子浓度缺失"] = ion_conc.isna().astype(int)
    out["机理_游离氟占比缺失"] = free_f.isna().astype(int)

    return out


def grouped_split_groups(groups: List[str], seed: int) -> Tuple[List[str], List[str], List[str]]:
    """27 groups -> 15 train, 6 valid, 6 test. Used only for coverage diagnostics."""
    groups = list(groups)
    rng = np.random.RandomState(seed)
    groups = sorted(groups)
    rng.shuffle(groups)
    n_groups = len(groups)
    n_train = int(round(n_groups * 15 / 27))
    n_valid = int(round(n_groups * 6 / 27))
    train_groups = groups[:n_train]
    valid_groups = groups[n_train:n_train + n_valid]
    test_groups = groups[n_train + n_valid:]
    return train_groups, valid_groups, test_groups


def unseen_fraction(train_values: pd.Series, test_values: pd.Series) -> float:
    train_set = set(train_values.dropna().astype(str))
    if len(test_values) == 0:
        return np.nan
    return float(~test_values.dropna().astype(str).isin(train_set).mean()) if False else float((~test_values.astype(str).isin(train_set)).mean())


def build_coverage_diagnosis(df: pd.DataFrame, source_col: str, coag_col: str, ion_col: str) -> pd.DataFrame:
    rows = []
    groups = sorted(df[source_col].dropna().astype(str).unique())

    for seed in SEEDS:
        train_g, valid_g, test_g = grouped_split_groups(groups, seed)
        train_df = df[df[source_col].astype(str).isin(train_g)]
        valid_df = df[df[source_col].astype(str).isin(valid_g)]
        test_df = df[df[source_col].astype(str).isin(test_g)]

        comparisons = [
            ("Raw_Coagulant", coag_col),
            ("Mechanism_Coagulant_Family", "机理_混凝剂大类"),
            ("Raw_Interfering_Ion", ion_col),
            ("Mechanism_Interfering_Ion_Group", "机理_干扰离子组"),
        ]
        for name, col in comparisons:
            tr_set = set(train_df[col].astype(str).fillna("缺失"))
            te_vals = test_df[col].astype(str).fillna("缺失")
            unseen_mask = ~te_vals.isin(tr_set)
            rows.append({
                "Seed": seed,
                "Feature_View": name,
                "Train_n_rows": len(train_df),
                "Valid_n_rows": len(valid_df),
                "Test_n_rows": len(test_df),
                "Train_n_papers": len(train_g),
                "Valid_n_papers": len(valid_g),
                "Test_n_papers": len(test_g),
                "Unseen_sample_fraction": float(unseen_mask.mean()) if len(te_vals) else np.nan,
                "Unseen_n_categories": int(te_vals[unseen_mask].nunique()) if len(te_vals) else 0,
                "Train_n_categories": int(train_df[col].astype(str).nunique()),
                "Test_n_categories": int(te_vals.nunique()),
            })

    return pd.DataFrame(rows)


def save_unseen_barplot(summary: pd.DataFrame, fig_dir: Path):
    if plt is None or summary.empty:
        return
    agg = summary.groupby("Feature_View")["Unseen_sample_fraction"].agg(["mean", "std"]).reset_index()
    order = ["Raw_Coagulant", "Mechanism_Coagulant_Family", "Raw_Interfering_Ion", "Mechanism_Interfering_Ion_Group"]
    agg["Feature_View"] = pd.Categorical(agg["Feature_View"], categories=order, ordered=True)
    agg = agg.sort_values("Feature_View")

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)
    x = np.arange(len(agg))
    bars = ax.bar(x, agg["mean"].values, yerr=agg["std"].fillna(0).values, capsize=4, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v).replace("_", "\n") for v in agg["Feature_View"]], fontsize=9)
    ax.set_ylabel("Mean unseen sample fraction")
    ax.set_title("Raw categories vs mechanism-aware categories\n(grouped split coverage diagnosis)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for rect, val in zip(bars, agg["mean"].values):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "75_unseen_category_reduction_after_mechanism_encoding.png", bbox_inches="tight")
    plt.close(fig)


def main():
    root = find_project_root()
    data_path = root / "data" / "processed" / "model_ready_v1_with_source.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find {data_path}. Please run 29c rebuild first.")

    table_dir = root / "outputs" / "tables"
    fig_dir = root / "outputs" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, encoding="utf-8-sig")

    target_col = pick_col(df, TARGET_COL_CANDIDATES, "target", required=True)
    source_col = pick_col(df, SOURCE_COL_CANDIDATES, "source", required=True)
    cols = {}
    for role, candidates in COL_CANDIDATES.items():
        cols[role] = pick_col(df, candidates, role, required=(role in ["coagulant", "ion"]))

    coag_col = cols["coagulant"]
    ion_col = cols["ion"]

    # Build descriptors row-wise
    coag_desc = pd.DataFrame([coagulant_descriptors(x) for x in df[coag_col]], index=df.index)
    ion_desc = pd.DataFrame([ion_descriptors(x) for x in df[ion_col]], index=df.index)
    num_desc = add_numeric_mechanism_features(df, cols)

    # Interactions between pH and coagulant-family flags
    ph_series = to_numeric_safe(df[cols["ph"]]) if cols.get("ph") else pd.Series(np.nan, index=df.index)
    dose_series = to_numeric_safe(df[cols["dose"]]) if cols.get("dose") else pd.Series(np.nan, index=df.index)

    interaction_desc = pd.DataFrame(index=df.index)
    for flag_col in ["机理_含Al", "机理_含Fe", "机理_含Zr", "机理_含Ca", "机理_含Mg", "机理_含Si", "机理_含La或稀土"]:
        interaction_desc[f"机理_pH乘{flag_col.replace('机理_', '')}"] = ph_series * coag_desc[flag_col]
        interaction_desc[f"机理_剂量乘{flag_col.replace('机理_', '')}"] = dose_series * coag_desc[flag_col]

    out_df = pd.concat([df.copy(), coag_desc, ion_desc, num_desc, interaction_desc], axis=1)

    out_data_path = root / "data" / "processed" / "model_ready_v1_with_source_mechanism31b.csv"
    out_df.to_csv(out_data_path, index=False, encoding="utf-8-sig")

    # Unique mapping outputs
    coag_map = pd.concat([df[[coag_col, source_col]].copy(), coag_desc], axis=1)
    coag_map = coag_map.groupby([coag_col] + list(coag_desc.columns), dropna=False).agg(
        n_rows=(source_col, "size"),
        n_papers=(source_col, lambda s: s.astype(str).nunique()),
        paper_ids=(source_col, lambda s: ";".join(sorted(s.astype(str).unique()))),
    ).reset_index().sort_values("n_rows", ascending=False)
    coag_map.to_csv(table_dir / "31b_coagulant_mechanism_mapping_applied.csv", index=False, encoding="utf-8-sig")

    ion_map = pd.concat([df[[ion_col, source_col]].copy(), ion_desc], axis=1)
    ion_map = ion_map.groupby([ion_col] + list(ion_desc.columns), dropna=False).agg(
        n_rows=(source_col, "size"),
        n_papers=(source_col, lambda s: s.astype(str).nunique()),
        paper_ids=(source_col, lambda s: ";".join(sorted(s.astype(str).unique()))),
    ).reset_index().sort_values("n_rows", ascending=False)
    ion_map.to_csv(table_dir / "31b_interfering_ion_mechanism_mapping_applied.csv", index=False, encoding="utf-8-sig")

    mechanism_cols = list(coag_desc.columns) + list(ion_desc.columns) + list(num_desc.columns) + list(interaction_desc.columns)
    pd.DataFrame({"mechanism_feature_column": mechanism_cols}).to_csv(
        table_dir / "31b_mechanism_feature_columns.csv", index=False, encoding="utf-8-sig"
    )

    coverage_df = build_coverage_diagnosis(out_df, source_col, coag_col, ion_col)
    coverage_df.to_csv(table_dir / "31b_raw_vs_mechanism_unseen_category_diagnosis.csv", index=False, encoding="utf-8-sig")
    save_unseen_barplot(coverage_df, fig_dir)

    coverage_summary = coverage_df.groupby("Feature_View").agg(
        mean_unseen_sample_fraction=("Unseen_sample_fraction", "mean"),
        std_unseen_sample_fraction=("Unseen_sample_fraction", "std"),
        mean_unseen_n_categories=("Unseen_n_categories", "mean"),
    ).reset_index()
    coverage_summary.to_csv(table_dir / "31b_raw_vs_mechanism_unseen_category_summary.csv", index=False, encoding="utf-8-sig")

    # Report
    n_groups = df[source_col].astype(str).nunique()
    report_lines = []
    report_lines.append("31B mechanism-aware feature build report")
    report_lines.append("=" * 72)
    report_lines.append(f"Project root: {root}")
    report_lines.append(f"Input data: {data_path}")
    report_lines.append(f"Output data: {out_data_path}")
    report_lines.append(f"Data shape before: {df.shape}")
    report_lines.append(f"Data shape after: {out_df.shape}")
    report_lines.append(f"Target column: {target_col}")
    report_lines.append(f"Source column: {source_col}")
    report_lines.append(f"Number of literature groups: {n_groups}")
    report_lines.append("")
    report_lines.append("Mechanism feature counts:")
    report_lines.append(f"  - Coagulant mechanism columns: {len(coag_desc.columns)}")
    report_lines.append(f"  - Interfering-ion mechanism columns: {len(ion_desc.columns)}")
    report_lines.append(f"  - Numeric mechanism columns: {len(num_desc.columns)}")
    report_lines.append(f"  - pH/dose × coagulant interaction columns: {len(interaction_desc.columns)}")
    report_lines.append(f"  - Total new mechanism columns: {len(mechanism_cols)}")
    report_lines.append("")
    report_lines.append("Raw vs mechanism unseen-category summary:")
    for _, row in coverage_summary.iterrows():
        report_lines.append(
            f"  - {row['Feature_View']}: mean unseen fraction = {row['mean_unseen_sample_fraction']:.4f} "
            f"± {row['std_unseen_sample_fraction']:.4f}; mean unseen categories = {row['mean_unseen_n_categories']:.2f}"
        )
    report_lines.append("")
    report_lines.append("Manual 31A review decisions implemented:")
    report_lines.append("  - 商业复合除氟混凝剂(神美除氟剂A) -> 商业复合/成分未知; keep composite/commercial flags; composition flags remain 0.")
    report_lines.append("  - NO3-, SO4(2-), Cl- -> 常见无机阴离子干扰; keep individual nitrate/sulfate/chloride flags.")
    report_lines.append("")
    report_lines.append("Next step:")
    report_lines.append("  - 31C should reuse the real 29_grouped_validation_candidate_screening.py workflow.")
    report_lines.append("  - Only replace/add input feature representation using model_ready_v1_with_source_mechanism31b.csv.")
    report_lines.append("  - Do not change seeds, grouped split logic, candidate models, weight-search objective, or evaluation metrics unless explicitly documented.")

    report_path = table_dir / "31b_mechanism_feature_build_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
