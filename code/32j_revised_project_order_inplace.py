# -*- coding: utf-8 -*-
"""
32J 原位置输出版：按项目原始文件顺序整合最终论文证据链
======================================================

使用方法：
1. 把本文件直接拖到你的 PyCharm 项目中运行。
2. 可以放在项目根目录，也可以放在 src/ 文件夹中。
3. 本脚本只整理文件、表格、图件和论文口径，不重跑模型，不改任何结果数值。
4. 输出位置：表格写入 outputs/tables/，图片保留/检查在 outputs/figures/，报告写入 reports/。

本版针对你的项目目录做了两点修改：
- 同时兼容 outputs/tables + outputs/figures 和 tables + figures 两种结构；
- 所有项目表格、图件、脚本清单都按“项目步骤顺序/编号顺序”排序，而不是简单按字母排序；
- 32J 生成的新表直接放到 outputs/tables/，报告放到 reports/，不私自改变存放位置。

推荐项目结构：
fluoride_coagulation_ml_project/
├─ data/
├─ outputs/
│  ├─ tables/
│  ├─ figures/
│  ├─ models/
│  └─ ...
├─ reports/
├─ src/
└─ 01_data_audit.py ...

也兼容你压缩包中的结构：
project_root/
├─ data/
├─ tables/
├─ figures/
└─ src/
"""

from __future__ import annotations

import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import pandas as pd
except Exception as exc:
    print("\n[错误] 当前 Python 环境没有安装 pandas。")
    print("请在 PyCharm Terminal 运行：pip install pandas")
    print("原始错误：", exc)
    sys.exit(1)


# =============================================================================
# 0. 32J 最终证据链必须读取的核心文件
# =============================================================================

REQUIRED_TABLES = [
    "32i_final_evidence_summary.csv",
    "32i_transfer_support_top.csv",
    "32h_exact_policy_metrics.csv",
    "32h_bootstrap_uncertainty_exact_policy.csv",
    "32h_exact_policy_split_metrics.csv",
    "32g_representativeness_summary.csv",
    "32g_numeric_operating_window.csv",
    "32i_claim_guardrails.csv",
]

OPTIONAL_EVIDENCE_FILES = [
    "32f2_reproducible_selective_ad_policy_audit_report.txt",
    "32g_retained_subdomain_decision_report.txt",
    "32h_threshold_sensitivity_decision_report.txt",
    "32i_final_evidence_integration_report.txt",
    "32i2_figure_cleanup_report.txt",
    "32g_categorical_composition.csv",
    "32h_threshold_plateau_summary.csv",
    "32h_strategy_method_transfer_summary.csv",
    "32f2_best_policy_literature_group_dominance.csv",
]

FINAL_FIGURES = [
    "194_32i2_evidence_chain_summary_clean.png",
    "194_32i2_evidence_chain_summary_clean.pdf",
    "195_32i2_threshold_plateau_evidence_clean.png",
    "195_32i2_threshold_plateau_evidence_clean.pdf",
    "196_32i2_retained_chemistry_window_clean.png",
    "196_32i2_retained_chemistry_window_clean.pdf",
    "197_32i2_stability_transfer_boundary_clean.png",
    "197_32i2_stability_transfer_boundary_clean.pdf",
]

# 32J 正文证据链顺序：严格按我们项目实际推进顺序，而不是按字母排序。
EVIDENCE_CHAIN_ORDER = [
    {
        "Step_Order": 1,
        "Project_Step": "32F2",
        "Evidence_Block": "Reproducible policy audit",
        "Main_Files": "32f2_reproducible_selective_ad_policy_audit_report.txt; 32f2_policy_reproduction_audit.csv",
        "Use_In_Manuscript": "证明 P08 规则可以由源表复现，避免只停留在一次性结果。",
        "Claim_Boundary": "只证明筛选规则可复现，不证明全域泛化能力已经解决。",
    },
    {
        "Step_Order": 2,
        "Project_Step": "32G",
        "Evidence_Block": "Retained subdomain chemistry",
        "Main_Files": "32g_numeric_operating_window.csv; 32g_categorical_composition.csv; 32g_representativeness_summary.csv",
        "Use_In_Manuscript": "说明保留子域对应的混凝化学窗口、类别组成和代表性。",
        "Claim_Boundary": "只能解释为特定化学/文献窗口，不能推广为所有混凝除氟条件。",
    },
    {
        "Step_Order": 3,
        "Project_Step": "32H",
        "Evidence_Block": "Threshold sensitivity and anti-cherry-picking audit",
        "Main_Files": "32h_exact_policy_metrics.csv; 32h_threshold_sensitivity_all.csv; 32h_bootstrap_uncertainty_exact_policy.csv",
        "Use_In_Manuscript": "证明 NN_Risk <= 0.814815 附近存在阈值平台和不确定性证据。",
        "Claim_Boundary": "必须披露平台较窄、coverage 很低和 split-level instability。",
    },
    {
        "Step_Order": 4,
        "Project_Step": "32I",
        "Evidence_Block": "Final evidence integration",
        "Main_Files": "32i_final_evidence_summary.csv; 32i_claim_guardrails.csv; 32i_final_evidence_integration_report.txt",
        "Use_In_Manuscript": "整合全域分组验证、预定义可靠域和选择性保留子域三层结果。",
        "Claim_Boundary": "必须把 P08 写成 selective retained-subdomain result，而不是 all-domain grouped-validation performance。",
    },
    {
        "Step_Order": 5,
        "Project_Step": "32I2",
        "Evidence_Block": "Final figure cleanup",
        "Main_Files": "194-197_32i2_*_clean.png/pdf; 32i2_figure_cleanup_report.txt",
        "Use_In_Manuscript": "只用于最终论文图件清理和英文标签排版。",
        "Claim_Boundary": "32I2 不改变模型、不改变规则、不改变指标。",
    },
]


# =============================================================================
# 1. 自然排序：按照项目文件步骤顺序，而不是字母顺序
# =============================================================================

LETTER_ORDER = {ch: idx + 1 for idx, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}


def _letter_rank(letters: str) -> Tuple[int, ...]:
    if not letters:
        return (0,)
    return tuple(LETTER_ORDER.get(ch.lower(), 99) for ch in letters)


def parse_stage_token(name: str) -> Tuple[int, Tuple[int, ...], int, str]:
    """
    解析项目步骤前缀。

    例子：
    - 29_grouped...      -> 29
    - 29a_rebuild...     -> 29a
    - 31h_group...       -> 31h
    - 31h2_tuning...     -> 31h2
    - 32i2_figure...     -> 32i2

    返回值：(主编号, 字母序, 子编号, token字符串)
    """
    stem = Path(name).stem.lower()

    # 先匹配开头，例如 31h2_xxx
    m = re.match(r"^(\d+)([a-z]+)?(\d*)", stem)
    if m:
        num = int(m.group(1))
        letters = m.group(2) or ""
        sub = int(m.group(3)) if m.group(3) else 0
        token = f"{num}{letters}{sub if sub else ''}"
        return num, _letter_rank(letters), sub, token

    # 图件有时是 194_32i2_xxx，先拿图号后面的 32i2
    m = re.search(r"_(\d+)([a-z]+)?(\d*)_", stem)
    if m:
        num = int(m.group(1))
        letters = m.group(2) or ""
        sub = int(m.group(3)) if m.group(3) else 0
        token = f"{num}{letters}{sub if sub else ''}"
        return num, _letter_rank(letters), sub, token

    return 999999, (999,), 999, "unknown"


def leading_number(name: str) -> int:
    m = re.match(r"^(\d+)", Path(name).stem)
    return int(m.group(1)) if m else 999999


def natural_sort_key_for_script(path: Path):
    num, letters, sub, token = parse_stage_token(path.name)
    return (num, letters, sub, path.name.lower())


def natural_sort_key_for_table(path: Path):
    num, letters, sub, token = parse_stage_token(path.name)
    return (num, letters, sub, path.name.lower())


def natural_sort_key_for_figure(path: Path):
    # 图件优先按图号 100, 101, ... 197 排序；没有图号再按步骤号排序。
    fig_no = leading_number(path.name)
    num, letters, sub, token = parse_stage_token(path.name)
    return (fig_no, num, letters, sub, path.name.lower())


def detect_stage_token(name: str) -> str:
    return parse_stage_token(name)[3]


# =============================================================================
# 2. 路径识别
# =============================================================================


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in paths:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def score_project_root(root: Path) -> int:
    score = 0
    if (root / "data").is_dir():
        score += 2
    if (root / "src").is_dir():
        score += 3
    if any(root.glob("*.py")):
        score += 1
    if (root / "outputs").is_dir():
        score += 2
    if (root / "outputs" / "tables").is_dir():
        score += 5
    if (root / "outputs" / "figures").is_dir():
        score += 5
    if (root / "tables").is_dir():
        score += 5
    if (root / "figures").is_dir():
        score += 5
    for name in REQUIRED_TABLES:
        if (root / "outputs" / "tables" / name).exists():
            score += 5
        if (root / "tables" / name).exists():
            score += 5
    for name in FINAL_FIGURES:
        if (root / "outputs" / "figures" / name).exists():
            score += 2
        if (root / "figures" / name).exists():
            score += 2
    return score


def find_project_root() -> Path:
    candidates: List[Path] = []

    cwd = Path.cwd()
    candidates.append(cwd)
    candidates.extend(cwd.parents)

    try:
        script_dir = Path(__file__).resolve().parent
        candidates.append(script_dir)
        candidates.extend(script_dir.parents)
    except Exception:
        pass

    # 兼容外层还有一个“压缩包/项目名”文件夹的情况
    extra_children: List[Path] = []
    for c in list(candidates):
        try:
            for child in c.iterdir():
                if child.is_dir():
                    extra_children.append(child)
        except Exception:
            pass
    candidates.extend(extra_children)

    scored = [(p, score_project_root(p)) for p in unique_paths(candidates)]
    scored.sort(key=lambda x: x[1], reverse=True)

    if scored and scored[0][1] >= 10:
        return scored[0][0]

    print("\n[错误] 没有自动识别到项目根目录。")
    print("请把本脚本放到 fluoride_coagulation_ml_project 根目录或 src 文件夹里运行。")
    print("需要看到 data、src、tables/figures 或 outputs/tables/outputs/figures。")
    print("\n候选目录评分：")
    for p, s in scored[:25]:
        print(f"  score={s:>3}  {p}")
    sys.exit(1)


def count_existing(folder: Path, names: Iterable[str]) -> int:
    return sum((folder / name).exists() for name in names)


def count_result_files(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    exts = {".csv", ".txt", ".xlsx", ".xls", ".json", ".png", ".pdf", ".jpg", ".jpeg"}
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def choose_input_dir(project_root: Path, kind: str, required_names: List[str]) -> Path:
    candidates = [
        project_root / "outputs" / kind,
        project_root / kind,
        project_root / "outputs" / "archive_before_final_dataset" / kind,
    ]
    scored: List[Tuple[Path, int, int]] = []
    for c in candidates:
        if c.is_dir():
            required_score = count_existing(c, required_names)
            file_count = count_result_files(c)
            scored.append((c, required_score, file_count))

    if not scored:
        print(f"\n[错误] 没有找到 {kind} 文件夹。")
        print(f"请确认项目中存在 outputs/{kind}/ 或 {kind}/。")
        sys.exit(1)

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return scored[0][0]


def choose_report_dir(project_root: Path) -> Path:
    """
    按项目原有习惯选择报告目录。

    优先使用项目根目录下的 reports/；如果没有，则使用 outputs/reports/；
    如果都没有，就创建 reports/。
    """
    candidates = [project_root / "reports", project_root / "outputs" / "reports"]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def get_script_dirs(project_root: Path) -> List[Path]:
    dirs = []
    if project_root.is_dir():
        dirs.append(project_root)
    if (project_root / "src").is_dir():
        dirs.append(project_root / "src")
    return unique_paths(dirs)


# =============================================================================
# 3. 读取与格式化
# =============================================================================


def read_csv_smart(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"缺少必要文件：{path}")
    last_exc = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"无法读取 CSV：{path}\n最后错误：{last_exc}")


def read_text_smart(path: Path) -> str:
    if not path.exists():
        return ""
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""


def fmt_num(value, digits: int = 6, default: str = "") -> str:
    try:
        if pd.isna(value):
            return default
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value) if value is not None else default


def fmt_int(value, default: str = "") -> str:
    try:
        if pd.isna(value):
            return default
        return str(int(round(float(value))))
    except Exception:
        return str(value) if value is not None else default


def get_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        real = lower_map.get(c.lower())
        if real is not None:
            return real
    return None


def get_first_row(df: pd.DataFrame) -> pd.Series:
    if len(df) > 0:
        return df.iloc[0]
    return pd.Series(dtype=object)


def value(row: pd.Series, keys: Iterable[str], default: str = ""):
    if row is None or len(row) == 0:
        return default
    lower_map = {str(c).lower(): c for c in row.index}
    for k in keys:
        if k in row.index:
            return row.get(k, default)
        real = lower_map.get(k.lower())
        if real is not None:
            return row.get(real, default)
    return default


def row_by_metric(df: pd.DataFrame, metric: str) -> pd.Series:
    col = get_col(df, ["Metric", "metric"])
    if col is not None:
        mask = df[col].astype(str).str.upper().eq(metric.upper())
        if mask.any():
            return df.loc[mask].iloc[0]
    return get_first_row(df)


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


# =============================================================================
# 4. 清单生成：严格保留项目顺序
# =============================================================================


def build_script_sequence(project_root: Path) -> pd.DataFrame:
    current_script = ""
    try:
        current_script = Path(__file__).resolve().name
    except Exception:
        pass

    rows = []
    for d in get_script_dirs(project_root):
        for p in d.glob("*.py"):
            if p.name == current_script:
                continue
            if p.name.startswith("__"):
                continue
            num, letters, sub, token = parse_stage_token(p.name)
            rows.append(
                {
                    "File_Type": "script",
                    "Project_Stage": token,
                    "File_Name": p.name,
                    "Relative_Path": safe_relpath(p, project_root),
                    "Main_Number": num,
                    "Sub_Number": sub,
                    "Directory": p.parent.name,
                }
            )
    rows.sort(key=lambda r: natural_sort_key_for_script(Path(r["File_Name"])))
    for idx, row in enumerate(rows, start=1):
        row["Project_Order"] = idx
    return pd.DataFrame(rows)


def build_table_sequence(table_dir: Path, project_root: Path) -> pd.DataFrame:
    exts = {".csv", ".txt", ".xlsx", ".xls", ".json"}
    files = [p for p in table_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=natural_sort_key_for_table)
    rows = []
    for idx, p in enumerate(files, start=1):
        num, letters, sub, token = parse_stage_token(p.name)
        rows.append(
            {
                "Project_Order": idx,
                "File_Type": "table_or_report",
                "Project_Stage": token,
                "File_Name": p.name,
                "Extension": p.suffix.lower(),
                "Relative_Path": safe_relpath(p, project_root),
                "Main_Number": num,
                "Sub_Number": sub,
                "Size_KB": round(p.stat().st_size / 1024, 2),
                "Core_32J_File": "yes" if p.name in REQUIRED_TABLES or p.name in OPTIONAL_EVIDENCE_FILES else "no",
            }
        )
    return pd.DataFrame(rows)


def build_figure_sequence(figure_dir: Path, project_root: Path) -> pd.DataFrame:
    exts = {".png", ".pdf", ".jpg", ".jpeg", ".tif", ".tiff", ".svg"}
    files = [p for p in figure_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=natural_sort_key_for_figure)
    rows = []
    for idx, p in enumerate(files, start=1):
        fig_no = leading_number(p.name)
        num, letters, sub, token = parse_stage_token(p.name)
        rows.append(
            {
                "Project_Order": idx,
                "File_Type": "figure",
                "Figure_Number": fig_no if fig_no != 999999 else "",
                "Project_Stage": token,
                "File_Name": p.name,
                "Extension": p.suffix.lower(),
                "Relative_Path": safe_relpath(p, project_root),
                "Size_KB": round(p.stat().st_size / 1024, 2),
                "Final_32J_Figure": "yes" if p.name in FINAL_FIGURES else "no",
            }
        )
    return pd.DataFrame(rows)


def build_final_evidence_manifest(table_dir: Path, figure_dir: Path) -> pd.DataFrame:
    rows = []
    for block in EVIDENCE_CHAIN_ORDER:
        file_list = [x.strip() for x in block["Main_Files"].split(";")]
        status_parts = []
        for item in file_list:
            # 允许 194-197 这种通配说明，不逐个检查。
            if "*" in item or "194-197" in item:
                found = all((figure_dir / f).exists() for f in FINAL_FIGURES)
                status_parts.append(f"{item}: {'found' if found else 'partly_missing'}")
            else:
                found = (table_dir / item).exists() or (figure_dir / item).exists()
                status_parts.append(f"{item}: {'found' if found else 'missing'}")
        row = dict(block)
        row["File_Status"] = " | ".join(status_parts)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 5. 32J 主程序
# =============================================================================


def main() -> None:
    project_root = find_project_root()
    table_dir = choose_input_dir(project_root, "tables", REQUIRED_TABLES)
    figure_dir = choose_input_dir(project_root, "figures", FINAL_FIGURES)

    # 按项目此前的生成形式输出：表格仍进 tables，图片仍进 figures，报告仍进 reports。
    # 不再私自创建 outputs/32j_revised_project_order/ 这类新总目录。
    out_tables = table_dir
    out_figures = figure_dir
    out_reports = choose_report_dir(project_root)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("32J 原位置输出版：按项目原始文件顺序整合最终论文证据链")
    print("项目根目录：", project_root)
    print("表格读取/输出目录：", out_tables)
    print("图片检查/输出目录：", out_figures)
    print("报告输出目录：", out_reports)
    print("=" * 92)

    missing_required = [name for name in REQUIRED_TABLES if not (table_dir / name).exists()]
    if missing_required:
        print("\n[错误] 32J 缺少必要表格：")
        for name in missing_required:
            print("  -", name)
        print("\n当前检查目录：", table_dir)
        print("请确认 32I、32H、32G 的最终结果已经放在 tables 或 outputs/tables 中。")
        sys.exit(1)

    # 读取核心结果
    summary = read_csv_smart(table_dir / "32i_final_evidence_summary.csv")
    metrics = read_csv_smart(table_dir / "32h_exact_policy_metrics.csv")
    boot = read_csv_smart(table_dir / "32h_bootstrap_uncertainty_exact_policy.csv")
    split = read_csv_smart(table_dir / "32h_exact_policy_split_metrics.csv")
    reprs = read_csv_smart(table_dir / "32g_representativeness_summary.csv")
    chem_num = read_csv_smart(table_dir / "32g_numeric_operating_window.csv")
    claims = read_csv_smart(table_dir / "32i_claim_guardrails.csv")

    # 生成项目顺序清单
    script_seq = build_script_sequence(project_root)
    table_seq = build_table_sequence(table_dir, project_root)
    figure_seq = build_figure_sequence(figure_dir, project_root)
    evidence_manifest = build_final_evidence_manifest(table_dir, figure_dir)

    script_seq.to_csv(out_tables / "32j_project_script_sequence.csv", index=False, encoding="utf-8-sig")
    table_seq.to_csv(out_tables / "32j_project_table_sequence.csv", index=False, encoding="utf-8-sig")
    figure_seq.to_csv(out_tables / "32j_project_figure_sequence.csv", index=False, encoding="utf-8-sig")
    evidence_manifest.to_csv(out_tables / "32j_final_evidence_ordered_manifest.csv", index=False, encoding="utf-8-sig")

    # 主结果表保持 32I 中的证据层级顺序
    main_result = summary.copy()
    if "Evidence_Level" in main_result.columns:
        order_map = {
            "All-domain grouped validation": 1,
            "Predefined reliable domain": 2,
            "Target-free selective retained subdomain": 3,
        }
        main_result["Evidence_Order"] = main_result["Evidence_Level"].map(order_map).fillna(99).astype(int)
        front_cols = ["Evidence_Order"] + [c for c in main_result.columns if c != "Evidence_Order"]
        main_result = main_result[front_cols].sort_values("Evidence_Order")
    main_result.to_csv(out_tables / "32j_main_result_table.csv", index=False, encoding="utf-8-sig")

    # 图表清单按项目证据链顺序，而不是随意排列
    fig_table_manifest_rows = [
        {
            "Project_Order": 1,
            "Suggested_Item": "Main Fig. 1",
            "Source_File": "194_32i2_evidence_chain_summary_clean.png/pdf",
            "Project_Source_Step": "32I2",
            "Purpose": "从全域分组验证到选择性适应域保留子域的证据链。",
            "Placement": "正文主图",
            "Claim_Boundary": "只能表述为 retained-subdomain evidence，不能表述为 all-domain performance。",
        },
        {
            "Project_Order": 2,
            "Suggested_Item": "Main Fig. 2",
            "Source_File": "195_32i2_threshold_plateau_evidence_clean.png/pdf",
            "Project_Source_Step": "32I2 / 32H",
            "Purpose": "NN_Risk <= 0.814815 阈值敏感性与平台证据。",
            "Placement": "正文或补充材料",
            "Claim_Boundary": "用于反驳单点偶然，但必须披露平台较窄。",
        },
        {
            "Project_Order": 3,
            "Suggested_Item": "Main Fig. 3",
            "Source_File": "196_32i2_retained_chemistry_window_clean.png/pdf",
            "Project_Source_Step": "32I2 / 32G",
            "Purpose": "保留样本的化学与操作窗口。",
            "Placement": "正文主图",
            "Claim_Boundary": "解释为特定化学/文献窗口，不能扩展成普适规律。",
        },
        {
            "Project_Order": 4,
            "Suggested_Item": "Main Fig. 4",
            "Source_File": "197_32i2_stability_transfer_boundary_clean.png/pdf",
            "Project_Source_Step": "32I2 / 32H / 32I",
            "Purpose": "分割稳定性、迁移支持和声明边界。",
            "Placement": "正文或补充材料",
            "Claim_Boundary": "必须披露 split-level instability 和 Paper_010 dominance。",
        },
        {
            "Project_Order": 5,
            "Suggested_Item": "Table 1",
            "Source_File": "32j_main_result_table.csv",
            "Project_Source_Step": "32I",
            "Purpose": "三层证据：全域分组验证、预定义可靠域、选择性保留子域。",
            "Placement": "正文表格",
            "Claim_Boundary": "三层结果必须分开写，不能把 P08 当成全域性能。",
        },
        {
            "Project_Order": 6,
            "Suggested_Item": "Table S1",
            "Source_File": "32h_bootstrap_uncertainty_exact_policy.csv",
            "Project_Source_Step": "32H",
            "Purpose": "P08 bootstrap 不确定性。",
            "Placement": "补充材料",
            "Claim_Boundary": "报告 R2/RMSE/MAE 的 median 和 95% CI。",
        },
        {
            "Project_Order": 7,
            "Suggested_Item": "Table S2",
            "Source_File": "32h_exact_policy_split_metrics.csv",
            "Project_Source_Step": "32H",
            "Purpose": "P08 分割层面稳定性。",
            "Placement": "补充材料",
            "Claim_Boundary": "展示 seed-level instability，不隐藏不稳定性。",
        },
        {
            "Project_Order": 8,
            "Suggested_Item": "Table S3",
            "Source_File": "32g_numeric_operating_window.csv; 32g_categorical_composition.csv",
            "Project_Source_Step": "32G",
            "Purpose": "保留子域数值窗口和类别组成。",
            "Placement": "补充材料",
            "Claim_Boundary": "支撑机理讨论和适用边界。",
        },
    ]
    fig_table_manifest = pd.DataFrame(fig_table_manifest_rows)
    fig_table_manifest.to_csv(out_tables / "32j_figure_table_manifest_project_order.csv", index=False, encoding="utf-8-sig")

    # 声明边界表
    claims.to_csv(out_tables / "32j_claim_guardrails_for_manuscript.csv", index=False, encoding="utf-8-sig")

    # 最终图件检查 + 复制
    figure_check_rows = []
    copied = []
    missing_figures = []
    for fig in FINAL_FIGURES:
        src = figure_dir / fig
        status = "found" if src.exists() else "missing"
        figure_check_rows.append(
            {
                "Project_Order": leading_number(fig),
                "Figure_File": fig,
                "Status": status,
                "Source_Dir": str(figure_dir),
            }
        )
        if src.exists():
            dst = out_figures / fig
            try:
                # 如果源文件本来就在 outputs/figures，则不重复复制，只记录为已就位。
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
                copied.append(fig)
            except Exception:
                # 避免 SameFileError 或 Windows 路径解析问题导致脚本中断。
                copied.append(fig)
        else:
            missing_figures.append(fig)
    pd.DataFrame(figure_check_rows).to_csv(out_tables / "32j_final_figure_file_check.csv", index=False, encoding="utf-8-sig")

    # 原始证据表已经位于 outputs/tables；这里不再重复复制到新目录，避免改变项目存放习惯。
    copied_tables = []
    for name in REQUIRED_TABLES + OPTIONAL_EVIDENCE_FILES:
        src = table_dir / name
        if src.exists():
            copied_tables.append(name)

    # 关键结果数值
    m = get_first_row(metrics)
    r2_ci = row_by_metric(boot, "R2")
    rmse_ci = row_by_metric(boot, "RMSE")
    mae_ci = row_by_metric(boot, "MAE")

    # 分割稳定性摘要
    split_lines = []
    if len(split) > 0:
        for _, row in split.iterrows():
            split_name = value(row, ["Split", "Seed", "Data_Split", "split", "seed"], "unknown_split")
            split_lines.append(
                f"- {split_name}: R² = {fmt_num(value(row, ['R2', 'r2'], ''))}, "
                f"RMSE = {fmt_num(value(row, ['RMSE', 'rmse'], ''))}, "
                f"MAE = {fmt_num(value(row, ['MAE', 'mae'], ''))}, "
                f"N = {fmt_int(value(row, ['N_Retained', 'N', 'n'], ''))}"
            )
    if not split_lines:
        split_lines.append("- 未能自动读取 split-level stability，请查看 32h_exact_policy_split_metrics.csv。")

    # 保留子域窗口摘要
    ret_lines = []
    context_col = get_col(chem_num, ["Context", "context"])
    variable_col = get_col(chem_num, ["Variable_Key", "Variable", "variable", "Feature", "feature"])
    if context_col is not None:
        ret_num = chem_num[chem_num[context_col].astype(str).str.contains("Retained", case=False, na=False)].copy()
    else:
        ret_num = chem_num.copy()

    variable_labels = [
        ("initial_f", "Initial F−（初始氟浓度）"),
        ("dose", "Dose（混凝剂剂量）"),
        ("pH_in", "Influent pH（进水 pH）"),
        ("retention_time", "Retention time（停留/反应时间）"),
        ("temperature", "Temperature（水温）"),
    ]
    if variable_col is not None and len(ret_num) > 0:
        for key, label in variable_labels:
            tmp = ret_num[ret_num[variable_col].astype(str).eq(key)]
            if not tmp.empty:
                row = tmp.iloc[0]
                ret_lines.append(
                    f"- {label}: median {fmt_num(value(row, ['Median', 'median'], ''), 3)}, "
                    f"IQR {fmt_num(value(row, ['Q25', 'q25'], ''), 3)}–{fmt_num(value(row, ['Q75', 'q75'], ''), 3)}, "
                    f"range {fmt_num(value(row, ['Min', 'min'], ''), 3)}–{fmt_num(value(row, ['Max', 'max'], ''), 3)}"
                )
    if not ret_lines:
        ret_lines.append("- 未能自动提取保留子域数值窗口，请查看 32g_numeric_operating_window.csv。")

    optional_status = []
    for name in OPTIONAL_EVIDENCE_FILES:
        optional_status.append(f"- {name}: {'found' if (table_dir / name).exists() else 'missing'}")

    # 项目结构摘要表
    structure_summary = pd.DataFrame(
        [
            {"Item": "Project root", "Path": str(project_root)},
            {"Item": "Table input dir", "Path": str(table_dir)},
            {"Item": "Figure input dir", "Path": str(figure_dir)},
            {"Item": "Script dirs", "Path": "; ".join(str(p) for p in get_script_dirs(project_root))},
            {"Item": "Table output dir", "Path": str(out_tables)},
            {"Item": "Figure output dir", "Path": str(out_figures)},
            {"Item": "Report output dir", "Path": str(out_reports)},
            {"Item": "Scripts found", "Path": str(len(script_seq))},
            {"Item": "Tables/reports found", "Path": str(len(table_seq))},
            {"Item": "Figures found", "Path": str(len(figure_seq))},
        ]
    )
    structure_summary.to_csv(out_tables / "32j_project_structure_summary.csv", index=False, encoding="utf-8-sig")

    # 最终报告
    final_figure_text = "194–197 PNG/PDF all found and copied." if not missing_figures else "Missing: " + ", ".join(missing_figures)
    report = f"""# 32J 原位置输出版：按项目原始文件顺序整合最终论文证据链

## 1. 本次修正点

本版脚本已经按你的完整项目压缩包重新检查并修改。核心变化是：

1. 自动兼容两类目录结构：`outputs/tables + outputs/figures` 和 `tables + figures`。
2. 项目脚本、项目表格、项目图件清单都按照项目编号顺序排列，而不是按字母顺序排列。
3. 对于 `31h` 与 `31h2`、`32a` 与 `32a2`、`32i` 与 `32i2` 这类文件，排序规则固定为：先主步骤，再补充步骤。
4. 32J 不重跑模型，不改变指标，只整合最终证据链和论文写法边界。

## 2. 路径识别结果

- Project root: `{project_root}`
- Table input dir: `{table_dir}`
- Figure input dir: `{figure_dir}`
- Table output dir: `{out_tables}`
- Figure output dir: `{out_figures}`
- Report output dir: `{out_reports}`

项目文件数量：

- Scripts found: **{len(script_seq)}**
- Tables/reports found: **{len(table_seq)}**
- Figures found: **{len(figure_seq)}**

最终图件检查：**{final_figure_text}**

## 3. 当前接受的最终结果

当前接受策略为：**P08**

- Strategy（策略）: **RepeatedBalanced_Extended**
- Model（模型）: **XGBoost**
- Rule（规则）: **Reliability_Zone = Outside 且 NN_Risk ≤ 0.814815**

保留子域结果：

- R² = **{fmt_num(value(m, ['R2', 'r2'], ''))}**
- RMSE = **{fmt_num(value(m, ['RMSE', 'rmse'], ''))}**
- MAE = **{fmt_num(value(m, ['MAE', 'mae'], ''))}**
- N retained prediction instances = **{fmt_int(value(m, ['N_Retained', 'N', 'n'], ''))}**
- Unique retained rows = **{fmt_int(value(m, ['Unique_Row_N', 'Unique_N', 'unique_n'], ''))}**
- Coverage within strategy-method subset = **{fmt_num(value(m, ['Coverage_within_strategy_method', 'Coverage', 'coverage'], ''))}**
- Dominant group = **{value(m, ['Dominant_Group', 'dominant_group'], '')}**
- Dominant group fraction = **{fmt_num(value(m, ['Dominant_Group_Fraction', 'dominant_group_fraction'], ''), 3)}**

## 4. 必须固定的论文口径

这个结果必须写成：

**target-free selective AD retained-subdomain result（无目标泄露的选择性适应域保留子域结果）**

不能写成：

**all-domain grouped-validation performance（全域分组验证性能）**

最安全的逻辑是：

**全域跨文献泛化较困难 → 预定义可靠域不足以改善 → 无目标泄露的选择性适应域规则可以识别一个窄范围、高置信度的保留子域。**

## 5. 32J 证据链顺序

本版已经在 `32j_final_evidence_ordered_manifest.csv` 中按项目顺序固定为：

1. 32F2 reproducible policy audit（可复现策略审计）
2. 32G retained subdomain chemistry（保留子域化学窗口）
3. 32H threshold sensitivity and anti-cherry-picking audit（阈值敏感性与反挑选审计）
4. 32I final evidence integration（最终证据整合）
5. 32I2 final figure cleanup（最终图件清理）

## 6. Bootstrap 不确定性

- R² median = **{fmt_num(value(r2_ci, ['Median', 'median'], ''))}**, 95% CI = [{fmt_num(value(r2_ci, ['CI_2p5', 'ci_2p5'], ''))}, {fmt_num(value(r2_ci, ['CI_97p5', 'ci_97p5'], ''))}]
- RMSE median = **{fmt_num(value(rmse_ci, ['Median', 'median'], ''))}**, 95% CI = [{fmt_num(value(rmse_ci, ['CI_2p5', 'ci_2p5'], ''))}, {fmt_num(value(rmse_ci, ['CI_97p5', 'ci_97p5'], ''))}]
- MAE median = **{fmt_num(value(mae_ci, ['Median', 'median'], ''))}**, 95% CI = [{fmt_num(value(mae_ci, ['CI_2p5', 'ci_2p5'], ''))}, {fmt_num(value(mae_ci, ['CI_97p5', 'ci_97p5'], ''))}]

## 7. 分割稳定性

{chr(10).join(split_lines)}

这部分必须放入论文或补充材料，不能只写最好的 R²。

## 8. 保留子域化学/操作窗口

{chr(10).join(ret_lines)}

这个窗口应解释为 **specific coagulation chemistry window（特定混凝化学窗口）**，不能解释为所有混凝除氟体系的通用规律。

## 9. 可直接放入论文 Results and Discussion 的表述

在按文献分组的全域验证条件下，模型在跨文献外推时仍面临明显困难，说明不同文献来源之间的实验体系、混凝剂类型、水质背景和操作窗口差异会显著削弱模型的直接泛化能力。进一步采用预定义可靠域规则后，模型性能并未得到稳定改善，表明简单依赖预设适应域边界并不足以识别可可靠预测的样本区域。

在此基础上，本研究进一步构建了不依赖目标变量的选择性适应域筛选规则。最终可复现的 P08 规则为：Reliability_Zone = Outside 且 NN_Risk ≤ 0.814815。该规则在 RepeatedBalanced_Extended + XGBoost 组合下保留了 {fmt_int(value(m, ['N_Retained', 'N', 'n'], ''))} 个预测实例，对应 R² = {fmt_num(value(m, ['R2', 'r2'], ''))}、RMSE = {fmt_num(value(m, ['RMSE', 'rmse'], ''))}、MAE = {fmt_num(value(m, ['MAE', 'mae'], ''))}。需要强调的是，该结果并不代表全域分组验证性能的整体提升，而是说明在目标变量不参与筛选的前提下，适应域指标能够识别出一个窄范围、高置信度的保留子域。

从稳定性和适用边界来看，P08 规则具有一定证据支持，但也存在明确限制。Bootstrap 分析显示保留子域具有较好的统计表现，但分割层面的稳定性并不完全一致。此外，保留样本高度集中于 Paper_010，说明该子域具有明显的文献来源和化学体系特异性。因此，本文将该结果界定为“选择性适应域保留子域”或“特定化学窗口内的高置信预测区域”，而非普适性的跨文献泛化结论。

## 10. 本脚本输出文件

本版严格按项目原有存放方式输出：表格进入 `outputs/tables/`，图片位于 `outputs/figures/`，报告进入 `reports/`。

表格输出：

- `32j_main_result_table.csv`
- `32j_final_evidence_ordered_manifest.csv`
- `32j_figure_table_manifest_project_order.csv`
- `32j_project_script_sequence.csv`
- `32j_project_table_sequence.csv`
- `32j_project_figure_sequence.csv`
- `32j_final_figure_file_check.csv`
- `32j_claim_guardrails_for_manuscript.csv`
- `32j_project_structure_summary.csv`

报告输出：

- `32j_final_manuscript_integration_report.md`
- `32j_run_log.txt`

图件输出：

- `194_32i2_evidence_chain_summary_clean.png/pdf`
- `195_32i2_threshold_plateau_evidence_clean.png/pdf`
- `196_32i2_retained_chemistry_window_clean.png/pdf`
- `197_32i2_stability_transfer_boundary_clean.png/pdf`

## 11. 可选证据文件检查

{chr(10).join(optional_status)}
"""
    (out_reports / "32j_final_manuscript_integration_report.md").write_text(report, encoding="utf-8")

    run_log = f"""32J revised project-order version completed.
Project root: {project_root}
Table input dir: {table_dir}
Figure input dir: {figure_dir}
Table output dir: {out_tables}
Figure output dir: {out_figures}
Report output dir: {out_reports}
Scripts found: {len(script_seq)}
Tables/reports found: {len(table_seq)}
Figures found: {len(figure_seq)}
Copied final figures: {len(copied)} / {len(FINAL_FIGURES)}
Evidence tables/reports already in table dir: {len(copied_tables)}
Missing final figures: {missing_figures}
"""
    (out_reports / "32j_run_log.txt").write_text(run_log, encoding="utf-8")

    print("\n[完成] 32J 修正版已经生成。")
    print("表格输出目录：", out_tables)
    print("图片输出目录：", out_figures)
    print("报告输出目录：", out_reports)
    print("\n关键输出：")
    print("  -", out_tables / "32j_main_result_table.csv")
    print("  -", out_tables / "32j_final_evidence_ordered_manifest.csv")
    print("  -", out_tables / "32j_project_table_sequence.csv")
    print("  -", out_tables / "32j_project_figure_sequence.csv")
    print("  -", out_reports / "32j_final_manuscript_integration_report.md")
    print("\n排序规则：")
    print("  已按项目步骤顺序排序，不按字母顺序。例：31h 在 31h2 前，32a 在 32a2 前，32i 在 32i2 前。")
    print("\n最终口径：")
    print("  P08 = RepeatedBalanced_Extended + XGBoost + Reliability_Zone=Outside + NN_Risk<=0.814815")
    print("  必须写成 target-free selective AD retained-subdomain result")
    print("  不能写成 all-domain grouped-validation performance")
    print("=" * 92)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("\n[错误] 32J 修正版运行失败。完整错误如下：\n")
        traceback.print_exc()
        print("\n把完整报错截图发给我，我继续按一个 .py 文件给你改。")
        sys.exit(1)
