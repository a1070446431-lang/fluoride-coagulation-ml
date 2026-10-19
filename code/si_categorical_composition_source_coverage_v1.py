# -*- coding: utf-8 -*-
"""Supplementary categorical composition and source coverage (论文1).

Place this complete file in fluoride_coagulation_ml_project/src/ and run it.
Dependencies: Python >= 3.9, numpy, pandas, matplotlib (no model training).
Uses config.py for project/data paths and the formal Figure 4 v9 package paths:
outputs/jece_final_package/{figures,tables,reports,logs}/.
Outputs: full category tables, source cross-tabs, four PNG/PDF figures,
a bilingual supplementary-text draft, and a reproducibility JSON log.
No final Table S/Figure S numbers are assigned because older manifests conflict.
Data are read only. Labels, source groups and model results are not recoded.
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
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

STEM = "SI_categorical_composition_source_coverage_v1"
N_EXPECTED, SOURCES_EXPECTED = 1011, 27
VARIABLES = {"氟形态": "Fluoride form", "干扰离子": "Interferent label", "混凝剂": "Coagulant"}
KEYS = {"氟形态": "fluoride_form", "干扰离子": "interferent", "混凝剂": "coagulant"}
PREFIXES = {"氟形态": "F", "干扰离子": "I", "混凝剂": "C"}

# Display translations only: never used for grouping or model input.
DISPLAY = {
    "氟离子": "Free fluoride", "氟离子+络合氟": "Free + complexed fluoride", "络合氟": "Complexed fluoride",
    "无": 'Recorded as "none"', "聚合氯化铝": "Polyaluminum chloride (PAC)",
    "硫酸铝": "Aluminum sulfate", "锆基复合混凝剂（ZXC）": "Zr-based composite (ZXC)",
    "聚合硫酸铁": "Polyferric sulfate", "氯化铝": "Aluminum chloride",
    "聚硅酸铁钙镧复合混凝剂（PSFCL）": "Fe-Ca-La polysilicate (PSFCL)",
    "水合氧化锆": "Hydrous zirconium oxide", "氯化钙": "Calcium chloride",
    "铝基废渣复合混凝剂（AW-CF）": "Al-waste composite (AW-CF)",
    "聚合硅酸锆（PZS）": "Polyzirconium silicate (PZS)",
    "聚合氯化铝铁": "Polyaluminum ferric chloride (PAFC)",
    "聚合氯化铝Ⅰ": "PAC I", "聚合氯化铝Ⅱ": "PAC II", "明矾": "Alum",
    "高铁酸钠": "Sodium ferrate", "氢氧化钙+聚合氯化铝": "Calcium hydroxide + PAC",
    "硫酸铝（CSAF单铝体系）": "Aluminum sulfate (CSAF Al-only)",
    "硫酸铝（CSAM单铝体系）": "Aluminum sulfate (CSAM Al-only)",
    "聚合氯化铝（单Al）": "PAC (Al-only)", "聚合氯化铝铁（Al/Fe）": "PAFC (Al/Fe)",
    "三氯化铁": "Ferric chloride", "氢氧化钙+三氯化铁": "Calcium hydroxide + ferric chloride",
    "商业复合除氟混凝剂（神美除氟剂A）": "Commercial defluoridant (Shenmei A)",
    "聚合氯化铝钛（PATC）": "Polyaluminum titanium chloride (PATC)",
    "氧化钙": "Calcium oxide", "氢氧化钙": "Calcium hydroxide",
    "氢氧化钙+聚合硫酸铁": "Calcium hydroxide + polyferric sulfate",
    "聚硅酸铁（PSF）": "Polyferric silicate (PSF)",
    "聚硅酸铁（PSI）": "Polyferric silicate (PSI)",
    "聚硅酸铁复合混凝剂（PSFC）": "Polyferric silicate composite (PSFC)",
    "含氟废水用聚硅酸铝铁复合混凝剂（FD-PSAF）": "Fluoride-wastewater composite (FD-PSAF)",
    "氢氧化铝": "Aluminum hydroxide", "氧化铁": "Iron oxide",
    "氯化钙+氢氧化钙": "Calcium chloride + calcium hydroxide",
    "氯化锆": "Zirconium chloride", "聚合氯化铝S型": "PAC S-type",
    "聚合铁铝硅复合混凝剂（PFASiC）": "Fe-Al-Si composite (PFASiC)",
    "聚硅酸铁/聚硅酸铁复合/聚硅酸铁钙镧体系": "PSF / PSFC / Fe-Ca-La system",
    "聚硅酸铁/聚硅酸铁复合混凝剂（PSF/PSFC）": "PSF / PSFC",
    "聚硅酸铝铁（PSAF）": "Polyaluminum ferric silicate (PSAF)",
    "铝铁复合电极": "Al-Fe composite electrode",
    "腐殖酸/海藻酸": "Humic acid / alginic acid",
    "腐殖酸/NO₃⁻/CO₃²⁻": "Humic acid / NO3(-) / CO3(2-)",
    "腐殖酸/BSA": "Humic acid / BSA",
    "腐殖酸/海藻酸/Ca²⁺": "Humic acid / alginic acid / Ca(2+)",
    "腐殖酸": "Humic acid", "天然有机物/Cl⁻/HCO₃⁻": "Natural organic matter / Cl(-) / HCO3(-)",
    "油类有机物/Cl⁻/SO₄²⁻": "Oily organics / Cl(-) / SO4(2-)",
}
for _ratio in ["1:1", "4:1", "6.5:1", "9:1", "14:1"]:
    DISPLAY[f"复合硫酸铝铁（CSAF，Al/Fe={_ratio}）"] = f"CSAF (Al/Fe = {_ratio})"
    DISPLAY[f"复合硫酸铝镁（CSAM，Al/Mg={_ratio}）"] = f"CSAM (Al/Mg = {_ratio})"
IONS = {"SO₄²⁻": "SO4(2-)", "CO₃²⁻": "CO3(2-)", "Ca²⁺": "Ca(2+)", "Mg²⁺": "Mg(2+)",
        "Cl⁻": "Cl(-)", "HCO₃⁻": "HCO3(-)", "PO₄³⁻": "PO4(3-)", "NO₃⁻": "NO3(-)",
        "SiO₃²⁻": "SiO3(2-)", "SiO₂": "SiO2"}


def display_label(raw):
    if raw == "":
        return "Missing (empty cell)"
    if raw in DISPLAY:
        return DISPLAY[raw]
    result = raw
    for a, b in IONS.items():
        result = result.replace(a, b)
    if any("\u4e00" <= ch <= "\u9fff" for ch in result):
        raise ValueError(f"Missing display translation for category {raw!r}; add it to DISPLAY, do not merge labels.")
    return result.replace("/", " / ")


def project_paths():
    root = Path(__file__).resolve().parents[1]
    config_file = root / "src" / "config.py"
    if not config_file.is_file():
        raise FileNotFoundError(f"Place this script inside the project src/ folder. Required config: {config_file}")
    spec = importlib.util.spec_from_file_location("_si_project_config", config_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configured_root = Path(module.PROJECT_ROOT).resolve()
    if configured_root != root:
        raise ValueError(f"config.py PROJECT_ROOT mismatch: {configured_root}; script project: {root}")
    processed = Path(module.PROCESSED_DIR)
    if not processed.is_absolute():
        processed = root / processed
    # Formal Figure 4 v9 uses this package, distinct from legacy config FIG_DIR.
    package = root / "outputs" / "jece_final_package"
    outputs = {key: package / key for key in ("figures", "tables", "reports", "logs")}
    inputs = [processed / "model_ready_v1.csv", processed / "model_ready_v1_with_source.csv"]
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required inputs missing; no fallback or synthetic data is allowed:\n" + "\n".join(missing))
    return root, config_file, inputs, outputs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checked(inputs):
    # Preserve literal labels such as '无' and 'NA'; an empty field alone is missing.
    main, data = [pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig") for p in inputs]
    if main.shape != (N_EXPECTED, 26) or data.shape != (N_EXPECTED, 27):
        raise ValueError(f"Expected storage matrices 1011 x 26 / 1011 x 27; got {main.shape} / {data.shape}")
    required = [*VARIABLES, "原始行号"]
    for df in (main, data):
        if not set(required).issubset(df.columns):
            raise ValueError(f"Required columns missing: {set(required) - set(df.columns)}")
        if (df['原始行号'].str.strip() == '').any() or not df['原始行号'].is_unique:
            raise ValueError("原始行号 must be present and unique for each record.")
    if set(main['原始行号']) != set(data['原始行号']):
        raise ValueError("The two storage matrices contain different raw row IDs.")
    main = main.set_index('原始行号').loc[data['原始行号']]
    for column in VARIABLES:
        if main[column].tolist() != data[column].tolist():
            raise ValueError(f"Categorical values differ between storage matrices: {column}")
        if (data[column] != data[column].str.strip()).any():
            raise ValueError(f"Leading/trailing whitespace in {column}; review inputs before category counting.")
    if "文献编号" not in data.columns or (data['文献编号'].str.strip() == '').any():
        raise ValueError("Source IDs must be present for every record.")
    if (data['文献编号'] != data['文献编号'].str.strip()).any():
        raise ValueError("Whitespace in source IDs; no silent relabeling is allowed.")
    if data['文献编号'].nunique() != SOURCES_EXPECTED:
        raise ValueError(f"Expected 27 distinct source groups, got {data['文献编号'].nunique()}")
    return data


def compute_tables(data):
    sources = sorted(data['文献编号'].unique())
    source_sizes = data.groupby('文献编号').size().reindex(sources)
    details, long_rows, summaries, matrices = [], [], [], {}
    for variable in VARIABLES:
        count = data.groupby(variable, sort=False).size()
        levels = sorted(count.index, key=lambda level: (-int(count[level]), level))
        matrix = pd.crosstab(data[variable], data['文献编号']).reindex(index=levels, columns=sources, fill_value=0)
        if matrix.to_numpy().sum() != len(data) or not (matrix.sum(axis=0) == source_sizes).all():
            raise AssertionError(f"Cross-tab total mismatch: {variable}")
        rows = []
        for rank, category in enumerate(levels, start=1):
            counts = matrix.loc[category]
            n, k = int(counts.sum()), int((counts > 0).sum())
            maximum = int(counts.max())
            record = dict(variable=KEYS[variable], variable_original=variable,
                category_id=f"{PREFIXES[variable]}{rank:02d}", category_original=category,
                category_display=display_label(category), label_status="empty_cell" if category == "" else ("literal_none" if category == "无" else "recorded"),
                n_records=n, pct_all_records=100*n/len(data), n_sources=k,
                pct_all_sources=100*k/len(sources), single_source=(k == 1),
                dominant_source_n=maximum, dominant_source_pct=100*maximum/n,
                dominant_sources=";".join(counts.index[counts == maximum]),
                all_sources=";".join(counts.index[counts > 0]))
            details.append(record); rows.append(record)
            for source in sources:
                nij = int(counts[source])
                long_rows.append(dict(variable=KEYS[variable], category_id=record['category_id'],
                    category_original=category, source_id=source, n_records=nij,
                    pct_within_category=100*nij/n,
                    pct_within_source=100*nij/int(source_sizes[source])))
        tab = pd.DataFrame(rows)
        if tab.n_records.sum() != len(data) or not np.isclose(tab.pct_all_records.sum(), 100):
            raise AssertionError(f"Category composition mismatch: {variable}")
        summaries.append(dict(variable=KEYS[variable], variable_original=variable,
            n_records=len(data), n_sources=len(sources), n_levels=len(tab),
            n_single_source_levels=int(tab.single_source.sum()),
            n_records_in_single_source_levels=int(tab.loc[tab.single_source, 'n_records'].sum()),
            n_empty_cells=int((data[variable] == '').sum()),
            n_literal_none=int((data[variable] == '无').sum())))
        matrix.index = tab.category_id.to_numpy()
        matrix.index.name = 'category_id'
        matrices[variable] = matrix
    return pd.DataFrame(details), pd.DataFrame(long_rows), pd.DataFrame(summaries), matrices, source_sizes


def create_figures(details, outputs):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.titlesize': 11, 'axes.labelsize': 10, 'xtick.labelsize': 9,
        'ytick.labelsize': 8.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False})
    written = []
    for variable, title in VARIABLES.items():
        tab = details.loc[details.variable == KEYS[variable]].reset_index(drop=True)
        parts = [tab.iloc[i:i+27] for i in range(0, len(tab), 27)]
        for number, sub in enumerate(parts, start=1):
            n = len(sub)
            fig = plt.figure(figsize=(12, max(3.6, 1.8 + 0.30*n)))
            grid = fig.add_gridspec(1, 2, left=.405, right=.967, bottom=max(.10, .73/fig.get_figheight()),
                                   top=1-.73/fig.get_figheight(), width_ratios=[1.85, 1], wspace=.34)
            ax, bx = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
            y = np.arange(n)
            colors = np.where(sub.single_source, '#B7503C', '#347C9E')
            ax.barh(y, sub.n_records, color=colors, height=.64)
            maximum = float(tab.n_records.max())
            ax.set_xlim(0, maximum*1.49)
            ax.set_yticks(y, [f"{row.category_id}  {row.category_display}" for row in sub.itertuples()])
            for yy, row in zip(y, sub.itertuples()):
                ax.text(row.n_records + maximum*.022, yy, f"{row.n_records} ({row.pct_all_records:.1f}%)", va='center', fontsize=8.5)
            ax.set_xlabel('Records (n); labels: n (% of 1,011)')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.set_title('(a) Sample composition', loc='left', pad=12)
            bx.hlines(y, 0, sub.n_sources, color=colors, lw=1.4, alpha=.7)
            bx.scatter(sub.n_sources, y, c=colors, s=24, zorder=3)
            for yy, value in zip(y, sub.n_sources):
                bx.text(value+.8, yy, str(value), va='center', fontsize=9)
            bx.set_xlim(0, 30); bx.set_xticks([0, 5, 10, 15, 20, 27])
            bx.set_yticks(y, ['']*n); bx.tick_params(axis='y', length=0)
            bx.set_xlabel('Distinct sources (out of 27)')
            bx.set_title('(b) Source coverage', loc='left', pad=12)
            for axis in (ax, bx):
                axis.set_ylim(n-.5, -.5); axis.grid(axis='x', color='#E3E6E8', linewidth=.5)
                axis.set_axisbelow(True)
            suffix = f' — part {number}/{len(parts)}' if len(parts)>1 else ''
            fig.text(.035, .975, f'{title}: recorded categories{suffix}', fontsize=13, weight='bold', va='top')
            fig.text(.035, .025, 'Red: category occurs in one source. Blue: category occurs in multiple sources.\n'
                     'Stored labels are preserved; categories are not merged. Source coverage is descriptive.', fontsize=8.5, va='bottom')
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            # Catch clipping and long category labels intruding into the left chart.
            for label in ax.get_yticklabels():
                bounds = label.get_window_extent(renderer)
                if bounds.x0 < 0 or bounds.x1 > ax.bbox.x0-1:
                    raise RuntimeError(f"Category label does not fit: {label.get_text()}")
            stem = outputs['figures'] / f'{STEM}_{KEYS[variable]}_part{number}'
            for extension in ('png', 'pdf'):
                path = stem.with_suffix('.'+extension)
                fig.savefig(path, dpi=600)
                written.append(path)
            plt.close(fig)
    return written


def make_report(details, summaries):
    lines = ['# Supplementary draft: categorical composition and source coverage', '',
        '## 方法与统计口径', '',
        '本项仅统计当前1011条记录、27个来源组中的氟形态、干扰物标签和混凝剂类别。'
        '逐行核对严格存储矩阵与带来源矩阵中的原始行号和三个类别字段后，按存储标签的精确取值统计，'
        '不合并配方、比例、复合标签或同义名称。26列是存储结构，不代表26个预测变量。'
        '文献编号仅用于本项来源覆盖审计，不作为模型预测输入。', '',
        '各类别样本比例以1011条记录为分母，来源覆盖比例以27个来源组为分母；'
        '最大来源占比以该类别的记录数为分母。并列最大来源全部列出。'
        '空字段与字面标签“无”分别统计；“无”仅表示当前矩阵的记录标签，不单独证明实验体系中完全不存在干扰物或药剂。'
        '含斜杠的标签按整体类别保留，不据此推断每种组分在每次实验中同时出现。', '',
        '## 可插入补充材料的结果文字', '']
    for row in summaries.itertuples():
        tab = details.loc[details.variable == row.variable]
        top = tab.iloc[0]
        lines.append(f'{row.variable_original}包含{row.n_levels}个存储类别，其中{row.n_single_source_levels}个类别'
            f'仅出现在一个来源组中，涉及{row.n_records_in_single_source_levels}条记录'
            f'（{100*row.n_records_in_single_source_levels/row.n_records:.2f}%）。'
            f'样本最多的类别为“{top.category_original}”，含{top.n_records}条记录'
            f'（{top.pct_all_records:.2f}%），覆盖{top.n_sources}个来源组。'
            f'空字段为{row.n_empty_cells}条，字面标签“无”为{row.n_literal_none}条。')
        lines.append('')
    lines.extend(['上述统计表明，较大的样本量并不必然对应广泛的来源覆盖。'
        '单来源类别提示该标签在其他文献中的数据支持不足，可为来源隔离验证和未见类别诊断提供背景，'
        '但不能单凭这一统计确定预测误差、因果机制或可靠适用域。精确标签形成的类别数也不等同于独立化学体系数。', '',
        '## Figure caption (English)', '',
        'Categorical composition and source coverage in the curated dataset (1,011 records from 27 source groups). '
        'For each recorded category, panel (a) shows the record count and percentage of all records; panel (b) shows '
        'the number of distinct source groups. Red indicates categories represented by a single source; blue indicates '
        'categories represented by multiple sources. Coagulant categories are displayed in two parts with a shared '
        'count scale. Categories are ordered by descending record count; ties are resolved by the original label. '
        'Display translations do not modify the stored categories. The complete original labels and source cross-tabulations '
        'are supplied in the accompanying tables. Source identifiers are used only for coverage auditing.', '',
        '## Table definitions / 表格说明', '',
        '- category_summary.csv：81个类别的原名、显示名、记录数、记录占比、来源数、来源占比、最大来源占比及全部来源。',
        '- source_cross_long.csv：类别×来源的完整交叉表，保留零计数；包含类别内比例和来源内比例。',
        '- *_source_counts.csv：三个类别变量各自的宽格式来源计数矩阵，行ID对应category_summary.csv。',
        '- variable_summary.csv：类别数、单来源类别数及其覆盖记录数、空字段与字面“无”计数。',
        '- source_sizes.csv：27个来源组的记录数；source_sizes之和为1011。', '',
        '本次文件不指定最终Table S/Figure S序号；应在现有补充材料合并阶段统一编号。'
        '本统计不读取目标值或预测误差，不变更模型、适用域阈值、正文或正式主图。', ''])
    return '\n'.join(lines).replace('81个类别', f'{len(details)}个类别')


def main():
    root, config_file, inputs, outputs = project_paths()
    hashes_before = {str(p.relative_to(root)): digest(p) for p in inputs}
    data = load_checked(inputs)
    details, cross, summaries, matrices, source_sizes = compute_tables(data)
    # Validate all labels before creating any deliverable.
    for label in details.category_original:
        display_label(label)
    for directory in outputs.values():
        directory.mkdir(parents=True, exist_ok=True)
    products = []
    tables = {'category_summary': details, 'source_cross_long': cross, 'variable_summary': summaries,
              'source_sizes': source_sizes.rename('n_records').rename_axis('source_id').reset_index()}
    for key, table in tables.items():
        path = outputs['tables'] / f'{STEM}_{key}.csv'
        table.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.6f'); products.append(path)
    for variable, matrix in matrices.items():
        path = outputs['tables'] / f'{STEM}_{KEYS[variable]}_source_counts.csv'
        matrix.to_csv(path, encoding='utf-8-sig'); products.append(path)
    products.extend(create_figures(details, outputs))
    report = outputs['reports'] / f'{STEM}_supplementary_text.md'
    report.write_text(make_report(details, summaries), encoding='utf-8'); products.append(report)
    hashes_after = {str(p.relative_to(root)): digest(p) for p in inputs}
    if hashes_after != hashes_before:
        raise RuntimeError('Input file changed during execution; audit failed.')
    audit = {'script': Path(__file__).name, 'script_sha256': digest(Path(__file__)),
        'config_sha256': digest(config_file), 'input_sha256': hashes_before, 'inputs_unchanged': True,
        'storage_shapes': [[1011,26],[1011,27]], 'n_source_groups': int(data['文献编号'].nunique()),
        'versions': {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'checks': ['unique raw row IDs', 'category equality across matrices by row ID', '27 complete source groups',
                   'category and source cross-tab totals equal 1011', 'display label bounds', 'input SHA256 unchanged'],
        'output_sha256': {str(p.relative_to(root)): digest(p) for p in products}}
    log = outputs['logs'] / f'{STEM}_audit.json'
    log.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summaries.to_string(index=False))
    print(f'Created {len(products)} deliverables + audit log under: {outputs["figures"].parent}')
    print('Inputs unchanged; original category labels and source IDs preserved.')


if __name__ == '__main__':
    started = time.perf_counter()
    try:
        main()
    finally:
        print(f'Total runtime: {time.perf_counter() - started:.2f} s')
