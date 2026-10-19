# -*- coding: utf-8 -*-
"""
32V EST-style storyline figure-suite generator and DOCX inserter
Project: fluoride coagulation defluoridation ML project

Purpose
-------
This script creates an EST-style figure suite from the previous project figures
and inserts the new polished figures into the manuscript draft.

What it does
------------
1. Keeps the original project output convention:
      outputs/figures/
      outputs/tables/
      reports/
2. Builds four EST-style composite figures:
      201: overall paper storyline / framework
      202: random split + model-screening evidence
      203: grouped validation + generalization boundary evidence
      204: selective AD + chemistry-window evidence
3. Inserts the suite into the manuscript near "3. Results and Discussion".
4. Writes an article-mainline report and figure manifest.

Important writing boundary
--------------------------
R2=0.747676 belongs ONLY to the target-free selective AD retained-subdomain result.
It must NOT be described as all-domain grouped validation performance.

This script does NOT retrain any model and does NOT modify final metrics.
"""

from __future__ import annotations

import csv
import os
import re
import sys
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
except Exception as e:
    print("[ERROR] matplotlib is required. Install it by: pip install matplotlib")
    print("Original error:", e)
    raise

try:
    from PIL import Image, ImageOps
except Exception as e:
    print("[ERROR] pillow is required. Install it by: pip install pillow")
    print("Original error:", e)
    raise

try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except Exception as e:
    print("[ERROR] python-docx is required. Install it by: pip install python-docx")
    print("Original error:", e)
    raise


# =============================
# Configuration
# =============================
SCRIPT_NAME = "32v_est_style_storyline_figure_suite.py"
RUN_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

EST_BLUE = "#4C78A8"
EST_GREEN = "#72A276"
EST_ORANGE = "#DDAA4F"
EST_RED = "#C44E52"
EST_GREY = "#9AA7B4"
EST_DARK = "#222222"
EST_LIGHT = "#F7F8FA"
EST_GRID = "#E5E7EB"

# Source figure candidates from previous steps.
# Each entry is a list because your project may have pdf/png variants or slightly different names.
SOURCE_FIGURE_GROUPS = {
    "dataset_group_size": ["69_", "literature_group_size_distribution"],
    "paper_target_dist": ["70_", "paper_level_target_distribution"],
    "feature_curve": ["63_", "selected_feature_curve"],
    "random_metrics": ["64_", "selected_fusion_metrics_dashboard"],
    "weight_stability": ["65_", "selected_fusion_weight_stability"],
    "tuned_xgb_cat_metrics": ["99_", "tuned_catboost_xgboost_metrics"],
    "feature_ablation": ["111_", "feature_ablation_metrics"],
    "smd_shift": ["72_", "train_test_numeric_shift_smd_heatmap"],
    "grouped_metrics": ["80_", "grouped_screen_metrics"],
    "grouped_pred_actual": ["81_", "grouped_screen_pooled_predicted_vs_actual"],
    "lolo_rmse": ["129_", "lolo_paper_level_rmse"],
    "ad_risk_error": ["139_", "ad_risk_error_relationship"],
    "reliable_domain": ["153_", "all_vs_reliable_domain_metrics", "reliable_domain"],
    "final_evidence_chain": ["194_", "evidence_chain_summary_clean"],
    "threshold_plateau": ["195_", "threshold_plateau_evidence_clean"],
    "chemistry_window": ["196_", "retained_chemistry_window_clean"],
    "stability_boundary": ["197_", "stability_transfer_boundary_clean"],
}

# Optional output figure names.
EST_FIGURES = {
    "framework": "201_32v_est_style_overall_storyline_framework",
    "random": "202_32v_est_style_random_split_modeling_evidence",
    "grouped": "203_32v_est_style_grouped_validation_boundary_evidence",
    "ad": "204_32v_est_style_selective_ad_boundary_evidence",
}

LOCKED_RESULT = {
    "result_type": "target-free selective AD retained-subdomain",
    "model": "RepeatedBalanced_Extended + XGBoost",
    "rule": "Zone=Outside; NN_Risk <= 0.814815",
    "r2": 0.747676,
    "rmse": 11.474674,
    "mae": 9.155856,
    "coverage": 0.014664,
    "n": 52,
    "dominant_group": "Paper_010",
}


@dataclass
class ProjectPaths:
    root: Path
    figures: Path
    tables: Path
    reports: Path
    src: Optional[Path] = None


def _norm(p: Path) -> Path:
    try:
        return p.resolve()
    except Exception:
        return p.absolute()


def find_project_root() -> Path:
    """Find the real PyCharm project root, preferring folders with src + outputs."""
    candidates: List[Path] = []

    # Script location, cwd, and their parents.
    try:
        candidates.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    candidates.append(Path.cwd())

    expanded: List[Path] = []
    for c in candidates:
        c = _norm(c)
        expanded.append(c)
        expanded.extend(list(c.parents)[:6])

    # If script is inside src, parent is almost certainly project root.
    for c in expanded:
        if c.name.lower() == "src" and (c.parent / "outputs").exists():
            return c.parent

    # Strong project root: has outputs, reports, and src.
    for c in expanded:
        if (c / "outputs").exists() and (c / "src").exists():
            return c

    # Medium: has outputs/figures.
    for c in expanded:
        if (c / "outputs" / "figures").exists():
            return c

    # Fallback cwd.
    return _norm(Path.cwd())


def prepare_paths() -> ProjectPaths:
    root = find_project_root()
    figures = root / "outputs" / "figures"
    tables = root / "outputs" / "tables"
    reports = root / "reports"
    src = root / "src" if (root / "src").exists() else None
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(root=root, figures=figures, tables=tables, reports=reports, src=src)


def find_docx_template(paths: ProjectPaths) -> Optional[Path]:
    """Find the manuscript draft/template. Prefer user's latest draft."""
    names = [
        "AI论文--混凝沉淀除氟初稿.docx",
        "32t2_full_manuscript_intro_results_random_grouped_ad.docx",
        "32s5_results_discussion_full_template_aligned_with_figures.docx",
        "32s4_results_discussion_template_aligned_final.docx",
        "AI论文--混凝沉淀除氟空稿.docx",
    ]
    locations = [
        paths.root,
        paths.reports,
        paths.root.parent,
        paths.root.parent.parent if paths.root.parent else paths.root,
        Path("/mnt/data"),
        Path("/mnt/data/reports"),
    ]
    for nm in names:
        for loc in locations:
            p = loc / nm
            if p.exists() and p.is_file():
                return p
    return None


def find_image_by_keys(fig_dir: Path, keys: Sequence[str], extra_dirs: Optional[Sequence[Path]] = None) -> Optional[Path]:
    """Find a PNG image matching any candidate key. Prefer exact numeric prefix."""
    dirs = [fig_dir]
    if extra_dirs:
        dirs.extend([d for d in extra_dirs if d and d.exists()])
    seen = set()
    candidates: List[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.png"):
            if p in seen:
                continue
            seen.add(p)
            name = p.name.lower()
            if any(str(k).lower() in name for k in keys):
                candidates.append(p)
    if not candidates:
        return None

    # Rank: exact numeric prefix first, shorter name next.
    def rank(p: Path) -> Tuple[int, int, str]:
        name = p.name.lower()
        score = 10
        for k in keys:
            k = str(k).lower()
            if re.match(r"^\d+_", k) and name.startswith(k):
                score = 0
            elif name.startswith(k):
                score = min(score, 1)
            elif k in name:
                score = min(score, 2)
        return (score, len(name), name)

    return sorted(candidates, key=rank)[0]


def collect_sources(paths: ProjectPaths) -> Dict[str, Optional[Path]]:
    extra = [paths.root, paths.root / "outputs", Path("/mnt/data"), Path("/mnt/data/outputs/figures")]
    found = {}
    for key, keys in SOURCE_FIGURE_GROUPS.items():
        found[key] = find_image_by_keys(paths.figures, keys, extra_dirs=extra)
    return found


def load_image_for_panel(path: Optional[Path], max_px: int = 1800) -> Image.Image:
    if path is None or not Path(path).exists():
        img = Image.new("RGB", (1000, 650), "white")
        return img
    img = Image.open(path).convert("RGB")
    # Crop very large white border conservatively.
    bg = Image.new("RGB", img.size, "white")
    diff = ImageOps.invert(ImageChops.difference(img, bg)) if False else None
    # Resize for memory.
    w, h = img.size
    scale = min(1.0, max_px / max(w, h))
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def imshow_panel(ax, path: Optional[Path], title: str, label: str, subtitle: str = "") -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if path is None or not Path(path).exists():
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="white", ec=EST_GREY, lw=1.1))
        ax.text(0.5, 0.55, "Source figure\nnot found", ha="center", va="center", fontsize=10, color=EST_GREY, transform=ax.transAxes)
        ax.text(0.5, 0.34, title, ha="center", va="center", fontsize=8, color=EST_DARK, transform=ax.transAxes)
    else:
        # Downscale very large source figures before plotting.
        # This keeps the EST composite creation fast and prevents oversized DOCX files.
        img_pil = Image.open(path).convert("RGB")
        w, h = img_pil.size
        max_side = 1800
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img_pil = img_pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        ax.imshow(np.asarray(img_pil))
    ax.text(0.012, 0.988, label, ha="left", va="top", transform=ax.transAxes,
            fontsize=16, fontweight="bold", color=EST_DARK,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88))
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6, color=EST_DARK)
    if subtitle:
        ax.text(0.5, -0.055, subtitle, ha="center", va="top", transform=ax.transAxes,
                fontsize=8.3, color="#555555")


def save_fig(fig, out_base: Path, dpi: int = 220) -> Tuple[Path, Path]:
    """Save PNG for Word insertion. PDF saving is intentionally skipped to keep PyCharm runs fast."""
    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def create_framework_figure(paths: ProjectPaths) -> Tuple[Path, Path]:
    out_base = paths.figures / EST_FIGURES["framework"]
    fig = plt.figure(figsize=(13.5, 7.5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    fig.suptitle("Mechanism-informed ML framework for heterogeneous coagulation-defluoridation data",
                 fontsize=18, fontweight="bold", y=0.985, color=EST_DARK)

    # Top-left problem statement
    ax.text(0.055, 0.875, "Environmental small-data challenge", fontsize=13.5, fontweight="bold", color=EST_DARK)
    bullets = [
        "multi-source literature data",
        "nonlinear chemistry and co-existing ions",
        "random split may overestimate interpolation performance",
        "grouped validation exposes transfer boundary",
    ]
    for i, b in enumerate(bullets):
        ax.text(0.06, 0.835 - i * 0.036, "• " + b, fontsize=10.5, color="#444444")

    # Flow boxes
    boxes = [
        (0.08, 0.52, 0.17, 0.18, "Data construction", "literature extraction\nunit harmonization\nsource-group labels", EST_BLUE),
        (0.30, 0.52, 0.17, 0.18, "Mechanism features", "pH window\ndose/F load\nion interference", EST_GREEN),
        (0.52, 0.52, 0.17, 0.18, "Random-split learning", "model screening\nfusion/stacking\nstability check", EST_ORANGE),
        (0.74, 0.52, 0.17, 0.18, "Grouped validation", "leave-source transfer\nerror diagnosis\nAD risk", EST_RED),
    ]
    for x, y, w, h, head, body, color in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                               fc=color, ec="none", alpha=0.92, transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h*0.68, head, ha="center", va="center", fontsize=11.5, fontweight="bold", color="white", transform=ax.transAxes)
        ax.text(x + w/2, y + h*0.33, body, ha="center", va="center", fontsize=9.3, color="white", transform=ax.transAxes)
    for (x1, y1, w1, h1, *_), (x2, y2, w2, h2, *__) in zip(boxes[:-1], boxes[1:]):
        arr = FancyArrowPatch((x1+w1+0.01, y1+h1/2), (x2-0.01, y2+h2/2),
                              arrowstyle="-|>", mutation_scale=18, lw=2.0, color="#6B7280", transform=ax.transAxes)
        ax.add_patch(arr)

    # Bottom result hierarchy
    ax.text(0.055, 0.405, "Result hierarchy used in this manuscript", fontsize=13.5, fontweight="bold", color=EST_DARK)
    hierarchy = [
        ("Random split", "internal interpolation\nmodel screening", "not sufficient for transfer claims", EST_BLUE),
        ("All-domain grouped validation", "cross-literature transfer\nstrict generalization test", "reveals poor whole-domain transfer", EST_GREY),
        ("Selective AD retained subdomain", "target-free risk constraint\nlocal reliable window", "R²=0.748, coverage=1.47%", EST_GREEN),
    ]
    start_x = 0.08
    for i, (head, body, note, color) in enumerate(hierarchy):
        x = start_x + i * 0.30
        y = 0.16
        patch = FancyBboxPatch((x, y), 0.24, 0.17, boxstyle="round,pad=0.012,rounding_size=0.018",
                               fc="white", ec=color, lw=2.0, transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(x+0.12, y+0.125, head, ha="center", va="center", fontsize=11.0, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x+0.12, y+0.080, body, ha="center", va="center", fontsize=9.1, color="#333333", transform=ax.transAxes)
        ax.text(x+0.12, y+0.030, note, ha="center", va="center", fontsize=8.6, color="#555555", transform=ax.transAxes)
        if i < len(hierarchy)-1:
            arr = FancyArrowPatch((x+0.245, y+0.085), (x+0.295, y+0.085),
                                  arrowstyle="-|>", mutation_scale=16, lw=1.8, color="#6B7280", transform=ax.transAxes)
            ax.add_patch(arr)

    # Locked boundary callout
    callout = FancyBboxPatch((0.67, 0.80), 0.27, 0.12, boxstyle="round,pad=0.012,rounding_size=0.015",
                             fc="#FFF7ED", ec=EST_ORANGE, lw=1.8, transform=ax.transAxes)
    ax.add_patch(callout)
    ax.text(0.805, 0.865, "Writing boundary", ha="center", va="center", fontsize=11.2, fontweight="bold", color=EST_DARK, transform=ax.transAxes)
    ax.text(0.805, 0.825, "R²=0.748 belongs only to the\nselective AD retained subdomain.", ha="center", va="center", fontsize=9.4, color="#333333", transform=ax.transAxes)

    return save_fig(fig, out_base)


def create_composite(paths: ProjectPaths, out_key: str, title: str, panels: List[Tuple[str, str, str, Optional[Path]]], grid=(2, 2)) -> Tuple[Path, Path]:
    out_base = paths.figures / EST_FIGURES[out_key]
    nrows, ncols = grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 9.0), facecolor="white")
    axes = np.array(axes).reshape(-1)
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.985, color=EST_DARK)
    for ax, (label, panel_title, subtitle, path) in zip(axes, panels):
        imshow_panel(ax, path, panel_title, label, subtitle)
    for ax in axes[len(panels):]:
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    return save_fig(fig, out_base)


def create_all_est_figures(paths: ProjectPaths, sources: Dict[str, Optional[Path]]) -> Dict[str, Path]:
    created: Dict[str, Path] = {}

    png, pdf = create_framework_figure(paths)
    created["201"] = png

    panels_202 = [
        ("A", "Literature and target distribution", "source imbalance and response heterogeneity", sources.get("dataset_group_size") or sources.get("paper_target_dist")),
        ("B", "Mechanism-informed feature selection", "feature engineering before model comparison", sources.get("feature_curve")),
        ("C", "Random-split model/fusion metrics", "internal interpolation ability", sources.get("random_metrics") or sources.get("tuned_xgb_cat_metrics")),
        ("D", "Weight or seed stability", "single-split results require stability checks", sources.get("weight_stability") or sources.get("feature_ablation")),
    ]
    png, pdf = create_composite(paths, "random", "Random-split modeling: model screening and stability evidence", panels_202)
    created["202"] = png

    panels_203 = [
        ("A", "Train-test distribution shift", "grouped split exposes covariate shift", sources.get("smd_shift")),
        ("B", "All-domain grouped validation metrics", "whole-domain transfer is difficult", sources.get("grouped_metrics")),
        ("C", "Grouped prediction scatter", "cross-literature prediction errors", sources.get("grouped_pred_actual")),
        ("D", "Leave-literature-out error landscape", "source-specific generalization boundary", sources.get("lolo_rmse")),
    ]
    png, pdf = create_composite(paths, "grouped", "Grouped validation: cross-literature generalization boundary", panels_203)
    created["203"] = png

    panels_204 = [
        ("A", "Evidence chain summary", "accepted result is a selective AD retained-subdomain result", sources.get("final_evidence_chain")),
        ("B", "NN-risk threshold plateau", "threshold support rather than a single isolated point", sources.get("threshold_plateau")),
        ("C", "Retained chemistry window", "where the model is locally supported", sources.get("chemistry_window")),
        ("D", "Stability and interpretation boundary", "low coverage and Paper_010 dominance must be stated", sources.get("stability_boundary")),
    ]
    png, pdf = create_composite(paths, "ad", "Selective applicability domain: reliable subdomain and interpretation boundary", panels_204)
    created["204"] = png

    return created


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    if not fieldnames:
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def add_run_style(paragraph, font_name: str = "宋体", font_size: int = 10, bold: bool = False, color: Optional[str] = None) -> None:
    for run in paragraph.runs:
        run.font.name = font_name
        run.font.size = Pt(font_size)
        run.bold = bold
        if color:
            run.font.color.rgb = RGBColor.from_string(color.replace("#", ""))


def insert_paragraph_after(paragraph, text: str = "", style=None):
    new_p = paragraph._p.addnext(paragraph._p.__class__())  # not reliable; replaced below


def _insert_paragraph_after(paragraph, text: str = "", style: Optional[str] = None):
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


def _insert_picture_after(paragraph, image_path: Path, width_inches: float = 6.35):
    p = _insert_paragraph_after(paragraph, "")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))
    return p


def find_results_anchor(doc: Document):
    patterns = ["3 Results and Discussion", "3. Results and Discussion", "3 Results", "3. Results", "结果与讨论"]
    for p in doc.paragraphs:
        txt = (p.text or "").strip()
        if any(pat in txt for pat in patterns):
            return p
    # fallback end of document
    if doc.paragraphs:
        return doc.paragraphs[-1]
    return doc.add_paragraph()


def insert_est_suite_into_docx(template: Optional[Path], paths: ProjectPaths, created: Dict[str, Path]) -> Path:
    if template and template.exists():
        doc = Document(str(template))
    else:
        doc = Document()
        title = doc.add_paragraph("基于机器学习建模的含氟废水混凝处理的工艺选择、优化及控制")
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_run_style(title, font_size=16, bold=True)
        doc.add_paragraph("3 Results and Discussion（结果与讨论）")

    anchor = find_results_anchor(doc)

    # Insert in reverse order after same anchor, so final order is correct.
    captions = [
        ("204", "图 3-4 EST风格选择性适应域证据链图。该图汇总最终选择性适应域保留子域的证据链、NN-risk阈值平台、保留化学窗口以及稳定性/解释边界，用于强调R² = 0.748仅属于无目标泄露的选择性适应域保留子域结果。"),
        ("203", "图 3-3 EST风格跨文献分组验证与泛化边界图。该图展示分组划分下训练集与测试集的分布偏移、未筛选全域分组验证性能、预测偏差以及按文献来源的外推误差，用于说明随机划分结果不能直接代表跨文献泛化能力。"),
        ("202", "图 3-2 EST风格随机划分建模与稳定性证据图。该图整合文献数据分布、机理特征选择、随机划分模型筛选/融合性能和权重或种子稳定性，用于说明树模型和集成模型在数据内部规律学习中的作用及其稳定性限制。"),
        ("201", "图 3-1 EST风格全文主线框架图。该图概括本研究从异质文献数据构建、机理特征工程、随机划分模型筛选、跨文献分组验证到选择性适应域边界识别的整体技术路线。"),
    ]
    p = _insert_paragraph_after(anchor, "")
    for run in p.runs:
        pass
    p.add_run("EST风格主线图与证据链套图").bold = True
    add_run_style(p, font_size=12, bold=True)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # Insert each figure after the title paragraph; update anchor each time.
    cur = p
    for fig_id in ["201", "202", "203", "204"]:
        fig_path = created.get(fig_id)
        if fig_path and fig_path.exists():
            cur = _insert_picture_after(cur, fig_path, width_inches=6.45)
        cap_text = dict(captions)[fig_id]
        cap = _insert_paragraph_after(cur, cap_text)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_run_style(cap, font_size=9, bold=False)
        cur = cap

    out = paths.reports / "32v_est_style_storyline_figure_suite_inserted.docx"
    doc.save(out)
    return out


def write_storyline_report(paths: ProjectPaths, created: Dict[str, Path], sources: Dict[str, Optional[Path]], docx_out: Path, template: Optional[Path]) -> None:
    report = paths.reports / "32v_est_style_storyline_and_figure_plan.md"
    lines = []
    lines.append("# 32V EST-style storyline and figure suite")
    lines.append("")
    lines.append(f"Generated: {RUN_TIME}")
    lines.append("")
    lines.append("## Paper storyline")
    lines.append("")
    lines.append("本研究的主线不是简单追求随机划分下的最高 R²，而是围绕“异质文献数据如何可靠建模”展开：")
    lines.append("")
    lines.append("1. **数据问题**：混凝除氟文献数据来源分散，存在变量缺失、目标分布不均衡和跨文献工况差异。")
    lines.append("2. **机理增强**：将 pH、投加量、初始氟、干扰离子和混凝剂类型等化学机理转化为模型可学习的特征。")
    lines.append("3. **随机划分建模**：用随机划分筛选模型、融合策略和特征处理方案，证明树模型更适合数据内部非线性规律学习。")
    lines.append("4. **分组验证转折**：按文献来源分组后，模型全域性能下降，说明随机划分结果不能等同于跨文献泛化能力。")
    lines.append("5. **适应域边界**：通过无目标泄露的选择性 AD 筛选，识别局部可靠保留子域，同时明确低覆盖率和 Paper_010 主导带来的解释边界。")
    lines.append("")
    lines.append("## Locked final selective-AD result")
    lines.append("")
    lines.append(f"- Model: {LOCKED_RESULT['model']}")
    lines.append(f"- Rule: {LOCKED_RESULT['rule']}")
    lines.append(f"- R²: {LOCKED_RESULT['r2']:.6f}")
    lines.append(f"- RMSE: {LOCKED_RESULT['rmse']:.6f}")
    lines.append(f"- MAE: {LOCKED_RESULT['mae']:.6f}")
    lines.append(f"- Coverage: {LOCKED_RESULT['coverage']:.6f}")
    lines.append(f"- Retained N: {LOCKED_RESULT['n']}")
    lines.append(f"- Dominant group: {LOCKED_RESULT['dominant_group']}")
    lines.append("")
    lines.append("> Writing boundary: R² = 0.747676 must be described only as a target-free selective AD retained-subdomain result, not as all-domain grouped-validation performance.")
    lines.append("")
    lines.append("## EST-style figure suite")
    for fig_id, path in created.items():
        lines.append(f"- {fig_id}: {path}")
    lines.append("")
    lines.append(f"DOCX output: {docx_out}")
    lines.append(f"Template/input DOCX: {template if template else 'new document created'}")
    report.write_text("\n".join(lines), encoding="utf-8")

    rows = []
    for fig_id, path in created.items():
        rows.append({"figure_id": fig_id, "est_style_output": str(path), "exists": path.exists()})
    write_csv(paths.tables / "32v_est_style_figure_suite_manifest.csv", rows)

    source_rows = []
    for key, path in sources.items():
        source_rows.append({"source_role": key, "path": str(path) if path else "", "exists": bool(path and path.exists())})
    write_csv(paths.tables / "32v_source_figure_matching_audit.csv", source_rows)

    hierarchy_rows = [
        {"stage": "Data challenge", "message": "heterogeneous literature data; missing key variables; imbalanced target distribution", "figure": "201/202"},
        {"stage": "Mechanism-informed features", "message": "pH, dose/F load, ion interference and coagulant category encode coagulation chemistry", "figure": "201/202"},
        {"stage": "Random split", "message": "useful for model screening and interpolation; not sufficient for external transfer claims", "figure": "202"},
        {"stage": "Grouped validation", "message": "all-domain cross-literature performance drops, exposing generalization boundary", "figure": "203"},
        {"stage": "Selective AD", "message": "target-free risk rule finds a small locally reliable subdomain; coverage is low", "figure": "204"},
    ]
    write_csv(paths.tables / "32v_article_storyline_hierarchy.csv", hierarchy_rows)


def main() -> None:
    paths = prepare_paths()
    print("=" * 70)
    print("32V EST-style storyline figure-suite generator")
    print("Run time:", RUN_TIME)
    print("Project root:", paths.root)
    print("Figure output dir:", paths.figures)
    print("Table output dir:", paths.tables)
    print("Report output dir:", paths.reports)
    print("=" * 70)

    template = find_docx_template(paths)
    print("[INPUT DOCX]", template if template else "not found; creating new document")

    sources = collect_sources(paths)
    for k, p in sources.items():
        print(f"[SOURCE] {k}: exists={bool(p and p.exists())} path={p if p else ''}")

    created = create_all_est_figures(paths, sources)
    for fig_id, path in created.items():
        print(f"[OUTPUT FIGURE] {fig_id}: {path}")

    docx_out = insert_est_suite_into_docx(template, paths, created)
    print("[OUTPUT DOCX]", docx_out)

    write_storyline_report(paths, created, sources, docx_out, template)

    # Final simple audit
    missing_final = [fid for fid in ["201", "202", "203", "204"] if fid not in created or not created[fid].exists()]
    print("=" * 70)
    if missing_final:
        print("32V finished with warnings. Missing EST figures:", missing_final)
    else:
        print("32V finished successfully.")
    print("Important boundary: R2=0.747676 belongs to selective AD retained-subdomain only.")
    print("Please open:")
    print(" ", docx_out)


if __name__ == "__main__":
    main()
