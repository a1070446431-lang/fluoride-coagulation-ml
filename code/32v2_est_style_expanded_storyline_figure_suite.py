# -*- coding: utf-8 -*-
"""
32V2 EST-style expanded storyline figure-suite generator
Project: fluoride coagulation defluoridation ML project

Purpose
-------
This version fixes the incompleteness of 32V by explicitly adding:
1) the earliest/highest random-split R2 result (CatBoost-HGB weighted two-model, R2≈0.825-0.829);
2) the step-by-step path from baseline models to the final two-model weighted fusion idea;
3) the exploratory best all-domain grouped-validation result around R2≈0.25;
4) the final audited all-domain grouped baseline (R2=0.194), predefined reliable domain, and selective AD retained-subdomain;
5) the previous key project figures as EST-style composite panels.

Important writing boundary
--------------------------
R2=0.747676 belongs ONLY to the target-free selective AD retained-subdomain result.
It must NOT be described as all-domain grouped-validation performance.

Output convention
-----------------
The script follows the original project convention and does NOT create a new output root:
    outputs/figures/
    outputs/tables/
    reports/

It does not retrain any model and does not change final metrics.
"""

from __future__ import annotations

import csv
import os
import re
import sys
import zipfile
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


SCRIPT_NAME = "32v2_est_style_expanded_storyline_figure_suite.py"
RUN_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# EST-like low-saturation palette
EST_BLUE = "#4C78A8"
EST_GREEN = "#72A276"
EST_ORANGE = "#DDAA4F"
EST_RED = "#C44E52"
EST_PURPLE = "#7E6AAE"
EST_GREY = "#9AA7B4"
EST_DARK = "#222222"
EST_LIGHT = "#F7F8FA"
EST_GRID = "#E5E7EB"

# Try common CJK fonts; silently fall back if not installed.
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

LOCKED_SELECTIVE_AD = {
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

# Manually locked milestones from the project conversation / accepted project record.
# These are used for figure annotation only. They are not recomputed here.
MILESTONES = [
    {
        "stage": "Linear / Ridge baseline",
        "r2": np.nan,
        "note": "linear baseline; insufficient for nonlinear coagulation chemistry",
        "level": "random_split_screening",
    },
    {
        "stage": "Random Forest baseline",
        "r2": 0.717,
        "note": "stable single-model reference in early random splits",
        "level": "random_split_screening",
    },
    {
        "stage": "CatBoost tuned",
        "r2": 0.741,
        "note": "strong single split after manual tuning, but less stable across seeds",
        "level": "random_split_screening",
    },
    {
        "stage": "RF-CatBoost Linear stacking",
        "r2": 0.694,
        "note": "multi-seed stable comparison; small gain over RF baseline",
        "level": "random_split_multiseed",
    },
    {
        "stage": "CatBoost-HGB Weighted 2Model",
        "r2": 0.829,
        "note": "highest random-split pooled result; internal interpolation benchmark",
        "level": "random_split_best",
    },
    {
        "stage": "Strict final-data random rerun",
        "r2": 0.730,
        "note": "strict reconstructed final dataset; 4-model weighted result",
        "level": "random_split_final_structure",
    },
    {
        "stage": "Exploratory all-domain grouped best",
        "r2": 0.250,
        "note": "best exploratory full-domain grouped result around 0.25; still below target",
        "level": "grouped_exploratory_best",
    },
    {
        "stage": "Audited all-domain grouped baseline",
        "r2": 0.194,
        "note": "32J/32I2 evidence-chain all-domain grouped baseline",
        "level": "grouped_audited_baseline",
    },
    {
        "stage": "Predefined reliable domain",
        "r2": -0.160,
        "note": "fixed reliable-domain rule did not improve generalization",
        "level": "predefined_reliable_domain",
    },
    {
        "stage": "Selective AD retained subdomain",
        "r2": 0.748,
        "note": "local retained-subdomain result; coverage=1.47%; not all-domain",
        "level": "selective_ad",
    },
]

SOURCE_FIGURE_GROUPS = {
    # Data and feature stage
    "missing_or_dataset_dist": ["1_", "missing", "缺失", "target_distribution", "dataset_distribution"],
    "dataset_group_size": ["69_", "literature_group_size_distribution"],
    "paper_target_dist": ["70_", "paper_level_target_distribution"],
    "feature_curve": ["63_", "selected_feature_curve"],
    "feature_importance": ["68_", "selected_feature_importance"],
    # Random split / two-model pathway
    "random_metrics": ["64_", "selected_fusion_metrics_dashboard"],
    "weight_stability": ["65_", "selected_fusion_weight_stability"],
    "random_pred_actual": ["66_", "selected_fusion_pooled_predicted_vs_actual"],
    "random_residual": ["67_", "selected_fusion_residual_plot"],
    "tuned_xgb_cat_metrics": ["99_", "tuned_catboost_xgboost_metrics"],
    "feature_ablation": ["111_", "feature_ablation_metrics"],
    # Grouped validation
    "smd_shift": ["72_", "train_test_numeric_shift_smd_heatmap"],
    "grouped_metrics": ["80_", "grouped_screen_metrics"],
    "grouped_pred_actual": ["81_", "grouped_screen_pooled_predicted_vs_actual"],
    "lolo_rmse": ["129_", "lolo_paper_level_rmse"],
    "grouped_tuning_metrics": ["106_", "tuning_grid_adequacy_metrics"],
    "grouped_feature_ablation": ["111_", "31k_feature_ablation_metrics"],
    # AD and reliable domain
    "ad_risk_error": ["139_", "ad_risk_error_relationship"],
    "ad_performance_bin": ["140_", "ad_performance_by_risk_bin"],
    "paper_ad_risk": ["141_", "paper_ad_risk_vs_rmse"],
    "reliable_tradeoff": ["148_", "reliable_rule_tradeoff"],
    "reliable_domain": ["153_", "all_vs_reliable_domain_metrics", "reliable_domain"],
    # Final selective AD
    "fast_policy_metrics": ["158_", "fast_policy_metrics"],
    "r2_coverage_frontier": ["159_", "r2_coverage_frontier", "166_32d_r2_coverage_frontier"],
    "policy_stability": ["168_", "best_policy_split_stability", "182_32f2_best_policy_split_stability"],
    "final_evidence_chain": ["194_", "evidence_chain_summary_clean"],
    "threshold_plateau": ["195_", "threshold_plateau_evidence_clean"],
    "chemistry_window": ["196_", "retained_chemistry_window_clean"],
    "stability_boundary": ["197_", "stability_transfer_boundary_clean"],
}

EST_FIGURES = {
    "framework": "205_32v2_est_full_storyline_framework",
    "random": "206_32v2_est_random_split_to_twomodel_evidence",
    "grouped": "207_32v2_est_grouped_validation_boundary_with_025",
    "ad_development": "208_32v2_est_ad_development_and_reliable_domain",
    "final_ad": "209_32v2_est_final_selective_ad_evidence_chain",
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
    candidates: List[Path] = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    candidates.append(Path.cwd())

    expanded: List[Path] = []
    for c in candidates:
        c = _norm(c)
        expanded.append(c)
        expanded.extend(list(c.parents)[:8])

    for c in expanded:
        if str(c) == os.path.sep:
            continue
        if c.name.lower() == "src" and (c.parent / "outputs" / "figures").exists():
            return c.parent
    for c in expanded:
        if str(c) == os.path.sep:
            continue
        if (c / "src").exists() and (c / "outputs" / "figures").exists() and (c / "outputs" / "tables").exists():
            return c
    for c in expanded:
        if str(c) == os.path.sep:
            continue
        if (c / "outputs" / "figures").exists() and (c / "outputs" / "tables").exists():
            return c
    # Safer fallback: use script folder rather than filesystem root/current working directory.
    try:
        return _norm(Path(__file__).resolve().parent)
    except Exception:
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
    names = [
        "32u_full_manuscript_with_random_grouped_ad_and_previous_figures.docx",
        "32t2_full_manuscript_intro_results_random_grouped_ad.docx",
        "32s5_results_discussion_full_template_aligned_with_figures.docx",
        "AI论文--混凝沉淀除氟初稿.docx",
        "AI论文--混凝沉淀除氟空稿.docx",
    ]
    locations = [
        paths.reports,
        paths.root,
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
    dirs = [fig_dir]
    if extra_dirs:
        dirs.extend([d for d in extra_dirs if d and d.exists()])
    seen = set()
    candidates: List[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        # Avoid recursively scanning very large parent folders such as Downloads or /mnt/data.
        # Project figure folders are searched recursively; generic fallback folders are searched only at top level.
        if d.name.lower() in {"figures", "outputs"}:
            iterator = d.rglob("*.png")
        else:
            iterator = d.glob("*.png")
        for p in iterator:
            if p in seen or not p.is_file():
                continue
            seen.add(p)
            name = p.name.lower()
            if any(str(k).lower() in name for k in keys):
                candidates.append(p)
    if not candidates:
        return None

    def rank(p: Path) -> Tuple[int, int, str]:
        name = p.name.lower()
        score = 100
        for k in keys:
            kk = str(k).lower()
            if re.match(r"^\d+_", kk) and name.startswith(kk):
                score = min(score, 0)
            elif name.startswith(kk):
                score = min(score, 1)
            elif kk in name:
                score = min(score, 2)
        # Prefer non-EST replot only when exact key is matched; shorter names often original.
        return (score, len(name), name)

    return sorted(candidates, key=rank)[0]


def collect_sources(paths: ProjectPaths) -> Dict[str, Optional[Path]]:
    extra = [paths.root, paths.root / "outputs", Path("/mnt/data"), Path("/mnt/data/outputs/figures")]
    found = {}
    for key, keys in SOURCE_FIGURE_GROUPS.items():
        found[key] = find_image_by_keys(paths.figures, keys, extra_dirs=extra)
    return found


def resize_image_for_panel(path: Optional[Path], max_side: int = 1700) -> Optional[np.ndarray]:
    if path is None or not Path(path).exists() or not Path(path).is_file():
        return None
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    return np.asarray(img)


def imshow_panel(ax, path: Optional[Path], title: str, label: str, subtitle: str = "") -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    arr = resize_image_for_panel(path)
    if arr is None:
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="white", ec=EST_GREY, lw=1.1))
        ax.text(0.5, 0.56, "Source figure\nnot found", ha="center", va="center", fontsize=10, color=EST_GREY, transform=ax.transAxes)
        ax.text(0.5, 0.35, title, ha="center", va="center", fontsize=8.2, color=EST_DARK, transform=ax.transAxes)
    else:
        ax.imshow(arr)
    ax.text(0.012, 0.988, label, ha="left", va="top", transform=ax.transAxes,
            fontsize=15.5, fontweight="bold", color=EST_DARK,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88))
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6, color=EST_DARK)
    if subtitle:
        ax.text(0.5, -0.055, subtitle, ha="center", va="top", transform=ax.transAxes,
                fontsize=8.3, color="#555555")


def save_fig(fig, out_base: Path, dpi: int = 230) -> Tuple[Path, Path]:
    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    # Keep PDF available for manuscript polishing; safe and not too large for vector/text figures.
    try:
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    except Exception:
        pass
    plt.close(fig)
    return png, pdf


def draw_badge(ax, x, y, text, color, w=0.18, h=0.055, fs=10.0):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                           fc=color, ec="none", alpha=0.94, transform=ax.transAxes)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color="white", transform=ax.transAxes)


def create_framework_figure(paths: ProjectPaths) -> Tuple[Path, Path]:
    out_base = paths.figures / EST_FIGURES["framework"]
    fig = plt.figure(figsize=(14.4, 8.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.suptitle("Mechanism-informed ML framework for heterogeneous coagulation-defluoridation data",
                 fontsize=18, fontweight="bold", y=0.987, color=EST_DARK)

    # Pipeline
    steps = [
        ("Heterogeneous\nliterature data", "1011 samples\n32 literature groups\nsource labels only for splitting", EST_BLUE),
        ("Mechanism-informed\nfeature engineering", "pH window\ndose/F loading\nion interference", EST_GREEN),
        ("Random split\nmodel screening", "highest internal R²≈0.829\nCatBoost-HGB weighted 2Model", EST_ORANGE),
        ("Grouped validation\ntransfer test", "all-domain best≈0.25\naudited baseline R²=0.194", EST_RED),
        ("Selective AD\nretained subdomain", "R²=0.748\ncoverage=1.47%\nPaper_010 boundary", EST_PURPLE),
    ]
    xs = [0.045, 0.235, 0.425, 0.615, 0.805]
    y = 0.56
    for i, ((head, body, color), x) in enumerate(zip(steps, xs)):
        patch = FancyBboxPatch((x, y), 0.155, 0.205, boxstyle="round,pad=0.012,rounding_size=0.018",
                               fc=color, ec="none", alpha=0.93, transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(x+0.0775, y+0.145, head, ha="center", va="center", fontsize=10.5, fontweight="bold", color="white", transform=ax.transAxes)
        ax.text(x+0.0775, y+0.065, body, ha="center", va="center", fontsize=8.5, color="white", transform=ax.transAxes)
        if i < len(xs)-1:
            arr = FancyArrowPatch((x+0.165, y+0.102), (xs[i+1]-0.012, y+0.102), arrowstyle="-|>",
                                  mutation_scale=16, lw=1.8, color="#6B7280", transform=ax.transAxes)
            ax.add_patch(arr)

    # Core message blocks
    ax.text(0.055, 0.88, "Main problem", fontsize=13.5, fontweight="bold", color=EST_DARK, transform=ax.transAxes)
    ax.text(0.055, 0.835,
            "Random splitting learns internal interpolation, but literature-grouped splitting reveals the real transfer boundary.",
            fontsize=10.7, color="#333333", transform=ax.transAxes)
    ax.text(0.055, 0.800,
            "The manuscript therefore reports three non-interchangeable levels: random split, all-domain grouped validation, and selective AD retained subdomain.",
            fontsize=10.7, color="#333333", transform=ax.transAxes)

    # Result hierarchy table-like region
    ax.text(0.055, 0.420, "Non-interchangeable result hierarchy", fontsize=13.5, fontweight="bold", color=EST_DARK, transform=ax.transAxes)
    rows = [
        ("Random split", "model selection / interpolation", "highest R²≈0.829", EST_ORANGE),
        ("All-domain grouped validation", "cross-literature generalization", "exploratory best≈0.25; audited R²=0.194", EST_RED),
        ("Predefined reliable domain", "fixed AD rule", "R²=-0.160; not sufficient", EST_GREY),
        ("Selective AD retained subdomain", "local reliable window", "R²=0.748; coverage=1.47%", EST_GREEN),
    ]
    x0, y0 = 0.06, 0.335
    colw = [0.22, 0.33, 0.30]
    ax.add_patch(Rectangle((x0, y0-0.18), sum(colw), 0.215, transform=ax.transAxes, fc="white", ec=EST_GRID, lw=1.2))
    for i, (level, purpose, result, color) in enumerate(rows):
        yy = y0 - i*0.05
        ax.add_patch(Rectangle((x0, yy-0.028), 0.025, 0.025, transform=ax.transAxes, fc=color, ec="none"))
        ax.text(x0+0.035, yy-0.015, level, fontsize=9.7, fontweight="bold", color=EST_DARK, transform=ax.transAxes, va="center")
        ax.text(x0+colw[0], yy-0.015, purpose, fontsize=9.3, color="#444444", transform=ax.transAxes, va="center")
        ax.text(x0+colw[0]+colw[1], yy-0.015, result, fontsize=9.3, color="#444444", transform=ax.transAxes, va="center")

    # Locked boundary callout
    callout = FancyBboxPatch((0.665, 0.275), 0.285, 0.145, boxstyle="round,pad=0.012,rounding_size=0.015",
                             fc="#FFF7ED", ec=EST_ORANGE, lw=1.8, transform=ax.transAxes)
    ax.add_patch(callout)
    ax.text(0.807, 0.372, "Writing boundary", ha="center", va="center", fontsize=12.0, fontweight="bold", color=EST_DARK, transform=ax.transAxes)
    ax.text(0.807, 0.325, "R²=0.748 is NOT all-domain performance.\nIt belongs only to selective AD retained subdomain.",
            ha="center", va="center", fontsize=9.4, color="#333333", transform=ax.transAxes)

    return save_fig(fig, out_base)


def create_random_split_figure(paths: ProjectPaths, sources: Dict[str, Optional[Path]]) -> Tuple[Path, Path]:
    out_base = paths.figures / EST_FIGURES["random"]
    fig = plt.figure(figsize=(15.0, 10.0), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], width_ratios=[1.15, 1.15, 1.2], wspace=0.22, hspace=0.32)
    fig.suptitle("Random-split modeling: from broad model screening to CatBoost-HGB weighted two-model", fontsize=17, fontweight="bold", y=0.985)

    # A: metric timeline including highest random R2
    ax = fig.add_subplot(gs[0, :2])
    labels = ["RF\nbaseline", "CatBoost\ntuned", "RF-CB\nstacking", "CatBoost-HGB\nWeighted 2Model", "Strict final\n4Model rerun"]
    vals = [0.717, 0.741, 0.694, 0.829, 0.730]
    colors = [EST_BLUE, EST_GREEN, EST_GREY, EST_ORANGE, EST_PURPLE]
    bars = ax.bar(np.arange(len(vals)), vals, color=colors, edgecolor=EST_DARK, linewidth=0.8)
    ax.axhline(0.8, ls="--", lw=1.2, color=EST_RED, alpha=0.7)
    ax.text(2.9, 0.812, "high random-split internal accuracy", fontsize=9.0, color=EST_RED)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("R² under random-split related evaluation")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=8.6)
    ax.grid(axis="y", color=EST_GRID, lw=0.8)
    ax.set_title("A  Model-performance milestones that led to the two-model solution", fontsize=11.0, fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.018, f"{v:.3f}", ha="center", va="bottom", fontsize=9.0)
    ax.text(3, 0.12, "Highest random-split result\nPooled R²≈0.825-0.829\nCatBoost + HGB weighted fusion", ha="center", va="center",
            fontsize=9.3, bbox=dict(boxstyle="round,pad=0.35", fc="#FFF7ED", ec=EST_ORANGE, lw=1.3))

    # B: path to two-model fusion
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.set_title("B  Why and how the two-model fusion was reached", fontsize=11.0, fontweight="bold")
    flow = [
        ("1. Baseline screening", "linear models insufficient; tree models dominate", EST_BLUE),
        ("2. Single-model tuning", "CatBoost improves but seed stability is limited", EST_GREEN),
        ("3. Fusion screening", "RF/ET/HGB/CatBoost/XGBoost candidates tested", EST_ORANGE),
        ("4. Weight search", "validation RMSE selects non-negative weights", EST_PURPLE),
        ("5. Accepted random-split benchmark", "CatBoost + HGB weighted 2Model", EST_RED),
    ]
    y = 0.88
    for i, (head, body, color) in enumerate(flow):
        patch = FancyBboxPatch((0.06, y-0.12), 0.88, 0.095, boxstyle="round,pad=0.012,rounding_size=0.018",
                               fc="white", ec=color, lw=1.8, transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(0.10, y-0.055, head, ha="left", va="center", fontsize=9.4, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.10, y-0.092, body, ha="left", va="center", fontsize=8.2, color="#444444", transform=ax.transAxes)
        if i < len(flow)-1:
            arr = FancyArrowPatch((0.50, y-0.125), (0.50, y-0.165), arrowstyle="-|>", mutation_scale=12, lw=1.4, color="#6B7280", transform=ax.transAxes)
            ax.add_patch(arr)
        y -= 0.17

    # C/D/E source panels
    ax = fig.add_subplot(gs[1, 0])
    imshow_panel(ax, sources.get("random_metrics") or sources.get("tuned_xgb_cat_metrics"), "Random-split/fusion metrics", "C", "metric dashboard for internal learning")
    ax = fig.add_subplot(gs[1, 1])
    imshow_panel(ax, sources.get("weight_stability"), "Fusion-weight stability", "D", "why weight search and seed checks were needed")
    ax = fig.add_subplot(gs[1, 2])
    imshow_panel(ax, sources.get("random_pred_actual") or sources.get("feature_ablation"), "Predicted vs observed / ablation evidence", "E", "random-split fit and feature simplification")

    return save_fig(fig, out_base)


def create_grouped_figure(paths: ProjectPaths, sources: Dict[str, Optional[Path]]) -> Tuple[Path, Path]:
    out_base = paths.figures / EST_FIGURES["grouped"]
    fig = plt.figure(figsize=(15.0, 10.0), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], width_ratios=[1.05, 1.2, 1.2], wspace=0.23, hspace=0.32)
    fig.suptitle("Grouped validation: random-split accuracy does not equal cross-literature transfer", fontsize=17, fontweight="bold", y=0.985)

    # A: result hierarchy with 0.25 included
    ax = fig.add_subplot(gs[0, 0])
    levels = [
        "Random-split\nbest", "Grouped\nexploratory best", "Grouped\naudited baseline", "Predefined\nreliable domain", "Selective AD\nretained subdomain"
    ]
    vals = [0.829, 0.250, 0.194, -0.160, 0.748]
    colors = [EST_ORANGE, EST_RED, EST_BLUE, EST_GREY, EST_GREEN]
    ypos = np.arange(len(vals))[::-1]
    ax.barh(ypos, vals, color=colors, edgecolor=EST_DARK, linewidth=0.7)
    ax.axvline(0.5, ls="--", lw=1.2, color=EST_RED, alpha=0.75)
    ax.text(0.51, ypos[0]+0.35, "target R²=0.5", fontsize=8.7, color=EST_RED)
    ax.set_xlim(-0.22, 0.90)
    ax.set_yticks(ypos)
    ax.set_yticklabels(levels, fontsize=8.5)
    ax.set_xlabel("R²")
    ax.grid(axis="x", color=EST_GRID, lw=0.8)
    ax.set_title("A  Metric hierarchy, including the ≈0.25 full-domain grouped result", fontsize=10.8, fontweight="bold")
    for y, v in zip(ypos, vals):
        ax.text(v + (0.025 if v >= 0 else -0.04), y, f"{v:.3f}", va="center", ha="left" if v>=0 else "right", fontsize=8.8)
    ax.text(0.10, -0.95, "Key: ≈0.25 was the best exploratory all-domain grouped result;\n0.194 is the audited all-domain baseline used in 32J/32I2.",
            fontsize=8.2, color="#444444", transform=ax.transData,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=EST_GRID, lw=1.0))

    # B/C top source panels
    ax = fig.add_subplot(gs[0, 1])
    imshow_panel(ax, sources.get("smd_shift"), "Train-test shift under grouped split", "B", "why grouped validation is harder")
    ax = fig.add_subplot(gs[0, 2])
    imshow_panel(ax, sources.get("grouped_metrics") or sources.get("grouped_tuning_metrics"), "All-domain grouped metrics", "C", "whole-domain transfer remains weak")
    # D/E/F bottom panels
    ax = fig.add_subplot(gs[1, 0])
    imshow_panel(ax, sources.get("grouped_pred_actual"), "Grouped predicted vs observed", "D", "error pattern under source transfer")
    ax = fig.add_subplot(gs[1, 1])
    imshow_panel(ax, sources.get("lolo_rmse"), "Leave-one-literature-out error landscape", "E", "paper-level transfer difficulty")
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.set_title("F  Interpretation boundary", fontsize=10.8, fontweight="bold")
    text = (
        "Why grouped R² drops:\n"
        "• literature-specific chemistry windows\n"
        "• different coagulants and dose scales\n"
        "• sparse low-removal and failure regions\n"
        "• source-level covariate shift\n\n"
        "Manuscript rule:\n"
        "random split → model screening;\n"
        "all-domain grouped → transfer test;\n"
        "selective AD → local reliable window."
    )
    ax.text(0.06, 0.88, text, ha="left", va="top", fontsize=10.2, color="#333333", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.45", fc="#F8FAFC", ec=EST_GRID, lw=1.2))
    draw_badge(ax, 0.10, 0.10, "Do not mix metrics", EST_RED, w=0.55, h=0.075, fs=10.5)

    return save_fig(fig, out_base)


def create_ad_development_figure(paths: ProjectPaths, sources: Dict[str, Optional[Path]]) -> Tuple[Path, Path]:
    out_base = paths.figures / EST_FIGURES["ad_development"]
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.4), facecolor="white")
    axes = axes.reshape(-1)
    fig.suptitle("Applicability-domain development: from risk diagnosis to reliable-domain limits", fontsize=17, fontweight="bold", y=0.985)
    panels = [
        ("A", "AD risk vs error", "target-free risk can diagnose but not automatically solve transfer", sources.get("ad_risk_error")),
        ("B", "Risk-bin performance", "high-risk regions remain uncertain", sources.get("ad_performance_bin") or sources.get("paper_ad_risk")),
        ("C", "Reliable-domain tradeoff", "predefined reliable domain was insufficient", sources.get("reliable_tradeoff") or sources.get("reliable_domain")),
        ("D", "All vs reliable domain", "fixed rule did not reach the required generalization target", sources.get("reliable_domain")),
    ]
    for ax, (label, title, subtitle, path) in zip(axes, panels):
        imshow_panel(ax, path, title, label, subtitle)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    return save_fig(fig, out_base)


def create_final_ad_figure(paths: ProjectPaths, sources: Dict[str, Optional[Path]]) -> Tuple[Path, Path]:
    out_base = paths.figures / EST_FIGURES["final_ad"]
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.4), facecolor="white")
    axes = axes.reshape(-1)
    fig.suptitle("Final selective AD evidence chain: high local accuracy with explicit coverage boundary", fontsize=17, fontweight="bold", y=0.985)
    panels = [
        ("A", "Evidence-chain summary", "R²=0.748 belongs only to retained subdomain", sources.get("final_evidence_chain")),
        ("B", "NN-risk threshold plateau", "accepted rule is supported by threshold sensitivity", sources.get("threshold_plateau")),
        ("C", "Retained chemistry window", "local chemical support and Paper_010 dominance", sources.get("chemistry_window")),
        ("D", "Stability and transfer boundary", "support exists, but coverage is low", sources.get("stability_boundary")),
    ]
    for ax, (label, title, subtitle, path) in zip(axes, panels):
        imshow_panel(ax, path, title, label, subtitle)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    return save_fig(fig, out_base)


def create_all_figures(paths: ProjectPaths, sources: Dict[str, Optional[Path]]) -> Dict[str, Path]:
    created: Dict[str, Path] = {}
    png, _ = create_framework_figure(paths); created["205"] = png
    png, _ = create_random_split_figure(paths, sources); created["206"] = png
    png, _ = create_grouped_figure(paths, sources); created["207"] = png
    png, _ = create_ad_development_figure(paths, sources); created["208"] = png
    png, _ = create_final_ad_figure(paths, sources); created["209"] = png
    return created


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def add_run_style(paragraph, font_name: str = "宋体", font_size: int = 10, bold: bool = False, color: Optional[str] = None) -> None:
    for run in paragraph.runs:
        run.font.name = font_name
        run.font.size = Pt(font_size)
        run.bold = bold
        if color:
            run.font.color.rgb = RGBColor.from_string(color.replace("#", ""))


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
    return doc.paragraphs[-1] if doc.paragraphs else doc.add_paragraph()


def insert_suite_into_docx(template: Optional[Path], paths: ProjectPaths, created: Dict[str, Path]) -> Path:
    if template and template.exists():
        doc = Document(str(template))
    else:
        doc = Document()
        title = doc.add_paragraph("基于机器学习建模的含氟废水混凝处理的工艺选择、优化及控制")
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_run_style(title, font_size=16, bold=True)
        doc.add_paragraph("3 Results and Discussion（结果与讨论）")

    anchor = find_results_anchor(doc)
    cur = _insert_paragraph_after(anchor, "EST风格扩充主线图与证据链套图（32V2）")
    add_run_style(cur, font_size=12, bold=True)

    captions = {
        "205": "图 3-1 全文主线框架图。该图概括本研究从异质文献数据构建、机理特征增强、随机划分模型筛选、跨文献分组验证到选择性适应域边界识别的整体技术路线，并区分随机划分、全域分组验证和选择性适应域三类评价口径。",
        "206": "图 3-2 随机划分建模与双模型形成过程图。该图补充展示从线性/树模型筛选、CatBoost 调参、融合候选比较到 CatBoost-HGB 加权双模型的形成过程；其中随机划分最高 R² 约为 0.825–0.829，代表数据内部插值条件下的模型学习能力。",
        "207": "图 3-3 跨文献分组验证与泛化边界图。该图补充展示全域分组验证阶段的探索性最优 R² 约为 0.25，以及 32J/32I2 证据链中采用的审计口径全域基准 R² = 0.194，说明随机划分高精度不能直接代表跨文献外推性能。",
        "208": "图 3-4 适应域风险诊断与预定义可靠域分析图。该图展示从适应域风险评分、误差关系、可靠域阈值到固定可靠域失效的分析过程，说明单一静态阈值难以充分解决文献异质性带来的泛化问题。",
        "209": "图 3-5 最终选择性适应域证据链图。该图汇总选择性适应域保留子域的性能证据、NN-risk 阈值平台、保留化学窗口和稳定性/解释边界；R² = 0.748 仅属于无目标泄露的选择性适应域保留子域结果。",
    }

    for fig_id in ["205", "206", "207", "208", "209"]:
        p = created.get(fig_id)
        if p and p.exists():
            cur = _insert_picture_after(cur, p, width_inches=6.55)
        else:
            cur = _insert_paragraph_after(cur, f"[缺失图件 {fig_id}] {p if p else ''}")
        cap = _insert_paragraph_after(cur, captions[fig_id])
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_run_style(cap, font_size=9)
        cur = cap

    out = paths.reports / "32v2_est_style_expanded_storyline_figure_suite_inserted.docx"
    doc.save(out)
    return out


def count_docx_media(docx_path: Path) -> int:
    try:
        with zipfile.ZipFile(docx_path, "r") as z:
            return len([n for n in z.namelist() if n.startswith("word/media/")])
    except Exception:
        return -1


def write_reports(paths: ProjectPaths, sources: Dict[str, Optional[Path]], created: Dict[str, Path], docx_out: Path, template: Optional[Path]) -> None:
    # Source matching audit
    source_rows = []
    for key, path in sources.items():
        source_rows.append({"source_role": key, "path": str(path) if path else "", "exists": bool(path and path.exists())})
    write_csv(paths.tables / "32v2_source_figure_matching_audit.csv", source_rows)

    # Output figure manifest
    fig_rows = []
    for fig_id, path in created.items():
        fig_rows.append({"figure_id": fig_id, "output_path": str(path), "exists": path.exists(), "size_kb": round(path.stat().st_size / 1024, 2) if path.exists() else ""})
    write_csv(paths.tables / "32v2_est_expanded_figure_suite_manifest.csv", fig_rows)

    # Metric milestones
    write_csv(paths.tables / "32v2_metric_milestone_hierarchy.csv", MILESTONES)

    # Corrections required by user
    corrections = [
        {"issue_in_32v": "Fig. 3-2 missed the earliest/highest random-split R2", "fix_in_32v2": "Added CatBoost-HGB Weighted 2Model with R2≈0.825-0.829 in Fig. 206 panel A."},
        {"issue_in_32v": "Fig. 3-2 did not explain how the two-model fusion was reached", "fix_in_32v2": "Added a five-step path from baseline screening to validation-RMSE weight search and final two-model benchmark."},
        {"issue_in_32v": "Fig. 3-3 missed the exploratory all-domain grouped best around R2≈0.25", "fix_in_32v2": "Added grouped metric hierarchy with exploratory grouped best≈0.25 and audited all-domain baseline R2=0.194."},
        {"issue_in_32v": "AD development before final selective AD was incomplete", "fix_in_32v2": "Added Fig. 208 for risk diagnosis, risk-bin performance, reliable-domain tradeoff and predefined reliable-domain limitation."},
    ]
    write_csv(paths.tables / "32v2_requested_corrections_and_fixes.csv", corrections)

    report = paths.reports / "32v2_est_expanded_storyline_and_figure_plan.md"
    lines = []
    lines.append("# 32V2 EST-style expanded figure suite")
    lines.append("")
    lines.append(f"Generated: {RUN_TIME}")
    lines.append(f"Template/input DOCX: {template if template else 'new document created'}")
    lines.append(f"Output DOCX: {docx_out}")
    lines.append(f"DOCX embedded media count: {count_docx_media(docx_out)}")
    lines.append("")
    lines.append("## Why this version was expanded")
    lines.append("32V was too compressed. It did not explicitly show the highest random-split result, the path toward the CatBoost-HGB two-model fusion, or the exploratory full-domain grouped-validation result around R²≈0.25. 32V2 adds these as explicit manuscript evidence.")
    lines.append("")
    lines.append("## Core storyline")
    lines.append("1. Build heterogeneous literature data and mechanism-informed features.")
    lines.append("2. Use random split to screen models and discover the high-performing CatBoost-HGB weighted two-model benchmark.")
    lines.append("3. Use grouped validation to reveal cross-literature transfer difficulty; the exploratory best all-domain grouped R² is only about 0.25, and the audited baseline is R²=0.194.")
    lines.append("4. Use AD risk analysis and reliable-domain attempts to diagnose why fixed rules are insufficient.")
    lines.append("5. Use target-free selective AD to identify a small retained subdomain with R²=0.748, while explicitly stating low coverage and Paper_010 dominance.")
    lines.append("")
    lines.append("## Locked writing boundary")
    lines.append("R²=0.747676 belongs only to the selective AD retained-subdomain result. It must not be described as all-domain grouped validation performance.")
    report.write_text("\n".join(lines), encoding="utf-8")

    # Run log
    log = paths.reports / "32v2_run_log.txt"
    log_lines = [
        "32V2 EST-style expanded figure suite",
        f"Run time: {RUN_TIME}",
        f"Project root: {paths.root}",
        f"Figure dir: {paths.figures}",
        f"Table dir: {paths.tables}",
        f"Report dir: {paths.reports}",
        f"Input DOCX: {template if template else 'new document'}",
        f"Output DOCX: {docx_out}",
        f"DOCX media count: {count_docx_media(docx_out)}",
        "Important boundary: R2=0.747676 belongs to selective AD retained-subdomain only.",
    ]
    log.write_text("\n".join(log_lines), encoding="utf-8")


def main() -> None:
    paths = prepare_paths()
    print("=" * 70)
    print("32V2 EST-style expanded storyline figure-suite generator")
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

    created = create_all_figures(paths, sources)
    for fig_id, p in created.items():
        print(f"[OUTPUT FIGURE] {fig_id}: {p} | exists={p.exists()}")

    docx_out = insert_suite_into_docx(template, paths, created)
    media_count = count_docx_media(docx_out)
    print("[OUTPUT DOCX]", docx_out)
    print("[DOCX MEDIA COUNT]", media_count)

    write_reports(paths, sources, created, docx_out, template)

    missing = [fid for fid, p in created.items() if not p.exists()]
    print("=" * 70)
    if missing:
        print("32V2 finished with warnings. Missing output figures:", missing)
    else:
        print("32V2 finished successfully.")
    print("Key additions:")
    print("  - Fig. 3-2 now includes the highest random-split R2≈0.825-0.829 and two-model formation path.")
    print("  - Fig. 3-3 now includes exploratory all-domain grouped best R2≈0.25 and audited all-domain R2=0.194.")
    print("  - Fig. 3-4 now includes AD development before final selective AD.")
    print("Important boundary: R2=0.747676 belongs to the selective AD retained-subdomain only.")
    print("Please open:")
    print(" ", docx_out)


if __name__ == "__main__":
    main()
