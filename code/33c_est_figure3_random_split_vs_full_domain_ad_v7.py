# -*- coding: utf-8 -*-
"""
Compact EST-style Figure 3: random split versus full-domain grouped validation and AD.
Place in fluoride_coagulation_ml_project/src/ and run directly in PyCharm.
"""
from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch
import pandas as pd

SCRIPT_NAME = "33c_est_figure3_random_split_vs_full_domain_ad_v7"
FIGURE_STEM = "Figure_3_random_split_vs_full_domain_ad_workflow_v7"
FIGURE_SIZE = (7.2, 5.45)
PNG_DPI = 600

COL = {
    "teal": "#2D7F8C",
    "teal_light": "#E2F0F2",
    "blue": "#668FA6",
    "blue_light": "#E3EDF2",
    "green": "#7C9E86",
    "green_light": "#E7EFE9",
    "gold": "#C9A252",
    "gold_light": "#F5ECD5",
    "gray": "#5F666C",
    "gray_mid": "#A8AFB5",
    "gray_light": "#F6F7F8",
    "black": "#222222",
    "white": "#FFFFFF",
}


@dataclass
class Summary:
    n_records: int = 1011
    n_groups: int = 27


@dataclass
class Paths:
    root: Path
    figures: Path
    tables: Path
    reports: Path
    logs: Path


def locate_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "data").exists() and (p / "outputs").exists():
            return p
    return here.parent.parent


def build_paths() -> Paths:
    root = locate_root()
    out = root / "outputs" / "jece_final_package"
    figures, tables, reports, logs = out / "figures", out / "tables", out / "reports", out / "logs"
    for d in (figures, tables, reports, logs):
        d.mkdir(parents=True, exist_ok=True)
    return Paths(root, figures, tables, reports, logs)


def load_summary(root: Path) -> Summary:
    s = Summary()
    p = root / "data" / "processed" / "model_ready_v1_with_source.csv"
    try:
        if p.exists():
            df = pd.read_csv(p)
            s.n_records = int(len(df))
            if "文献编号" in df.columns:
                s.n_groups = int(df["文献编号"].astype(str).nunique())
    except Exception:
        pass
    return s


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def box(ax, x, y, w, h, fc, ec=COL["gray_mid"], lw=0.8, radius=0.008, ls="solid"):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.003,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color=COL["gray_mid"], lw=0.9, ms=9, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms, linewidth=lw, color=color, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, fs=7.0, weight="normal", ha="center", va="center", color=COL["black"]):
    ax.text(x, y, s, fontsize=fs, fontweight=weight, ha=ha, va=va, color=color)


def document_cluster(ax, x, y):
    colors = [COL["blue_light"], COL["teal_light"], COL["gold_light"], COL["green_light"]]
    for i, c in enumerate(colors):
        bx = x + i * 0.037
        box(ax, bx, y, 0.029, 0.048, c, ec=COL["gray_mid"], lw=0.55, radius=0.003)
        text(ax, bx + 0.0145, y + 0.030, f"L{i+1}", fs=5.2, weight="bold", color=COL["gray"])
        for j in range(3):
            ax.add_patch(Circle((bx + 0.008 + j * 0.0065, y + 0.012), 0.0025, fc=COL["gray_mid"], ec="none"))


def split_panel(ax, x, y, w, h, grouped=False):
    box(ax, x, y, w, h, COL["white"], ec=COL["gray_mid"], lw=0.7, radius=0.005)
    train_x, test_x = x + 0.025, x + w - 0.155
    box(ax, train_x, y + 0.038, 0.130, 0.072, COL["blue_light"], ec=COL["gray_mid"], lw=0.6, radius=0.005)
    box(ax, test_x, y + 0.038, 0.130, 0.072, COL["gold_light"], ec=COL["gray_mid"], lw=0.6, radius=0.005)
    text(ax, train_x + 0.065, y + 0.094, "Train", fs=6.2, weight="bold", color=COL["blue"])
    text(ax, test_x + 0.065, y + 0.094, "Test", fs=6.2, weight="bold", color=COL["gold"])

    if grouped:
        for i, c in enumerate([COL["teal"], COL["blue"], COL["green"]]):
            ax.add_patch(Circle((train_x + 0.035 + i * 0.028, y + 0.061), 0.0068, fc=c, ec=COL["white"], lw=0.3))
        ax.add_patch(Circle((test_x + 0.065, y + 0.061), 0.0068, fc=COL["gold"], ec=COL["white"], lw=0.3))
        text(ax, x + w / 2, y + 0.018, "source-isolated split", fs=5.5, color=COL["gray"])
    else:
        colors = [COL["teal"], COL["blue"], COL["gold"]]
        for i, c in enumerate(colors):
            ax.add_patch(Circle((train_x + 0.035 + i * 0.028, y + 0.061), 0.0068, fc=c, ec=COL["white"], lw=0.3))
            ax.add_patch(Circle((test_x + 0.035 + i * 0.028, y + 0.061), 0.0068, fc=c, ec=COL["white"], lw=0.3))
        text(ax, x + w / 2, y + 0.018, "same-source overlap", fs=5.5, color=COL["gray"])


def nested_domains(ax, cx, cy):
    ax.add_patch(Ellipse((cx, cy), 0.225, 0.116, fc=COL["blue_light"], ec=COL["blue"], lw=0.8))
    ax.add_patch(Ellipse((cx + 0.006, cy), 0.152, 0.078, fc=COL["teal_light"], ec=COL["teal"], lw=0.8))
    ax.add_patch(Ellipse((cx + 0.012, cy), 0.078, 0.040, fc=COL["gold_light"], ec=COL["gold"], lw=0.8))
    text(ax, cx - 0.072, cy + 0.047, "Full domain", fs=5.9, weight="bold", color=COL["blue"])
    text(ax, cx + 0.009, cy + 0.018, "Applicability", fs=5.4, weight="bold", color=COL["teal"])
    text(ax, cx + 0.012, cy - 0.006, "Reliable", fs=5.1, weight="bold", color=COL["gold"])


def create_figure(s: Summary):
    fig = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Shared input
    box(ax, 0.080, 0.855, 0.840, 0.105, COL["gray_light"], ec=COL["teal"], lw=0.9, radius=0.005)
    document_cluster(ax, 0.115, 0.883)
    text(ax, 0.345, 0.922, "Shared literature-derived dataset", fs=8.1, weight="bold", ha="left")
    text(ax, 0.345, 0.892, f"{s.n_groups} literature groups  |  {s.n_records:,} experimental records", fs=6.2, ha="left", color=COL["gray"])

    # Route headers
    arrow(ax, 0.320, 0.855, 0.320, 0.815, lw=0.8, ms=8)
    arrow(ax, 0.680, 0.855, 0.680, 0.815, lw=0.8, ms=8)
    box(ax, 0.075, 0.760, 0.390, 0.055, COL["blue_light"], ec=COL["blue"], lw=0.8, radius=0.006)
    box(ax, 0.535, 0.760, 0.390, 0.055, COL["teal_light"], ec=COL["teal"], lw=0.8, radius=0.006)
    text(ax, 0.270, 0.787, "Random split benchmark", fs=7.3, weight="bold", color=COL["blue"])
    text(ax, 0.730, 0.787, "Full-domain grouped challenge", fs=7.3, weight="bold", color=COL["teal"])

    # Two comparison panels
    box(ax, 0.075, 0.485, 0.390, 0.240, COL["gray_light"], ec=COL["gray_mid"], lw=0.7, radius=0.005)
    split_panel(ax, 0.095, 0.555, 0.350, 0.130, grouped=False)
    box(ax, 0.115, 0.500, 0.310, 0.032, COL["white"], ec=COL["gray_mid"], lw=0.6, radius=0.003, ls="dashed")
    text(ax, 0.270, 0.516, "internal learning / optimistic baseline", fs=5.9, weight="bold", color=COL["gray"])

    # Group-aware branch: aligned split box + validation ladder
    box(ax, 0.535, 0.485, 0.390, 0.240, COL["gray_light"], ec=COL["gray_mid"], lw=0.7, radius=0.005)
    split_panel(ax, 0.555, 0.603, 0.350, 0.092, grouped=True)
    # keep the grouping-key note separate from Train/Test labels
    box(ax, 0.585, 0.570, 0.290, 0.022, COL["teal_light"], ec=COL["teal"], lw=0.55, radius=0.002)
    text(ax, 0.730, 0.581, "literature ID = grouping key only", fs=5.25, weight="bold", color=COL["teal"])
    stages = ["grouped holdout", "repeated / balanced", "LOLO + shift audit"]
    stage_y = [0.536, 0.506, 0.476]
    for yy, stage in zip(stage_y, stages):
        box(ax, 0.585, yy, 0.290, 0.026, COL["white"], ec=COL["gray_mid"], lw=0.6, radius=0.002, ls="dashed")
        text(ax, 0.730, yy + 0.013, stage, fs=5.55, weight="bold", color=COL["gray"])

    arrow(ax, 0.465, 0.605, 0.535, 0.605, color=COL["teal"], lw=1.0, ms=11)
    box(ax, 0.477, 0.622, 0.046, 0.022, COL["white"], ec=COL["gray_mid"], lw=0.5, radius=0.002)
    text(ax, 0.500, 0.633, "stricter", fs=5.35, weight="bold", color=COL["teal"])

    # Full-domain to AD flow
    box(ax, 0.075, 0.145, 0.850, 0.310, COL["gray_light"], ec=COL["teal"], lw=0.9, radius=0.005)
    text(ax, 0.500, 0.425, "Full-domain challenge  ->  target-free AD  ->  reliable prediction region", fs=7.6, weight="bold")

    text(ax, 0.175, 0.372, "All-domain result", fs=6.8, weight="bold", color=COL["teal"])
    box(ax, 0.105, 0.245, 0.150, 0.100, COL["white"], ec=COL["gray_mid"], lw=0.7, radius=0.005)
    text(ax, 0.180, 0.312, "complete space", fs=6.2, weight="bold")
    text(ax, 0.180, 0.284, "maximum heterogeneity", fs=5.5, color=COL["gray"])
    text(ax, 0.180, 0.260, "weak local support", fs=5.5, color=COL["gray"])

    arrow(ax, 0.260, 0.295, 0.345, 0.295, color=COL["teal"], lw=1.0, ms=12)
    text(ax, 0.455, 0.372, "Target-free AD", fs=6.8, weight="bold", color=COL["teal"])
    box(ax, 0.350, 0.245, 0.210, 0.100, COL["white"], ec=COL["gray_mid"], lw=0.7, radius=0.005)
    text(ax, 0.455, 0.315, "distance / density", fs=5.9, weight="bold")
    text(ax, 0.455, 0.291, "category novelty", fs=5.9, weight="bold")
    text(ax, 0.455, 0.262, "training-data support", fs=5.5, color=COL["gray"])

    arrow(ax, 0.565, 0.295, 0.635, 0.295, color=COL["teal"], lw=1.0, ms=12)
    nested_domains(ax, 0.775, 0.295)
    text(ax, 0.775, 0.216, "narrower region; stronger claim", fs=5.7, weight="bold", color=COL["gold"])

    box(ax, 0.180, 0.155, 0.640, 0.032, COL["white"], ec=COL["gray_mid"], lw=0.6, radius=0.003)
    text(ax, 0.500, 0.171, "Random split = internal learning; grouped validation + AD = trustworthy claim boundary", fs=5.8, color=COL["gray"])

    return fig


def save_metadata(paths: Paths, s: Summary, runtime: float) -> None:
    pd.DataFrame({"item": ["records", "groups"], "value": [s.n_records, s.n_groups]}).to_csv(
        paths.tables / f"{FIGURE_STEM}_summary.csv", index=False, encoding="utf-8-sig"
    )
    report = (
        f"{SCRIPT_NAME}\nrecords={s.n_records}\ngroups={s.n_groups}\nruntime_s={runtime:.2f}\n"
        f"python={sys.version.split()[0]}\nplatform={platform.platform()}\n"
    )
    (paths.reports / f"{SCRIPT_NAME}_audit.txt").write_text(report, encoding="utf-8")
    (paths.logs / f"{SCRIPT_NAME}.log").write_text(report, encoding="utf-8")


def main() -> None:
    start = time.time()
    set_style()
    paths = build_paths()
    summary = load_summary(paths.root)
    fig = create_figure(summary)
    fig.savefig(paths.figures / f"{FIGURE_STEM}.png", dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(paths.figures / f"{FIGURE_STEM}.pdf", bbox_inches="tight")
    plt.close(fig)
    runtime = time.time() - start
    save_metadata(paths, summary, runtime)
    print(f"[{SCRIPT_NAME}] done in {runtime:.2f} s")


if __name__ == "__main__":
    main()
