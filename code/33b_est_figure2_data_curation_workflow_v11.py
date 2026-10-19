# -*- coding: utf-8 -*-
"""
EST-style Figure 2: data curation and dual-matrix construction workflow (v11).
Place this file in fluoride_coagulation_ml_project/src/ and run directly in PyCharm.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import pandas as pd

SCRIPT_NAME = "33b_est_figure2_data_curation_workflow_v11"
FIGURE_STEM = "Figure_2_data_curation_and_dual_matrix_workflow_v11"
FIGURE_SIZE = (7.5, 5.8)
PNG_DPI = 600

COL = {
    "teal": "#2D7F8C",
    "teal_dark": "#256C77",
    "teal_light": "#E2F0F2",
    "blue": "#6A91A9",
    "blue_dark": "#5B839C",
    "blue_light": "#E4EEF3",
    "green": "#7E9E88",
    "green_light": "#E8EFEA",
    "gold": "#C9A252",
    "gold_dark": "#B78F41",
    "gold_light": "#F4EBD5",
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
    strict_cols: int = 26
    source_cols: int = 27
    n_predictors: int = 10
    n_target: int = 1


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
    figures = out / "figures"
    tables = out / "tables"
    reports = out / "reports"
    logs = out / "logs"
    for d in (figures, tables, reports, logs):
        d.mkdir(parents=True, exist_ok=True)
    return Paths(root, figures, tables, reports, logs)


def load_summary(root: Path) -> Summary:
    s = Summary()
    source = root / "data" / "processed" / "model_ready_v1_with_source.csv"
    strict = root / "data" / "processed" / "model_ready_v1.csv"

    try:
        if source.exists():
            df = pd.read_csv(source)
            s.n_records = int(len(df))
            s.source_cols = int(df.shape[1])
            if "文献编号" in df.columns:
                s.n_groups = int(df["文献编号"].astype(str).nunique())
        if strict.exists():
            s.strict_cols = int(pd.read_csv(strict).shape[1])
    except Exception:
        pass

    return s


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def box(ax, x, y, w, h, fc, ec=COL["gray_mid"], lw=0.8, radius=0.008, ls="solid"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.003,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color=COL["gray_mid"], lw=0.95, ms=10, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def text(
    ax,
    x,
    y,
    s,
    fs=8.0,
    weight="normal",
    ha="center",
    va="center",
    color=COL["black"],
    rot=0,
):
    ax.text(
        x,
        y,
        s,
        fontsize=fs,
        fontweight=weight,
        ha=ha,
        va=va,
        color=color,
        rotation=rot,
    )


# =========================
# Icons
# =========================
def icon_documents(ax, cx, cy):
    for dx, dy in [(0.014, 0.012), (0.007, 0.006), (0, 0)]:
        ax.add_patch(
            Rectangle(
                (cx - 0.028 + dx, cy - 0.034 + dy),
                0.056,
                0.068,
                fc=COL["white"],
                ec=COL["gray_mid"],
                lw=0.7,
            )
        )
        for i in range(4):
            yy = cy + 0.021 + dy - i * 0.013
            ax.plot(
                [cx - 0.018 + dx, cx + 0.017 + dx],
                [yy, yy],
                color=COL["gray_mid"],
                lw=0.5,
            )


def icon_table(ax, cx, cy):
    x, y, w, h = cx - 0.034, cy - 0.030, 0.068, 0.060
    ax.add_patch(Rectangle((x, y), w, h, fc=COL["white"], ec=COL["gray_mid"], lw=0.7))
    ax.add_patch(Rectangle((x, y + h * 0.68), w, h * 0.32, fc=COL["teal_light"], ec="none"))
    for i in range(1, 3):
        ax.plot([x, x + w], [y + i * h / 3, y + i * h / 3], color=COL["gray_mid"], lw=0.5)
    for j in range(1, 4):
        ax.plot([x + j * w / 4, x + j * w / 4], [y, y + h], color=COL["gray_mid"], lw=0.5)


def icon_units(ax, cx, cy):
    text(ax, cx - 0.035, cy + 0.016, "mg L$^{-1}$", fs=6.4, color=COL["gray"])
    arrow(ax, cx - 0.003, cy + 0.016, cx + 0.028, cy + 0.016, color=COL["teal"], lw=0.8, ms=8)
    text(ax, cx + 0.046, cy + 0.016, "mM", fs=6.4, color=COL["gray"])
    text(ax, cx - 0.035, cy - 0.019, "h", fs=6.4, color=COL["gray"])
    arrow(ax, cx - 0.020, cy - 0.019, cx + 0.020, cy - 0.019, color=COL["teal"], lw=0.8, ms=8)
    text(ax, cx + 0.046, cy - 0.019, "min", fs=6.4, color=COL["gray"])


def icon_check(ax, cx, cy):
    pts_x = [cx - 0.024, cx + 0.024, cx + 0.020, cx, cx - 0.020, cx - 0.024]
    pts_y = [cy + 0.030, cy + 0.030, cy - 0.005, cy - 0.032, cy - 0.005, cy + 0.030]
    ax.fill(pts_x, pts_y, fc=COL["teal_light"], ec=COL["teal"], lw=0.8)
    ax.plot(
        [cx - 0.013, cx - 0.002, cx + 0.015],
        [cy - 0.002, cy - 0.015, cy + 0.013],
        color=COL["teal"],
        lw=1.5,
    )


def icon_network(ax, cx, cy):
    pts = [(cx - 0.028, cy), (cx, cy + 0.026), (cx + 0.028, cy), (cx, cy - 0.026)]
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)):
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=COL["gray_mid"], lw=0.75)
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.0072, fc=COL["gold_light"], ec=COL["gold"], lw=0.7))


def icon_matrices(ax, cx, cy):
    ax.add_patch(Rectangle((cx - 0.032, cy - 0.024), 0.038, 0.056, fc=COL["white"], ec=COL["blue"], lw=0.8))
    ax.add_patch(Rectangle((cx - 0.002, cy - 0.031), 0.038, 0.056, fc=COL["white"], ec=COL["gold"], lw=0.8))


# =========================
# Main components
# =========================
def top_card(ax, x, y, w, h, num, title, note, icon_fn, fc):
    box(ax, x, y, w, h, fc, ec=COL["teal"], lw=0.95, radius=0.008)

    text(ax, x + 0.015, y + h - 0.020, num, fs=10.0, weight="bold", ha="left", color=COL["teal"])
    text(ax, x + w / 2, y + h - 0.064, title, fs=9.0, weight="bold")

    icon_fn(ax, x + w / 2, y + 0.120)

    text(ax, x + w / 2, y + 0.033, note, fs=6.8, color=COL["gray"])

    # subtle divider to make card feel fuller and more designed
    ax.plot([x + 0.012, x + w - 0.012], [y + 0.062, y + 0.062], color=COL["gray_mid"], lw=0.35, alpha=0.35)


def matrix_card(ax, x, y, w, h, accent, title, descriptor, dims, chips):
    box(ax, x, y, w, h, COL["white"], ec=accent, lw=1.0, radius=0.0075)
    ax.add_patch(Rectangle((x, y + h - 0.038), w, 0.038, fc=accent, ec="none"))
    text(ax, x + w / 2, y + h - 0.019, title, fs=8.7, weight="bold", color=COL["white"])

    # inner content region
    gx, gy, gw, gh = x + 0.022, y + 0.066, 0.058, 0.052
    ax.add_patch(Rectangle((gx, gy), gw, gh, fc=COL["gray_light"], ec=COL["gray_mid"], lw=0.65))
    for i in range(1, 3):
        ax.plot([gx, gx + gw], [gy + i * gh / 3, gy + i * gh / 3], color=COL["gray_mid"], lw=0.45)
    for j in range(1, 4):
        ax.plot([gx + j * gw / 4, gx + j * gw / 4], [gy, gy + gh], color=COL["gray_mid"], lw=0.45)

    text(ax, x + 0.095, y + 0.123, descriptor, fs=6.9, ha="left", weight="bold", color=COL["gray"])
    text(ax, x + 0.095, y + 0.085, dims, fs=7.9, ha="left", weight="bold", color=accent)

    chip_y = y + 0.024
    chip_h = 0.030
    chip_w = (w - 0.040) / len(chips)
    for i, chip in enumerate(chips):
        cx = x + 0.020 + i * chip_w
        box(ax, cx, chip_y, chip_w - 0.006, chip_h, COL["gray_light"], ec=COL["gray_mid"], lw=0.55, radius=0.0035)
        text(ax, cx + (chip_w - 0.006) / 2, chip_y + chip_h / 2, chip, fs=6.4, color=COL["gray"])


def create_figure(s: Summary):
    fig = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    titles = [
        ("1", "Literature\nsources", f"{s.n_groups} groups\n{s.n_records:,} records", icon_documents, COL["blue_light"]),
        ("2", "Structured\nextraction", "water / reagent\noperation / response", icon_table, COL["teal_light"]),
        ("3", "Harmonization", "names / units\ncategories", icon_units, COL["gold_light"]),
        ("4", "Quality\ncontrol", "range checks\nresponse verification", icon_check, COL["green_light"]),
        ("5", "Feature\nrepresentation", f"{s.n_predictors} predictors\n{s.n_target} target", icon_network, COL["blue_light"]),
        ("6", "Dual\nmatrices", f"{s.strict_cols} / {s.source_cols} fields\nsame records", icon_matrices, COL["teal_light"]),
    ]

    x0, card_w, gap = 0.035, 0.147, 0.010
    card_y, card_h = 0.705, 0.265

    for i, (num, title, note, icon_fn, fc) in enumerate(titles):
        x = x0 + i * (card_w + gap)
        top_card(ax, x, card_y, card_w, card_h, num, title, note, icon_fn, fc)
        if i < 5:
            arrow(ax, x + card_w + 0.001, card_y + 0.133, x + card_w + gap - 0.002, card_y + 0.133, lw=0.9, ms=10)

    # middle flow strip
    box(ax, 0.063, 0.585, 0.874, 0.072, COL["gray_light"], ec=COL["gray_mid"], lw=0.8, radius=0.004)
    stages = ["sample pool", "cleaning", "standardization", "audit trail"]
    for i, stage in enumerate(stages):
        sx = 0.095 + i * 0.215
        box(ax, sx, 0.601, 0.160, 0.034, COL["white"], ec=COL["gray_mid"], lw=0.65, radius=0.0035, ls="dashed")
        text(ax, sx + 0.080, 0.618, stage, fs=7.1, weight="bold")
        if i < 3:
            arrow(ax, sx + 0.164, 0.618, sx + 0.205, 0.618, lw=0.8, ms=8)

    arrow(ax, 0.500, 0.585, 0.500, 0.548, color=COL["teal"], lw=1.2, ms=14)

    # lower main panel
    box(ax, 0.045, 0.055, 0.910, 0.485, COL["gray_light"], ec=COL["teal"], lw=1.0, radius=0.004)
    text(ax, 0.270, 0.510, "Model-ready output", fs=10.2, weight="bold")
    text(ax, 0.730, 0.510, "Validation-ready output", fs=10.2, weight="bold")

    matrix_card(
        ax,
        0.080,
        0.262,
        0.382,
        0.205,
        COL["blue"],
        "Model-ready storage matrix",
        "26 archived fields; predictor subset\nextracted before model fitting",
        f"{s.n_records:,} records",
        ["training", "random split", "SHAP / PDP"],
    )

    matrix_card(
        ax,
        0.538,
        0.262,
        0.382,
        0.205,
        COL["gold"],
        "Source-aware audit matrix",
        "Same 26 archived fields +\nsource-group ID",
        f"{s.n_records:,} records × {s.source_cols} fields",
        ["grouped validation", "LOLO", "AD audit"],
    )

    arrow(ax, 0.462, 0.365, 0.500, 0.365, lw=0.8, ms=8)
    arrow(ax, 0.538, 0.365, 0.500, 0.365, lw=0.8, ms=8)

    # bottom note box made fuller
    box(ax, 0.232, 0.108, 0.524, 0.090, COL["white"], ec=COL["gray_mid"], lw=0.8, radius=0.005)
    text(ax, 0.494, 0.157, "Same records; source-group ID used differently", fs=8.5, weight="bold", color=COL["teal"])
    text(ax, 0.494, 0.129, "excluded from predictors; retained only for grouping and audit", fs=6.9, color=COL["gray"])

    return fig


def save_metadata(paths: Paths, s: Summary, runtime: float) -> None:
    pd.DataFrame(
        {
            "item": ["records", "groups", "strict_cols", "source_cols", "n_predictors", "n_target"],
            "value": [s.n_records, s.n_groups, s.strict_cols, s.source_cols, s.n_predictors, s.n_target],
        }
    ).to_csv(paths.tables / f"{FIGURE_STEM}_summary.csv", index=False, encoding="utf-8-sig")

    report = (
        f"{SCRIPT_NAME}\n"
        f"records={s.n_records}\n"
        f"groups={s.n_groups}\n"
        f"strict_cols={s.strict_cols}\n"
        f"source_cols={s.source_cols}\n"
        f"n_predictors={s.n_predictors}\n"
        f"n_target={s.n_target}\n"
        f"runtime_s={runtime:.2f}\n"
        f"python={sys.version.split()[0]}\n"
        f"platform={platform.platform()}\n"
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