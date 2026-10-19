# -*- coding: utf-8 -*-
"""
33a_est_figure1_dataset_overview_v7.py

Publication-size Figure 1 for the coagulation-fluoride removal manuscript.
The figure is generated at its intended two-column print width (7.2 in),
so font sizes are not reduced by later scaling in Word or journal layout.

Place this file in:
    fluoride_coagulation_ml_project/src/

Input:
    data/processed/model_ready_v1_with_source.csv

Outputs:
    outputs/jece_final_package/figures/
    outputs/jece_final_package/tables/
    outputs/jece_final_package/reports/
    outputs/jece_final_package/logs/

Figure structure:
    a  Numerical descriptor distributions
    b  Fluoride-removal distributions across descriptor intervals
    c  Literature-source imbalance
    d  Prediction-target distribution
    e  Categorical descriptor composition
    f  Final data-matrix structure
"""

from __future__ import annotations

import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import gaussian_kde, shapiro

warnings.filterwarnings("ignore")

SCRIPT_NAME = "33a_est_figure1_dataset_overview_v7"
FIGURE_STEM = "Figure_1_dataset_structure_and_heterogeneity_v7"
PNG_DPI = 600

# Final publication dimensions: approximately 183 mm wide.
FIGURE_WIDTH_IN = 7.2
FIGURE_HEIGHT_IN = 7.25

PALETTE = {
    "teal": "#2F6F73",
    "blue": "#4E79A7",
    "blue_light": "#AFC8D6",
    "green": "#6F9E75",
    "green_light": "#C9D8C7",
    "orange": "#D99545",
    "orange_light": "#E8CCAE",
    "purple": "#8E7DBE",
    "purple_light": "#CFC5E4",
    "rose": "#B85C6F",
    "gray": "#70757A",
    "gray_mid": "#A8ADB2",
    "gray_light": "#E7EAEC",
    "grid": "#D9DEE2",
    "black": "#222222",
    "white": "#FFFFFF",
}

# Refined low-saturation palette inspired by recent EST figures: cyan/teal,
# muted blue, pale gold, warm coral, and subdued violet. The colors are used
# to distinguish descriptor families without introducing high-saturation clutter.
EST_FEATURE_COLORS = [
    {"fill": "#BFDDE1", "line": "#1E7480"},
    {"fill": "#C9D7E8", "line": "#486B91"},
    {"fill": "#D9E1B8", "line": "#708548"},
    {"fill": "#F0D79D", "line": "#B27A25"},
    {"fill": "#E8C3B7", "line": "#A45D51"},
    {"fill": "#CEC6E1", "line": "#71628E"},
]

EST_CATEGORY_COLORS = {
    "coagulant_low": "#D4E8E6",
    "coagulant_high": "#267A83",
    "ion_low": "#F4E8B8",
    "ion_high": "#C79232",
}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_file: Path
    output_root: Path
    figures: Path
    tables: Path
    reports: Path
    logs: Path


def locate_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "data" / "processed").exists() and (parent / "outputs").exists():
            return parent
    return current.parent.parent


def build_paths() -> ProjectPaths:
    root = locate_project_root()
    output_root = root / "outputs" / "jece_final_package"
    paths = ProjectPaths(
        root=root,
        data_file=root / "data" / "processed" / "model_ready_v1_with_source.csv",
        output_root=output_root,
        figures=output_root / "figures",
        tables=output_root / "tables",
        reports=output_root / "reports",
        logs=output_root / "logs",
    )
    for folder in (paths.output_root, paths.figures, paths.tables, paths.reports, paths.logs):
        folder.mkdir(parents=True, exist_ok=True)
    return paths


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_data(paths: ProjectPaths) -> pd.DataFrame:
    if not paths.data_file.exists():
        raise FileNotFoundError(f"Final source-aware matrix not found: {paths.data_file}")
    df = pd.read_csv(paths.data_file)
    df.columns = [str(column).strip() for column in df.columns]
    required = [
        "初始氟_mg_L",
        "进水pH",
        "水温_C",
        "剂量_mM",
        "停留时间_min",
        "干扰离子浓度_mmol_L",
        "氟去除率_pct",
        "混凝剂",
        "干扰离子",
        "文献编号",
    ]
    require_columns(df, required)
    return df


# column, display title, original unit, apply log10(value+1) for the histogram
FEATURES: List[Tuple[str, str, str, bool]] = [
    ("初始氟_mg_L", "Initial fluoride", "mg L$^{-1}$", True),
    ("进水pH", "Influent pH", "pH", False),
    ("水温_C", "Water temperature", r"$^{\circ}$C", False),
    ("剂量_mM", "Coagulant dose", "mM", True),
    ("停留时间_min", "Residence time", "min", True),
    ("干扰离子浓度_mmol_L", "Interfering-ion level", "mmol L$^{-1}$", True),
]

# Physically interpretable intervals. Sparse groups are retained but are shown
# without a violin envelope when n < 20 to avoid over-interpreting tiny groups.
VIOLIN_BINS: Dict[str, Tuple[Sequence[float], Sequence[str]]] = {
    "初始氟_mg_L": ([-np.inf, 5, 10, 20, np.inf], ["<=5", "5-10", "10-20", ">20"]),
    "进水pH": ([-np.inf, 5, 7, 8, np.inf], ["<=5", "5-7", "7-8", ">8"]),
    "水温_C": ([-np.inf, 24.999, 25.001, np.inf], ["<25", "25", ">25"]),
    "剂量_mM": ([-np.inf, 1, 10, 50, np.inf], ["<=1", "1-10", "10-50", ">50"]),
    "停留时间_min": ([-np.inf, 30, 40, 60, np.inf], ["<=30", "30-40", "40-60", ">60"]),
    "干扰离子浓度_mmol_L": ([-0.001, 0, 2, np.inf], ["0", ">0-2", ">2"]),
}


def panel_label(ax: plt.Axes, label: str, x: float = -0.055, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=PALETTE["black"],
        clip_on=False,
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=PALETTE["grid"], linestyle="--", linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)


def normality_test(values: pd.Series) -> Tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) < 3:
        return np.nan, np.nan
    result = shapiro(clean)
    return float(result.statistic), float(result.pvalue)


def transform_for_hist(values: pd.Series, use_log: bool) -> Tuple[np.ndarray, str]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if use_log:
        clean = np.log10(np.clip(clean, a_min=0, a_max=None) + 1.0)
        return clean, "log$_{10}$(value + 1)"
    return clean, ""


def draw_histogram_matrix(fig: plt.Figure, spec, df: pd.DataFrame) -> pd.DataFrame:
    outer = fig.add_subplot(spec)
    outer.axis("off")
    panel_label(outer, "a", x=-0.02, y=1.015)
    inner = GridSpecFromSubplotSpec(2, 3, subplot_spec=spec, wspace=0.20, hspace=0.54)
    rows: List[dict] = []

    for index, (column, title, unit, use_log) in enumerate(FEATURES):
        ax = fig.add_subplot(inner[index // 3, index % 3])
        raw = pd.to_numeric(df[column], errors="coerce").dropna()
        display, scale_note = transform_for_hist(raw, use_log)

        scheme = EST_FEATURE_COLORS[index]
        ax.hist(
            display,
            bins=18,
            color=scheme["fill"],
            edgecolor=PALETTE["white"],
            linewidth=0.42,
            alpha=0.96,
        )
        if len(np.unique(display)) > 3 and np.nanstd(display) > 0:
            try:
                kde = gaussian_kde(display)
                x_grid = np.linspace(np.nanmin(display), np.nanmax(display), 240)
                density = kde(x_grid)
                bin_width = (
                    (np.nanmax(display) - np.nanmin(display)) / 18
                    if np.nanmax(display) > np.nanmin(display)
                    else 1.0
                )
                ax.plot(
                    x_grid,
                    density * len(display) * bin_width,
                    color=scheme["line"],
                    linewidth=1.15,
                )
            except Exception:
                pass

        # Keep statistical testing in the exported source table, not inside the
        # main figure. This preserves readability at final journal size.
        w_value, p_value = normality_test(raw)
        ax.set_title(f"{title} ({unit})", fontsize=6.9, pad=2.2)
        ax.set_xlabel(scale_note if use_log else unit, fontsize=6.0, labelpad=1.2)
        ax.set_ylabel("Count" if index % 3 == 0 else "", fontsize=6.0, labelpad=1.5)
        clean_axes(ax, grid_axis="y")
        ax.spines["left"].set_color("#50565B")
        ax.spines["bottom"].set_color("#50565B")

        rows.append(
            {
                "feature": column,
                "display_name": title,
                "unit": unit,
                "display_scale": scale_note if use_log else "original scale",
                "n": len(raw),
                "mean": raw.mean(),
                "median": raw.median(),
                "std": raw.std(ddof=1),
                "min": raw.min(),
                "max": raw.max(),
                "shapiro_W": w_value,
                "shapiro_p": p_value,
            }
        )

    return pd.DataFrame(rows)


def _draw_single_violin_panel(
    ax: plt.Axes,
    grouped: List[np.ndarray],
    labels: List[str],
    colors: List[str],
) -> None:
    positions = np.arange(1, len(grouped) + 1)

    # Draw violin envelopes only when the sample size supports a density estimate.
    for pos, vals, color in zip(positions, grouped, colors):
        if len(vals) >= 20 and np.unique(vals).size > 1:
            violin = ax.violinplot(
                [vals],
                positions=[pos],
                widths=0.72,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            body = violin["bodies"][0]
            body.set_facecolor(color)
            body.set_edgecolor(PALETTE["gray"])
            body.set_linewidth(0.55)
            body.set_alpha(0.92)
        else:
            # Sparse groups are shown as lightly jittered points instead of a
            # potentially misleading density envelope.
            rng = np.random.default_rng(20260720 + int(pos))
            jitter = rng.normal(pos, 0.035, size=len(vals))
            ax.scatter(
                jitter,
                vals,
                s=5,
                facecolor=color,
                edgecolor=PALETTE["gray"],
                linewidth=0.25,
                alpha=0.80,
                zorder=2,
            )

    ax.boxplot(
        grouped,
        positions=positions,
        widths=0.16,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE["white"], "linewidth": 0.95},
        whiskerprops={"color": PALETTE["black"], "linewidth": 0.55},
        capprops={"color": PALETTE["black"], "linewidth": 0.55},
        boxprops={"facecolor": PALETTE["black"], "edgecolor": PALETTE["black"], "linewidth": 0.45},
    )
    for pos, vals in zip(positions, grouped):
        ax.scatter(
            pos,
            np.mean(vals),
            marker="D",
            s=8,
            color=PALETTE["orange"],
            edgecolor=PALETTE["white"],
            linewidth=0.30,
            zorder=4,
        )

    ax.set_xticks(positions)
    tick_labels = [f"{label}\n(n={len(vals)})" for label, vals in zip(labels, grouped)]
    ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=5.35, linespacing=0.95)


def draw_violin_matrix(fig: plt.Figure, spec, df: pd.DataFrame) -> pd.DataFrame:
    outer = fig.add_subplot(spec)
    outer.axis("off")
    panel_label(outer, "b", x=-0.02, y=1.015)
    inner = GridSpecFromSubplotSpec(2, 3, subplot_spec=spec, wspace=0.20, hspace=0.58)
    rows: List[dict] = []
    violin_colors = [
        PALETTE["orange_light"],
        PALETTE["green_light"],
        PALETTE["purple_light"],
        PALETTE["blue_light"],
    ]
    target = pd.to_numeric(df["氟去除率_pct"], errors="coerce")

    for index, (column, title, unit, _) in enumerate(FEATURES):
        ax = fig.add_subplot(inner[index // 3, index % 3])
        bins, labels = VIOLIN_BINS[column]
        feature_values = pd.to_numeric(df[column], errors="coerce")
        categories = pd.cut(feature_values, bins=bins, labels=labels, include_lowest=True, right=True)

        grouped: List[np.ndarray] = []
        valid_labels: List[str] = []
        for label in labels:
            vals = target[categories == label].dropna().to_numpy(dtype=float)
            if len(vals) >= 2:
                grouped.append(vals)
                valid_labels.append(str(label))
                rows.append(
                    {
                        "feature": column,
                        "interval": str(label),
                        "n": len(vals),
                        "removal_mean": float(np.mean(vals)),
                        "removal_median": float(np.median(vals)),
                        "removal_q25": float(np.quantile(vals, 0.25)),
                        "removal_q75": float(np.quantile(vals, 0.75)),
                    }
                )

        if grouped:
            _draw_single_violin_panel(
                ax,
                grouped,
                valid_labels,
                violin_colors[: len(grouped)],
            )

        ax.set_ylim(-3, 108)
        ax.set_title(f"{title} ({unit})", fontsize=6.9, pad=2.2)
        ax.set_ylabel("Removal efficiency (%)" if index % 3 == 0 else "", fontsize=6.0, labelpad=1.5)
        clean_axes(ax, grid_axis="y")

    return pd.DataFrame(rows)


def draw_group_distribution(ax: plt.Axes, df: pd.DataFrame) -> pd.DataFrame:
    panel_label(ax, "c")
    full_counts = df["文献编号"].dropna().astype(str).value_counts().sort_values(ascending=False)
    shown = full_counts.head(15).sort_values(ascending=True)
    omitted_n = len(full_counts) - len(shown)
    omitted_records = int(full_counts.iloc[15:].sum()) if omitted_n > 0 else 0

    colors = [PALETTE["blue"]] * len(shown)
    if colors:
        colors[-1] = PALETTE["orange"]

    y = np.arange(len(shown))
    ax.barh(y, shown.values, color=colors, height=0.70, edgecolor=PALETTE["white"], linewidth=0.35)
    ax.set_yticks(y)
    ax.set_yticklabels([name.replace("_", " ") for name in shown.index], fontsize=5.8)
    ax.set_xlabel("Number of records")
    ax.set_title("Literature-source imbalance (top 15 groups)", pad=4)
    clean_axes(ax, grid_axis="x")

    for yi, value in zip(y, shown.values):
        ax.text(value + max(full_counts.max() * 0.012, 0.65), yi, f"{int(value)}", va="center", ha="left", fontsize=5.5, color=PALETTE["gray"])

    top3_share = float(full_counts.head(3).sum() / full_counts.sum() * 100.0)
    ax.text(
        0.98,
        0.07,
        f"N = {int(full_counts.sum()):,}; groups = {len(full_counts)}\nTop 3 groups = {top3_share:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        color=PALETTE["gray"],
    )

    ax.set_xlim(0, full_counts.max() * 1.18)
    return full_counts.rename_axis("literature_group").reset_index(name="record_count")


def draw_target_distribution(ax: plt.Axes, df: pd.DataFrame) -> pd.DataFrame:
    panel_label(ax, "c")
    target = pd.to_numeric(df["氟去除率_pct"], errors="coerce").dropna()
    counts, edges, _ = ax.hist(
        target,
        bins=np.linspace(0, 100, 21),
        color=PALETTE["blue_light"],
        edgecolor=PALETTE["white"],
        linewidth=0.42,
    )

    if target.nunique() > 3:
        kde = gaussian_kde(target)
        x_grid = np.linspace(0, 100, 300)
        bin_width = edges[1] - edges[0]
        ax.plot(x_grid, kde(x_grid) * len(target) * bin_width, color=PALETTE["teal"], linewidth=1.15)

    mean = target.mean()
    median = target.median()
    ax.axvline(mean, color=PALETTE["orange"], linestyle="--", linewidth=0.9, label=f"Mean = {mean:.1f}%")
    ax.axvline(median, color=PALETTE["teal"], linestyle="-.", linewidth=0.9, label=f"Median = {median:.1f}%")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Fluoride removal efficiency (%)")
    ax.set_ylabel("Number of records")
    ax.set_title("Prediction-target distribution", pad=4)
    ax.legend(frameon=False, loc="upper left", handlelength=2.0)
    clean_axes(ax, grid_axis="y")
    ax.text(
        0.98,
        0.96,
        f"N = {len(target):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=PALETTE["gray"],
    )

    return pd.DataFrame(
        [
            {
                "N": len(target),
                "mean": mean,
                "median": median,
                "std": target.std(ddof=1),
                "q25": target.quantile(0.25),
                "q75": target.quantile(0.75),
                "min": target.min(),
                "max": target.max(),
            }
        ]
    )


CATEGORY_LABELS = {
    "聚合氯化铝": "PACl",
    "硫酸铝": "Aluminum sulfate",
    "锆基复合混凝剂（ZXC）": "Zr composite (ZXC)",
    "聚合硫酸铁": "PFS",
    "氯化铝": "AlCl$_3$",
    "聚硅酸铁钙镧复合混凝剂（PSFCL）": "PSFCL",
    "复合硫酸铝镁（CSAM，Al/Mg=14:1）": "CSAM",
    "复合硫酸铝铁（CSAF，Al/Fe=4:1）": "CSAF",
    "无": "None",
    "SO₄²⁻/CO₃²⁻": "Sulfate/carbonate",
    "Ca²⁺/Mg²⁺/Cl⁻/SO₄²⁻/HCO₃⁻": "Mixed inorganic ions",
    "PO₄³⁻/SO₄²⁻/Cl⁻/Ca²⁺": "Phosphate/sulfate mix",
    "腐殖酸/海藻酸": "Humic acid/alginate",
    "HCO₃⁻": "Bicarbonate",
    "NO₃⁻": "Nitrate",
    "腐殖酸/NO₃⁻/CO₃²⁻": "Humic/nitrate/carbonate",
    "Other": "Other",
    "Missing": "Missing",
}


def display_category_label(value: str) -> str:
    value = str(value)
    return CATEGORY_LABELS.get(value, value if value.isascii() else "Other formulation")


def compact_category(series: pd.Series, top_n: int) -> pd.DataFrame:
    counts = series.fillna("Missing").astype(str).value_counts()
    top = counts.head(top_n).copy()
    other = int(counts.iloc[top_n:].sum())
    if other > 0:
        top.loc["Other"] = other
    return top.rename_axis("category").reset_index(name="count")


def draw_categorical_composition(fig: plt.Figure, spec, df: pd.DataFrame) -> pd.DataFrame:
    outer = fig.add_subplot(spec)
    outer.axis("off")
    panel_label(outer, "d", x=-0.02, y=1.015)
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.52)
    frames: List[pd.DataFrame] = []
    configurations = [
        ("混凝剂", "Coagulant types", 7, "coagulant"),
        ("干扰离子", "Interfering-ion settings", 6, "ion"),
    ]

    for index, (column, title, top_n, family) in enumerate(configurations):
        ax = fig.add_subplot(inner[0, index])
        comp = compact_category(df[column], top_n=top_n).sort_values("count", ascending=True)
        y = np.arange(len(comp))
        values = comp["count"].to_numpy(dtype=float)
        norm = mcolors.Normalize(vmin=float(values.min()), vmax=float(values.max()))
        if family == "coagulant":
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "est_coagulant", [EST_CATEGORY_COLORS["coagulant_low"], EST_CATEGORY_COLORS["coagulant_high"]]
            )
        else:
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "est_ion", [EST_CATEGORY_COLORS["ion_low"], EST_CATEGORY_COLORS["ion_high"]]
            )
        bar_colors = [cmap(norm(value)) for value in values]
        ax.barh(
            y,
            values,
            color=bar_colors,
            alpha=0.98,
            edgecolor=PALETTE["white"],
            linewidth=0.45,
        )
        translated = [display_category_label(label) for label in comp["category"].astype(str)]
        labels = [label if len(label) <= 22 else label[:21] + "..." for label in translated]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=5.8)
        ax.set_xlabel("Records", fontsize=6.3)
        ax.set_title(title, fontsize=7.0, pad=2.0, fontweight="semibold")
        clean_axes(ax, grid_axis="x")
        for yi, value in zip(y, comp["count"]):
            ax.text(value + max(comp["count"].max() * 0.012, 0.5), yi, f"{int(value)}", va="center", fontsize=5.4, color=PALETTE["gray"])
        ax.set_xlim(0, comp["count"].max() * 1.18)

        comp.insert(0, "descriptor", column)
        comp["display_label"] = [display_category_label(label) for label in comp["category"].astype(str)]
        frames.append(comp)

    return pd.concat(frames, ignore_index=True)


def draw_matrix_roles(ax: plt.Axes, df: pd.DataFrame) -> pd.DataFrame:
    panel_label(ax, "f", x=-0.02, y=1.03)
    ax.axis("off")
    ax.set_title("Final data-matrix structure", pad=2.0)

    # Matrix relationship at the top.
    left_box = FancyBboxPatch(
        (0.04, 0.64),
        0.37,
        0.27,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=ax.transAxes,
        facecolor=PALETTE["blue_light"],
        edgecolor=PALETTE["gray"],
        linewidth=0.65,
    )
    right_box = FancyBboxPatch(
        (0.59, 0.64),
        0.37,
        0.27,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=ax.transAxes,
        facecolor="#DCE6E2",
        edgecolor=PALETTE["gray"],
        linewidth=0.65,
    )
    ax.add_patch(left_box)
    ax.add_patch(right_box)
    ax.text(0.225, 0.82, "Strict matrix", transform=ax.transAxes, ha="center", va="center", fontsize=6.9, fontweight="bold")
    ax.text(0.225, 0.71, "1,011 x 26 fields\nwithout literature ID", transform=ax.transAxes, ha="center", va="center", fontsize=5.7, color=PALETTE["gray"])
    ax.text(0.775, 0.82, "Source-aware matrix", transform=ax.transAxes, ha="center", va="center", fontsize=6.9, fontweight="bold")
    ax.text(0.775, 0.71, "1,011 x 27 fields\nadds literature ID", transform=ax.transAxes, ha="center", va="center", fontsize=5.7, color=PALETTE["gray"])
    arrow = FancyArrowPatch(
        (0.43, 0.775),
        (0.57, 0.775),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.75,
        color=PALETTE["gray"],
    )
    ax.add_patch(arrow)

    roles = [
        ("Predictors", "10", "water/process", PALETTE["blue_light"]),
        ("Target", "1", "removal efficiency", PALETTE["green_light"]),
        ("Response/QC", "4", "effluent/consistency", PALETTE["orange_light"]),
        ("Raw audit", "11", "original values", PALETTE["gray_light"]),
        ("Group label", "1", "literature ID", "#DCE6E2"),
    ]
    x_positions = np.linspace(0.03, 0.81, len(roles))
    card_width = 0.16
    output_rows: List[dict] = []
    for x, (title, count, subtitle, fill) in zip(x_positions, roles):
        patch = FancyBboxPatch(
            (x, 0.14),
            card_width,
            0.34,
            boxstyle="round,pad=0.010,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor=fill,
            edgecolor=PALETTE["gray"],
            linewidth=0.55,
        )
        ax.add_patch(patch)
        ax.text(x + card_width / 2, 0.385, title, transform=ax.transAxes, ha="center", va="center", fontsize=6.2, fontweight="bold")
        ax.text(x + card_width / 2, 0.285, count, transform=ax.transAxes, ha="center", va="center", fontsize=7.0, fontweight="bold", color=PALETTE["teal"])
        ax.text(x + card_width / 2, 0.205, subtitle, transform=ax.transAxes, ha="center", va="center", fontsize=5.0, color=PALETTE["gray"])
        output_rows.append({"role": title, "field_count": count, "description": subtitle})

    ax.text(
        0.5,
        0.02,
        "Literature ID was excluded from model inputs and used only for grouped validation.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.7,
        color=PALETTE["gray"],
    )
    return pd.DataFrame(output_rows)


def save_outputs(
    fig: plt.Figure,
    paths: ProjectPaths,
    tables: Dict[str, pd.DataFrame],
    elapsed: float,
    df: pd.DataFrame,
) -> Tuple[Path, Path]:
    png = paths.figures / f"{FIGURE_STEM}.png"
    pdf = paths.figures / f"{FIGURE_STEM}.pdf"
    fig.savefig(png, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.025, facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.025, facecolor="white")
    plt.close(fig)

    for suffix, table in tables.items():
        table.to_csv(paths.tables / f"{FIGURE_STEM}_{suffix}.csv", index=False, encoding="utf-8-sig")

    report_lines = [
        "EST-style Figure 1 publication audit",
        "=" * 80,
        f"Script: {SCRIPT_NAME}.py",
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
        f"Project root: {paths.root}",
        f"Input: {paths.data_file}",
        f"Dataset shape: {df.shape}",
        f"Observed literature groups: {df['文献编号'].nunique(dropna=True)}",
        f"Final figure size: {FIGURE_WIDTH_IN:.2f} x {FIGURE_HEIGHT_IN:.2f} in",
        "",
        "Publication revisions relative to v5",
        "-" * 80,
        "- Main-text Figure 1 retains only the former panels a, b, d, and e.",
        "- Retained content is relabeled sequentially as panels a-d.",
        "- Literature-group imbalance and matrix-role structure are reserved for Supplementary Figure S1.",
        "- Panels a and b remain full-width and use tighter internal and inter-panel spacing.",
        "- Panel a uses a refined six-color EST-inspired muted palette; panel d uses teal and pale-gold gradients.",
        "- The target distribution and categorical composition share the final row.",
        "",
        "Data-integrity note",
        "-" * 80,
        (
            "The final matrix contains one residence-time field but no separate mixing-time and "
            "settling-time fields. The figure therefore uses residence time and interfering-ion "
            "concentration rather than inventing unavailable variables."
        ),
        "",
        f"PNG: {png}",
        f"PDF: {pdf}",
        f"Total runtime: {elapsed:.2f} s",
    ]
    report = "\n".join(report_lines)
    (paths.reports / f"{SCRIPT_NAME}_audit.txt").write_text(report, encoding="utf-8")
    (paths.logs / f"{SCRIPT_NAME}_run_log.txt").write_text(report, encoding="utf-8")
    return png, pdf


def main() -> None:
    start = time.perf_counter()
    configure_matplotlib()
    paths = build_paths()
    df = load_data(paths)

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), constrained_layout=False)
    grid = GridSpec(
        nrows=3,
        ncols=2,
        figure=fig,
        height_ratios=[1.64, 1.86, 1.24],
        width_ratios=[0.42, 0.58],
        hspace=0.31,
        wspace=0.40,
        left=0.085,
        right=0.985,
        bottom=0.055,
        top=0.982,
    )

    histogram_summary = draw_histogram_matrix(fig, grid[0, :], df)
    violin_summary = draw_violin_matrix(fig, grid[1, :], df)

    ax_c = fig.add_subplot(grid[2, 0])
    target_summary = draw_target_distribution(ax_c, df)
    category_summary = draw_categorical_composition(fig, grid[2, 1], df)

    elapsed = time.perf_counter() - start
    png, pdf = save_outputs(
        fig,
        paths,
        {
            "panel_a_histogram_statistics": histogram_summary,
            "panel_b_violin_interval_statistics": violin_summary,
            "panel_c_target_summary": target_summary,
            "panel_d_categorical_composition": category_summary,
        },
        elapsed,
        df,
    )

    print("=" * 80)
    print("Figure 1 generated successfully.")
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")
    print(f"Total runtime: {elapsed:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
