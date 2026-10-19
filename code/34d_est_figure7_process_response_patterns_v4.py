# -*- coding: utf-8 -*-
"""
Figure 7 — Process–response patterns and mechanistic clues

Place this file in:
    fluoride_coagulation_ml_project/src/

Run directly in PyCharm. Outputs are saved to:
    outputs/jece_final_package/figures/
    outputs/jece_final_package/tables/
"""
from __future__ import annotations

import time
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder

SCRIPT_NAME = "34d_est_figure7_process_response_patterns_v4"
FIGURE_STEM = "Figure_7_process_response_patterns_and_mechanistic_clues_v4"

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
def locate_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "data" / "processed").exists() and (candidate / "outputs").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root containing data/processed and outputs/.")


ROOT = locate_project_root()
DATA_PATH = ROOT / "data" / "processed" / "model_ready_v1_with_source.csv"
OUT_ROOT = ROOT / "outputs" / "jece_final_package"
FIG_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"
REPORT_DIR = OUT_ROOT / "reports"
LOG_DIR = OUT_ROOT / "logs"
for folder in (FIG_DIR, TABLE_DIR, REPORT_DIR, LOG_DIR):
    folder.mkdir(parents=True, exist_ok=True)

PNG_PATH = FIG_DIR / f"{FIGURE_STEM}.png"
PDF_PATH = FIG_DIR / f"{FIGURE_STEM}.pdf"

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
COLORS = {
    "blue": "#5E88AE",
    "teal": "#5E9A9B",
    "gold": "#C79A4B",
    "green": "#8FA987",
    "orange": "#D08C60",
    "slate": "#707C88",
    "light_blue": "#B7C9D9",
    "light_teal": "#C7D9D8",
    "light_gold": "#E2D4B6",
    "grid": "#D8DDE3",
}

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.6,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.4,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "grid.color": COLORS["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.85,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

FEATURES = [
    "氟形态",
    "游离氟占比",
    "初始氟_mg_L",
    "干扰离子",
    "干扰离子浓度_mmol_L",
    "水温_C",
    "混凝剂",
    "剂量_mM",
    "进水pH",
    "停留时间_min",
]
TARGET = "氟去除率_pct"
NUMERIC = [
    "游离氟占比",
    "初始氟_mg_L",
    "干扰离子浓度_mmol_L",
    "水温_C",
    "剂量_mM",
    "进水pH",
    "停留时间_min",
]
CATEGORICAL = ["氟形态", "干扰离子", "混凝剂"]


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        ha="left",
        va="top",
    )


def feature_label(name: str) -> str:
    mapping = {
        "氟形态": "Fluoride form",
        "游离氟占比": "Free-fluoride fraction",
        "初始氟_mg_L": "Initial fluoride",
        "干扰离子": "Interfering ion",
        "干扰离子浓度_mmol_L": "Interfering-ion concentration",
        "水温_C": "Water temperature",
        "混凝剂": "Coagulant type",
        "剂量_mM": "Coagulant dosage",
        "进水pH": "Influent pH",
        "停留时间_min": "Residence time",
    }
    return mapping.get(name, name)


def compact_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.1e}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def quantile_groups(series: pd.Series, n_groups: int = 5, prefix: str = "Q"):
    numeric = pd.to_numeric(series, errors="coerce")
    grouped = pd.qcut(numeric, q=n_groups, duplicates="drop")
    categories = list(grouped.cat.categories)
    labels = []
    for i, interval in enumerate(categories, start=1):
        median_value = numeric[grouped == interval].median()
        labels.append(f"{prefix}{i}\n({compact_number(float(median_value))})")
    renamed = grouped.cat.rename_categories(labels)
    return renamed, labels


def pH_groups(series: pd.Series):
    numeric = pd.to_numeric(series, errors="coerce")
    boundaries = np.unique(np.quantile(numeric.dropna(), np.linspace(0, 1, 6)))
    grouped = pd.cut(numeric, bins=boundaries, include_lowest=True, duplicates="drop")
    labels = []
    for interval in grouped.cat.categories:
        labels.append(f"{interval.left:.1f}–{interval.right:.1f}")
    return grouped.cat.rename_categories(labels), labels


def violin_box(
    ax: plt.Axes,
    values: pd.Series,
    groups: pd.Series,
    color: str,
    xlabel: str,
    title: str,
    letter: str,
) -> None:
    panel_label(ax, letter)
    levels = list(groups.cat.categories)
    arrays = [values[groups == level].dropna().to_numpy() for level in levels]

    violin = ax.violinplot(arrays, positions=np.arange(1, len(levels) + 1), widths=0.86, showextrema=True)
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25)
        body.set_linewidth(0.9)
    for key in ("cbars", "cmins", "cmaxes"):
        violin[key].set_color("#7D8791")
        violin[key].set_linewidth(0.8)

    box = ax.boxplot(
        arrays,
        positions=np.arange(1, len(levels) + 1),
        widths=0.26,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.3},
        whiskerprops={"color": "#7D8791", "linewidth": 0.8},
        capprops={"color": "#7D8791", "linewidth": 0.8},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.92)

    ax.set_xticks(np.arange(1, len(levels) + 1))
    ax.set_xticklabels(levels, rotation=0, ha="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fluoride removal (%)")
    ax.set_title(title, pad=10)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_ylim(-5, 106)
    for position, values_here in enumerate(arrays, start=1):
        ax.text(position, 102, f"n={len(values_here)}", ha="center", va="bottom", fontsize=7.7, color=COLORS["slate"])


def mutual_information_ranking(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df[FEATURES].copy()
    for column in NUMERIC:
        encoded[column] = pd.to_numeric(encoded[column], errors="coerce")
        encoded[column] = encoded[column].fillna(encoded[column].median())
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoded[CATEGORICAL] = encoder.fit_transform(encoded[CATEGORICAL].astype(str).fillna("Missing"))
    discrete_mask = [column in CATEGORICAL for column in FEATURES]
    scores = mutual_info_regression(
        encoded[FEATURES],
        pd.to_numeric(df[TARGET], errors="coerce"),
        discrete_features=discrete_mask,
        random_state=42,
    )
    return (
        pd.DataFrame({"feature": FEATURES, "mutual_information": scores})
        .sort_values("mutual_information", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    start = time.time()
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Required dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    mi = mutual_information_ranking(df)
    mi.to_csv(TABLE_DIR / "Figure_7_mutual_information_ranking_v3.csv", index=False, encoding="utf-8-sig")

    dose_group, dose_labels = quantile_groups(df["剂量_mM"], 5, "Q")
    fluoride_group, fluoride_labels = quantile_groups(df["初始氟_mg_L"], 5, "Q")
    ph_group, ph_labels = pH_groups(df["进水pH"])

    # Interaction map: median target and sample size in dose × initial-fluoride quantile bins
    interaction_df = df.assign(dose_group=dose_group, fluoride_group=fluoride_group)
    median_map = interaction_df.pivot_table(
        values=TARGET,
        index="fluoride_group",
        columns="dose_group",
        aggfunc="median",
        observed=False,
    ).reindex(index=fluoride_labels, columns=dose_labels)
    count_map = interaction_df.pivot_table(
        values=TARGET,
        index="fluoride_group",
        columns="dose_group",
        aggfunc="count",
        observed=False,
    ).reindex(index=fluoride_labels, columns=dose_labels)

    # Coagulant summary and manuscript-safe code map
    top_coagulants = df["混凝剂"].astype(str).value_counts().head(8).index.tolist()
    code_map = {name: f"C{i + 1}" for i, name in enumerate(top_coagulants)}
    coag = df[df["混凝剂"].astype(str).isin(top_coagulants)].copy()
    coag["coagulant_code"] = coag["混凝剂"].astype(str).map(code_map)
    coag_stats = (
        coag.groupby(["coagulant_code", "混凝剂"])[TARGET]
        .agg(
            median="median",
            count="count",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
        )
        .reset_index()
        .sort_values("median", ascending=True)
        .reset_index(drop=True)
    )
    coag_stats.to_csv(TABLE_DIR / "Figure_7_coagulant_summary_v3.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {"coagulant_code": list(code_map.values()), "coagulant_full_name": list(code_map.keys())}
    ).to_csv(TABLE_DIR / "Figure_7_coagulant_code_map_v3.csv", index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(13.8, 9.2))
    grid = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.03, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.42,
        hspace=0.42,
    )

    # a: association ranking
    ax_a = fig.add_subplot(grid[0, 0])
    panel_label(ax_a, "a")
    shown = mi.head(8).iloc[::-1]
    bar_colors = [COLORS["gold"] if feature in CATEGORICAL else COLORS["blue"] for feature in shown["feature"]]
    positions = np.arange(len(shown))
    ax_a.barh(positions, shown["mutual_information"], color=bar_colors, height=0.66, edgecolor="none")
    ax_a.set_yticks(positions)
    ax_a.set_yticklabels([feature_label(feature) for feature in shown["feature"]])
    ax_a.set_xlabel("Mutual-information score")
    ax_a.set_title("Association ranking of process descriptors", pad=10)
    ax_a.grid(axis="x")
    ax_a.set_axisbelow(True)
    maximum = shown["mutual_information"].max()
    ax_a.set_xlim(0, maximum * 1.13)
    for position, value in zip(positions, shown["mutual_information"]):
        ax_a.text(value + maximum * 0.018, position, f"{value:.3f}", va="center", fontsize=8.2)
    ax_a.text(
        0.01,
        0.02,
        "Higher values indicate stronger nonlinear association with removal.",
        transform=ax_a.transAxes,
        fontsize=7.9,
        color=COLORS["slate"],
    )

    # b-d: binned distributions
    ax_b = fig.add_subplot(grid[0, 1])
    violin_box(
        ax_b,
        df[TARGET],
        dose_group,
        COLORS["teal"],
        "Coagulant dosage quintile\n(median dose in parentheses, mM)",
        "Removal pattern across dosage levels",
        "b",
    )

    ax_c = fig.add_subplot(grid[0, 2])
    violin_box(
        ax_c,
        df[TARGET],
        fluoride_group,
        COLORS["green"],
        "Initial-fluoride quintile\n(median concentration in parentheses, mg L$^{-1}$)",
        "Removal pattern across initial-fluoride levels",
        "c",
    )

    ax_d = fig.add_subplot(grid[1, 0])
    violin_box(
        ax_d,
        df[TARGET],
        ph_group,
        COLORS["gold"],
        "Influent-pH interval",
        "Removal pattern across influent-pH intervals",
        "d",
    )

    # e: interaction heatmap
    ax_e = fig.add_subplot(grid[1, 1])
    panel_label(ax_e, "e")
    image = ax_e.imshow(median_map.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax_e.set_title("Combined dosage–initial fluoride response map", pad=10)
    ax_e.set_xticks(np.arange(len(dose_labels)))
    ax_e.set_xticklabels(dose_labels)
    ax_e.set_yticks(np.arange(len(fluoride_labels)))
    ax_e.set_yticklabels(fluoride_labels)
    ax_e.set_xlabel("Coagulant dosage quintile")
    ax_e.set_ylabel("Initial-fluoride quintile")
    for row in range(median_map.shape[0]):
        for column in range(median_map.shape[1]):
            value = median_map.iloc[row, column]
            sample_count = count_map.iloc[row, column]
            if pd.notna(value):
                text_color = "white" if value >= 58 else "#1E2B34"
                ax_e.text(
                    column,
                    row,
                    f"{value:.0f}\n(n={int(sample_count)})",
                    ha="center",
                    va="center",
                    fontsize=7.3,
                    color=text_color,
                )
    colorbar = fig.colorbar(image, ax=ax_e, fraction=0.038, pad=0.018)
    colorbar.ax.set_title("Median removal (%)", fontsize=7.8, pad=4)
    colorbar.ax.tick_params(labelsize=7.7)

    # f: coagulant summary using manuscript-safe codes
    ax_f = fig.add_subplot(grid[1, 2])
    panel_label(ax_f, "f")
    y_positions = np.arange(len(coag_stats))
    ax_f.hlines(y_positions, coag_stats["q1"], coag_stats["q3"], color=COLORS["light_blue"], linewidth=5.5)
    ax_f.scatter(coag_stats["median"], y_positions, color=COLORS["orange"], s=44, zorder=3)
    ax_f.set_yticks(y_positions)
    ax_f.set_yticklabels(coag_stats["coagulant_code"])
    ax_f.set_xlabel("Fluoride removal (%)")
    ax_f.set_title("Coagulant-type response summary", pad=10)
    ax_f.grid(axis="x")
    ax_f.set_axisbelow(True)
    x_min = max(0, float(coag_stats["q1"].min()) - 5)
    x_max = min(105, float(coag_stats["q3"].max()) + 16)
    ax_f.set_xlim(x_min, x_max)
    for row_number, row in coag_stats.iterrows():
        ax_f.text(x_max - 0.8, row_number, f"n={int(row['count'])}", ha="right", va="center", fontsize=7.9, color=COLORS["slate"])
    ax_f.text(0.02, 0.02, "Bar = IQR; dot = median.", transform=ax_f.transAxes, fontsize=8.0, color=COLORS["slate"])
    ax_f.text(
        0.02,
        -0.14,
        "C1–C8 are defined in the exported code-map table.",
        transform=ax_f.transAxes,
        fontsize=7.8,
        color=COLORS["slate"],
    )

    plt.subplots_adjust(left=0.075, right=0.985, top=0.95, bottom=0.095)
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)

    runtime = time.time() - start
    report = (
        f"{SCRIPT_NAME}\n"
        f"records={len(df)}\n"
        f"figure_png={PNG_PATH}\n"
        f"figure_pdf={PDF_PATH}\n"
        f"runtime_s={runtime:.2f}\n"
    )
    (REPORT_DIR / f"{SCRIPT_NAME}_audit.txt").write_text(report, encoding="utf-8")
    (LOG_DIR / f"{SCRIPT_NAME}.log").write_text(report, encoding="utf-8")

    print(f"Figure PNG: {PNG_PATH}")
    print(f"Figure PDF: {PDF_PATH}")
    print(f"Total runtime: {runtime:.2f} s")


if __name__ == "__main__":
    main()
